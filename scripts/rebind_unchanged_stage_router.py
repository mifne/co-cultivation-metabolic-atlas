"""Explicitly rebind unchanged stage weights after OTHER stages are rebuilt.

Deployment SHA guards are NOT relaxed. Both manifests, every selected artifact,
GEM identities and the full stage record must agree, before a NEW router with
new whole-bank provenance is written. No fit, score change or online LP solve.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.compact_training_data import checked_child, checked_npz, sha256_file
from src.coverage_router import CoverageRouter
from src.coverage_router_binding import load_bound_coverage_router


def rebind_router(router_path, router_sha256, old_bank, old_bank_sha256,
                  new_bank, new_bank_sha256, output, *, stage_name='maxmin'):
    if stage_name not in ('maxmin','aggregate','exchange'):
        raise ValueError('Router rebinding requires an original LP stage')
    old_bank, new_bank, output = map(Path, (old_bank, new_bank, output))
    if output.exists():
        raise FileExistsError(output)
    manifests, stages, schemas = [], [], []
    for directory, digest in ((old_bank, old_bank_sha256), (new_bank, new_bank_sha256)):
        manifest_path = checked_child(directory, 'manifest.json', digest)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'completed':
            raise ValueError('Only completed bank manifests may be rebound')
        candidates = [stage for stage in manifest['stages'] if stage['stage'] == stage_name]
        if len(candidates) != 1:
            raise ValueError('Exactly one unchanged selected stage is required')
        stage = candidates[0]
        root = checked_npz(directory/stage_name, 'root.npz', stage['root_sha256'])
        nearest = checked_npz(directory/stage_name, 'router.npz', stage['router_sha256'])
        for entry in stage['entries']:
            checked_child(directory/stage_name, entry['filename'], entry['sha256'])
        bank = SimpleNamespace(host_a=SimpleNamespace(shape=tuple(root['a_shape'])),
            variable_rows=root['variable_rows'], feature_indices=nearest['indices'],
            evaluators=[None]*len(stage['entries']))
        manifests.append(manifest); stages.append(stage); schemas.append(bank)
    if stages[0] != stages[1]:
        raise ValueError('Selected stage changed: router must be retrained, not rebound')
    if manifests[0]['model_fingerprints'] != manifests[1]['model_fingerprints']:
        raise ValueError('GEM fingerprints changed')
    _, old_binding = load_bound_coverage_router(router_path, router_sha256,
        bank_manifest_sha256=old_bank_sha256, stage_manifest=stages[0], bank=schemas[0],
        model_fingerprints=manifests[0]['model_fingerprints'], xp=np)
    with np.load(router_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        arrays = {name:archive[name].copy() for name in archive.files if name != 'metadata'}
    original = CoverageRouter(arrays, metadata)
    metadata = copy.deepcopy(metadata)
    metadata['provenance']['bank_sha256'] = new_bank_sha256
    metadata.setdefault('unchanged_stage_rebindings', []).append(dict(
        source_router_sha256=router_sha256, old_bank_sha256=old_bank_sha256,
        new_bank_sha256=new_bank_sha256, stage=stage_name,
        full_stage_record_equal=True, all_stage_files_sha_verified=True,
        weights_and_feature_statistics_unchanged=True, training_performed=False,
        script_sha256=sha256_file(__file__)))
    rebound = CoverageRouter(arrays, metadata)
    for name in arrays:
        if not np.array_equal(original.arrays[name], rebound.arrays[name]):
            raise ValueError('Router arrays changed during metadata-only rebinding')
    saved = rebound.save(output)
    _, binding = load_bound_coverage_router(output, saved['sha256'],
        bank_manifest_sha256=new_bank_sha256, stage_manifest=stages[1], bank=schemas[1],
        model_fingerprints=manifests[1]['model_fingerprints'], xp=np)
    return dict(status='rebound_unchanged_stage', **saved,
        old_binding=old_binding, new_binding=binding,
        training_performed=False, arrays_bitwise_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--router', type=Path, required=True)
    parser.add_argument('--router-sha256', required=True)
    parser.add_argument('--old-bank', type=Path, required=True)
    parser.add_argument('--old-bank-sha256', required=True)
    parser.add_argument('--new-bank', type=Path, required=True)
    parser.add_argument('--new-bank-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['maxmin','aggregate','exchange'], default='maxmin')
    args = parser.parse_args()
    report = rebind_router(args.router, args.router_sha256, args.old_bank,
        args.old_bank_sha256, args.new_bank, args.new_bank_sha256, args.output, stage_name=args.stage)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
