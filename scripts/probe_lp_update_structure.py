"""Input-only changing-state LP structure audit; never solves, calls CUDA or QR.

Sparse forest/zero-face maps may be constructed to inspect coordinate changes.
Exact-equality proof reuse is only a fingerprint eligibility check: this script
does not discover QR proofs, certify an optimum or advance a dFBA environment.
Recorded reference solution arrays are deliberately never loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import select_entries
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_exact_equalities import equality_fingerprint
from src.lp_trace import problem_hash
from src.lp_zero_face import ZeroFaceReduction

STAGES = ('maxmin', 'aggregate', 'exchange')


def digest_arrays(items):
    digest = hashlib.sha256()
    for name, value in items:
        array = np.asarray(value)
        dtype = '<i8' if array.dtype.kind in 'biu' else '<f8'
        array = np.asarray(array, dtype=dtype)
        digest.update(name.encode('ascii'))
        digest.update(dtype.encode('ascii'))
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sparse_key(a, *, include_values):
    items = [('shape', a.shape), ('indptr', a.indptr), ('indices', a.indices)]
    if include_values:
        items.append(('values', a.data))
    return digest_arrays(items)


def bound_topology(lo, hi):
    """Numeric endpoints are not topology, but fixed/free/finite placement is."""
    fixed = np.isfinite(lo) & (lo == hi)
    return digest_arrays([('fixed', fixed), ('finite_lower', np.isfinite(lo)),
                          ('finite_upper', np.isfinite(hi))])


def problem_structure(problem):
    a, rhs, lo, hi, c, neq = problem
    equality = a[:neq].tocsr()
    fixed = np.isfinite(lo) & (lo == hi)
    result = dict(problem_sha256=problem_hash(problem), rows=a.shape[0], columns=a.shape[1],
                  nonzeros=a.nnz, equalities=int(neq),
                  csr_support_key=sparse_key(a, include_values=False),
                  a_numeric_key=sparse_key(a, include_values=True),
                  equality_numeric_key=sparse_key(equality, include_values=True),
                  exact_equality_rhs_key=equality_fingerprint(problem),
                  rhs_key=digest_arrays([('rhs', rhs)]), cost_key=digest_arrays([('cost', c)]),
                  lower_key=digest_arrays([('lower', lo)]), upper_key=digest_arrays([('upper', hi)]),
                  bound_topology_key=bound_topology(lo, hi),
                  fixed_columns=int(fixed.sum()), finite_lower=int(np.isfinite(lo).sum()),
                  finite_upper=int(np.isfinite(hi).sum()))
    # Sufficient construction cache key. A nonzero RHS becoming zero can add
    # forest edges even if every equality coefficient remains unchanged.
    result['forest_construction_key'] = digest_json(dict(
        algorithm='HomogeneousEqualityReduction:v1', max_scale_ratio=1e8,
        original_shape=a.shape, neq=int(neq), equality=result['equality_numeric_key'],
        homogeneous_mask=digest_arrays([('zero_rhs', rhs[:neq] == 0.)])))
    # Full numeric A/RHS/box is deliberately conservative: zero inequality rows
    # and duplicate equality RHS also affect which original rows survive.
    result['zero_face_input_key_without_config'] = digest_json({key: result[key] for key in
        ('a_numeric_key', 'rhs_key', 'lower_key', 'upper_key', 'equalities')})
    return result


def vector_difference(old, new):
    if old.shape != new.shape:
        return dict(shape_changed=True, old_shape=list(old.shape), new_shape=list(new.shape),
                    changed=None, max_abs_finite_delta=None, nonfinite_transitions=None)
    changed = old != new
    both_finite = np.isfinite(old) & np.isfinite(new)
    return dict(shape_changed=False, changed=int(changed.sum()),
                changed_indices=np.flatnonzero(changed).tolist(),
                max_abs_finite_delta=float(np.max(np.abs(new[both_finite] - old[both_finite]), initial=0.)),
                nonfinite_transitions=int(np.count_nonzero(changed & ~both_finite)))


def matrix_difference(old, new):
    if old.shape != new.shape:
        return dict(shape_changed=True, old_shape=list(old.shape), new_shape=list(new.shape))
    same_support = (np.array_equal(old.indptr, new.indptr)
                    and np.array_equal(old.indices, new.indices))
    delta = (new - old).tocsr()
    delta.eliminate_zeros()
    rows = np.flatnonzero(np.diff(delta.indptr))
    cols = np.unique(delta.indices)
    old_presence, new_presence = old.copy(), new.copy()
    old_presence.data = np.ones(old.nnz)
    new_presence.data = np.ones(new.nnz)
    support_delta = (new_presence - old_presence).tocsr()
    support_delta.eliminate_zeros()
    return dict(shape_changed=False, same_csr_support=bool(same_support),
                coefficient_changes=int(delta.nnz), changed_rows=rows.tolist(),
                changed_columns=cols.tolist(), support_added=int(np.sum(support_delta.data > 0)),
                support_removed=int(np.sum(support_delta.data < 0)),
                max_abs_delta=float(np.max(np.abs(delta.data), initial=0.)))


def problem_difference(old, new):
    answer = dict(A=matrix_difference(old[0], new[0]), neq_changed=old[-1] != new[-1])
    for index, name in enumerate(('rhs', 'lower', 'upper', 'cost'), 1):
        answer[name] = vector_difference(old[index], new[index])
    if old[0].shape == new[0].shape and old[-1] == new[-1]:
        neq = old[-1]
        answer['A_equalities'] = matrix_difference(old[0][:neq], new[0][:neq])
        answer['A_inequalities'] = matrix_difference(old[0][neq:], new[0][neq:])
        answer['rhs_equalities'] = vector_difference(old[1][:neq], new[1][:neq])
        answer['rhs_inequalities'] = vector_difference(old[1][neq:], new[1][neq:])
    before, after = problem_structure(old), problem_structure(new)
    answer['same_keys'] = {key: before[key] == after[key] for key in before if key.endswith('_key')}
    return answer


def forest_map_keys(plan, reduction):
    return dict(coordinate_key=digest_json(dict(
        kept=digest_arrays([('kept', plan.kept_rows), ('eliminated', plan.eliminated_rows)]),
        transform=sparse_key(plan.transform, include_values=True),
        compression=sparse_key(plan.compression, include_values=True),
        dual_lift=sparse_key(plan.dual_lift_map, include_values=True))),
        bound_witness_key=digest_arrays([('lower', reduction.lower_witness),
            ('upper', reduction.upper_witness), ('fallback', reduction.fallback_witness)]))


def map_snapshot(problem, *, fix_singletons=False, second_forest=False):
    """Only sparse algebra, no ExactEqualityReduction constructor or optimizer."""
    forest = HomogeneousEqualityReduction.from_problem(problem)
    first = forest.reduce(problem)
    face = ZeroFaceReduction(first.problem, fix_singleton_equalities=fix_singletons)
    config = dict(remove_duplicate_equalities=True, fix_singleton_equalities=fix_singletons,
                  singleton_min_coefficient=1e-12)
    face_keys = dict(coordinate_key=digest_json(dict(
        indices=digest_arrays([('columns', face.columns), ('rows', face.rows),
            ('fixed_columns', face.fixed_columns)]),
        witnesses=[asdict(w) for w in face.witnesses],
        duplicates=[asdict(d) for d in face.duplicate_equalities])),
        fixed_value_key=digest_arrays([('fixed_values', face.fixed_values)]),
        construction_key=digest_json(dict(input=problem_structure(first.problem)[
            'zero_face_input_key_without_config'], config=config)))
    levels = dict(forest=problem_structure(first.problem), zero_face=problem_structure(face.reduced))
    maps = dict(forest=forest_map_keys(forest, first), zero_face=face_keys)
    working = face.reduced
    if second_forest and len(working[4]):
        second = HomogeneousEqualityReduction.from_problem(working)
        second_reduced = second.reduce(working)
        levels['second_forest'] = problem_structure(second_reduced.problem)
        maps['second_forest'] = forest_map_keys(second, second_reduced)
        working = second_reduced.problem
    # Equality proof maps not built: only test exact input identity for a
    # separately proved cache entry. This key is NOT an existence certificate.
    return dict(levels=levels, maps=maps,
        exact_equality_proof_reuse_input_key=equality_fingerprint(working),
        exact_equality_proof_computed=False,
        working_bound_topology_key=bound_topology(working[2], working[3]),
        working_csr_support_key=sparse_key(working[0], include_values=False),
        working_shape=list(working[0].shape), working_neq=int(working[-1]))


def compare_snapshots(old, new):
    if old is None or new is None:
        return None
    result = dict(same_maps={name: {key: value == new['maps'].get(name, {}).get(key)
        for key, value in mapping.items()} for name, mapping in old['maps'].items()},
        same_level_keys={name: {key: value == new['levels'].get(name, {}).get(key)
            for key, value in level.items() if key.endswith('_key')}
            for name, level in old['levels'].items()})
    for key in ('exact_equality_proof_reuse_input_key', 'working_bound_topology_key',
                'working_csr_support_key', 'working_shape', 'working_neq'):
        result['same_' + key] = old[key] == new[key]
    return result


def audit(trace, *, steps=(1, 2, 3), batch=4, stages=STAGES, maps=True,
          fix_singletons=False, second_forest=False):
    trace = Path(trace).resolve()
    if (type(batch) is not int or not 1 <= batch <= 32 or not steps
            or list(steps) != sorted(set(steps)) or any(type(s) is not int or s < 1 for s in steps)
            or not stages or len(set(stages)) != len(stages) or any(s not in STAGES for s in stages)):
        raise ValueError('Bounded batch, increasing unique steps and unique known stages required')
    raw = (trace / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if batch > len(manifest.get('seeds', [])):
        raise ValueError('Distinct recorded environments required')
    records, temporal, cross_environment = [], [], []
    for stage in stages:
        previous = {}
        for step in steps:
            anchor = None
            for environment in range(batch):
                entry = select_entries(manifest, stage, environment, [step])[0]
                problem = _load_problem_without_reference(trace, entry)
                snapshot = map_snapshot(problem, fix_singletons=fix_singletons,
                    second_forest=second_forest) if maps else None
                identity = dict(stage=stage, step=step, environment_id=environment,
                                persistent_owner_key=[stage, environment])
                records.append(dict(**identity, original=problem_structure(problem), maps=snapshot))
                if environment in previous:
                    old_step, old_problem, old_snapshot = previous[environment]
                    temporal.append(dict(stage=stage, environment_id=environment, from_step=old_step,
                        to_step=step, input_difference=problem_difference(old_problem, problem),
                        map_difference=compare_snapshots(old_snapshot, snapshot)))
                if anchor is None:
                    anchor = problem, snapshot
                else:
                    cross_environment.append(dict(stage=stage, step=step, from_environment_id=0,
                        to_environment_id=environment, input_difference=problem_difference(anchor[0], problem),
                        map_difference=compare_snapshots(anchor[1], snapshot)))
                previous[environment] = step, problem, snapshot
    return dict(role='input_only_structure_diagnostic_not_training_or_closed_loop',
        trace_directory=str(trace), input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        configuration=dict(batch=batch, steps=list(steps), stages=list(stages), sparse_maps=maps,
            fix_singleton_equalities=fix_singletons, second_forest=second_forest),
        cpu_lp_calls=0, gpu_calls=0, qr_calls=0, reference_vectors_loaded=False,
        certificate_claimed=False, exact_equality_proof_maps_computed=False,
        caution='Recorded downstream inputs are conditional on their recorded predecessor solves; not a causal replay.',
        records=records, temporal_transitions=temporal, cross_environment_comparisons=cross_environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT / 'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--stages', choices=STAGES, nargs='+', default=list(STAGES))
    parser.add_argument('--skip-maps', action='store_true')
    parser.add_argument('--fix-singleton-equalities', action='store_true')
    parser.add_argument('--second-forest', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace an existing diagnostic result')
    record = audit(args.trace, steps=args.steps, batch=args.batch, stages=args.stages,
        maps=not args.skip_maps, fix_singletons=args.fix_singleton_equalities, second_forest=args.second_forest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), records=len(record['records']),
        temporal_transitions=len(record['temporal_transitions']), cpu_lp_calls=0, gpu_calls=0, qr_calls=0)))


if __name__ == '__main__':
    main()
