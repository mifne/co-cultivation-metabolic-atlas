"""Strict paired ablation of common host/CPU runtime options, not GPU-only gains."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_pipeline_run import summarize_pipeline_report


def cpu_rows(history):
    rows = []
    for batch in history:
        # Some CPU dictionary batches expose the same rows both directly and
        # grouped by dispatch route. These are two views, not two LP solves.
        if 'rows' in batch:
            rows.extend(batch['rows'])
        else:
            rows.extend(row for group in batch.get('groups', [])
                        if group.get('route') == 'cpu_fallback' for row in group.get('rows', []))
    return rows


def compare(before, after):
    summaries = [summarize_pipeline_report(r) for r in (before, after)]
    cfg0, cfg1 = before['configuration'], after['configuration']
    flags = ('frozen_inputs', 'cpu_exchange_support_updates')
    changes = {flag:[cfg0.get(flag, False), cfg1.get(flag, False)] for flag in flags
               if cfg0.get(flag, False) != cfg1.get(flag, False)}
    if not changes or any(pair != [False, True] for pair in changes.values()):
        raise ValueError('Expected one or both common runtime options disabled then enabled')
    ignored = {'output', *flags}
    if {k:v for k,v in cfg0.items() if k not in ignored} != {k:v for k,v in cfg1.items() if k not in ignored}:
        raise ValueError('Unrelated configuration changed')
    for key in ('source_hashes','model_fingerprints','offline_bank_manifest','repair_operator_manifest'):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError('Missing or changed provenance: '+key)
    if len(before['runs']) != len(after['runs']):
        raise ValueError('Repeat counts changed')
    pairs = []
    for old, new in zip(before['runs'], after['runs']):
        if any(old[k] != new[k] for k in ('seeds','repeat','execution_order')):
            raise ValueError('Seed or execution order changed')
        errors = {}
        for side in ('cpu','gpu'):
            differences = []
            metabolite_changes = []
            for a,b in zip(old[side+'_rows'], new[side+'_rows']):
                if set(a['biomass']) != set(b['biomass']):
                    raise ValueError('Endpoint species identity changed')
                # The simulator materializes previously absent medium keys
                # on secretion, including roundoff-sized values. Report these
                # explicitly; model identity is verified above. Its absent
                # concentration convention is zero, not a missing observation.
                am,bm = a['metabolites'],b['metabolites']
                union = set(am)|set(bm)
                metabolite_changes.append(dict(
                    max_abs_mmol_l=max((abs(am.get(k,0.)-bm.get(k,0.)) for k in union),default=0.),
                    added={k:bm[k] for k in sorted(set(bm)-set(am))},
                    removed={k:am[k] for k in sorted(set(am)-set(bm))}))
                differences.append(dict(pha_relative=abs(a['pha']-b['pha'])/max(abs(a['pha']),1e-9),
                    biomass_g_l=max(abs(a['biomass'][k]-b['biomass'][k]) for k in a['biomass']),
                    phv_fraction=abs(a['phv_fraction']-b['phv_fraction'])))
            errors[side] = {key:max(row[key] for row in differences) for key in differences[0]}
            errors[side]['metabolite_diagnostics'] = metabolite_changes
        if any(side[key] > .01 for side in errors.values() for key in ('pha_relative','biomass_g_l','phv_fraction')):
            raise ValueError('Option change exceeds original endpoint criteria')
        diagnostics = {}
        for side in ('cpu','gpu'):
            diagnostics[side] = []
            for run in (old,new):
                rows = cpu_rows(run[side+'_history'])
                diagnostics[side].append(dict(cpu_requests=len(rows),
                    model_rebuilds=sum(bool(r['model_rebuilt']) for r in rows),
                    exchange_rebuilds=sum(bool(r['model_rebuilt']) for r in rows if r['stage']=='exchange'),
                    support_updates=sum(bool(r.get('exchange_support_updated')) for r in rows),
                    simplex_iterations=sum(r['simplex_iterations'] for r in rows)))
        pairs.append(dict(repeat=old['repeat'], seeds=old['seeds'],
            cpu_seconds=[old['cpu_seconds'],new['cpu_seconds']],
            hybrid_seconds=[old['gpu_seconds'],new['gpu_seconds']],
            common_cpu_time_reduction=1-new['cpu_seconds']/old['cpu_seconds'],
            common_hybrid_time_reduction=1-new['gpu_seconds']/old['gpu_seconds'],
            optimized_cpu_over_hybrid=new['cpu_seconds']/new['gpu_seconds'],
            endpoint_changes=errors, cpu_diagnostics=diagnostics))
    return dict(scope='Common host/CPU changes applied to both CPU and hybrid; '
        'descriptive paired development measurements; no CI or GPU-only speed attribution',
        changed_options=changes, pairs=pairs, validated_run_summaries=summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    report['sources'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.before,args.after,Path(__file__))}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['pairs'], indent=2))


if __name__ == '__main__':
    main()
