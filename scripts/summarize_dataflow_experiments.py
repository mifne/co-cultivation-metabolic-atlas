"""Strict summary of measured pipelines, including source snapshots and spans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_pipeline_run import summarize_pipeline_report,NONOVERLAP_FIELDS


def summarize(path):
    path=Path(path);raw=json.loads(path.read_text());summary=summarize_pipeline_report(raw)
    directory=path.with_suffix('.sources').resolve()
    for name,digest in raw['source_hashes'].items():
        file=(directory/name).resolve()
        if not file.is_relative_to(directory) or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('Source snapshot mismatch: '+name)
    stages=[s for r in raw['runs'] for s in r['gpu_history'] if s['stage']=='maxmin']
    phase_rows=[s['gpu_host_observed_phases'] for s in stages if 'gpu_host_observed_phases' in s]
    for s in stages:
        phase=s.get('gpu_host_observed_phases')
        if phase is not None and not np.isclose(sum(v for k,v in phase.items() if k.endswith('_seconds')),
                s['gpu_probe_span_seconds'],rtol=1e-9,atol=1e-9):
            raise ValueError('GPU host phase accounting mismatch')
    # Each trajectory's first step is excluded from warm diagnostic medians,
    # but remains in the total wall comparison and all LP counters.
    warm=[s['gpu_host_observed_phases'] for r in raw['runs']
        for s in [v for v in r['gpu_history'] if v['stage']=='maxmin'][1:]
        if 'gpu_host_observed_phases' in s]
    nonoverlap={side:{key:sum(s[key] for r in raw['runs'] for s in r[side+'_pipeline_history'])
        for key in NONOVERLAP_FIELDS} for side in ('cpu','gpu')}
    o=summary['overall']
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=summary['status'],configuration=raw['configuration'],
        source_snapshot_files_verified=len(raw['source_hashes']),
        overall=o,nonoverlap_seconds=nonoverlap,
        gpu_warm_host_phase_medians={key:float(np.median([row.get(key,0.) for row in warm]))
            for key in set().union(*(r.keys() for r in warm)) if key.endswith('_seconds')},
        runs=[dict(repeat=r['repeat'],seeds=r['seeds'],execution_order=r['execution_order'],
            cpu_seconds=r['cpu_seconds'],gpu_seconds=r['gpu_seconds'],
            gpu_maxmin_accepts=sum(v['bank_accepts'] for v in r['gpu_history'] if v['stage']=='maxmin'),
            cpu_cancelled=sum(v.get('cpu_speculative_cancelled',0) for v in r['gpu_history']))
            for r in raw['runs']])


def main():
    p=argparse.ArgumentParser();p.add_argument('reports',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[summarize(path) for path in args.reports]
    report=dict(status='validated',scope='Descriptive measured end-to-end comparisons. No statistical significance, long-horizon, or GPU-only claim.',
        experiments=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    for row in rows:
        o=row['overall']
        print(json.dumps(dict(file=row['file'],wall=o['wall_seconds'],ratio=o['cpu_over_hybrid_ratio'],
            gpu_accepts=o['gpu_maxmin_bank_accepts'],cancelled=o['gpu_total_cpu_speculative_cancelled'],
            unused=o['gpu_total_cpu_speculative_unused'],errors=o['endpoint_error_maxima'],
            nonoverlap=row['nonoverlap_seconds'],warm_phases=row['gpu_warm_host_phase_medians']),indent=2),flush=True)


if __name__=='__main__':main()
