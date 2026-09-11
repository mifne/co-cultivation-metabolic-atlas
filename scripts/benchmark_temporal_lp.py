"""Causal single-stage LP replay; NOT a closed-loop dFBA GPU speed claim.

Inputs of the selected stage were produced by a CPU dFBA trajectory, including
its upstream lexicographic stages. Current reference solutions are scoring
targets only. Every method maintains its own accepted previous-solution state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.temporal_lp_data import load_trajectories, TemporalLPCodec
from src.lp_trace import problem_request
from src.cpu_repeated_lp import RepeatedCpuLP
from scripts.benchmark_compact_gpu import json_finite_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--environments',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--check-interval',type=int,default=100)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--primal-weight',type=float,default=1.)
    parser.add_argument('--equality-reduction',action='store_true',
                        help='Exact fixed equality elimination with original-LP GPU certification')
    parser.add_argument('--resident-execution',choices=['loop','graph'],
                        help='Reuse fused GPU workspace; requires --equality-reduction')
    parser.add_argument('--graph-chunk',type=int,default=64)
    parser.add_argument('--modes',nargs='+',choices=['cold','previous','mean','mlp','gru'],default=['cold','previous','mean','mlp','gru'])
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.steps,args.environments,args.workers,args.check_interval) < 1 or args.iterations < 0:
        raise ValueError('Invalid replay budget')
    if args.resident_execution and not args.equality_reduction:
        raise ValueError('--resident-execution requires --equality-reduction')
    report = dict(status='initializing',configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope='Single-stage LP replay on CPU-generated states/upstream stages; not closed-loop dFBA, '
              'not all-GPU qualification; no current/previous CPU solutions in GPU warm starts',
        cpu_comparator='Persistent HiGHS with 4 workers by default; no offline dictionary initialization',
        results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(json_finite_values(report),indent=2,allow_nan=False))
        temp.replace(args.output)
    try:
        save()
        codec = TemporalLPCodec.load(args.artifact/'codec.npz')
        manifest, trajectories, sha = load_trajectories(args.trace,codec.metadata['stage'])
        if manifest['model_fingerprints'] != codec.metadata['model_fingerprints']:
            raise ValueError('Model provenance mismatch')
        used = set(codec.metadata['train_seeds']) | set(codec.metadata['development_seeds'])
        if used & set(manifest['seeds']):
            raise ValueError('Evaluation trajectories overlap training/development seeds')
        if len(trajectories) < args.environments or min(map(len,trajectories)) < args.steps:
            raise ValueError('Incomplete evaluation cohort/horizon')
        trajectories = [trajectory[:args.steps] for trajectory in trajectories[:args.environments]]
        report.update(trace_sha256=sha,trace_role=manifest['role'],seeds=manifest['seeds'][:args.environments],
            model_fingerprints=manifest['model_fingerprints'],
            artifact_sha256={f:hashlib.sha256((args.artifact/f).read_bytes()).hexdigest()
                             for f in ('codec.npz','manifest.json','mlp.pt','gru.pt')})
        source_dir = args.output.with_suffix('.sources');source_dir.mkdir(exist_ok=False)
        report['source_hashes'] = {}
        sources = ['scripts/benchmark_temporal_lp.py','src/temporal_gpu_lp.py','src/temporal_lp_data.py',
                   'src/temporal_lp_model.py','src/gpu_pdhg_corrector.py','src/gpu_block_lp.py','src/cpu_repeated_lp.py']
        if args.equality_reduction:
            sources += ['src/lp_equality_reduction.py','src/gpu_reduced_pdhg.py']
        if args.resident_execution:
            sources += ['src/gpu_pdhg_workspace.py','src/gpu_resident_reduced_pdhg.py']
        for f in sources:
            target = source_dir/f; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/f,target);report['source_hashes'][f]=hashlib.sha256(target.read_bytes()).hexdigest()
        report['status']='running';save()
        from src.temporal_gpu_lp import TemporalGpuLPBackend
        for mode in ['cpu',*args.modes]:
            before = time.perf_counter()
            backend = RepeatedCpuLP(args.workers) if mode=='cpu' else TemporalGpuLPBackend(
                args.artifact,mode=mode,iterations=args.iterations,check_interval=args.check_interval,
                primal_weight=args.primal_weight,equality_reduction=args.equality_reduction,
                resident_execution=args.resident_execution,graph_chunk=args.graph_chunk)
            row = dict(mode=mode,backend_setup_seconds=time.perf_counter()-before,steps=[])
            try:
                for t in range(args.steps):
                    items = [trajectory[t] for trajectory in trajectories]
                    requests = [problem_request(p,stage=codec.metadata['stage']) for p,_,_ in items]
                    before = time.perf_counter()
                    solved = backend.solve_batch(requests)
                    seconds = time.perf_counter()-before
                    errors = [abs(r.fun-float(p[4]@ref)) if r.success else None
                              for (p,ref,_),r in zip(items,solved)]
                    row['steps'].append(dict(step=t+1,seconds=seconds,accepted=sum(r.success for r in solved),
                        reference_objective_errors=errors,diagnostics=backend.history[-1]))
                    if t == 0 or (t+1)%10 == 0:
                        print(f'{mode} step {t+1}: accepted {sum(r.success for r in solved)}/{len(solved)}, {seconds:.3f}s',flush=True)
                row['total_seconds'] = sum(r['seconds'] for r in row['steps'])
                row['accepted'] = sum(r['accepted'] for r in row['steps'])
                row['requested'] = args.steps*args.environments
                row['all_accepted'] = row['accepted']==row['requested']
                if mode!='cpu' and row['all_accepted'] and report['results'][0]['all_accepted']:
                    row['local_persistent_cpu_over_gpu_ratio'] = report['results'][0]['total_seconds']/row['total_seconds']
                report['results'].append(row);save()
            finally:
                if mode=='cpu':backend.close()
                del backend
        report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save()


if __name__=='__main__':main()
