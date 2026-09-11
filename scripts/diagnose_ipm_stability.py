"""Development-only failed Newton-system audit; no CPU LP or reference x/y.

CPU sparse LU here solves a saved *linear equation*, exclusively for diagnosis.
It never supplies a warm start, a runtime fallback, or a dFBA state update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import select_entries
from src.gpu_batched_ipm import GpuBatchedIPM


def magnitude(a):
    a=np.abs(np.asarray(a))
    nonzero=a[a>0]
    return dict(maximum=float(a.max(initial=0.)), minimum_nonzero=float(nonzero.min()) if len(nonzero) else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='exchange',choices=['maxmin','aggregate','exchange'])
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--regularization',type=float,default=1e-9)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=ROOT/'results/pf_coverage_holdout4x120_20260905'
    manifest=json.loads((root/'manifest.json').read_text())
    entries=[select_entries(manifest,args.stage,i,[args.step])[0] for i in range(4)]
    problems=[_load_problem_without_reference(root,e) for e in entries]
    report=dict(role='development_newton_diagnostic_not_training_or_fallback', stage=args.stage,
        step=args.step, entries=entries, cpu_lp_calls=0, cpu_linear_diagnostic_calls=0,
        regularization=args.regularization, reference_vectors_loaded=False,
        input_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        sources={p:(ROOT/p).read_text() for p in ('src/gpu_batched_ipm.py','src/gpu_sparse_factor.py',
                                                'scripts/diagnose_ipm_stability.py')})
    with GpuBatchedIPM(problems,regularization=args.regularization) as solver:
        result=solver.solve(iterations=80,capture_failure=True)
        report.update(status=result['status'],iterations=result['iterations'],metrics=result['metrics'],
                      gpu_seconds=result['total_seconds'])
        snapshot=solver.failure_snapshot
        report['environments']=[]
        if snapshot:
            host={k:v.get() for k,v in snapshot.items()}
            offsets=solver.factor.indptr.get();indices=solver.factor.indices.get()
            for i in range(4):
                a=csr_matrix((host['kkt_values'][i],indices,offsets),shape=(solver.size,solver.size))
                rhs=host['rhs'][i]
                gpu=host['answer'][i]
                before=time.perf_counter()
                lu=splu(a.tocsc())
                cpu=lu.solve(rhs)
                # Three ordinary CPU IR steps, kept out of the GPU runtime.
                for _ in range(3):cpu+=lu.solve(rhs-a@cpu)
                elapsed=time.perf_counter()-before
                slices=[slice(0,solver.n),slice(solver.n,solver.n+solver.ne),slice(solver.n+solver.ne,None)]
                report['cpu_linear_diagnostic_calls']+=4
                report['environments'].append(dict(env=i,cpu_linear_seconds=elapsed,
                    gpu_absolute_residual=magnitude(rhs-a@gpu),cpu_absolute_residual=magnitude(rhs-a@cpu),
                    gpu_residual_by_block=[magnitude((rhs-a@gpu)[s]) for s in slices],
                    cpu_residual_by_block=[magnitude((rhs-a@cpu)[s]) for s in slices],
                    rhs_by_block=[magnitude(rhs[s]) for s in slices],
                    direction_by_block=[magnitude(gpu[s]) for s in slices],
                    current_rd=magnitude(host['rd'][i]),current_rp=magnitude(host['rp'][i]),
                    current_rg=magnitude(host['rg'][i]),s=magnitude(host['s'][i]),z=magnitude(host['z'][i]),
                    ratio=magnitude(host['s'][i]/host['z'][i]),
                    fixed_variables=int(len(solver.fixed)),equalities=solver.ne,inequalities=solver.ng))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','entries')},indent=2))


if __name__=='__main__':main()
