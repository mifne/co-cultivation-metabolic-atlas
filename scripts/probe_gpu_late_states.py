"""Offline late-state LP regression, never an online CPU fallback or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.augment_cooperative_surrogate_dagger import label_exact_context_chunk
from src.gpu_lexicographic_lp import PresolvedCuOptBackend


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,nargs="+",default=[90,119])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device-pivots",action="store_true")
    args=p.parse_args()
    data=np.load(ROOT/"results/pf_multi_same_state_baseline.npz")
    if any(not np.any(data["steps"]==step) for step in args.steps):
        raise ValueError(f"Requested state not saved; available: {np.unique(data['steps']).tolist()}")
    backend=PresolvedCuOptBackend(method="tableau",tolerance=1e-8,time_limit=120,log_to_console=True)
    if args.device_pivots:
        from src.gpu_device_bounded_simplex import GpuDeviceBoundedSimplex
        backend.engine=GpuDeviceBoundedSimplex(tolerance=1e-8,time_limit=120,log_to_console=True)
        backend.validation_engine=backend.engine
    report=dict(scope="offline_CPU_relabelled_GPU_visited_contexts_not_trajectory_qualification",records=[],
        device_pivots=args.device_pivots,
        implementation_sha256=hashlib.sha256((ROOT/"src/gpu_bounded_simplex.py").read_bytes()).hexdigest())
    for step in args.steps:
        index=np.flatnonzero(data["steps"]==step)[0]
        context=data["contexts"][index:index+1]
        calls=[]
        def record(c,**kw):
            cpu=linprog(c,**kw)
            calls.append((np.asarray(c),kw,cpu))
            return cpu
        with patch("src.community_solver.linprog",record):
            label_exact_context_chunk(context,"pf-helper3")
        for stage,(c,kw,cpu) in enumerate(calls,1):
            gpu=backend.solve(c,**kw)
            row=dict(step=step,stage=stage,cpu_success=cpu.success,gpu_success=gpu.success,
                cpu_objective=float(cpu.fun),gpu_objective=gpu.fun,
                objective_abs_error=abs(cpu.fun-gpu.fun) if gpu.success else None,
                history=backend.history[-1])
            report["records"].append(row)
            args.output.write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!="history"}),flush=True)
            if not gpu.success: return


if __name__=="__main__": main()
