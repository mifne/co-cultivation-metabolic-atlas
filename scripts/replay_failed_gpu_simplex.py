"""Replay an exact saved GPU LP failure, without calling a CPU optimizer."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.gpu_bounded_simplex import GpuBoundedSimplex

def main():
    p=argparse.ArgumentParser()
    p.add_argument("snapshot",type=Path)
    p.add_argument("--cold",action="store_true")
    p.add_argument("--repair",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    d=np.load(args.snapshot)
    meta=json.loads(str(d["metadata_json"]))
    a=csr_matrix((d["data"],d["indices"],d["indptr"]),shape=tuple(d["shape"]))
    neq=int(d["neq"])
    backend=GpuBoundedSimplex(tolerance=1e-8,time_limit=120,log_to_console=True)
    previous=meta.pop("previous")
    if not args.cold: backend.basis_cache[meta["stage_key"]]=previous
    result=backend.solve(d["c"],A_eq=a[:neq],b_eq=d["rhs"][:neq],
        A_ub=a[neq:],b_ub=d["rhs"][neq:],bounds=list(zip(d["lower"],d["upper"])),**meta)
    if args.repair:
        for _ in range(3):
            if result.success: break
            if backend.numerical_repair_basis is None: break
            backend.basis_cache[meta["stage_key"]]=backend.numerical_repair_basis
            backend.last_basis=backend.numerical_repair_basis
            result=backend.solve(d["c"],A_eq=a[:neq],b_eq=d["rhs"][:neq],
                A_ub=a[neq:],b_ub=d["rhs"][neq:],bounds=list(zip(d["lower"],d["upper"])),**meta)
    args.output.write_text(json.dumps(dict(success=result.success,history=backend.history,
        snapshot=str(args.snapshot),cold=args.cold),indent=2))
    if backend.failure_snapshot is not None:
        np.savez_compressed(args.output.with_suffix(".npz"),**backend.failure_snapshot)

if __name__=="__main__": main()
