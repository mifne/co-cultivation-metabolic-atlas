"""Separate numerical probe; do not mutate the frozen qualification backend."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.gpu_lexicographic_lp import PresolvedCuOptBackend
from scripts.benchmark_gpu_lexicographic import run


class OriginalBoundPresolvedBackend(PresolvedCuOptBackend):
    name="original_bounds_presolved_cuopt_probe"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        original_solve=self.engine.solve
        def restored(c,**kw):
            bound=[]
            for name,(lo,hi) in zip(kw["column_ids"],kw["bounds"]):
                index=int(name[1:])
                original_lo,original_hi=self.original_bounds[index]
                bound.append((max(lo,-np.inf if original_lo is None else original_lo),
                              min(hi,np.inf if original_hi is None else original_hi)))
            return original_solve(c,**dict(kw,bounds=bound))
        self.engine.solve=restored
    def solve(self,c,**kw):
        self.original_bounds=kw["bounds"]
        return super().solve(c,**kw)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method",default="hybrid")
    p.add_argument("--augmented",type=int,default=0)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    actions=np.random.default_rng(20260913).uniform(.05,.95,(3,5)).astype(np.float32)
    exact=run(actions,20260913)
    backend=OriginalBoundPresolvedBackend(method=args.method,augmented=args.augmented,
        presolve=0,tolerance=1e-7,time_limit=30,log_to_console=True)
    gpu=run(actions,20260913,backend)
    args.output.write_text(json.dumps(dict(exact=exact,gpu=gpu),indent=2))
    print("completed",gpu["steps"],all(r["accepted"] for r in gpu["trajectory"]),flush=True)


if __name__=="__main__": main()
