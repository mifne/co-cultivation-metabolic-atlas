"""Short real-GEM test of GPU basis reuse and GPU-only bounded repair.

Offline CPU compilation and transfer-inclusive online timing are reported
separately. This is an LP-component experiment, not end-to-end dFBA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_certified_basis import load
from src.gpu_certified_basis import CommunityCoordinates, compile_basis
from src.gpu_revised_basis import GpuRevisedBasis


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stages",nargs="+",type=int,default=[1,2,3])
    p.add_argument("--steps",nargs="+",type=int,default=[16,17,18])
    p.add_argument("--pivots",type=int,default=24)
    p.add_argument("--diagnostics",action="store_true")
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    import cupy as cp
    meta=json.loads((ROOT/"results/pf_seed20286001_cpu_diagnostic/metadata.json").read_text())
    report=dict(status="running",scope="original three CPU LP objectives; GPU repair component, not full dFBA",
        pivot_budget=args.pivots,source_hashes={str(f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
            for f in ["src/gpu_certified_basis.py","src/gpu_revised_basis.py","scripts/benchmark_revised_basis.py"]},records=[])
    def save():
        tmp=args.output.with_suffix(".json.tmp"); tmp.write_text(json.dumps(report,indent=2)); tmp.replace(args.output)
    save()
    for stage in args.stages:
        d,a,path=load(16,stage); neq=int(d["neq"])
        coords=CommunityCoordinates(meta,a,neq)
        anchor=compile_basis(a,d["rhs"],d["lower"],d["upper"],d["c"],neq,coords)
        rows=list(range(neq,neq+3))
        if stage==3: rows.append(neq+3+len(meta["exchange_terms"]))
        started=time.perf_counter(); solver=GpuRevisedBasis(anchor,rows,max_pivots=args.pivots)
        cp.cuda.get_current_stream().synchronize(); setup=time.perf_counter()-started
        problems=[]; expected=[]
        for step in args.steps:
            query,matrix,_=load(step,stage)
            problems.append(coords.normalize(matrix,query["rhs"],query["lower"],query["upper"],query["c"],int(query["neq"])))
            expected.append(float(query["cpu_objective"]))
        data=solver.prepare_host(problems)
        # One unmeasured warm-up on anchor only; compilation is reported.
        solver.solve_device(**solver.prepare_host([anchor["lp"]]))
        cp.cuda.get_current_stream().synchronize()
        started=time.perf_counter(); result=solver.solve_device(**data,diagnostics=args.diagnostics)
        cp.cuda.get_current_stream().synchronize(); resident=time.perf_counter()-started
        cases=[]
        if args.diagnostics:
            np.savez_compressed(args.output.with_suffix(f".stage{stage}.npz"),
                **{k:result[k].get() for k in ("trace","basis","raw_values","raw_kind","raw_y")})
        for i,step in enumerate(args.steps):
            row=dict(step=step,saved_cpu_objective=expected[i])
            for k in ["accepted","objective","primal_residual","dual_violation","relative_kkt_gap","pivots"]:
                value=result[k].get()[i].item()
                row[k]=value if not isinstance(value,float) or np.isfinite(value) else None
            if row["accepted"]: row["objective_absolute_error"]=abs(row["objective"]-expected[i])
            cases.append(row)
        entry=dict(stage=stage,offline_cpu_lp_calls=1,compilation_seconds=anchor["compilation_seconds"],
            gpu_setup_seconds=setup,resident_batch_seconds=resident,batch_size=len(problems),cases=cases,
            gpu_pool_MiB=cp.get_default_memory_pool().used_bytes()/2**20,online_cpu_lp_calls=0)
        report["records"].append(entry);save();print(json.dumps(entry),flush=True)
        del solver,data,result,anchor,problems
        cp.get_default_memory_pool().free_all_blocks()
    report["status"]="completed";save()


if __name__=="__main__":main()
