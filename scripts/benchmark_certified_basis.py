"""Offline-compiled GPU basis evaluation on real saved GEM LPs.

Reports compile cost, acceptance, GPU resident and transfer-inclusive timing.
No claim of full dFBA speedup or successful unseen-state coverage is inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.gpu_certified_basis import CommunityCoordinates, compile_basis, GpuBasisEvaluator
from scripts.cpu_basis_comparator import CpuBasisComparator


def load(step,stage):
    path=ROOT/f"results/pf_seed20286001_cpu_diagnostic/step_{step:03d}_stage_{stage}.npz"
    d=np.load(path)
    a=csr_matrix((d["data"],d["indices"],d["indptr"]),shape=tuple(d["shape"]))
    return d,a,path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stages",type=int,nargs="+",default=[1,2,3])
    p.add_argument("--batch-sizes",type=int,nargs="+",default=[1,8,32])
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    import cupy as cp
    meta=json.load(open(ROOT/"results/pf_seed20286001_cpu_diagnostic/metadata.json"))
    report=dict(status="running",scope="offline compiled anchor; saved-state kernel experiment, not end-to-end dFBA",records=[],
        source_hashes={str(f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ["src/gpu_certified_basis.py","scripts/benchmark_certified_basis.py","scripts/cpu_basis_comparator.py"]})
    def save():
        temporary=args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report,indent=2)); temporary.replace(args.output)
    save()
    for stage in args.stages:
        d,a,path=load(16,stage); neq=int(d["neq"])
        coordinates=CommunityCoordinates(meta,a,neq)
        anchor=compile_basis(a,d["rhs"],d["lower"],d["upper"],d["c"],neq,coordinates)
        # Three growth rows plus the retained-production row when present.
        variable=list(range(neq,neq+3))
        if stage==3: variable.append(neq+3+len(meta["exchange_terms"]))
        evaluator=GpuBasisEvaluator(anchor,variable)
        cpu_same=CpuBasisComparator(anchor,variable)
        entry=dict(stage=stage,compilation_seconds=anchor["compilation_seconds"],
            offline_cpu_lp_calls=1,basis_size=len(anchor["basic"]),inverse_density=anchor["inverse_density"],
            inverse_CSR_MiB=(anchor["inverse"].data.nbytes+anchor["inverse"].indices.nbytes+anchor["inverse"].indptr.nbytes)/2**20,
            cases=[],timings=[],anchor_lp_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report["records"].append(entry); save()
        for step in (16,17,18,113,114,115,116):
            query,matrix,_=load(step,stage)
            problem=coordinates.normalize(matrix,query["rhs"],query["lower"],query["upper"],query["c"],int(query["neq"]))
            try:
                result=evaluator.evaluate_device(**evaluator.prepare_host([problem]))
                accepted=bool(result["accepted"].get()[0])
                entry["cases"].append(dict(step=step,accepted=accepted,objective=float(result["objective"].get()[0]) if accepted else None,
                    saved_cpu_objective=float(query["cpu_objective"]),primal=float(result["primal_residual"].get()[0]),
                    dual=float(result["dual_violation"].get()[0]),gap=float(result["relative_kkt_gap"].get()[0])))
            except ValueError as error:
                entry["cases"].append(dict(step=step,accepted=False,structure_error=str(error)))
            save()
        # Perturb RHS around the anchor without supplying new CPU answers to GPU.
        rng=np.random.default_rng(20286301+stage)
        anchor_norm=anchor["lp"]
        for batch_size in args.batch_sizes:
            problems=[]
            for i in range(batch_size):
                rhs=d["rhs"].copy()
                rhs[neq+3:neq+3+len(meta["exchange_terms"]) ] *= rng.uniform(.999,1.001,len(meta["exchange_terms"]))
                problems.append(coordinates.normalize(a,rhs,d["lower"],d["upper"],d["c"],neq))
            device=evaluator.prepare_host(problems)
            for _ in range(2): evaluator.evaluate_device(**device)
            cp.cuda.get_current_stream().synchronize()
            times=[]
            for _ in range(5):
                started=time.perf_counter(); result=evaluator.evaluate_device(**device)
                cp.cuda.get_current_stream().synchronize(); times.append(time.perf_counter()-started)
            started=time.perf_counter()
            end_to_end=evaluator.evaluate_device(**evaluator.prepare_host(problems))
            host_values=end_to_end["values"].get(); acceptance=end_to_end["accepted"].get()
            transfer_seconds=time.perf_counter()-started
            cpu_started=time.perf_counter(); cpu_values=[]
            for problem in problems:
                cpu=linprog(problem.c,A_eq=problem.a[:neq],b_eq=problem.rhs[:neq],
                    A_ub=problem.a[neq:],b_ub=problem.rhs[neq:],bounds=list(zip(problem.lower,problem.upper)),method="highs-ds")
                cpu_values.append(float(cpu.fun) if cpu.success else None)
            cpu_seconds=time.perf_counter()-cpu_started
            cpu_data=cpu_same.prepare_host(problems)
            for _ in range(2): cpu_same.evaluate_host(**cpu_data)
            cpu_reuse_times=[]
            for _ in range(5):
                started=time.perf_counter(); cpu_reuse=cpu_same.evaluate_host(**cpu_data)
                cpu_reuse_times.append(time.perf_counter()-started)
            cpu_valid=cpu_reuse["accepted"]
            joint_valid=cpu_valid&acceptance
            same_errors=np.abs(cpu_reuse["objective"][joint_valid]-end_to_end["objective"].get()[joint_valid])
            values=result["objective"].get()
            errors=[abs(values[i]-cpu_values[i]) for i in range(batch_size) if acceptance[i] and cpu_values[i] is not None]
            timing=dict(batch_size=batch_size,accepted=int(acceptance.sum()),resident_seconds=times,
                transfer_inclusive_seconds=transfer_seconds,cpu_sequential_seconds=cpu_seconds,
                max_objective_error=max(errors,default=None),cpu_comparator="SciPy HiGHS dual simplex on equivalent normalized LP",
                cpu_same_algorithm_batched_seconds=cpu_reuse_times,cpu_same_algorithm_accepted=int(cpu_valid.sum()),
                same_algorithm_objective_error=float(same_errors.max()) if same_errors.size else None,
                excludes="offline compilation and state/LP assembly; no correction for rejected queries")
            entry["timings"].append(timing); save(); print(json.dumps(dict(stage=stage,**timing)),flush=True)
        del evaluator,cpu_same,cpu_data,anchor
        cp.get_default_memory_pool().free_all_blocks()
    report["status"]="completed"; save()


if __name__=="__main__": main()
