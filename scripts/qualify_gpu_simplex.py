#!/usr/bin/env python3
"""Frozen five-seed verification; independent processes share one GPU.

Parallel wall times are not used as a CPU/GPU speed comparison. This script
refuses to begin until a full development trajectory passes every gate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SEEDS=list(range(20286001,20286006))
SUPPORTING_SOURCES=("scripts/benchmark_cooperative_surrogate_e2e.py", "src/rl_environment.py",
                    "src/utils.py", "src/metabolite_ids.py", "src/fba_surrogate.py")


def supporting_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SUPPORTING_SOURCES}


def environment_info():
    versions={}
    for name in ("numpy","scipy","cobra","highspy","torch","stable-baselines3","gymnasium",
                 "cuopt-cu12","cuopt-cu13","cupy-cuda12x","cupy-cuda13x","cupy"):
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: pass
    try:
        gpu=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"],capture_output=True,text=True,timeout=10,check=True).stdout.strip()
    except (OSError,subprocess.SubprocessError) as exc: gpu=f"unavailable: {exc!r}"
    return dict(python=sys.version,packages=versions,gpu_query_at_start=gpu,
                numerical_threads=dict(OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=3)
    args=p.parse_args()
    dev=json.loads(args.development.read_text())
    if (dev["config"]["steps"]!=120 or not dev["runs"] or
            not all(r.get("passed",False) for r in dev["runs"])):
        raise RuntimeError("Full development trajectory has not passed")
    expected=dict(method="tableau",host_presolve=True,rank_reduce=False,objective_scale=1.,
                  quadratic_regularization=0.,reaction_support=None)
    if any(dev["config"].get(k)!=v for k,v in expected.items()):
        raise RuntimeError("Development configuration differs from this qualification backend")
    for row in dev["runs"]:
        for side in ("exact","gpu"):
            result=row.get(side,{})
            trace=result.get("trajectory",[])
            if result.get("steps")!=120 or len(trace)!=120 or not all(t.get("accepted",False) for t in trace):
                raise RuntimeError("Complete accepted trajectories required; a pass flag alone is insufficient")
        cpu,gpu=row["exact"],row["gpu"]
        if any("stage=parsimonious_exchange" not in t.get("status","")
               for t in cpu["trajectory"]+gpu["trajectory"]):
            raise RuntimeError("Every reference and GPU step must complete all three LP stages")
        if gpu["cpu_lp_stage_calls"]!=0 or gpu["gpu_lp_stage_calls"]!=360:
            raise RuntimeError("Development LP counters do not qualify")
        if cpu["model_fingerprints"]!=gpu["model_fingerprints"]:
            raise RuntimeError("Development CPU/GPU GEM mismatch")
        a,b=cpu["trajectory"][-1],gpu["trajectory"][-1]
        errors=[abs(a["pha"]-b["pha"])/max(abs(a["pha"]),1e-9),
            max(abs(a["biomass"][k]-b["biomass"][k]) for k in a["biomass"]),
            abs(a["phv_fraction"]-b["phv_fraction"])]
        if not all(0<=error<=.01 for error in errors):
            raise RuntimeError("Recomputed development errors do not qualify")
    for name,digest in dev["implementation_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Implementation changed since development: {name}")
    if not 1<=args.workers<=5: raise ValueError("workers must be 1--5")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    # Parallel scheduling increases wall time, not the required accuracy.
    # Fix this operational limit before seeing any final-seed outcome.
    solver_time_limit=dev["config"]["time_limit"]*args.workers
    supporting=supporting_hashes()
    manifest=dict(status="running_frozen_test",seeds=SEEDS,development=str(args.development),
        started_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        environment=environment_info(),supporting_source_sha256=supporting,
        workload=dict(consortium="pf-helper3",initial_nh4_mM=.05,dt_h=.2,steps=120,
            action_dimensions=5,action_uniform_range=[.05,.95],maximum_common_growth_h_inverse=.005),
        implementation_sha256=dev["implementation_sha256"],
        config=dict(dev["config"],time_limit=solver_time_limit,seeds=SEEDS,cpu_reference=None,
                    output=str(args.output_dir)),
        concurrency=args.workers,
        timing_note=("Contended verification; not a speed benchmark" if args.workers>1 else
                     "Sequential independent verification; not a separately controlled speed benchmark"),
        gates=dict(pha_relative=.01,biomass_g_l=.01,phv_mole_fraction=.01,steps=120,
                   cpu_lp_calls=0,original_lp_feasibility_residual=1e-5),
        runs=[])
    output=args.output_dir/"qualification.json"
    output.write_text(json.dumps(manifest,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    def run(seed):
        if supporting_hashes()!=supporting:
            return dict(seed=seed,passed=False,error="supporting workload source changed before run")
        path=args.output_dir/f"seed_{seed}.json"
        command=[sys.executable,str(ROOT/"scripts/benchmark_gpu_lexicographic.py"),
            "--steps","120","--seeds",str(seed),"--method","tableau","--host-presolve",
            "--tolerance",str(dev["config"]["tolerance"]),"--time-limit",str(solver_time_limit),
            "--log-solver","--output",str(path)]
        started=time.perf_counter()
        with (args.output_dir/f"seed_{seed}.log").open("w") as log:
            try:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                return dict(seed=seed,passed=False,error="verification interrupted by an external timeout")
        if result.returncode or not path.exists():
            return dict(seed=seed,passed=False,error=f"exit code {result.returncode}")
        data=json.loads(path.read_text())
        if supporting_hashes()!=supporting:
            return dict(seed=seed,passed=False,error="supporting workload source changed during run")
        if data["implementation_sha256"]!=dev["implementation_sha256"]:
            return dict(seed=seed,passed=False,error="frozen implementation mismatch")
        row=data["runs"][0]
        if row["seed"]!=seed:
            return dict(seed=seed,passed=False,error="validation seed mismatch")
        if row["exact"]["model_fingerprints"]!=dev["runs"][0]["exact"]["model_fingerprints"]:
            return dict(seed=seed,passed=False,error="frozen GEM model mismatch")
        if any("stage=parsimonious_exchange" not in t.get("status","")
               for side in ("exact","gpu") for t in row[side]["trajectory"]):
            return dict(seed=seed,passed=False,error="incomplete three-stage reference or GPU solve",report=str(path))
        return dict(seed=seed,passed=row["passed"],errors=row["errors"],report=str(path),
                    report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    seconds=time.perf_counter()-started)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run,seed):seed for seed in SEEDS}
        for job in as_completed(jobs):
            try: row=job.result()
            except Exception as exc:
                row=dict(seed=jobs[job],passed=False,error=repr(exc))
            manifest["runs"].append(row)
            output.write_text(json.dumps(manifest,indent=2))
            print(json.dumps(row),flush=True)
    manifest["runs"].sort(key=lambda x:x["seed"])
    manifest["status"]="five_seed_accuracy_pass" if all(x["passed"] for x in manifest["runs"]) else "failed"
    manifest["finished_at_utc"]=datetime.now(timezone.utc).isoformat(timespec="seconds")
    output.write_text(json.dumps(manifest,indent=2))
    print(manifest["status"],flush=True)


if __name__=="__main__": main()
