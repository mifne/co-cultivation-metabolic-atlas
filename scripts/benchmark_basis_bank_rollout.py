"""Independent-seed short dFBA integration probe, never a five-seed qualification.

Offline training uses CPU solves and is charged separately. The test trajectory
is not added to the bank. Misses stop the run; no CPU optimizer is available to
the GPU online phase. The unchanged original three-LP CPU model is comparator.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil
import traceback
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix,vstack

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment,PHB_REPEAT_G_PER_MMOL,PHV_REPEAT_G_PER_MMOL
from src.community_solver import CooperativeCommunityFbaSolver
from src.fba_surrogate import model_fingerprint
from src.gpu_certified_basis import CommunityCoordinates,compile_basis,factor_compiled_basis
from src.gpu_basis_bank import GpuBasisBank
from src.gpu_compiled_community_backend import GpuCompiledCommunityBackend,lp_arrays,stage_key
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend
from src.compiled_basis_artifact import save_anchor,load_anchor


def environment(seed):
    env=make_environment(None,120,consortium_profile="pf-helper3",initial_nh4=.05)
    env.reset(seed=seed);sim=env.simulator
    layout=CooperativeCommunityFbaSolver(sim.models,original_exchange_bounds=sim.original_bounds,
        maximum_coexistence_growth=.005,optimize_live_objectives=True,parsimonious_exchange=True,highs_method="highs-ds")
    sim._cooperative_solver=layout
    return env,layout


def snapshot(env):
    species=env.simulator.state.species
    phb=sum(s.phb_accumulated for s in species.values());phv=sum(s.phv_accumulated for s in species.values())
    return dict(pha=phb*PHB_REPEAT_G_PER_MMOL+phv*PHV_REPEAT_G_PER_MMOL,
        phv_fraction=phv/(phb+phv) if phb+phv>1e-12 else 0.,
        biomass={k:float(s.biomass) for k,s in species.items()},
        metabolites={k:float(v) for k,v in env.simulator.state.metabolites.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--train-seed",type=int,default=20286311)
    p.add_argument("--test-seed",type=int,default=20286312)
    p.add_argument("--train-steps",type=int,default=4)
    p.add_argument("--test-steps",type=int,default=8)
    p.add_argument("--repair-pivots",type=int,default=0)
    p.add_argument("--bank-cache",type=Path)
    p.add_argument("--optimal-face-tie",action="store_true")
    p.add_argument("--previous-compiler-source",type=Path)
    p.add_argument("--warm-updates",action='store_true')
    p.add_argument("--capture-repair",action='store_true')
    args=p.parse_args()
    if args.train_seed==args.test_seed: raise ValueError("Training and test seeds must differ")
    if args.output.exists(): raise FileExistsError(args.output)
    import cupy as cp
    report=dict(status="offline_training",train_seed=args.train_seed,test_seed=args.test_seed,
        train_steps=args.train_steps,test_steps=args.test_steps,scope="short independent-seed dFBA integration, not final qualification",
        repair_pivot_budget=args.repair_pivots,
        optimal_face_tie=args.optimal_face_tie,
        warm_updates=args.warm_updates,
        capture_repair=args.capture_repair,
        source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ["src/gpu_certified_basis.py","src/gpu_basis_bank.py","src/gpu_revised_basis.py","src/gpu_optimal_face.py","src/gpu_compiled_community_backend.py","src/compiled_basis_artifact.py",
             "src/gpu_capture_math.py","src/gpu_conditional_capture.py","src/gpu_replay_call.py","src/compiled_basis_compatibility.py",
             "src/gpu_exchange_tie_break.py","src/community_solver.py","src/dfba_simulator.py",str(Path(__file__).relative_to(ROOT))]})
    snapshot_dir=args.output.with_suffix(".sources")
    snapshot_dir.mkdir(parents=True,exist_ok=False)
    for filename in report["source_hashes"]:
        destination=snapshot_dir/filename;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/filename,destination)
    report["source_snapshot_directory"]=str(snapshot_dir)
    def save():
        tmp=args.output.with_suffix(".json.tmp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.output)
    save()
    env,layout=environment(args.train_seed)
    fingerprints={k:model_fingerprint(v) for k,v in env.simulator.models.items()}
    meta=dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for model in env.simulator.models.values() for r in model.reactions],
        reaction_species=[k for k,model in env.simulator.models.items() for r in model.reactions])
    report["model_fingerprints"]=fingerprints
    groups=defaultdict(dict);coords=None;offline_calls=0
    def collect(c,**kw):
        nonlocal coords,offline_calls
        arrays=lp_arrays(c,**kw)
        if coords is None: coords=CommunityCoordinates(meta,arrays[0],arrays[-1])
        anchor=compile_basis(*arrays,coordinates=coords,factorize=False)
        offline_calls+=1
        key=stage_key(arrays[4],arrays[0],arrays[-1],layout.n_fluxes)
        signature=hashlib.sha256(b"".join(anchor[k].tobytes() for k in ("basic","active","kind","row_kind"))).hexdigest()
        groups[key].setdefault(signature,anchor)
        return SimpleNamespace(success=True,x=anchor["cpu_anchor_values"],fun=anchor["cpu_anchor_objective"],message="offline CPU anchor")
    def training_solve(c,**kw):
        primary=collect(c,**kw)
        if len(c)>layout.n_fluxes+1:
            objective=np.zeros_like(c)
            aux=[m for m,terms in layout._exchange_terms.items() for _ in terms]
            objective[layout.n_fluxes+1:]=[0. if m in {"h2o_e","h_e","oh1_e"} else 1. for m in aux]
            # Compile the existing GPU fourth-objective selection policy, but
            # retain the CPU primary result for the OFFLINE training trajectory.
            collect(objective,**dict(kw,A_ub=vstack((kw["A_ub"],csr_matrix(np.asarray(c)[None])),format="csr"),
                b_ub=np.r_[kw["b_ub"],primary.fun+1e-10*max(1.,abs(primary.fun))]))
        return primary
    layout.linear_program_backend=SimpleNamespace(name="offline_CPU_training",method="highs-simplex",solve=training_solve)
    actions=np.random.default_rng(args.train_seed).uniform(.05,.95,(120,5)).astype(np.float32)
    started=time.perf_counter()
    cache_exists=args.bank_cache is not None and (args.bank_cache/"manifest.json").exists()
    for step,action in enumerate(actions[:0 if cache_exists else args.train_steps],1):
        env.step(action)
        if not layout.stats.status.startswith("optimal;"): raise RuntimeError(layout.stats.status)
        print(f"offline training {step}/{args.train_steps}; distinct bases {sum(map(len,groups.values()))}",flush=True)
    report.update(offline_cpu_lp_calls=offline_calls,offline_training_seconds=time.perf_counter()-started,
        distinct_bases={str(k):len(v) for k,v in groups.items()},status="offline_factorization")
    save()
    banks={};started=time.perf_counter()
    cache_identity=dict(train_seed=args.train_seed,train_steps=args.train_steps,model_fingerprints=fingerprints,
        compiler_source_sha256=report["source_hashes"]["src/gpu_certified_basis.py"],
        artifact_source_sha256=report["source_hashes"]["src/compiled_basis_artifact.py"])
    manifest=dict(identity=cache_identity,entries=[],original_offline_cpu_lp_calls=offline_calls,
        original_offline_training_seconds=report["offline_training_seconds"])
    if cache_exists:
        manifest=json.loads((args.bank_cache/"manifest.json").read_text())
        previous_identity=dict(manifest["identity"])
        if args.previous_compiler_source is not None:
            from src.compiled_basis_compatibility import equivalent_offline_compiler
            previous_source=args.previous_compiler_source.read_bytes()
            current_source=(ROOT/'src/gpu_certified_basis.py').read_bytes()
            if not equivalent_offline_compiler(previous_source,current_source,previous_identity['compiler_source_sha256']):
                raise ValueError('Offline compiler definitions changed or untrusted previous source')
            report['offline_compiler_AST_equivalent']=True
            report['previous_compiler_source']=str(args.previous_compiler_source)
            previous_identity['compiler_source_sha256']=cache_identity['compiler_source_sha256']
        if previous_identity!=cache_identity:raise ValueError("Compiled bank identity mismatch")
        groups=defaultdict(dict)
        for entry in manifest["entries"]:
            path=(args.bank_cache/entry["filename"]).resolve()
            if not path.is_relative_to(args.bank_cache.resolve()):raise ValueError("Invalid artifact path")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:raise ValueError("Artifact hash mismatch")
            anchor=load_anchor(path);key=tuple(entry["key"])
            groups[key][entry["filename"]]=anchor
        sample=next(iter(next(iter(groups.values())).values()))
        # Stoichiometric equality row membership is invariant under scaling.
        coords=CommunityCoordinates(meta,sample["lp"].a,sample["lp"].neq)
        report["distinct_bases"]={str(k):len(v) for k,v in groups.items()}
        report["cached_training_provenance"]={k:v for k,v in manifest.items() if k!="entries"}
    elif args.bank_cache is not None:
        args.bank_cache.mkdir(parents=True,exist_ok=False)
    for key,anchors in groups.items():
        factored=[]
        for i,a in enumerate(anchors.values(),1):
            factored.append(factor_compiled_basis(a))
            if args.bank_cache is not None and not cache_exists:
                filename=f"anchor_{len(manifest['entries']):04d}.npz"
                path=args.bank_cache/filename;save_anchor(path,factored[-1])
                manifest["entries"].append(dict(key=list(key),filename=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(f"offline factorization {key[0]} {i}/{len(anchors)}",flush=True)
        neq=key[-1];rows=list(range(neq,neq+3))
        if key[0] in {"exchange","exchange_tie"}:rows.append(neq+3+len(meta["exchange_terms"]))
        banks[key]=GpuBasisBank(factored,rows)
    if args.bank_cache is not None and not cache_exists:
        (args.bank_cache/"manifest.json").write_text(json.dumps(manifest,indent=2))
    report.update(offline_factorization_gpu_setup_seconds=time.perf_counter()-started,
        gpu_resident_pool_MiB=cp.get_default_memory_pool().used_bytes()/2**20,status="cpu_test")
    save()
    actions=np.random.default_rng(args.test_seed).uniform(.05,.95,(120,5)).astype(np.float32)
    reference,reference_layout=environment(args.test_seed)
    assert {k:model_fingerprint(v) for k,v in reference.simulator.models.items()}==fingerprints
    cpu_rows=[];started=time.perf_counter()
    for action in actions[:args.test_steps]:
        reference.step(action)
        if not reference_layout.stats.status.startswith("optimal;"):raise RuntimeError(reference_layout.stats.status)
        cpu_rows.append(snapshot(reference))
    report.update(cpu_seconds=time.perf_counter()-started,cpu_rows=cpu_rows,status="gpu_test")
    save()
    test,test_layout=environment(args.test_seed)
    assert {k:model_fingerprint(v) for k,v in test.simulator.models.items()}==fingerprints
    inner=GpuCompiledCommunityBackend(coords,banks,repair_pivots=args.repair_pivots,optimal_face_tie=args.optimal_face_tie,warm_updates=args.warm_updates,capture_repair=args.capture_repair)
    backend=GpuExchangeTieBreakBackend(test_layout,inner=inner)
    test_layout.linear_program_backend=backend
    gpu_rows=[]; failure=None
    def forbidden(*args,**kw):raise AssertionError("Online CPU optimization forbidden")
    # Warm kernels using training inputs only; no test solution enters the bank.
    for bank in banks.values():bank.evaluate_device(**bank.prepare_host([bank.root.anchor["lp"]]))
    cp.cuda.get_current_stream().synchronize()
    started=time.perf_counter()
    with patch("highspy.Highs.run",forbidden),patch("src.community_solver.linprog",forbidden),patch("scipy.optimize.linprog",forbidden):
        for step,action in enumerate(actions[:args.test_steps],1):
            try:
                test.step(action)
                if not test_layout.stats.status.startswith("optimal;"):raise RuntimeError(test_layout.stats.status)
            except (RuntimeError,AssertionError) as error:
                failure=dict(step=step,error=str(error),traceback=traceback.format_exc());break
            gpu_rows.append(snapshot(test))
            print(f"GPU independent-seed test {step}/{args.test_steps}",flush=True)
    cp.cuda.get_current_stream().synchronize()
    report.update(gpu_seconds=time.perf_counter()-started,gpu_rows=gpu_rows,gpu_history=inner.history,
        gpu_outer_history=backend.history,online_cpu_lp_calls=test_layout.cpu_lp_stage_calls,failure=failure,
        status="completed" if failure is None and len(gpu_rows)==args.test_steps else "incomplete_bank_coverage")
    if report["status"]=="completed":
        a,b=cpu_rows[-1],gpu_rows[-1]
        errors=dict(pha_relative=abs(a["pha"]-b["pha"])/max(abs(a["pha"]),1e-9),
            biomass_g_l=max(abs(a["biomass"][k]-b["biomass"][k]) for k in a["biomass"]),
            phv_fraction=abs(a["phv_fraction"]-b["phv_fraction"]))
        report.update(endpoint_errors=errors,short_endpoint_gates_passed=all(v<=.01 for v in errors.values()),
            cpu_over_gpu_time_ratio=report["cpu_seconds"]/report["gpu_seconds"])
    save();print(json.dumps({k:v for k,v in report.items() if k not in {"cpu_rows","gpu_rows","gpu_history","gpu_outer_history"}}),flush=True)


if __name__=="__main__":main()
