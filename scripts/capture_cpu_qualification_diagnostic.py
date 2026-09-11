"""Offline CPU replay of an already tested seed, with exact state/LP capture.

Never changes the frozen GPU qualification or supplies online GPU answers.
Step numbers in this diagnostic are one-based.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment, PHB_REPEAT_G_PER_MMOL, PHV_REPEAT_G_PER_MMOL
from src.community_solver import CooperativeCommunityFbaSolver
from src.fba_surrogate import model_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lp-steps", type=int, nargs="+", default=[16, 17, 18, 113, 114, 115, 116])
    p.add_argument("--backend", choices=["cpu", "tableau", "tableau-exchange-tie"], default="cpu")
    p.add_argument("--stop-after", type=int, default=120)
    args = p.parse_args()
    if not 1 <= args.stop_after <= 120: raise ValueError("stop-after must be 1--120")
    reference = json.loads(args.reference.read_text())
    for name, digest in reference["implementation_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reference implementation changed: {name}")
    seed = reference["runs"][0]["seed"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    actions = np.random.default_rng(seed).uniform(.05, .95, (120, 5)).astype(np.float32)
    env = make_environment(None, 120, consortium_profile="pf-helper3", initial_nh4=.05)
    env.reset(seed=seed)
    sim = env.simulator
    fingerprints = {k:model_fingerprint(v) for k,v in sim.models.items()}
    if fingerprints != reference["runs"][0]["exact"]["model_fingerprints"]:
        raise ValueError("Reference GEM changed")
    solver = CooperativeCommunityFbaSolver(sim.models, original_exchange_bounds=sim.original_bounds,
        maximum_coexistence_growth=.005, optimize_live_objectives=True,
        parsimonious_exchange=True, highs_method="highs-ds")
    sim._cooperative_solver = solver
    reaction_ids = [r.id for model in sim.models.values() for r in model.reactions]
    species_labels = [name for name,model in sim.models.items() for _ in model.reactions]
    metadata = dict(seed=seed, scope="offline_replay_of_already_tested_seed_not_training",
        backend=args.backend, stop_after=args.stop_after,
        step_indexing="one_based", reference=str(args.reference), model_fingerprints=fingerprints,
        implementation_sha256=reference["implementation_sha256"],
        reaction_ids=reaction_ids, reaction_species=species_labels,
        growth_terms=solver._growth_terms, exchange_terms=solver._exchange_terms)
    metadata["diagnostic_source_sha256"] = {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ("scripts/capture_cpu_qualification_diagnostic.py", "src/gpu_exchange_tie_break.py")}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rows, calls = [], []
    started = time.perf_counter()
    step = 0
    stage = 0
    backend = None
    if args.backend == "tableau":
        from src.gpu_lexicographic_lp import PresolvedCuOptBackend
        backend = PresolvedCuOptBackend(method="tableau", tolerance=1e-8, time_limit=600)
    elif args.backend == "tableau-exchange-tie":
        from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend
        backend = GpuExchangeTieBreakBackend(solver, tolerance=1e-8, time_limit=600)
    def save_checkpoint(status, error=None):
        payload = dict(metadata=metadata, status=status, error=error,
            elapsed_seconds=time.perf_counter()-started, rows=rows, calls=calls,
            max_pha_replay_difference=max((r["pha_reference_difference"] for r in rows), default=None),
            max_biomass_replay_difference=max((r["biomass_reference_difference"] for r in rows), default=None),
            backend_history=None if backend is None else backend.history)
        temporary = args.output_dir / "replay.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(args.output_dir / "replay.json")
    def record(c, **kw):
        nonlocal stage
        stage += 1
        result = linprog(c, **kw) if backend is None else backend.solve(c, **kw)
        if not result.success:
            save_checkpoint("failed_lp", f"step={step}, stage={stage}: {result.message}")
            raise RuntimeError(f"Diagnostic LP failed: {result.message}")
        if step in args.lp_steps:
            eq, ub = csr_matrix(kw["A_eq"]), csr_matrix(kw["A_ub"])
            matrix = vstack((eq, ub), format="csr")
            bounds = kw["bounds"]
            solution_fields = (dict(cpu_x=result.x, cpu_objective=np.asarray(result.fun),
                cpu_lower_marginals=result.lower.marginals,cpu_upper_marginals=result.upper.marginals)
                if backend is None else dict(gpu_x=result.x,gpu_objective=np.asarray(result.fun)))
            np.savez_compressed(args.output_dir / f"step_{step:03d}_stage_{stage}.npz",
                c=np.asarray(c), data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                shape=np.asarray(matrix.shape), neq=np.asarray(eq.shape[0]),
                rhs=np.r_[kw["b_eq"],kw["b_ub"]],
                lower=np.asarray([-np.inf if a is None else a for a,b in bounds]),
                upper=np.asarray([np.inf if b is None else b for a,b in bounds]),**solution_fields)
            calls.append(dict(step=step, stage=stage, success=result.success, objective=float(result.fun)))
        return result
    if backend is not None:
        from types import SimpleNamespace
        solver.linear_program_backend = SimpleNamespace(name=backend.name,method=backend.method,solve=record)
    with patch("src.community_solver.linprog", record):
        for step, action in enumerate(actions[:args.stop_after], 1):
            stage = 0
            before = dict(metabolites={k:float(v) for k,v in sim.state.metabolites.items()},
                biomass={k:float(v.biomass) for k,v in sim.state.species.items()})
            _, _, terminated, truncated, _ = env.step(action)
            phb = sum(s.phb_accumulated for s in sim.state.species.values())
            phv = sum(s.phv_accumulated for s in sim.state.species.values())
            after = dict(metabolites={k:float(v) for k,v in sim.state.metabolites.items()},
                biomass={k:float(v.biomass) for k,v in sim.state.species.items()},
                pha=phb*PHB_REPEAT_G_PER_MMOL+phv*PHV_REPEAT_G_PER_MMOL,
                phv_fraction=phv/(phb+phv) if phb+phv>1e-12 else 0.)
            ref = reference["runs"][0]["exact"]["trajectory"][step-1]
            row = dict(step=step, before=before, after=after, status=solver.stats.status,
                pha_reference_difference=abs(after["pha"]-ref["pha"]),
                biomass_reference_difference=max(abs(after["biomass"][k]-ref["biomass"][k]) for k in after["biomass"]))
            rows.append(row)
            if backend is not None:
                original_gpu = reference["runs"][0]["gpu"]["trajectory"][step-1]
                row["original_gpu_pha_difference"] = abs(after["pha"]-original_gpu["pha"])
                row["original_gpu_biomass_difference"] = max(abs(after["biomass"][k]-original_gpu["biomass"][k]) for k in after["biomass"])
                row["original_gpu_nh4_difference"] = abs(after["metabolites"].get("nh4_e",0)-original_gpu["nh4"])
            if step % 5 == 0:
                save_checkpoint("running")
                print(f"offline {args.backend} replay {step}/{args.stop_after}", flush=True)
            if not solver.stats.status.startswith("optimal;") or terminated or truncated: break
    save_checkpoint("completed" if len(rows) == args.stop_after else "incomplete")
    print("Offline replay saved", flush=True)


if __name__ == "__main__": main()
