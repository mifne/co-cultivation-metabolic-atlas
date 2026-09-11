#!/usr/bin/env python3
"""Serial identical-action CPU vs exact-formulation GPU dFBA qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment, PHB_REPEAT_G_PER_MMOL, PHV_REPEAT_G_PER_MMOL
from src.community_solver import CooperativeCommunityFbaSolver
from src.gpu_lexicographic_lp import CuOptLexicographicBackend, HybridCuOptLexicographicBackend, PresolvedCuOptBackend
from src.fba_surrogate import model_fingerprint


def ppo_telemetry(state,observation,reward,action,terminated,truncated,info):
    """Raw task-level quantities; threshold flags describe the END state.

    These are not claims about the branch actually executed earlier in the
    transition. PHA repeat amount and mass must not be conflated.
    """
    repeat=sum(float(v.pha_accumulated) for v in state.species.values())
    concentrations={str(k):float(v) for k,v in state.metabolites.items()}
    biomass=[float(v.biomass) for v in state.species.values()]
    return dict(observation=np.asarray(observation,dtype=float).tolist(),
        reward_raw=float(reward),action=np.asarray(action,dtype=float).tolist(),
        terminated=bool(terminated),truncated=bool(truncated),
        pha_repeat_mmol_l=repeat,ph=float(info['ph']),do_mmol_l=float(info['do']),
        metabolites_mmol_l=concentrations,
        state_thresholds_after_step=dict(all_biomass_below_001=all(x<.01 for x in biomass),
            any_biomass_below_002=any(x<.02 for x in biomass),
            nh4_below_01=concentrations.get('nh4_e',0.)<.1,
            rubber_intermediate_present=any(concentrations.get(k,0.)>1e-12
                                           for k in ('C30_oligo_e','odtd_e')),
            oxygen_at_or_below_zero=concentrations.get('o2_e',0.)<=0.))


def run(actions, seed, backend=None):
    env = make_environment(None, len(actions), consortium_profile="pf-helper3", initial_nh4=.05)
    env.reset(seed=seed)
    sim = env.simulator
    initial_fingerprints = {k:model_fingerprint(v) for k,v in sim.models.items()}
    sim._cooperative_solver = CooperativeCommunityFbaSolver(sim.models,
        original_exchange_bounds=sim.original_bounds, maximum_coexistence_growth=.005,
        optimize_live_objectives=True, parsimonious_exchange=True, highs_method="highs-ds",
        linear_program_backend=backend)
    rows = []
    started = time.perf_counter()
    for step, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        solver = sim._cooperative_solver
        accepted = solver.stats.status.startswith("optimal;")
        state = sim.state
        phb = sum(v.phb_accumulated for v in state.species.values())
        phv = sum(v.phv_accumulated for v in state.species.values())
        rows.append(dict(step=step, accepted=accepted, status=solver.stats.status,
            biomass={k:float(v.biomass) for k,v in state.species.items()},
            pha=phb*PHB_REPEAT_G_PER_MMOL+phv*PHV_REPEAT_G_PER_MMOL,
            phv_fraction=phv/(phb+phv) if phb+phv > 1e-12 else 0.0,
            nh4=float(state.metabolites.get("nh4_e", 0.0)),
            **ppo_telemetry(state,observation,reward,action,terminated,truncated,info)))
        if (step+1) % 10 == 0 or not accepted:
            print(f"{seed} {'CPU' if backend is None else backend.method} step {step+1}/{len(actions)}: "
                  f"PHA={rows[-1]['pha']:.8f}, accepted={accepted}", flush=True)
        if terminated or truncated or not accepted:
            break
    return dict(seconds=time.perf_counter()-started, steps=len(rows), trajectory=rows,
        cpu_lp_stage_calls=solver.cpu_lp_stage_calls, gpu_lp_stage_calls=solver.gpu_lp_stage_calls,
        model_fingerprints=initial_fingerprints,
        telemetry_scope='raw_environment_quantities_not_policy_or_GAE_qualification',
        telemetry_units=dict(pha='g/L',pha_repeat_mmol_l='mmol/L',biomass='g/L',
            metabolites_mmol_l='mmol/L',phv_fraction='mol/mol',ph='pH'),
        backend_history=None if backend is None else backend.history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260913])
    p.add_argument("--method", choices=["barrier", "pdlp", "hybrid", "tableau"], default="barrier")
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--presolve", type=int, default=2)
    p.add_argument("--augmented", type=int, default=1)
    p.add_argument("--objective-scale", type=float, default=1.)
    p.add_argument("--rank-reduce", action="store_true")
    p.add_argument("--host-presolve", action="store_true")
    p.add_argument("--quadratic-regularization", type=float, default=0.)
    p.add_argument("--pdlp-mode", choices=["Stable3", "Methodical1", "Fast1"], default="Stable3")
    p.add_argument("--reaction-support", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--log-solver", action="store_true")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--cpu-reference", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Qualification diagnostics never overwrite existing results')
    report = dict(status="in_progress", config=vars(args).copy(), runs=[],
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("src/community_solver.py", "src/gpu_lexicographic_lp.py",
                         "src/dfba_simulator.py", "src/gpu_bounded_simplex.py", "scripts/benchmark_gpu_lexicographic.py")})
    report["config"]["output"] = str(args.output)
    report["config"]["cpu_reference"] = str(args.cpu_reference) if args.cpu_reference else None
    report["config"]["reaction_support"] = str(args.reaction_support) if args.reaction_support else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        actions = np.random.default_rng(seed).uniform(.05, .95, (args.steps, 5)).astype(np.float32)
        if args.cpu_reference:
            reference = json.loads(args.cpu_reference.read_text())
            for name in ("src/community_solver.py", "src/dfba_simulator.py"):
                if reference["implementation_sha256"][name] != report["implementation_sha256"][name]:
                    raise ValueError(f"CPU reference implementation changed: {name}")
            if reference["config"]["steps"] != args.steps:
                raise ValueError("CPU reference step count mismatch")
            matches = [r["exact"] for r in reference["runs"] if r["seed"] == seed]
            if len(matches) != 1:
                raise ValueError("CPU reference seed mismatch")
            exact = matches[0]
            report["timing_note"] = "CPU reference reused; no matched-load speedup claim"
        else:
            exact = run(actions, seed)
        if args.cpu_only:
            report["runs"].append(dict(seed=seed, exact=exact))
            report["status"] = "cpu_reference_only"
            args.output.write_text(json.dumps(report, indent=2))
            continue
        factory = HybridCuOptLexicographicBackend if args.method == "hybrid" else CuOptLexicographicBackend
        if args.host_presolve:
            factory = PresolvedCuOptBackend
        backend = factory(**({} if args.method == "hybrid" and not args.host_presolve else {"method":args.method}), tolerance=args.tolerance,
            time_limit=args.time_limit, presolve=args.presolve, log_to_console=args.log_solver,
            augmented=args.augmented, objective_scale=args.objective_scale, rank_reduce=args.rank_reduce,
            quadratic_regularization=args.quadratic_regularization, pdlp_mode=args.pdlp_mode,
            reaction_support=np.load(args.reaction_support)["support"] if args.reaction_support else None)
        try:
            gpu = run(actions, seed, backend)
        except Exception as exc:
            report["status"] = "execution_error"
            report["runs"].append(dict(seed=seed, exact=exact, error=repr(exc), backend_history=backend.history))
            args.output.write_text(json.dumps(report, indent=2))
            raise
        same_horizon = exact["steps"] == gpu["steps"] == args.steps
        # Partial failures are never labelled as same-horizon endpoint errors.
        a, b = exact["trajectory"][gpu["steps"]-1], gpu["trajectory"][-1]
        errors = dict(pha_relative=abs(a["pha"]-b["pha"])/max(abs(a["pha"]), 1e-9),
            biomass_max=max(abs(a["biomass"][k]-b["biomass"][k]) for k in a["biomass"]),
            phv_fraction=abs(a["phv_fraction"]-b["phv_fraction"]))
        passed = (same_horizon and not args.reaction_support and
            exact["model_fingerprints"] == gpu["model_fingerprints"] and
            all(r["accepted"] for r in exact["trajectory"]+gpu["trajectory"]) and
            gpu["cpu_lp_stage_calls"] == 0 and gpu["gpu_lp_stage_calls"] == 3*args.steps and
            errors["pha_relative"] <= .01 and errors["biomass_max"] <= .01 and errors["phv_fraction"] <= .01)
        failed_problem = getattr(getattr(backend,"engine",backend),"failure_snapshot",None)
        if failed_problem is not None:
            dump=args.output.with_name(args.output.stem+f"_seed{seed}_failed_lp.npz")
            np.savez_compressed(dump,**failed_problem)
            report.setdefault("failed_lp_snapshots",[]).append(str(dump))
        report["runs"].append(dict(seed=seed, exact=exact, gpu=gpu, errors=errors, passed=passed,
            error_scope="full_horizon_endpoint" if same_horizon else "partial_horizon_diagnostic_not_qualification"))
        report["status"] = "development_pass" if all(r["passed"] for r in report["runs"]) else "failed"
        if set(args.seeds) == set(range(20286001, 20286006)) and len(report["runs"]) == 5 and args.steps == 120:
            report["status"] = "five_seed_accuracy_pass" if all(r["passed"] for r in report["runs"]) else "failed"
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(seed=seed, errors=errors, passed=passed,
            cpu_seconds=exact["seconds"], gpu_seconds=gpu["seconds"])), flush=True)


if __name__ == "__main__":
    main()
