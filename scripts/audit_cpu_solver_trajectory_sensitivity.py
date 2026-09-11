"""Same biological LP and actions, alternate CPU solver: offline sensitivity only."""
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
from src.fba_surrogate import model_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["highs-ds", "highs-ipm"], required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference.read_text())
    for name, digest in reference["implementation_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reference implementation changed: {name}")
    ref_run = reference["runs"][0]
    seed = ref_run["seed"]
    env = make_environment(None, 120, consortium_profile="pf-helper3", initial_nh4=.05)
    env.reset(seed=seed)
    sim = env.simulator
    fingerprints = {k: model_fingerprint(v) for k, v in sim.models.items()}
    if fingerprints != ref_run["exact"]["model_fingerprints"]:
        raise ValueError("Reference GEM changed")
    solver = CooperativeCommunityFbaSolver(sim.models,
        original_exchange_bounds=sim.original_bounds, maximum_coexistence_growth=.005,
        optimize_live_objectives=True, parsimonious_exchange=True, highs_method=args.method)
    sim._cooperative_solver = solver
    actions = np.random.default_rng(seed).uniform(.05, .95, (120, 5)).astype(np.float32)
    result = dict(scope="post-test CPU solver sensitivity, not new GPU qualification",
        seed=seed, method=args.method, reference=str(args.reference),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        implementation_sha256=reference["implementation_sha256"],
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_fingerprints=fingerprints, status="running", rows=[])
    started = time.perf_counter()
    for step, action in enumerate(actions, 1):
        _, _, terminated, truncated, _ = env.step(action)
        phb = sum(s.phb_accumulated for s in sim.state.species.values())
        phv = sum(s.phv_accumulated for s in sim.state.species.values())
        after = dict(metabolites={k: float(v) for k, v in sim.state.metabolites.items()},
            biomass={k: float(v.biomass) for k, v in sim.state.species.items()},
            pha=phb*PHB_REPEAT_G_PER_MMOL+phv*PHV_REPEAT_G_PER_MMOL,
            phv_fraction=phv/(phb+phv) if phb+phv > 1e-12 else 0.)
        ref = ref_run["exact"]["trajectory"][step-1]
        errors = dict(pha_relative=abs(after["pha"]-ref["pha"])/max(abs(ref["pha"]), 1e-9),
            biomass_g_l=max(abs(after["biomass"][k]-ref["biomass"][k]) for k in after["biomass"]),
            phv_fraction=abs(after["phv_fraction"]-ref["phv_fraction"]))
        accepted = "stage=parsimonious_exchange" in solver.stats.status
        result["rows"].append(dict(step=step, after=after, errors=errors,
            accepted=accepted, solver_status=solver.stats.status))
        if step % 5 == 0:
            result["elapsed_seconds"] = time.perf_counter()-started
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(step=step, errors=errors)), flush=True)
        if not accepted or terminated or truncated:
            break
    result["elapsed_seconds"] = time.perf_counter()-started
    result["status"] = "completed" if len(result["rows"]) == 120 and all(r["accepted"] for r in result["rows"]) else "incomplete"
    result["endpoint_errors"] = result["rows"][-1]["errors"]
    result["within_original_endpoint_gates"] = (result["status"] == "completed"
        and all(v <= .01 for v in result["endpoint_errors"].values()))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
