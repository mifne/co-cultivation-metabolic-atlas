#!/usr/bin/env python3
"""Equal-action end-to-end exact vs GPU cooperative-dictionary benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from cobra.io import read_sbml_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.utils import (
    get_initial_params,
    load_sbml_models,
    select_consortium_models,
    select_or16_ns21_models,
)


CONSORTIUM_PROFILES = ("legacy3", "or16-ns21", "pf-helper3")
PHB_REPEAT_G_PER_MMOL = 86.09 / 1000.0
PHV_REPEAT_G_PER_MMOL = 100.12 / 1000.0
HELPER_MODEL = (
    PROJECT_ROOT / "models" / "sbml" / "helper_candidates"
    / "Propionibacterium_freudenreichii_shermanii_curated.xml"
)


def load_consortium_profile(profile: str):
    if profile not in CONSORTIUM_PROFILES:
        raise ValueError(f"unknown consortium profile: {profile}")
    loaded = load_sbml_models(
        PROJECT_ROOT / "models" / "sbml" / "final_consortium"
    )
    if profile == "legacy3":
        return select_consortium_models(loaded)
    models = select_or16_ns21_models(loaded)
    if profile == "pf-helper3":
        models["Propionibacterium_freudenreichii_shermanii"] = (
            read_sbml_model(str(HELPER_MODEL))
        )
    return models


def make_environment(
    artifact: Path | None,
    steps: int,
    top_k: int = 16,
    interpolation_k: int = 1,
    exact_interval: int = 0,
    gpu_qp_projection: bool = False,
    gpu_qp_only: bool = False,
    gpu_qp_candidates: int = 64,
    gpu_qp_rerank_pool: int | None = None,
    gpu_qp_decision_strength: float | None = None,
    gpu_qp_blend_candidates: int = 1,
    gpu_qp_blend_distance_power: float = 2.0,
    gpu_qp_match_pha: bool = False,
    consortium_profile: str = "legacy3",
    initial_nh4: float | None = None,
    gpu_qp_max_iterations: int = 400,
    gpu_qp_multioutput_strength: float = 0.0,
    gpu_qp_independent_species: bool = False,
) -> ConsortiumEnv:
    models = load_consortium_profile(consortium_profile)
    biomass, metabolites = get_initial_params(models)
    if initial_nh4 is not None:
        metabolites["nh4_e"] = max(0.0, float(initial_nh4))
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=metabolites,
        dt=0.2,
        solver_backend="highs",
        fba_mode="cooperative",
        cooperative_optimize_live_objectives=True,
        cooperative_parsimony=True,
        cooperative_highs_method="highs-ds",
        cooperative_surrogate_artifact=None if artifact is None else str(artifact),
        cooperative_surrogate_device="cuda",
        cooperative_surrogate_top_k=top_k,
        cooperative_surrogate_interpolation_k=interpolation_k,
        cooperative_surrogate_exact_interval=exact_interval,
        cooperative_surrogate_require_qualified=False,
        cooperative_gpu_qp_projection=gpu_qp_projection,
        cooperative_gpu_qp_only=gpu_qp_only,
        cooperative_gpu_qp_candidates=gpu_qp_candidates,
        cooperative_gpu_qp_rerank_pool=gpu_qp_rerank_pool,
        cooperative_gpu_qp_decision_strength=gpu_qp_decision_strength,
        cooperative_gpu_qp_blend_candidates=gpu_qp_blend_candidates,
        cooperative_gpu_qp_blend_distance_power=gpu_qp_blend_distance_power,
        cooperative_gpu_qp_match_pha=gpu_qp_match_pha,
        cooperative_gpu_qp_max_iterations=gpu_qp_max_iterations,
        cooperative_gpu_qp_multioutput_strength=gpu_qp_multioutput_strength,
        cooperative_gpu_qp_independent_species=gpu_qp_independent_species,
        ph_control_target=6.5,
    )
    return ConsortiumEnv(
        simulator=simulator,
        max_time=(steps + 1) * 0.2,
        max_common_feed_early=0.02,
        max_common_feed_late=0.01,
        max_specific_feed_per_step=0.1,
    )


def run_case(
    artifact: Path | None,
    actions: np.ndarray,
    top_k: int = 16,
    interpolation_k: int = 1,
    exact_interval: int = 0,
    gpu_qp_projection: bool = False,
    gpu_qp_only: bool = False,
    gpu_qp_candidates: int = 64,
    gpu_qp_rerank_pool: int | None = None,
    gpu_qp_decision_strength: float | None = None,
    gpu_qp_blend_candidates: int = 1,
    gpu_qp_blend_distance_power: float = 2.0,
    gpu_qp_match_pha: bool = False,
    consortium_profile: str = "legacy3",
    initial_nh4: float | None = None,
    gpu_qp_max_iterations: int = 400,
    gpu_qp_multioutput_strength: float = 0.0,
    gpu_qp_independent_species: bool = False,
) -> dict:
    implementation_hashes = {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/community_solver.py", "src/gpu_batch_qp.py",
            "src/gpu_multioutput_qp.py",
            "src/cooperative_neural_surrogate.py", "src/dfba_simulator.py",
            "scripts/benchmark_cooperative_surrogate_e2e.py",
        )
    }
    if artifact is not None:
        import torch

        torch.cuda.reset_peak_memory_stats()
    env = make_environment(
        artifact,
        len(actions),
        top_k=top_k,
        interpolation_k=interpolation_k,
        exact_interval=exact_interval,
        gpu_qp_projection=gpu_qp_projection,
        gpu_qp_only=gpu_qp_only,
        gpu_qp_candidates=gpu_qp_candidates,
        gpu_qp_rerank_pool=gpu_qp_rerank_pool,
        gpu_qp_decision_strength=gpu_qp_decision_strength,
        gpu_qp_blend_candidates=gpu_qp_blend_candidates,
        gpu_qp_blend_distance_power=gpu_qp_blend_distance_power,
        gpu_qp_match_pha=gpu_qp_match_pha,
        consortium_profile=consortium_profile,
        initial_nh4=initial_nh4,
        gpu_qp_max_iterations=gpu_qp_max_iterations,
        gpu_qp_multioutput_strength=gpu_qp_multioutput_strength,
        gpu_qp_independent_species=gpu_qp_independent_species,
    )
    observation, _ = env.reset(seed=20260901)
    started = time.perf_counter()
    reward_sum = 0.0
    info = {}
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - started
    state = env.simulator.state
    result = {
        "implementation_sha256": implementation_hashes,
        "elapsed_seconds": elapsed,
        "steps": int(env.simulator.current_step),
        "steps_per_second": env.simulator.current_step / elapsed,
        "reward_sum": reward_sum,
        "observation": np.asarray(observation, dtype=float).tolist(),
        "biomass_g_l": {
            name: float(value.biomass) for name, value in state.species.items()
        },
        "pha_g_l": float(
            sum(
                value.phb_accumulated * PHB_REPEAT_G_PER_MMOL
                + value.phv_accumulated * PHV_REPEAT_G_PER_MMOL
                for value in state.species.values()
            )
        ),
        "phb_mmol_l": float(
            sum(value.phb_accumulated for value in state.species.values())
        ),
        "phv_mmol_l": float(
            sum(value.phv_accumulated for value in state.species.values())
        ),
        "rubber_remaining_g_l": float(state.rubber_concentration),
        "solver": env.simulator.get_solver_diagnostics(),
        "last_info": info,
    }
    repeat_units = result["phb_mmol_l"] + result["phv_mmol_l"]
    result["phv_mol_fraction"] = (
        result["phv_mmol_l"] / repeat_units if repeat_units > 1e-12 else None
    )
    if artifact is not None:
        import torch

        result["gpu_memory_allocated_mib"] = float(
            torch.cuda.memory_allocated() / 2**20
        )
        result["gpu_peak_memory_allocated_mib"] = float(
            torch.cuda.max_memory_allocated() / 2**20
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--interpolation-k",
        type=int,
        default=1,
        help="convexly interpolate this many feasible neighbours (1 is legacy nearest)",
    )
    parser.add_argument(
        "--exact-interval",
        type=int,
        default=0,
        help="force an exact HiGHS solve every N steps (0 disables re-anchoring)",
    )
    parser.add_argument("--gpu-qp-projection", action="store_true")
    parser.add_argument("--gpu-qp-only", action="store_true")
    parser.add_argument("--gpu-qp-candidates", type=int, default=64)
    parser.add_argument("--gpu-qp-rerank-pool", type=int, default=None)
    parser.add_argument("--gpu-qp-decision-strength", type=float, default=None)
    parser.add_argument("--gpu-qp-blend-candidates", type=int, default=1)
    parser.add_argument("--gpu-qp-blend-distance-power", type=float, default=2.0)
    parser.add_argument("--gpu-qp-match-pha", action="store_true")
    parser.add_argument("--gpu-qp-max-iterations", type=int, default=400)
    parser.add_argument("--gpu-qp-multioutput-strength", type=float, default=0.0)
    parser.add_argument("--gpu-qp-independent-species", action="store_true")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--initial-nh4", type=float, default=None)
    parser.add_argument(
        "--consortium",
        choices=CONSORTIUM_PROFILES,
        default="legacy3",
    )
    parser.add_argument(
        "--exact-reference",
        type=Path,
        default=None,
        help="reuse the exact row from a prior identical-seed benchmark report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "cooperative_surrogate_e2e.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(0.05, 0.95, size=(args.steps, 5)).astype(np.float32)
    if args.exact_reference is None:
        exact = run_case(
            None, actions, top_k=args.top_k,
            consortium_profile=args.consortium,
            initial_nh4=args.initial_nh4,
        )
    else:
        reference = json.loads(args.exact_reference.read_text(encoding="utf-8"))
        if int(reference["steps"]) != args.steps:
            raise ValueError("exact reference step count does not match")
        if int(reference["seed"]) != args.seed:
            raise ValueError("exact reference seed does not match")
        if reference.get("consortium_profile", "legacy3") != args.consortium:
            raise ValueError("exact reference consortium does not match")
        if reference.get("initial_nh4_mmol_l") != args.initial_nh4:
            raise ValueError("exact reference initial NH4 does not match")
        exact = reference["exact"]
    surrogate = run_case(
        args.artifact,
        actions,
        top_k=args.top_k,
        interpolation_k=max(1, args.interpolation_k),
        exact_interval=max(0, args.exact_interval),
        gpu_qp_projection=args.gpu_qp_projection,
        gpu_qp_only=args.gpu_qp_only,
        gpu_qp_candidates=max(2, args.gpu_qp_candidates),
        gpu_qp_rerank_pool=args.gpu_qp_rerank_pool,
        gpu_qp_decision_strength=args.gpu_qp_decision_strength,
        gpu_qp_blend_candidates=args.gpu_qp_blend_candidates,
        gpu_qp_blend_distance_power=args.gpu_qp_blend_distance_power,
        gpu_qp_match_pha=args.gpu_qp_match_pha,
        consortium_profile=args.consortium,
        initial_nh4=args.initial_nh4,
        gpu_qp_max_iterations=args.gpu_qp_max_iterations,
        gpu_qp_multioutput_strength=args.gpu_qp_multioutput_strength,
        gpu_qp_independent_species=args.gpu_qp_independent_species,
    )
    report = {
        "benchmark": "exact_vs_gpu_aligned_cooperative_dictionary",
        "artifact": str(args.artifact),
        "consortium_profile": args.consortium,
        "steps": args.steps,
        "seed": args.seed,
        "initial_nh4_mmol_l": args.initial_nh4,
        "top_k": args.top_k,
        "interpolation_k": max(1, args.interpolation_k),
        "exact_interval": max(0, args.exact_interval),
        "gpu_qp_projection": bool(args.gpu_qp_projection),
        "gpu_qp_only": bool(args.gpu_qp_only),
        "gpu_qp_candidates": max(2, args.gpu_qp_candidates),
        "gpu_qp_rerank_pool": args.gpu_qp_rerank_pool,
        "gpu_qp_decision_strength": args.gpu_qp_decision_strength,
        "gpu_qp_blend_candidates": args.gpu_qp_blend_candidates,
        "gpu_qp_blend_distance_power": args.gpu_qp_blend_distance_power,
        "gpu_qp_match_pha": bool(args.gpu_qp_match_pha),
        "gpu_qp_max_iterations": args.gpu_qp_max_iterations,
        "gpu_qp_multioutput_strength": args.gpu_qp_multioutput_strength,
        "gpu_qp_independent_species":args.gpu_qp_independent_species,
        "exact": exact,
        "surrogate": surrogate,
        "speedup": exact["elapsed_seconds"] / surrogate["elapsed_seconds"],
        "max_observation_abs_difference": float(
            np.max(
                np.abs(
                    np.asarray(exact["observation"])
                    - np.asarray(surrogate["observation"])
                )
            )
        ),
        "reward_abs_difference": abs(exact["reward_sum"] - surrogate["reward_sum"]),
        "pha_abs_difference_g_l": abs(exact["pha_g_l"] - surrogate["pha_g_l"]),
        "rubber_abs_difference_g_l": abs(
            exact["rubber_remaining_g_l"] - surrogate["rubber_remaining_g_l"]
        ),
        "max_biomass_abs_difference_g_l": max(
            abs(exact["biomass_g_l"][name] - surrogate["biomass_g_l"][name])
            for name in exact["biomass_g_l"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        key: report[key]
        for key in (
            "speedup", "max_observation_abs_difference", "reward_abs_difference",
            "pha_abs_difference_g_l", "rubber_abs_difference_g_l",
            "max_biomass_abs_difference_g_l",
        )
    }, indent=2))
    print(json.dumps(surrogate["solver"]["cooperative"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
