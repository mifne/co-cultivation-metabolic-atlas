#!/usr/bin/env python3
"""Benchmark exact cooperative FBA HiGHS options without changing biology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.utils import get_initial_params, load_sbml_models, select_consortium_models


def make_environment(method: str, presolve: bool, max_time: float) -> ConsortiumEnv:
    models = select_consortium_models(
        load_sbml_models(PROJECT_ROOT / "models" / "sbml" / "final_consortium")
    )
    biomass, metabolites = get_initial_params(models)
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=metabolites,
        dt=0.2,
        solver_backend="highs",
        fba_mode="cooperative",
        cooperative_parsimony=True,
        cooperative_highs_presolve=presolve,
        cooperative_highs_method=method,
        ph_control_target=6.5,
    )
    return ConsortiumEnv(
        simulator=simulator,
        max_time=max_time,
        max_common_feed_early=0.02,
        max_common_feed_late=0.01,
        max_specific_feed_per_step=0.1,
    )


def run_case(method: str, presolve: bool, actions: np.ndarray) -> dict:
    env = make_environment(method, presolve, max_time=(len(actions) + 1) * 0.2)
    observation, _ = env.reset(seed=20260901)
    del observation
    started = time.perf_counter()
    rewards = []
    last_info = {}
    for action in actions:
        _, reward, terminated, truncated, last_info = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - started
    stats = env.simulator._cooperative_solver.stats
    diagnostics = env.simulator.get_solver_diagnostics()
    state = env.simulator.state
    return {
        "method": method,
        "presolve": presolve,
        "steps": len(rewards),
        "elapsed_seconds": elapsed,
        "steps_per_second": len(rewards) / elapsed,
        "reward_sum": float(sum(rewards)),
        "biomass_g_l": {
            name: float(species.biomass) for name, species in state.species.items()
        },
        "rubber_g_l": float(state.rubber_concentration),
        "pha_g_l": float(
            sum(species.pha_accumulated for species in state.species.values())
        ),
        "solver_status": stats.status,
        "last_solve_timing_seconds": {
            "build": stats.build_seconds,
            "stage1": stats.stage1_seconds,
            "stage2": stats.stage2_seconds,
            "stage3": stats.stage3_seconds,
            "total": stats.solve_seconds,
        },
        "solver_success_rate": float(diagnostics["solve_success_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "cooperative_highs_options.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(20260901)
    actions = rng.uniform(0.05, 0.95, size=(args.steps, 5)).astype(np.float32)
    cases = []
    for method in ("highs", "highs-ds", "highs-ipm"):
        for presolve in (True, False):
            row = run_case(method, presolve, actions)
            cases.append(row)
            print(
                f"{method:8s} presolve={presolve!s:5s}: "
                f"{row['steps_per_second']:.3f} steps/s, "
                f"success={row['solver_success_rate']:.3f}",
                flush=True,
            )
    reference = cases[0]
    for row in cases:
        row["speedup_vs_default"] = (
            row["steps_per_second"] / reference["steps_per_second"]
        )
        row["max_biomass_abs_difference_vs_default"] = max(
            abs(row["biomass_g_l"][name] - reference["biomass_g_l"][name])
            for name in reference["biomass_g_l"]
        )
        row["pha_abs_difference_vs_default"] = abs(
            row["pha_g_l"] - reference["pha_g_l"]
        )
        row["rubber_abs_difference_vs_default"] = abs(
            row["rubber_g_l"] - reference["rubber_g_l"]
        )
    report = {
        "benchmark": "exact_cooperative_fba_highs_options",
        "steps_per_case": args.steps,
        "seed": 20260901,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
