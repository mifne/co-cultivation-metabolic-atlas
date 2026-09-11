#!/usr/bin/env python3
"""Multi-seed scientific qualification of a cooperative GPU surrogate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_cooperative_surrogate_e2e import CONSORTIUM_PROFILES, run_case


def _exact_case(task: tuple[int, int, int, str, float | None]):
    seed, steps, top_k, consortium_profile, initial_nh4 = task
    rng = np.random.default_rng(seed)
    actions = rng.uniform(0.05, 0.95, size=(steps, 5)).astype(np.float32)
    return seed, run_case(
        None, actions, top_k=top_k,
        consortium_profile=consortium_profile,
        initial_nh4=initial_nh4,
    )


def _confidence_interval_95(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return mean, mean
    from scipy.stats import t

    half = float(t.ppf(0.975, len(data) - 1) * data.std(ddof=1) / np.sqrt(len(data)))
    return mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260901, 20260902, 20260903, 20260904, 20260905])
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--initial-nh4", type=float, default=None)
    parser.add_argument(
        "--consortium", choices=CONSORTIUM_PROFILES, default="legacy3"
    )
    parser.add_argument("--gpu-qp-projection", action="store_true")
    parser.add_argument("--gpu-qp-only", action="store_true")
    parser.add_argument("--gpu-qp-match-pha", action="store_true")
    parser.add_argument("--gpu-qp-max-iterations", type=int, default=400)
    parser.add_argument("--gpu-qp-multioutput-strength", type=float, default=0.0)
    parser.add_argument("--gpu-qp-independent-species", action="store_true")
    parser.add_argument("--gpu-qp-candidates", type=int, default=128)
    parser.add_argument("--gpu-qp-rerank-pool", type=int, default=None)
    parser.add_argument("--gpu-qp-decision-strength", type=float, default=None)
    parser.add_argument("--gpu-qp-blend-candidates", type=int, default=1)
    parser.add_argument("--gpu-qp-blend-distance-power", type=float, default=2.0)
    parser.add_argument("--max-pha-relative-error", type=float, default=0.01)
    parser.add_argument("--min-acceptance", type=float, default=0.90)
    parser.add_argument("--max-biomass-absolute-error", type=float, default=0.01)
    parser.add_argument("--max-phv-mol-fraction-error", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exact-workers",
        type=int,
        default=1,
        help="parallel CPU workers used only to build exact reference trajectories",
    )
    args = parser.parse_args()

    exact_workers = max(1, min(int(args.exact_workers), len(args.seeds)))
    tasks = [
        (int(seed), int(args.steps), int(args.top_k), args.consortium, args.initial_nh4)
        for seed in args.seeds
    ]
    if exact_workers == 1:
        exact_rows = [_exact_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=exact_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            exact_rows = list(executor.map(_exact_case, tasks))
    exact_by_seed = dict(exact_rows)

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        actions = rng.uniform(0.05, 0.95, size=(args.steps, 5)).astype(np.float32)
        exact = exact_by_seed[int(seed)]
        surrogate = run_case(
            args.artifact,
            actions,
            top_k=args.top_k,
            exact_interval=0,
            gpu_qp_projection=args.gpu_qp_projection,
            gpu_qp_only=args.gpu_qp_only,
            gpu_qp_match_pha=args.gpu_qp_match_pha,
            gpu_qp_candidates=args.gpu_qp_candidates,
            gpu_qp_rerank_pool=args.gpu_qp_rerank_pool,
            gpu_qp_decision_strength=args.gpu_qp_decision_strength,
            gpu_qp_blend_candidates=args.gpu_qp_blend_candidates,
            gpu_qp_blend_distance_power=args.gpu_qp_blend_distance_power,
            consortium_profile=args.consortium,
            initial_nh4=args.initial_nh4,
            gpu_qp_max_iterations=args.gpu_qp_max_iterations,
            gpu_qp_multioutput_strength=args.gpu_qp_multioutput_strength,
            gpu_qp_independent_species=args.gpu_qp_independent_species,
        )
        pha_denominator = max(abs(float(exact["pha_g_l"])), 1e-9)
        cooperative = surrogate["solver"]["cooperative"]
        row = {
            "implementation_sha256": surrogate.get("implementation_sha256"),
            "seed": int(seed),
            "exact_completed_steps": int(exact["steps"]),
            "gpu_completed_steps": int(surrogate["steps"]),
            "speedup": float(exact["elapsed_seconds"] / surrogate["elapsed_seconds"]),
            "exact_seconds": float(exact["elapsed_seconds"]),
            "surrogate_seconds": float(surrogate["elapsed_seconds"]),
            "exact_pha_g_l": float(exact["pha_g_l"]),
            "gpu_pha_g_l": float(surrogate["pha_g_l"]),
            "exact_phv_mol_fraction": exact["phv_mol_fraction"],
            "gpu_phv_mol_fraction": surrogate["phv_mol_fraction"],
            "phv_mol_fraction_absolute_error": (
                abs(exact["phv_mol_fraction"] - surrogate["phv_mol_fraction"])
                if exact["phv_mol_fraction"] is not None
                and surrogate["phv_mol_fraction"] is not None else None
            ),
            "exact_biomass_g_l": exact["biomass_g_l"],
            "gpu_biomass_g_l": surrogate["biomass_g_l"],
            "exact_rubber_remaining_g_l": float(exact["rubber_remaining_g_l"]),
            "gpu_rubber_remaining_g_l": float(surrogate["rubber_remaining_g_l"]),
            "pha_relative_error": float(abs(exact["pha_g_l"] - surrogate["pha_g_l"]) / pha_denominator),
            "pha_absolute_error_g_l": float(abs(exact["pha_g_l"] - surrogate["pha_g_l"])),
            "max_biomass_absolute_error_g_l": float(
                max(abs(exact["biomass_g_l"][name] - surrogate["biomass_g_l"][name]) for name in exact["biomass_g_l"])
            ),
            "rubber_absolute_error_g_l": float(abs(exact["rubber_remaining_g_l"] - surrogate["rubber_remaining_g_l"])),
            "reward_absolute_error": float(abs(exact["reward_sum"] - surrogate["reward_sum"])),
            "acceptance_rate": float(cooperative["surrogate_acceptance_rate"]),
            "fallback_calls": int(cooperative["cpu_cooperative_solve_calls"]),
            "online_cpu_lp_calls": int(cooperative["cpu_lp_stage_calls"]),
            "gpu_qp_accepts": int(cooperative.get("gpu_qp_accepts", 0)),
            "gpu_qp_failures": int(cooperative.get("gpu_qp_failures", 0)),
            "fallback_reasons": cooperative["surrogate_rejection_reasons"],
            "gpu_peak_memory_allocated_mib": float(surrogate.get("gpu_peak_memory_allocated_mib", 0.0)),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    speedups = [row["speedup"] for row in rows]
    pha_errors = [row["pha_relative_error"] for row in rows]
    acceptances = [row["acceptance_rate"] for row in rows]
    biomass_errors = [row["max_biomass_absolute_error_g_l"] for row in rows]
    phv_errors = [
        row["phv_mol_fraction_absolute_error"] for row in rows
        if row["phv_mol_fraction_absolute_error"] is not None
    ]
    qualified = (
        max(pha_errors) <= args.max_pha_relative_error
        and min(acceptances) >= args.min_acceptance
        and max(biomass_errors) <= args.max_biomass_absolute_error
        and (not phv_errors or max(phv_errors) <= args.max_phv_mol_fraction_error)
        and min(speedups) > 1.0
        and all(row["exact_completed_steps"] == args.steps
                and row["gpu_completed_steps"] == args.steps for row in rows)
        and all(row["exact_pha_g_l"] > 1e-9 and row["gpu_pha_g_l"] > 1e-9 for row in rows)
        and len(phv_errors) == len(rows)
        and (not args.gpu_qp_only or all(
            row["online_cpu_lp_calls"] == 0 and row["gpu_qp_failures"] == 0
            and row["gpu_qp_accepts"] == args.steps for row in rows))
    )
    report = {
        "status": "qualified" if qualified else "experimental_not_qualified",
        "artifact": str(args.artifact),
        "consortium_profile": args.consortium,
        "validation": "equal_action_cooperative_dfba_multi_seed",
        "simulated_hours": float(args.steps * 0.2),
        "initial_nh4_mmol_l": args.initial_nh4,
        "steps": int(args.steps),
        "dt_hours": 0.2,
        "gpu_qp_projection": bool(args.gpu_qp_projection),
        "gpu_qp_only": bool(args.gpu_qp_only),
        "gpu_qp_match_pha": bool(args.gpu_qp_match_pha),
        "gpu_qp_max_iterations": args.gpu_qp_max_iterations,
        "gpu_qp_multioutput_strength": args.gpu_qp_multioutput_strength,
        "gpu_qp_independent_species":args.gpu_qp_independent_species,
        "gpu_qp_candidates": int(args.gpu_qp_candidates),
        "gpu_qp_rerank_pool": args.gpu_qp_rerank_pool,
        "gpu_qp_decision_strength": args.gpu_qp_decision_strength,
        "gpu_qp_blend_candidates": int(args.gpu_qp_blend_candidates),
        "gpu_qp_blend_distance_power": float(
            args.gpu_qp_blend_distance_power
        ),
        "n": len(rows),
        "exact_reference_workers": exact_workers,
        "timing_caveat": (
            "CPU reference trajectories ran concurrently; per-run speedups are diagnostic, not matched-load benchmarks"
            if exact_workers > 1 else None
        ),
        "seeds": [int(seed) for seed in args.seeds],
        "criteria": {
            "gpu_only_requires_every_step_accepted": bool(args.gpu_qp_only),
            "cpu_lp_counter_unit": "individual SciPy linprog stage invocation",
            "maximum_pha_relative_error": float(args.max_pha_relative_error),
            "minimum_gpu_acceptance_rate": float(args.min_acceptance),
            "maximum_biomass_absolute_error_g_l": float(args.max_biomass_absolute_error),
            "maximum_phv_mol_fraction_absolute_error": float(args.max_phv_mol_fraction_error),
            "minimum_speedup": 1.0,
        },
        "summary": {
            "speedup_mean": float(np.mean(speedups)),
            "speedup_95_ci": list(_confidence_interval_95(speedups)),
            "pha_relative_error_mean": float(np.mean(pha_errors)),
            "pha_relative_error_max": float(np.max(pha_errors)),
            "acceptance_rate_mean": float(np.mean(acceptances)),
            "acceptance_rate_min": float(np.min(acceptances)),
            "max_biomass_absolute_error_g_l": float(np.max(biomass_errors)),
            "phv_mol_fraction_error_max": float(max(phv_errors)) if phv_errors else None,
        },
        "runs": rows,
    }
    output = args.output or args.artifact.with_suffix(".validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"status: {report['status']}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
