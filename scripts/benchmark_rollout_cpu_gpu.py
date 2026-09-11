#!/usr/bin/env python3
"""Repeat an identical dFBA rollout on CPU HiGHS and the CUDA surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.utils import get_initial_params, load_sbml_models, select_consortium_models


def make_environment(sbml_dir: Path, backend: str, artifact_dir: Path) -> ConsortiumEnv:
    models = select_consortium_models(load_sbml_models(sbml_dir))
    biomass, metabolites = get_initial_params(models)
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2,
        solver_backend=backend,
        surrogate_dir=artifact_dir if backend == "surrogate" else None,
        surrogate_device="cuda",
        surrogate_audit_interval=128,
    )
    return ConsortiumEnv(simulator=simulator, max_time=24 * 0.2)


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_once(
    sbml_dir: Path,
    artifact_dir: Path,
    backend: str,
    actions: np.ndarray,
) -> dict:
    env = make_environment(sbml_dir, backend, artifact_dir)
    env.reset(seed=0)

    # Exclude model/artifact loading and first-use CUDA setup from steady-state timing.
    env.step(actions[0])
    env.reset(seed=0)
    synchronize_cuda()
    started = time.perf_counter()
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    synchronize_cuda()
    elapsed = time.perf_counter() - started
    diagnostics = env.simulator.get_solver_diagnostics()
    return {
        "seconds": elapsed,
        "steps": int(len(actions)),
        "steps_per_second": float(len(actions) / elapsed),
        "diagnostics": diagnostics,
    }


def summarize(rows: list[dict]) -> dict:
    values = [row["seconds"] for row in rows]
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * stdev / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean_seconds": mean,
        "stdev_seconds": stdev,
        "ci95_seconds": float(ci95),
        "mean_steps_per_second": statistics.fmean(
            row["steps_per_second"] for row in rows
        ),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbml-dir", type=Path, default=PROJECT_ROOT / "models" / "sbml")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "fba_surrogate_gpu_lp",
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "rollout_cpu_vs_gpu_rtx4060.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU comparison")
    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(0.05, 0.95, size=(args.steps, 5)).astype(np.float32)
    raw = {"highs": [], "surrogate_cuda": []}
    # Alternate backends to reduce bias from background system load.
    for _ in range(args.repeats):
        for backend, key in (("highs", "highs"), ("surrogate", "surrogate_cuda")):
            raw[key].append(run_once(args.sbml_dir, args.artifact_dir, backend, actions))

    report = {
        "timestamp_date": "2026-08-25",
        "gpu_name": torch.cuda.get_device_name(0),
        "steps_per_run": args.steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "highs": summarize(raw["highs"]),
        "surrogate_cuda": summarize(raw["surrogate_cuda"]),
    }
    report["speedup_mean_time"] = (
        report["highs"]["mean_seconds"]
        / report["surrogate_cuda"]["mean_seconds"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
