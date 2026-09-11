#!/usr/bin/env python3
"""Benchmark exact CPU FBA against batched CPU/CUDA surrogate inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dfba_simulator import dFBASimulator
from src.fba_surrogate import FbaSurrogateBackend, safe_species_filename
from src.utils import load_sbml_models, select_consortium_models


def benchmark_network(surrogate, features, batch_sizes, repeats):
    results = []
    torch_device = surrogate.device
    for batch_size in batch_sizes:
        rows = np.resize(features, (batch_size, features.shape[1])).astype(np.float32)
        tensor = torch.as_tensor(rows, device=torch_device)
        with torch.inference_mode():
            surrogate.network(tensor)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                surrogate.network(tensor)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        elapsed = time.perf_counter() - started
        guarded_repeats = min(repeats, 20)
        guarded_started = time.perf_counter()
        for _ in range(guarded_repeats):
            surrogate.predict_features(rows)
        guarded_elapsed = time.perf_counter() - guarded_started
        results.append(
            {
                "batch_size": batch_size,
                "seconds_per_batch": elapsed / repeats,
                "lp_predictions_per_second": batch_size * repeats / elapsed,
                "guarded_predictions_per_second": (
                    batch_size * guarded_repeats / guarded_elapsed
                ),
            }
        )
    return results


def exact_cpu_rate(model, features, limit):
    n = len(model.reactions)
    solved = 0
    started = time.perf_counter()
    for row in features[:limit]:
        for index, reaction in enumerate(model.reactions):
            reaction.bounds = (float(row[index]), float(row[n + index]))
        coefficients = {
            reaction: float(row[2 * n + index])
            for index, reaction in enumerate(model.reactions)
            if row[2 * n + index] != 0.0
        }
        model.objective = coefficients
        model.objective_direction = "max" if row[-1] > 0 else "min"
        solved += dFBASimulator._solve_exact_cpu_lp(model) is not None
    elapsed = time.perf_counter() - started
    return {
        "solves": solved,
        "seconds": elapsed,
        "lp_solves_per_second": solved / elapsed if elapsed else 0.0,
    }


def accuracy(surrogate, features, exact_fluxes):
    tensor = torch.as_tensor(features, device=surrogate.device)
    with torch.inference_mode():
        predicted = surrogate.network(tensor).cpu().numpy().astype(np.float64)
    exact = exact_fluxes.astype(np.float64)
    n = exact.shape[1]
    objective = features[:, 2 * n : 3 * n]
    objective_error = np.abs(
        np.sum(predicted * objective, axis=1) - np.sum(exact * objective, axis=1)
    )
    guarded = surrogate.predict_features(features)
    return {
        "flux_rmse": float(np.sqrt(np.mean((predicted - exact) ** 2))),
        "objective_mae": float(np.mean(objective_error)),
        "objective_max_abs_error": float(np.max(objective_error)),
        "guard_acceptance_rate": float(np.mean([item.accepted for item in guarded])),
        "max_mass_balance_residual": float(
            max(item.max_mass_balance_residual for item in guarded)
        ),
        "max_relative_mass_balance_residual": float(
            max(item.max_relative_mass_balance_residual for item in guarded)
        ),
        "max_bound_violation": float(max(item.max_bound_violation for item in guarded)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbml-dir", type=Path, default=PROJECT_ROOT / "models" / "sbml")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "models" / "fba_surrogate")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--exact-limit", type=int, default=16)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "fba_surrogate_benchmark.json")
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = args.artifact_dir / "exact_training_data.npz"
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = select_consortium_models(load_sbml_models(args.sbml_dir))
    backend = FbaSurrogateBackend(models, args.artifact_dir, device=device)
    archive = np.load(args.dataset)
    report = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "species": {},
    }
    for species, model in models.items():
        prefix = safe_species_filename(species)
        features = archive[f"{prefix}__features"]
        fluxes = archive[f"{prefix}__fluxes"]
        surrogate = backend.species[species]
        report["species"][species] = {
            "exact_cpu": exact_cpu_rate(model, features, args.exact_limit),
            "surrogate": benchmark_network(
                surrogate, features, args.batch_sizes, args.repeats
            ),
            "accuracy": accuracy(surrogate, features, fluxes),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure_path = args.output.with_suffix(".png")
    fig, axes = plt.subplots(1, len(report["species"]), figsize=(15, 4), sharey=True)
    if len(report["species"]) == 1:
        axes = [axes]
    for axis, (species, values) in zip(axes, report["species"].items()):
        batches = [row["batch_size"] for row in values["surrogate"]]
        throughput = [
            row["guarded_predictions_per_second"] for row in values["surrogate"]
        ]
        cpu = values["exact_cpu"]["lp_solves_per_second"]
        axis.plot(
            batches, throughput, marker="o", label=f"{device} surrogate + guards"
        )
        axis.axhline(cpu, color="tab:red", linestyle="--", label="CPU HiGHS")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(species.replace("_", " "))
        axis.set_xlabel("Concurrent environments / batch")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("FBA-equivalent predictions / second")
    axes[-1].legend(loc="best")
    fig.suptitle("Exact CPU FBA vs physics-constrained batched surrogate")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    print(json.dumps(report, indent=2))
    print(f"saved: {args.output}\nsaved: {figure_path}")


if __name__ == "__main__":
    main()
