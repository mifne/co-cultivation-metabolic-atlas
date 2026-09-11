#!/usr/bin/env python3
"""Benchmark device-resident neural retrieval + QP projection by batch size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.community_solver import CooperativeCommunityFbaSolver
from src.utils import load_sbml_models, select_consortium_models


def _layout(metadata: dict) -> dict[str, tuple[int, int]]:
    result = {}
    cursor = 0
    for name, width in metadata["context_layout"]:
        result[str(name)] = (cursor, cursor + int(width))
        cursor += int(width)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "models" / "cooperative_surrogate" / "production_32768",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260931)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpu_qp_batch_scaling.json",
    )
    args = parser.parse_args()

    models = select_consortium_models(
        load_sbml_models(PROJECT_ROOT / "models" / "sbml" / "final_consortium")
    )
    original = {
        name: {reaction.id: tuple(reaction.bounds) for reaction in model.exchanges}
        for name, model in models.items()
    }
    solver = CooperativeCommunityFbaSolver(
        models,
        original_exchange_bounds=original,
        maximum_coexistence_growth=0.005,
        optimize_live_objectives=False,
        surrogate_artifact=str(args.artifact),
        surrogate_device="cuda",
        surrogate_require_qualified=False,
        surrogate_exact_interval=0,
        gpu_qp_projection=True,
        gpu_qp_only=True,
        gpu_qp_candidates=args.candidates,
    )
    if solver._gpu_qp_projector is None or solver._surrogate_dictionary is None:
        raise RuntimeError("GPU QP path was not constructed")

    contexts = np.load(args.dataset / "contexts.npy", mmap_mode="r")
    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    layout = _layout(metadata)
    biomass_slice = slice(*layout["biomass_g_l"])
    supply_slice = slice(*layout["shared_supply_mmol_l_h"])
    lower_slice = slice(*layout["flux_lower_bounds"])
    upper_slice = slice(*layout["flux_upper_bounds"])
    rng = np.random.default_rng(args.seed)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for batch_size in args.batch_sizes:
        indices = rng.choice(len(contexts), size=int(batch_size), replace=False)
        context = np.asarray(contexts[indices], dtype=np.float32)

        def execute():
            candidates = solver._surrogate_dictionary.rank_device(
                context, top_k=args.candidates
            )
            return solver._gpu_qp_projector.project(
                candidates.fluxes,
                context[:, lower_slice],
                context[:, upper_slice],
                context[:, biomass_slice],
                context[:, supply_slice],
            )

        execute()  # warm-up allocations and CUDA kernels
        elapsed = []
        feasible = []
        iterations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = execute()
            elapsed.append(time.perf_counter() - started)
            feasible.append(float(np.mean(result.feasible)))
            iterations.append(int(result.iterations))
        mean_seconds = float(np.mean(elapsed))
        rows.append(
            {
                "batch_size": int(batch_size),
                "mean_seconds": mean_seconds,
                "environments_per_second": float(batch_size / mean_seconds),
                "per_environment_milliseconds": float(1000.0 * mean_seconds / batch_size),
                "feasible_fraction": float(np.mean(feasible)),
                "maximum_iterations": int(max(iterations)),
                "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    report = {
        "artifact": str(args.artifact),
        "gpu": torch.cuda.get_device_name(0),
        "candidate_count_per_environment": int(args.candidates),
        "repeats": int(args.repeats),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

