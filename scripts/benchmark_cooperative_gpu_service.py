#!/usr/bin/env python3
"""Verify cross-process-style micro-batching of cooperative GPU QP requests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cooperative_gpu_service import start_cooperative_gpu_qp_service
from src.utils import load_sbml_models, select_consortium_models
from scripts.benchmark_cooperative_surrogate_e2e import load_consortium_profile, CONSORTIUM_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--batch-window-ms", type=float, default=5.0)
    parser.add_argument("--consortium", choices=CONSORTIUM_PROFILES, default="legacy3")
    parser.add_argument("--contexts", type=Path, default=None)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--retry-candidates", type=int, default=2048)
    parser.add_argument("--rerank-pool", type=int, default=4096)
    parser.add_argument("--decision-strength", type=float, default=0)
    parser.add_argument("--multioutput-strength", type=float, default=0)
    parser.add_argument("--independent-species", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/cooperative_gpu_service.json")
    args = parser.parse_args()
    models = load_consortium_profile(args.consortium)
    original = {
        name: {reaction.id: tuple(reaction.bounds) for reaction in model.exchanges}
        for name, model in models.items()
    }
    if args.contexts is None and args.consortium != "legacy3":
        raise ValueError("provide matching --contexts for non-legacy consortium")
    context_path = args.contexts or PROJECT_ROOT / "models/cooperative_surrogate/production_32768/contexts.npy"
    contexts = np.load(context_path, mmap_mode="r")
    if context_path.suffix == ".npz":
        contexts = contexts["contexts"]
    if len(contexts) < args.workers*args.rounds:
        raise ValueError("not enough distinct context rows for requested rounds")
    manager, service = start_cooperative_gpu_qp_service(
        models,
        original,
        str(args.artifact),
        candidates=args.candidates,
        retry_candidates=args.retry_candidates,
        rerank_pool=args.rerank_pool,
        decision_strength=args.decision_strength,
        multioutput_strength=args.multioutput_strength,
        independent_species=args.independent_species,
        qp_max_iterations=args.max_iterations,
        block_composition_candidates=4,
        batch_window_ms=args.batch_window_ms,
        max_batch_size=args.workers,
    )
    elapsed = []
    feasible = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for round_index in range(args.rounds):
                rows = [
                    np.asarray(contexts[round_index * args.workers + index], dtype=np.float32)
                    for index in range(args.workers)
                ]
                started = time.perf_counter()
                results = list(executor.map(service.predict, rows))
                elapsed.append(time.perf_counter() - started)
                feasible.extend(bool(result["feasible"]) for result in results)
        diagnostics = service.diagnostics()
    finally:
        service.close()
        manager.shutdown()
    report = {
        "status":"parallel_request_smoke_not_trajectory_qualification",
        "consortium":args.consortium,
        "artifact":str(args.artifact), "contexts":str(context_path),
        "candidates":args.candidates, "retry_candidates":args.retry_candidates,
        "note":"includes startup warmup; not a matched CPU/GPU speedup benchmark",
        "workers": args.workers,
        "rounds": args.rounds,
        "requests": args.workers * args.rounds,
        "mean_round_seconds": float(np.mean(elapsed)),
        "requests_per_second": float(args.workers / np.mean(elapsed)),
        "feasible_fraction": float(np.mean(feasible)),
        "service": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
