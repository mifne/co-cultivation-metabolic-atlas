#!/usr/bin/env python3
"""Exercise PPO-like parallel actions through the GPU-only QP service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_until_effect_wcfs1 import LOW_AIR, MODEL_DIR, make_train_env
from src.cooperative_gpu_service import start_cooperative_gpu_qp_service
from src.utils import load_sbml_models, select_consortium_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--retry-candidates", type=int, default=2048)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT
        / "models/cooperative_surrogate/cooperative_neural_reranker_34448_dagger2_rebased.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/gpu_qp_parallel_training_smoke.json",
    )
    args = parser.parse_args()

    models = select_consortium_models(load_sbml_models(MODEL_DIR))
    original = {
        name: {reaction.id: tuple(reaction.bounds) for reaction in model.exchanges}
        for name, model in models.items()
    }
    manager, service = start_cooperative_gpu_qp_service(
        models,
        original,
        str(args.artifact),
        candidates=128,
        retry_candidates=args.retry_candidates,
        batch_window_ms=2.0,
        max_batch_size=max(4, args.n_envs),
    )
    env = None
    started = time.perf_counter()
    try:
        env = SubprocVecEnv(
            [
                make_train_env(
                    rank,
                    args.seed,
                    str(args.artifact),
                    0,
                    float("inf"),
                    service,
                )
                for rank in range(args.n_envs)
            ],
            start_method="fork",
        )
        env.reset()
        rng = np.random.default_rng(args.seed)
        mean = 2.0 * LOW_AIR - 1.0
        for _ in range(args.steps):
            actions = np.clip(
                rng.normal(mean, np.exp(-1.2), size=(args.n_envs, 5)),
                -1.0,
                1.0,
            ).astype(np.float32)
            env.step(actions)
        worker_diagnostics = env.env_method("get_solver_diagnostics")
        service_diagnostics = service.diagnostics()
    finally:
        if env is not None:
            env.close()
        service.close()
        manager.shutdown()

    report = {
        "n_envs": args.n_envs,
        "steps_per_env": args.steps,
        "transitions": args.n_envs * args.steps,
        "elapsed_seconds": time.perf_counter() - started,
        "service": service_diagnostics,
        "workers": worker_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["service"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
