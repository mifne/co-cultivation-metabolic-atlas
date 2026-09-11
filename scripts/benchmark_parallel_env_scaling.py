#!/usr/bin/env python3
"""Fair end-to-end scaling comparison for CPU and batched CUDA environments.

For each environment count both backends receive the same action tensor and
execute the same number of vector-environment steps.  Model/CUDA warm-up is
excluded.  Resource samples and surrogate micro-batch diagnostics are stored
with the timing result so high throughput cannot be mistaken for GPU use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import make_env
from scripts.benchmark_resource_usage import ResourceSampler, summarize_resource
from src.fba_surrogate_service import start_surrogate_service
from src.utils import load_sbml_models, select_consortium_models


def aggregate_solver_diagnostics(rows: list[dict]) -> dict:
    totals = {
        "solve_attempts": 0,
        "solve_successes": 0,
        "cpu_fallback_solves": 0,
        "surrogate_attempts": 0,
        "surrogate_accepts": 0,
        "surrogate_rejections": 0,
        "surrogate_rejection_reasons": {},
    }
    for row in rows:
        totals["solve_attempts"] += int(row.get("solve_attempts", 0))
        totals["solve_successes"] += int(row.get("solve_successes", 0))
        totals["cpu_fallback_solves"] += int(row.get("cpu_fallback_solves", 0))
        surrogate = row.get("surrogate", {})
        totals["surrogate_attempts"] += int(surrogate.get("attempts", 0))
        totals["surrogate_accepts"] += int(surrogate.get("accepted", 0))
        totals["surrogate_rejections"] += int(surrogate.get("rejections", 0))
        for reason, count in surrogate.get("rejection_reasons", {}).items():
            totals["surrogate_rejection_reasons"][reason] = (
                totals["surrogate_rejection_reasons"].get(reason, 0) + int(count)
            )
    totals["surrogate_acceptance_rate"] = (
        totals["surrogate_accepts"] / totals["surrogate_attempts"]
        if totals["surrogate_attempts"]
        else None
    )
    return totals


def run_case(
    backend: str,
    n_envs: int,
    actions: np.ndarray,
    artifact_dir: Path,
    interval_seconds: float,
    batch_window_ms: float,
    max_batch_size: int,
    worker_start_method: str,
) -> dict:
    manager = None
    service = None
    env_params = {
        "max_time": (len(actions) + 4) * 0.2,
        "solver_backend": backend,
        "fba_mode": "separate",
        "surrogate_dir": str(artifact_dir) if backend == "surrogate" else None,
        "surrogate_device": "cuda",
        "surrogate_audit_interval": 0,
        "surrogate_ood_threshold": 8.0,
    }
    models = select_consortium_models(
        load_sbml_models(PROJECT_ROOT / "models" / "sbml")
    )
    if backend == "surrogate":
        manager, service = start_surrogate_service(
            models,
            str(artifact_dir),
            device="cuda",
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
            ood_threshold=8.0,
        )
        env_params["surrogate_service"] = service

    env_fns = [
        make_env(
            str(PROJECT_ROOT / "models" / "sbml"),
            env_params,
            data_log_path=None,
            rank=rank,
            preloaded_models=models,
        )
        for rank in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method=worker_start_method)
    try:
        vec_env.seed(20260827)
        vec_env.reset()
        # One complete vector step warms model objects, CUDA and RPC paths.
        vec_env.step(actions[0])
        observations = vec_env.reset()
        sampler = ResourceSampler(interval_seconds, backend)
        sampler.start()
        started = time.perf_counter()
        cumulative_rewards = np.zeros(n_envs, dtype=np.float64)
        last_infos = None
        for action_batch in actions:
            observations, rewards, _, last_infos = vec_env.step(action_batch)
            cumulative_rewards += rewards
        elapsed = time.perf_counter() - started
        sampler.stop()
        worker_diagnostics = vec_env.env_method("get_solver_diagnostics")
        service_diagnostics = service.diagnostics() if service is not None else None
        transitions = int(len(actions) * n_envs)
        return {
            "backend": "cpu_highs" if backend == "highs" else "cuda_surrogate_batched",
            "n_envs": n_envs,
            "vector_steps": len(actions),
            "transitions": transitions,
            "elapsed_seconds": elapsed,
            "transitions_per_second": transitions / elapsed,
            "resource": summarize_resource(sampler.rows),
            "resource_samples": sampler.rows,
            "solver": aggregate_solver_diagnostics(worker_diagnostics),
            "service": service_diagnostics,
            "final_observations": np.asarray(observations).tolist(),
            "cumulative_rewards": cumulative_rewards.tolist(),
            "last_infos": last_infos,
        }
    finally:
        vec_env.close()
        if service is not None:
            service.close()
        if manager is not None:
            manager.shutdown()


def accuracy_summary(cpu: dict, gpu: dict) -> dict:
    cpu_obs = np.asarray(cpu["final_observations"], dtype=np.float64)
    gpu_obs = np.asarray(gpu["final_observations"], dtype=np.float64)
    cpu_rewards = np.asarray(cpu["cumulative_rewards"], dtype=np.float64)
    gpu_rewards = np.asarray(gpu["cumulative_rewards"], dtype=np.float64)
    return {
        "final_observation_max_abs_difference": float(np.max(np.abs(cpu_obs - gpu_obs))),
        "final_observation_mean_abs_difference": float(np.mean(np.abs(cpu_obs - gpu_obs))),
        "cumulative_reward_max_abs_difference": float(np.max(np.abs(cpu_rewards - gpu_rewards))),
        "cumulative_reward_mean_abs_difference": float(np.mean(np.abs(cpu_rewards - gpu_rewards))),
    }


def make_plot(report: dict, output: Path) -> None:
    cpu = report["cpu_highs"]
    gpu = report["cuda_surrogate_batched"]
    envs = [row["n_envs"] for row in cpu]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(envs, [r["transitions_per_second"] for r in cpu], "o-", label="CPU / HiGHS")
    axes[0, 0].plot(envs, [r["transitions_per_second"] for r in gpu], "o-", label="Batched CUDA surrogate")
    axes[0, 0].set_title("End-to-end rollout throughput")
    axes[0, 0].set_ylabel("Transitions / second")
    axes[0, 0].legend()
    axes[0, 1].plot(envs, [r["resource"]["gpu_utilization_percent"]["mean"] for r in gpu], "o-", label="Mean")
    axes[0, 1].plot(envs, [r["resource"]["gpu_utilization_percent"]["p95"] for r in gpu], "s--", label="P95")
    axes[0, 1].set_title("GPU utilization during CUDA rollout")
    axes[0, 1].set_ylabel("Percent (%)")
    axes[0, 1].legend()
    axes[1, 0].plot(envs, [r["resource"]["gpu_memory_used_mib"]["max"] for r in gpu], "o-")
    axes[1, 0].set_title("Peak VRAM")
    axes[1, 0].set_ylabel("MiB")
    axes[1, 1].plot(envs, [r["service"]["mean_batch_size"] for r in gpu], "o-", label="Mean batch")
    axes[1, 1].plot(envs, [r["service"]["max_observed_batch"] for r in gpu], "s--", label="Max batch")
    axes[1, 1].set_title("Observed GPU micro-batch size")
    axes[1, 1].set_ylabel("Requests / batch")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Parallel environments")
        axis.set_xticks(envs)
        axis.grid(alpha=0.25)
    fig.suptitle("Equal-work CPU vs batched CUDA environment scaling")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-counts", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--vector-steps", type=int, default=16)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--batch-window-ms", type=float, default=4.0)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument(
        "--worker-start-method",
        choices=["fork", "forkserver", "spawn"],
        default="fork",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "models" / "fba_surrogate_gpu_lp")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "parallel_env_scaling_rtx4060.json")
    parser.add_argument("--plot", type=Path, default=PROJECT_ROOT / "results" / "parallel_env_scaling_rtx4060.png")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    max_envs = max(args.env_counts)
    rng = np.random.default_rng(args.seed)
    all_actions = rng.uniform(
        0.05, 0.95, size=(args.vector_steps, max_envs, 5)
    ).astype(np.float32)
    report = {
        "measurement_type": "equal-work steady-state vector-environment rollout",
        "gpu_name": torch.cuda.get_device_name(0),
        "vector_steps": args.vector_steps,
        "sample_interval_seconds": args.interval,
        "batch_window_ms": args.batch_window_ms,
        "max_batch_size": args.max_batch_size,
        "cpu_highs": [],
        "cuda_surrogate_batched": [],
        "accuracy": {},
    }
    for n_envs in args.env_counts:
        actions = all_actions[:, :n_envs]
        cpu = run_case(
            "highs", n_envs, actions, args.artifact_dir, args.interval,
            args.batch_window_ms, args.max_batch_size, args.worker_start_method,
        )
        gpu = run_case(
            "surrogate", n_envs, actions, args.artifact_dir, args.interval,
            args.batch_window_ms, args.max_batch_size, args.worker_start_method,
        )
        report["cpu_highs"].append(cpu)
        report["cuda_surrogate_batched"].append(gpu)
        report["accuracy"][str(n_envs)] = accuracy_summary(cpu, gpu)
        print(
            f"n_envs={n_envs}: CPU={cpu['transitions_per_second']:.2f}, "
            f"GPU={gpu['transitions_per_second']:.2f} transitions/s, "
            f"GPU util={gpu['resource']['gpu_utilization_percent']['mean']:.1f}%"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(report, args.plot)
    print(f"saved: {args.output}\nsaved: {args.plot}")


if __name__ == "__main__":
    main()
