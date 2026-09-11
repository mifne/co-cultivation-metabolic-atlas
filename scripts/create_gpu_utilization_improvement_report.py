#!/usr/bin/env python3
"""Plot before/after GPU batching and dictionary-size measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_map(report: dict) -> dict[int, dict]:
    return {row["n_envs"]: row for row in report["cuda_surrogate_batched"]}


def cpu_map(report: dict) -> dict[int, dict]:
    return {row["n_envs"]: row for row in report["cpu_highs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=PROJECT_ROOT / "results" / "parallel_env_scaling_rtx4060.json")
    parser.add_argument("--batched", type=Path, default=PROJECT_ROOT / "results" / "parallel_env_scaling_multispecies_rtx4060.json")
    parser.add_argument("--expanded", type=Path, default=PROJECT_ROOT / "results" / "parallel_env_scaling_512_rtx4060.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "gpu_utilization_improvement_rtx4060.png")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "gpu_utilization_improvement_rtx4060.json")
    args = parser.parse_args()
    initial = load(args.initial)
    batched = load(args.batched)
    expanded = load(args.expanded)
    envs = sorted(gpu_map(expanded))
    initial_gpu = gpu_map(initial)
    batched_gpu = gpu_map(batched)
    expanded_gpu = gpu_map(expanded)
    cpu = cpu_map(expanded)

    summary = {
        "gpu_name": expanded["gpu_name"],
        "windows_adapter": "GPU 1 (NVIDIA); WSL/PyTorch cuda:0",
        "measurement": expanded["measurement_type"],
        "environment_counts": envs,
        "rows": [],
    }
    for n in envs:
        row = {
            "n_envs": n,
            "cpu_transitions_per_second": cpu[n]["transitions_per_second"],
            "initial_gpu_transitions_per_second": initial_gpu[n]["transitions_per_second"],
            "batched_gpu_transitions_per_second": batched_gpu[n]["transitions_per_second"],
            "expanded_gpu_transitions_per_second": expanded_gpu[n]["transitions_per_second"],
            "expanded_speedup_vs_cpu": expanded_gpu[n]["transitions_per_second"] / cpu[n]["transitions_per_second"],
            "initial_gpu_utilization_mean_percent": initial_gpu[n]["resource"]["gpu_utilization_percent"]["mean"],
            "expanded_gpu_utilization_mean_percent": expanded_gpu[n]["resource"]["gpu_utilization_percent"]["mean"],
            "expanded_gpu_utilization_p95_percent": expanded_gpu[n]["resource"]["gpu_utilization_percent"]["p95"],
            "expanded_gpu_utilization_max_percent": expanded_gpu[n]["resource"]["gpu_utilization_percent"]["max"],
            "expanded_peak_vram_mib": expanded_gpu[n]["resource"]["gpu_memory_used_mib"]["max"],
            "expanded_acceptance_rate": expanded_gpu[n]["solver"]["surrogate_acceptance_rate"],
            "expanded_mean_batch_size": expanded_gpu[n]["service"]["mean_batch_size"],
        }
        summary["rows"].append(row)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), constrained_layout=True)
    axes[0, 0].plot(envs, [cpu[n]["transitions_per_second"] for n in envs], "o-", label="CPU / HiGHS")
    axes[0, 0].plot(envs, [initial_gpu[n]["transitions_per_second"] for n in envs], "o-", label="Initial CUDA")
    axes[0, 0].plot(envs, [batched_gpu[n]["transitions_per_second"] for n in envs], "o-", label="3-GEM batched / 102 candidates")
    axes[0, 0].plot(envs, [expanded_gpu[n]["transitions_per_second"] for n in envs], "o-", label="3-GEM batched / 512 candidates")
    axes[0, 0].set_title("Equal-work rollout throughput")
    axes[0, 0].set_ylabel("Transitions / second")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(envs, [initial_gpu[n]["resource"]["gpu_utilization_percent"]["mean"] for n in envs], "o-", label="Initial mean")
    axes[0, 1].plot(envs, [expanded_gpu[n]["resource"]["gpu_utilization_percent"]["mean"] for n in envs], "o-", label="Optimized mean")
    axes[0, 1].plot(envs, [expanded_gpu[n]["resource"]["gpu_utilization_percent"]["p95"] for n in envs], "s--", label="Optimized P95")
    axes[0, 1].set_title("NVIDIA GPU utilization")
    axes[0, 1].set_ylabel("Percent (%)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(envs, [initial_gpu[n]["service"]["mean_batch_size"] for n in envs], "o-", label="Initial")
    axes[1, 0].plot(envs, [expanded_gpu[n]["service"]["mean_batch_size"] for n in envs], "o-", label="Optimized")
    axes[1, 0].plot(envs, envs, ":", color="gray", label="Ideal synchronized batch")
    axes[1, 0].set_title("Observed micro-batch size")
    axes[1, 0].set_ylabel("Requests / batch")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(envs, [100 * initial_gpu[n]["solver"]["surrogate_acceptance_rate"] for n in envs], "o-", label="Initial")
    axes[1, 1].plot(envs, [100 * expanded_gpu[n]["solver"]["surrogate_acceptance_rate"] for n in envs], "o-", label="512 candidates")
    axes[1, 1].set_title("GPU solution acceptance")
    axes[1, 1].set_ylabel("Accepted without CPU fallback (%)")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Parallel environments")
        axis.set_xticks(envs)
        axis.grid(alpha=0.25)
    fig.suptitle("RTX 4060: GPU utilization and end-to-end improvement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"saved: {args.summary}\nsaved: {args.output}")


if __name__ == "__main__":
    main()
