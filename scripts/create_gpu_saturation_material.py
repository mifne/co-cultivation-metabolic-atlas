#!/usr/bin/env python3
"""Create an evidence-only RTX 4060 utilization and throughput figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequential",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpu_saturation_2048_raw_rtx4060.json",
    )
    parser.add_argument(
        "--multistream",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpu_saturation_2048_multistream_rtx4060.json",
    )
    parser.add_argument(
        "--end-to-end",
        type=Path,
        default=PROJECT_ROOT / "results" / "parallel_env_16_long_multistream_rtx4060.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpu_saturation_evidence_rtx4060.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpu_saturation_evidence_rtx4060.json",
    )
    args = parser.parse_args()

    sequential = load(args.sequential)
    multistream = load(args.multistream)
    end_to_end = load(args.end_to_end)
    before = {row["batch_size_per_species"]: row for row in sequential["rows"]}
    after = {row["batch_size_per_species"]: row for row in multistream["rows"]}
    batches = sorted(set(before) & set(after))
    cpu = end_to_end["cpu_highs"][0]
    gpu = end_to_end["cuda_surrogate_batched"][0]

    throughput_optimum = max(
        batches, key=lambda size: after[size]["predictions_per_second"]
    )
    utilization_optimum = max(
        batches,
        key=lambda size: after[size]["resource"]["gpu_utilization_percent"]["mean"],
    )
    summary = {
        "gpu_name": multistream["gpu_name"],
        "workload": multistream["measurement_type"],
        "candidate_count_per_large_gem": 2048,
        "throughput_optimum": {
            "batch_size_per_gem": throughput_optimum,
            "gpu_utilization_mean_percent": after[throughput_optimum]["resource"]["gpu_utilization_percent"]["mean"],
            "gpu_utilization_max_percent": after[throughput_optimum]["resource"]["gpu_utilization_percent"]["max"],
            "vram_max_mib": after[throughput_optimum]["resource"]["gpu_memory_used_mib"]["max"],
            "predictions_per_second": after[throughput_optimum]["predictions_per_second"],
        },
        "utilization_optimum": {
            "batch_size_per_gem": utilization_optimum,
            "gpu_utilization_mean_percent": after[utilization_optimum]["resource"]["gpu_utilization_percent"]["mean"],
            "gpu_utilization_max_percent": after[utilization_optimum]["resource"]["gpu_utilization_percent"]["max"],
            "vram_max_mib": after[utilization_optimum]["resource"]["gpu_memory_used_mib"]["max"],
            "predictions_per_second": after[utilization_optimum]["predictions_per_second"],
        },
        "end_to_end_16_env": {
            "transitions": gpu["transitions"],
            "cpu_highs_transitions_per_second": cpu["transitions_per_second"],
            "cuda_transitions_per_second": gpu["transitions_per_second"],
            "speedup": gpu["transitions_per_second"] / cpu["transitions_per_second"],
            "gpu_solution_acceptance_rate": gpu["solver"]["surrogate_acceptance_rate"],
            "cpu_fallback_solves": gpu["solver"]["cpu_fallback_solves"],
            "gpu_utilization_mean_percent": gpu["resource"]["gpu_utilization_percent"]["mean"],
            "gpu_utilization_max_percent": gpu["resource"]["gpu_utilization_percent"]["max"],
            "vram_max_mib": gpu["resource"]["gpu_memory_used_mib"]["max"],
        },
        "interpretation": "Batch 64 maximizes useful throughput; batch 128 maximizes utilization but is memory-pressure limited.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), constrained_layout=True)
    series = [
        ("Sequential GEMs", before, "o--"),
        ("Concurrent CUDA streams", after, "o-"),
    ]
    for label, rows, style in series:
        axes[0, 0].plot(
            batches,
            [rows[size]["resource"]["gpu_utilization_percent"]["mean"] for size in batches],
            style,
            label=label,
        )
        axes[0, 1].plot(
            batches,
            [rows[size]["predictions_per_second"] for size in batches],
            style,
            label=label,
        )
    axes[0, 0].axhline(100, color="gray", linewidth=1)
    axes[0, 0].set_title("A. Sustained GPU utilization")
    axes[0, 0].set_ylabel("GPU utilization (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("B. Useful guarded FBA throughput")
    axes[0, 1].set_ylabel("3-GEM predictions / second")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        batches,
        [after[size]["resource"]["gpu_memory_used_mib"]["max"] / 1024 for size in batches],
        "o-",
    )
    axes[1, 0].axhline(8188 / 1024, color="gray", linewidth=1, linestyle=":", label="RTX 4060 total")
    axes[1, 0].set_title("C. VRAM used by production workload")
    axes[1, 0].set_ylabel("Peak VRAM (GiB)")
    axes[1, 0].legend(fontsize=8)

    labels = ["CPU / HiGHS", "CUDA surrogate"]
    values = [cpu["transitions_per_second"], gpu["transitions_per_second"]]
    bars = axes[1, 1].bar(labels, values, color=["#777777", "#2c7fb8"])
    for bar, value in zip(bars, values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value:.1f}",
            ha="center",
        )
    axes[1, 1].set_ylim(0, max(values) * 1.18)
    axes[1, 1].set_title("D. End-to-end, 16 environments / 960 transitions")
    axes[1, 1].set_ylabel("Transitions / second")
    axes[1, 1].text(
        0.98,
        0.90,
        f"{values[1] / values[0]:.2f}x",
        transform=axes[1, 1].transAxes,
        ha="right",
        fontsize=14,
    )

    for axis in axes.flat[:3]:
        axis.set_xlabel("Concurrent environments / GEM batch")
        axis.set_xscale("log", base=2)
        axis.set_xticks(batches, labels=[str(size) for size in batches])
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "RTX 4060 measured evidence: utilization ceiling vs throughput optimum"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"saved: {args.summary}\nsaved: {args.output}")


if __name__ == "__main__":
    main()
