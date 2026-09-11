#!/usr/bin/env python3
"""Create the publication Resource Usage figure for the purchase case."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Muted, colour-vision-deficiency-aware palette with distinct luminance.
BLUE = "#245A73"
GREEN = "#3E8E82"
ORANGE = "#C6613A"
GREY = "#777B80"
INK = "#20252B"
GRID = "#D7DCE1"


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def configure() -> None:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", colors=INK)


def style_twin(ax, color: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_color(INK)
    ax.tick_params(axis="y", direction="out", colors=color)


def save(fig) -> None:
    metadata = {
        "Title": "Hardware resource pressure in the three-GEM dFBA workflow",
        "Creator": "co-cultivation benchmark pipeline",
    }
    for ext in ("png", "pdf", "svg"):
        path = RESULTS / f"resource_usage_publication.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03, "metadata": metadata}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)


def main() -> None:
    configure()
    saturation = read("gpu_saturation_2048_multistream_rtx4060.json")
    parallel = read("hardware_resource_parallel_rtx4060.json")
    evidence = read("workstation_procurement_evidence.json")

    rows = saturation["rows"]
    batch = np.asarray([row["batch_size_per_species"] for row in rows])
    gpu_mean = np.asarray([row["resource"]["gpu_utilization_percent"]["mean"] for row in rows])
    gpu_max = np.asarray([row["resource"]["gpu_utilization_percent"]["max"] for row in rows])
    vram_gib = np.asarray([row["resource"]["gpu_memory_used_mib"]["max"] / 1024 for row in rows])
    predictions_k = np.asarray([row["predictions_per_second"] / 1000 for row in rows])
    current_vram_gib = rows[0]["resource_samples"][0]["gpus"][0]["memory_total_mib"] / 1024

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.55))
    fig.subplots_adjust(left=0.10, right=0.93, top=0.96, bottom=0.11, hspace=0.52, wspace=0.43)

    # (a) Production CUDA kernel: compute saturation.
    ax = axes[0, 0]
    upper = np.maximum(gpu_max - gpu_mean, 0)
    ax.errorbar(
        batch,
        gpu_mean,
        yerr=np.vstack([np.zeros_like(upper), upper]),
        color=ORANGE,
        marker="s",
        fillstyle="none",
        markersize=3.8,
        linewidth=1.2,
        elinewidth=0.8,
        capsize=2.2,
        label="Mean; maximum whisker",
    )
    for xpos, value in zip(batch, gpu_mean):
        ax.text(xpos, value - 5.5 if value > 92 else value + 3.2, f"{value:.1f}", ha="center", va="center", fontsize=6.7)
    ax.axhline(100, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), label="Hardware ceiling")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch, [str(value) for value in batch])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Batch size per GEM")
    ax.set_ylabel("GPU utilization (%)")
    ax.set_title("(a) GPU compute saturation", loc="left", fontweight="normal")
    ax.legend(frameon=False, loc="lower right")
    style(ax)

    # (b) Production CUDA kernel: VRAM pressure and throughput trade-off.
    ax = axes[0, 1]
    x = np.arange(len(batch))
    bars = ax.bar(x, vram_gib, width=0.55, color=GREEN, edgecolor=INK, linewidth=0.35, hatch="////", label="Peak VRAM")
    for bar, value in zip(bars, vram_gib):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.16, f"{value:.1f}", ha="center", va="bottom", fontsize=6.7)
    capacity = ax.axhline(current_vram_gib, color=GREY, linewidth=0.9, linestyle=(0, (3, 2)), label="RTX 4060 capacity")
    ax.set_xticks(x, [str(value) for value in batch])
    ax.set_ylim(0, current_vram_gib * 1.10)
    ax.set_xlabel("Batch size per GEM")
    ax.set_ylabel("Peak VRAM (GiB)", color=GREEN)
    ax.tick_params(axis="y", colors=GREEN)
    twin = ax.twinx()
    throughput_line, = twin.plot(x, predictions_k, color=BLUE, marker="o", markersize=3.5, linewidth=1.2, label="Throughput")
    twin.set_ylim(0, max(predictions_k) * 1.25)
    twin.set_ylabel(r"Predictions ($10^3$ s$^{-1}$)", color=BLUE)
    style_twin(twin, BLUE)
    ax.set_title("(b) VRAM pressure and throughput", loc="left", fontweight="normal")
    ax.legend([bars, capacity, throughput_line], ["Peak VRAM", "RTX 4060 capacity", "Throughput"], frameon=False, loc="upper left")
    style(ax)

    # (c) Equal-work CPU/CUDA rollouts: host compute and unique memory load.
    ax = axes[1, 0]
    cases = []
    for backend_key, backend_label in (("cpu_highs", "CPU"), ("cuda_surrogate_batched", "CUDA")):
        for row in parallel[backend_key]:
            cases.append(
                {
                    "label": f"{backend_label}\n{row['n_envs']} env",
                    "backend": backend_label,
                    "environments": row["n_envs"],
                    "system_cpu_percent_mean": row["resource"]["system_cpu_percent"]["mean"],
                    "process_pss_gib_max": row["resource"]["process_pss_mib"]["max"] / 1024,
                    "transitions_per_second": row["transitions_per_second"],
                }
            )
    x = np.arange(len(cases))
    cpu_load = [row["system_cpu_percent_mean"] for row in cases]
    pss = [row["process_pss_gib_max"] for row in cases]
    colors = [GREY if row["backend"] == "CPU" else BLUE for row in cases]
    bars = ax.bar(x, cpu_load, width=0.58, color=colors, edgecolor=INK, linewidth=0.35, label="System CPU")
    for bar, value in zip(bars, cpu_load):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}", ha="center", va="bottom", fontsize=6.7)
    ax.set_xticks(x, [row["label"] for row in cases])
    ax.set_ylim(0, 108)
    ax.set_ylabel("System CPU utilization (%)")
    twin = ax.twinx()
    twin.plot(x, pss, color=GREEN, marker="D", markersize=3.8, linewidth=1.1, label="Process PSS")
    for xpos, value in zip(x, pss):
        twin.text(xpos, value + 0.17, f"{value:.2f}", ha="center", va="bottom", fontsize=6.5, color=GREEN)
    twin.set_ylim(0, max(pss) * 1.35)
    twin.set_ylabel("Process PSS (GiB)", color=GREEN)
    style_twin(twin, GREEN)
    ax.set_title("(c) Parallel-environment host load", loc="left", fontweight="normal")
    style(ax)

    # (d) Nominal capacity of the proposed workstation, normalized to current.
    ax = axes[1, 1]
    current = evidence["measurement_hardware"]
    proposed = evidence["proposed_hardware"]
    capacity_labels = ["CPU\nthreads", "System\nRAM", "VRAM /\nGPU", "Aggregate\nVRAM"]
    proposed_ratio = np.asarray(
        [
            proposed["threads"] / current["logical_threads_available"],
            proposed["ram_gb"] / current["physical_ram_gb"],
            proposed["vram_gb_per_gpu"] / current["vram_gb_per_gpu"],
            proposed["aggregate_vram_gb_not_pooled"] / (current["gpu_count"] * current["vram_gb_per_gpu"]),
        ]
    )
    x = np.arange(len(capacity_labels))
    width = 0.34
    ax.bar(x - width / 2, np.ones_like(proposed_ratio), width, color=GREY, edgecolor=INK, linewidth=0.35, label="Current")
    proposed_bars = ax.bar(x + width / 2, proposed_ratio, width, color=BLUE, edgecolor=INK, linewidth=0.35, hatch="////", label="Proposed")
    for bar, value in zip(proposed_bars, proposed_ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.22, f"{value:.2g}×", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x, capacity_labels)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Capacity relative to current")
    ax.set_title("(d) Proposed hardware capacity", loc="left", fontweight="normal")
    ax.legend(frameon=False, loc="upper left")
    style(ax)

    save(fig)
    plt.close(fig)

    output = {
        "figure": "resource_usage_publication",
        "definition": {
            "a": "production CUDA-kernel compute saturation",
            "b": "production CUDA-kernel VRAM pressure and throughput trade-off",
            "c": "system CPU utilization and proportional set size during equal-work parallel rollouts",
            "d": "nominal proposed-hardware capacity relative to the current system",
        },
        "source_logs": [
            "gpu_saturation_2048_multistream_rtx4060.json",
            "hardware_resource_parallel_rtx4060.json",
            "workstation_procurement_evidence.json",
        ],
        "excluded_source_logs": {
            "resource_usage_short_rtx4060.json": "24-step smoke-test context baseline; not used in this purchase figure"
        },
        "measurement_hardware": current,
        "proposed_hardware": proposed,
        "gpu_kernel": [
            {
                "batch_size_per_gem": int(batch_value),
                "gpu_utilization_percent_mean": float(mean),
                "gpu_utilization_percent_max": float(maximum),
                "peak_vram_gib": float(memory),
                "predictions_per_second": float(rate * 1000),
            }
            for batch_value, mean, maximum, memory, rate in zip(batch, gpu_mean, gpu_max, vram_gib, predictions_k)
        ],
        "parallel_environment_host_load": cases,
        "proposed_capacity_ratio": dict(zip(capacity_labels, map(float, proposed_ratio))),
        "notes": [
            "PSS is used instead of summed RSS so shared pages are apportioned rather than double-counted.",
            "Panel d reports nominal capacity, not measured application speed.",
            "Aggregate VRAM is distributed across three independent GPUs and is not pooled.",
        ],
    }
    (RESULTS / "resource_usage_publication.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved resource_usage_publication.png/.pdf/.svg/.json")


if __name__ == "__main__":
    main()
