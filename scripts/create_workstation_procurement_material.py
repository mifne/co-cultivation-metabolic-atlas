#!/usr/bin/env python3
"""Create evidence-separated figures for the proposed 9960X/3-GPU workstation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# Okabe-Ito color-blind-safe palette.  The figures must also remain legible
# when printed in grayscale, so series use distinct markers and/or hatching.
MEASURED = "#245A73"
MEASURED_2 = "#3E8E82"
PROJECTED = "#C6613A"
CURRENT = "#777B80"
GRID = "#D7DCE1"
INK = "#20252B"


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def build_summary() -> dict:
    saturation = load_json("gpu_saturation_2048_multistream_rtx4060.json")
    env_scaling = load_json("parallel_env_scaling_2048_shared_fork_rtx4060.json")
    long_run = load_json("parallel_env_16_long_multistream_rtx4060.json")
    exact = load_json("exact_fba_collection_scaling_current_cpu.json")

    saturation_rows = [
        {
            "batch": row["batch_size_per_species"],
            "gpu_util_mean_percent": row["resource"]["gpu_utilization_percent"]["mean"],
            "gpu_util_max_percent": row["resource"]["gpu_utilization_percent"]["max"],
            "vram_max_mib": row["resource"]["gpu_memory_used_mib"]["max"],
            "predictions_per_second": row["predictions_per_second"],
        }
        for row in saturation["rows"]
    ]
    cpu_env = env_scaling["cpu_highs"]
    gpu_env = env_scaling["cuda_surrogate_batched"]
    long_cpu = long_run["cpu_highs"][0]
    long_gpu = long_run["cuda_surrogate_batched"][0]

    # RTX PRO 4000 Blackwell full-height official specs (June 2026 datasheet):
    # 672 GB/s and 37 TFLOPS.
    # Local RTX 4060 Laptop reference is derived from its 128-bit, 16 Gbps GDDR6
    # configuration and 3,072 CUDA cores at the official maximum 2.37 GHz boost.
    bandwidth_ratio = 672.0 / 256.0
    fp32_ratio = 37.0 / (3072 * 2 * 2.37 / 1000)
    upper_single = min(bandwidth_ratio, fp32_ratio)
    projection = {
        "single_gpu_factor": {"low": 1.7, "planning": 2.2, "high": upper_single},
        "three_gpu_efficiency": {"low": 0.80, "planning": 0.90, "high": 0.95},
    }
    projection["three_gpu_factor"] = {
        key: projection["single_gpu_factor"][key]
        * 3
        * projection["three_gpu_efficiency"][key]
        for key in ("low", "planning", "high")
    }
    best_gpu_row = max(saturation_rows, key=lambda row: row["predictions_per_second"])
    projection["three_gpu_predictions_per_second"] = {
        key: best_gpu_row["predictions_per_second"] * value
        for key, value in projection["three_gpu_factor"].items()
    }

    return {
        "created_date": "2026-08-27",
        "workload": "3-GEM dFBA reinforcement-learning exploration",
        "measurement_hardware": {
            "cpu": "Intel Core i7-12650H",
            "logical_threads_available": 16,
            "physical_ram_gb": 48,
            "wsl_ram_visible_gib": exact["physical_ram_gib_visible_to_wsl"],
            "gpu": saturation["gpu_name"],
            "gpu_count": 1,
            "vram_gb_per_gpu": 8,
        },
        "proposed_hardware": {
            "cpu": "AMD Ryzen Threadripper 9960X",
            "cores": 24,
            "threads": 48,
            "cpu_tdp_w": 350,
            "platform": "TRX50 / sTR5 / 4-channel DDR5 RDIMM ECC",
            "ram_gb": 128,
            "gpu": "NVIDIA RTX PRO 4000 Blackwell (full-height)",
            "gpu_count": 3,
            "vram_gb_per_gpu": 24,
            "aggregate_vram_gb_not_pooled": 72,
            "memory_bandwidth_gbs_per_gpu": 672,
            "fp32_tflops_per_gpu": 37,
            "power_w_per_gpu": 145,
        },
        "measured": {
            "end_to_end_16_env": {
                "cpu_highs_transitions_per_second": long_cpu["transitions_per_second"],
                "gpu_transitions_per_second": long_gpu["transitions_per_second"],
                "speedup": long_gpu["transitions_per_second"] / long_cpu["transitions_per_second"],
                "gpu_acceptance_rate": long_gpu["solver"]["surrogate_acceptance_rate"],
                "cpu_fallback_solves": long_gpu["solver"]["cpu_fallback_solves"],
                "solve_attempts": long_gpu["solver"]["solve_attempts"],
            },
            "environment_scaling": {
                "cpu": [
                    {"environments": row["n_envs"], "transitions_per_second": row["transitions_per_second"]}
                    for row in cpu_env
                ],
                "gpu": [
                    {"environments": row["n_envs"], "transitions_per_second": row["transitions_per_second"]}
                    for row in gpu_env
                ],
            },
            "gpu_saturation": saturation_rows,
            "exact_fba_collection": exact["results"],
        },
        "projection": projection,
        "recommended_topology": {
            "independent_gpu_shards": 3,
            "environments_per_shard_initial": 16,
            "total_environments_initial": 48,
            "purpose": "one seed/agent/environment group per GPU; VRAM is not pooled",
        },
        "interpretation": [
            "Measured GPU FBA is 7.01x faster end-to-end than CPU HiGHS at 16 environments.",
            "The current 16-thread CPU is saturated: 32 environments are slower than 16.",
            "The actual FBA kernel reaches 98.3% mean GPU utilization at batch 128, but batch 64 maximizes throughput.",
            "Three GPUs are justified by independent seed/environment shards, not by pooling VRAM or accelerating one environment.",
            "Projected GPU-stage factors are scenarios, not measurements on the proposed workstation.",
        ],
    }


def configure_plotting() -> None:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style(ax, xgrid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", colors=INK)


def save_publication_figure(fig, basename: str) -> None:
    metadata = {
        "Title": basename.replace("_", " "),
        "Creator": "co-cultivation benchmark pipeline",
        "Subject": "3-GEM dFBA reinforcement-learning workstation benchmark",
    }
    fig.savefig(
        RESULTS / f"{basename}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=metadata,
    )
    fig.savefig(
        RESULTS / f"{basename}.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=metadata,
    )
    fig.savefig(
        RESULTS / f"{basename}.svg",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Title": metadata["Title"]},
    )


def make_overview(summary: dict) -> None:
    configure_plotting()
    measured = summary["measured"]
    projection = summary["projection"]
    proposed = summary["proposed_hardware"]
    current = summary["measurement_hardware"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(
        left=0.09,
        right=0.94,
        top=0.97,
        bottom=0.10,
        hspace=0.48,
        wspace=0.39,
    )

    # A. End-to-end evidence and current CPU saturation.
    ax = axes[0, 0]
    long = measured["end_to_end_16_env"]
    labels = ["CPU\n16 env", "GPU\n16 env", "GPU\n32 env"]
    values = [
        long["cpu_highs_transitions_per_second"],
        long["gpu_transitions_per_second"],
        measured["environment_scaling"]["gpu"][1]["transitions_per_second"],
    ]
    bars = ax.bar(
        labels,
        values,
        color=[CURRENT, MEASURED, MEASURED_2],
        edgecolor=INK,
        linewidth=0.5,
        width=0.62,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    ax.set_ylim(0, max(values) * 1.16)
    ax.set_ylabel(r"Transitions s$^{-1}$")
    ax.set_title("(a) Environment throughput", loc="left", fontweight="normal")
    style(ax)

    # B. Real GPU saturation: utilization and throughput do not peak together.
    ax = axes[0, 1]
    rows = measured["gpu_saturation"]
    batches = [row["batch"] for row in rows]
    util = [row["gpu_util_mean_percent"] for row in rows]
    pps = [row["predictions_per_second"] for row in rows]
    throughput_line = ax.plot(
        batches,
        pps,
        color=MEASURED,
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        label="Throughput",
    )[0]
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, labels=[str(x) for x in batches])
    ax.set_xlabel("Batch size per GEM")
    ax.set_ylabel(r"Predictions s$^{-1}$")
    twin = ax.twinx()
    utilization_line = twin.plot(
        batches,
        util,
        color=PROJECTED,
        marker="s",
        fillstyle="none",
        markersize=3.5,
        linewidth=1.1,
        label="GPU utilization",
    )[0]
    twin.set_ylabel("GPU utilization (%)")
    twin.set_ylim(0, 105)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    twin.spines["right"].set_color(INK)
    twin.tick_params(direction="out", colors=INK)
    ax.legend(
        [throughput_line, utilization_line],
        [throughput_line.get_label(), utilization_line.get_label()],
        frameon=False,
        loc="upper left",
        handlelength=1.6,
        borderaxespad=0.2,
    )
    ax.set_title("(b) GPU saturation", loc="left", fontweight="normal")
    style(ax)

    # C. Workstation resource capacity.
    ax = axes[1, 0]
    categories = ["CPU\nthreads", "RAM", "VRAM\nper GPU", "Aggregate\nVRAM"]
    old = np.array([current["logical_threads_available"], current["physical_ram_gb"], 8, 8], dtype=float)
    new = np.array([proposed["threads"], proposed["ram_gb"], 24, 72], dtype=float)
    ratios = new / old
    positions = np.arange(len(categories))
    bars = ax.bar(
        positions,
        ratios,
        width=0.64,
        color=MEASURED_2,
        edgecolor=INK,
        linewidth=0.5,
        hatch="//",
    )
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + 0.18,
            f"{ratio:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    ax.set_xticks(positions, categories)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Capacity ratio")
    ax.set_title("(c) Hardware capacity", loc="left", fontweight="normal")
    style(ax)

    # D. Projection range for the GPU-only stage.
    ax = axes[1, 1]
    labels = ["RTX 4060\n×1", "RTX PRO 4000\n×1", "RTX PRO 4000\n×3"]
    plan = [1.0, projection["single_gpu_factor"]["planning"], projection["three_gpu_factor"]["planning"]]
    low = [1.0, projection["single_gpu_factor"]["low"], projection["three_gpu_factor"]["low"]]
    high = [1.0, projection["single_gpu_factor"]["high"], projection["three_gpu_factor"]["high"]]
    errors = np.vstack([np.array(plan) - np.array(low), np.array(high) - np.array(plan)])
    bars = ax.bar(
        labels,
        plan,
        yerr=errors,
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        color=[CURRENT, PROJECTED, PROJECTED],
        edgecolor=INK,
        linewidth=0.5,
        width=0.58,
    )
    ax.set_ylim(0, max(high) * 1.14)
    ax.set_ylabel("Relative GPU-stage throughput")
    ax.set_title("(d) Projected GPU capacity", loc="left", fontweight="normal")
    style(ax)
    save_publication_figure(fig, "workstation_procurement_overview")
    plt.close(fig)


def make_cpu_and_memory(summary: dict) -> None:
    configure_plotting()
    measured = summary["measured"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.21, wspace=0.34)

    ax = axes[0]
    rows = measured["exact_fba_collection"]
    workers = [row["workers"] for row in rows]
    solves = [row["exact_lp_solves_per_second"] for row in rows]
    ax.plot(
        workers,
        solves,
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        color=MEASURED,
    )
    for x, value in zip(workers, solves):
        ax.text(x, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(workers)
    ax.set_ylim(4.2, 8.0)
    ax.set_xlabel("CPU workers")
    ax.set_ylabel(r"Exact LP solves s$^{-1}$")
    ax.set_title("(e) Exact-label collection", loc="left", fontweight="normal")
    style(ax)

    ax = axes[1]
    categories = ["Current\n1 GPU", "Proposed\n3 GPUs"]
    envs = [16, 48]
    bars = ax.bar(
        categories,
        envs,
        color=[CURRENT, MEASURED_2],
        edgecolor=INK,
        linewidth=0.5,
        hatch=[None, "//"],
        width=0.56,
    )
    for bar, value in zip(bars, envs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    ax.set_ylabel("Concurrent environments")
    ax.set_ylim(0, 56)
    ax.set_title("(f) Sharded execution target", loc="left", fontweight="normal")
    style(ax)
    save_publication_figure(fig, "workstation_cpu_ram_parallelism")
    plt.close(fig)


def main() -> None:
    summary = build_summary()
    (RESULTS / "workstation_procurement_evidence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_overview(summary)
    make_cpu_and_memory(summary)
    print("saved workstation_procurement_evidence.json and six figure files")


if __name__ == "__main__":
    main()
