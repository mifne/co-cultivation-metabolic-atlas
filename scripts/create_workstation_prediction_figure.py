#!/usr/bin/env python3
"""Create a publication figure for the proposed 9960X/RTX PRO workstation.

Measured RTX 4060 behavior is kept separate from scenario projections.  The
projection combines an Amdahl model for one 16-environment shard with explicit
multi-GPU efficiency assumptions for independent shards.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PDF_OUTPUT = ROOT / "output" / "pdf"

MEASURED = "#245A73"
MEASURED_ALT = "#3E8E82"
PROJECTED = "#C6613A"
CURRENT = "#777B80"
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
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style(ax, *, xgrid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", colors=INK)


def efficiency_curve(n_gpu: np.ndarray, efficiency_at_three: float) -> np.ndarray:
    """Linear interpolation from 100% at one GPU to the 3-GPU scenario."""
    return 1.0 - (1.0 - efficiency_at_three) * (n_gpu - 1.0) / 2.0


def amdahl_speedup(accelerated_fraction: float, acceleration: float) -> float:
    return 1.0 / ((1.0 - accelerated_fraction) + accelerated_fraction / acceleration)


def build_summary() -> dict:
    long_run = read("parallel_env_16_long_multistream_rtx4060.json")
    saturation = read("gpu_saturation_2048_multistream_rtx4060.json")
    resources = read("resource_usage_publication.json")

    long_gpu = long_run["cuda_surrogate_batched"][0]
    measured_rollout_seconds = float(long_gpu["elapsed_seconds"])
    measured_inference_seconds = float(long_gpu["service"]["inference_seconds"])
    accelerated_fraction = measured_inference_seconds / measured_rollout_seconds
    baseline_transitions_per_second = float(long_gpu["transitions_per_second"])
    best_kernel = max(saturation["rows"], key=lambda row: row["predictions_per_second"])
    baseline_predictions_per_second = float(best_kernel["predictions_per_second"])

    # Official RTX PRO 4000 Blackwell full-height values (NVIDIA datasheet,
    # June 2026): 37 TFLOPS FP32, 672 GB/s, 24 GB, 145 W.
    # The local RTX 4060 Laptop reference follows the measured workstation
    # evidence: 256 GB/s and 3,072 CUDA cores at 2.37 GHz maximum boost.
    rtx4060_fp32_tflops = 3072 * 2 * 2.37 / 1000
    upper_single_factor = min(672.0 / 256.0, 37.0 / rtx4060_fp32_tflops)
    scenarios = {
        "low": {"single_gpu_factor": 1.70, "efficiency_at_three": 0.80},
        "planning": {"single_gpu_factor": 2.20, "efficiency_at_three": 0.90},
        "high": {"single_gpu_factor": upper_single_factor, "efficiency_at_three": 0.95},
    }
    gpu_counts = np.asarray([1.0, 2.0, 3.0])
    predictions = {}
    for name, values in scenarios.items():
        single_factor = values["single_gpu_factor"]
        efficiency = efficiency_curve(gpu_counts, values["efficiency_at_three"])
        gpu_stage_factor = single_factor * gpu_counts * efficiency
        per_shard_e2e = amdahl_speedup(accelerated_fraction, single_factor)
        e2e_factor = per_shard_e2e * gpu_counts * efficiency
        predictions[name] = {
            "gpu_count": gpu_counts.astype(int).tolist(),
            "efficiency": efficiency.tolist(),
            "gpu_stage_factor": gpu_stage_factor.tolist(),
            "gpu_stage_predictions_per_second": (baseline_predictions_per_second * gpu_stage_factor).tolist(),
            "per_shard_end_to_end_factor": per_shard_e2e,
            "aggregate_end_to_end_factor": e2e_factor.tolist(),
            "aggregate_transitions_per_second": (baseline_transitions_per_second * e2e_factor).tolist(),
            "fixed_work_end_to_end_time_percent_at_three": float(100.0 / e2e_factor[-1]),
            "fixed_work_gpu_stage_time_percent_at_three": float(100.0 / gpu_stage_factor[-1]),
        }

    current = resources["measurement_hardware"]
    proposed = resources["proposed_hardware"]
    capacity = {
        "gpu_shards": 3.0,
        "concurrent_environments": 48.0 / 16.0,
        "cpu_threads": proposed["threads"] / current["logical_threads_available"],
        "system_ram": proposed["ram_gb"] / current["physical_ram_gb"],
        "vram_per_gpu": proposed["vram_gb_per_gpu"] / current["vram_gb_per_gpu"],
        "aggregate_vram": proposed["aggregate_vram_gb_not_pooled"] / (
            current["gpu_count"] * current["vram_gb_per_gpu"]
        ),
    }

    return {
        "figure": "workstation_predicted_behavior",
        "created_date": "2026-08-28",
        "scope": "three-GEM dFBA reinforcement-learning exploration",
        "source_logs": [
            "parallel_env_16_long_multistream_rtx4060.json",
            "gpu_saturation_2048_multistream_rtx4060.json",
            "resource_usage_publication.json",
        ],
        "measured_baseline": {
            "hardware": current,
            "parallel_environments": 16,
            "rollout_transitions": int(long_gpu["transitions"]),
            "rollout_elapsed_seconds": measured_rollout_seconds,
            "rollout_transitions_per_second": baseline_transitions_per_second,
            "gpu_service_inference_seconds": measured_inference_seconds,
            "gpu_accelerated_fraction": accelerated_fraction,
            "kernel_optimal_batch_per_gem": int(best_kernel["batch_size_per_species"]),
            "kernel_predictions_per_second": baseline_predictions_per_second,
        },
        "proposed_hardware": {
            **proposed,
            "fp32_tflops_per_gpu": 37,
            "power_w_per_gpu": 145,
            "official_sources": {
                "cpu": "https://www.amd.com/en/products/processors/ryzen-threadripper/9000-series/amd-ryzen-threadripper-9960x.html",
                "gpu": "https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-4000/workstation-datasheet-rtx-pro-4000-nvidia-us-web.pdf",
            },
        },
        "projection_model": {
            "amdahl_accelerated_fraction": accelerated_fraction,
            "single_gpu_factor_method": "scenario range bounded by memory-bandwidth and FP32 ratios",
            "rtx4060_reference_fp32_tflops": rtx4060_fp32_tflops,
            "rtx_pro_4000_official_fp32_tflops": 37.0,
            "rtx4060_reference_memory_bandwidth_gbs": 256.0,
            "rtx_pro_4000_official_memory_bandwidth_gbs": 672.0,
            "scenarios": scenarios,
            "two_gpu_efficiency_method": "linear interpolation between 100% at one GPU and the 3-GPU scenario",
            "multi_gpu_scope": "independent environment/seed shards; no VRAM pooling",
        },
        "predictions": predictions,
        "capacity_ratio": capacity,
        "interpretation": [
            "The planning scenario predicts about 5.94x aggregate GPU-stage capacity with three GPUs.",
            "The planning scenario predicts about 3.50x aggregate end-to-end throughput for three independent shards.",
            "The difference is the non-GPU fraction measured in the current 16-environment rollout.",
            "Scenario ranges are planning bounds, not confidence intervals or measurements on the proposed workstation.",
        ],
    }


def plot(summary: dict) -> None:
    configure()
    baseline = summary["measured_baseline"]
    predictions = summary["predictions"]
    counts = np.asarray(predictions["planning"]["gpu_count"])
    low = predictions["low"]
    plan = predictions["planning"]
    high = predictions["high"]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.965, bottom=0.105, hspace=0.50, wspace=0.40)

    # (a) Measured wall-time decomposition used by the Amdahl model.
    ax = axes[0, 0]
    accelerated = baseline["gpu_accelerated_fraction"] * 100
    other = 100 - accelerated
    ax.barh([0], [other], color=CURRENT, edgecolor=INK, linewidth=0.35, height=0.48, label="Environment and overhead")
    ax.barh([0], [accelerated], left=[other], color=MEASURED, edgecolor=INK, linewidth=0.35, hatch="////", height=0.48, label="GPU inference")
    ax.text(other / 2, 0, f"Environment + overhead\n{other:.1f}%", ha="center", va="center", color="white", fontsize=7.0)
    ax.text(other + accelerated / 2, 0, f"GPU inference\n{accelerated:.1f}%", ha="center", va="center", color="white", fontsize=7.0)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Measured wall-time share (%)")
    ax.set_title("(a) Rollout decomposition", loc="left", fontweight="normal")
    style(ax, xgrid=True)

    # (b) Projected GPU-only stage capacity.
    ax = axes[0, 1]
    low_gpu = np.asarray(low["gpu_stage_factor"])
    plan_gpu = np.asarray(plan["gpu_stage_factor"])
    high_gpu = np.asarray(high["gpu_stage_factor"])
    ax.fill_between(counts, low_gpu, high_gpu, color=PROJECTED, alpha=0.18, linewidth=0, label="Scenario range")
    ax.plot(counts, plan_gpu, color=PROJECTED, marker="s", fillstyle="none", linewidth=1.3, markersize=4, label="Planning")
    ax.axhline(1, color=CURRENT, linewidth=0.8, linestyle=(0, (3, 2)), label="Current RTX 4060 ×1")
    for x, value in zip(counts, plan_gpu):
        ax.text(x, value + 0.22, f"{value:.2f}×", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(counts)
    ax.set_ylim(0, max(high_gpu) * 1.15)
    ax.set_xlabel("Independent RTX PRO 4000 GPUs")
    ax.set_ylabel("Relative GPU-stage throughput")
    ax.set_title("(b) GPU-stage scale-out", loc="left", fontweight="normal")
    ax.legend(frameon=False, loc="upper left")
    style(ax)

    # (c) End-to-end aggregate throughput for independent 16-env shards.
    ax = axes[0, 2]
    low_e2e = np.asarray(low["aggregate_transitions_per_second"])
    plan_e2e = np.asarray(plan["aggregate_transitions_per_second"])
    high_e2e = np.asarray(high["aggregate_transitions_per_second"])
    ax.fill_between(counts, low_e2e, high_e2e, color=MEASURED, alpha=0.16, linewidth=0, label="Scenario range")
    ax.plot(counts, plan_e2e, color=MEASURED, marker="o", linewidth=1.3, markersize=4, label="Planning")
    ax.axhline(baseline["rollout_transitions_per_second"], color=CURRENT, linewidth=0.8, linestyle=(0, (3, 2)), label="Measured current")
    for x, value in zip(counts, plan_e2e):
        ax.text(x, value + 10, f"{value:.0f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(counts)
    ax.set_ylim(0, max(high_e2e) * 1.17)
    ax.set_xlabel("Independent 16-env GPU shards")
    ax.set_ylabel(r"Aggregate transitions s$^{-1}$")
    ax.set_title("(c) End-to-end throughput", loc="left", fontweight="normal")
    ax.legend(frameon=False, loc="upper left")
    style(ax)

    # (d) Absolute production-kernel capacity.
    ax = axes[1, 0]
    low_kernel = np.asarray(low["gpu_stage_predictions_per_second"]) / 1000
    plan_kernel = np.asarray(plan["gpu_stage_predictions_per_second"]) / 1000
    high_kernel = np.asarray(high["gpu_stage_predictions_per_second"]) / 1000
    ax.fill_between(counts, low_kernel, high_kernel, color=MEASURED_ALT, alpha=0.17, linewidth=0, label="Scenario range")
    ax.plot(counts, plan_kernel, color=MEASURED_ALT, marker="D", linewidth=1.3, markersize=3.8, label="Planning")
    ax.axhline(baseline["kernel_predictions_per_second"] / 1000, color=CURRENT, linewidth=0.8, linestyle=(0, (3, 2)), label="Measured RTX 4060")
    for x, value in zip(counts, plan_kernel):
        ax.text(x, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(counts)
    ax.set_ylim(0, max(high_kernel) * 1.17)
    ax.set_xlabel("Independent RTX PRO 4000 GPUs")
    ax.set_ylabel(r"Predictions ($10^3$ s$^{-1}$)")
    ax.set_title("(d) Production-kernel capacity", loc="left", fontweight="normal")
    ax.legend(frameon=False, loc="upper left")
    style(ax)

    # (e) Fixed-work completion time, current normalized to 100%.
    ax = axes[1, 1]
    labels = ["End-to-end\nrollout", "GPU-only\nstage"]
    plan_time = np.asarray([
        plan["fixed_work_end_to_end_time_percent_at_three"],
        plan["fixed_work_gpu_stage_time_percent_at_three"],
    ])
    low_time = np.asarray([
        high["fixed_work_end_to_end_time_percent_at_three"],
        high["fixed_work_gpu_stage_time_percent_at_three"],
    ])
    high_time = np.asarray([
        low["fixed_work_end_to_end_time_percent_at_three"],
        low["fixed_work_gpu_stage_time_percent_at_three"],
    ])
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, [100, 100], width, color=CURRENT, edgecolor=INK, linewidth=0.35, label="Current")
    projected_bars = ax.bar(
        x + width / 2,
        plan_time,
        width,
        yerr=np.vstack([plan_time - low_time, high_time - plan_time]),
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        color=MEASURED,
        edgecolor=INK,
        linewidth=0.35,
        hatch="////",
        label="Proposed ×3",
    )
    for bar, value, upper_value in zip(projected_bars, plan_time, high_time):
        ax.text(bar.get_x() + bar.get_width() / 2, upper_value + 2.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Relative completion time (%)")
    ax.set_title("(e) Fixed-work time", loc="left", fontweight="normal")
    ax.legend(frameon=False, loc="upper right")
    style(ax)

    # (f) Nominal execution capacity of the complete workstation.
    ax = axes[1, 2]
    capacity = summary["capacity_ratio"]
    names = ["GPU shards", "Concurrent env", "CPU threads", "System RAM", "VRAM / GPU", "Aggregate VRAM"]
    values = [
        capacity["gpu_shards"],
        capacity["concurrent_environments"],
        capacity["cpu_threads"],
        capacity["system_ram"],
        capacity["vram_per_gpu"],
        capacity["aggregate_vram"],
    ]
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=[MEASURED, MEASURED, PROJECTED, PROJECTED, MEASURED_ALT, MEASURED_ALT], edgecolor=INK, linewidth=0.35, hatch="////")
    for bar, value in zip(bars, values):
        ax.text(value + 0.18, bar.get_y() + bar.get_height() / 2, f"{value:.2g}×", ha="left", va="center", fontsize=6.8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 10)
    ax.set_xlabel("Capacity relative to current")
    ax.set_title("(f) Workstation capacity", loc="left", fontweight="normal")
    style(ax, xgrid=True)

    metadata = {
        "Title": "Predicted behavior of the proposed 9960X and three-GPU workstation",
        "Creator": "co-cultivation benchmark pipeline",
        "Subject": "Scenario projection for three-GEM dFBA reinforcement learning",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "workstation_predicted_behavior.png", dpi=600, bbox_inches="tight", pad_inches=0.03, metadata=metadata)
    fig.savefig(RESULTS / "workstation_predicted_behavior.svg", bbox_inches="tight", pad_inches=0.03, metadata={"Title": metadata["Title"]})
    fig.savefig(PDF_OUTPUT / "workstation_predicted_behavior.pdf", bbox_inches="tight", pad_inches=0.03, metadata=metadata)
    plt.close(fig)


def main() -> None:
    summary = build_summary()
    (RESULTS / "workstation_predicted_behavior.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(summary)
    print("saved workstation_predicted_behavior.png/.svg/.json and output/pdf/workstation_predicted_behavior.pdf")


if __name__ == "__main__":
    main()
