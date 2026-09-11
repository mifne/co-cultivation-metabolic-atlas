#!/usr/bin/env python3
"""Create a one-page, evidence-separated GPU procurement figure and data file."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

MEASURED_COLOR = "#0B6E4F"
MEASURED_ALT = "#2F80ED"
PROJECTED_COLOR = "#F2994A"
CPU_COLOR = "#9B2C2C"
GRID_COLOR = "#D8DEE9"
TEXT_COLOR = "#14213D"


def load_inputs() -> tuple[dict, dict]:
    rollout = json.loads(
        (RESULTS / "rollout_cpu_vs_gpu_rtx4060.json").read_text(encoding="utf-8")
    )
    batch = json.loads(
        (RESULTS / "fba_surrogate_gpu_lp_rtx4060.json").read_text(encoding="utf-8")
    )
    return rollout, batch


def mean_timing(rows: list[dict]) -> dict[str, float]:
    keys = ["pre_solve_seconds", "solve_seconds", "post_solve_seconds"]
    return {
        key: statistics.fmean(row["diagnostics"]["mean_step_timing"][key] for row in rows)
        for key in keys
    }


def build_summary(rollout: dict, batch: dict) -> dict:
    cpu_rows = rollout["highs"]["runs"]
    gpu_rows = rollout["surrogate_cuda"]["runs"]
    paired_speedups = [
        cpu["seconds"] / gpu["seconds"] for cpu, gpu in zip(cpu_rows, gpu_rows)
    ]
    paired_mean = statistics.fmean(paired_speedups)
    paired_ci95 = 1.96 * statistics.stdev(paired_speedups) / np.sqrt(len(paired_speedups))
    cpu_timing = mean_timing(cpu_rows)
    gpu_timing = mean_timing(gpu_rows)

    # RTX PRO 4000 SFF official: 432 GB/s, 24 TFLOPS FP32, 24 GB.
    # Local RTX 4060 Laptop reference: 256 GB/s, 14.6 TFLOPS, 8 GB.
    memory_ratio = 432.0 / 256.0
    fp32_ratio = 24.0 / 14.6
    conservative_gpu_ratio = min(memory_ratio, fp32_ratio)
    scaling_efficiency = 0.85
    pure_gpu_3x_plan = conservative_gpu_ratio * 3.0 * scaling_efficiency
    pure_gpu_3x_low = conservative_gpu_ratio * 3.0 * 0.80
    pure_gpu_3x_high = conservative_gpu_ratio * 3.0

    full_vs_sff_gpu_ratio = min(672.0 / 432.0, 40.0 / 24.0)
    full_gpu_ratio = conservative_gpu_ratio * full_vs_sff_gpu_ratio
    full_pure_gpu_3x_plan = full_gpu_ratio * 3.0 * scaling_efficiency
    full_pure_gpu_3x_low = full_gpu_ratio * 3.0 * 0.80
    full_pure_gpu_3x_high = full_gpu_ratio * 3.0

    inference_seconds = statistics.median(
        row["diagnostics"]["surrogate"]["inference_seconds"] for row in gpu_rows
    )
    measured_gpu_seconds = rollout["surrogate_cuda"]["mean_seconds"]
    inference_fraction = inference_seconds / measured_gpu_seconds
    single_pro_rollout_ratio = 1.0 / (
        (1.0 - inference_fraction) + inference_fraction / conservative_gpu_ratio
    )
    end_to_end_3x_plan = single_pro_rollout_ratio * 3.0 * scaling_efficiency
    end_to_end_3x_low = single_pro_rollout_ratio * 3.0 * 0.80
    end_to_end_3x_high = single_pro_rollout_ratio * 3.0
    single_full_rollout_ratio = 1.0 / (
        (1.0 - inference_fraction) + inference_fraction / full_gpu_ratio
    )
    full_end_to_end_3x_plan = single_full_rollout_ratio * 3.0 * scaling_efficiency
    full_end_to_end_3x_low = single_full_rollout_ratio * 3.0 * 0.80
    full_end_to_end_3x_high = single_full_rollout_ratio * 3.0

    species_aliases = {
        "Actinoplanes_sp_OR16_lcp": "OR16",
        "NS21_lcp_pha": "NS21",
        "Lactobacillus_plantarum_iNF517": "L. plantarum",
    }
    batch_series = {}
    for species, values in batch["species"].items():
        batch_series[species_aliases[species]] = {
            "batch_size": [row["batch_size"] for row in values["surrogate"]],
            "guarded_lp_per_second": [
                row["guarded_predictions_per_second"] for row in values["surrogate"]
            ],
            "cpu_lp_per_second": values["exact_cpu"]["lp_solves_per_second"],
            "guard_acceptance_rate": values["accuracy"]["guard_acceptance_rate"],
        }

    return {
        "measurement": {
            "date": rollout["timestamp_date"],
            "gpu": rollout["gpu_name"],
            "steps_per_run": rollout["steps_per_run"],
            "repeats": rollout["repeats"],
            "cpu_mean_seconds": rollout["highs"]["mean_seconds"],
            "cpu_ci95_seconds": rollout["highs"]["ci95_seconds"],
            "gpu_mean_seconds": measured_gpu_seconds,
            "gpu_ci95_seconds": rollout["surrogate_cuda"]["ci95_seconds"],
            "paired_speedup_mean": paired_mean,
            "paired_speedup_ci95": float(paired_ci95),
            "cpu_mean_step_timing": cpu_timing,
            "gpu_mean_step_timing": gpu_timing,
            "batch_series": batch_series,
        },
        "hardware": {
            "rtx_4060_laptop": {
                "vram_gb": 8,
                "memory_bandwidth_gbs": 256,
                "fp32_tflops_reference": 14.6,
                "ai_tops": 233,
            },
            "rtx_pro_4000_blackwell_sff": {
                "count": 3,
                "vram_gb_per_card": 24,
                "aggregate_vram_gb_not_pooled": 72,
                "memory_bandwidth_gbs_per_card": 432,
                "fp32_tflops_per_card": 24,
                "ai_tops_per_card": 770,
                "power_w_per_card": 70,
            },
            "rtx_pro_4000_blackwell_full_height": {
                "count": 3,
                "vram_gb_per_card": 24,
                "aggregate_vram_gb_not_pooled": 72,
                "memory_bandwidth_gbs_per_card": 672,
                "fp32_tflops_per_card": 40,
                "ai_tops_per_card": 1290,
                "power_w_per_card": 140,
            },
        },
        "projection": {
            "method": "min(memory-bandwidth ratio, FP32 ratio); multi-GPU 80-100%, planning 85%",
            "single_pro_pure_gpu_factor": conservative_gpu_ratio,
            "three_pro_pure_gpu_factor_low": pure_gpu_3x_low,
            "three_pro_pure_gpu_factor_plan": pure_gpu_3x_plan,
            "three_pro_pure_gpu_factor_high": pure_gpu_3x_high,
            "measured_inference_fraction_of_rollout": inference_fraction,
            "single_pro_end_to_end_factor": single_pro_rollout_ratio,
            "three_pro_end_to_end_factor_low": end_to_end_3x_low,
            "three_pro_end_to_end_factor_plan": end_to_end_3x_plan,
            "three_pro_end_to_end_factor_high": end_to_end_3x_high,
            "full_vs_sff_pure_gpu_factor": full_vs_sff_gpu_ratio,
            "single_full_pure_gpu_factor": full_gpu_ratio,
            "three_full_pure_gpu_factor_low": full_pure_gpu_3x_low,
            "three_full_pure_gpu_factor_plan": full_pure_gpu_3x_plan,
            "three_full_pure_gpu_factor_high": full_pure_gpu_3x_high,
            "single_full_end_to_end_factor": single_full_rollout_ratio,
            "three_full_end_to_end_factor_low": full_end_to_end_3x_low,
            "three_full_end_to_end_factor_plan": full_end_to_end_3x_plan,
            "three_full_end_to_end_factor_high": full_end_to_end_3x_high,
        },
        "scope": {
            "artifact_candidates": 102,
            "artifact_disk_mb": 6.8,
            "conclusion": "three GPUs are justified for concurrent environment/seed throughput, not single-environment latency",
        },
    }


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(colors=TEXT_COLOR)


def plot(summary: dict, png_path: Path, pdf_path: Path) -> None:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    font_manager.fontManager.addfont(font_path)
    font = font_manager.FontProperties(fname=font_path)
    plt.rcParams.update({
        "font.family": font.get_name(),
        "axes.unicode_minus": False,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "figure.facecolor": "#F7F9FC",
        "axes.facecolor": "white",
    })

    m = summary["measurement"]
    p = summary["projection"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.11, hspace=0.42, wspace=0.26)
    fig.suptitle(
        "RTX PRO 4000 Blackwell ×3：SFF版・非SFF版の導入比較",
        fontsize=22,
        fontweight="bold",
        x=0.07,
        ha="left",
    )
    fig.text(
        0.07,
        0.915,
        "3 GEM dFBA-RL／RTX 4060 Laptop 実測（n=5）｜橙色は仕様に基づく予測",
        fontsize=11,
        color="#52606D",
    )

    # A: repeatable end-to-end evidence.
    ax = axes[0, 0]
    labels = ["CPU HiGHS", "RTX 4060\nGPUサロゲート"]
    means = [m["cpu_mean_seconds"], m["gpu_mean_seconds"]]
    errors = [m["cpu_ci95_seconds"], m["gpu_ci95_seconds"]]
    bars = ax.bar(labels, means, yerr=errors, capsize=6, color=[CPU_COLOR, MEASURED_COLOR], width=0.62)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(means)*0.055, f"{value:.2f}秒", ha="center", fontweight="bold")
    ax.text(0.98, 0.86, f"{m['paired_speedup_mean']:.2f}× ± {m['paired_speedup_ci95']:.2f}", transform=ax.transAxes, ha="right", fontsize=17, fontweight="bold", color=MEASURED_COLOR)
    ax.text(0.98, 0.76, "ペア比較・95% CI", transform=ax.transAxes, ha="right", fontsize=9, color="#52606D")
    ax.set_ylabel("24ステップ実行時間（秒、低いほど良い）")
    ax.set_title("A. GPU化の全体効果（実測）", loc="left", fontweight="bold")
    style_axis(ax)

    # B: where time remains after acceleration.
    ax = axes[0, 1]
    timing_labels = ["CPU HiGHS", "RTX 4060 GPU"]
    cpu_t = m["cpu_mean_step_timing"]
    gpu_t = m["gpu_mean_step_timing"]
    timing_keys = ["pre_solve_seconds", "solve_seconds", "post_solve_seconds"]
    timing_names = ["FBA前処理", "FBA／fallback", "状態更新"]
    timing_colors = ["#7EA6E0", "#D9534F", "#9AC6A7"]
    left = np.zeros(2)
    for key, label, color in zip(timing_keys, timing_names, timing_colors):
        values = np.array([cpu_t[key], gpu_t[key]])
        ax.barh(timing_labels, values, left=left, label=label, color=color, height=0.55)
        left += values
    cpu_share = cpu_t["solve_seconds"] / sum(cpu_t.values())
    gpu_share = gpu_t["solve_seconds"] / sum(gpu_t.values())
    ax.text(left[0] * 1.01, 0, f"FBA {cpu_share:.0%}", va="center", fontweight="bold", color=CPU_COLOR)
    ax.text(left[1] * 1.01, 1, f"FBA {gpu_share:.0%}", va="center", fontweight="bold", color=MEASURED_COLOR)
    ax.set_xlabel("平均1ステップ時間（秒）")
    ax.set_title("B. 律速段階の移動（実測）", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, -0.34))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    # C: current GPU batching scalability.
    ax = axes[1, 0]
    palette = ["#0B6E4F", "#2F80ED", "#8E5BB7"]
    for (species, values), color in zip(m["batch_series"].items(), palette):
        ax.plot(values["batch_size"], values["guarded_lp_per_second"], marker="o", linewidth=2.2, markersize=5, label=species, color=color)
        ax.axhline(values["cpu_lp_per_second"], color=color, linestyle=":", linewidth=1, alpha=0.45)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 8, 16, 32, 64], labels=["1", "4", "8", "16", "32", "64"])
    ax.set_xlabel("同時環境数／GPUバッチ")
    ax.set_ylabel("安全検査込みFBA相当処理数／秒")
    ax.set_title("C. 環境並列化でGPUを使い切る（実測）", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    style_axis(ax)

    # D: compare SFF and full-height projections on the same workload metrics.
    ax = axes[1, 1]
    labels = ["単一run\n1GPU", "総rollout\n3GPU", "純GPU FBA容量\n3GPU"]
    sff_values = [p["single_pro_end_to_end_factor"], p["three_pro_end_to_end_factor_plan"], p["three_pro_pure_gpu_factor_plan"]]
    full_values = [p["single_full_end_to_end_factor"], p["three_full_end_to_end_factor_plan"], p["three_full_pure_gpu_factor_plan"]]
    positions = np.arange(len(labels))
    width = 0.34
    sff_bars = ax.bar(positions - width / 2, sff_values, width, color="#F5B971", hatch="//", label="SFF 70W/枚")
    full_bars = ax.bar(positions + width / 2, full_values, width, color=PROJECTED_COLOR, hatch="xx", label="非SFF 140W/枚")
    for bars, values in ((sff_bars, sff_values), (full_bars, full_values)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.12, f"{value:.2f}×", ha="center", fontweight="bold", fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, max(p["three_full_pure_gpu_factor_high"] * 1.12, 7.4))
    ax.set_ylabel("現行RTX 4060比")
    ax.set_title("D. SFF版 vs 非SFF版（予測）", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.text(0.98, 0.95, "非SFF: 帯域 +56% / FP32 +67%", transform=ax.transAxes, ha="right", fontsize=9.5, color="#8A4B08")
    ax.text(0.98, 0.88, "ただし消費電力は2倍", transform=ax.transAxes, ha="right", fontsize=9.5, color=CPU_COLOR)
    style_axis(ax)

    fig.text(
        0.07,
        0.035,
        "判断：現行規模ではSFF/非SFFの単一run差は約2%。大規模バッチでは非SFFの純GPU容量がSFF比約1.56倍。"
        " 3GPU効率80–100%（計画85%）。VRAMは両者24GB/枚、合計72GB・非共有。消費電力はSFF 210W、非SFF 420W。",
        fontsize=9.5,
        color="#3D4A5C",
    )
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    rollout, batch = load_inputs()
    summary = build_summary(rollout, batch)
    data_path = RESULTS / "gpu_procurement_case_rtx_pro_4000x3.json"
    png_path = RESULTS / "gpu_procurement_case_rtx_pro_4000x3.png"
    pdf_path = RESULTS / "gpu_procurement_case_rtx_pro_4000x3.pdf"
    data_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(summary, png_path, pdf_path)
    print(f"saved: {data_path}\nsaved: {png_path}\nsaved: {pdf_path}")


if __name__ == "__main__":
    main()
