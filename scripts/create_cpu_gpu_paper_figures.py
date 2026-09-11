#!/usr/bin/env python3
"""Create publication figures for the CPU-to-GPU cooperative dFBA study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "models/cooperative_surrogate/cooperative_neural_reranker_34448_dagger2_rebased.validation.json"
BASELINE_VALIDATION = ROOT / "models/cooperative_surrogate/cooperative_neural_reranker_33728_uniform_late_trained_pha256.validation.json"
DAGGER1_VALIDATION = ROOT / "results/pha_validation_dagger_rebased_5seed.json"
CALIBRATION = ROOT / "results/pha_hyperparameter_calibration_3seed.json"
PROFILE = ROOT / "results/cooperative_gpu_qp_only_33728_trained_k128_24h_seed20260901.json"
SCALING = ROOT / "results/gpu_qp_batch_scaling_rtx4060.json"
SMOKE = ROOT / "results/gpu_qp_parallel_training_smoke_dagger2.json"

COLORS = {
    "cpu": "#4C566A",
    "gpu": "#0072B2",
    "accent": "#D55E00",
    "green": "#009E73",
    "light_blue": "#DCEAF4",
    "light_gray": "#ECEFF4",
    "light_green": "#DCEFE8",
    "ink": "#1F2933",
    "muted": "#68737D",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def clean_axis(ax, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9DEE3", lw=0.55, alpha=0.8, zorder=0)


def save(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, face, edge, fontsize=8.0, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        transform=ax.transAxes,
    )
    return patch


def arrow(ax, start, end, color=None, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        lw=0.9,
        color=color or COLORS["muted"],
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)


def architecture_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_axis_off()
    ax.text(0.01, 0.94, "a", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.055, 0.94, "CPU reference path", fontsize=10, weight="bold", transform=ax.transAxes)
    ax.text(0.01, 0.46, "b", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.055, 0.46, "GPU online path", fontsize=10, weight="bold", transform=ax.transAxes)

    cpu_y = 0.67
    cpu_boxes = [
        (0.03, "Live dFBA\nstate", 0.12),
        (0.19, "Assemble\n6,585-flux LP", 0.15),
        (0.39, "Stage 1\nmax–min growth", 0.15),
        (0.59, "Stage 3\nparsimonious flux", 0.15),
        (0.79, "Flux vector\n+ state update", 0.16),
    ]
    for x, label, width in cpu_boxes:
        box(ax, (x, cpu_y), (width, 0.14), label, COLORS["light_gray"], COLORS["cpu"])
    for left, right in zip(cpu_boxes[:-1], cpu_boxes[1:]):
        arrow(ax, (left[0] + left[2], cpu_y + 0.07), (right[0], cpu_y + 0.07), COLORS["cpu"])
    ax.text(0.495, 0.60, "CPU HiGHS at every time step", ha="center", color=COLORS["cpu"], fontsize=8, transform=ax.transAxes)

    gpu_y = 0.17
    gpu_boxes = [
        (0.03, "Parallel RL\nenvironments", 0.12),
        (0.18, "CUDA\nmicrobatch\nservice", 0.15),
        (0.37, "Neural ranking\n34,448 anchors", 0.15),
        (0.56, "128-anchor\nfeasibility scan", 0.14),
        (0.76, "Batched QP\nprojection", 0.14),
    ]
    faces = [COLORS["light_blue"], COLORS["light_blue"], COLORS["light_blue"], COLORS["light_green"], COLORS["light_green"]]
    edges = [COLORS["gpu"], COLORS["gpu"], COLORS["gpu"], COLORS["green"], COLORS["green"]]
    for (x, label, width), face, edge in zip(gpu_boxes, faces, edges):
        box(ax, (x, gpu_y), (width, 0.14), label, face, edge, fontsize=7.2)
    for left, right in zip(gpu_boxes[:-1], gpu_boxes[1:]):
        arrow(ax, (left[0] + left[2], gpu_y + 0.07), (right[0], gpu_y + 0.07), COLORS["gpu"])
    box(ax, (0.56, 0.005), (0.14, 0.095), "Retry: 2,048\nanchors", "#FBE8DD", COLORS["accent"], fontsize=7.7)
    arrow(ax, (0.63, gpu_y), (0.63, 0.10), COLORS["accent"])
    arrow(ax, (0.70, 0.052), (0.83, gpu_y), COLORS["accent"], connectionstyle="arc3,rad=-0.18")
    arrow(ax, (0.90, gpu_y + 0.07), (0.965, gpu_y + 0.07), COLORS["green"])
    ax.text(0.97, gpu_y + 0.07, "dFBA", va="center", ha="left", fontsize=8, weight="bold", transform=ax.transAxes)
    ax.text(
        0.76,
        0.39,
        r"$v=V^{T}w$;  $w\geq0$;  $\mathbf{1}^{T}w=1$" + "\n" + r"$Sv=0$;  $l\leq v\leq u$;  $A_{shared}v\leq b$",
        ha="center",
        va="center",
        fontsize=8.2,
        transform=ax.transAxes,
    )
    ax.text(0.18, 0.115, "one CUDA context", ha="left", fontsize=7.4, color=COLORS["muted"], transform=ax.transAxes)
    save(fig, output, "Figure_1_CPU_GPU_architecture")


def performance_figure(validation: dict, output: Path) -> None:
    rows = validation["runs"]
    seed = np.arange(1, len(rows) + 1)
    cpu = np.array([row["exact_seconds"] for row in rows])
    gpu = np.array([row["surrogate_seconds"] for row in rows])
    speedup = np.array([row["speedup"] for row in rows])
    pha = 100 * np.array([row["pha_relative_error"] for row in rows])
    summary = validation["summary"]

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1), hspace=0.45, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    width = 0.34
    ax_a.bar(seed - width / 2, cpu, width, color=COLORS["cpu"], label="CPU HiGHS", zorder=3)
    ax_a.bar(seed + width / 2, gpu, width, color=COLORS["gpu"], label="GPU surrogate + QP", zorder=3)
    ax_a.set_ylabel("Runtime per 120-step trajectory (s)")
    ax_a.set_xticks(seed, [f"Seed {i}" for i in seed])
    ax_a.set_ylim(0, max(cpu) * 1.18)
    ax_a.legend(frameon=False, ncol=2, loc="upper left")
    ax_a.text(-0.09, 1.04, "a", transform=ax_a.transAxes, fontsize=11, weight="bold")
    clean_axis(ax_a)

    ax_b.scatter(seed, speedup, s=28, color=COLORS["gpu"], edgecolor="white", linewidth=0.5, zorder=4)
    mean = float(summary["speedup_mean"])
    lo, hi = map(float, summary["speedup_95_ci"])
    ax_b.axhline(mean, color=COLORS["accent"], lw=1.2)
    ax_b.fill_between([0.6, 5.4], lo, hi, color=COLORS["accent"], alpha=0.16, linewidth=0)
    ax_b.text(0.68, mean + 0.018, f"Mean {mean:.2f}×", color=COLORS["accent"], fontsize=8)
    ax_b.set_xlim(0.6, 5.4)
    ax_b.set_xticks(seed)
    ax_b.set_ylabel("Speed-up (CPU/GPU)")
    ax_b.set_xlabel("Seed")
    ax_b.text(-0.19, 1.04, "b", transform=ax_b.transAxes, fontsize=11, weight="bold")
    clean_axis(ax_b)

    ax_c.bar(seed, pha, color=COLORS["green"], width=0.64, zorder=3)
    ax_c.axhline(1.0, color=COLORS["accent"], lw=1.0, ls="--")
    ax_c.set_xlim(0.4, 5.6)
    ax_c.set_xticks(seed)
    ax_c.set_ylabel("Terminal PHA relative error (%)")
    ax_c.set_xlabel("Seed")
    ax_c.text(
        5.35,
        1.04,
        "Predefined 1% gate",
        ha="right",
        va="bottom",
        color=COLORS["accent"],
        fontsize=7.8,
    )
    ax_c.text(-0.19, 1.04, "c", transform=ax_c.transAxes, fontsize=11, weight="bold")
    clean_axis(ax_c)
    save(fig, output, "Figure_3_performance_and_accuracy")


def scaling_figure(scaling: dict, output: Path) -> None:
    rows = scaling["rows"]
    batch = np.array([row["batch_size"] for row in rows])
    throughput = np.array([row["environments_per_second"] for row in rows])
    latency = np.array([row["per_environment_milliseconds"] for row in rows])
    vram = np.array([row["gpu_peak_allocated_mib"] for row in rows]) / 1024
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch, [str(value) for value in batch])
        ax.set_xlabel("Batch size")
        clean_axis(ax)

    axes[0].plot(batch, throughput, marker="o", color=COLORS["gpu"], lw=1.5, ms=4.5, zorder=3)
    axes[0].set_ylabel("Throughput (env s$^{-1}$)")
    axes[0].text(0.03, 0.93, f"{throughput[-1] / throughput[0]:.2f}×", transform=axes[0].transAxes, color=COLORS["gpu"], weight="bold")
    axes[1].plot(batch, latency, marker="o", color=COLORS["green"], lw=1.5, ms=4.5, zorder=3)
    axes[1].set_ylabel("Per-environment latency (ms)")
    axes[1].text(0.23, 0.93, f"{100 * (1 - latency[-1] / latency[0]):.1f}% lower", transform=axes[1].transAxes, color=COLORS["green"], weight="bold")
    axes[2].plot(batch, vram, marker="o", color=COLORS["accent"], lw=1.5, ms=4.5, zorder=3)
    axes[2].set_ylabel("Peak allocated VRAM (GiB)")
    for label, ax in zip("abc", axes):
        ax.text(-0.20, 1.06, label, transform=ax.transAxes, fontsize=11, weight="bold")
    save(fig, output, "Figure_4_GPU_batch_scaling")


def service_figure(smoke: dict, output: Path) -> None:
    service = smoke["service"]
    histogram = {int(k): int(v) for k, v in service["batch_size_histogram"].items()}
    batch_sizes = np.array(sorted(histogram))
    batches = np.array([histogram[k] for k in batch_sizes])
    requests = int(service["requests"])
    retry = int(service["retry_requests"])
    primary = requests - retry
    recovered = int(service["retry_recovered"])

    fig = plt.figure(figsize=(7.2, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.35), wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_a.bar(batch_sizes, batches, color=COLORS["gpu"], width=0.68, zorder=3)
    ax_a.set_xlabel("Observed microbatch size")
    ax_a.set_ylabel("Number of batches")
    ax_a.set_xticks(batch_sizes)
    ax_a.text(-0.16, 1.04, "a", transform=ax_a.transAxes, fontsize=11, weight="bold")
    clean_axis(ax_a)

    ax_b.set_axis_off()
    box(ax_b, (0.02, 0.40), (0.20, 0.20), f"{requests}\nrequests", COLORS["light_blue"], COLORS["gpu"], 9, "bold")
    box(ax_b, (0.39, 0.60), (0.24, 0.18), f"{primary}\nprimary feasible", COLORS["light_green"], COLORS["green"], 8.2, "bold")
    box(ax_b, (0.39, 0.18), (0.24, 0.18), f"{retry}\nGPU retries", "#FBE8DD", COLORS["accent"], 8.2, "bold")
    box(ax_b, (0.76, 0.40), (0.20, 0.20), f"{primary + recovered}\nvalid fluxes", COLORS["light_green"], COLORS["green"], 9, "bold")
    arrow(ax_b, (0.22, 0.52), (0.39, 0.69), COLORS["green"])
    arrow(ax_b, (0.22, 0.48), (0.39, 0.27), COLORS["accent"])
    arrow(ax_b, (0.63, 0.69), (0.76, 0.54), COLORS["green"])
    arrow(ax_b, (0.63, 0.27), (0.76, 0.46), COLORS["accent"])
    ax_b.text(0.50, 0.06, "CPU LP calls = 0", ha="center", fontsize=8.5, weight="bold", color=COLORS["cpu"], transform=ax_b.transAxes)
    ax_b.text(-0.08, 1.04, "b", transform=ax_b.transAxes, fontsize=11, weight="bold")
    save(fig, output, "Figure_5_parallel_service_robustness")


def dagger_workflow_figure(output: Path) -> None:
    """Show why teacher-forced anchors drift and how rollout relabeling fixes it."""
    fig = plt.figure(figsize=(7.2, 4.55))
    gs = fig.add_gridspec(2, 1, height_ratios=(0.8, 1.35), hspace=0.27)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    for ax in (ax_a, ax_b):
        ax.set_axis_off()

    ax_a.text(0.00, 0.93, "a", fontsize=11, weight="bold", transform=ax_a.transAxes)
    box(ax_a, (0.05, 0.31), (0.22, 0.34), "CPU exact trajectories\n(teacher distribution)", COLORS["light_gray"], COLORS["cpu"], 8.2, "bold")
    box(ax_a, (0.39, 0.31), (0.22, 0.34), "GPU free rollout\n(visited distribution)", COLORS["light_blue"], COLORS["gpu"], 8.2, "bold")
    box(ax_a, (0.73, 0.31), (0.22, 0.34), "Accumulated endpoint\nPHA error", "#FBE8DD", COLORS["accent"], 8.2, "bold")
    arrow(ax_a, (0.27, 0.48), (0.39, 0.48), COLORS["muted"])
    arrow(ax_a, (0.61, 0.48), (0.73, 0.48), COLORS["accent"])
    ax_a.text(0.50, 0.08, "one-step feasibility does not prevent rollout distribution shift", ha="center", fontsize=7.8, color=COLORS["muted"], transform=ax_a.transAxes)

    ax_b.text(0.00, 0.96, "b", fontsize=11, weight="bold", transform=ax_b.transAxes)
    labels = [
        (0.02, "GPU-only\ndFBA rollout", COLORS["light_blue"], COLORS["gpu"]),
        (0.205, "Capture live\n97-feature\ncontext", COLORS["light_blue"], COLORS["gpu"]),
        (0.395, "Offline exact\n3-stage HiGHS\nlabel", COLORS["light_gray"], COLORS["cpu"]),
        (0.595, "Append\nexact-feasible\nflux anchors", COLORS["light_green"], COLORS["green"]),
        (0.795, "Rebase model\n+ held-out\nvalidation", COLORS["light_green"], COLORS["green"]),
    ]
    widths = [0.145, 0.155, 0.16, 0.16, 0.18]
    for (x, label, face, edge), width in zip(labels, widths):
        box(ax_b, (x, 0.43), (width, 0.29), label, face, edge, 6.8, "bold")
    for idx in range(len(labels) - 1):
        arrow(ax_b, (labels[idx][0] + widths[idx], 0.565), (labels[idx + 1][0], 0.565), COLORS["muted"])
    ax_b.text(0.485, 0.25, "Iteration 1: +360 anchors", ha="center", fontsize=8.0, color=COLORS["green"], weight="bold", transform=ax_b.transAxes)
    ax_b.text(0.485, 0.10, "Iteration 2: +360 anchors   |   validation seeds excluded", ha="center", fontsize=8.0, color=COLORS["green"], weight="bold", transform=ax_b.transAxes)
    arrow(ax_b, (0.88, 0.43), (0.88, 0.31), COLORS["green"])
    arrow(ax_b, (0.88, 0.31), (0.105, 0.31), COLORS["green"], connectionstyle="arc3,rad=0.13")
    save(fig, output, "Figure_2_DAgger_rollout_relabeling")


def endpoint_accuracy_figure(baseline: dict, validation: dict, output: Path) -> None:
    old_rows = baseline["runs"]
    rows = validation["runs"]
    seed = np.arange(1, len(rows) + 1)
    old_error = 100 * np.array([row["pha_relative_error"] for row in old_rows])
    new_error = 100 * np.array([row["pha_relative_error"] for row in rows])
    exact_pha = np.array([row["exact_pha_g_l"] for row in rows])
    gpu_pha = np.array([row["gpu_pha_g_l"] for row in rows])
    abs_error = 1000 * np.array([row["pha_absolute_error_g_l"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    for x, before, after in zip(seed, old_error, new_error):
        ax_a.plot([0, 1], [before, after], color="#AAB2BB", lw=1.0, zorder=1)
        ax_a.scatter([0], [before], color=COLORS["cpu"], s=28, zorder=3)
        ax_a.scatter([1], [after], color=COLORS["green"], s=28, zorder=3)
        ax_a.text(1.045, after, str(x), va="center", fontsize=6.8, color=COLORS["muted"])
    ax_a.axhline(1.0, color=COLORS["accent"], ls="--", lw=1.0)
    ax_a.set_xticks([0, 1], ["Baseline", "DAgger-2"])
    ax_a.set_ylabel("Terminal PHA relative error (%)")
    ax_a.set_xlim(-0.25, 1.28)
    clean_axis(ax_a)

    lo = min(exact_pha.min(), gpu_pha.min()) - 0.0004
    hi = max(exact_pha.max(), gpu_pha.max()) + 0.0004
    ax_b.plot([lo, hi], [lo, hi], color=COLORS["muted"], ls="--", lw=1.0)
    ax_b.scatter(exact_pha, gpu_pha, color=COLORS["gpu"], edgecolor="white", linewidth=0.5, s=35, zorder=3)
    offsets = [(3, 3), (3, 3), (4, 7), (8, 1), (3, 3)]
    for index, (x, y, offset) in enumerate(zip(exact_pha, gpu_pha, offsets), 1):
        ax_b.annotate(str(index), (x, y), xytext=offset, textcoords="offset points", fontsize=6.8)
    ax_b.set_xlabel("CPU exact terminal PHA (g L$^{-1}$)")
    ax_b.set_ylabel("GPU terminal PHA (g L$^{-1}$)")
    ax_b.set_xlim(lo, hi)
    ax_b.set_ylim(lo, hi)
    clean_axis(ax_b)

    ax_c.bar(seed, abs_error, color=COLORS["gpu"], width=0.62, zorder=3)
    ax_c.set_xlabel("Held-out seed")
    ax_c.set_ylabel("Absolute PHA error (mg L$^{-1}$)")
    ax_c.set_xticks(seed)
    clean_axis(ax_c)

    metric = ["Mean", "Maximum"]
    old_summary = [100 * baseline["summary"]["pha_relative_error_mean"], 100 * baseline["summary"]["pha_relative_error_max"]]
    new_summary = [100 * validation["summary"]["pha_relative_error_mean"], 100 * validation["summary"]["pha_relative_error_max"]]
    x = np.arange(2)
    width = 0.34
    ax_d.bar(x - width / 2, old_summary, width, color=COLORS["cpu"], label="Baseline", zorder=3)
    ax_d.bar(x + width / 2, new_summary, width, color=COLORS["green"], label="DAgger-2", zorder=3)
    ax_d.axhline(1.0, color=COLORS["accent"], ls="--", lw=1.0)
    ax_d.set_xticks(x, metric)
    ax_d.set_ylabel("Terminal PHA relative error (%)")
    ax_d.legend(frameon=False, ncol=2, loc="upper right")
    clean_axis(ax_d)

    for label, ax in zip("abcd", axes.ravel()):
        ax.text(-0.18, 1.04, label, transform=ax.transAxes, fontsize=11, weight="bold")
    fig.subplots_adjust(hspace=0.43, wspace=0.38)
    save(fig, output, "Figure_6_PHA_endpoint_accuracy")


def dagger_progress_figure(baseline: dict, dagger1: dict, validation: dict, output: Path) -> None:
    datasets = [baseline, dagger1, validation]
    labels = ["Baseline", "DAgger-1", "DAgger-2"]
    candidate_count = np.array([33728, 34088, 34448])
    mean_error = 100 * np.array([item["summary"]["pha_relative_error_mean"] for item in datasets])
    max_error = 100 * np.array([item["summary"]["pha_relative_error_max"] for item in datasets])
    speedup = np.array([item["summary"]["speedup_mean"] for item in datasets])
    x = np.arange(3)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax_a, ax_b, ax_c = axes
    ax_a.plot(x, mean_error, marker="o", color=COLORS["gpu"], lw=1.5, label="Mean", zorder=3)
    ax_a.plot(x, max_error, marker="s", color=COLORS["accent"], lw=1.5, label="Maximum", zorder=3)
    ax_a.axhline(1.0, color=COLORS["accent"], ls="--", lw=0.9, alpha=0.8)
    ax_a.set_xticks(x, labels, rotation=18, ha="right")
    ax_a.set_ylabel("PHA relative error (%)")
    ax_a.legend(frameon=False)
    clean_axis(ax_a)

    added = candidate_count - candidate_count[0]
    ax_b.bar(x, added, color=[COLORS["cpu"], COLORS["gpu"], COLORS["green"]], width=0.62, zorder=3)
    for xi, total, inc in zip(x, candidate_count, added):
        ax_b.text(xi, inc + 28, f"{total:,}", ha="center", fontsize=7.2)
    ax_b.set_xticks(x, labels, rotation=18, ha="right")
    ax_b.set_ylabel("Exact anchors added")
    ax_b.set_ylim(0, 860)
    clean_axis(ax_b)

    ax_c.plot(x, speedup, marker="o", color=COLORS["green"], lw=1.5, ms=5, zorder=3)
    ax_c.axhline(1.0, color=COLORS["muted"], ls="--", lw=0.9)
    for xi, value in zip(x, speedup):
        ax_c.text(xi, value + 0.06, f"{value:.2f}×", ha="center", fontsize=7.2)
    ax_c.set_xticks(x, labels, rotation=18, ha="right")
    ax_c.set_ylabel("Mean speed-up")
    ax_c.set_ylim(0, max(speedup) * 1.2)
    clean_axis(ax_c)
    for label, ax in zip("abc", axes):
        ax.text(-0.20, 1.06, label, transform=ax.transAxes, fontsize=11, weight="bold")
    save(fig, output, "Figure_7_DAgger_iteration_progress")


def bottleneck_figure(profile: dict, output: Path) -> None:
    exact = profile["exact"]["solver"]["mean_step_timing"]
    gpu = profile["surrogate"]["solver"]["mean_step_timing"]
    coop = profile["surrogate"]["solver"]["cooperative"]
    qp_per_step = float(coop["gpu_qp_seconds"]) / float(profile["steps"])
    dictionary = float(coop["build_seconds"])
    other_solve = max(0.0, float(gpu["solve_seconds"]) - dictionary - qp_per_step)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.35]})
    ax_a, ax_b = axes
    categories = ["State preparation", "Optimization", "State update"]
    cpu_values = np.array([exact["pre_solve_seconds"], exact["solve_seconds"], exact["post_solve_seconds"]])
    gpu_values = np.array([gpu["pre_solve_seconds"], gpu["solve_seconds"], gpu["post_solve_seconds"]])
    x = np.arange(2)
    bottom = np.zeros(2)
    colors = [COLORS["light_blue"], COLORS["gpu"], COLORS["green"]]
    for idx, (category, color) in enumerate(zip(categories, colors)):
        vals = np.array([cpu_values[idx], gpu_values[idx]])
        ax_a.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, label=category, zorder=3)
        bottom += vals
    ax_a.set_xticks(x, ["CPU HiGHS", "GPU surrogate"])
    ax_a.set_ylabel("Mean time per dFBA step (s)")
    ax_a.legend(frameon=False, fontsize=7.0, loc="upper right")
    clean_axis(ax_a)

    parts = np.array([gpu["pre_solve_seconds"], dictionary, qp_per_step, other_solve, gpu["post_solve_seconds"]], dtype=float)
    labels = ["State preparation", "Dictionary search / materialization", "QP projection", "Other solve overhead", "State update"]
    colors = [COLORS["light_blue"], COLORS["gpu"], COLORS["green"], "#B8C2CC", COLORS["accent"]]
    left = 0.0
    for value, label, color in zip(parts, labels, colors):
        ax_b.barh([0], [value], left=left, color=color, edgecolor="white", linewidth=0.5, label=label)
        if value / parts.sum() >= 0.08:
            ax_b.text(left + value / 2, 0, f"{100*value/parts.sum():.1f}%", ha="center", va="center", fontsize=7.0, color="white" if color == COLORS["gpu"] else COLORS["ink"], weight="bold")
        left += value
    ax_b.set_xlabel("Mean time per GPU dFBA step (s)")
    ax_b.set_yticks([])
    ax_b.legend(frameon=False, fontsize=6.9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    clean_axis(ax_b, grid=False)
    ax_b.text(0.5, 1.04, f"QP = {100*qp_per_step/parts.sum():.1f}% of GPU step; search/materialization is dominant", ha="center", transform=ax_b.transAxes, fontsize=7.7, color=COLORS["muted"])
    for label, ax in zip("ab", axes):
        ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=11, weight="bold")
    fig.subplots_adjust(bottom=0.27, wspace=0.35)
    save(fig, output, "Figure_8_runtime_bottleneck_profile")


def calibration_figure(calibration: dict, output: Path) -> None:
    pools = [64, 128, 256, 512, 1024]
    strengths = [1, 2, 4, 8, 16]
    lookup = {(int(row["pool"]), float(row["strength"])): row for row in calibration["all"]}
    mean = np.full((len(strengths), len(pools)), np.nan)
    maximum = np.full_like(mean, np.nan)
    for yi, strength in enumerate(strengths):
        for xi, pool in enumerate(pools):
            row = lookup.get((pool, float(strength)))
            if row:
                mean[yi, xi] = 100 * float(row["mean"])
                maximum[yi, xi] = 100 * float(row["max"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for label, ax, matrix, title in zip("ab", axes, [mean, maximum], ["Mean teacher-forced PHA error", "Maximum teacher-forced PHA error"]):
        image = ax.imshow(matrix, cmap="viridis_r", vmin=0.0, vmax=max(1.05, float(np.nanmax(maximum))), aspect="auto")
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                value = matrix[yi, xi]
                if np.isfinite(value):
                    ax.text(xi, yi, f"{value:.2f}", ha="center", va="center", fontsize=6.8, color="white" if value > 0.72 else COLORS["ink"])
        ax.set_xticks(np.arange(len(pools)), [str(value) for value in pools])
        ax.set_yticks(np.arange(len(strengths)), [str(value) for value in strengths])
        ax.set_xlabel("Rerank pool")
        ax.set_ylabel("Decision strength")
        ax.set_title(title, pad=7)
        ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=11, weight="bold")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative error (%)")
    save(fig, output, "Figure_9_hyperparameter_calibration")


def qualification_figure(validation: dict, smoke: dict, output: Path) -> None:
    summary = validation["summary"]
    criteria = validation["criteria"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"width_ratios": [1.2, 1]})
    ax_a, ax_b = axes
    names = ["PHA endpoint", "Biomass endpoint"]
    measured = np.array([
        summary["pha_relative_error_max"] / criteria["maximum_pha_relative_error"],
        summary["max_biomass_absolute_error_g_l"] / criteria["maximum_biomass_absolute_error_g_l"],
    ])
    y = np.arange(2)
    ax_a.barh(y, measured, color=[COLORS["green"], COLORS["gpu"]], height=0.55, zorder=3)
    ax_a.axvline(1.0, color=COLORS["accent"], ls="--", lw=1.0)
    ax_a.set_yticks(y, names)
    ax_a.set_xlabel("Measured / acceptance limit")
    ax_a.set_xlim(0, 1.12)
    ax_a.invert_yaxis()
    for yi, value in zip(y, measured):
        ax_a.text(value + 0.025, yi, f"{value:.2f}", va="center", fontsize=7.5)
    clean_axis(ax_a)

    service = smoke["service"]
    labels = ["Validation accepted", "Smoke-test accepted", "Retry recovered", "CPU LP fallback"]
    values = [validation["n"] * validation["steps"], service["requests"], service["retry_recovered"], 0]
    colors = [COLORS["gpu"], COLORS["green"], COLORS["accent"], COLORS["cpu"]]
    y = np.arange(4)
    plotted = np.maximum(values, 0.12)
    ax_b.barh(y, plotted, color=colors, height=0.56, zorder=3)
    ax_b.set_yticks(y, labels)
    ax_b.set_xlabel("Number of requests")
    ax_b.set_xscale("symlog", linthresh=1)
    ax_b.invert_yaxis()
    for yi, value, display in zip(y, plotted, values):
        ax_b.text(value * 1.18, yi, str(display), va="center", fontsize=7.5)
    clean_axis(ax_b)
    for label, ax in zip("ab", axes):
        ax.text(-0.17, 1.05, label, transform=ax.transAxes, fontsize=11, weight="bold")
    fig.subplots_adjust(wspace=0.52)
    save(fig, output, "Figure_10_qualification_and_safety")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/cpu_gpu_paper_20260902/figures",
    )
    args = parser.parse_args()
    configure()
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_VALIDATION.read_text(encoding="utf-8"))
    dagger1 = json.loads(DAGGER1_VALIDATION.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    architecture_figure(args.output_dir)
    performance_figure(validation, args.output_dir)
    scaling_figure(scaling, args.output_dir)
    service_figure(smoke, args.output_dir)
    dagger_workflow_figure(args.output_dir)
    endpoint_accuracy_figure(baseline, validation, args.output_dir)
    dagger_progress_figure(baseline, dagger1, validation, args.output_dir)
    bottleneck_figure(profile, args.output_dir)
    calibration_figure(calibration, args.output_dir)
    qualification_figure(validation, smoke, args.output_dir)
    print(f"saved figures: {args.output_dir}")


if __name__ == "__main__":
    main()
