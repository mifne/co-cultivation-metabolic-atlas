#!/usr/bin/env python3
"""Plot held-out PHA accuracy and performance before/after DAgger relabeling."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "models/cooperative_surrogate/cooperative_neural_reranker_33728_uniform_late_trained_pha256.validation.json"
DAGGER1 = ROOT / "results/pha_validation_dagger_rebased_5seed.json"
DAGGER2 = ROOT / "models/cooperative_surrogate/cooperative_neural_reranker_34448_dagger2_rebased.validation.json"
OUT = ROOT / "outputs/pha_accuracy_improvement_20260902"


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    one = json.loads(DAGGER1.read_text(encoding="utf-8"))
    two = json.loads(DAGGER2.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    old_error = 100 * np.asarray([row["pha_relative_error"] for row in old["runs"]])
    one_error = 100 * np.asarray([row["pha_relative_error"] for row in one["runs"]])
    two_error = 100 * np.asarray([row["pha_relative_error"] for row in two["runs"]])
    old_speed = np.asarray([row["speedup"] for row in old["runs"]])
    two_speed = np.asarray([row["speedup"] for row in two["runs"]])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue, green, orange, dark = "#0072B2", "#009E73", "#D55E00", "#30343B"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)

    x = np.arange(1, 6)
    axes[0].plot(x, old_error, "o-", color=orange, label="Before")
    axes[0].plot(x, one_error, "s-", color=blue, label="DAgger-1")
    axes[0].plot(x, two_error, "^-", color=green, label="DAgger-2")
    axes[0].axhline(1.0, color=dark, lw=0.9, ls="--")
    axes[0].set(xticks=x, xlabel="Held-out seed", ylabel="Terminal PHA relative error (%)")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=1)
    axes[0].set_title("a", loc="left", fontweight="bold")

    iterations = np.arange(3)
    mean = [old_error.mean(), one_error.mean(), two_error.mean()]
    maximum = [old_error.max(), one_error.max(), two_error.max()]
    axes[1].plot(iterations, mean, "o-", color=blue, label="Mean")
    axes[1].plot(iterations, maximum, "s-", color=orange, label="Maximum")
    axes[1].axhline(1.0, color=dark, lw=0.9, ls="--")
    axes[1].set(
        xticks=iterations,
        xticklabels=["Before", "1", "2"],
        xlabel="DAgger iteration",
        ylabel="Terminal PHA relative error (%)",
    )
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].set_title("b", loc="left", fontweight="bold")

    width = 0.36
    axes[2].bar(x - width / 2, old_speed, width, color=orange, label="Before")
    axes[2].bar(x + width / 2, two_speed, width, color=green, label="DAgger-2")
    axes[2].axhline(1.0, color=dark, lw=0.9, ls="--")
    axes[2].set(xticks=x, xlabel="Held-out seed", ylabel="Speed-up (CPU/GPU)")
    axes[2].set_ylim(0.0, 3.35)
    axes[2].legend(frameon=False, fontsize=7.5)
    axes[2].set_title("c", loc="left", fontweight="bold")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DEE5", lw=0.6)
        axis.set_axisbelow(True)

    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"Figure_PHA_GPU_DAgger_accuracy.{suffix}",
            dpi=500,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
