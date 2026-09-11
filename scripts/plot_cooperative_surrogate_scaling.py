#!/usr/bin/env python3
"""Publication-style summary of measured cooperative dictionary scaling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def row(candidate_count: int, path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    exact_pha = float(report["exact"]["pha_g_l"])
    return {
        "candidate_count": candidate_count,
        "gpu_valid_fraction": float(
            report["surrogate"]["solver"]["cooperative"][
                "surrogate_acceptance_rate"
            ]
        ),
        "speedup": float(report["speedup"]),
        "pha_relative_error_percent": (
            100.0 * float(report["pha_abs_difference_g_l"]) / exact_pha
            if exact_pha
            else np.nan
        ),
        "exact_seconds": float(report["exact"]["elapsed_seconds"]),
        "gpu_hybrid_seconds": float(report["surrogate"]["elapsed_seconds"]),
        "rubber_absolute_error_g_l": float(report["rubber_abs_difference_g_l"]),
        "biomass_max_absolute_error_g_l": float(
            report["max_biomass_abs_difference_g_l"]
        ),
    }


def main() -> None:
    rows = [
        row(128, PROJECT_ROOT / "results" / "cooperative_surrogate_e2e_120.json"),
        row(
            1024,
            PROJECT_ROOT / "results" / "cooperative_surrogate_e2e_1024_120.json",
        ),
    ]
    output_dir = PROJECT_ROOT / "outputs" / "cooperative_surrogate_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "measured_scaling.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    count = np.asarray([item["candidate_count"] for item in rows])
    color = "#2468A2"
    accent = "#C84B31"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)
    values = [
        100 * np.asarray([item["gpu_valid_fraction"] for item in rows]),
        np.asarray([item["speedup"] for item in rows]),
        np.asarray([item["pha_relative_error_percent"] for item in rows]),
    ]
    ylabels = ["GPU-valid solutions (%)", "End-to-end speedup (×)", "24 h PHA error (%)"]
    labels = ["a", "b", "c"]
    for axis, y, ylabel, label in zip(axes, values, ylabels, labels):
        axis.plot(count, y, marker="o", color=color, linewidth=1.5, markersize=4.5)
        axis.set_xscale("log", base=2)
        axis.set_xticks(count, [f"{value:,}" for value in count])
        axis.set_xlabel("Aligned exact candidates")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.18, 1.04, label, transform=axis.transAxes, fontweight="bold")
    axes[0].axhline(95, color=accent, linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 105)
    axes[1].axhline(1, color="#666666", linestyle=":", linewidth=1)
    axes[1].set_ylim(0.8, max(values[1]) * 1.12)
    axes[2].axhline(1, color=accent, linestyle="--", linewidth=1)
    axes[2].set_ylim(0, max(2.6, max(values[2]) * 1.12))
    for axis, y in zip(axes, values):
        for x_value, y_value in zip(count, y):
            axis.annotate(
                f"{y_value:.2f}",
                (x_value, y_value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"cooperative_surrogate_scaling.{extension}",
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
