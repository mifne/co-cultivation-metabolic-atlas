#!/usr/bin/env python3
"""Validate the selected live-helper hypothesis across N and oxygen transfer."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.screen_nonproducer_helper import ROOT, Scenario, run_scenario


OUTPUT = ROOT / "results/nonproducer_helper_sensitivity_20260902"
HOURS = 12.0
DT = 1.0
RATE = 0.10
CONDITIONS = [
    (0.05, 20.0),
    (0.05, 50.0),
    (0.05, 100.0),
    (0.50, 50.0),
    (2.00, 50.0),
]


def scenarios(kla: float) -> list[Scenario]:
    return [
        Scenario("glucose_direct", False, "glc__D_e", RATE, kla_per_h=kla),
        Scenario("dextrin_direct", False, "dextrin_e", RATE, kla_per_h=kla),
        Scenario("bsub_starch", True, "starch_e", RATE, 0.01, kla),
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for nh4, kla in CONDITIONS:
        for scenario in scenarios(kla):
            print(f"NH4={nh4:.2f}; kLa={kla:.0f}; {scenario.name}", flush=True)
            summary, _ = run_scenario(scenario, HOURS, DT, nh4)
            rows.append({"condition_nh4_mmol_l": nh4, **summary})

    fields = list(rows[0])
    with (OUTPUT / "sensitivity_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    condition_results = []
    for nh4, kla in CONDITIONS:
        subset = [
            row
            for row in rows
            if row["condition_nh4_mmol_l"] == nh4 and row["kla_per_h"] == kla
        ]
        helper = next(row for row in subset if row["name"] == "bsub_starch")
        direct = max(
            (row for row in subset if row["name"] != "bsub_starch"),
            key=lambda row: row["final_pha_g_l_as_phb"],
        )
        condition_results.append(
            {
                "initial_nh4_mmol_l": nh4,
                "kla_per_h": kla,
                "helper_pha_g_l_as_phb": helper["final_pha_g_l_as_phb"],
                "best_direct_name": direct["name"],
                "best_direct_pha_g_l_as_phb": direct["final_pha_g_l_as_phb"],
                "helper_to_direct_ratio": (
                    helper["final_pha_g_l_as_phb"]
                    / direct["final_pha_g_l_as_phb"]
                    if direct["final_pha_g_l_as_phb"] > 0
                    else None
                ),
                "helper_solver_success_rate": helper["solver_success_rate"],
                "helper_starch_uptake_mmol_l": helper[
                    "helper_starch_uptake_mmol_l"
                ],
                "helper_dextrin_secretion_mmol_l": helper[
                    "helper_dextrin_secretion_mmol_l"
                ],
                "helper_o2_uptake_mmol_l": helper["helper_o2_uptake_mmol_l"],
                "helper_nh4_uptake_mmol_l": helper["helper_nh4_uptake_mmol_l"],
            }
        )

    with (OUTPUT / "condition_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_results[0]))
        writer.writeheader()
        writer.writerows(condition_results)

    _plot(rows, condition_results)
    ratios = [
        row["helper_to_direct_ratio"]
        for row in condition_results
        if row["helper_to_direct_ratio"] is not None
    ]
    accepted = [row for row in condition_results if row["helper_to_direct_ratio"] > 1.05]
    payload = {
        "conditions": condition_results,
        "hours": HOURS,
        "dt_h": DT,
        "feed_rate_mmol_l_h": RATE,
        "helper_initial_biomass_g_l": 0.01,
        "helper_to_direct_ratio_range": [min(ratios), max(ratios)],
        "conditions_meeting_5pct_rule": len(accepted),
        "verdict": (
            "conditional_go" if len(accepted) == len(condition_results)
            else "no_go_as_live_third_strain"
        ),
    }
    (OUTPUT / "sensitivity_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# 第三菌候補の窒素・酸素移動感度",
        "",
        f"- 12 h、流加 {RATE:.2f} mmol/L/h、第三菌初期量 0.01 g/L。",
        f"- 第三菌/最良直接流加のPHA比: {min(ratios):.3f}–{max(ratios):.3f}。",
        f"- 直接流加を5%以上上回った条件: {len(accepted)}/{len(condition_results)}。",
        f"- 判定: `{payload['verdict']}`。",
        "- この感度解析は初期NH4量とkLaの軸上評価であり、実測速度論による校正ではない。",
    ]
    (OUTPUT / "sensitivity_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Output: {OUTPUT}")


def _plot(rows: list[dict], comparisons: list[dict]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "glucose_direct": "#E69F00",
        "dextrin_direct": "#6A51A3",
        "bsub_starch": "#009E73",
    }
    labels = {
        "glucose_direct": "Glucose direct",
        "dextrin_direct": "Dextrin direct",
        "bsub_starch": "B. subtilis + starch",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    for name in colors:
        nitrogen = sorted(
            (
                row
                for row in rows
                if row["name"] == name and row["kla_per_h"] == 50.0
            ),
            key=lambda row: row["condition_nh4_mmol_l"],
        )
        axes[0].plot(
            [row["condition_nh4_mmol_l"] for row in nitrogen],
            [row["final_pha_g_l_as_phb"] for row in nitrogen],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
        oxygen = sorted(
            (
                row
                for row in rows
                if row["name"] == name
                and row["condition_nh4_mmol_l"] == 0.05
            ),
            key=lambda row: row["kla_per_h"],
        )
        axes[1].plot(
            [row["kla_per_h"] for row in oxygen],
            [row["final_pha_g_l_as_phb"] for row in oxygen],
            marker="o",
            color=colors[name],
        )
    axes[0].set_xlabel(r"Initial NH$_4^+$ (mmol L$^{-1}$)")
    axes[0].set_ylabel(r"PHA model output as PHB (g L$^{-1}$)")
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].set_xlabel(r"$k_La$ (h$^{-1}$)")
    axes[1].set_ylabel(r"PHA model output as PHB (g L$^{-1}$)")

    labels_x = [
        f"N{row['initial_nh4_mmol_l']:g}/k{row['kla_per_h']:g}"
        for row in comparisons
    ]
    axes[2].bar(
        range(len(comparisons)),
        [row["helper_to_direct_ratio"] for row in comparisons],
        color="#009E73",
    )
    axes[2].axhline(1.0, color="#666666", lw=0.9)
    axes[2].axhline(1.05, color="#666666", lw=0.9, ls="--")
    axes[2].set_xticks(range(len(labels_x)), labels_x, rotation=35, ha="right")
    axes[2].set_ylabel("Helper / best direct-feed PHA")

    for label, axis in zip("abc", axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.17, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    base = OUTPUT / "Figure_nonproducer_helper_sensitivity"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
