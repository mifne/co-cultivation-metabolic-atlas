#!/usr/bin/env python3
"""Audit three-member GEM fluxes, polymer chemistry, and nutrient rescues.

This program is deliberately read-only with respect to the production SBML
files.  Two model views are compared:

* ``raw`` keeps every published/reconstructed reaction bound; and
* ``zero_feasible`` expands exchange bounds just enough to allow zero flux.

The second view prevents experimental flux ranges embedded in an SBML model
from being mistaken for compulsory culture-medium requirements.  It is an
audit view, not a replacement model.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_rl_problem_feasibility import (  # noqa: E402
    YEAST_EXTRACT_COMPOSITION,
)
from src.coexistence_audit import (  # noqa: E402
    SharedMediumCommunityLP,
    canonical_metabolite_id,
    find_growth_reaction,
)
from src.utils import (  # noqa: E402
    COEXISTENCE_DEFINED_FEED_MMOL_L_H,
    get_initial_params,
    load_sbml_models,
    select_consortium_models,
)


POLYMER_REACTION_IDS = (
    "R_LCP",
    "R_ROXB",
    "R_ROXA",
    "R_ROXA_BULK",
    "R_C30t",
    "R_C30_cat",
    "R_ODTDt",
    "R_ODTD_cat",
    "PHB_PhaB",
    "PHV_PhaB",
    "PHB_syn",
    "PHV_syn",
    "PHB_PhaZ",
    "PHV_PhaZ",
    "EX_pha_c",
    "EX_phv_c",
    "R_GLYCOLIPOPROTEIN_SYN_SEC",
)

EXPECTED_GPRS = {
    "R_LCP": {"ACTI_59630", "ACTI_59640", "ACTI_69520"},
    "R_ROXB": {"A4W93_01825"},
    "R_ROXA": {"A4W93_07150"},
    "R_ROXA_BULK": {"A4W93_07150"},
    "PHB_PhaB": {"A4W93_10495"},
    "PHV_PhaB": {"A4W93_10495"},
    "PHB_syn": {"A4W93_10485"},
    "PHV_syn": {"A4W93_10485"},
    "PHB_PhaZ": {"A4W93_09590", "A4W93_20030"},
    "PHV_PhaZ": {"A4W93_09590", "A4W93_20030"},
}

FLUX_METABOLITES = (
    "glc__D_e",
    "o2_e",
    "nh4_e",
    "arg__L_e",
    "glu__L_e",
    "leu__L_e",
    "ac_e",
    "lac__L_e",
    "etoh_e",
    "for_e",
    "succ_e",
    "lys__L_e",
)

DISPLAY_SPECIES = {
    "Actinoplanes_sp_OR16_lcp": "OR16",
    "Rhizobacter_gummiphilus_NS21": "NS21",
    "Lactobacillus_plantarum": "L. plantarum",
}

DISPLAY_METABOLITES = {
    "glc__D_e": "Glucose",
    "o2_e": "Oxygen",
    "nh4_e": "Ammonium",
    "arg__L_e": "Arginine",
    "glu__L_e": "Glutamate",
    "leu__L_e": "Leucine",
    "ac_e": "Acetate",
    "lac__L_e": "Lactate",
    "etoh_e": "Ethanol",
    "for_e": "Formate",
    "succ_e": "Succinate",
    "lys__L_e": "Lysine",
}


def zero_feasible_exchange_copies(models: Mapping[str, Any]) -> dict[str, Any]:
    """Copy models and remove compulsory exchange flux without opening uptake."""

    copies = {name: model.copy() for name, model in models.items()}
    for model in copies.values():
        for reaction in model.exchanges:
            reaction.bounds = (
                min(float(reaction.lower_bound), 0.0),
                max(float(reaction.upper_bound), 0.0),
            )
        # The simulator itself removes this compulsory synthetic secretion.
        if "R_GLYCOLIPOPROTEIN_SYN_SEC" in model.reactions:
            reaction = model.reactions.get_by_id("R_GLYCOLIPOPROTEIN_SYN_SEC")
            reaction.lower_bound = min(0.0, float(reaction.lower_bound))
    return copies


def forced_exchange_rows(models: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species, model in models.items():
        for reaction in model.exchanges:
            lower, upper = map(float, reaction.bounds)
            if lower <= 0.0 <= upper:
                continue
            metabolite = next(iter(reaction.metabolites), None)
            rows.append(
                {
                    "species": species,
                    "reaction": reaction.id,
                    "metabolite": metabolite.id if metabolite is not None else "",
                    "metabolite_name": metabolite.name if metabolite is not None else "",
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "forced_direction": "secretion" if lower > 0 else "uptake",
                }
            )
    return rows


def _mass_balance_status(reaction) -> tuple[str, dict[str, float]]:
    if any(not metabolite.formula for metabolite in reaction.metabolites):
        return "indeterminate", {}
    residual = {
        key: float(value)
        for key, value in reaction.check_mass_balance().items()
        if abs(float(value)) > 1e-9
    }
    return ("unbalanced" if residual else "balanced"), residual


def model_integrity_rows(models: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species, model in models.items():
        growth, _ = find_growth_reaction(model)
        boundary = set(model.boundary)
        counts = {"balanced": 0, "unbalanced": 0, "indeterminate": 0}
        for reaction in model.reactions:
            if reaction in boundary or reaction is growth:
                continue
            status, _ = _mass_balance_status(reaction)
            counts[status] += 1
        total = sum(counts.values())
        rows.append(
            {
                "species": species,
                "reactions": len(model.reactions),
                "metabolites": len(model.metabolites),
                "genes": len(model.genes),
                "internal_checked": total,
                **counts,
                "balanced_fraction": counts["balanced"] / max(total, 1),
                "unbalanced_fraction": counts["unbalanced"] / max(total, 1),
                "indeterminate_fraction": counts["indeterminate"] / max(total, 1),
            }
        )
    return rows


def polymer_reaction_rows(models: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species, model in models.items():
        for reaction_id in POLYMER_REACTION_IDS:
            if reaction_id not in model.reactions:
                continue
            reaction = model.reactions.get_by_id(reaction_id)
            status, residual = _mass_balance_status(reaction)
            if reaction_id.startswith("EX_"):
                # A boundary drain intentionally has no element-balanced
                # product side; it must not be reported as a chemical error.
                status = "boundary"
            gpr = str(reaction.gene_reaction_rule)
            expected = EXPECTED_GPRS.get(reaction_id, set())
            actual_tokens = {
                token.strip("()")
                for token in gpr.replace(" and ", " ").replace(" or ", " ").split()
                if token.strip("()")
            }
            gpr_status = "not assessed"
            if expected:
                gpr_status = "match" if expected.issubset(actual_tokens) else "mismatch"
            rows.append(
                {
                    "species": species,
                    "reaction": reaction_id,
                    "equation": reaction.reaction,
                    "lower_bound": float(reaction.lower_bound),
                    "upper_bound": float(reaction.upper_bound),
                    "mass_balance_status": status,
                    "mass_balance_residual": residual,
                    "residual_l1": float(sum(abs(value) for value in residual.values())),
                    "gpr": gpr,
                    "expected_gpr": sorted(expected),
                    "gpr_status": gpr_status,
                }
            )
    return rows


def merge_feed_rates(
    medium: Mapping[str, float], dt: float, additions: Iterable[Mapping[str, float]]
) -> dict[str, float]:
    """Return environmental supply caps for only the explicitly added solutes."""

    additions_total: dict[str, float] = {}
    for addition in additions:
        for metabolite, rate in addition.items():
            metabolite = canonical_metabolite_id(metabolite)
            additions_total[metabolite] = additions_total.get(metabolite, 0.0) + float(rate)
    return {
        metabolite: max(0.0, float(medium.get(metabolite, 0.0))) / dt + rate
        for metabolite, rate in additions_total.items()
    }


def feed_groups(specific_rate: float, yeast_proxy_rate: float) -> dict[str, dict[str, float]]:
    limiting_amino_acids = (
        "his__L_e",
        "phe__L_e",
        "asn__L_e",
        "asp__L_e",
        "ser__L_e",
        "gly_e",
        "ile__L_e",
        "ala__L_e",
    )
    return {
        "Defined-10": dict(COEXISTENCE_DEFINED_FEED_MMOL_L_H),
        "AA8": {metabolite: specific_rate for metabolite in limiting_amino_acids},
        "G3PS": {"g3ps_e": specific_rate},
        "Species C": {
            "succ_e": specific_rate,
            "glu__L_e": specific_rate,
            "mnl_e": specific_rate,
        },
        "Vitamins": {
            "nac_e": specific_rate * 0.1,
            "ribflv_e": specific_rate * 0.1,
            "pnto__R_e": specific_rate * 0.1,
            "thm_e": specific_rate * 0.1,
            "btn_e": specific_rate * 0.01,
            "4abz_e": specific_rate * 0.1,
            "fol_e": specific_rate * 0.01,
            "nicnt_e": specific_rate * 0.1,
        },
        "YE proxy": {
            metabolite: yeast_proxy_rate * coefficient
            for metabolite, coefficient in YEAST_EXTRACT_COMPOSITION.items()
        },
    }


def screen_nutrients(
    solver: SharedMediumCommunityLP,
    medium: Mapping[str, float],
    groups: Mapping[str, Mapping[str, float]],
    dt: float,
) -> tuple[list[dict[str, Any]], Any]:
    rows: list[dict[str, Any]] = []
    names = list(groups)
    selected_result = None
    for size in range(len(names) + 1):
        for selected in itertools.combinations(names, size):
            caps = merge_feed_rates(medium, dt, (groups[name] for name in selected))
            result = solver.solve(caps)
            row = {
                "groups": list(selected),
                "label": "Baseline" if not selected else " + ".join(selected),
                "group_count": len(selected),
                "feasible": bool(result.feasible),
                "common_growth_per_h": float(result.common_growth_per_h),
                "species_growth_per_h": result.species_growth_per_h,
                "limiting_metabolites": result.limiting_metabolites,
            }
            rows.append(row)
            if selected == ("Defined-10",):
                selected_result = result
    if selected_result is None:
        raise RuntimeError("Defined-10 scenario was not evaluated")
    return rows, selected_result


def exchange_flux_rows(result) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metabolite in FLUX_METABOLITES:
        balance = result.exchange_balance.get(metabolite, {})
        rates = balance.get("species_rates_mmol_l_h", {})
        for species in result.members:
            rows.append(
                {
                    "metabolite": metabolite,
                    "species": species,
                    # SharedMediumCommunityLP uses positive for environmental
                    # consumption and negative for production.
                    "net_consumption_mmol_l_h": float(rates.get(species, 0.0)),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def plot_summary(payload: Mapping[str, Any], output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "red": "#D55E00",
        "gray": "#8A8A8A",
        "light": "#D9E6EE",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.4), constrained_layout=True)

    # A: internal reaction elemental-balance audit.
    integrity = payload["model_integrity"]
    species_labels = [DISPLAY_SPECIES.get(row["species"], row["species"]) for row in integrity]
    fractions = np.asarray(
        [
            [row["balanced_fraction"], row["unbalanced_fraction"], row["indeterminate_fraction"]]
            for row in integrity
        ]
    )
    left = np.zeros(len(integrity))
    for index, (label, color) in enumerate(
        [("Balanced", colors["green"]), ("Unbalanced", colors["red"]), ("Indeterminate", colors["gray"])]
    ):
        axes[0, 0].barh(species_labels, 100 * fractions[:, index], left=left, color=color, label=label)
        left += 100 * fractions[:, index]
    axes[0, 0].set_xlim(0, 100)
    axes[0, 0].set_xlabel("Internal reactions (%)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axes[0, 0].text(-0.15, 1.08, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=10)

    # B: raw and normalized coexistence result plus best defined correction.
    comparison = payload["boundary_sensitivity"]
    labels = [row["label"] for row in comparison]
    values = [row["common_growth_per_h"] for row in comparison]
    bar_colors = [colors["gray"], colors["blue"], colors["green"]]
    bars = axes[0, 1].bar(range(len(labels)), values, color=bar_colors)
    axes[0, 1].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Common growth rate (h$^{-1}$)")
    axes[0, 1].set_ylim(0, max(values) * 1.23 if max(values) else 1)
    for bar, value in zip(bars, values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value + 0.0015, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0, 1].text(-0.15, 1.08, "b", transform=axes[0, 1].transAxes, fontweight="bold", fontsize=10)

    # C: representative nutrient combinations, including the production feed.
    show_groups = {
        (),
        ("Defined-10",),
        ("AA8",),
        ("G3PS",),
        ("Species C",),
        ("Vitamins",),
        ("YE proxy",),
        ("AA8", "Species C"),
        ("Species C", "YE proxy"),
    }
    nutrient_rows = [
        row for row in payload["nutrient_combinations"] if tuple(row["groups"]) in show_groups
    ]
    nutrient_rows = sorted(nutrient_rows, key=lambda row: (row["common_growth_per_h"], row["group_count"]))
    nutrient_rows = list(reversed(nutrient_rows))
    y = np.arange(len(nutrient_rows))
    growth = [row["common_growth_per_h"] for row in nutrient_rows]
    group_count = [row["group_count"] for row in nutrient_rows]
    palette = mpl.colormaps["viridis"](np.linspace(0.18, 0.82, max(group_count) + 1))
    axes[1, 0].barh(y, growth, color=[palette[count] for count in group_count])
    axes[1, 0].set_yticks(y, [row["label"] for row in nutrient_rows], fontsize=6.5)
    axes[1, 0].set_xlabel("Common growth rate (h$^{-1}$)")
    axes[1, 0].set_xlim(0, max(growth) * 1.06)
    axes[1, 0].text(-0.15, 1.08, "c", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=10)

    # D: environmental exchange fluxes under the selected production feed.
    exchange = payload["selected_exchange_fluxes"]
    members = list(payload["selected_members"])
    matrix = np.zeros((len(FLUX_METABOLITES), len(members)), dtype=float)
    for row in exchange:
        matrix[FLUX_METABOLITES.index(row["metabolite"]), members.index(row["species"])] = row[
            "net_consumption_mmol_l_h"
        ]
    vmax = max(float(np.max(np.abs(matrix))), 1e-9)
    image = axes[1, 1].imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1, 1].set_xticks(range(len(members)), [DISPLAY_SPECIES.get(name, name) for name in members], rotation=25, ha="right")
    axes[1, 1].set_yticks(range(len(FLUX_METABOLITES)), [DISPLAY_METABOLITES[item] for item in FLUX_METABOLITES], fontsize=6.5)
    colorbar = fig.colorbar(image, ax=axes[1, 1], fraction=0.045, pad=0.03)
    colorbar.set_label(r"Net consumption (mmol L$^{-1}$ h$^{-1}$)", fontsize=7)
    axes[1, 1].text(-0.15, 1.08, "d", transform=axes[1, 1].transAxes, fontweight="bold", fontsize=10)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure_base = output_dir / "Figure_flux_nutrition_audit"
    fig.savefig(figure_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_polymer_audit(payload: Mapping[str, Any], output_dir: Path) -> None:
    """Create a compact chemistry/topology audit for the rubber pathway."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    green, orange, red, blue, gray = "#009E73", "#E69F00", "#D55E00", "#0072B2", "#666666"
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), constrained_layout=True)

    # A: literature-aware topology versus the implemented coarse-graining.
    axis = axes[0]
    axis.set_xlim(-0.2, 4.25)
    axis.set_ylim(-1.72, 1.2)
    axis.axis("off")
    nodes = [
        (0.0, 0.0, "Rubber\n$(C_5H_8)_n$"),
        (1.35, 0.0, "C30 proxy\n$C_{30}H_{48}O_2$"),
        (2.7, 0.0, "ODTD\n$C_{15}H_{24}O_2$"),
        (4.0, 0.0, "AcCoA +\nPrCoA"),
        (4.0, -0.70, "PHB + PHBV\n(3HB / 3HV)"),
    ]
    for x, y, label in nodes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": blue, "linewidth": 1.1},
        )
    axis.annotate("", xy=(1.05, 0), xytext=(0.35, 0), arrowprops={"arrowstyle": "->", "color": orange, "lw": 1.5})
    axis.annotate("", xy=(2.4, 0), xytext=(1.7, 0), arrowprops={"arrowstyle": "->", "color": green, "lw": 1.5})
    axis.annotate("", xy=(3.7, 0), xytext=(3.05, 0), arrowprops={"arrowstyle": "->", "color": red, "lw": 1.5})
    axis.annotate("", xy=(4.0, -0.45), xytext=(4.0, -0.22), arrowprops={"arrowstyle": "->", "color": blue, "lw": 1.5})
    axis.text(0.7, 0.34, "Lcp/RoxB", ha="center", color=orange, fontweight="bold", fontsize=7.5)
    axis.text(2.05, 0.34, "RoxA", ha="center", color=green, fontweight="bold", fontsize=7.5)
    axis.text(3.37, 0.34, "Catabolism", ha="center", color=red, fontweight="bold", fontsize=7.5)
    axis.text(3.72, -0.35, "PhaA/B/C", ha="right", color=blue, fontweight="bold", fontsize=7.5)
    axis.text(-0.08, -1.01, "1  C30 denotes a measured-product distribution proxy", ha="left", va="top", fontsize=6.5, color=gray)
    axis.text(-0.08, -1.27, "2  RoxA has both direct and RoxB-assisted routes", ha="left", va="top", fontsize=6.5, color=gray)
    axis.text(-0.08, -1.53, "3  3HB and 3HV are tracked independently", ha="left", va="top", fontsize=6.5, color=green)
    axis.text(-0.08, 1.08, "a", transform=axis.transAxes, fontweight="bold", fontsize=10)

    # B: element/charge residuals in the custom reactions.
    keep = {
        "R_LCP", "R_ROXB", "R_ROXA", "R_ROXA_BULK", "R_C30_cat",
        "R_ODTD_cat", "PHB_PhaB", "PHV_PhaB", "PHB_syn", "PHV_syn",
        "PHB_PhaZ", "PHV_PhaZ",
    }
    rows = [row for row in payload["polymer_reactions"] if row["reaction"] in keep]
    elements = ("C", "H", "O", "N", "P", "S", "charge")
    matrix = np.asarray(
        [[float(row["mass_balance_residual"].get(element, 0.0)) for element in elements] for row in rows],
        dtype=float,
    )
    labels = [f"{DISPLAY_SPECIES.get(row['species'], row['species'])}: {row['reaction']}" for row in rows]
    vmax = max(float(np.max(np.abs(matrix))), 1.0)
    image = axes[1].imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_xticks(range(len(elements)), elements)
    axes[1].set_yticks(range(len(rows)), labels, fontsize=6.5)
    axes[1].tick_params(length=0)
    for y, row in enumerate(rows):
        marker = "GPR mismatch" if row["gpr_status"] == "mismatch" else "GPR absent" if row["gpr_status"] == "missing" else ""
        if marker:
            axes[1].text(len(elements) - 0.45, y, marker, ha="right", va="center", fontsize=5.7, color="black")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.045, pad=0.03)
    colorbar.set_label("Element/charge residual", fontsize=7)
    axes[1].text(-0.13, 1.08, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    figure_base = output_dir / "Figure_polymer_model_audit"
    fig.savefig(figure_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(payload: Mapping[str, Any], output_dir: Path) -> None:
    forced = payload["forced_exchange_bounds"]
    polymer = payload["polymer_reactions"]
    mismatched = [row for row in polymer if row["gpr_status"] in {"mismatch", "missing"}]
    unbalanced = [row for row in polymer if row["mass_balance_status"] == "unbalanced"]
    combinations = payload["nutrient_combinations"]
    baseline = next(row for row in combinations if not row["groups"])
    defined10 = next(row for row in combinations if row["groups"] == ["Defined-10"])
    baseline_growth = float(baseline["common_growth_per_h"])
    defined10_growth = float(defined10["common_growth_per_h"])
    gain = (
        100 * (defined10_growth / baseline_growth - 1)
        if baseline_growth > 1e-12
        else None
    )
    gain_text = f"{gain:+.1f}%" if gain is not None else "基準値0のため比率未定義"
    lines = [
        "# 3種共存系の代謝フラックス・栄養補正監査",
        "",
        "## 結論",
        "",
        "ポリマー炭素の二重計上、Lcp/RoxのGPR、RoxA経路トポロジー、追加反応の元素収支を修正した。現段階では培地候補の比較に使用できるが、ゴムからPHAへの絶対収率には未校正の細胞外酵素速度と仮説的輸送・下流redox量論が残る。",
        "",
        "## 境界条件",
        "",
        f"- 生SBMLでゼロフラックスを取れない交換反応は{len(forced)}本（修正後は0本）。",
        "- L. plantarumに残っていた13本の測定由来範囲は、方向性を維持してゼロを許す基底境界へ正規化した。",
        "- 特定実験の交換範囲は、再利用するGEM本体ではなくシナリオ側で付与する。",
        "",
        "## ポリマー分解モデル",
        "",
        f"- 元素収支が合わない追加反応: {', '.join(DISPLAY_SPECIES.get(row['species'], row['species']) + ':' + row['reaction'] for row in unbalanced) or 'なし'}。",
        f"- GPRが欠落または文献と不一致: {', '.join(DISPLAY_SPECIES.get(row['species'], row['species']) + ':' + row['reaction'] for row in mismatched) or 'なし'}。",
        "- Lcp/RoxBの生成物はC20以上の分布であり、C30は炭素保存された代表プールとして明示した。",
        "- RoxAにはポリマー末端からの直接ODTD生成とRoxB生成オリゴマーの処理を併設し、絶対的な直列依存を解消した。",
        "- バルクゴム交換とLcp/Rox反応はFBA内で0固定し、細胞外速度式だけがゴムと酸素を消費して可溶性C30/ODTDを供給する。",
        "",
        "## 可溶性培地の組合せスクリーニング",
        "",
        f"- zero-feasible基準条件の共通成長率: {baseline['common_growth_per_h']:.5f} h^-1。",
        f"- Defined-10流加: {defined10['common_growth_per_h']:.5f} h^-1（モデル予測で{gain_text}）。",
        "- Defined-10はグルコース、アルギニン、トリプトファン、ロイシン、グルタミン酸、イソロイシン、プロリン、フェニルアラニン、グルタミン、ピリドキサミンからなる。各速度は実装済みの24時間定義流加条件と一致させた。",
        "- ピリドキサミンは修復したWCFS1 2022モデルで顕在化した要求であり、旧AA8条件だけでは共通成長率がゼロのままである。",
        "- コハク酸＋グルタミン酸＋マンニトールは、現モデルのmax-min成長率を改善しなかった。実験上の種維持には有用な可能性があるが、GEM上の次の律速栄養ではない。",
        "- 酵母エキス近似はg/Lに校正されていないため、実レシピや用量反応として解釈しない。",
        "",
        "## 次の培養試験",
        "",
        "強化学習前に、基礎培地、Defined-10、Defined-10から各成分を1つずつ除いた条件、酵母エキス対照、コハク酸/グルタミン酸/マンニトール対照の小規模要因実験を行う。種別存在量、DO、NH4、pH・塩基要求量、細胞外アミノ酸・有機酸、ゴム質量減少、ODTD/オリゴマー、PHAを測定し、細胞外酵素Vmaxと輸送仮定を校正する。",
        "",
        "## 出力ファイル",
        "",
        "- `Figure_flux_nutrition_audit.pdf/png`",
        "- `Figure_polymer_model_audit.pdf/png`",
        "- `model_integrity.csv`",
        "- `forced_exchange_bounds.csv`",
        "- `polymer_reaction_audit.csv`",
        "- `nutrient_combinations.csv`",
        "- `selected_exchange_fluxes.csv`",
        "- `flux_nutrition_audit.json`",
    ]
    (output_dir / "flux_nutrition_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_models = select_consortium_models(load_sbml_models(args.sbml_dir))
    biomass, medium = get_initial_params(raw_models)
    forced = forced_exchange_rows(raw_models)
    integrity = model_integrity_rows(raw_models)
    polymer = polymer_reaction_rows(raw_models)

    raw_solver = SharedMediumCommunityLP(raw_models, medium, biomass, dt=args.dt)
    raw_baseline = raw_solver.solve()

    normalized_models = zero_feasible_exchange_copies(raw_models)
    normalized_solver = SharedMediumCommunityLP(
        normalized_models,
        medium,
        biomass,
        dt=args.dt,
        oxygen_transfer_mmol_l_h=args.oxygen_transfer,
    )
    groups = feed_groups(args.specific_rate, args.yeast_proxy_rate)
    nutrient_rows, selected_result = screen_nutrients(normalized_solver, medium, groups, args.dt)
    normalized_baseline = next(row for row in nutrient_rows if not row["groups"])
    selected_row = next(row for row in nutrient_rows if row["groups"] == ["Defined-10"])
    exchange = exchange_flux_rows(selected_result)

    payload = {
        "schema_version": 1,
        "parameters": {
            "dt_h": args.dt,
            "specific_supply_mmol_l_h": args.specific_rate,
            "yeast_proxy_scale": args.yeast_proxy_rate,
            "oxygen_transfer_mmol_l_h": args.oxygen_transfer,
            "polymer_supply_in_nutrient_screen": False,
        },
        "model_integrity": integrity,
        "forced_exchange_bounds": forced,
        "polymer_reactions": polymer,
        "simulator_findings": {
            "bulk_rubber_exchange_fixed_zero": True,
            "empirical_degrade_rubber_called_after_fba": True,
            "degradation_rates_argument_used": True,
            "extracellular_carbon_and_oxygen_conserved": True,
            "double_counting_risk": "controlled_by_boundary_separation",
        },
        "boundary_sensitivity": [
            {
                "label": "Raw SBML",
                "feasible": raw_baseline.feasible,
                "common_growth_per_h": raw_baseline.common_growth_per_h,
            },
            {
                "label": "Zero-feasible",
                "feasible": normalized_baseline["feasible"],
                "common_growth_per_h": normalized_baseline["common_growth_per_h"],
            },
            {
                "label": "Zero-feasible + Defined-10",
                "feasible": selected_row["feasible"],
                "common_growth_per_h": selected_row["common_growth_per_h"],
            },
        ],
        "feed_groups": groups,
        "nutrient_combinations": nutrient_rows,
        "selected_flux_scenario": "Defined-10",
        "selected_members": list(selected_result.members),
        "selected_exchange_fluxes": exchange,
        "selected_cross_feeding": [edge.__dict__ for edge in selected_result.cross_feeding],
        "interpretation_limits": [
            "FBA values are model predictions, not biological replicates or confidence intervals.",
            "Exchange-bound normalization is a sensitivity analysis, not an SBML correction.",
            "The yeast-extract action is a pseudo-composition and is not calibrated to g/L.",
            "Polymer carbon was excluded from nutrient ranking so that soluble-medium limitations are not confounded with uncalibrated extracellular enzyme rates.",
        ],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sbml-dir",
        type=Path,
        default=ROOT / "models" / "sbml" / "final_consortium",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "flux_nutrition_audit",
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--specific-rate", type=float, default=0.5)
    parser.add_argument("--yeast-proxy-rate", type=float, default=2.5)
    parser.add_argument("--oxygen-transfer", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = run(args)
    (args.output_dir / "flux_nutrition_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        args.output_dir / "model_integrity.csv",
        payload["model_integrity"],
        [
            "species",
            "reactions",
            "metabolites",
            "genes",
            "internal_checked",
            "balanced",
            "unbalanced",
            "indeterminate",
            "balanced_fraction",
            "unbalanced_fraction",
            "indeterminate_fraction",
        ],
    )
    _write_csv(
        args.output_dir / "forced_exchange_bounds.csv",
        payload["forced_exchange_bounds"],
        [
            "species",
            "reaction",
            "metabolite",
            "metabolite_name",
            "lower_bound",
            "upper_bound",
            "forced_direction",
        ],
    )
    _write_csv(
        args.output_dir / "polymer_reaction_audit.csv",
        payload["polymer_reactions"],
        [
            "species",
            "reaction",
            "equation",
            "lower_bound",
            "upper_bound",
            "mass_balance_status",
            "mass_balance_residual",
            "residual_l1",
            "gpr",
            "expected_gpr",
            "gpr_status",
        ],
    )
    _write_csv(
        args.output_dir / "nutrient_combinations.csv",
        payload["nutrient_combinations"],
        [
            "label",
            "groups",
            "group_count",
            "feasible",
            "common_growth_per_h",
            "species_growth_per_h",
            "limiting_metabolites",
        ],
    )
    _write_csv(
        args.output_dir / "selected_exchange_fluxes.csv",
        payload["selected_exchange_fluxes"],
        ["metabolite", "species", "net_consumption_mmol_l_h"],
    )
    plot_summary(payload, args.output_dir)
    plot_polymer_audit(payload, args.output_dir)
    write_report(payload, args.output_dir)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
