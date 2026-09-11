#!/usr/bin/env python3
"""Screen a non-PHA-producing helper for the OR16--NS21 fed-batch system.

The selected test organism is *Bacillus subtilis* 168.  Selection is based on
direct PHA co-culture evidence, availability of a curated GEM, and a feedable
role that is absent from the current OR16/NS21 reconstructions: extracellular
starch hydrolysis.  The script compares the live helper against rubber-only,
helper-without-starch, and matched-carbon direct-feed controls.  It therefore
does not assume that adding a third organism is beneficial.

The fed-batch implementation is a constant-volume, pulse-equivalent dFBA
approximation: ``rate * dt`` is added before every solve and dilution is not
modelled.  Results are hypotheses for culture design, not wet-lab validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cobra.io import load_json_model, read_sbml_model, write_sbml_model


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dfba_simulator import dFBASimulator  # noqa: E402
from src.metabolite_ids import canonical_metabolite_id  # noqa: E402
from src.utils import get_initial_params  # noqa: E402


OR16_PATH = ROOT / "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
NS21_PATH = ROOT / "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
HELPER_JSON = ROOT / "models/sbml/helper_candidates/Bacillus_subtilis_168_iYO844.json"
HELPER_SBML = ROOT / "models/sbml/helper_candidates/Bacillus_subtilis_168_iYO844_curated.xml"
HELPER_PROVENANCE = ROOT / "models/sbml/helper_candidates/Bacillus_subtilis_168_iYO844_provenance.json"
PHB_MONOMER_G_PER_MMOL = 86.09 / 1000.0


CANDIDATES = [
    {
        "candidate": "Bacillus subtilis 168",
        "direct_nonproducer_evidence": 5,
        "gem_quality_access": 5,
        "fed_batch_controllability": 4,
        "strain_accessibility": 5,
        "mechanistic_fit": 3,
        "competition_penalty": 3,
        "evidence": "PHA co-culture; sucrose hydrolysis and precursor supply",
        "gem": "iYO844 (844 genes; BiGG)",
        "reference": "https://doi.org/10.1016/j.biortech.2018.02.056",
    },
    {
        "candidate": "engineered Saccharomyces cerevisiae",
        "direct_nonproducer_evidence": 5,
        "gem_quality_access": 4,
        "fed_batch_controllability": 4,
        "strain_accessibility": 2,
        "mechanistic_fit": 3,
        "competition_penalty": 3,
        "evidence": "mcl-PHA +150% by xylose-to-lactate cross-feeding",
        "gem": "iMM904 exists, but not the exact engineered strain",
        "reference": "https://doi.org/10.1002/cjce.24751",
    },
    {
        "candidate": "Bacillus megaterium DSM319",
        "direct_nonproducer_evidence": 3,
        "gem_quality_access": 5,
        "fed_batch_controllability": 4,
        "strain_accessibility": 4,
        "mechanistic_fit": 3,
        "competition_penalty": 3,
        "evidence": "industrial 2-KLG helper; growth-factor export",
        "gem": "iJA1121 (1121 genes; manually curated)",
        "reference": "https://doi.org/10.1038/s41598-019-55041-w",
    },
    {
        "candidate": "Bacillus amyloliquefaciens",
        "direct_nonproducer_evidence": 4,
        "gem_quality_access": 2,
        "fed_batch_controllability": 4,
        "strain_accessibility": 3,
        "mechanistic_fit": 2,
        "competition_penalty": 4,
        "evidence": "B12 helper; amino acids/vitamins and oxygen removal",
        "gem": "iJYQ746 reported; strain/model artifact not openly standardized",
        "reference": "https://doi.org/10.1186/s12934-022-01773-w",
    },
    {
        "candidate": "Propionibacterium freudenreichii",
        "direct_nonproducer_evidence": 1,
        "gem_quality_access": 3,
        "fed_batch_controllability": 2,
        "strain_accessibility": 4,
        "mechanistic_fit": 1,
        "competition_penalty": 2,
        "evidence": "B12 producer, but no direct PHA-helper evidence",
        "gem": "published reconstructions exist",
        "reference": "https://doi.org/10.1186/s12934-022-01773-w",
    },
]


@dataclass(frozen=True)
class Scenario:
    name: str
    helper: bool
    feed_metabolite: str | None
    feed_rate_mmol_l_h: float
    helper_biomass_g_l: float = 0.03
    kla_per_h: float = 50.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_helper_model(write_artifacts: bool = True):
    """Load iYO844 and expose only the already gene-supported starch route."""

    if not HELPER_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {HELPER_JSON}. Download the BiGG iYO844 JSON artifact first."
        )
    model = load_json_model(str(HELPER_JSON))
    required = {"EX_starch_e", "AAMYL_1", "BIOMASS_BS_10"}
    missing = sorted(required.difference(reaction.id for reaction in model.reactions))
    if missing:
        raise ValueError(f"iYO844 is missing required reactions: {missing}")
    amylase = model.reactions.get_by_id("AAMYL_1")
    if "BSU03040" not in amylase.gene_reaction_rule:
        raise ValueError("The starch-hydrolysis reaction lacks the expected amylase GPR")

    # BiGG distributes this exchange closed.  Opening the boundary does not
    # add a reaction; it declares starch as the tested experimental feed.
    model.reactions.get_by_id("EX_starch_e").lower_bound = -10.0
    model.id = "iYO844_starch_helper"
    model.name = "Bacillus subtilis 168 iYO844; starch-helper test configuration"

    pha_terms = ("polyhydroxyalkanoate", "polyhydroxybutyrate")
    pha_like = [
        reaction.id
        for reaction in model.reactions
        if any(term in f"{reaction.id} {reaction.name or ''}".lower() for term in pha_terms)
    ]
    if pha_like:
        raise ValueError(f"The proposed nonproducer GEM contains PHA reactions: {pha_like}")

    if write_artifacts:
        HELPER_SBML.parent.mkdir(parents=True, exist_ok=True)
        write_sbml_model(model, str(HELPER_SBML))
        provenance = {
            "model": "iYO844",
            "organism": "Bacillus subtilis subsp. subtilis str. 168",
            "culture_collection_ids": [
                "NBRC 111470",
                "JCM 10629",
                "DSM 402",
                "ATCC 23857",
            ],
            "download_url": "http://bigg.ucsd.edu/api/v2/models/iYO844/download",
            "database": "BiGG Models 1.6",
            "retrieved": "2026-09-02",
            "source_sha256": _sha256(HELPER_JSON),
            "curated_sbml_sha256": _sha256(HELPER_SBML),
            "counts": {
                "reactions": len(model.reactions),
                "metabolites": len(model.metabolites),
                "genes": len(model.genes),
                "exchanges": len(model.exchanges),
            },
            "curation": [
                "EX_starch_e lower bound changed from 0 to -10 mmol gDW-1 h-1",
                "No reaction or gene was added",
                "AAMYL_1 is retained with GPR BSU03040",
                "No named PHA/PHB biosynthetic reaction was detected",
            ],
            "model_reference": "https://doi.org/10.1074/jbc.M703759200",
            "co_culture_reference": "https://doi.org/10.1016/j.biortech.2018.02.056",
            "limitations": [
                "iYO844 represents strain 168, not the exact Bacillus strain used in every PHA co-culture study",
                "FBA secretion is a feasible cooperative phenotype, not proof of secretion in an unengineered culture",
            ],
        }
        HELPER_PROVENANCE.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return model


def candidate_rows() -> list[dict[str, Any]]:
    rows = []
    for item in CANDIDATES:
        row = dict(item)
        positive = sum(
            row[key]
            for key in (
                "direct_nonproducer_evidence",
                "gem_quality_access",
                "fed_batch_controllability",
                "strain_accessibility",
                "mechanistic_fit",
            )
        )
        row["score"] = positive - row["competition_penalty"]
        rows.append(row)
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _base_models(helper: bool) -> dict[str, Any]:
    models = {
        "Actinoplanes_sp_OR16_lcp": read_sbml_model(str(OR16_PATH)),
        "Rhizobacter_gummiphilus_NS21": read_sbml_model(str(NS21_PATH)),
    }
    if helper:
        models["Bacillus_subtilis_168_iYO844"] = prepare_helper_model(
            write_artifacts=False
        )
    return models


def _state_row(simulator: dFBASimulator, scenario: Scenario, feed_added: float) -> dict[str, Any]:
    ns21 = simulator.state.species["Rhizobacter_gummiphilus_NS21"]
    helper = simulator.state.species.get("Bacillus_subtilis_168_iYO844")
    return {
        "scenario": scenario.name,
        "time_h": float(simulator.state.time),
        "pha_mmol_l": float(ns21.pha_accumulated),
        "pha_g_l_as_phb": float(ns21.pha_accumulated * PHB_MONOMER_G_PER_MMOL),
        "rubber_g_l": float(simulator.state.rubber_concentration),
        "nh4_mmol_l": float(simulator.state.metabolites.get("nh4_e", 0.0)),
        "o2_mmol_l": float(simulator.state.metabolites.get("o2_e", 0.0)),
        "or16_biomass_g_l": float(simulator.state.species["Actinoplanes_sp_OR16_lcp"].biomass),
        "ns21_biomass_g_l": float(ns21.biomass),
        "helper_biomass_g_l": float(helper.biomass) if helper is not None else 0.0,
        "cumulative_feed_mmol_l": float(feed_added),
        "helper_starch_uptake": float(helper.metabolite_uptake.get("starch_e", 0.0)) if helper else 0.0,
        "helper_dextrin_secretion": float(helper.metabolite_secretion.get("dextrin_e", 0.0)) if helper else 0.0,
        "helper_o2_uptake": float(helper.metabolite_uptake.get("o2_e", 0.0)) if helper else 0.0,
        "helper_nh4_uptake": float(helper.metabolite_uptake.get("nh4_e", 0.0)) if helper else 0.0,
        "ns21_dextrin_uptake": float(ns21.metabolite_uptake.get("dextrin_e", 0.0)),
        "ns21_glucose_uptake": float(ns21.metabolite_uptake.get("glc__D_e", 0.0)),
    }


def run_scenario(
    scenario: Scenario,
    hours: float,
    dt: float,
    initial_nh4_mmol_l: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models = _base_models(scenario.helper)
    biomass, medium = get_initial_params(models)
    biomass["Actinoplanes_sp_OR16_lcp"] = 0.5
    biomass["Rhizobacter_gummiphilus_NS21"] = 0.1
    if scenario.helper:
        biomass["Bacillus_subtilis_168_iYO844"] = scenario.helper_biomass_g_l
    medium.update(
        {
            "yeast_extract_e": 0.0,
            "mnl_e": 0.0,
            "glc__D_e": 0.0,
            "starch_e": 0.0,
            "nh4_e": float(initial_nh4_mmol_l),
        }
    )
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=10.0,
        volume=1.0,
        dt=dt,
        solver_backend="highs",
        fba_mode="cooperative",
        ph_control_target=7.0,
        polymer_oxygen_fraction=0.25,
        cooperative_optimize_live_objectives=True,
        cooperative_parsimony=False,
    )
    feed_added = 0.0
    trajectory = [_state_row(simulator, scenario, feed_added)]
    for _ in range(max(1, int(np.ceil(hours / dt)))):
        if scenario.feed_metabolite and scenario.feed_rate_mmol_l_h > 0.0:
            addition = scenario.feed_rate_mmol_l_h * dt
            simulator.state.metabolites[scenario.feed_metabolite] = (
                simulator.state.metabolites.get(scenario.feed_metabolite, 0.0)
                + addition
            )
            feed_added += addition
        simulator.step({}, {}, dynamic_kla=scenario.kla_per_h)
        trajectory.append(_state_row(simulator, scenario, feed_added))

    end = trajectory[-1]
    feed_repeat_mw = {
        "starch_e": 162.14,
        "glc__D_e": 180.16,
        "dextrin_e": 162.14,
    }.get(scenario.feed_metabolite or "", 0.0)
    feed_g_l = feed_added * feed_repeat_mw / 1000.0

    def integrated_exchange(flux_key: str, biomass_key: str) -> float:
        # Fluxes are mmol gDW-1 h-1 and biomass is gDW L-1.  The solve uses the
        # biomass at the beginning of each step, so integrate the reported
        # step flux against the preceding state (left-endpoint rectangle).
        return float(
            sum(
                current[flux_key] * previous[biomass_key] * dt
                for previous, current in zip(trajectory[:-1], trajectory[1:])
            )
        )

    summary = {
        **asdict(scenario),
        "hours": float(hours),
        "dt_h": float(dt),
        "initial_nh4_mmol_l": float(initial_nh4_mmol_l),
        "final_pha_mmol_l": end["pha_mmol_l"],
        "final_pha_g_l_as_phb": end["pha_g_l_as_phb"],
        "final_rubber_g_l": end["rubber_g_l"],
        "rubber_removed_g_l": 10.0 - end["rubber_g_l"],
        "final_nh4_mmol_l": end["nh4_mmol_l"],
        "final_or16_biomass_g_l": end["or16_biomass_g_l"],
        "final_ns21_biomass_g_l": end["ns21_biomass_g_l"],
        "final_helper_biomass_g_l": end["helper_biomass_g_l"],
        "cumulative_feed_mmol_l": feed_added,
        "cumulative_feed_g_l": feed_g_l,
        "pha_g_per_g_feed": (
            end["pha_g_l_as_phb"] / feed_g_l if feed_g_l > 0 else None
        ),
        "helper_starch_uptake_mmol_l": integrated_exchange(
            "helper_starch_uptake", "helper_biomass_g_l"
        ),
        "helper_dextrin_secretion_mmol_l": integrated_exchange(
            "helper_dextrin_secretion", "helper_biomass_g_l"
        ),
        "helper_o2_uptake_mmol_l": integrated_exchange(
            "helper_o2_uptake", "helper_biomass_g_l"
        ),
        "helper_nh4_uptake_mmol_l": integrated_exchange(
            "helper_nh4_uptake", "helper_biomass_g_l"
        ),
        "ns21_dextrin_uptake_mmol_l": integrated_exchange(
            "ns21_dextrin_uptake", "ns21_biomass_g_l"
        ),
        "solver_success_rate": simulator.get_solver_diagnostics()["solve_success_rate"],
    }
    return summary, trajectory


def build_scenarios(kla: float) -> list[Scenario]:
    scenarios = [
        Scenario("two_rubber_only", False, None, 0.0, kla_per_h=kla),
        Scenario("three_no_starch", True, None, 0.0, 0.03, kla),
    ]
    # Every helper feed rate has a rate-matched direct-feed control.  This is
    # essential because an apparently superior helper can otherwise just be
    # receiving a different amount of carbon.
    for rate in (0.10, 0.25, 0.50):
        scenarios.extend(
            [
                Scenario(
                    f"two_glucose_{rate:.2f}",
                    False,
                    "glc__D_e",
                    rate,
                    kla_per_h=kla,
                ),
                Scenario(
                    f"two_dextrin_{rate:.2f}",
                    False,
                    "dextrin_e",
                    rate,
                    kla_per_h=kla,
                ),
            ]
        )
    for biomass in (0.01, 0.03, 0.05):
        for rate in (0.10, 0.25, 0.50):
            scenarios.append(
                Scenario(
                    f"three_starch_b{biomass:.2f}_r{rate:.2f}",
                    True,
                    "starch_e",
                    rate,
                    biomass,
                    kla,
                )
            )
    return scenarios


def plot_results(
    candidates: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue, orange, green, purple, gray = "#0072B2", "#E69F00", "#009E73", "#6A51A3", "#666666"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

    ranked = list(reversed(candidates))
    axes[0, 0].barh(
        range(len(ranked)), [row["score"] for row in ranked], color=blue
    )
    axes[0, 0].set_yticks(
        range(len(ranked)), [row["candidate"] for row in ranked], fontsize=6.5
    )
    axes[0, 0].set_xlabel("Evidence-weighted candidate score")

    ordered = sorted(summaries, key=lambda row: row["final_pha_g_l_as_phb"])
    labels = [row["name"] for row in ordered]
    bar_colors = [green if row["helper"] else gray for row in ordered]
    axes[0, 1].barh(
        range(len(ordered)),
        [row["final_pha_g_l_as_phb"] for row in ordered],
        color=bar_colors,
    )
    axes[0, 1].set_yticks(range(len(labels)), labels, fontsize=5.3)
    axes[0, 1].set_xlabel(r"PHA model output as PHB (g L$^{-1}$)")

    best_helper = max(
        (row for row in summaries if row["helper"] and row["feed_metabolite"] == "starch_e"),
        key=lambda row: row["final_pha_g_l_as_phb"],
    )
    matched_rate = best_helper["feed_rate_mmol_l_h"]
    selected_names = [
        "two_rubber_only",
        f"two_glucose_{matched_rate:.2f}",
        f"two_dextrin_{matched_rate:.2f}",
        best_helper["name"],
    ]
    palette = [gray, orange, purple, green]
    for name, color in zip(selected_names, palette):
        rows = [row for row in trajectories if row["scenario"] == name]
        axes[1, 0].plot(
            [row["time_h"] for row in rows],
            [row["pha_g_l_as_phb"] for row in rows],
            label=name,
            color=color,
            lw=1.5,
        )
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel(r"PHA model output as PHB (g L$^{-1}$)")
    axes[1, 0].legend(frameon=False, fontsize=5.6)

    for name, color in zip(
        [f"two_glucose_{matched_rate:.2f}", best_helper["name"]],
        [orange, green],
    ):
        rows = [row for row in trajectories if row["scenario"] == name]
        axes[1, 1].plot(
            [row["time_h"] for row in rows],
            [row["nh4_mmol_l"] for row in rows],
            label=name,
            color=color,
            lw=1.5,
        )
    axes[1, 1].axhline(0.1, color=gray, ls="--", lw=0.9)
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel(r"NH$_4^+$ (mmol L$^{-1}$)")
    axes[1, 1].legend(frameon=False, fontsize=5.6)

    for label, axis in zip("abcd", axes.flat):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    base = output_dir / "Figure_nonproducer_helper_screen"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    candidates: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    rubber = next(row for row in summaries if row["name"] == "two_rubber_only")
    helper_no_feed = next(row for row in summaries if row["name"] == "three_no_starch")
    best = max(
        (
            row
            for row in summaries
            if row["helper"]
            and row["feed_metabolite"] == "starch_e"
            and row["solver_success_rate"] >= 0.95
        ),
        key=lambda row: row["final_pha_g_l_as_phb"],
    )
    matched_rate = best["feed_rate_mmol_l_h"]
    glucose = next(
        row
        for row in summaries
        if row["name"] == f"two_glucose_{matched_rate:.2f}"
    )
    dextrin = next(
        row
        for row in summaries
        if row["name"] == f"two_dextrin_{matched_rate:.2f}"
    )

    def gain(value: float, base: float) -> str:
        if base <= 0:
            return "比較不能"
        return f"{100.0 * (value / base - 1.0):+.1f}%"

    direct_reference = max(
        glucose["final_pha_g_l_as_phb"], dextrin["final_pha_g_l_as_phb"]
    )
    rubber_loss_fraction = (
        1.0 - best["rubber_removed_g_l"] / rubber["rubber_removed_g_l"]
        if rubber["rubber_removed_g_l"] > 0
        else float("inf")
    )
    verdict = (
        "conditional_go"
        if best["final_pha_g_l_as_phb"] > direct_reference * 1.05
        and rubber_loss_fraction <= 0.05
        else "no_go_as_live_third_strain"
    )
    lines = [
        "# 非生産側ヘルパー候補の選定と流加dFBA",
        "",
        "## 結論",
        "",
        f"- 候補比較の首位: **{candidates[0]['candidate']}**（スコア {candidates[0]['score']}）",
        f"- GEM: iYO844、{len(prepare_helper_model(False).reactions)} reactions。PHA/PHBという名称の生合成反応は検出されなかった。",
        f"- 最良の三種条件: `{best['name']}`、PHAモデル出力 {best['final_pha_g_l_as_phb']:.5f} g/L-as-PHB。",
        f"- ゴムのみ二種対照比: {gain(best['final_pha_g_l_as_phb'], rubber['final_pha_g_l_as_phb'])}",
        f"- 同一速度（{matched_rate:.2f} mmol/L/h）グルコース二種対照比: {gain(best['final_pha_g_l_as_phb'], glucose['final_pha_g_l_as_phb'])}",
        f"- 同一速度（{matched_rate:.2f} mmol/L/h）デキストリン二種対照比: {gain(best['final_pha_g_l_as_phb'], dextrin['final_pha_g_l_as_phb'])}",
        f"- ゴム除去量のゴムのみ対照比: {gain(best['rubber_removed_g_l'], rubber['rubber_removed_g_l'])}",
        f"- 第三菌の積算デンプン取込: {best['helper_starch_uptake_mmol_l']:.3f} mmol/L、デキストリン分泌: {best['helper_dextrin_secretion_mmol_l']:.3f} mmol/L。",
        f"- 第三菌の積算酸素取込: {best['helper_o2_uptake_mmol_l']:.3f} mmol/L、NH4取込: {best['helper_nh4_uptake_mmol_l']:.3f} mmol/L。",
        f"- 無給餌三種対照のLP成功率: {helper_no_feed['solver_success_rate']:.1%}（維持代謝を満たせず、PHA量の比較対象にはしない）",
        f"- 生菌第三菌としての判定: `{verdict}`",
        "",
        "## 選定理由",
        "",
        "*B. subtilis* は本系でPHA目的生産を担わせないヘルパー役であり、安価な糖質の分解と有機酸前駆体供給を担わせたPHA共培養の原著例がある。iYO844には遺伝子BSU03040に対応する細胞外α-amylase反応があり、現行OR16/NS21 GEMが直接取り込めないstarch_eを専用の流加入力として試験できる。株168はNBRC 111470、JCM 10629、DSM 402等として入手可能である。",
        "",
        "## 重要な限定",
        "",
        "- 協調FBAは三菌が全体目的に従う上限を示し、自然な分泌制御や進化的競争を保証しない。",
        "- starch_eはグルコース反復単位として扱った。実際の可溶化、粘度、α-amylase活性は未校正である。",
        "- PHA値はNS21モデル内プールをPHB単量体分子量で換算したもので、実測g/Lではない。",
        "- スターチ由来炭素がPHAへ入るため、製品を『ゴム由来PHA』と主張するには13C追跡が必要である。",
        "- 生菌ヘルパーが直接糖質流加を上回らない場合、第三菌ではなく酵素または直接流加を優先する。",
        "",
        "## 実験へ進む判定基準",
        "",
        "生菌ヘルパーは、同一炭素量の直接流加に対して総PHA濃度を5%以上改善し、OR16/NS21の両方が維持され、ゴム除去量を5%以上低下させない場合だけ採用候補とする。",
    ]
    (output_dir / "nonproducer_helper_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_dir / "decision.json").write_text(
        json.dumps(
            {
                "selected_candidate": candidates[0]["candidate"],
                "best_scenario": best,
                "controls": {
                    "rubber_only": rubber,
                    "matched_glucose": glucose,
                    "matched_dextrin": dextrin,
                    "helper_no_feed": helper_no_feed,
                },
                "verdict": verdict,
                "acceptance_rule": {
                    "minimum_gain_over_best_direct_feed": 0.05,
                    "maximum_rubber_removal_loss": 0.05,
                    "minimum_solver_success_rate": 0.95,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/nonproducer_helper_screen_20260902",
    )
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--initial-nh4", type=float, default=0.05)
    parser.add_argument("--kla", type=float, default=50.0)
    parser.add_argument("--scenario", action="append", default=[])
    args = parser.parse_args()
    if args.hours <= 0 or args.dt <= 0 or args.initial_nh4 < 0 or args.kla < 0:
        parser.error("hours/dt must be positive and initial-nh4/kla non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_helper_model(write_artifacts=True)
    candidates = candidate_rows()
    scenarios = build_scenarios(args.kla)
    if args.scenario:
        requested = set(args.scenario)
        scenarios = [item for item in scenarios if item.name in requested]
        missing = requested.difference(item.name for item in scenarios)
        if missing:
            parser.error(f"unknown scenarios: {sorted(missing)}")

    summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, 1):
        print(f"[{index}/{len(scenarios)}] {scenario.name}", flush=True)
        summary, rows = run_scenario(
            scenario,
            hours=args.hours,
            dt=args.dt,
            initial_nh4_mmol_l=args.initial_nh4,
        )
        summaries.append(summary)
        trajectories.extend(rows)

    _write_csv(args.output_dir / "candidate_ranking.csv", candidates)
    _write_csv(args.output_dir / "scenario_summary.csv", summaries)
    _write_csv(args.output_dir / "trajectories.csv", trajectories)
    payload = {
        "schema_version": 1,
        "study_type": "exact_HiGHS_cooperative_fed_batch_dFBA",
        "candidate_ranking": candidates,
        "scenarios": summaries,
        "parameters": {
            "hours": args.hours,
            "dt_h": args.dt,
            "initial_nh4_mmol_l": args.initial_nh4,
            "kla_per_h": args.kla,
            "feed_model": "constant-volume pulse-equivalent; no dilution",
            "pha_gate": "EX_pha_c open only when NH4 < 0.1 mmol/L",
        },
    }
    (args.output_dir / "nonproducer_helper_screen.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if len(summaries) == len(build_scenarios(args.kla)):
        plot_results(candidates, summaries, trajectories, args.output_dir)
        write_report(candidates, summaries, args.output_dir)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
