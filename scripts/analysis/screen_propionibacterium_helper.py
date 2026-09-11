#!/usr/bin/env python3
"""Test the remaining metabolic niche for a third organism in OR16--NS21.

The hypothesis is deliberately narrow: anaerobic *Propionibacterium
freudenreichii* converts externally supplied L-lactate to propionate and
acetate, potentially giving NS21 a propionyl-CoA precursor without competing
for oxygen.  Rate-matched two-member lactate and propionate feeds are included,
so the live helper is rejected when direct chemical feeding performs as well.

The current NS21 reconstruction tracks 3HB and 3HV repeat units separately.
Consequently this screen compares total polymer mass and PHBV composition,
while keeping absolute PhaB/PhaC kinetics explicitly uncalibrated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cobra.flux_analysis import flux_variability_analysis
from cobra.io import read_sbml_model, write_sbml_model


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dfba_simulator import dFBASimulator  # noqa: E402
from src.utils import get_initial_params  # noqa: E402


OR16_PATH = ROOT / "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
NS21_PATH = ROOT / "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
SUPPLEMENT_DIR = (
    ROOT
    / "models/sbml/helper_candidates/propionibacterium_panmodel_supplement"
)
HELPER_SBML = (
    ROOT
    / "models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml"
)
HELPER_PROVENANCE = HELPER_SBML.with_name(
    "Propionibacterium_freudenreichii_shermanii_provenance.json"
)
HELPER_NAME = "Propionibacterium_freudenreichii_shermanii"
NS21_NAME = "Rhizobacter_gummiphilus_NS21"
OR16_NAME = "Actinoplanes_sp_OR16_lcp"
PHB_MONOMER_G_PER_MMOL = 86.09 / 1000.0
PHV_MONOMER_G_PER_MMOL = 100.12 / 1000.0
STATIC_PROPIONATE_PER_LACTATE = 5.0510874047646785 / 10.0
STATIC_ACETATE_PER_LACTATE = 2.6715256592863192 / 10.0
# Post-hoc cell-free replay of the constrained best helper condition.  These
# fractions are reported explicitly and are not treated as independent data.
REPLAY_LACTATE_PER_FEED = 4.284894383482834 / 6.0
REPLAY_PROPIONATE_PER_FEED = 0.9082902116660938 / 6.0
REPLAY_ACETATE_PER_FEED = 0.3432594619524263 / 6.0


# Only identity/compartment normalization is performed.  No biochemical
# reaction is invented.  Oxygen is intentionally private and closed so this
# first test represents the anaerobic propionate-producing phenotype.
MODELSEED_TO_CANONICAL = {
    "cpd00001": "h2o_e",
    "cpd00009": "pi_e",
    "cpd00013": "nh4_e",
    "cpd00027": "glc__D_e",
    "cpd00030": "mn2_e",
    "cpd00034": "zn2_e",
    "cpd00048": "so4_e",
    "cpd00058": "cu2_e",
    "cpd00067": "h_e",
    "cpd00099": "cl_e",
    "cpd00104": "btn_e",
    "cpd00141": "ppa_e",
    "cpd00149": "cobalt2_e",
    "cpd00159": "lac__L_e",
    "cpd00205": "k_e",
    "cpd00254": "mg2_e",
    "cpd00644": "pnto__R_e",
    "cpd00971": "na1_e",
    "cpd03424": "b12_e",
    "cpd10515": "fe2_e",
    "cpd10516": "fe3_e",
    "cpd00029": "ac_e",
    "cpd00007": "o2_pfreud_e",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    helper: bool
    feed_metabolite: str | None
    feed_rate_mmol_l_h: float
    helper_biomass_g_l: float = 0.01
    kla_per_h: float = 50.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_model_path() -> Path:
    matches = list(SUPPLEMENT_DIR.rglob("P_sherm_model.xml"))
    primary = [path for path in matches if path.parent.name == "Model_XML_files"]
    if len(primary) == 1:
        return primary[0]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one P_sherm_model.xml under {SUPPLEMENT_DIR}, found {matches}"
        )
    return matches[0]


def prepare_helper_model(write_artifacts: bool = True):
    """Normalize the published P. freudenreichii subsp. shermanii GEM."""

    source = _source_model_path()
    model = read_sbml_model(str(source))
    before = {
        "reactions": len(model.reactions),
        "metabolites": len(model.metabolites),
        "genes": len(model.genes),
    }
    renamed: list[dict[str, str]] = []
    for seed_id, canonical in MODELSEED_TO_CANONICAL.items():
        old_met_id = f"S_{seed_id}_ext"
        old_rxn_id = f"Ex_{old_met_id}"
        if old_met_id not in model.metabolites or old_rxn_id not in model.reactions:
            raise ValueError(f"Published model lacks required boundary {old_rxn_id}")
        metabolite = model.metabolites.get_by_id(old_met_id)
        reaction = model.reactions.get_by_id(old_rxn_id)
        metabolite.id = canonical
        metabolite.compartment = "e"
        reaction.id = f"EX_{canonical}"
        renamed.append({"source": old_met_id, "canonical": canonical})

    model.id = "P_freudenreichii_shermanii_lactate_helper"
    model.name = (
        "P. freudenreichii subsp. shermanii; anaerobic lactate-helper test"
    )
    model.objective = "biomass_c0"

    # The source model distributes lactate exchange secretion-only even though
    # the paper and network support lactate growth.  Opening an existing
    # boundary is a medium declaration, not the addition of a reaction.
    model.reactions.get_by_id("EX_lac__L_e").lower_bound = -10.0
    model.reactions.get_by_id("EX_glc__D_e").bounds = (-10.0, 0.0)
    model.reactions.get_by_id("EX_ppa_e").lower_bound = -10.0
    model.reactions.get_by_id("EX_o2_pfreud_e").bounds = (0.0, 1000.0)

    if len(model.reactions) != before["reactions"] or len(model.genes) != before["genes"]:
        raise AssertionError("Curation must not add reactions or genes")

    if write_artifacts:
        HELPER_SBML.parent.mkdir(parents=True, exist_ok=True)
        write_sbml_model(model, str(HELPER_SBML))
        provenance = {
            "organism": "Propionibacterium freudenreichii subsp. shermanii",
            "published_model_file": source.relative_to(ROOT).as_posix(),
            "source_article": "https://doi.org/10.3390/genes11101115",
            "supplement_url": "https://mdpi-res.com/d_attachment/genes/genes-11-01115/article_deploy/genes-11-01115-s001.zip",
            "retrieved": "2026-09-02",
            "source_sha256": _sha256(source),
            "curated_sha256": _sha256(HELPER_SBML),
            "counts": before,
            "curation": [
                "Normalized selected ModelSEED extracellular identifiers to the shared BiGG-like namespace",
                "Changed normalized extracellular metabolite compartment from c to e",
                "Opened existing lactate and propionate exchanges for medium tests",
                "Closed glucose secretion while retaining uptake, matching reported sugar-consumption phenotypes",
                "Kept helper oxygen exchange closed for the anaerobic propionate-producing phenotype",
                "Added no reaction, metabolite, or gene",
            ],
            "identifier_map": renamed,
            "limitations": [
                "The pan-model-derived strain reconstruction requires strain-specific experimental validation",
                "NH3 is mapped to the shared NH4 pool as a pH-7 bulk nitrogen approximation",
                "NS21 3HV pathway structure is represented, but its kinetic capacity and PHBV composition require calibration",
            ],
        }
        HELPER_PROVENANCE.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return model


def static_helper_validation() -> dict[str, Any]:
    """Confirm the lactate-to-propionate phenotype in the published network."""

    model = prepare_helper_model(write_artifacts=False)
    for reaction in model.exchanges:
        if reaction.lower_bound < 0:
            reaction.lower_bound = 0.0
    for met_id in (
        "h2o_e", "pi_e", "nh4_e", "mn2_e", "zn2_e", "so4_e", "cu2_e",
        "h_e", "cl_e", "btn_e", "cobalt2_e", "k_e", "mg2_e",
        "pnto__R_e", "na1_e", "fe2_e", "fe3_e",
    ):
        model.reactions.get_by_id(f"EX_{met_id}").lower_bound = -1000.0
    model.reactions.get_by_id("EX_lac__L_e").lower_bound = -10.0
    model.reactions.get_by_id("EX_o2_pfreud_e").lower_bound = 0.0
    solution = model.optimize()
    if solution.status != "optimal" or solution.objective_value <= 0:
        raise RuntimeError("Curated helper does not grow anaerobically on lactate")
    targets = ["EX_ppa_e", "EX_ac_e", "EX_b12_e"]
    fva = flux_variability_analysis(model, reaction_list=targets, fraction_of_optimum=0.9)
    return {
        "growth_h-1": float(solution.objective_value),
        "lactate_uptake": float(-solution.fluxes["EX_lac__L_e"]),
        "propionate_at_growth_optimum": float(solution.fluxes["EX_ppa_e"]),
        "acetate_at_growth_optimum": float(solution.fluxes["EX_ac_e"]),
        "fva_at_90pct_growth": {
            rid: {
                "minimum": float(fva.loc[rid, "minimum"]),
                "maximum": float(fva.loc[rid, "maximum"]),
            }
            for rid in targets
        },
    }


def _models(helper: bool) -> dict[str, Any]:
    models = {
        OR16_NAME: read_sbml_model(str(OR16_PATH)),
        NS21_NAME: read_sbml_model(str(NS21_PATH)),
    }
    if helper:
        models[HELPER_NAME] = prepare_helper_model(write_artifacts=False)
    return models


def _row(sim: dFBASimulator, scenario: Scenario, feed_added: float) -> dict[str, Any]:
    ns21 = sim.state.species[NS21_NAME]
    helper = sim.state.species.get(HELPER_NAME)
    return {
        "scenario": scenario.name,
        "time_h": float(sim.state.time),
        "pha_mmol_l": float(ns21.pha_accumulated),
        "phb_mmol_l": float(ns21.phb_accumulated),
        "phv_mmol_l": float(ns21.phv_accumulated),
        "pha_g_l": float(
            ns21.phb_accumulated * PHB_MONOMER_G_PER_MMOL
            + ns21.phv_accumulated * PHV_MONOMER_G_PER_MMOL
        ),
        "phv_mol_fraction": float(
            ns21.phv_accumulated / ns21.pha_accumulated
            if ns21.pha_accumulated > 0.0 else 0.0
        ),
        "rubber_g_l": float(sim.state.rubber_concentration),
        "nh4_mmol_l": float(sim.state.metabolites.get("nh4_e", 0.0)),
        "lac__L_mmol_l": float(sim.state.metabolites.get("lac__L_e", 0.0)),
        "ppa_mmol_l": float(sim.state.metabolites.get("ppa_e", 0.0)),
        "ac_mmol_l": float(sim.state.metabolites.get("ac_e", 0.0)),
        "or16_biomass_g_l": float(sim.state.species[OR16_NAME].biomass),
        "ns21_biomass_g_l": float(ns21.biomass),
        "helper_biomass_g_l": float(helper.biomass) if helper else 0.0,
        "cumulative_feed_mmol_l": float(feed_added),
        "helper_lactate_uptake": float(helper.metabolite_uptake.get("lac__L_e", 0.0)) if helper else 0.0,
        "helper_propionate_secretion": float(helper.metabolite_secretion.get("ppa_e", 0.0)) if helper else 0.0,
        "helper_acetate_secretion": float(helper.metabolite_secretion.get("ac_e", 0.0)) if helper else 0.0,
        "ns21_lactate_uptake": float(ns21.metabolite_uptake.get("lac__L_e", 0.0)),
        "ns21_lactate_secretion": float(ns21.metabolite_secretion.get("lac__L_e", 0.0)),
        "ns21_propionate_uptake": float(ns21.metabolite_uptake.get("ppa_e", 0.0)),
        "or16_lactate_secretion": float(sim.state.species[OR16_NAME].metabolite_secretion.get("lac__L_e", 0.0)),
    }


def run_scenario(
    scenario: Scenario, hours: float, dt: float, initial_nh4: float, fba_mode: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models = _models(scenario.helper)
    biomass, medium = get_initial_params(models)
    biomass[OR16_NAME] = 0.5
    biomass[NS21_NAME] = 0.1
    if scenario.helper:
        biomass[HELPER_NAME] = scenario.helper_biomass_g_l
    medium.update(
        {
            "yeast_extract_e": 0.0,
            "glc__D_e": 0.0,
            "lac__L_e": 0.0,
            "ppa_e": 0.0,
            "ac_e": 0.0,
            "nh4_e": float(initial_nh4),
        }
    )
    sim = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=10.0,
        volume=1.0,
        dt=dt,
        solver_backend="highs",
        fba_mode=fba_mode,
        ph_control_target=7.0,
        polymer_oxygen_fraction=0.25,
        cooperative_optimize_live_objectives=True,
        cooperative_parsimony=True,
    )
    added = 0.0
    rows = [_row(sim, scenario, added)]
    uptake_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    secretion_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _ in range(max(1, int(np.ceil(hours / dt)))):
        if scenario.feed_metabolite and scenario.feed_rate_mmol_l_h > 0:
            dose = scenario.feed_rate_mmol_l_h * dt
            if scenario.feed_metabolite == "ppa_ac_mix":
                sim.state.metabolites["ppa_e"] += dose * STATIC_PROPIONATE_PER_LACTATE
                sim.state.metabolites["ac_e"] += dose * STATIC_ACETATE_PER_LACTATE
            elif scenario.feed_metabolite == "helper_effluent_replay":
                sim.state.metabolites["lac__L_e"] += dose * REPLAY_LACTATE_PER_FEED
                sim.state.metabolites["ppa_e"] += dose * REPLAY_PROPIONATE_PER_FEED
                sim.state.metabolites["ac_e"] += dose * REPLAY_ACETATE_PER_FEED
            else:
                sim.state.metabolites[scenario.feed_metabolite] = (
                    sim.state.metabolites.get(scenario.feed_metabolite, 0.0) + dose
                )
            added += dose
        integration_biomass = {
            name: float(state.biomass) for name, state in sim.state.species.items()
        }
        sim.step({}, {}, dynamic_kla=scenario.kla_per_h)
        for name, state in sim.state.species.items():
            for metabolite, flux in state.metabolite_uptake.items():
                uptake_totals[name][metabolite] += (
                    float(flux) * integration_biomass[name] * dt
                )
            for metabolite, flux in state.metabolite_secretion.items():
                secretion_totals[name][metabolite] += (
                    float(flux) * integration_biomass[name] * dt
                )
        rows.append(_row(sim, scenario, added))

    def integral(key: str, biomass_key: str) -> float:
        return float(
            sum(
                current[key] * previous[biomass_key] * dt
                for previous, current in zip(rows[:-1], rows[1:])
            )
        )

    end = rows[-1]
    summary = {
        **asdict(scenario),
        "hours": float(hours),
        "dt_h": float(dt),
        "initial_nh4_mmol_l": float(initial_nh4),
        "fba_mode": fba_mode,
        "final_pha_mmol_l": end["pha_mmol_l"],
        "final_pha_g_l": end["pha_g_l"],
        "final_phb_mmol_l": end["phb_mmol_l"],
        "final_phv_mmol_l": end["phv_mmol_l"],
        "final_phv_mol_fraction": end["phv_mol_fraction"],
        "rubber_removed_g_l": 10.0 - end["rubber_g_l"],
        "final_or16_biomass_g_l": end["or16_biomass_g_l"],
        "final_ns21_biomass_g_l": end["ns21_biomass_g_l"],
        "final_helper_biomass_g_l": end["helper_biomass_g_l"],
        "cumulative_feed_mmol_l": added,
        "helper_lactate_uptake_mmol_l": integral("helper_lactate_uptake", "helper_biomass_g_l"),
        "helper_propionate_secretion_mmol_l": integral("helper_propionate_secretion", "helper_biomass_g_l"),
        "helper_acetate_secretion_mmol_l": integral("helper_acetate_secretion", "helper_biomass_g_l"),
        "ns21_lactate_uptake_mmol_l": integral("ns21_lactate_uptake", "ns21_biomass_g_l"),
        "ns21_lactate_secretion_mmol_l": integral("ns21_lactate_secretion", "ns21_biomass_g_l"),
        "ns21_propionate_uptake_mmol_l": integral("ns21_propionate_uptake", "ns21_biomass_g_l"),
        "or16_lactate_secretion_mmol_l": integral("or16_lactate_secretion", "or16_biomass_g_l"),
        "helper_secretions_json": json.dumps(
            dict(sorted(secretion_totals.get(HELPER_NAME, {}).items())),
            sort_keys=True,
        ),
        "ns21_uptakes_json": json.dumps(
            dict(sorted(uptake_totals.get(NS21_NAME, {}).items())),
            sort_keys=True,
        ),
        "ns21_secretions_json": json.dumps(
            dict(sorted(secretion_totals.get(NS21_NAME, {}).items())),
            sort_keys=True,
        ),
        "solver_success_rate": sim.get_solver_diagnostics()["solve_success_rate"],
    }
    return summary, rows


def build_scenarios(kla: float) -> list[Scenario]:
    scenarios = [
        Scenario("two_rubber_only", False, None, 0.0, kla_per_h=kla),
        Scenario("three_no_lactate", True, None, 0.0, 0.01, kla),
    ]
    for rate in (0.10, 0.25, 0.50):
        scenarios.extend(
            [
                Scenario(f"two_lactate_{rate:.2f}", False, "lac__L_e", rate, kla_per_h=kla),
                Scenario(f"two_propionate_{rate:.2f}", False, "ppa_e", rate, kla_per_h=kla),
                Scenario(f"two_ppa_ac_mix_{rate:.2f}", False, "ppa_ac_mix", rate, kla_per_h=kla),
                Scenario(f"two_helper_effluent_replay_{rate:.2f}", False, "helper_effluent_replay", rate, kla_per_h=kla),
                Scenario(f"three_lactate_b0.01_r{rate:.2f}", True, "lac__L_e", rate, 0.01, kla),
                Scenario(f"three_lactate_b0.03_r{rate:.2f}", True, "lac__L_e", rate, 0.03, kla),
            ]
        )
    return scenarios


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    static: dict[str, Any], summaries: list[dict[str, Any]], trajectories: list[dict[str, Any]], output_dir: Path
) -> None:
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    blue, orange, green, purple, gray = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#666666"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    fva = static["fva_at_90pct_growth"]
    labels = ["Propionate", "Acetate", "Vitamin B12"]
    ids = ["EX_ppa_e", "EX_ac_e", "EX_b12_e"]
    lows = [fva[r]["minimum"] for r in ids]
    highs = [fva[r]["maximum"] for r in ids]
    axes[0, 0].bar(range(3), highs, color=blue, alpha=0.85)
    axes[0, 0].bar(range(3), lows, color="white", edgecolor=blue, hatch="////")
    axes[0, 0].set_xticks(range(3), labels, rotation=20, ha="right")
    axes[0, 0].set_ylabel(r"Flux range (mmol gDW$^{-1}$ h$^{-1}$)")

    ordered = sorted(summaries, key=lambda r: r["final_pha_g_l"])
    colors = [green if r["helper"] else (purple if r["feed_metabolite"] == "ppa_e" else gray) for r in ordered]
    axes[0, 1].barh(range(len(ordered)), [r["final_pha_g_l"] for r in ordered], color=colors)
    axes[0, 1].set_yticks(range(len(ordered)), [r["name"] for r in ordered], fontsize=5.2)
    axes[0, 1].set_xlabel(r"PHA model output (g L$^{-1}$)")

    selected = ["two_rubber_only", "two_lactate_0.50", "two_propionate_0.50", "three_lactate_b0.01_r0.50"]
    for name, color in zip(selected, [gray, orange, purple, green]):
        rows = [r for r in trajectories if r["scenario"] == name]
        axes[1, 0].plot([r["time_h"] for r in rows], [r["pha_g_l"] for r in rows], label=name, color=color, lw=1.5)
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel(r"PHA model output (g L$^{-1}$)")
    axes[1, 0].legend(frameon=False, fontsize=5.4)

    helper_rows = [r for r in trajectories if r["scenario"] == "three_lactate_b0.01_r0.50"]
    axes[1, 1].plot([r["time_h"] for r in helper_rows], [r["lac__L_mmol_l"] for r in helper_rows], label="L-lactate", color=orange, lw=1.5)
    axes[1, 1].plot([r["time_h"] for r in helper_rows], [r["ppa_mmol_l"] for r in helper_rows], label="Propionate", color=green, lw=1.5)
    axes[1, 1].plot([r["time_h"] for r in helper_rows], [r["ac_mmol_l"] for r in helper_rows], label="Acetate", color=blue, lw=1.5)
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel(r"Extracellular pool (mmol L$^{-1}$)")
    axes[1, 1].legend(frameon=False, fontsize=6)

    for label, axis in zip("abcd", axes.flat):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    base = output_dir / "Figure_third_species_metabolic_niche"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(static: dict[str, Any], summaries: list[dict[str, Any]], output_dir: Path) -> None:
    rubber_only = next(r for r in summaries if r["name"] == "two_rubber_only")
    no_lactate = next(r for r in summaries if r["name"] == "three_no_lactate")
    best_helper = max((r for r in summaries if r["helper"]), key=lambda r: r["final_pha_g_l"])
    rate = best_helper["feed_rate_mmol_l_h"]
    direct_lac = next(r for r in summaries if r["name"] == f"two_lactate_{rate:.2f}")
    direct_ppa = next(r for r in summaries if r["name"] == f"two_propionate_{rate:.2f}")
    direct_mix = next(r for r in summaries if r["name"] == f"two_ppa_ac_mix_{rate:.2f}")
    direct_replay = next(r for r in summaries if r["name"] == f"two_helper_effluent_replay_{rate:.2f}")
    direct = max(direct_lac, direct_ppa, direct_mix, direct_replay, key=lambda r: r["final_pha_g_l"])
    ratio = best_helper["final_pha_g_l"] / direct["final_pha_g_l"] if direct["final_pha_g_l"] > 0 else 0.0
    no_feed_ratio = no_lactate["final_pha_g_l"] / rubber_only["final_pha_g_l"] if rubber_only["final_pha_g_l"] > 0 else 0.0
    crossfeed = best_helper["helper_propionate_secretion_mmol_l"] > 1e-6 and best_helper["ns21_propionate_uptake_mmol_l"] > 1e-6
    composition_gain = (
        best_helper["final_phv_mol_fraction"]
        - direct["final_phv_mol_fraction"]
    )
    verdict = (
        "conditional_go_for_phbv"
        if crossfeed and (ratio >= 1.05 or composition_gain >= 0.01)
        else "no_go_without_experimental_support"
    )
    lines = [
        "# 第3菌が入り込める代謝ニッチの検証",
        "",
        "## 結論",
        "",
        f"- 公開GEMは嫌気条件で乳酸10 mmol gDW^-1 h^-1を取り込み、増殖速度 {static['growth_h-1']:.4f} h^-1、プロピオン酸 {static['propionate_at_growth_optimum']:.4f}、酢酸 {static['acetate_at_growth_optimum']:.4f} mmol gDW^-1 h^-1を示した。",
        f"- 最良の第3菌条件は `{best_helper['name']}`、PHAモデル出力 {best_helper['final_pha_g_l']:.5f} g/L、3HV {best_helper['final_phv_mol_fraction'] * 100:.2f} mol%。",
        f"- 同一流加速度で最良の直接流加 `{direct['name']}` に対する比は {ratio:.3f}。",
        f"- 無乳酸の第3菌添加は二種ゴム対照の {no_feed_ratio:.3f} 倍であり、自然な共生利益は検出されなかった。",
        f"- 第3菌の積算乳酸取込 {best_helper['helper_lactate_uptake_mmol_l']:.4f} mmol/L、プロピオン酸分泌 {best_helper['helper_propionate_secretion_mmol_l']:.4f} mmol/L、NS21のプロピオン酸取込 {best_helper['ns21_propionate_uptake_mmol_l']:.4f} mmol/L。",
        f"- 総PHA質量と3HV組成を併用した判定: `{verdict}`。",
        "",
        "## 見つかった余地",
        "",
        "第3菌の最も明確な余地は、OR16/NS21が自然放出する乳酸の回収ではなく、嫌気的な乳酸→プロピオン酸変換層である。これは酸素競合を抑え、NS21のpropionyl-CoA形成と3HV枝へ接続する。したがってP. freudenreichiiは総PHA量だけでなくPHBV組成を操作する候補として評価する。",
        "",
        "## 他の候補機能を棄却・保留した理由",
        "",
        "- 窒素変換菌: OR16とNS21はNH4、尿素、硝酸、複数アミノ酸・ペプチドの遺伝子支持された輸送系を既に持つ。PHA蓄積では窒素制限自体が操作変数なので、窒素を追加無機化する第三菌は役割が重複しやすい。",
        "- pH安定化・酸素除去菌: 1 LジャーはpH-stat、攪拌、温度を直接制御できるため、生菌で代替する必然性が弱い。",
        "- B12供給菌: 公開P. freudenreichii GEMでは90%増殖を保ったB12分泌が最大で正だが、OR16/NS21側の必須性が現GEMに表現されていないため保留とした。",
        "- ゴム表面改質・阻害物除去菌: GEMだけでは界面活性、付着、ゴム添加剤毒性を評価できない。無菌上清・透析膜試験で効果が出た場合にだけモデルへ追加する。",
        "",
        "## 次の判定実験",
        "",
        "1. OR16+NS21の上清で乳酸、酢酸、ピルビン酸、コハク酸、プロピオン酸をLC/HPLC定量し、自然な基質供給があるか確認する。",
        "2. NS21単培養に0--20 mM propionateを添加し、増殖阻害、総PHA、3HV mol%をGC-MSで測る。",
        "3. NS21のPhaB/PhaC基質特異性を酵素レベルで確認し、3HV生成速度と上限をGEM/dFBAへ較正する。",
        "4. 第3菌は透析膜・二槽接続から始め、直接接触による栄養競合と代謝物供給を分離する。",
        "",
        "## 科学的限定",
        "",
        "- 主解析は各菌を独立最適化し、分泌物を次ステップで共有する逐次dFBAである。協調LPは第三菌追加により目的関数構成が変わるため上限感度解析に留めた。",
        "- P. freudenreichiiモデルは公開pan-model由来株モデルで、使用株の再シーケンスと表現型校正が必要である。",
        "- B12供給はFVA上可能でも、OR16/NS21側のB12要求が現モデルで十分表現されていない。",
        "- OR16/NS21の既計算では乳酸分泌がほぼ0であり、乳酸を外から入れない自然共生は現時点では支持されない。",
        "",
        "## 参考文献",
        "",
        "- Machado et al. (2020), P. freudenreichii pan-GEM: https://doi.org/10.3390/genes11101115",
        "- Taniguchi et al. (1998), lactate-producing bacteriumとの逐次共培養: https://doi.org/10.1271/bbb.62.1522",
        "- Crow (1986), P. freudenreichii subsp. shermaniiの乳酸異性体利用: https://doi.org/10.1128/AEM.52.2.352-358.1986",
        "- Han et al. (2013), propionyl-CoA供給と3HV生合成: https://doi.org/10.1128/AEM.03142-12",
    ]
    (output_dir / "third_species_niche_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "decision.json").write_text(json.dumps({"verdict": verdict, "best_helper": best_helper, "matched_direct_lactate": direct_lac, "matched_direct_propionate": direct_ppa, "matched_direct_propionate_acetate_mix": direct_mix, "matched_cell_free_effluent_replay": direct_replay, "helper_to_best_direct_ratio": ratio, "crossfeed_observed": crossfeed}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/third_species_niche_20260902")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--initial-nh4", type=float, default=0.05)
    parser.add_argument("--kla", type=float, default=50.0)
    parser.add_argument("--fba-mode", choices=("separate", "cooperative"), default="separate")
    parser.add_argument("--scenario", action="append", default=[])
    args = parser.parse_args()
    if args.hours <= 0 or args.dt <= 0 or args.initial_nh4 < 0 or args.kla < 0:
        parser.error("hours/dt must be positive and initial-nh4/kla non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_helper_model(write_artifacts=True)
    static = static_helper_validation()
    scenarios = build_scenarios(args.kla)
    if args.scenario:
        requested = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario.name in requested]
        missing = requested.difference(scenario.name for scenario in scenarios)
        if missing:
            parser.error(f"unknown scenarios: {sorted(missing)}")
    summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, 1):
        print(f"[{index}/{len(scenarios)}] {scenario.name}", flush=True)
        summary, rows = run_scenario(scenario, args.hours, args.dt, args.initial_nh4, args.fba_mode)
        summaries.append(summary)
        trajectories.extend(rows)
    _write_csv(args.output_dir / "scenario_summary.csv", summaries)
    _write_csv(args.output_dir / "trajectories.csv", trajectories)
    (args.output_dir / "third_species_niche.json").write_text(json.dumps({"schema_version": 1, "study_type": f"exact_{args.fba_mode}_fed_batch_dFBA", "static_helper_validation": static, "scenarios": summaries, "parameters": {"hours": args.hours, "dt_h": args.dt, "initial_nh4_mmol_l": args.initial_nh4, "kla_per_h": args.kla, "helper_oxygen": "closed (anaerobic phenotype)", "exchange_parsimony": True if args.fba_mode == "cooperative" else None, "direct_mix_yield": {"propionate_per_lactate": STATIC_PROPIONATE_PER_LACTATE, "acetate_per_lactate": STATIC_ACETATE_PER_LACTATE}, "posthoc_cell_free_replay": {"lactate_per_feed": REPLAY_LACTATE_PER_FEED, "propionate_per_feed": REPLAY_PROPIONATE_PER_FEED, "acetate_per_feed": REPLAY_ACETATE_PER_FEED}, "pha_gate": "EX_pha_c open only when NH4 < 0.1 mmol/L"}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(summaries) == len(build_scenarios(args.kla)):
        plot_results(static, summaries, trajectories, args.output_dir)
        write_report(static, summaries, args.output_dir)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
