#!/usr/bin/env python3
"""Fine-grained feed selection for the repaired WCFS1 consortium.

The workflow separates inexpensive static sensitivity analysis from dynamic
confirmation.  It does not treat model predictions as biological replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coexistence_audit import SharedMediumCommunityLP  # noqa: E402
from src.dfba_simulator import dFBASimulator  # noqa: E402
from src.utils import (  # noqa: E402
    COEXISTENCE_DEFINED_FEED_MMOL_L_H,
    COEXISTENCE_DEFINED_MEDIUM_24H,
    get_initial_params,
    load_sbml_models,
    select_consortium_models,
)


MODEL_DIR = ROOT / "models/sbml/final_consortium"
TARGET_GROWTH_PER_H = 0.005
MOLECULAR_WEIGHT_G_MOL = {
    "glc__D_e": 180.16,
    "arg__L_e": 174.20,
    "trp__L_e": 204.23,
    "leu__L_e": 131.17,
    "glu__L_e": 147.13,
    "ile__L_e": 131.17,
    "pro__L_e": 115.13,
    "phe__L_e": 165.19,
    "gln__L_e": 146.14,
    "pydam_e": 168.20,
    "pydxn_e": 169.18,
    "val__L_e": 117.15,
    "met__L_e": 149.21,
    "cys__L_e": 121.16,
    "tyr__L_e": 181.19,
    "his__L_e": 155.15,
    "malt_e": 342.30,
}
DISPLAY = {
    "glc__D_e": "Glucose",
    "arg__L_e": "Arg",
    "trp__L_e": "Trp",
    "leu__L_e": "Leu",
    "glu__L_e": "Glu",
    "ile__L_e": "Ile",
    "pro__L_e": "Pro",
    "phe__L_e": "Phe",
    "gln__L_e": "Gln",
    "pydam_e": "Pyridoxamine",
    "pydxn_e": "Pyridoxine",
    "val__L_e": "Val",
    "met__L_e": "Met",
    "cys__L_e": "Cys",
    "tyr__L_e": "Tyr",
    "his__L_e": "His",
    "malt_e": "Maltose",
}
OPTIONAL_FEED = {
    "val__L_e": 0.0300,
    "malt_e": 0.02373214285714429,
    "tyr__L_e": 0.0140,
    "met__L_e": 0.0110,
    "his__L_e": 0.0085,
    "cys__L_e": 0.0055,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_name(value: str) -> str:
    if "OR16" in value:
        return "OR16"
    if "NS21" in value:
        return "NS21"
    if "Lactobacillus" in value or "Lactiplantibacillus" in value:
        return "WCFS1"
    return value


def base_models_and_medium():
    models = select_consortium_models(load_sbml_models(MODEL_DIR))
    biomass, medium = get_initial_params(models)
    for metabolite in COEXISTENCE_DEFINED_MEDIUM_24H:
        medium[metabolite] = 0.0
    return models, biomass, medium


def feed_mass_g_l(feed_rate: dict[str, float], hours: float) -> float:
    return float(
        sum(
            rate * hours * MOLECULAR_WEIGHT_G_MOL[metabolite] / 1000.0
            for metabolite, rate in feed_rate.items()
            if rate > 0
        )
    )


def static_row(label: str, feed: dict[str, float], common_growth: float) -> dict:
    return {
        "label": label,
        "feasible": bool(common_growth >= TARGET_GROWTH_PER_H),
        "common_growth_per_h": float(common_growth),
        "target_met": bool(common_growth >= TARGET_GROWTH_PER_H * 0.999),
        "feed_mass_g_l_24h": feed_mass_g_l(feed, 24.0),
        "component_count": sum(value > 0 for value in feed.values()),
    }


def run_static_screen() -> dict:
    models, biomass, medium = base_models_and_medium()
    solver = SharedMediumCommunityLP(
        models,
        medium,
        biomass,
        dt=0.5,
        oxygen_transfer_mmol_l_h=100.0,
        growth_threshold=TARGET_GROWTH_PER_H,
    )
    reference = dict(COEXISTENCE_DEFINED_FEED_MMOL_L_H)
    metabolites = list(reference)

    global_scales = np.unique(
        np.concatenate(
            (
                np.asarray([0.0]),
                np.geomspace(0.001, 0.1, 17),
                np.linspace(0.125, 1.0, 8),
                np.asarray([1.25, 1.5, 2.0]),
            )
        )
    )
    global_rows = []
    for scale in global_scales:
        feed = {key: value * float(scale) for key, value in reference.items()}
        row = static_row(f"D10 x {scale:g}", feed, solver.solve_common_growth(feed))
        row["scale"] = float(scale)
        global_rows.append(row)

    low, high = 0.0, 0.1
    for _ in range(24):
        midpoint = (low + high) / 2
        growth = solver.solve_common_growth(
            {key: value * midpoint for key, value in reference.items()}
        )
        if growth >= TARGET_GROWTH_PER_H:
            high = midpoint
        else:
            low = midpoint
    uniform_threshold = high

    component_factors = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0)
    component_rows = []
    for metabolite in metabolites:
        for factor in component_factors:
            feed = dict(reference)
            feed[metabolite] *= factor
            row = static_row(
                f"{DISPLAY[metabolite]} x {factor:g}",
                feed,
                solver.solve_common_growth(feed),
            )
            row.update({"metabolite": metabolite, "factor": factor})
            component_rows.append(row)

    single_rows = []
    for metabolite in metabolites:
        feed = dict(reference)
        feed[metabolite] = 0.0
        row = static_row(
            f"minus {DISPLAY[metabolite]}", feed, solver.solve_common_growth(feed)
        )
        row["removed"] = [metabolite]
        single_rows.append(row)

    pair_rows = []
    for first, second in itertools.combinations(metabolites, 2):
        feed = dict(reference)
        feed[first] = 0.0
        feed[second] = 0.0
        row = static_row(
            f"minus {DISPLAY[first]} + {DISPLAY[second]}",
            feed,
            solver.solve_common_growth(feed),
        )
        row["removed"] = [first, second]
        pair_rows.append(row)

    alternatives = []
    reference_result = solver.solve_common_growth(reference)
    alternatives.append(static_row("Defined-10", reference, reference_result))
    for metabolite, rate in OPTIONAL_FEED.items():
        feed = dict(reference)
        feed[metabolite] = rate
        alternatives.append(
            static_row(
                f"D10 + {DISPLAY[metabolite]}",
                feed,
                solver.solve_common_growth(feed),
            )
        )
    pyridoxine = dict(reference)
    pyridoxine["pydam_e"] = 0.0
    pyridoxine["pydxn_e"] = reference["pydam_e"]
    alternatives.append(
        static_row(
            "Pyridoxine replaces pyridoxamine",
            pyridoxine,
            solver.solve_common_growth(pyridoxine),
        )
    )

    required = ("ile__L_e", "pydam_e")
    optional = [metabolite for metabolite in metabolites if metabolite not in required]
    subset_rows = []
    for size in range(len(optional) + 1):
        for selected_optional in itertools.combinations(optional, size):
            selected = tuple((*required, *selected_optional))
            feed = {
                key: (value if key in selected else 0.0)
                for key, value in reference.items()
            }
            row = static_row(
                " + ".join(DISPLAY[key] for key in selected),
                feed,
                solver.solve_common_growth(feed),
            )
            row["metabolites"] = list(selected)
            subset_rows.append(row)
    feasible_subsets = [row for row in subset_rows if row["target_met"]]
    minimum_components = min(row["component_count"] for row in feasible_subsets)
    minimum_subsets = [
        row for row in feasible_subsets if row["component_count"] == minimum_components
    ]
    subset_candidates = []
    for row in minimum_subsets:
        selected = set(row["metabolites"])
        low, high = 0.0, 1.0
        for _ in range(24):
            midpoint = (low + high) / 2
            feed = {
                key: (value * midpoint if key in selected else 0.0)
                for key, value in reference.items()
            }
            if solver.solve_common_growth(feed) >= TARGET_GROWTH_PER_H:
                high = midpoint
            else:
                low = midpoint
        operating_scale = high * 2.0
        operating_feed = {
            key: (value * operating_scale if key in selected else 0.0)
            for key, value in reference.items()
        }
        subset_candidates.append(
            {
                **row,
                "threshold_scale": high,
                "operating_safety_factor": 2.0,
                "operating_scale": operating_scale,
                "operating_feed_mmol_l_h": {
                    key: value for key, value in operating_feed.items() if value > 0
                },
                "operating_mass_g_l_24h": feed_mass_g_l(operating_feed, 24.0),
            }
        )
    selected_subset = min(
        subset_candidates, key=lambda row: row["operating_mass_g_l_24h"]
    )

    return {
        "reference_feed_mmol_l_h": reference,
        "target_growth_per_h": TARGET_GROWTH_PER_H,
        "uniform_scale_threshold": uniform_threshold,
        "uniform_scale_safety_5x": uniform_threshold * 5.0,
        "global_scale_response": global_rows,
        "component_dose_response": component_rows,
        "single_dropout": single_rows,
        "pair_dropout": pair_rows,
        "alternatives": alternatives,
        "subset_screen": subset_rows,
        "minimum_component_count": minimum_components,
        "minimum_subset_candidates": subset_candidates,
        "selected_subset": selected_subset,
    }


def run_dynamic(label: str, feed: dict[str, float], hours: float, dt: float) -> dict:
    models, initial_biomass, medium = base_models_and_medium()
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=medium,
        initial_rubber=100.0,
        dt=dt,
        solver_backend="highs",
        fba_mode="cooperative",
        ph_control_target=6.5,
    )
    trajectory = []
    steps = int(np.ceil(hours / dt))
    for step in range(steps):
        for metabolite, rate in feed.items():
            simulator.state.metabolites[metabolite] = (
                simulator.state.metabolites.get(metabolite, 0.0) + rate * dt
            )
        state = simulator.step({}, {"coexistence_feed_rate": 0.0}, dynamic_kla=100.0)
        trajectory.append(
            {
                "time_h": float(state.time),
                "biomass_g_l": {
                    short_name(name): float(member.biomass)
                    for name, member in state.species.items()
                },
                "growth_per_h": {
                    short_name(name): float(member.growth_rate)
                    for name, member in state.species.items()
                },
                "rubber_g_l": float(state.rubber_concentration),
            }
        )
        if (step + 1) % max(1, int(6.0 / dt)) == 0:
            print(f"[{label}] {state.time:.1f}/{hours:.1f} h", flush=True)

    initial = {short_name(name): float(value) for name, value in initial_biomass.items()}
    final = trajectory[-1]["biomass_g_l"]
    ratios = {name: final[name] / initial[name] for name in final}
    lp_positive = np.mean(
        [row["growth_per_h"]["WCFS1"] > 1e-7 for row in trajectory]
    )
    ratio_values = np.asarray(list(ratios.values()), dtype=float)
    balance_cv = float(np.std(ratio_values) / max(np.mean(ratio_values), 1e-12))
    active = bool(min(ratios.values()) >= 0.90 and lp_positive >= 0.90)
    balanced = bool(active and balance_cv <= 0.15)
    return {
        "label": label,
        "hours": hours,
        "dt_h": dt,
        "feed_rate_mmol_l_h": feed,
        "feed_mass_g_l": feed_mass_g_l(feed, hours),
        "component_count": sum(value > 0 for value in feed.values()),
        "initial_biomass_g_l": initial,
        "final_biomass_g_l": final,
        "final_fraction_of_initial": ratios,
        "minimum_final_fraction": min(ratios.values()),
        "balance_cv": balance_cv,
        "lp_positive_growth_fraction": float(lp_positive),
        "active_coexistence": active,
        "balanced_coexistence": balanced,
        "trajectory": trajectory,
    }


def run_dynamic_screen(static: dict, screen_hours: float, confirm_hours: float, dt: float):
    reference = static["reference_feed_mmol_l_h"]
    threshold = static["uniform_scale_threshold"]
    raw_scales = [
        0.0,
        threshold,
        threshold * 2,
        threshold * 5,
        threshold * 10,
        0.05,
        0.10,
        0.25,
        0.50,
        1.0,
    ]
    scales = sorted({float(np.clip(value, 0.0, 1.0)) for value in raw_scales})
    screen = []
    for scale in scales:
        feed = {key: value * scale for key, value in reference.items()}
        result = run_dynamic(f"D10 x {scale:.5g}", feed, screen_hours, dt)
        result["uniform_scale"] = scale
        screen.append(result)

    feasible = [row for row in screen if row["balanced_coexistence"]]
    minimum_d10 = min(feasible, key=lambda row: row["feed_mass_g_l"]) if feasible else max(
        screen, key=lambda row: row["minimum_final_fraction"]
    )
    selected_feed = dict(static["selected_subset"]["operating_feed_mmol_l_h"])
    selected_metabolites = list(selected_feed)
    confirmation_specs = {
        "Selected 3-component feed": selected_feed,
        "Selected minus isoleucine": {
            key: (0.0 if key == "ile__L_e" else value)
            for key, value in selected_feed.items()
        },
        "Selected minus pyridoxamine": {
            key: (0.0 if key == "pydam_e" else value)
            for key, value in selected_feed.items()
        },
        "Selected minus glutamate": {
            key: (0.0 if key == "glu__L_e" else value)
            for key, value in selected_feed.items()
        },
        "Selected with pyridoxine": {
            **{key: value for key, value in selected_feed.items() if key != "pydam_e"},
            "pydxn_e": selected_feed["pydam_e"],
        },
        "Reference D10": dict(reference),
    }
    confirmation = [
        run_dynamic(label, feed, confirm_hours, dt)
        for label, feed in confirmation_specs.items()
    ]
    selected = next(
        row for row in confirmation if row["label"] == "Selected 3-component feed"
    )
    return {
        "screen_hours": screen_hours,
        "confirmation_hours": confirm_hours,
        "screen": screen,
        "minimum_balanced_d10_scale": minimum_d10["uniform_scale"],
        "selected_metabolites": selected_metabolites,
        "confirmation": confirmation,
        "selected": selected,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def plot_static(static: dict, output_dir: Path) -> None:
    reference = static["reference_feed_mmol_l_h"]
    metabolites = list(reference)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    colors = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73"}

    rows = static["global_scale_response"]
    positive = [row for row in rows if row["scale"] > 0]
    axes[0, 0].plot(
        [row["scale"] for row in positive],
        [row["common_growth_per_h"] for row in positive],
        marker="o",
        markersize=3,
        linewidth=1.2,
        color=colors["blue"],
    )
    axes[0, 0].axhline(TARGET_GROWTH_PER_H, color=colors["orange"], linestyle="--")
    axes[0, 0].axvline(static["uniform_scale_threshold"], color="#666666", linestyle=":")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Uniform Defined-10 scale")
    axes[0, 0].set_ylabel("Common growth rate (h$^{-1}$)")
    axes[0, 0].set_title("a", loc="left", fontweight="bold")

    single = static["single_dropout"]
    values = [row["common_growth_per_h"] for row in single]
    y = np.arange(len(single))
    axes[0, 1].barh(y, values, color=colors["green"])
    axes[0, 1].axvline(TARGET_GROWTH_PER_H, color=colors["orange"], linestyle="--")
    axes[0, 1].set_yticks(y, [DISPLAY[row["removed"][0]] for row in single])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Common growth after single dropout (h$^{-1}$)")
    axes[0, 1].set_title("b", loc="left", fontweight="bold")

    matrix = np.full((len(metabolites), len(metabolites)), np.nan)
    for index, row in enumerate(single):
        matrix[index, index] = row["common_growth_per_h"]
    lookup = {
        tuple(row["removed"]): row["common_growth_per_h"]
        for row in static["pair_dropout"]
    }
    for i, first in enumerate(metabolites):
        for j, second in enumerate(metabolites):
            if i < j:
                matrix[i, j] = matrix[j, i] = lookup[(first, second)]
    image = axes[1, 0].imshow(matrix, cmap="viridis", aspect="auto")
    labels = [DISPLAY[item] for item in metabolites]
    axes[1, 0].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[1, 0].set_yticks(range(len(labels)), labels)
    colorbar = fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    colorbar.set_label("Common growth rate (h$^{-1}$)")
    axes[1, 0].set_title("c", loc="left", fontweight="bold")

    alternatives = static["alternatives"]
    labels = [
        row["label"].replace("D10 + ", "+").replace(
            "Pyridoxine replaces pyridoxamine", "PydXn substitution"
        )
        for row in alternatives
    ]
    x = np.arange(len(alternatives))
    bars = axes[1, 1].bar(
        x,
        [row["common_growth_per_h"] for row in alternatives],
        color=[colors["blue"] if row["target_met"] else colors["orange"] for row in alternatives],
    )
    axes[1, 1].axhline(TARGET_GROWTH_PER_H, color="#555555", linestyle="--")
    axes[1, 1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1, 1].set_ylabel("Common growth rate (h$^{-1}$)")
    axes[1, 1].set_title("d", loc="left", fontweight="bold")
    for bar, row in zip(bars, alternatives):
        if not row["target_met"]:
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, 0.001, "fail", ha="center", fontsize=6)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"Figure_feed_selection_static.{suffix}", dpi=500)
    plt.close(fig)


def plot_dynamic(dynamic: dict, output_dir: Path) -> None:
    screen = dynamic["screen"]
    confirmation = dynamic["confirmation"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    colors = {"OR16": "#0072B2", "NS21": "#E69F00", "WCFS1": "#009E73"}

    positive_screen = [row for row in screen if row["uniform_scale"] > 0]
    scales = [row["uniform_scale"] for row in positive_screen]
    for species in ("OR16", "NS21", "WCFS1"):
        axes[0, 0].plot(
            scales,
            [row["final_fraction_of_initial"][species] for row in positive_screen],
            marker="o",
            markersize=3,
            label=species,
            color=colors[species],
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].axhline(1.0, color="#666666", linestyle="--")
    axes[0, 0].set_xlabel("Uniform Defined-10 scale")
    axes[0, 0].set_ylabel("Final / initial biomass")
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[0, 0].set_title("a", loc="left", fontweight="bold")

    axes[0, 1].plot(
        scales,
        [row["lp_positive_growth_fraction"] for row in positive_screen],
        marker="o",
        color=colors["WCFS1"],
        label="WCFS1 positive-growth steps",
    )
    axes[0, 1].plot(
        scales,
        [row["balance_cv"] for row in positive_screen],
        marker="s",
        color="#CC79A7",
        label="Final-ratio CV",
    )
    axes[0, 1].axhline(0.90, color="#666666", linestyle="--", linewidth=0.8)
    axes[0, 1].axhline(0.15, color="#999999", linestyle=":", linewidth=0.8)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Uniform Defined-10 scale")
    axes[0, 1].set_ylabel("Fraction or CV")
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[0, 1].set_title("b", loc="left", fontweight="bold")

    x = np.arange(len(confirmation))
    labels = [
        "Selected-3",
        "-Ile",
        "-PydAm",
        "-Glu",
        "PydXn",
        "Reference",
    ]
    width = 0.23
    for index, species in enumerate(("OR16", "NS21", "WCFS1")):
        axes[1, 0].bar(
            x + (index - 1) * width,
            [row["final_fraction_of_initial"][species] for row in confirmation],
            width,
            color=colors[species],
            label=species,
        )
    axes[1, 0].axhline(1.0, color="#666666", linestyle="--")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Final / initial biomass (24 h)")
    axes[1, 0].set_title("c", loc="left", fontweight="bold")

    selected = dynamic["selected"]
    for species, linestyle in zip(
        ("OR16", "NS21", "WCFS1"), ("-", "--", ":")
    ):
        axes[1, 1].plot(
            [row["time_h"] for row in selected["trajectory"]],
            [row["biomass_g_l"][species] for row in selected["trajectory"]],
            color=colors[species],
            linestyle=linestyle,
            label=species,
        )
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel("Biomass (g L$^{-1}$)")
    axes[1, 1].set_title("d", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"Figure_feed_selection_dynamic.{suffix}", dpi=500)
    plt.close(fig)


def plot_decision(static: dict, dynamic: dict, output_dir: Path) -> None:
    """Summarize the formulation decision without implying replication."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    blue, orange, green, purple = "#0072B2", "#E69F00", "#009E73", "#CC79A7"

    component_counts = sorted({row["component_count"] for row in static["subset_screen"]})
    tested = [
        sum(row["component_count"] == count for row in static["subset_screen"])
        for count in component_counts
    ]
    passed = [
        sum(
            row["component_count"] == count and row["target_met"]
            for row in static["subset_screen"]
        )
        for count in component_counts
    ]
    x = np.arange(len(component_counts))
    axes[0, 0].bar(x - 0.18, tested, 0.36, color="#B8B8B8", label="Tested")
    axes[0, 0].bar(x + 0.18, passed, 0.36, color=green, label="Target met")
    axes[0, 0].set_xticks(x, component_counts)
    axes[0, 0].set_xlabel("Number of feed components")
    axes[0, 0].set_ylabel("Number of formulations")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("a", loc="left", fontweight="bold")

    candidates = sorted(
        static["minimum_subset_candidates"],
        key=lambda row: row["operating_mass_g_l_24h"],
    )
    mass_labels = [
        "+".join(DISPLAY[item] for item in row["metabolites"])
        for row in candidates
    ] + ["Defined-10"]
    masses = [row["operating_mass_g_l_24h"] for row in candidates] + [
        feed_mass_g_l(static["reference_feed_mmol_l_h"], 24.0)
    ]
    bars = axes[0, 1].bar(
        np.arange(len(masses)), masses, color=[green, blue, "#777777"]
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(np.arange(len(masses)), mass_labels, rotation=25, ha="right")
    axes[0, 1].set_ylabel("Feed mass over 24 h (g L$^{-1}$)")
    axes[0, 1].set_title("b", loc="left", fontweight="bold")
    for bar, value in zip(bars, masses):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.18,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    selected_feed = static["selected_subset"]["operating_feed_mmol_l_h"]
    rate_labels = [DISPLAY[key] for key in selected_feed]
    rates = [selected_feed[key] for key in selected_feed]
    axes[1, 0].bar(np.arange(len(rates)), rates, color=[blue, orange, purple])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(np.arange(len(rates)), rate_labels)
    axes[1, 0].set_ylabel("Operating supply (mmol L$^{-1}$ h$^{-1}$)")
    axes[1, 0].set_title("c", loc="left", fontweight="bold")

    confirmation = dynamic["confirmation"]
    labels = ["Selected-3", "-Ile", "-PydAm", "-Glu", "PydXn", "Defined-10"]
    x = np.arange(len(confirmation))
    axes[1, 1].bar(
        x - 0.18,
        [row["minimum_final_fraction"] for row in confirmation],
        0.36,
        color=blue,
        label="Minimum final/initial biomass",
    )
    axes[1, 1].bar(
        x + 0.18,
        [row["lp_positive_growth_fraction"] for row in confirmation],
        0.36,
        color=green,
        label="WCFS1 positive-growth steps",
    )
    axes[1, 1].axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_ylim(0.0, 1.22)
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 1].set_title("d", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"Figure_feed_selection_decision.{suffix}", dpi=500)
    plt.close(fig)


def write_outputs(payload: dict, output_dir: Path) -> None:
    static = payload["static"]
    dynamic = payload["dynamic"]
    (output_dir / "feed_selection_fine.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "global_scale_response.csv",
        static["global_scale_response"],
        ["scale", "common_growth_per_h", "target_met", "feed_mass_g_l_24h"],
    )
    _write_csv(
        output_dir / "component_dose_response.csv",
        static["component_dose_response"],
        ["metabolite", "factor", "common_growth_per_h", "target_met", "feed_mass_g_l_24h"],
    )
    _write_csv(
        output_dir / "single_dropout.csv",
        static["single_dropout"],
        ["label", "removed", "common_growth_per_h", "target_met", "feed_mass_g_l_24h"],
    )
    _write_csv(
        output_dir / "pair_dropout.csv",
        static["pair_dropout"],
        ["label", "removed", "common_growth_per_h", "target_met", "feed_mass_g_l_24h"],
    )
    _write_csv(
        output_dir / "alternatives.csv",
        static["alternatives"],
        ["label", "common_growth_per_h", "target_met", "feed_mass_g_l_24h", "component_count"],
    )
    _write_csv(
        output_dir / "subset_screen.csv",
        static["subset_screen"],
        [
            "label",
            "metabolites",
            "component_count",
            "common_growth_per_h",
            "target_met",
            "feed_mass_g_l_24h",
        ],
    )
    _write_csv(
        output_dir / "dynamic_screen.csv",
        dynamic["screen"],
        [
            "label",
            "uniform_scale",
            "feed_mass_g_l",
            "minimum_final_fraction",
            "balance_cv",
            "lp_positive_growth_fraction",
            "active_coexistence",
            "balanced_coexistence",
            "final_fraction_of_initial",
        ],
    )
    _write_csv(
        output_dir / "selected_trajectory.csv",
        dynamic["selected"]["trajectory"],
        ["time_h", "biomass_g_l", "growth_per_h", "rubber_g_l"],
    )
    plot_static(static, output_dir)
    plot_dynamic(dynamic, output_dir)
    plot_decision(static, dynamic, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/feed_selection_wcfs1_fine"
    )
    parser.add_argument("--screen-hours", type=float, default=12.0)
    parser.add_argument("--confirm-hours", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static = run_static_screen()
    dynamic = run_dynamic_screen(static, args.screen_hours, args.confirm_hours, args.dt)
    payload = {
        "schema_version": 1,
        "model_provenance": {
            path.name: {"sha256": sha256(path)} for path in sorted(MODEL_DIR.glob("*.xml"))
        },
        "parameters": {
            "target_growth_per_h": TARGET_GROWTH_PER_H,
            "screen_hours": args.screen_hours,
            "confirmation_hours": args.confirm_hours,
            "dt_h": args.dt,
            "oxygen_transfer_mmol_l_h": 100.0,
            "ph_setpoint": 6.5,
        },
        "static": static,
        "dynamic": dynamic,
        "interpretation_limits": [
            "Each point is a deterministic model scenario, not a biological replicate.",
            "Cross-feeding secretion bounds and extracellular enzyme kinetics are not experimentally calibrated.",
            "The selected feed is a minimum experimental starting formulation, not a validated culture recipe.",
            "Pyridoxamine and pyridoxine equivalence must be tested experimentally.",
        ],
    }
    write_outputs(payload, args.output_dir)
    summary = {
        "static_uniform_threshold": static["uniform_scale_threshold"],
        "minimum_balanced_d10_scale": dynamic["minimum_balanced_d10_scale"],
        "selected_metabolites": dynamic["selected_metabolites"],
        "selected_feed_mass_g_l_24h": dynamic["selected"]["feed_mass_g_l"],
        "selected_final_fraction": dynamic["selected"]["final_fraction_of_initial"],
        "selected_lp_positive_growth_fraction": dynamic["selected"]["lp_positive_growth_fraction"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
