#!/usr/bin/env python3
"""Dynamic validation of the corrected low-cost WCFS1 coexistence strategy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dfba_simulator import dFBASimulator  # noqa: E402
from src.utils import (  # noqa: E402
    COEXISTENCE_DEFINED_FEED_MMOL_L_H,
    COEXISTENCE_DEFINED_MEDIUM_24H,
    get_initial_params,
    load_sbml_models,
    select_consortium_models,
)


MOLECULAR_WEIGHT_G_MOL = {
    "glc__D_e": 180.16,
    "arg__L_e": 174.20,
    "trp__L_e": 204.23,
    "leu__L_e": 131.17,
    "glu__L_e": 147.13,
    "ile__L_e": 131.17,
    "phe__L_e": 165.19,
    "pro__L_e": 115.13,
    "val__L_e": 117.15,
    "malt_e": 342.30,
    "tyr__L_e": 181.19,
    "met__L_e": 149.21,
    "his__L_e": 155.15,
    "cys__L_e": 121.16,
    "gln__L_e": 146.14,
    "pydam_e": 168.20,
}

# Single-WCFS1 continuous feed plus common pools and pyridoxamine. This is the
# non-cross-fed comparator for the repaired 2022 network.
COMPLETE_16_FEED = {
    "glc__D_e": 0.5000,
    "arg__L_e": 0.0500,
    "trp__L_e": 0.0500,
    "leu__L_e": 0.0500,
    "glu__L_e": 0.0500,
    "val__L_e": 0.0300,
    "malt_e": 0.02373214285714429,
    "ile__L_e": 0.0215,
    "pro__L_e": 0.0200,
    "phe__L_e": 0.0170,
    "tyr__L_e": 0.0140,
    "met__L_e": 0.0110,
    "his__L_e": 0.0085,
    "cys__L_e": 0.0055,
    "gln__L_e": 0.0030,
    "pydam_e": 0.0010,
}
COMPLETE_16_OVERFEED = {
    metabolite: 4.0 * rate for metabolite, rate in COMPLETE_16_FEED.items()
}

SCENARIOS = {
    "No feed + cooperative": {"mode": "cooperative", "defined_scale": 0.0},
    "Defined-10 + separate": {
        "mode": "separate",
        "defined_scale": 1.0,
    },
    "Defined-10 + cooperative": {
        "mode": "cooperative",
        "defined_scale": 1.0,
    },
    "Complete-16 (4x) + separate": {
        "mode": "separate",
        "defined_scale": 0.0,
        "direct_feed": COMPLETE_16_OVERFEED,
    },
}


def supplement_mass_g_l(supplement: dict[str, float]) -> float:
    return float(
        sum(
            concentration_mmol_l * MOLECULAR_WEIGHT_G_MOL[metabolite] / 1000.0
            for metabolite, concentration_mmol_l in supplement.items()
        )
    )


def short_name(value: str) -> str:
    if "OR16" in value:
        return "OR16"
    if "NS21" in value:
        return "NS21"
    if "Lactobacillus" in value or "Lactiplantibacillus" in value:
        return "LP"
    return value


def run_scenario(label: str, config: dict, hours: float, dt: float) -> dict:
    models = select_consortium_models(
        load_sbml_models(ROOT / "models/sbml/final_consortium")
    )
    initial_biomass, medium = get_initial_params(models)
    # get_initial_params includes the optimized reserve by default. Rebuild
    # the comparison from a common no-rescue base.
    for metabolite in COEXISTENCE_DEFINED_MEDIUM_24H:
        medium[metabolite] = 0.0
    direct_feed = dict(config.get("direct_feed", {}))
    defined_scale = float(config.get("defined_scale", 0.0))
    feed_rates = (
        {
            metabolite: rate * defined_scale
            for metabolite, rate in COEXISTENCE_DEFINED_FEED_MMOL_L_H.items()
        }
        if defined_scale > 0
        else direct_feed
    )

    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=medium,
        initial_rubber=100.0,
        dt=dt,
        solver_backend="highs",
        fba_mode=config["mode"],
        ph_control_target=6.5,
    )
    steps = int(np.ceil(hours / dt))
    trajectory = []
    crossfeeding = {
        metabolite: 0.0
        for metabolite in (
            "cys__L_e",
            "met__L_e",
            "val__L_e",
            "lys__L_e",
            "thr__L_e",
            "tyr__L_e",
        )
    }
    for step in range(steps):
        if direct_feed:
            for metabolite, rate in direct_feed.items():
                simulator.state.metabolites[metabolite] = (
                    simulator.state.metabolites.get(metabolite, 0.0) + rate * dt
                )
        before = {
            species: float(state.biomass)
            for species, state in simulator.state.species.items()
        }
        state = simulator.step(
            {},
            {"coexistence_feed_rate": defined_scale},
            dynamic_kla=100.0,
        )
        for species, species_state in state.species.items():
            for metabolite in crossfeeding:
                crossfeeding[metabolite] += (
                    float(species_state.metabolite_secretion.get(metabolite, 0.0))
                    * before[species]
                    * dt
                )
        trajectory.append(
            {
                "time_h": float(state.time),
                "biomass_g_l": {
                    short_name(species): float(species_state.biomass)
                    for species, species_state in state.species.items()
                },
                "growth_per_h": {
                    short_name(species): float(species_state.growth_rate)
                    for species, species_state in state.species.items()
                },
                "rubber_g_l": float(state.rubber_concentration),
                "ph": float(
                    -np.log10(max(1e-12, state.metabolites["h_e"]) / 1000.0)
                ),
            }
        )
        if (step + 1) % max(1, int(6.0 / dt)) == 0:
            print(f"[{label}] {state.time:.1f}/{hours:.1f} h", flush=True)

    initial_short = {short_name(name): float(value) for name, value in initial_biomass.items()}
    final = trajectory[-1]["biomass_g_l"]
    minimum_fraction = min(final[name] / initial_short[name] for name in final)
    positive_lp_steps = sum(
        row["growth_per_h"]["LP"] > 1e-7 for row in trajectory
    )
    diagnostics = simulator.get_solver_diagnostics()
    return {
        "scenario": label,
        "fba_mode": config["mode"],
        "feed_rate_mmol_l_h": feed_rates,
        "supplement_mmol_l": {
            metabolite: rate * hours for metabolite, rate in feed_rates.items()
        },
        "supplement_component_count": len(feed_rates),
        "supplement_mass_g_l": supplement_mass_g_l(
            {metabolite: rate * hours for metabolite, rate in feed_rates.items()}
        ),
        "initial_biomass_g_l": initial_short,
        "final_biomass_g_l": final,
        "final_fraction_of_initial": {
            name: final[name] / initial_short[name] for name in final
        },
        "minimum_final_fraction": minimum_fraction,
        "all_members_persist": bool(minimum_fraction >= 0.90),
        "lp_positive_growth_fraction": positive_lp_steps / len(trajectory),
        "active_coexistence": bool(
            minimum_fraction >= 0.90 and positive_lp_steps / len(trajectory) >= 0.90
        ),
        "crossfeeding_mmol_l": crossfeeding,
        "solver_diagnostics": diagnostics,
        "trajectory": trajectory,
    }


def plot(results: list[dict], output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "No feed + cooperative": "#999999",
        "Defined-10 + separate": "#D55E00",
        "Defined-10 + cooperative": "#009E73",
        "Complete-16 (4x) + separate": "#0072B2",
    }
    species_colors = {"OR16": "#0072B2", "NS21": "#E69F00", "LP": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    for result in results:
        axes[0, 0].plot(
            [row["time_h"] for row in result["trajectory"]],
            [row["biomass_g_l"]["LP"] for row in result["trajectory"]],
            color=colors[result["scenario"]],
            label=result["scenario"],
            linewidth=1.6,
        )
    axes[0, 0].axhline(0.09, color="#666666", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xlabel("Time (h)")
    axes[0, 0].set_ylabel(r"LP biomass (g L$^{-1}$)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    axes[0, 0].set_title("a", loc="left", fontweight="bold")

    labels = [result["scenario"] for result in results]
    x = np.arange(len(labels))
    optimized = next(r for r in results if r["scenario"] == "Defined-10 + cooperative")
    width = 0.23
    for species_index, species in enumerate(("OR16", "NS21", "LP")):
        axes[0, 1].bar(
            x + (species_index - 1) * width,
            [result["final_fraction_of_initial"][species] for result in results],
            width,
            color=species_colors[species],
            label=species,
        )
    axes[0, 1].axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    short_labels = ["No feed", "D10 sep.", "D10 coop.", "C16×4 sep."]
    axes[0, 1].set_xticks(x, short_labels)
    axes[0, 1].set_ylabel("Final / initial biomass")
    axes[0, 1].legend(frameon=False, ncol=3, fontsize=7)
    axes[0, 1].set_title("b", loc="left", fontweight="bold")

    bars = axes[1, 0].bar(
        x,
        [result["supplement_mass_g_l"] for result in results],
        color=[colors[label] for label in labels],
    )
    axes[1, 0].set_xticks(x, short_labels)
    axes[1, 0].set_ylabel(r"Defined supplement (g L$^{-1}$)")
    for bar, result in zip(bars, results):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{result['supplement_component_count']} solutes",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    axes[1, 0].set_title("c", loc="left", fontweight="bold")

    metabolites = [
        key for key, value in optimized["crossfeeding_mmol_l"].items() if value > 1e-7
    ]
    metabolite_labels = {
        "cys__L_e": "Cys",
        "met__L_e": "Met",
        "val__L_e": "Val",
        "lys__L_e": "Lys",
        "thr__L_e": "Thr",
        "tyr__L_e": "Tyr",
    }
    axes[1, 1].barh(
        [metabolite_labels[key] for key in metabolites],
        [optimized["crossfeeding_mmol_l"][key] for key in metabolites],
        color="#56B4E9",
    )
    axes[1, 1].set_xlabel(r"Predicted secretion (mmol L$^{-1}$, 24 h)")
    axes[1, 1].set_title("d", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"wcfs1_coexistence_strategy.{suffix}", dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/wcfs1_coexistence_strategy",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from the existing JSON without rerunning dFBA",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "wcfs1_coexistence_strategy.json"
    if args.plot_only:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        results = payload["results"]
    else:
        results = [
            run_scenario(label, config, args.hours, args.dt)
            for label, config in SCENARIOS.items()
        ]
        payload = {
            "parameters": {
                "hours": args.hours,
                "dt_h": args.dt,
                "ph_setpoint": 6.5,
                "maintenance_growth_target_per_h": 0.005,
            },
            "results": results,
            "interpretation_limits": [
                "Predicted cross-feeding is a cooperative LP design hypothesis, not an experimentally measured secretion rate.",
                "The ten-solute formulation requires monoculture and consortium dropout validation before wet-lab scale-up.",
                "Reagent mass is reported; purchase price and solution stability are not represented in the LP objective.",
            ],
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    plot(results, args.output_dir)
    summary = [
        {
            key: result[key]
            for key in (
                "scenario",
                "fba_mode",
                "supplement_component_count",
                "supplement_mass_g_l",
                "final_biomass_g_l",
                "minimum_final_fraction",
                "all_members_persist",
                "lp_positive_growth_fraction",
                "active_coexistence",
            )
        }
        for result in results
    ]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
