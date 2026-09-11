#!/usr/bin/env python3
"""Run exact-FBA virtual smoke tests for the physical 1 L jar wrapper."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dfba_simulator import dFBASimulator  # noqa: E402
from src.one_l_jar import OneLJarConfig, OneLJarProfile  # noqa: E402
from src.utils import get_initial_params, load_sbml_models, select_consortium_models  # noqa: E402


MODEL_DIR = ROOT / "models/sbml/final_consortium"
PUMP_NAMES = (
    "p1_or16_virtual",
    "p2_ns21_virtual",
    "p3_wcfs1_virtual",
    "p4_defined10_virtual",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    policy: str
    variant: str
    seed: int
    initial_volume_l: float
    initial_rubber_g_l: float
    biomass_scale_or16: float
    biomass_scale_ns21: float
    biomass_scale_wcfs1: float
    medium_scale: float
    pump_efficiency: float
    kla_multiplier: float
    polymer_multiplier: float


VARIANTS = {
    "nominal": dict(
        initial_volume_l=0.75,
        initial_rubber_g_l=100.0,
        biomass_scale_or16=1.0,
        biomass_scale_ns21=1.0,
        biomass_scale_wcfs1=1.0,
        medium_scale=1.0,
        pump_efficiency=1.0,
        kla_multiplier=1.0,
        polymer_multiplier=1.0,
    ),
    "low_activity": dict(
        initial_volume_l=0.75,
        initial_rubber_g_l=100.0,
        biomass_scale_or16=0.8,
        biomass_scale_ns21=0.8,
        biomass_scale_wcfs1=0.8,
        medium_scale=0.9,
        pump_efficiency=0.9,
        kla_multiplier=0.8,
        polymer_multiplier=0.7,
    ),
    "high_load": dict(
        initial_volume_l=0.70,
        initial_rubber_g_l=120.0,
        biomass_scale_or16=1.1,
        biomass_scale_ns21=1.1,
        biomass_scale_wcfs1=1.1,
        medium_scale=1.1,
        pump_efficiency=1.05,
        kla_multiplier=0.85,
        polymer_multiplier=0.8,
    ),
    "imbalanced": dict(
        initial_volume_l=0.80,
        initial_rubber_g_l=80.0,
        biomass_scale_or16=0.7,
        biomass_scale_ns21=0.8,
        biomass_scale_wcfs1=1.2,
        medium_scale=0.95,
        pump_efficiency=1.0,
        kla_multiplier=1.1,
        polymer_multiplier=1.2,
    ),
}
POLICIES = ("no_control", "fixed25", "budget_max_air", "two_phase")


def build_scenarios(seed: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for variant_index, (variant, parameters) in enumerate(VARIANTS.items()):
        for policy_index, policy in enumerate(POLICIES):
            scenarios.append(
                Scenario(
                    name=f"{policy}__{variant}",
                    policy=policy,
                    variant=variant,
                    seed=seed + 100 * variant_index + policy_index,
                    **parameters,
                )
            )
    nominal = next(row for row in scenarios if row.name == "budget_max_air__nominal")
    scenarios.append(
        Scenario(
            **{
                **asdict(nominal),
                "name": "budget_max_air__nominal_repeat",
            }
        )
    )
    return scenarios


def species_scale(name: str, scenario: Scenario) -> float:
    if "OR16" in name:
        return scenario.biomass_scale_or16
    if "NS21" in name:
        return scenario.biomass_scale_ns21
    return scenario.biomass_scale_wcfs1


def control(policy: str, time_h: float, pump_efficiency: float) -> tuple[dict, float, float]:
    flows = {name: 0.0 for name in PUMP_NAMES}
    if policy == "no_control":
        return flows, 250.0, 0.20
    if policy == "fixed25":
        flows = {name: 0.5 * pump_efficiency for name in PUMP_NAMES}
        return flows, 300.0, 0.30
    if policy == "budget_max_air":
        flows["p4_defined10_virtual"] = 0.5 * pump_efficiency
        return flows, 600.0, 0.80
    if policy == "two_phase":
        if time_h < 3.0:
            flows["p4_defined10_virtual"] = 0.8 * pump_efficiency
        return flows, 600.0, 0.80
    raise ValueError(policy)


def run_scenario(task: tuple[Scenario, float, float]) -> tuple[dict, list[dict]]:
    scenario, horizon_h, dt_h = task
    # Loading inside the worker keeps mutable COBRA bounds isolated between
    # scenarios; smoke-test fidelity is preferred over startup micro-optimizing.
    models = select_consortium_models(load_sbml_models(MODEL_DIR))
    biomass, medium = get_initial_params(models)
    rng = np.random.default_rng(scenario.seed)
    biomass = {
        name: value
        * species_scale(name, scenario)
        * float(rng.uniform(0.98, 1.02))
        for name, value in biomass.items()
    }
    medium = {
        name: (
            value * scenario.medium_scale * float(rng.uniform(0.99, 1.01))
            if value > 0.0
            else value
        )
        for name, value in medium.items()
    }
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=scenario.initial_rubber_g_l,
        volume=scenario.initial_volume_l,
        dt=dt_h,
        solver_backend="highs",
        fba_mode="cooperative",
        ph_control_target=6.5,
        cooperative_highs_method="highs-ds",
    )
    jar = OneLJarProfile(
        simulator,
        config=OneLJarConfig(
            initial_volume_l=scenario.initial_volume_l,
            vessel_capacity_l=1.0,
            minimum_operating_volume_l=0.55,
            base_stock_mol_l=2.0,
            acid_stock_mol_l=1.0,
            sample_events_h_ml=((3.0, 3.0), (6.0, 3.0)),
            rpm_max=800.0,
            airflow_l_min_max=2.0,
        ),
    )
    initial_biomass = {
        name: float(state.biomass) for name, state in simulator.state.species.items()
    }
    initial_rubber_g = scenario.initial_rubber_g_l * scenario.initial_volume_l
    polymer_rates = {
        name: value * scenario.polymer_multiplier
        for name, value in dFBASimulator.DEFAULT_POLYMER_RATES.items()
    }
    trajectory: list[dict] = []
    finite_state = True
    negative_concentrations = 0
    while simulator.state.time < horizon_h - 1e-12:
        flows, rpm, airflow = control(
            scenario.policy, simulator.state.time, scenario.pump_efficiency
        )
        state = jar.step(
            flows,
            rpm=rpm,
            airflow_l_min=airflow,
            rubber_degradation_rates=polymer_rates,
            kla_multiplier=scenario.kla_multiplier,
        )
        values = [
            jar.volume_l,
            state.rubber_concentration,
            *state.metabolites.values(),
            *(member.biomass for member in state.species.values()),
            *(member.pha_accumulated for member in state.species.values()),
        ]
        finite_state = finite_state and bool(np.all(np.isfinite(values)))
        negative_concentrations += sum(
            float(value) < -1e-12 for value in state.metabolites.values()
        )
        h_mM = max(1e-12, float(state.metabolites.get("h_e", 1e-4)))
        trajectory.append(
            {
                "scenario": scenario.name,
                "time_h": float(state.time),
                "volume_l": jar.volume_l,
                "ph": -math.log10(h_mM / 1000.0),
                "rubber_g_l": float(state.rubber_concentration),
                "total_pha_mmol_l": float(
                    sum(member.pha_accumulated for member in state.species.values())
                ),
                "minimum_biomass_g_l": float(
                    min(member.biomass for member in state.species.values())
                ),
                "kla_h": jar.records[-1]["kla_h"],
            }
        )
    diagnostics = simulator.get_solver_diagnostics()
    jar_diagnostics = jar.diagnostics()
    final_biomass = {
        name: float(state.biomass) for name, state in simulator.state.species.items()
    }
    biomass_ratios = {
        name: final_biomass[name] / initial_biomass[name] for name in final_biomass
    }
    final_rubber_g = float(state.rubber_concentration * jar.volume_l)
    degraded_rubber_g = (
        initial_rubber_g - final_rubber_g - jar.cumulative_sampled_rubber_g
    )
    final_ph = trajectory[-1]["ph"]
    local_checks = {
        "finite_state": finite_state,
        "no_negative_concentrations": negative_concentrations == 0,
        "solver_success": diagnostics["solve_success_rate"] >= 0.99,
        "three_species_present": min(final_biomass.values()) >= 0.01,
        "ph_in_range": 6.3 <= final_ph <= 6.7,
        "volume_in_range": (
            jar.config.minimum_operating_volume_l
            <= jar.volume_l
            <= jar.config.vessel_capacity_l
        ),
        "volume_balance": abs(jar.volume_balance_error_ml) <= 1e-6,
        "no_command_violation": jar.command_violations == 0,
        "no_capacity_violation": jar.capacity_violations == 0,
    }
    result = {
        **asdict(scenario),
        "horizon_h": horizon_h,
        "steps": simulator.current_step,
        "passed": all(local_checks.values()),
        "checks": local_checks,
        "final_volume_l": jar.volume_l,
        "volume_balance_error_ml": jar.volume_balance_error_ml,
        "nutrient_feed_ml": jar.cumulative_nutrient_feed_ml,
        "base_ml": jar.cumulative_base_ml,
        "acid_ml": jar.cumulative_acid_ml,
        "sample_ml": jar.cumulative_sample_ml,
        "rubber_degraded_g": degraded_rubber_g,
        "final_pha_mmol_l": float(
            sum(member.pha_accumulated for member in state.species.values())
        ),
        "minimum_final_initial_biomass_ratio": min(biomass_ratios.values()),
        "final_ph": final_ph,
        "solver_success_rate": diagnostics["solve_success_rate"],
        "mean_kla_h": float(np.mean([row["kla_h"] for row in trajectory])),
        "command_violations": jar.command_violations,
        "capacity_violations": jar.capacity_violations,
        "negative_concentrations": negative_concentrations,
        "final_biomass_g_l": final_biomass,
        "final_initial_biomass_ratio": biomass_ratios,
        "jar_diagnostics": jar_diagnostics,
    }
    return result, trajectory


def flatten_for_csv(row: dict) -> dict:
    return {
        key: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list))
        else value
        for key, value in row.items()
    }


def plot_results(rows: list[dict], output_dir: Path) -> None:
    mpl.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    colors = {
        "no_control": "#777777",
        "fixed25": "#E69F00",
        "budget_max_air": "#0072B2",
        "two_phase": "#009E73",
    }
    base_rows = [row for row in rows if not row["name"].endswith("repeat")]
    x = np.arange(len(base_rows))
    bar_colors = [colors[row["policy"]] for row in base_rows]
    metrics = (
        ("final_volume_l", "Final volume (L)"),
        ("solver_success_rate", "Solver success rate"),
        ("rubber_degraded_g", "Rubber degraded (g)"),
        ("minimum_final_initial_biomass_ratio", "Minimum biomass ratio"),
    )
    for axis, (metric, label) in zip(axes.flat, metrics):
        axis.bar(x, [row[metric] for row in base_rows], color=bar_colors)
        axis.set_ylabel(label)
        axis.set_xticks([])
        axis.spines[["top", "right"]].set_visible(False)
    for index, axis in enumerate(axes.flat):
        axis.set_title(chr(ord("a") + index), loc="left", fontweight="bold")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in POLICIES
    ]
    fig.legend(
        handles,
        ["No control", "Fixed-25", "Budget/max air", "Two-phase"],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Figure_one_l_jar_smoke.{suffix}", dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/one_l_jar_smoke"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon-h", type=float, default=6.0)
    parser.add_argument("--dt-h", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.horizon_h <= 0.0 or args.dt_h <= 0.0:
        raise ValueError("horizon and dt must be positive")
    if abs(args.horizon_h / args.dt_h - round(args.horizon_h / args.dt_h)) > 1e-9:
        raise ValueError("horizon must be an integer number of dt steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(args.seed)
    started = time.perf_counter()
    tasks = [(scenario, args.horizon_h, args.dt_h) for scenario in scenarios]
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        outputs = pool.map(run_scenario, tasks)
    rows = [item[0] for item in outputs]
    trajectories = [row for item in outputs for row in item[1]]
    nominal = next(row for row in rows if row["name"] == "budget_max_air__nominal")
    repeat = next(
        row for row in rows if row["name"] == "budget_max_air__nominal_repeat"
    )
    reproducibility_keys = (
        "final_volume_l",
        "rubber_degraded_g",
        "final_pha_mmol_l",
        "minimum_final_initial_biomass_ratio",
        "final_ph",
        "solver_success_rate",
    )
    reproducible = all(
        np.isclose(nominal[key], repeat[key], rtol=0.0, atol=1e-10)
        for key in reproducibility_keys
    )
    nonrepeat = [row for row in rows if not row["name"].endswith("repeat")]
    global_checks = {
        "all_scenarios_pass_local_checks": all(row["passed"] for row in rows),
        "deterministic_repeat": reproducible,
        "rubber_degradation_signal": max(
            row["rubber_degraded_g"] for row in nonrepeat
        )
        > 0.0,
        "feed_volume_signal": max(row["nutrient_feed_ml"] for row in nonrepeat)
        > 0.0,
        "aeration_signal": max(row["mean_kla_h"] for row in nonrepeat)
        > min(row["mean_kla_h"] for row in nonrepeat),
        "zero_command_violations": sum(
            row["command_violations"] for row in rows
        )
        == 0,
        "zero_capacity_violations": sum(
            row["capacity_violations"] for row in rows
        )
        == 0,
    }
    global_checks = {key: bool(value) for key, value in global_checks.items()}
    payload = {
        "schema_version": 1,
        "purpose": "implementation smoke test; not biological validation",
        "model_dir": str(MODEL_DIR),
        "scenario_count": len(rows),
        "independent_scenario_count": len(nonrepeat),
        "horizon_h": args.horizon_h,
        "dt_h": args.dt_h,
        "solver": "exact cooperative HiGHS dual simplex",
        "virtual_parameter_status": "uncalibrated provisional ranges",
        "passed": all(global_checks.values()),
        "global_checks": global_checks,
        "elapsed_seconds": time.perf_counter() - started,
        "scenarios": rows,
    }
    (args.output_dir / "smoke_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "scenario_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(flatten_for_csv(row) for row in rows)
    with (args.output_dir / "trajectories.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)
    plot_results(rows, args.output_dir)
    print(json.dumps({key: payload[key] for key in ("passed", "global_checks", "elapsed_seconds")}, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
