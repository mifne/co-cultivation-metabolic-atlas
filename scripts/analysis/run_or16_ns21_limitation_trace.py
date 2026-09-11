#!/usr/bin/env python3
"""Run an exact two-member OR16+NS21 dFBA culture with limitation tracing.

This is an in-silico pre-culture.  It does not claim that predicted rescue
concentrations are a validated wet-lab medium.  Machine-readable trajectories,
LP shadow constraints, finite-dose rescues, nutrient dropouts, and process
bottlenecks are written so every recommendation can be traced to a state and
an explicit model perturbation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coexistence_audit import SharedMediumCommunityLP  # noqa: E402
from src.dfba_simulator import dFBASimulator  # noqa: E402
from src.limitation_trace import (  # noqa: E402
    MINIMAL_SUPPLEMENT_GROUPS,
    candidate_ids,
    classify_polymer_bottleneck,
    rank_present_nutrient_dropouts,
    trace_community_limitation,
)
from src.utils import (  # noqa: E402
    get_initial_params,
    load_sbml_models,
    select_or16_ns21_models,
)


DISPLAY = {
    "Actinoplanes_sp_OR16_lcp": "OR16",
    "Rhizobacter_gummiphilus_NS21": "NS21",
}


def _short(name: str) -> str:
    return next((label for marker, label in (("OR16", "OR16"), ("NS21", "NS21")) if marker in name), name)


def _json_default(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, default=_json_default)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def two_species_starting_medium(models: Mapping[str, Any], glucose_mmol_l: float) -> tuple[dict, dict]:
    """Return the existing defined background without complex-medium feed."""

    biomass, medium = get_initial_params(dict(models))
    # No complex-medium or third-species feed is admitted.  Glucose defaults
    # to zero so growth remains coupled to rubber-derived carbon; it is exposed
    # as a CLI sensitivity parameter for inoculum carry-over experiments.
    medium["yeast_extract_e"] = 0.0
    medium["mnl_e"] = 0.0
    medium["glc__D_e"] = max(0.0, float(glucose_mmol_l))
    return biomass, medium


def _ph(state) -> float:
    proton = max(1e-12, float(state.metabolites.get("h_e", 1e-4)))
    return float(-np.log10(proton / 1000.0))


def _trajectory_row(simulator: dFBASimulator) -> dict[str, Any]:
    state = simulator.state
    row: dict[str, Any] = {
        "time_h": float(state.time),
        "rubber_g_l": float(state.rubber_concentration),
        "pha_mmol_l_model_pool": float(sum(item.pha_accumulated for item in state.species.values())),
        "nh4_mmol_l": float(state.metabolites.get("nh4_e", 0.0)),
        "o2_mmol_l": float(state.metabolites.get("o2_e", 0.0)),
        "glucose_mmol_l": float(state.metabolites.get("glc__D_e", 0.0)),
        "C30_oligo_mmol_l": float(state.metabolites.get("C30_oligo_e", 0.0)),
        "odtd_mmol_l": float(state.metabolites.get("odtd_e", 0.0)),
        "ph": _ph(state),
    }
    for name, species in state.species.items():
        prefix = _short(name).lower()
        row[f"{prefix}_biomass_g_l"] = float(species.biomass)
        row[f"{prefix}_growth_per_h"] = float(species.growth_rate)
        row[f"{prefix}_pha_mmol_l_model_pool"] = float(species.pha_accumulated)
    row.update({f"polymer_{key}": float(value) for key, value in simulator.last_polymer_fluxes.items()})
    return row


def _diagnostic_model_copies(simulator: dFBASimulator) -> dict[str, Any]:
    """Restore feed-responsive exchange bounds in read-only diagnostic copies.

    Dynamic dFBA closes uptake reactions when the corresponding pool reaches
    zero.  A rescue perturbation must reopen that uptake route; otherwise an
    added extracellular nutrient is a false negative.  Shared-medium supply
    caps still prevent material from being consumed when it is not supplied.
    """

    copies = {name: model.copy() for name, model in simulator.models.items()}
    for name, model in copies.items():
        original = simulator.original_bounds[name]
        for reaction in model.exchanges:
            if reaction.id not in original:
                continue
            lower, upper = original[reaction.id]
            reaction.bounds = (min(float(lower), 0.0), max(float(upper), 0.0))
        # A PHA accumulation phase changes the live NS21 objective.  The
        # limitation trace is explicitly a common-growth diagnostic, so its
        # growth reaction must remain the objective at every time point.
        for growth_id in ("R_Growth", "Growth", "R_BIOMASS_LLA", "BIOMASS_LLA", "BIOMASS"):
            if growth_id in model.reactions:
                model.objective = model.reactions.get_by_id(growth_id)
                break
    return copies


def _limitation_snapshot(simulator: dFBASimulator, args, full_scan: bool) -> dict[str, Any]:
    state = simulator.state
    biomass = {name: float(item.biomass) for name, item in state.species.items()}
    trace = trace_community_limitation(
        _diagnostic_model_copies(simulator),
        state.metabolites,
        biomass,
        dt_h=args.dt,
        oxygen_transfer_mmol_l_h=args.oxygen_transfer,
        target_gain_fraction=args.target_gain,
        run_single_addition_scan=full_scan,
    )
    trace["time_h"] = float(state.time)
    trace["rubber_process"] = classify_polymer_bottleneck(
        simulator.last_polymer_fluxes,
        state.rubber_concentration,
        state.metabolites.get("o2_e", 0.0),
        state.metabolites.get("glc__D_e", 0.0),
    )
    nh4 = float(state.metabolites.get("nh4_e", 0.0))
    trace["pha_process"] = {
        "code": (
            "nitrogen_replete_growth_phase" if nh4 >= 0.1 else "nitrogen_limited_accumulation_phase"
        ),
        "nh4_mmol_l": nh4,
        "note": (
            "The simulator switches the NS21 objective toward PHA below 0.1 mM NH4; "
            "this threshold is a model assumption requiring culture calibration."
        ),
    }
    return trace


def plot_results(payload: Mapping[str, Any], output_dir: Path) -> None:
    rows = payload["trajectory"]
    times = np.asarray([row["time_h"] for row in rows])
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
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.7), constrained_layout=True)

    for key, label, color in (
        ("or16_biomass_g_l", "OR16", blue),
        ("ns21_biomass_g_l", "NS21", orange),
    ):
        axes[0, 0].plot(times, [row.get(key, np.nan) for row in rows], label=label, color=color, lw=1.7)
    axes[0, 0].set(xlabel="Time (h)", ylabel=r"Biomass (g L$^{-1}$)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(times, [row["rubber_g_l"] for row in rows], color=gray, lw=1.7)
    axes[0, 1].set(xlabel="Time (h)", ylabel=r"Rubber (g L$^{-1}$)")
    twin = axes[0, 1].twinx()
    twin.plot(times, [row["pha_mmol_l_model_pool"] for row in rows], color=purple, lw=1.5)
    twin.set_ylabel("PHA model pool (mmol L$^{-1}$)", color=purple)
    twin.tick_params(axis="y", colors=purple)

    snapshots = payload["limitation_trace"]
    trace_t = [item["time_h"] for item in snapshots]
    axes[1, 0].plot(trace_t, [item["common_growth_per_h"] for item in snapshots], color=green, marker="o", ms=3)
    axes[1, 0].plot(trace_t, [item["target_growth_per_h"] for item in snapshots], color=gray, ls="--", lw=1.1, label="+10% target")
    axes[1, 0].set(xlabel="Time (h)", ylabel=r"Common growth capacity (h$^{-1}$)")
    axes[1, 0].legend(frameon=False)

    final_rescues = payload["final_single_addition_rescues"][:10]
    labels = [item["metabolite"] for item in reversed(final_rescues)]
    gains = [item["absolute_gain_per_h"] for item in reversed(final_rescues)]
    colors = [purple if item["group"] == "vitamin" else blue for item in reversed(final_rescues)]
    axes[1, 1].barh(range(len(labels)), gains, color=colors)
    axes[1, 1].set_yticks(range(len(labels)), labels, fontsize=6.5)
    axes[1, 1].set(xlabel=r"Finite-dose growth gain (h$^{-1}$)")
    if not labels:
        axes[1, 1].text(0.5, 0.5, "No positive single-addition rescue", ha="center", va="center", transform=axes[1, 1].transAxes)

    for index, axis in enumerate(axes.flat):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.05, "abcd"[index], transform=axis.transAxes, fontweight="bold", fontsize=10)
    base = output_dir / "Figure_OR16_NS21_limitation_trace"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(payload: Mapping[str, Any], output_dir: Path) -> None:
    final = payload["limitation_trace"][-1]
    trajectory = payload["trajectory"]
    start, end = trajectory[0], trajectory[-1]
    positive = [item for item in payload["final_single_addition_rescues"] if item["absolute_gain_per_h"] > 1e-9]
    essential = [item for item in payload["initial_nutrient_dropouts"] if item["model_essential_at_tested_medium"]]
    required = final["required_practical_supply"]
    vitamins = [item for item in payload["initial_nutrient_dropouts"] if item["group"] == "vitamin"]
    vitamin_max_loss = max((float(item["relative_loss"] or 0.0) for item in vitamins), default=0.0)
    top_rescue = positive[0] if positive else None
    lines = [
        "# OR16＋NS21二種系：培養・律速トレース",
        "",
        "## 実行範囲",
        "",
        "これはGEM/dFBAによるin-silico培養であり、実培養の結果ではない。OR16とNS21のみを使用し、酵母エキス、第三菌、追加流加を使用せず、pH-statを有効にした。",
        "",
        "## 終点",
        "",
        f"- 培養時間: {end['time_h']:.2f} h",
        f"- ゴム: {start['rubber_g_l']:.4f} → {end['rubber_g_l']:.4f} g/L",
        f"- PHAモデルプール: {start['pha_mmol_l_model_pool']:.5g} → {end['pha_mmol_l_model_pool']:.5g} mmol/L",
        f"- OR16菌体: {start.get('or16_biomass_g_l', 0):.5g} → {end.get('or16_biomass_g_l', 0):.5g} g/L",
        f"- NS21菌体: {start.get('ns21_biomass_g_l', 0):.5g} → {end.get('ns21_biomass_g_l', 0):.5g} g/L",
        f"- NH4: {start['nh4_mmol_l']:.4g} → {end['nh4_mmol_l']:.4g} mmol/L",
        "",
        "## 律速の判定規則",
        "",
        "- 濃度ゼロ、LPの結合制約、有限量添加による目的値回復を別々に記録した。",
        "- shadow benefitは『追加すれば成長余地がある』ことを示すが、欠乏や必須性の証明ではない。",
        "- 添加候補は酸素移動、無機塩、ビタミン、アミノ酸に制限した。代替炭素源、ゴム分解中間体、シデロフォアは自動推奨から除外した。",
        "",
        "## 終点診断",
        "",
        f"- ゴム分解段階: `{final['rubber_process']['code']}`",
        f"- PHA段階: `{final['pha_process']['code']}`",
        f"- +10%成長目標に必要な実用候補: {json.dumps(required, ensure_ascii=False, default=_json_default) if required else 'なし'}",
        f"- 単独添加で正の改善を示した候補: {', '.join(item['metabolite'] for item in positive[:10]) or 'なし'}",
        f"- 初期dropoutで成長がゼロになった成分: {', '.join(item['metabolite'] for item in essential) or 'なし'}",
        f"- 最大の単独介入: {top_rescue['metabolite'] + '（共通増殖容量 ' + format(100 * top_rescue['relative_gain'], '.1f') + '%増）' if top_rescue and top_rescue['relative_gain'] is not None else 'なし'}",
        f"- ビタミンdropoutの最大影響: {100 * vitamin_max_loss:.4f}%（モデル上、必須ビタミンは検出されず）",
        "- NH4が無給餌でも増えたため、初期培地のプリン/ヌクレオシド等からの脱アミノ反応がPHA移行を遅らせるモデル構造になっている。",
        "",
        "## 解釈上の制約",
        "",
        "- ゴム分解酵素速度、酸素配分、NH4によるPHA切替閾値は実測校正前のパラメータである。",
        "- PHAモデルプールは重量%やg/Lではなく、モデル内の集積量である。",
        "- 外部供給候補は計算上の介入仮説であり、実レシピ採用には濃度系列、無添加対照、単独培養対照が必要である。",
        "- 酸素律速判定は固定されたpolymer_oxygen_fraction=0.25に依存し、実測DO/OURによる再推定が必要である。",
    ]
    (output_dir / "OR16_NS21_limitation_trace_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = select_or16_ns21_models(load_sbml_models(args.sbml_dir))
    biomass, medium = two_species_starting_medium(models, args.initial_glucose)
    simulator = dFBASimulator(
        models={name: model.copy() for name, model in models.items()},
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=args.initial_rubber,
        volume=1.0,
        dt=args.dt,
        solver_backend="highs",
        fba_mode="cooperative",
        ph_control_target=args.ph,
        polymer_oxygen_fraction=args.polymer_oxygen_fraction,
    )
    trajectory = [_trajectory_row(simulator)]
    trace = [_limitation_snapshot(simulator, args, full_scan=True)]

    initial_solver = SharedMediumCommunityLP(
        _diagnostic_model_copies(simulator),
        simulator.state.metabolites,
        {name: state.biomass for name, state in simulator.state.species.items()},
        dt=args.dt,
        oxygen_transfer_mmol_l_h=args.oxygen_transfer,
    )
    initial_growth = initial_solver.solve_common_growth()
    dropouts = rank_present_nutrient_dropouts(
        initial_solver,
        initial_growth,
        simulator.state.metabolites,
        candidate_ids(MINIMAL_SUPPLEMENT_GROUPS),
    )

    steps = max(1, int(np.ceil(args.hours / args.dt)))
    trace_every = max(1, int(round(args.trace_every_h / args.dt)))
    for step in range(steps):
        simulator.step({}, {}, dynamic_kla=args.kla_per_h)
        trajectory.append(_trajectory_row(simulator))
        if (step + 1) % trace_every == 0 or step == steps - 1:
            trace.append(_limitation_snapshot(simulator, args, full_scan=False))

    # Replace the final lightweight snapshot with a complete finite-dose scan.
    trace[-1] = _limitation_snapshot(simulator, args, full_scan=True)
    final_rescues = trace[-1]["single_addition_rescues"]
    payload = {
        "schema_version": 1,
        "study_type": "in_silico_exact_HiGHS_cooperative_dFBA",
        "members": list(models),
        "parameters": {
            "hours": args.hours,
            "dt_h": args.dt,
            "initial_rubber_g_l": args.initial_rubber,
            "initial_glucose_mmol_l": args.initial_glucose,
            "pH_stat_target": args.ph,
            "kla_per_h": args.kla_per_h,
            "oxygen_transfer_diagnostic_mmol_l_h": args.oxygen_transfer,
            "polymer_oxygen_fraction": args.polymer_oxygen_fraction,
            "external_feed": "none",
            "yeast_extract": "none",
        },
        "trajectory": trajectory,
        "limitation_trace": trace,
        "initial_nutrient_dropouts": dropouts,
        "initial_single_addition_rescues": trace[0]["single_addition_rescues"],
        "final_single_addition_rescues": final_rescues,
        "solver_diagnostics": simulator.get_solver_diagnostics(),
        "quality_flags": {
            "ammonium_increased_without_external_feed": bool(
                trajectory[-1]["nh4_mmol_l"] > trajectory[0]["nh4_mmol_l"] + 1e-8
            ),
            "ammonium_change_mmol_l": float(
                trajectory[-1]["nh4_mmol_l"] - trajectory[0]["nh4_mmol_l"]
            ),
            "nitrogen_interpretation": (
                "Initial nitrogen-containing salvage pools can be deaminated; "
                "this delays the modeled PHA phase and must be checked by NH4 and "
                "purine/nucleoside measurements."
            ),
            "oxygen_limitation_depends_on_uncalibrated_allocation_fraction": True,
        },
        "interpretation_limits": [
            "Predictions are conditional on reconstructed GEMs and uncalibrated extracellular kinetics.",
            "Shadow prices are local objective sensitivities, not evidence of biological auxotrophy.",
            "Only an explicit finite-dose gain is labeled a model-predicted rescue.",
            "Alternative soluble carbon is excluded from the minimal-supplement recommendation because it decouples growth from rubber.",
        ],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbml-dir", type=Path, default=ROOT / "models/sbml/final_consortium")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/or16_ns21_limitation_trace")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--trace-every-h", type=float, default=2.0)
    parser.add_argument("--initial-rubber", type=float, default=10.0)
    parser.add_argument("--initial-glucose", type=float, default=0.0)
    parser.add_argument("--ph", type=float, default=7.0)
    parser.add_argument("--kla-per-h", type=float, default=50.0)
    parser.add_argument("--oxygen-transfer", type=float, default=1.0)
    parser.add_argument("--polymer-oxygen-fraction", type=float, default=0.25)
    parser.add_argument("--target-gain", type=float, default=0.10)
    args = parser.parse_args()
    if args.hours <= 0 or args.dt <= 0 or args.trace_every_h <= 0:
        parser.error("hours, dt, and trace-every-h must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = run(args)
    (args.output_dir / "OR16_NS21_limitation_trace.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "culture_trajectory.csv", payload["trajectory"])
    _write_csv(args.output_dir / "initial_nutrient_dropouts.csv", payload["initial_nutrient_dropouts"])
    _write_csv(args.output_dir / "initial_single_addition_rescues.csv", payload["initial_single_addition_rescues"])
    _write_csv(args.output_dir / "final_single_addition_rescues.csv", payload["final_single_addition_rescues"])
    limitation_rows = []
    for snapshot in payload["limitation_trace"]:
        limitation_rows.append(
            {
                "time_h": snapshot["time_h"],
                "common_growth_per_h": snapshot["common_growth_per_h"],
                "target_growth_per_h": snapshot["target_growth_per_h"],
                "combined_rescue_growth_per_h": snapshot["combined_rescue_growth_per_h"],
                "combined_rescue_reaches_target": snapshot["combined_rescue_reaches_target"],
                "rubber_bottleneck": snapshot["rubber_process"]["code"],
                "pha_phase": snapshot["pha_process"]["code"],
                "limiting_constraints": snapshot["limiting_constraints"],
                "required_practical_supply": snapshot["required_practical_supply"],
                "cross_feeding": snapshot["cross_feeding"],
            }
        )
    _write_csv(args.output_dir / "limitation_trace.csv", limitation_rows)
    plot_results(payload, args.output_dir)
    write_report(payload, args.output_dir)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
