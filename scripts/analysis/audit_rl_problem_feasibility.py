#!/usr/bin/env python3
"""Pre-training feasibility gate for the three-member dFBA-RL problem.

This audit asks whether the biological target is reachable with the controls
actually exposed to the RL agent. It is intentionally independent of PPO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coexistence_audit import (
    SharedMediumCommunityLP,
    apply_ideal_ph_control,
    find_growth_reaction,
)
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.utils import get_initial_params, load_sbml_models, select_consortium_models


YEAST_EXTRACT_COMPOSITION = {
    "glc__D_e": 1.0,
    "nh4_e": 0.5,
    "arg__L_e": 0.1,
    "trp__L_e": 0.05,
    "leu__L_e": 0.1,
    "ile__L_e": 0.1,
    "val__L_e": 0.1,
    "lys__L_e": 0.1,
    "met__L_e": 0.05,
    "phe__L_e": 0.05,
    "his__L_e": 0.05,
    "tyr__L_e": 0.05,
    "thr__L_e": 0.1,
    "cys__L_e": 0.05,
    "ala__L_e": 0.1,
    "asp__L_e": 0.1,
    "glu__L_e": 0.1,
    "gly_e": 0.1,
    "pro__L_e": 0.1,
    "ser__L_e": 0.1,
    "asn__L_e": 0.1,
    "gln__L_e": 0.1,
    "nac_e": 0.01,
    "ribflv_e": 0.01,
    "pnto__R_e": 0.01,
    "thm_e": 0.01,
    "btn_e": 0.001,
    "4abz_e": 0.01,
    "fol_e": 0.001,
    "nicnt_e": 0.01,
    "ade_e": 0.01,
    "gua_e": 0.01,
    "ura_e": 0.01,
    "cytd_e": 0.01,
}

ACTION_SEMANTICS = [
    {"index": 0, "name": "OR16-specific feed", "metabolites": ["mlttr_e"]},
    {"index": 1, "name": "NS21-specific feed", "metabolites": ["ptrc_e"]},
    {"index": 2, "name": "LP-specific feed", "metabolites": ["mnl_e"]},
    {
        "index": 3,
        "name": "yeast extract",
        "metabolites": sorted(YEAST_EXTRACT_COMPOSITION),
    },
    {"index": 4, "name": "KLa", "metabolites": ["o2_e"]},
]


def short_name(name: str) -> str:
    if "OR16" in name:
        return "OR16"
    if "NS21" in name:
        return "NS21"
    if "Lactobacillus" in name:
        return "LP"
    return name


def maximum_action_medium(initial: dict[str, float]) -> dict[str, float]:
    """Return medium after one maximum early-phase RL action."""

    medium = dict(initial)
    for metabolite in ("mlttr_e", "ptrc_e", "mnl_e"):
        medium[metabolite] = medium.get(metabolite, 0.0) + 0.1
    for metabolite, coefficient in YEAST_EXTRACT_COMPOSITION.items():
        medium[metabolite] = medium.get(metabolite, 0.0) + 0.5 * coefficient
    return medium


def static_reachability(models, dt: float) -> dict:
    biomass, initial = get_initial_params(models)
    baseline_solver = SharedMediumCommunityLP(models, initial, biomass, dt=dt)
    baseline = baseline_solver.solve()
    maximum_medium = maximum_action_medium(initial)
    maximum = SharedMediumCommunityLP(models, maximum_medium, biomass, dt=dt).solve()
    rare_member = {}
    for rare_name in models:
        rare_biomass = dict(biomass)
        rare_biomass[rare_name] = 1e-3
        result = SharedMediumCommunityLP(
            models, maximum_medium, rare_biomass, dt=dt
        ).solve()
        rare_member[short_name(rare_name)] = {
            "feasible": result.feasible,
            "common_growth_per_h": result.common_growth_per_h,
            "species_growth_per_h": {
                short_name(name): value
                for name, value in result.species_growth_per_h.items()
            },
        }
    return {
        "baseline": baseline.to_dict(),
        "maximum_one_step_action": maximum.to_dict(),
        "rare_member_screen": rare_member,
    }


def pha_reachability(models, dt: float) -> dict:
    biomass, medium = get_initial_params(models)
    medium = maximum_action_medium(medium)
    medium["nh4_e"] = 0.05
    simulator = dFBASimulator(
        models={name: model.copy() for name, model in models.items()},
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=100.0,
        volume=1.0,
        dt=dt,
        solver_backend="highs",
    )
    ns21_name = next(name for name in simulator.models if "NS21" in name)
    simulator.set_uptake_constraints(ns21_name, simulator.state.metabolites, 200.0)
    model = simulator.models[ns21_name]
    pha_id = simulator.exchange_reactions[ns21_name].get("pha_c")
    if not pha_id or pha_id not in model.reactions:
        return {"reachable": False, "reason": "No PHA sink/exchange mapped for NS21"}
    pha_reaction = model.reactions.get_by_id(pha_id)
    growth_reaction, _ = find_growth_reaction(model)
    with model:
        model.objective = pha_reaction
        unconstrained = model.optimize()
    with model:
        growth_reaction.lower_bound = max(growth_reaction.lower_bound, 1e-3)
        model.objective = pha_reaction
        coupled = model.optimize()
    unconstrained_flux = (
        float(unconstrained.fluxes[pha_id]) if unconstrained.status == "optimal" else 0.0
    )
    coupled_flux = float(coupled.fluxes[pha_id]) if coupled.status == "optimal" else 0.0
    return {
        "reachable": coupled_flux > 1e-9,
        "reaction": pha_id,
        "bounds": [float(pha_reaction.lower_bound), float(pha_reaction.upper_bound)],
        "max_flux_without_growth_constraint_mmol_gdcw_h": unconstrained_flux,
        "max_flux_at_growth_ge_0_001_mmol_gdcw_h": coupled_flux,
        "medium_nh4_mmol_l": medium["nh4_e"],
    }


def action_rollout(models, hours: float, dt: float, policy: str) -> dict:
    biomass, medium = get_initial_params(models)
    simulator = dFBASimulator(
        models={name: model.copy() for name, model in models.items()},
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=100.0,
        volume=1.0,
        dt=dt,
        solver_backend="highs",
    )
    env = ConsortiumEnv(simulator=simulator, max_time=hours)
    env.reset(seed=0)
    steps = max(1, int(np.ceil(hours / dt)))
    info = {}
    terminated = truncated = False
    base_added = 0.0
    for _ in range(steps):
        if policy == "current_maximum":
            action = np.ones(5, dtype=np.float32)
        elif policy == "oracle_two_phase_ph_control":
            base_added += apply_ideal_ph_control(simulator, target_ph=7.0)
            action = np.ones(5, dtype=np.float32)
            if simulator.state.time >= 12.0:
                # Preserve species-specific carbon feeds and aeration but stop
                # nitrogen-containing yeast extract to allow PHA induction.
                action[3] = 0.0
        else:
            raise ValueError(f"Unknown rollout policy: {policy}")
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    growth = {
        short_name(name): float(species.growth_rate)
        for name, species in simulator.state.species.items()
    }
    biomass_final = {
        short_name(name): float(species.biomass)
        for name, species in simulator.state.species.items()
    }
    return {
        "policy": policy,
        "hours": float(simulator.state.time),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_ph": float(info.get("ph", np.nan)),
        "final_biomass_g_l": biomass_final,
        "final_growth_per_h": growth,
        "total_pha_mmol": float(info.get("total_pha", 0.0)),
        "final_nh4_mmol_l": float(simulator.state.metabolites.get("nh4_e", 0.0)),
        "base_added_mmol_l": float(base_added),
        "rubber_remaining_g_l": float(info.get("rubber_remaining", np.nan)),
        "all_members_above_0_01": all(value >= 0.01 for value in biomass_final.values()),
        "all_members_nonnegative_growth": all(value >= -1e-3 for value in growth.values()),
    }


def build_verdict(static: dict, pha: dict, rollout: dict | None) -> dict:
    required = static["baseline"].get("required_additional_supply", [])
    reachable_metabolites = {
        metabolite
        for action in ACTION_SEMANTICS
        for metabolite in action["metabolites"]
    }
    missing_controls = sorted(
        item["metabolite"]
        for item in required
        if item["metabolite"] not in reachable_metabolites
    )
    rare_pass = all(
        result["feasible"] for result in static["rare_member_screen"].values()
    )
    checks = {
        "baseline_three_member_growth": bool(static["baseline"]["feasible"]),
        "maximum_action_static_growth": bool(
            static["maximum_one_step_action"]["feasible"]
        ),
        "rare_member_static_invasion_screen": rare_pass,
        "pha_with_positive_growth": bool(pha.get("reachable", False)),
        "all_phase_i_deficits_directly_controllable": not missing_controls,
        "direct_ph_control_available": False,
        "individual_member_loss_terminates_episode": False,
        "limiting_metabolites_observed": False,
        "nitrogen_switch_state_observed": False,
        "carbon_nitrogen_feed_decoupled": False,
    }
    if rollout is not None:
        dynamic_key = (
            "maximum_action_dynamic_stability"
            if rollout.get("policy") == "current_maximum"
            else "oracle_modified_problem_stability"
        )
        checks[dynamic_key] = bool(
            rollout["all_members_above_0_01"]
            and rollout["all_members_nonnegative_growth"]
            and 4.0 <= rollout["final_ph"] <= 9.5
        )
    essential = (
        "maximum_action_static_growth",
        "rare_member_static_invasion_screen",
        "pha_with_positive_growth",
        "direct_ph_control_available",
        "nitrogen_switch_state_observed",
    )
    ready = all(checks[name] for name in essential)
    return {
        "rl_ready": ready,
        "classification": "READY" if ready else "NOT_READY_STRUCTURALLY_UNDERCONTROLLED",
        "checks": checks,
        "unreachable_required_metabolites": missing_controls,
        "reason": (
            "Static coexistence, rare-member growth, and PHA flux are reachable, but "
            "the current action space has no pH/base control and the NH4 state that "
            "switches the NS21 objective is not observed. PPO cannot learn an unavailable "
            "intervention or reliably condition on a hidden switch."
            if not ready
            else "All pre-training reachability gates passed."
        ),
    }


def write_markdown(path: Path, payload: dict) -> None:
    verdict = payload["verdict"]
    static = payload["static_reachability"]
    pha = payload["pha_reachability"]
    lines = [
        "# 3種共存・RL問題設定の事前実行可能性監査",
        "",
        f"総合判定: **{verdict['classification']}**",
        "",
        verdict["reason"],
        "",
        "## 合否ゲート",
        "",
        "| 項目 | 合格 |",
        "|---|:---:|",
    ]
    for key, value in verdict["checks"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## 静的到達可能性",
        "",
        f"現行培地: 成立={static['baseline']['feasible']}、"
        f"共通増殖率={static['baseline']['common_growth_per_h']:.6g} h^-1。",
        f"最大1ステップ操作後: 成立={static['maximum_one_step_action']['feasible']}、"
        f"共通増殖率={static['maximum_one_step_action']['common_growth_per_h']:.6g} h^-1。",
        "",
        "## PHA到達可能性",
        "",
        f"NS21のPHA反応={pha.get('reaction', 'none')}、正の増殖を課した最大PHAフラックス="
        f"{pha.get('max_flux_at_growth_ge_0_001_mmol_gdcw_h', 0.0):.6g} mmol/gDCW/h。",
    ]
    if payload.get("maximum_action_rollout"):
        rollout = payload["maximum_action_rollout"]
        lines += [
            "",
            "## 最大操作dFBA",
            "",
            f"方策={rollout['policy']}、{rollout['hours']:.1f} h後: pH={rollout['final_ph']:.3f}、"
            f"PHA={rollout['total_pha_mmol']:.6g} mmol、"
            f"全種生存={rollout['all_members_above_0_01']}、"
            f"全種非負増殖={rollout['all_members_nonnegative_growth']}。",
        ]
    lines += [
        "",
        "## RL開始前に必要な修正",
        "",
        "1. g3ps_eまたは生物学的に妥当な代替炭素源を培地・操作空間へ追加する。",
        "2. 塩基滴定またはpH-stat操作を行動空間へ追加する。",
        "3. g3ps_e、ile__L_e、主要有機酸、滴定量を観測へ追加する。",
        "4. 1種脱落を終了または大きな制約違反として扱う。",
        "5. 共存可能性を確認後、PHA正フラックスを満たす対照制御を先に確立する。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sbml-dir",
        type=Path,
        default=ROOT / "models" / "sbml" / "final_consortium",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "rl_problem_feasibility",
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--rollout-hours", type=float, default=0.0)
    parser.add_argument(
        "--rollout-policy",
        choices=("current_maximum", "oracle_two_phase_ph_control"),
        default="current_maximum",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = select_consortium_models(load_sbml_models(args.sbml_dir))
    static = static_reachability(models, args.dt)
    pha = pha_reachability(models, args.dt)
    rollout = None
    if args.rollout_hours > 0:
        rollout = action_rollout(models, args.rollout_hours, args.dt, args.rollout_policy)
    payload = {
        "schema_version": 1,
        "action_semantics": ACTION_SEMANTICS,
        "static_reachability": static,
        "pha_reachability": pha,
        "maximum_action_rollout": rollout,
    }
    payload["verdict"] = build_verdict(static, pha, rollout)
    json_path = args.output_dir / "rl_problem_feasibility.json"
    md_path = args.output_dir / "rl_problem_feasibility.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
