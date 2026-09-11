#!/usr/bin/env python3
"""Audit whether the intended three-GEM consortium can coexist.

Outputs a machine-readable JSON record, a compact CSV, a Japanese Markdown
report, and a four-panel diagnostic figure.  The audit does not alter SBML
models or training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coexistence_audit import (  # noqa: E402
    SharedMediumCommunityLP,
    all_member_combinations,
    relaxed_medium_requirements,
    simulate_dynamic_coexistence,
    validate_three_member_models,
)
from src.utils import get_initial_params, load_sbml_models, select_consortium_models  # noqa: E402


COLORS = {
    "OR16": "#3366A3",
    "NS21": "#D07A2D",
    "LP": "#3D8B6D",
}

SCENARIO_LABELS = {
    "no_feed": "No feed",
    "maintenance_feed": "Maintenance",
    "candidate_rescue": "Nutrient rescue",
    "candidate_rescue_ph_control": "Rescue + pH-stat",
}


def short_name(name: str) -> str:
    if "OR16" in name:
        return "OR16"
    if "NS21" in name:
        return "NS21"
    if "Lactobacillus" in name:
        return "LP"
    return name


def combination_label(names) -> str:
    order = {"OR16": 0, "NS21": 1, "LP": 2}
    labels = sorted((short_name(name) for name in names), key=lambda item: order.get(item, 99))
    return "+".join(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-GEM consortium coexistence pre-audit (shared-medium LP + exact dFBA)"
    )
    parser.add_argument(
        "--sbml-dir",
        type=Path,
        default=ROOT / "models" / "sbml" / "final_consortium",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "coexistence_audit")
    parser.add_argument("--hours", type=float, default=24.0, help="dFBA horizon per run")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument(
        "--dynamic",
        choices=("none", "triplet", "all"),
        default="triplet",
        help="Which subsets receive time-course dFBA tests",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=(
            "no_feed",
            "maintenance_feed",
            "candidate_rescue",
            "candidate_rescue_ph_control",
        ),
        default=(
            "no_feed",
            "maintenance_feed",
            "candidate_rescue",
            "candidate_rescue_ph_control",
        ),
    )
    parser.add_argument("--growth-threshold", type=float, default=1e-5)
    parser.add_argument("--survival-threshold", type=float, default=0.01)
    parser.add_argument("--record-every", type=int, default=10)
    return parser.parse_args()


def static_audit(models, args) -> dict[str, Any]:
    results = {}
    for subset in all_member_combinations(models):
        biomass, medium = get_initial_params(subset)
        lp = SharedMediumCommunityLP(
            subset,
            medium,
            biomass,
            dt=args.dt,
            growth_threshold=args.growth_threshold,
        )
        result = lp.solve()
        results[combination_label(subset)] = result.to_dict()
    return results


def dynamic_audit(models, args) -> list[dict[str, Any]]:
    if args.dynamic == "none":
        return []
    subsets = list(all_member_combinations(models))
    if args.dynamic == "triplet":
        subsets = [models]
    results = []
    for subset in subsets:
        label = combination_label(subset)
        for scenario in args.scenarios:
            print(f"[dynamic] {label:12s} scenario={scenario:16s} hours={args.hours:g}", flush=True)
            result = simulate_dynamic_coexistence(
                subset,
                hours=args.hours,
                scenario=scenario,
                dt=args.dt,
                survival_threshold=args.survival_threshold,
                record_every_steps=args.record_every,
            )
            payload = result.to_dict()
            payload["combination"] = label
            results.append(payload)
    return results


def diagnose(models, static: Mapping[str, Any], dynamic: list[dict[str, Any]], args) -> dict[str, Any]:
    issues = []
    triplet = static.get("OR16+NS21+LP", {})
    rescue_test = None
    if not triplet.get("feasible", False):
        issues.append(
            {
                "severity": "critical",
                "code": "no_balanced_growth_solution",
                "message": "共有培地max–min LPで3種すべての正の増殖を同時に実現できない。",
            }
        )
        required = triplet.get("required_additional_supply", [])
        if required:
            biomass, medium = get_initial_params(models)
            suggested = {
                item["metabolite"]: max(
                    0.01, 2.0 * float(item["additional_supply_mmol_l_h"])
                )
                for item in required
            }
            rescued = SharedMediumCommunityLP(
                models,
                medium,
                biomass,
                dt=args.dt,
                growth_threshold=args.growth_threshold,
            ).solve(suggested)
            rescue_payload = rescued.to_dict()
            # Keep the diagnosis compact. Full per-metabolite balances are
            # retained in the ordinary static results where they are useful.
            rescue_payload.pop("exchange_balance", None)
            rescue_test = {
                "supply_overrides_mmol_l_h": suggested,
                "result": rescue_payload,
            }
            if rescued.feasible:
                issues.append(
                    {
                        "severity": "high",
                        "code": "medium_rescue_identified",
                        "message": "不足候補を補うと3種max–min LPが成立するため、主因はモデル構造破綻ではなく培地定義。",
                        "supply_overrides_mmol_l_h": suggested,
                        "rescued_common_growth_per_h": rescued.common_growth_per_h,
                    }
                )
    else:
        issues.append(
            {
                "severity": "info",
                "code": "balanced_growth_feasible",
                "message": f"共有培地LP上は共通増殖率 {triplet['common_growth_per_h']:.4g} 1/h で3種同時増殖が可能。",
            }
        )

    triplet_dynamic = [item for item in dynamic if item["combination"] == "OR16+NS21+LP"]
    by_scenario = {item["scenario"]: item for item in triplet_dynamic}
    no_feed = by_scenario.get("no_feed")
    maintenance = by_scenario.get("maintenance_feed")
    ph_control = by_scenario.get("candidate_rescue_ph_control")
    if no_feed and not no_feed["all_survive"]:
        extinct = [name for name, value in no_feed["final_biomass_g_l"].items() if value < args.survival_threshold]
        issues.append(
            {
                "severity": "high",
                "code": "dynamic_extinction_without_feed",
                "message": "無給餌dFBAで生存閾値を下回る: " + ", ".join(short_name(name) for name in extinct),
                "depleted_metabolites": no_feed.get("depleted_metabolites", []),
            }
        )
    if no_feed and no_feed["all_survive"]:
        issues.append(
            {
                "severity": "info",
                "code": "dynamic_persistence_without_feed",
                "message": f"無給餌で{no_feed['hours']:.1f} h、3種すべてが生存閾値以上を維持。",
            }
        )
        if not no_feed.get("stable_at_horizon", False):
            issues.append(
                {
                    "severity": "high",
                    "code": "survival_without_stability",
                    "message": (
                        "生存閾値は維持したが、終点で負の増殖率または許容外pHを示すため、"
                        "安定共生とは判定しない。"
                    ),
                    "final_ph": no_feed.get("final_ph"),
                    "final_growth_per_h": no_feed.get("final_growth_per_h"),
                }
            )
    if maintenance and maintenance["all_survive"] and no_feed and not no_feed["all_survive"]:
        issues.append(
            {
                "severity": "medium",
                "code": "feed_dependent_coexistence",
                "message": "維持給餌でのみ3種が残るため、共生は栄養供給条件付き。",
            }
        )
    if ph_control and ph_control.get("stable_at_horizon", False):
        issues.append(
            {
                "severity": "high",
                "code": "ph_control_rescues_dynamic_stability",
                "message": (
                    "不足候補の補完と理想pH-statを併用すると終点安定条件を満たす。"
                    "酸生成・緩衝能・塩基滴定を主要設計変数として扱う必要がある。"
                ),
                "base_added_mmol_l": ph_control.get("base_added_mmol_l", 0.0),
            }
        )
    elif ph_control:
        issues.append(
            {
                "severity": "high",
                "code": "ph_control_not_sufficient",
                "message": (
                    "不足候補の補完と理想pH-statを併用しても終点安定条件を満たさない。"
                    "pH以外の培地成分、目的反応、維持代謝を追加点検する必要がある。"
                ),
                "final_ph": ph_control.get("final_ph"),
                "final_growth_per_h": ph_control.get("final_growth_per_h"),
            }
        )
    if triplet.get("feasible") and no_feed and not no_feed["all_survive"]:
        issues.append(
            {
                "severity": "high",
                "code": "static_dynamic_mismatch",
                "message": "定常LPでは成立するがdFBAで崩れる。速度論、初期比率、pH、資源枯渇、目的関数切替を優先点検する。",
            }
        )

    relaxed = {}
    for name, model in models.items():
        mono = static.get(short_name(name), {})
        if not mono.get("feasible", False):
            relaxed[short_name(name)] = relaxed_medium_requirements(model)
    cross_feeding = triplet.get("cross_feeding", [])
    if not cross_feeding and rescue_test:
        cross_feeding = rescue_test.get("result", {}).get("cross_feeding", [])
    if not cross_feeding:
        issues.append(
            {
                "severity": "medium",
                "code": "no_required_cross_feeding_detected",
                "message": "max–min解で明確な生産者→消費者の代謝物授受が検出されない。3種は共生ではなく単なる共存の可能性がある。",
            }
        )

    coexistence_supported = bool(
        triplet.get("feasible", False)
        and (not triplet_dynamic or all(item.get("stable_at_horizon", False) for item in triplet_dynamic))
    )
    rescue_feasible = bool(
        rescue_test and rescue_test.get("result", {}).get("feasible", False)
    )
    if triplet.get("feasible") and triplet_dynamic and any(item.get("stable_at_horizon", False) for item in triplet_dynamic):
        classification = "conditional_or_supported"
    elif triplet.get("feasible"):
        classification = "static_only"
    elif rescue_feasible:
        classification = "medium_correction_required"
    else:
        classification = "not_supported"
    return {
        "classification": classification,
        "coexistence_supported_under_all_tested_conditions": coexistence_supported,
        "issues": issues,
        "relaxed_medium_diagnostics": relaxed,
        "rescue_test": rescue_test,
    }


def write_csv(path: Path, static: Mapping[str, Any], dynamic: list[dict[str, Any]]) -> None:
    dynamic_map = {(item["combination"], item["scenario"]): item for item in dynamic}
    scenarios = sorted({item["scenario"] for item in dynamic})
    fields = ["combination", "static_feasible", "common_growth_per_h"]
    for scenario in scenarios:
        fields += [f"{scenario}_all_survive", f"{scenario}_stable", f"{scenario}_final_total_biomass_g_l", f"{scenario}_final_ph"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, result in static.items():
            row = {
                "combination": label,
                "static_feasible": result["feasible"],
                "common_growth_per_h": result["common_growth_per_h"],
            }
            for scenario in scenarios:
                item = dynamic_map.get((label, scenario))
                if item:
                    row[f"{scenario}_all_survive"] = item["all_survive"]
                    row[f"{scenario}_stable"] = item.get("stable_at_horizon", False)
                    row[f"{scenario}_final_total_biomass_g_l"] = sum(item["final_biomass_g_l"].values())
                    row[f"{scenario}_final_ph"] = item["final_ph"]
            writer.writerow(row)


def make_figure(
    path: Path,
    static: Mapping[str, Any],
    dynamic: list[dict[str, Any]],
    diagnosis: Mapping[str, Any],
) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)

    labels = list(static)
    growth = [static[label]["common_growth_per_h"] for label in labels]
    colors = ["#3366A3" if static[label]["feasible"] else "#B8BEC7" for label in labels]
    axes[0, 0].bar(np.arange(len(labels)), growth, color=colors, edgecolor="white")
    axes[0, 0].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axes[0, 0].set_ylabel("max–min growth (1/h)")
    axes[0, 0].set_title("A  Shared-medium feasibility")
    axes[0, 0].grid(axis="y", alpha=0.2)

    triplet_runs = [item for item in dynamic if item["combination"] == "OR16+NS21+LP"]
    scenarios = [item["scenario"] for item in triplet_runs]
    x = np.arange(len(scenarios))
    width = 0.23
    for offset, member in enumerate(("OR16", "NS21", "LP")):
        values = []
        for item in triplet_runs:
            match = next((value for name, value in item["final_biomass_g_l"].items() if short_name(name) == member), 0.0)
            values.append(match)
        axes[0, 1].bar(x + (offset - 1) * width, values, width, label=member, color=COLORS[member])
    axes[0, 1].set_xticks(
        x,
        [SCENARIO_LABELS.get(item, item) for item in scenarios],
        rotation=18,
        ha="right",
    )
    axes[0, 1].set_ylabel("final biomass (g/L)")
    axes[0, 1].set_title("B  Triplet persistence")
    axes[0, 1].legend(frameon=False, ncol=3)
    axes[0, 1].grid(axis="y", alpha=0.2)

    preferred = next((item for item in triplet_runs if item["scenario"] == "no_feed"), None)
    if preferred is None and triplet_runs:
        preferred = triplet_runs[0]
    if preferred:
        time = [row["time_h"] for row in preferred["trajectory"]]
        for member in ("OR16", "NS21", "LP"):
            values = [
                next((value for name, value in row["biomass_g_l"].items() if short_name(name) == member), np.nan)
                for row in preferred["trajectory"]
            ]
            axes[1, 0].plot(time, values, label=member, color=COLORS[member], lw=1.8)
        ph_axis = axes[1, 0].twinx()
        ph_axis.plot(
            time,
            [row["ph"] for row in preferred["trajectory"]],
            color="#5B5B5B",
            lw=1.5,
            ls="--",
            label="pH",
        )
        ph_axis.axhline(4.0, color="#B04A4A", lw=0.9, ls=":", label="pH 4 threshold")
        ph_axis.set_ylabel("pH")
        ph_axis.set_ylim(2.0, 8.0)
        axes[1, 0].set_xlabel("time (h)")
        axes[1, 0].set_ylabel("biomass (g/L)")
        axes[1, 0].set_title(f"C  Triplet trajectory ({preferred['scenario']})")
        lines_a, labels_a = axes[1, 0].get_legend_handles_labels()
        lines_b, labels_b = ph_axis.get_legend_handles_labels()
        axes[1, 0].legend(lines_a + lines_b, labels_a + labels_b, frameon=False, ncol=3)
        axes[1, 0].grid(alpha=0.2)
    else:
        axes[1, 0].text(0.5, 0.5, "dynamic audit not run", ha="center", va="center")
        axes[1, 0].set_axis_off()

    triplet = static.get("OR16+NS21+LP", {})
    edges = triplet.get("cross_feeding", [])
    if not edges:
        rescue_test = diagnosis.get("rescue_test") or {}
        rescue_result = rescue_test.get("result") or {}
        edges = rescue_result.get("cross_feeding", [])
    edges = edges[:12]
    if edges:
        edge_labels = [f"{short_name(e['producer'])}→{short_name(e['consumer'])}: {e['metabolite']}" for e in edges]
        values = [min(e["producer_rate"], e["consumer_rate"]) for e in edges]
        order = np.argsort(values)
        axes[1, 1].barh(np.arange(len(edges)), np.asarray(values)[order], color="#6D8EAD")
        axes[1, 1].set_yticks(np.arange(len(edges)), np.asarray(edge_labels)[order])
        axes[1, 1].set_xlabel("matched transfer rate (mmol/L/h)")
        axes[1, 1].set_title("D  Predicted cross-feeding")
        axes[1, 1].grid(axis="x", alpha=0.2)
    else:
        axes[1, 1].text(0.5, 0.5, "no cross-feeding edge detected", ha="center", va="center")
        axes[1, 1].set_title("D  Predicted cross-feeding")
        axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    static = payload["static"]
    dynamic = payload["dynamic"]
    diagnosis = payload["diagnosis"]
    triplet = static["OR16+NS21+LP"]
    classification_labels = {
        "conditional_or_supported": "試験条件下で成立",
        "static_only": "静的LPのみ成立",
        "medium_correction_required": "現行培地では不成立（培地補正が必要）",
        "not_supported": "試験条件では不成立",
    }
    ph_control = next(
        (item for item in dynamic if item.get("scenario") == "candidate_rescue_ph_control"),
        None,
    )
    lines = [
        "# 3種コンソーシアム共生成立性・事前監査",
        "",
        "## 結論",
        "",
        f"判定: **{classification_labels.get(diagnosis['classification'], diagnosis['classification'])}**。"
        f"共有培地max–min LPの3種共通増殖率は "
        f"**{triplet['common_growth_per_h']:.6g} 1/h**、静的成立判定は **{triplet['feasible']}**。",
        "",
    ]
    if ph_control and ph_control.get("stable_at_horizon", False):
        lines += [
            "不足成分補完と理想pH-statの併用時のみ48 h終点の安定条件を満たし、"
            f"累積塩基需要は **{ph_control.get('base_added_mmol_l', 0.0):.3f} mmol/L**だった。",
            "",
        ]
    lines += [
        "この判定はin silicoの成立可能性であり、実験的な安定共生の証明ではない。",
        "",
        "## 静的な共有培地LP",
        "",
        "| 組合せ | 共通増殖率 (1/h) | 成立 |",
        "|---|---:|:---:|",
    ]
    for label, result in static.items():
        lines.append(f"| {label} | {result['common_growth_per_h']:.6g} | {result['feasible']} |")
    lines += ["", "## dFBA持続性", "", "| 組合せ | 条件 | 全種生存 | 安定判定 | 最終pH | 累積塩基 (mmol/L) | 枯渇候補 |", "|---|---|:---:|:---:|---:|---:|---|"]
    for item in dynamic:
        depleted = ", ".join(item["depleted_metabolites"][:8]) or "—"
        lines.append(
            f"| {item['combination']} | {SCENARIO_LABELS.get(item['scenario'], item['scenario'])} | "
            f"{item['all_survive']} | {item.get('stable_at_horizon', False)} | "
            f"{item['final_ph']:.3f} | {item.get('base_added_mmol_l', 0.0):.3f} | {depleted} |"
        )
    lines += ["", "## 検出した問題・条件", ""]
    for issue in diagnosis["issues"]:
        lines.append(f"- **{issue['severity']} / {issue['code']}**: {issue['message']}")
    rescue = diagnosis.get("rescue_test")
    if rescue:
        lines += ["", "## 培地レスキュー試験", ""]
        lines.append(
            f"追加供給候補 `{rescue['supply_overrides_mmol_l_h']}` により、"
            f"共通増殖率は {rescue['result']['common_growth_per_h']:.6g} 1/h、"
            f"成立={rescue['result']['feasible']}。"
        )
    lines += ["", "## 推定される代謝物授受", ""]
    edges = triplet.get("cross_feeding", [])
    if not edges and rescue:
        edges = rescue.get("result", {}).get("cross_feeding", [])
    if edges:
        for edge in edges[:20]:
            lines.append(
                f"- {short_name(edge['producer'])} → {short_name(edge['consumer'])}: "
                f"{edge['metabolite']}（生産 {edge['producer_rate']:.4g}, 消費 {edge['consumer_rate']:.4g} mmol/L/h）"
            )
    else:
        lines.append("- 明確な授受エッジは検出されなかった。")
    lines += [
        "",
        "## 方法上の注意",
        "",
        "- 静的LPは固定した初期バイオマス比と共有培地供給上限を用い、最小の菌種増殖率を最大化する。",
        "- dFBAは厳密SciPy/HiGHSで計算し、GPUサロゲートは使用しない。",
        "- 正の共通増殖は成立可能性を示すが、侵入可能性・撹乱後回復・実験的安定性は別途検証が必要。",
        "- maintenance_feedは成立条件探索用の保守的固定給餌であり、最適培地ではない。",
        "- Rescue + pH-statは理想的な塩基滴定を仮定した原因切り分け条件であり、実培養条件では滴定速度・浸透圧・毒性を再評価する。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_sbml_models(args.sbml_dir)
    models = select_consortium_models(loaded)
    validate_three_member_models(models)
    static = static_audit(models, args)
    dynamic = dynamic_audit(models, args)
    diagnosis = diagnose(models, static, dynamic, args)
    payload = {
        "schema_version": 1,
        "method": {
            "static": "shared-medium max-min balanced-growth LP with fixed initial biomass",
            "dynamic": "exact SciPy/HiGHS dFBA",
            "hours": args.hours,
            "dt": args.dt,
            "growth_threshold": args.growth_threshold,
            "survival_threshold": args.survival_threshold,
        },
        "models": {name: {"reactions": len(model.reactions), "metabolites": len(model.metabolites), "genes": len(model.genes)} for name, model in models.items()},
        "static": static,
        "dynamic": dynamic,
        "diagnosis": diagnosis,
    }
    json_path = args.output_dir / "coexistence_audit.json"
    csv_path = args.output_dir / "coexistence_summary.csv"
    md_path = args.output_dir / "coexistence_report.md"
    fig_path = args.output_dir / "coexistence_audit.png"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, static, dynamic)
    write_markdown(md_path, payload)
    make_figure(fig_path, static, dynamic, diagnosis)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print(f"FIG:  {fig_path}")


if __name__ == "__main__":
    main()
