#!/usr/bin/env python3
"""Compare practical feed strategies with the corrected three-member dFBA."""

from __future__ import annotations

import argparse
import csv
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
from src.utils import get_initial_params, load_sbml_models, select_consortium_models  # noqa: E402

AA8 = (
    "his__L_e", "phe__L_e", "asn__L_e", "asp__L_e",
    "ser__L_e", "gly_e", "ile__L_e", "ala__L_e",
)
DEFINED_GROWTH_MIX = AA8 + (
    "gln__L_e", "lys__L_e", "cys__L_e", "tyr__L_e", "met__L_e",
)
TRACKED_FLUXES = (
    "o2_e", "nh4_e", "glc__D_e", "mnl_e", "C30_oligo_e", "odtd_e",
    "ac_e", "lac__L_e", "succ_e", "co2_e",
)


def _add_defined(state, members, amount):
    for metabolite in members:
        state.metabolites[metabolite] = state.metabolites.get(metabolite, 0.0) + amount


def _feed(strategy, simulator, time_h, dt):
    """Return simulator feed controls after applying direct defined solutes."""
    if strategy == "No feed":
        return {}
    if strategy == "N-limited production":
        return {}
    if strategy == "AA8 continuous":
        _add_defined(simulator.state, AA8, 0.05 * dt)
        return {}
    if strategy == "YE continuous":
        return {"yeast_extract": 0.02 * dt}
    if strategy == "YE pulse":
        return {"yeast_extract": 0.05 * dt} if time_h < 6.0 else {}
    if strategy == "Defined pulse + mannitol":
        if time_h < 6.0:
            _add_defined(simulator.state, DEFINED_GROWTH_MIX, 0.02 * dt)
        return {"sn_lp": 0.02 * dt} if time_h >= 6.0 else {"sn_lp": 0.05 * dt}
    raise ValueError(strategy)


def run_strategy(strategy, model_dir, hours, dt):
    models = select_consortium_models(load_sbml_models(model_dir))
    biomass, medium = get_initial_params(models)
    if strategy == "N-limited production":
        # A production-stage medium after separate inoculum preparation. This
        # is not achieved by silently deleting nitrogen during a fed-batch.
        medium["nh4_e"] = 0.05
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=100.0,
        dt=dt,
        solver_backend="highs",
        fba_mode="separate",
        polymer_oxygen_fraction=0.25,
    )
    integrated = {
        metabolite: {name: 0.0 for name in models} for metabolite in TRACKED_FLUXES
    }
    trajectory = []
    for _ in range(max(1, int(np.ceil(hours / dt)))):
        feed = _feed(strategy, simulator, simulator.state.time, dt)
        state = simulator.step({}, feed, dynamic_kla=100.0)
        for name, species in state.species.items():
            for metabolite in TRACKED_FLUXES:
                uptake = species.metabolite_uptake.get(metabolite, 0.0)
                secretion = species.metabolite_secretion.get(metabolite, 0.0)
                integrated[metabolite][name] += (
                    (uptake - secretion) * species.biomass * dt
                )
        trajectory.append(
            {
                "strategy": strategy,
                "time_h": float(state.time),
                "rubber_g_l": float(state.rubber_concentration),
                "c30_mmol_l": float(state.metabolites.get("C30_oligo_e", 0.0)),
                "odtd_mmol_l": float(state.metabolites.get("odtd_e", 0.0)),
                "oxygen_mmol_l": float(state.metabolites.get("o2_e", 0.0)),
                "nh4_mmol_l": float(state.metabolites.get("nh4_e", 0.0)),
                "ph": float(-np.log10(max(1e-12, state.metabolites.get("h_e", 1e-4)) / 1000.0)),
                "pha_mmol": float(sum(item.pha_accumulated for item in state.species.values())),
                **{
                    f"biomass_{name}": float(item.biomass)
                    for name, item in state.species.items()
                },
            }
        )
    final = simulator.state
    final_biomass = {name: float(item.biomass) for name, item in final.species.items()}
    return {
        "strategy": strategy,
        "final_biomass_g_l": final_biomass,
        "minimum_final_biomass_g_l": min(final_biomass.values()),
        "rubber_degraded_g_l": 100.0 - float(final.rubber_concentration),
        "pha_mmol": float(sum(item.pha_accumulated for item in final.species.values())),
        "c30_mmol_l": float(final.metabolites.get("C30_oligo_e", 0.0)),
        "odtd_mmol_l": float(final.metabolites.get("odtd_e", 0.0)),
        "nh4_mmol_l": float(final.metabolites.get("nh4_e", 0.0)),
        "ph": float(-np.log10(max(1e-12, final.metabolites.get("h_e", 1e-4)) / 1000.0)),
        "base_added_mmol_l": float(simulator.cumulative_base_added_mmol_l),
        "acid_added_mmol_l": float(simulator.cumulative_acid_added_mmol_l),
        "integrated_net_consumption_mmol_l": integrated,
        "trajectory": trajectory,
    }


def _display_species(name):
    if "OR16" in name:
        return "OR16"
    if "NS21" in name:
        return "NS21"
    return "L. plantarum"


def plot_results(results, output_dir):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = {"OR16": "#0072B2", "NS21": "#D55E00", "L. plantarum": "#009E73"}
    strategies = [item["strategy"] for item in results]
    species = list(results[0]["final_biomass_g_l"])
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.1), constrained_layout=True)

    x = np.arange(len(strategies))
    width = 0.24
    for index, name in enumerate(species):
        label = _display_species(name)
        axes[0, 0].bar(
            x + (index - 1) * width,
            [item["final_biomass_g_l"][name] for item in results],
            width=width, color=colors[label], label=label,
        )
    axes[0, 0].set_xticks(x, strategies, rotation=25, ha="right")
    axes[0, 0].set_ylabel("Final biomass (g L$^{-1}$)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=7)

    axes[0, 1].bar(x, [item["pha_mmol"] for item in results], color="#CC79A7")
    axes[0, 1].set_xticks(x, strategies, rotation=25, ha="right")
    axes[0, 1].set_ylabel("PHA accumulated (mmol L$^{-1}$)")

    selected = next(item for item in results if item["strategy"] == "N-limited production")
    time = [row["time_h"] for row in selected["trajectory"]]
    for name in species:
        label = _display_species(name)
        axes[1, 0].plot(
            time, [row[f"biomass_{name}"] for row in selected["trajectory"]],
            color=colors[label], label=label, linewidth=1.5,
        )
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel("Biomass (g L$^{-1}$)")

    flux = selected["integrated_net_consumption_mmol_l"]
    matrix = np.asarray([[flux[metabolite][name] for name in species] for metabolite in TRACKED_FLUXES])
    vmax = max(float(np.max(np.abs(matrix))), 1e-9)
    image = axes[1, 1].imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1, 1].set_xticks(range(len(species)), [_display_species(name) for name in species], rotation=25, ha="right")
    axes[1, 1].set_yticks(range(len(TRACKED_FLUXES)), TRACKED_FLUXES, fontsize=6.5)
    bar = fig.colorbar(image, ax=axes[1, 1], fraction=0.045, pad=0.03)
    bar.set_label("Integrated net consumption (mmol L$^{-1}$)", fontsize=7)

    for label, axis in zip("abcd", axes.flat):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    base = output_dir / "Figure_corrected_feed_screen"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(results, output_dir):
    by_name = {item["strategy"]: item for item in results}
    continuous = by_name["YE continuous"]
    pulse = by_name["YE pulse"]
    nitrogen_limited = by_name["N-limited production"]
    lines = [
        "# 修正版3種dFBAによる流加条件スクリーニング",
        "",
        "## 選定結果",
        "",
        "低濃度酵母エキスは接種菌・初期菌体の確立時だけパルス供給し、その後は低窒素の生産培地へ移行または接種する。PHA生産期には酵母エキスやアミノ酸を連続供給しない。",
        "",
        "## モデル上の根拠（24時間）",
        "",
        f"- 酵母エキス連続供給は菌体維持量が最大だったが、PHAは{continuous['pha_mmol']:.4g} mmolで、NH4は{continuous['nh4_mmol_l']:.3f} mM残った。",
        f"- 6時間酵母エキスパルスでも三種菌体を維持できたが、NH4は{pulse['nh4_mmol_l']:.3f} mM残った。",
        f"- PHAを生成したのは低窒素生産条件だけであった（{nitrogen_limited['pha_mmol']:.4g} mmol）。ただしGEMは窒素再無機化を予測し、24時間後のNH4は{nitrogen_limited['nh4_mmol_l']:.3f} mMまで再上昇した。",
        "- AA8は静的共有培地max-min成長を改善したが、動的計算ではL. plantarumのより広いアミノ酸・ビタミン要求を代替できなかった。",
        "- 現GEMではマンニトールを主にNS21が消費したため、実測取り込みデータなしにL. plantarum特異的流加液とは見なせない。",
        "",
        "## 実用上のポンプ割当案",
        "",
        "1. 増殖期共通液：希釈酵母エキスを有限パルスで供給し、生産期前に停止する。",
        "2. 生産期流加液：無窒素の無機塩・微量元素液。モデル上のLcp抑制を避けるためグルコースは含めない。",
        "3. 小規模実験で効果を確認した救済栄養用に1系統を空ける。AA8は最初の定義組成候補であり、まだ確定レシピではない。",
        "4. 第4プロセスポンプを流加／排出に使う場合は、希釈と菌体washoutを明示的にモデル化する。手動サンプリングだけなら常時割当は不要である。",
        "",
        "独立したpH用2ポンプはpH 7.0の理想pH-statとして実装した。酵素Vmax、酸素の25%を細胞外切断へ配分する係数、流加濃度は実測フィット値ではなく、今後校正すべきパラメータである。",
    ]
    (output_dir / "corrected_feed_screen_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/sbml/final_consortium")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/corrected_feed_screen")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strategies = (
        "No feed", "N-limited production", "AA8 continuous", "YE continuous", "YE pulse",
        "Defined pulse + mannitol",
    )
    results = [run_strategy(name, args.model_dir, args.hours, args.dt) for name in strategies]
    payload = {
        "parameters": {
            "hours": args.hours, "dt_h": args.dt,
            "polymer_oxygen_fraction": 0.25,
            "warning": "Feed amounts and extracellular enzyme rates are model-screening assumptions, not calibrated recipes.",
        },
        "results": results,
    }
    (args.output_dir / "corrected_feed_screen.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "corrected_feed_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "strategy", "minimum_final_biomass_g_l", "rubber_degraded_g_l",
            "pha_mmol", "c30_mmol_l", "odtd_mmol_l", "final_biomass_g_l",
            "nh4_mmol_l", "ph",
            "base_added_mmol_l", "acid_added_mmol_l",
        ))
        writer.writeheader()
        for item in results:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                for key, value in item.items() if key not in {"trajectory", "integrated_net_consumption_mmol_l"}
            })
    plot_results(results, args.output_dir)
    write_report(results, args.output_dir)
    for item in results:
        print(item["strategy"], item["minimum_final_biomass_g_l"], item["rubber_degraded_g_l"], item["pha_mmol"])
    print(args.output_dir)


if __name__ == "__main__":
    main()
