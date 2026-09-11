#!/usr/bin/env python3
"""Create a transparent diagnostic report, not a qualification or speed claim."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    sources = [
        ("Scalar", "results/pf_phbv_final_strict_large_hull120.json"),
        ("Multi", "results/pf_multi_e2e_oldhead120.json"),
        ("Retrained", "results/pf_multi_e2e_newhead120.json"),
        ("Sensitivity", "results/pf_multi_e2e_sensitivity120.json"),
    ]
    rows = []
    for label, path in sources:
        report = read(path)
        assert report["seed"] == 20260913 and report["steps"] == 120
        for row in report["runs"]:
            row = dict(row)
            row["label"] = label if label != "Sensitivity" else f"λ={row['multioutput_strength']:g}"
            row["source"] = path
            rows.append(row)
    assert len({row["exact_pha_g_l"] for row in rows}) == 1
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.7,
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":300})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), layout="constrained")
    colors = ["#8B95A1", "#577B9E", "#258A8A", "#CC9950", "#A35C7A"]
    metrics = [("pha_relative_error", 100, 1, "PHA endpoint error (%)"),
               ("biomass_max_error_g_l", 1, .01, "Max. biomass error (g L$^{-1}$)"),
               ("phv_mol_fraction_error", 1, .01, "3HV mole-fraction error")]
    for i, (ax, (key, multiplier, threshold, ylabel)) in enumerate(zip(axes, metrics)):
        values = [row[key]*multiplier for row in rows]
        ax.scatter(np.arange(len(rows)), values, c=colors[:len(rows)], s=30, zorder=3)
        ax.axhline(threshold, color="#555555", linestyle="--", linewidth=.8)
        ax.set_xticks(np.arange(len(rows)), [row["label"] for row in rows], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97+i)})", loc="left", fontsize=9)
        ax.set_xlim(-.5, len(rows)-.5)
        if key == "phv_mol_fraction_error":
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.tick_params(width=.7, length=3)
    figure = ROOT/"results/pf_multioutput_endpoint_errors"
    fig.savefig(figure.with_suffix(".png"))
    fig.savefig(figure.with_suffix(".pdf"))
    plt.close(fig)
    table = ["|構成|PHA終点誤差|菌体量最大誤差 (g/L)|3HVモル分率誤差|GPU採用/120|オンラインCPU LP|実行秒数*|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"|{row['label']}|{100*row['pha_relative_error']:.3f}%|{row['biomass_max_error_g_l']:.6f}|"
            f"{row['phv_mol_fraction_error']:.6f}|{row['gpu_accepts']}|{row['cpu_lp_stage_calls']}|{row['seconds']:.2f}|")
    hull = read("results/pf_multi_hull_capacity.json")
    full = read("results/pf_multi_hull_capacity_full.json")
    independent = read("results/pf_multi_hull_independent_species.json")
    hull_table = ["|変更前GPU軌跡のstep|局所候補 2,108本|全候補＋合成候補|菌種独立混合（各2,048本）|", "|---|---:|---:|---:|"]
    for a in full["rows"]:
        b = next(r for r in hull["rows"] if r["step"] == a["step"])
        c = next(r for r in independent["rows"] if r["step"] == a["step"])
        hull_table.append(f"|{a['step']}|{b['best_worst_scaled_rate_error']:.3f}|{a['best_worst_scaled_rate_error']:.3f}|{c['best_worst_scaled_rate_error']:.3f}|")
    report = """# GPU複数出力補正：実装・検証報告（2026-09-03）

対象は OR16＋修正済みNS21（PHB/PHV分離）＋P. freudenreichii の3種系。
GEM、生物学的仮定、流加条件、CPU参照の目的関数は変更していない。
この報告は診断実験であり、実験培養による妥当性検証でも、正式な精度認証でもない。

結論：開発用1 seedでは、再学習＋複数出力補正（λ=30）でPHA終点誤差を
13.053%から0.513%へ縮小し、120/120 stepをGPUで処理できた。
ただし菌体量最大誤差0.028451 g/Lは0.01 g/L基準外であり、総合認証は未達。
GPU処理の実行時間も従来構成より増えており、高速化を達成したという結果ではない。

## 実装

155個の増殖・PHB/PHV・交換フラックスを同時に合わせるGPU QPを追加した。
予測値を硬い等式制約にはせず、標準化した予測誤差の重み付き二乗和と
参照混合係数からの変化量を最小化する。元の反応境界と共有培地制約を維持する。
物質収支は、同じGEMの厳密解を非負・総和1の係数で混合する構造で保持する。
GPU上のADMMに加え、未収束の候補からも制約内の改善だけを採用する線分探索を実装した。
全境界と共有培地制約を再検査し、目標誤差が改善しない補正は採用しない。
これはCPUの3段階LP最適解を保証するアルゴリズムではない。

予測器も再学習した。従来の32,768件に、既に独立収集済みの1,576件を加えた
34,344件を使用し、500 epochs、512×512のネットワークで学習した。
診断seedの状態を教師データに混ぜていない。サンプル分割の検証誤差と、
時系列全体の検証誤差は別物である。

## 120ステップの同一操作列比較

dt=0.2 h、24 h相当、seed=20260913、初期NH4=0.05 mM、探索候補2,048、
検索プール4,096、QP上限2,000反復。ランダム操作列によるdFBA検証で、PPO学習時間ではない。
Scalarは総PHAのみの従来補正、Multiは旧予測器＋複数出力補正、Retrainedは
再学習予測器＋複数出力補正（λ=100）。λ=30/300は再学習予測器の感度分析である。

""" + "\n".join(table) + """

*秒数は各GPU実行の記録。CPU参照は既存の同一条件の終点値を使用しているため、
同一負荷でのCPU対GPUの速度倍率や信頼区間は算出しない。
誤差基準はPHA ≤1%、菌体量 ≤0.01 g/L、3HVモル分率 ≤0.01で、全step採用も必要。
このseedは開発・調整用であり、n=1の結果から「最大誤差1%達成」を一般化しない。
最終検証用の5 seeds（20286001〜20286005）は未使用のまま保持した。

![終点誤差](../results/pf_multioutput_endpoint_errors.png)

図：いずれも小さいほど良い。破線は事前に定めた誤差基準。3HVパネルは対数軸。
独立反復がないためエラーバーは付けていない。条件・図の元データは各JSONに保存した。

## 誤差の切り分け

変更前GPU軌跡から26状態を採り、同一の反応境界・菌体量・培地でCPU LPを再計算した。
窒素の正味取り込み・排出が逆向きになる状態を確認し、NH4の1 step相当の
濃度変化の差は最大0.500 mMだった（正味フラックス差×dt。実際の更新では
非負化・流加等が入るため、更新後の濃度誤差そのものとは異なる）。
後半ではニューラル予測だけでなく、候補集合自体の表現範囲も調べる必要があった。

そこでCPUのオフライン診断LPで、同じ物理制約下における候補混合の最良到達誤差を求めた。
3種の増殖とPHB/PHVの5フラックスについて、各CPU値の1%（ゼロ近傍は絶対値1e-4）を
1単位とした最大誤差を最小化した。下表の値が1を超える場合、その候補集合では
この5フラックスを同時に当該範囲へ合わせられない。終点PHA誤差の基準とは別の診断指標である。

""" + "\n".join(hull_table) + """

このオフラインLPは推論時のCPUフォールバックには使っていない。また、この診断は
変更前GPUが訪れた状態に限られ、全ての状態・最終軌跡の誤差下限を意味しない。
菌種ごとに独立した混合係数を持たせるCPU診断も実施した。到達範囲は改善したが、
それだけでは3状態とも診断の1%範囲に届かなかった。独立混合のGPU実装は未実装である。

## 残る課題と次の順序

1. 別のseedで長時間・低窒素境界・窒素相切替えを重点的に収集する。
   現状の辞書全体でも到達できない状態があり、検索件数の増加だけでは不十分だった。
2. 各菌種の独立混合をGPU化し、追加データと組み合わせて検証する。
   検証状態そのものを学習に戻して合格扱いにはしない。
3. CPUと同じ共存→目的関数→交換最小化の順序をGPU上でも近づけ、QP反復回数と
   速度を改善する。使用率を上げること自体を目標にはしない。
4. 開発用の長時間試験を通過してから、未使用5 seedsで厳格な最大誤差検証を行う。

現状は研究用の非認証構成として保持し、本番用の既定設定は変更していない。
オンラインFBAのCPU LPは0回だが、環境更新・Python制御・入出力はホスト側に残る。
したがって「プログラム全体がGPU内だけで完結した」とは表現しない。
複数出力モードはローカルGPU QP経路で利用可能。未対応の別GPUサービス経路との
組み合わせは黙って無視せずエラーにし、従来モードの認証結果の流用も拒否する。

## 再現・テスト

- 実装：`src/gpu_multioutput_qp.py`、`src/gpu_batch_qp.py`、`src/community_solver.py`
- 比較：`scripts/diagnose_cooperative_gpu_candidates.py --multioutput-strength 100`
- 状態診断：`scripts/audit_cooperative_same_state.py`、`scripts/audit_multioutput_projection.py`
- 到達範囲診断：`scripts/audit_multioutput_hull_capacity.py`（オフライン専用）
- 図・本報告生成：`scripts/report_multioutput_gpu.py`
- 関連44 tests合格（CUDAを含む）。6件の既存fork関連警告あり。
- 新予測器：`models/cooperative_surrogate/pf_phbv_multihead_34344_e500_20260903.pt`
- 未認証。正式なqualification manifestや本番既定設定は昇格させていない。
"""
    (ROOT/"docs/GPU_MULTIOUTPUT_REPORT_20260903.md").write_text(report, encoding="utf-8")
    (ROOT/"results/pf_multioutput_comparison.json").write_text(json.dumps(
        {"status":"experimental_not_qualified", "seed":20260913, "steps":120, "rows":rows}, indent=2))
    print("\n".join(table))


if __name__ == "__main__":
    main()
