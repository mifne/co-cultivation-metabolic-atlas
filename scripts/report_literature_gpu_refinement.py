#!/usr/bin/env python3
"""Summarize all controls, including negative results, without qualification."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    baseline = read("results/pf_multi_e2e_sensitivity120.json")["runs"][0]
    rows = [{"label":"Previous", "description":"前回・34,344件", **baseline}]
    files = [("Frozen", "予測器固定・辞書拡張", "frozen"),
             ("Retrain", "追加データ・一から再学習", "dataonly"),
             ("Bounds", "再学習＋反応境界損失", "mechanistic"),
             ("Warm", "前回予測器から継続校正", "warm"),
             ("Independent", "菌種独立混合（K=512）", "independent")]
    for label, description, name in files:
        path = f"results/pf_literature_{name}120.json"
        report = read(path)
        assert report["seed"] == 20260913 and report["steps"] == 120
        rows.append({"label":label, "description":description, "source":path, **report["runs"][0]})
    service = read("results/pf_literature_service_final.json")
    training_a = read("models/cooperative_surrogate/pf_literature_dagger_neural_20260903.json")
    training_b = read("models/cooperative_surrogate/pf_literature_mechanistic_neural_20260903.json")
    table = ["|構成|PHA誤差|菌体量最大誤差 (g/L)|3HVモル分率誤差|GPU採用/120|CPU LP|",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"|{row['description']}|{100*row['pha_relative_error']:.3f}%|"
            f"{row['biomass_max_error_g_l']:.6f}|{row['phv_mol_fraction_error']:.6f}|{row['gpu_accepts']}|{row['cpu_lp_stage_calls']}|")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.7})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), layout="constrained")
    colors = ["#7B8792", "#4C7399", "#CB9350", "#359083", "#725F9B", "#B36472"]
    metrics = [("pha_relative_error", 100, 1, "PHA endpoint error (%)"),
        ("biomass_max_error_g_l", 1, .01, "Max. biomass error (g L$^{-1}$)"),
        ("phv_mol_fraction_error", 1, .01, "3HV mole-fraction error")]
    for index, (axis, (metric, multiplier, limit, ylabel)) in enumerate(zip(axes, metrics)):
        for position, row in enumerate(rows):
            axis.scatter(position, row[metric]*multiplier, c=colors[position], s=28,
                         marker="o" if row["gpu_accepts"] == 120 else "x")
        axis.axhline(limit, color="#555555", linestyle="--", linewidth=.8)
        axis.set_xticks(range(len(rows)), [r["label"] for r in rows], rotation=45, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(f"({chr(97+index)})", loc="left")
        if metric == "pha_relative_error":
            axis.set_yscale("log")
            axis.set_ylim(.15, 50)
        else:
            axis.set_ylim(bottom=0)
    fig.savefig(ROOT/"results/pf_literature_refinement.png", dpi=300)
    fig.savefig(ROOT/"results/pf_literature_refinement.pdf")
    plt.close(fig)
    text = """# 論文に基づくGPU機構層の改修・比較（2026-09-03）

対象：OR16＋NS21 PHBV＋P. freudenreichii。GEM・流加条件・CPUの3段階LP目的関数は変更していない。
実装が動くこと、物理制約を満たすこと、CPU軌跡に一致することは区別して評価した。

## 論文と実装の対応

|一次文献|今回の適用|適用していない／保証されない部分|
|---|---|---|
|[Faure et al., AMN (2023)](https://www.nature.com/articles/s41467-023-40380-0)|ニューラル予測と機構制約の組合せ、学習時のライブ反応境界損失、低次元フラックス表現の再検討|AMNをそのまま再現したわけではない。GEM-QP全体の勾配を通す学習は未実装。実験値への生物学的校正ではなくCPUモデルの近似|
|[Amos & Kolter, OptNet (2017)](https://proceedings.mlr.press/v70/amos17a.html)|GPU上の制約付きQPと環境間バッチ処理|本実装はADMM等の独自の近似解法で、OptNetの内点法そのものではない。CPU最適解一致を自動的に保証しない|
|[Ross et al., DAgger (2011)](https://proceedings.mlr.press/v15/ross11a.html)|現在のGPU軌跡が訪れる状態をCPUで別途ラベル付けし、辞書・学習データを集積|シミュレータ代理モデルへの応用であり、原論文の保証がそのまま1%誤差保証になるわけではない|
|[Hallmann et al., gsMOBO (2026)](https://spj.science.org/doi/full/10.34133/csbj.0072)|培地・増殖・生産・費用を扱う外側探索という役割を再確認|内側FBAをGPU化する手法ではない。今回BO部分は変更していない|

## 実装した変更

1. 現行構成（λ=30、K=2,048、最大2,000反復）のGPU軌跡から、独立した3 seeds
   （20294001、20295010、20296019）×120状態を採取。GPU不採用だった2状態も含めて、
   同一の上下限・共有培地・菌体量・目的関数でCPU HiGHS厳密解を付け直した。
   360状態を追加し、辞書を34,344件から34,704件にした。生ラベルとrow seedも保存した。
2. 追加データの学習重みを4とする再学習と、正規化した反応境界違反の二乗損失を
   加えた再学習を分けて比較した。閉じたPHB/PHV反応等の誤予測に学習時から勾配が働く。
3. 再学習の悪化を受け、予測器を変えない辞書拡張と、前回の正規化・重みを保持する
   低学習率の継続校正（100 epochs、学習率1e-5）を対照として追加した。
4. 菌種別の独立な混合係数をGPU QPへ導入した。菌種内の物質収支は維持し、
   共有栄養は引き続き菌種間で連成する。明らかに使えない境界面の候補を除く前処理も追加した。
5. 共有GPUサービスの古い目的関数モード固定を修正し、現在のPHB/PHV切替えモデル、
   複数出力QP、再試行に対応させた。大きい再試行は小分けし、空きVRAMに応じてバッチ上限を抑える。
6. 辞書の凸結合に限定しない符号付きアフィン投影を診断用に試作した。
   凸結合なら省略できた制約も検査対象に戻し、全化学量論行列の残差をGPUで確認する。
   小規模診断では後半状態の収束不足が残り、オンライン既定経路へは組み込んでいない。

## 同一操作列での120ステップ比較

dt=0.2 h、24 h相当、初期NH4=0.05 mM、開発用seed=20260913。
これは強化学習そのものではなく、固定したランダム操作列でのdFBA比較である。
判定基準：PHA終点誤差≤1%、菌体量最大誤差≤0.01 g/L、3HV分率誤差≤0.01、
120/120 GPU採用・オンラインCPU LP 0回。各条件は1試験で、信頼区間は付けない。

""" + "\n".join(table) + """

独立混合だけはメモリ・時間を抑えたK=512であり、他のK=2,048と同じ条件ではない。
その悪化を、独立混合という構造全般の否定と解釈しない。
不採用stepがある軌跡の終点値は、失敗を含む診断値であって有効な代替シミュレーションではない。
今回の秒数は開発中のGPU実行時間としてJSONへ残したが、一部は他の診断・学習と重なる。
CPU参照も既存の同一条件の終点値であり、同一負荷の速度比較・購入根拠には使用しない。

![比較図](../results/pf_literature_refinement.png)

図：破線は事前基準。誤差は小さいほどよい。(a)は対数軸。
丸印は120/120 GPU採用、×印は不採用stepを含む診断値。n=1の開発用比較で、エラーバーはない。
各実行JSONに菌種別終点値・失敗回数・実装ハッシュを保存している。

## 学習指標と並列経路

""" + f"""境界違反の正規化RMSEは、同じサンプル分割で
{training_a['validation_scaled_bound_violation_rmse']:.6f} → {training_b['validation_scaled_bound_violation_rmse']:.6f}
へ低下した。ただし、これはランダムサンプル分割の指標であり、独立時系列の精度保証ではない。

最終共有GPUサービス試験：{service['requests']}要求、可行率{100*service['feasible_fraction']:.1f}%、
最大実測バッチ{service['service']['max_observed_batch']}、オンラインCPU LP
{service['service']['online_cpu_lp_stage_calls']}回。現行live-objectiveモードで確認した。
これは問い合わせ単位のsmoke testで、PPO全体のスケーラビリティ試験ではない。
""" + """

## 結論と残る作業

実装・回帰テストと並列経路の修正は実施した。しかし、全精度基準を満たしたと認定していない。
継続校正ではPHA終点誤差が前回の0.513%から0.276%へ低下した一方、
3HVモル分率誤差は0.001307から0.030078へ悪化した。菌体量最大誤差も0.026461 g/Lで基準外である。
したがって、この校正を総合的な精度改善とは扱わず、採用を見送る。
失敗した構成を本番既定へ昇格させず、前回の比較結果と新しい負の結果を両方保持した。
最終検証用5 seeds（20286001〜20286005）は未使用。開発基準を通過してから評価する。

次の主要課題は、制約が厳しくなる状態での機構QPの収束と、補正後の出力・
正味の栄養移動・菌体増殖を意識した学習である。辞書件数だけ、あるいは単発の
ニューラルRMSEだけを改善しても、逐次誤差やCPU LPの最適性までは保証できない。
CPU LPをGPU推論中に隠れて呼ぶ、誤差基準を緩める、検証状態を教師に混ぜる、
PHA終点に補正式を掛ける、といった操作は行っていない。

オンラインFBAの数値処理はGPUだが、環境更新・Python制御・通信はCPUに残る。
生物学的に3種の共存が実証されたことを意味する結果でもない。

関連テストは54件通過、8件の警告（疎行列API・fork）があった。リポジトリ全体のテストではない。
最終結果は `results/pf_literature_pytest.log`、実装計画は
`docs/GPU_LITERATURE_REFINEMENT_PLAN_20260903.md` を参照。
"""
    (ROOT/"docs/GPU_LITERATURE_REFINEMENT_REPORT_20260903.md").write_text(text, encoding="utf-8")
    (ROOT/"results/pf_literature_comparison.json").write_text(json.dumps(
        {"status":"experimental_not_qualified", "seed":20260913, "rows":rows}, indent=2))
    print("\n".join(table))


if __name__ == "__main__":
    main()
