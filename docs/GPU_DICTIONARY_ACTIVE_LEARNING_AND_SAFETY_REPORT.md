# GPU dictionary refinement and scientific-safety report

## 結論

32,768候補辞書は、厳密HiGHS解を保存した実行可能候補集合としては妥当である。しかし、最近傍候補を現在状態の厳密最適解として置換する方式は、候補数、特徴重み、境界追加、凸補間、周期的CPU実行を変更しても、事前に定めた24 h PHA相対誤差1%以下を満たさなかった。

このため、辞書の単純拡大を本番化せず、未認定辞書を科学評価で指定した場合はGPUへロードせずCPU HiGHSへ自動的に差し戻す資格ゲートを実装した。GPU辞書は、明示的に許可した近似探索およびRL訓練rolloutだけに限定し、評価軌跡と報告値は厳密HiGHSで計算する。

## 先行研究から採用した原則

1. Song et al. (2025) はFBA反復をANNで置換して高速な代謝切替シミュレーションを示したが、炭素源ごとに代理モデルを構築し、交換フラックスを対象として訓練・検証している。本系のような3-GEM・20,200次元contextに対し、全フラックス最近傍を一つの距離で置換する設計とは異なる。
2. Fan et al. (ICML 2023) は学習モデルでLPの初期basisを予測し、線形代数検査と厳密simplex solverを組み合わせている。学習予測を最終解そのものではなく厳密解法の加速に使う構成である。
3. Sambharya et al. (L4DC 2023) も、学習器の出力を反復最適化のwarm startとして使用し、最終解は最適化アルゴリズムで得ている。
4. HiGHSはbasisの再利用と `setBasis` を公式Python例で提供している。したがって将来の厳密高速化は、辞書フラックスの直接採用より、活性basis予測とHiGHS warm startを優先する。

参考:

- https://doi.org/10.1038/s41598-025-89997-9
- https://proceedings.mlr.press/v202/fan23d.html
- https://proceedings.mlr.press/v211/sambharya23a.html
- https://github.com/ERGO-Code/HiGHS/blob/master/examples/call_highs_from_python.py

## 実装・検証した方式

同じ現行3-GEM、同じ120-step操作列、RTX 4060 Laptopで比較した。

| 方式 | GPU採用率 | PHA相対誤差 | 高速化 | 判定 |
|---|---:|---:|---:|---|
| 32,768最近傍（再測定） | 97.50% | 2.807% | 2.746× | 不合格 |
| decision-aware強重み | 60.00% | 4.244% | 1.616× | 不合格 |
| 低NH4・高距離512件追加（33,280件） | 99.17% | 6.751% | 2.748× | 不合格 |
| 実行可能2近傍の凸補間 | 98.33% | 2.911% | 2.806× | 不合格 |
| 実行可能4近傍の凸補間 | 98.33% | 3.515% | 2.714× | 不合格 |
| 実行可能8近傍の凸補間 | 98.33% | 4.113% | 2.758× | 不合格 |
| 4-stepごと厳密HiGHS | 100.00% | 3.604% | 1.943× | 不合格 |
| 8-stepごと厳密HiGHS | 100.00% | 2.408% | 2.327× | 不合格 |
| 16-stepごと厳密HiGHS | 100.00% | 2.130% | 2.628× | 不合格 |
| 32-stepごと厳密HiGHS | 99.15% | 2.141% | 2.781× | 不合格 |

候補追加でGPU採用率が上がってもPHA誤差が悪化したため、辞書密度と動的出力精度は単調な関係ではない。FBAには複数の最適・準最適フラックスが存在し、近傍追加で別の縮退枝が最近傍になると、微小な1-step差が120-stepで累積する。

## 最終的に採用した処理系

- artifactごとに小さなvalidation manifestを置く。
- `status = qualified` が明示されたartifactだけを科学評価モードでGPUへロードする。
- manifest欠落、破損、`experimental_not_qualified` はGPU試行前にCPU HiGHSへ差し戻す。
- 診断へ `surrogate_disabled_reason` を記録する。
- 近似GPU辞書を使用するベンチマーク・RL訓練は、コード上で明示的に資格ゲートを解除する。
- RLの評価軌跡、上位候補、図表、論文値はCPU HiGHSで再計算する。

実際の統合確認では、未認定32,768辞書を指定しても `backend = scipy`、GPU試行0回、`surrogate_disabled_reason = validation_status:experimental_not_qualified` となった。

## 次の技術候補

厳密性を維持したGPU/ML高速化として、次は全フラックス辞書置換ではなく、厳密HiGHSへ渡すbasisまたは初期解の予測を試す。現環境にはhighspy 1.15.1が導入済みである。basis warm startの有効性を独立benchmarkで確認し、厳密最適性を保持したまま速度向上が得られた場合のみスクリーニング経路へ採用する。

## 検証

- 全pytest: 84 passed、27 warnings
- 新規資格ゲートの統合確認: CPU HiGHSへ自動差し戻し
- 未認定manifest:
  - `models/cooperative_surrogate/cooperative_dictionary_32768.validation.json`
  - `models/cooperative_surrogate/cooperative_dictionary_33280_active.validation.json`
