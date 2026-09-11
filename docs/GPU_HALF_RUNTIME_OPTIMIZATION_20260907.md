# 約50%短縮を目標とした構造最適化の結果

## 結論

初回の構築・求解・検証は約半分に短縮できたが、**準備後の繰返し計算の半減は未達**。
CPU16並列に対する速度優位も未達である。長時間PPO学習の41%短縮を意味しない。

32個の異なる環境、保存maxmin LP入力の時刻2～8、各方式3反復の中央値。
入力読み込みは計時外。sequence lifecycleは初回構築、更新、求解、独立検証、終了処理を含む。
PPO全体、全LP stage、GPU出力から次の状態を生成する閉ループではない。

| 計時区間 | 従来GPU block diagonal | 証明再利用GPU | CPU16並列 | GPU変更の効果 |
|---|---:|---:|---:|---|
| 初回準備のみ | 17.095 s | 7.014 s | — | 59.0%短縮 |
| 初回準備＋最初の求解・検証 | 21.541 s | 10.889 s | 0.763 s | 49.4%短縮 |
| 初回準備込み7時刻のlifecycle | 25.563 s | 15.018 s | 3.041 s | 41.3%短縮 |
| 準備後の6時刻合計 | 3.413 s | 3.470 s | 2.229 s | 改善なし（中央値1.7%増） |

各方式のlifecycle範囲は、従来GPU 24.919～26.151秒、新GPU 14.663～15.276秒、
CPU16 3.032～3.102秒。これは実測最小・最大であり95%信頼区間ではない。
順序をローテーションした3反復の開発測定で、無作為化試験ではない。

## 実装した構造変更

`--reuse-equality-proofs`で有効化する。PPOのCPU既定やGPUの数値精度は変更していない。

1. 等式forestの構築証明を同一batch内で再利用する。等式行列・RHS・形状が完全一致する
   ことを検査する。hashは検索だけに使い、実配列の比較も行う。各環境のplanは独立所有する。
2. zero-faceの証明に既存の厳密rebind検査を使う。境界の符号、固定値、等式、proofの整合性を
   現在の入力で再検査する。不適合なら新規証明を構築する。現在の数値と境界witnessは別々に作る。
3. 独立の再証明でも、現在入力から新しく構築したcanonical forestの重複構築を省く。
   元planとcanonical planの行列・座標、各環境の縮約LP・witnessの照合自体は残す。

これは初回のCPU側構造準備の最適化であり、GPU算術の高速化とは区別する。
GPU経路のCPU LP呼出しは0だが、host入力のpack/hash、初回証明、制御、独立監査が残るため、
完全GPU内完結とも呼ばない。

主要実装: `src/lp_equality_reduction.py`, `src/gpu_forest_ipm.py`,
`src/gpu_zero_face_ipm.py`, `src/gpu_forest_map.py`, `src/gpu_ipm_device_update.py`。

## 精度と比較条件

- 元LPの主実行可能性≤1e-5、双対違反≤1e-7、相対KKT gap≤1e-7、独立direct dual gap≤1e-7。
- 各方式224入力×3反復＝672件が全て合格。672件の独立な生物条件という意味ではない。
- 現在または未来時刻のCPU正解ベクトルはGPUへ渡していない。前時刻のGPU状態のみwarm startに使用。
- CPUはHiGHS 1.15.1、16 worker、各LPのmodel/basis reuseを使用。CPUを単一workerに制限しない。
- 元LP入力・ソース・精度閾値のhash/identity一致を集計時に検査した。
- 再利用対象の行列破損、誤ったwitness、変更された目的、hash衝突、異なる境界をテストした。

## 試したが採用しなかった数値解法

| 案 | 実測結果・採否 |
|---|---|
| 50% reserve補助LPから前時刻解を再利用 | 4環境×7時刻で再利用0回。全件認証したが毎回の再求解が必要で遅い |
| 等式アフィン空間内のGPU Motzkin投影 | cold 2,000反復で違反約0.0025、warm 5,000反復で約0.045が残り不合格 |
| AMDによる疎行列の並べ替え | 32環境hot2時刻で1.896/1.560秒。悪化のため既定にしない |
| cuDSS 0.7の別factor algorithm | 0.682/0.525秒。小幅な変化で半減ではない |
| FP32 factor/三角求解＋FP64残差補正 | toyは通過。実GEMの初回Newton検査は正則化1e-6/1e-4とも不合格 |

数値候補は元のLP・認証閾値を変えていない。不合格候補の短い実行時間は高速化実績に数えない。
FP32 factorはFP64原Newton残差を検査するglobalized経路に限定し、native buffer整合性、
変換overflow、非有限RHS、lane間分離を検査する。FP64は引き続き既定である。

参考文献・仕様（本モデルでの速度保証を意味しない）:

- [De Loera et al., Sampling Kaczmarz–Motzkin](https://arxiv.org/abs/1605.01418):
  線形不等式への投影の着想。本試作は全行greedy候補器で、論文の完全実装ではない。
- [Carson & Higham, 2018](https://doi.org/10.1137/17M1140819):
  低精度分解と高精度残差補正の着想。GMRESが速く収束する保証は本モデルでは得られていない。
- [cuDSS 0.7世代の設定仕様](https://docs.nvidia.com/cuda/archive/13.0.0/cudss/types.html):
  導入済み0.7.1 headerと照合して並べ替え・分解方式を指定。ライブラリ更新はしていない。

## 残る律速と次の判定基準

準備後は、元Newton系を満たすためのFP64疎行列factorと三角求解が残る。
初回準備の再利用を増やしても、長いPPO学習の支配的な費用は解消しない。
FP32や単純投影の不合格を受け、次は残差が落ちない方向を診断し、少数の難しい方向だけを
倍精度で補正する二段の前処理を検討する。原因を数値条件の悪さと断定せず、原Newton残差・
補正回数・総時間を確認する。GPU内FP64再分解が増えて費用を相殺するなら採用しない。

採用基準は同じ32環境・同じ元LP精度で、更新・求解・検証込みhot6時刻を旧目標1.762秒以下、
かつCPU16より短くすること。現時点の3.470秒では未達。全stage/閉ループ/PPOへの昇格は行わない。

## 再現用の記録

- 比較実行: `scripts/run_ipm_proof_compare.py`
- 整合性検査・集計: `scripts/summarize_ipm_proof_compare.py`
- 集計: `results/pf_ipm_proof_compare_summary_20260907.json`
- raw: `results/pf_ipm_proof_{baseline,proof_reuse,cpu16}_32_2to8_20260907_r{1,2,3}.json`
- rawにはソースsnapshotとhash、元LP入力hash、精度指標、全測定値を保存している。
- 最終回帰結果: `results/pf_ipm_half_target_final_regression_20260907.xml`
- 固定ソースでの最終回帰: **1,594 passed / 4 skipped / 11 warnings、34.33秒**。
  skipは別GPUを必要とする既存テスト。warningsは既存のSciPy option伝達と意図的なCSR変更テスト。
