# GNN+GRU Supervised Correction and GPU Acceleration: Integrated Report

**要約:** GNN·GRUを用いた教師あり学習とGPU内点法/PDHGによるdFBA高速化を実装・検証した。学習・GPU推論・補正の接続は動作したが、実GEMの全3段階LPにおいてGPU補正が数値的に不安定で、元LP認証に至らないケースが多く、CPU速度超えは未達である。最新の結論は「前回GPU解の再利用」が現時点で最も効率的であり、GNN·GRUはそれを上回らなかった。学習データの増量や等式縮約による改善も、現状では補正反復の削減に寄与しておらず、構造共有型GPU内点法など新たな方針の検討が必要である。

---

## 1. 全体的な実施状況と最新結論

- **GNN+GRUの実装・学習・GPU補正パイプラインは構築され、実験的に動作する。** しかし、実GEM（OR16, NS21, P. freudenreichii）の全3段階LPにおいて、GPU補正で元LP認証を得ることはできず、CPU速度を超える高速化も達成していない。  
  （出典: GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md, 2026-09-05）

- **教師あり学習によりGNN・GNN+GRUモデルを訓練し、maxmin段階ではGPU補正で全件認証できたが、補正反復数が「前回GPU解の再利用」方式より多く、総時間で劣った。**  
  （出典: GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md, 2026-09-07）

- **等式保存型縮約（固定二項同次等式の消去）はCPU同値性検証で合格したが、縮約空間でのGRU+PDHG補正はexchange段階で0/164合格と失敗。** 精度維持とCPU超えは未達。  
  （出典: GPU_EQUALITY_REDUCTION_RESULTS_20260905.md, 2026-09-05）

- **固定辞書の拡張によるアプローチは停止し、構造共有型GPU内点法など、辞書に依存しない全空間直接解法への移行を検討中。**  
  （出典: DICTIONARY_FREE_GPU_APPROACH_REVIEW_20260905.md, 2026-09-05）

- **学習データの大規模化（512軌道〜1024軌道）と学習曲線プロトコルは設計済みだが、未実施。** データ増量が補正反復削減に寄与するかは未確認。  
  （出典: GRAPH_LEARNING_CURVE_PROTOCOL_20260907.md, 2026-09-07）

---

## 2. アーキテクチャと実装

### 2.1 GNN・GRUモデル

- **入力グラフ構造:** 変数–制約二部グラフ。LPの係数A、rhs、上下限、目的関数から動的に構築。反応・代謝物に加え、成長率、共有培地、parsimony等の補助変数・制約も保持。  
  （出典: GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md）

- **GNN構成:** hidden=32, message passing=2 rounds, 12,802パラメータ（GNNのみ）。符号付きmessage passing。  
- **GNN+GRU構成:** 25,474パラメータ。環境別のGRU hidden state (hidden=64, 1層) を持ち、前時刻の認証済みlatentを入力に予測。  
  （同上）

- **出力:** 全変数の主解xと全行の双対解yを直接出力（PCA圧縮は使用せず）。上下限制約と不等式双対の符号は出力変換で満たすが、物質収支と最適性は保証しない。  
  （出典: GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md; GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md）

### 2.2 GPU補正パイプライン

1. **候補生成**: GNN/GNN+GRUまたは前回CPU解などから主双対候補を生成。
2. **GPU受け渡し**: DLPack経由でTorchテンソルをCuPy配列として共有。
3. **GPU内点法/PDHG補正**: 全変数空間でprimal-dual内点法（cuDSSを利用したNewton系）または対角前処理付きPDHGを実行。
   - 内点法: 4環境バッチでcuDSS 0.7のsymbolic解析を再利用、反復ごとに数値分解。predictor/correctorは同一分解を再利用。
   - PDHG: FP64, omega=1（安全積固定）、最大2048反復、256反復ごとに認証。
4. **元LP認証**: 主残差≤1e-5、双対違反≤1e-7、相対KKT gap≤1e-7。全条件満足で合格。
5. **履歴管理**: 環境ID・段階・モデルhashによる分離。未認証候補は解キヤッシュに入れず、hidden stateも更新しない（GRUの場合はpropose/commmit分離）。
   （出典: GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md; GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md; GPU_EQUALITY_REDUCTION_RESULTS_20260905.md）

### 2.3 安全性と回帰テスト

- 環境ID・段階・モデル/グラフidentityによる履歴分離、reset後の古いcommit禁止、異なる提案とのtoken取り違え禁止、入力変更/順序変更の拒否、非有限値拒否。
- 回帰テスト: 最終報告時点で1664 passed, 4 skipped (GNN+GRU), 829件独立合格（等価縮約回帰）など。
  （出典: GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md; GPU_EQUALITY_REDUCTION_RESULTS_20260905.md）

---

## 3. 教師あり学習の詳細

### 3.1 学習条件

| 項目 | 値 |
|---|---|
| 対象GEM | OR16, NS21, P. freudenreichii (固定) |
| 学習データ | 12環境×60時刻 = 720件 (maxmin段階のみ) |
| 教師生成 | offine CPU计算 (HiGHS), 全LP元LP認証完了 |
| モデル選択 | 別4環境×60時刻 = 240件 |
| 入緯 | 現在LP全係数・上下限・目的 (変数/制約二部グラフ, hidden=16, message passing 2回) |
| パラメータ数 | GNN: 191,890; GNN+GRU: 195,154 |
| 損失 | train-only RMS (下限1) 正規化 + 元単位最大行違反 + 物質収支/不等式違反/目的値損失 |
| 学習epoch | 160 epoch (選択epoch: GNN 130, GNN+GRU 90) |
| 学習時簡 | GNN: 184.1秒, GNN+GRU: 197.4秒 (前処理含395.6秒) |
| GPÚ | RTX 4060 Laptop (PyTorch 2.11.0+cu130) |
| 訓練seed | 独立 (learning curve protocol参照) |

（出典: GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md; 補足: GRU_GPU_PILOT_RESULTS_20260905.md）

### 3.2 初回学習と第二試験

- **初回:** 標準偏差（下限1e-3）正規化、60epoch。開発損失が変動の小さいfluxに支配され、開発速度試験でバッチ分解ラウンドが前解19→GNN35→GNN+GRU46に増加。
- **第二試験:** train-only RMS（下限1）正規化、元単位最大行違反損失を追加。160epoch。選択epoch: GNN130, GNN+GRU90。

（出典: GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md）

---

## 4. GPU補正実験の結果

### 4.1 GNN推論費用（実GEM入力、exchange段階）

| 構或 | パラメータ数 | 同期付きwall中央値 | CUDA event中央値 |
|---|---|---|---|
| GNN | 12,802 | 2.935 ms | 2.882 ms |
| GNN+GRU | 25,474 | 3.626 ms | 3.528 ms |

* 入力グラフ構築・転送は別枠0.641秒。Torch peak allocated ~84.5 MiB。 （出典: GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md）

### 4.2 GPU内点法の基礎性能（exchange、cold、4環境）

- 初期Newton行列: 27,799次元, 118,993非零要素。
- 数値分解＋1回solveの3呼出し: 0.0980, 0.0265, 0.0119秒。
- 最大絶対残差 3.87e-12〜1.09e-11、解誤差 7.60e-7〜1.12e-6。
- LP反復でslack/dual比が変化するとNewton系精度悪化。対称スケーリング3回と正則化追加後も不十分。
- 結果: 各段階とも0/4元LP認証。終了理由はNewton方向の残差超過。

| 段階 | 完了更新回数/分解試行回数 | 補正wall | 元LP認証 |
|------|---------------------------|----------|----------|
| maxmin | 24/25 | 1.188秒 | 0/4 |
| aggregate | 26/27 | 1.226秒 | 0/4 |
| exchange | 30/31 | 1.395秒 | 0/4 |

* setup時間別途1.273〜1.100秒。 （出典: GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md）

### 4.3 GPU PDHG補正（cold/GRU比較、exchange段階）

*GRU + PDHG（2048反復、256反復毎認証）の4環境×41ステップ:*

| 方式 | 164 LP時間 | 元LP合格 |
|------|-----------|---------|
| CPU persitent HiGHS | 10.166秒 | 164/164 |
| 等価縮約＋cold PDHG | 29.926秒 | 0/164 |
| 等価縮約＋GRU+PDHG | 29.964秒 | 0/164 |

* GRU候補の元LP主残差（バッチ最大）2.949-3.530。補正反復が時間の約92%を占める。 （出典: GPU_EQUALITY_REDUCTION_RESULTS_20260905.md）

### 4.4 学習済みGNN/GNN+GRU + GPU補正の比較（maxmin段階、hot24LP）

| 方式 | 補正反復/LP | バッチ分解ラウンド合計 | hot24LP総時間 | 速度比（前解=1） |
|---|---|---|---|---|
| 前回GPU解再利用 | 7.46 | 48 | 1.645秒 | 1.000 |
| 学習済GNN | 17.38 | 110 | 3.350秒 | 0.491 |
| 学習済GNN+GRU | 19.71 | 123 | 3.721秒 | 0.442 |

* 全72/72件が元LP認証通過（GPU補正による）。生候補での直接認証は0/72。
* GNN+GRUのグラフ準備＋推論はhot6時刻合計0.092秒（総時間の2.5%）。
* 補正前の元LP primal残差（バッチ最大）2.949–3.530。 （出典: GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md）

### 4.5 BLOCK-diagonal GPU診断（cuOpt PDLP/Barrier）

初期診断（4環境、cold）:

| step | 段階 | CPU秒 | GPU秒 | 元LP採用 |
|------|------|-------|-------|---------|
| 1 | maxmin | 0.159 | 1.221 | 0/4 |
| 1 | aggregate | 0.249 | 3.030 | 0/4 |
| 1 | exchange | 0.365 | 3.035 | 0/4 |
| 41 | maxmin | 0.193 | 0.900 | 0/4 |
| 41 | aggregate | 0.329 | 3.030 | 0/4 |
| 41 | exchange | 0.491 | 3.033 | 0/4 |

* maxminはcuOpt内部optimalだが外部相補性gap基準超過。後段は時間上限。 （出典: GPU_TEMPORAL_AND_BLOCK_STRATEGY_20260905.md）

**maxmin解析的双対証明追加後（4環境）:** 全件採用（32/32）するもGPU 1.063秒 vs CPU 0.175秒（step1）で速度優位なし。 （同上）

**32環境追試:** maxminは解析証明で全件採用、後段は時間上限で0/32。GPUはCPUより遅い。 （同上）

### 4.6 固定配列入力（`--frozen-inputs`）の効果

- 4環境×8ステップ2組でPHA・菌体量完全一致、CPU LP数・simplex反復数一致、元LP証明全件合格。
- 従来→配列版: CPU 4.993→4.148秒, Hybrid 5.344→4.622秒（短縮13.5-17.7%）。
- しかしHybrid vs CPU: 配列版同士ではHybrid 4.622秒 vs CPU 4.148秒でCPU優位。 （出典: GPU_TEMPORAL_AND_BLOCK_STRATEGY_20260905.md）

---

## 5. 等式保存型縮約

### 5.1 構造監査

- 教師16×60×3＝2,880 LP + 独立診断36 LPの等式行列（4,651行）は全2,916件で同一hash、RHS全て0。
- 2非零行2,444本のうち独立forest 2,411本を消去、残り33本（閉路等）は保存。
- 消去对象: 同次等式 `E x=0`、疎行列Tで `x=T z, ET=0`。

### 5.2 縮約後のLPサイズ

| 段階 | 元の行×変数 | 縮約後の行×変数 | 変数削減率 |
|---|---:|---:|---:|
| maxmin | 5,051×6,734 | 2,640×4,323 | 35.8% |
| aggregate | 5,051×6,734 | 2,640×4,323 | 35.8% |
| exchange | 6,330×7,373 | 3,919×4,962 | ‍32.7% |

* 非零係数: exchange 31,501→24,397（22.6%減），復元写像T非零数7,373。

### 5.3 CPU同値性検証

- 参照主双対の縮約→復元→36/36元LP認証合格。
- 縮約LPを別途CPU HiGHSで求解→復元→36/36合格。
- 多重最適性によるflux差異があっても元LP制約＋最適性で判定。
  (出典: GPU_EQUALITY_REDUCIION_RCSULTS_20260905.md)

### 5.4 GPU補正結果（縮約空間PDHG）

- 4環境×41ステップexchange: CPU 10.166秒, 縮約cold PDHG 29.926秒, 縮約+GRU PDHG 29.964秒、いずれも0/164合格。
- 主残差は未消去の多項等式（例：1,193非零の行960）に残り、縮約と元LPの残差差は丸め誤ざ水準。
- 補正反復時間90%超、推論時間はごく僅か。
  (出典: 同上)

---

## 6. 性能比較とディスパリティに関する注記

- **GNN+GRU教師あり学習（maxminのみ）ではGPU補正により全件認証できたが、exchange段階では同一手法でも補正失敗**。これはLPの性質（等式・変数規模・active setの複雑さ）が異なるため。
- **等式縮約はCPU同値性を満たすが、GPU補正の収束改善には寄与せず**。縮約により変数削減率約35%だが、PDHG反復数が減らない（最大2048反復でも収束せず）。
- **「前回GPU解の再利用」がmaxmin段階で最速（補正反復7.46/LP）であり、学習モデルは反復増加により劣る**。学習データを増しても同傾向が変わる保証はない。
- **ディスパリティ（食い違い）**: 
  - GNN_GRU_SUPERVISED_CORRECTION_REPORTではhot24LPで全件認証成功（GPU補正後）。一方、GPU_EQUALITY_REDUCTION_RESULTSではexchange 164LPで全件失敗。これは対象段階の違い（maxmin vs exchange）による。
  - 同一ファイル内でも、内点法の初回試験（0/12)とPDHGの後続試験（0/164）が存在するが、いずれも補正未達で一貫。
  - 学習曲線プロトコルは未実施であり、データ増量が改善するかは未確認。

---

## 7. 学習データ大規模化プロトコル（未実施）

**目的**: データ量増加が補正反復削減に与える影響を系統的に評価する。  
**計画規模**: 32,64,128,256,512,1024軌道（1軌道120時刻, 3段階合計3,840〜368,640 LP）。  
**条件**: GEM固定、初期菌体量0.75-1.25倍、初期アンモニウム0.5-2倍、action範囲0.05-0.95。一様乱数/段階変化/ランダムウォーク/パルスを含む。  
**評価**: 同一LPで前解warm start, GNN, GNN+GRUの総時間、補正反復数、認証率を比較。2方向クラスターbootstrapで95%信頼区間。  
**速度目標**: 基準方式の2倍以上（時間半減）、区間下限が2倍超。連続2回のデータ増量で短縮率上限2%以下なら飽和候補。  
**現状**: 小規模trainer動作確認のみ。本格収集・学習・比較は未実施。  
（出典: GRAPH_LEARNING_CURVE_PROTOCOL_20260907.md）

---

## 8. 限界と今後の方向性

### 8.1 現時点での明確な限界

- **GPU補正（内点法/PDHG）がexchange段階で数値的に不安定。** スケーリング・正則化・反復改良を追加しても、元LP証を満たす収束が得られない。
- **学習候補からの直接認証は0件。** 補正に頼る限り、補正反復数が律速であり、推論時間の短縮だけでは高速化できない。
- **「前回GPU解の再利用」が現状最速。** 学習モデルはそれを上回っていない。
- **完全GPU内完結（host制御無し）には至らず。** hostでの配列準備・停止判定・元LP認証が残る。
- **全3段階・120step閉ループでの比較は未実施。** PHA終点誤差、菌体量、PHV分率の維持も未検証。
- **PPO統合は行われていない。** 環境計算の高速化未達のため。

### 8.2 文献との関係

| 文献 | 採用部品 | 注意点 |
|---|---|---|
| Chen et al., ICLR 2023 (LP-GNN) | 変数–制約二部グラフ | 本実装は有限データで厳密解を保証するものではない |
| FlowGAT, 2024 | 代謝網構造利用の着想 | FBA代替ソルバの実績ではない |
| Sambharya et al., JMLR 2024 (warm-start + fixed-point) | NN初期値＋反復最適化の枠組み | 本GEMや内点法速度への転用実績はない |
| DC3, Donti et al., ICLR 2021 | 部分予測→等式補完→不等式補正 | 本実装の等式保存層は限定版、DC3の完全再現ではない |
| cuPDLP.jl, Lu and Yang | GPU再始動PDHG | 本モデルでの速度は別途実測が必要 |
| rHPDHG/cuPDLPx | 再始動・primal weight制御 | 旧PDHG失敗がこれら全方式の不可能性を証明しない |

（出典: GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md; GPU_EQUALITY_REDUCTION_RESULTS_20260905.md; GRU_GPU_IMPLEMENTATION_PLAN_20260905.md）

### 8.3 次フェーズの優先順位

1. **構造共有型GPU内点法（cuDSS batch）の評価:** 同一CSR superpattern・係数変化のbatchで、数値分解の安定性・費用を測定。安定かつ軽ければ内点法を優先、重ければ高度化PDLGを再評価。
2. **GPU simplexの限定比較:** 双対実行可能な初期basisが確保できる場合のみ、dynamic dual simplexを検討。
3. **coldで収束する解法を確立:** NN/GRU無しで元LP基準を満たす解法を先に確立し、それから初期化手法を比較。
4. **閉ループ総費用評価:** 4×3で機能確認後、32×8での強いCPU対照と比較。CPU超えなければ120step・PPOへ拡大しない。
5. **学習目的の見直し:** 補正後残差・必要反復数を損失に含めるresidual型候補器の検討。全flux置換方式は継続しない。

（出典: DICTIONARY_FREE_GPU_APPROACH_REVIEW_20260905.md; GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md; GRU_GPU_IMPLEMENTATION_PLAN_20260905.md Rev 7）

---

## 9. 構成元ファイル

- `docs/GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md` (2026-09-05)
- `docs/GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md` (2026-09-07)
- `docs/GRAPH_LEARNING_CURVE_PROTOCOL_20260907.md` (2026-09-07)
- `docs/GRU_GPU_IMPLEMENTATION_PLAN_20260905.md` (2026-09-05)
- `docs/GRU_GPU_PILOT_RESULTS_20260905.md` (2026-09-05)
- `docs/GPU_TEMPORAL_AND_BLOCK_STRATEGY_20260905.md` (2026-09-05)
- `docs/GPU_EQUALITY_REDUCTION_RESULTS_20260905.md` (2026-09-05)
- `docs/DICTIONARY_FREE_GPU_APPROACH_REVIEW_20260905.md` (2026-09-05)
