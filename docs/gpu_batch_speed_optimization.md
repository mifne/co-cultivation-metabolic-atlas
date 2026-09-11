# GPU加速dFBA/LPソルバー：バッチ処理最適化の統合レポート

本レポートは、GPUバッチ処理によるdFBA/LPソルバーの速度最適化に関する一連の開発結果（2026-09-06～2026-09-07）を統合したものである。初期の基本性能測定ではGPUがCPU 1 workerより約15%速いことを確認したが、CPU 4 workersには約2.9倍遅く、その後CPU 16 workersとの比較では常にGPUが劣る結果となった。時刻間再利用、静的構造再利用、デバイス数値更新、証明再利用などの最適化により初回準備時間は最大59%短縮されたが、反復計算（hot求解）の半減は達成できず、CPU 16並列（HiGHS 1.15.1、16 workers、basis再利用）に対する速度優位は依然として未達である。すべての実験は保存済み開発用LP入力（maxmin段）の再生検証であり、閉ループdFBAやPPO全体の性能を保証するものではない。

---

## 1. 基本性能測定 — 同一32入力の速度比較 (2026-09-06)

**(出典: GPU_BATCH_SPEED_RESULTS_20260906.md, 2026-09-06)**

対象は現行GEMから保存した開発用LP入力 `results/pf_lp_trace_dev32x41_20260905`、physical step 1、maxmin段、異なる32環境。GPUはRTX 4060 Laptop、cuDSS uniform batch 32。元の参照x/yは未読取。

実験条件: balanced initialization、PC予測fraction 0.995、globalized full forcing 0.1、Krylov 16、exact equalities、GPU融合MGS/Givens、device finite check、cuDSS内部refinement 0。外側の元Newton残差と元LP精度閾値は維持。

| 実装 | 32 LPの秒数（中央値） | 認証済みLP/秒 | 反復 |
|---|---:|---:|---:|
| GPU新方式・setup除外 | 3.572876 | 8.96 | 3 |
| CPU HiGHS 1 worker・fresh model | 4.220701 | 7.58 | 3 |
| CPU HiGHS 4 workers・fresh model | 1.239123 | 25.82 | 3 |

GPUは3.908860/3.457349/3.572876秒で96/96解が元LP基準を通過。CPUも全試行通過。中央値比でCPU 1 workerに対して1.181倍、約15.35%の時間短縮。CPU 4 workersには約2.883倍遅い。n=3の開発測定で、統計的有意性/広い条件への一般化は主張しない。

GPU constructor/setupには別途8.54～9.10秒を要しており、cold総時間ではCPU超えではない。

測定ファイル（`results/`配下）:
- `pf_ipm_microall_device_ir0_maxmin32_20260906.json`
- `pf_ipm_microall_device_ir0_maxmin32_repeat_20260906.json`
- `pf_ipm_microall_device_ir0_maxmin32_repeat3_20260906.json`
- `pf_ipm_cpu_maxmin32_speed_recheck_20260906.json`

### 1.1 並列化済みでも遅い理由

repeat3の3.572876秒の内訳: 分解0.924743秒（25.9%）、三角solve/Krylov/更新2.402237秒（67.2%）、元LP認証0.147187秒（4.1%）。外側反復38回、batched solve 193回。受理反復は31～38。factorだけ無限に高速化しても約2.65秒残り、CPU 4 workersに届かない。

### 1.2 今回の修正と採否

- 内部cuDSS solveのfinite判定をdeviceに残してhost読出し削減。無効laneはNaNで返し、元のゲートが拒否。
- 2-pass逐次MGSとGivens/後退代入を融合。単独では32環境5.596秒に留まった。
- cuDSS内部固定2回の補正と外側の元Newton補正を分離。内部0回が上記3.57秒の主な改善。
- 有理数で有限境界上の差を検証した1行作業緩和を限定実装。
- 直接双対目的差と等式dual×残差の監査を追加。
- 等式のみ2冪でlossless正規化する実験（exchange4環境は0/4認証。既定化しない）。
- cuOpt native barrier方式は20.13秒でtime limit、0/4認証。

### 1.3 追加測定と実装

| GPUの環境数 | solve秒 | 認証済みLP/秒 | 測定 |
|---|---:|---:|---|
| 8 | 1.787036 | 4.48 | 単回・8/8認証 |
| 16 | 2.712076 | 5.90 | 単回・16/16認証 |
| 32 | 3.572876 | 8.96 | 前述3回中央値・96/96認証 |

追加ファイル: `pf_ipm_batchscale8_maxmin_20260906.json`、`pf_ipm_batchscale16_isolated_maxmin_20260906.json`。同じtraceの先頭8/16/32環境の入れ子部分集合。8→32で1秒あたりの処理数は約2倍。

2段目forestを `--second-forest` として接続。実GEM exchange4の新作業系は16,895次元（従来17,192）で7.341秒、0/4認証。この数値を有効解までの高速化とは扱わず、オプションのまま保持。

### 1.4 再開時の復元

再開時、`gpu_zero_face_ipm.py`、`gpu_forest_ipm.py`、`probe_graph_ipm.py`が空だった。前2本はnative barrier測定、後1本はrepeat3測定の保存済みsource_snapshotsからapply_patchで復元。復元後の関連テスト304件合格・1件skip。

---

## 2. CPU比較の強化と単項等式 (2026-09-06, Revision 23)

**(出典: GPU_BATCH_SPEED_RESULTS_20260906.md, 2026-09-06)**

CPU 4並列を最良値とする前提を取り消し、1/2/4/8/10/16 workersを調査。

HiGHS 1.14の同一process内連続測定はnative allocatorエラーで異常終了。完了した性能比較として採用しない。

HiGHS 1.15.1を別置き（`tmp/highspy-1.15.1-comparison`）して測定。同一32入力、各設定cold 3試行と同一入力hot 1回/試行、数値retry 0。

| CPU worker数 | cold 3試行中央値（秒） |
|---|---:|
| 1 | 4.4460 |
| 2 | 2.3732 |
| 4 | 1.3933 |
| 8 | 0.9312 |
| 10 | 0.8747 |
| 16 | 0.7644 |

保存: `results/pf_ipm_cpu_highs1151_owner_maxmin32_sweep_20260906.json`。

**食い違いに関する注記:** 上記CPU 1 workerの中央値4.4460秒は、第1節のCPU 1 worker中央値4.220701秒（fresh model）と異なる。これはHiGHSバージョン（1.14 vs 1.15.1）および測定方式（同一process連続 vs 個別process cold試行）の違いによる。両者とも開発診断であり、統一された条件での比較ではない。

### 2.1 GPUのopt-in処理

- GMRES中間host判定3箇所を抑制。残差・finite・active maskを保持。
- ゼロRHS単項等式a*x=0から、ゼロを跨ぐ境界の変数もゼロ固定。
- 2段目forestと厳密等式縮約へ接続。

| GPU構成 | maxmin32 solve（秒） | 認証 | setup（秒） |
|---|---:|---:|---:|
| 中間判定削減のみ | 3.5041 | 32/32 | 8.8331 |
| 単項等式＋2段目forest＋中間判定削減 | 3.2789 | 32/32 | 12.8038 |

どちらも単回の開発診断。新CPU 16 workersより速いとは言えない。後者のexchange4は8.0747秒で0/4認証、相補性gap 4.63e-6～1.87e-5で基準未達。

保存: `results/pf_ipm_deferred_control_maxmin32_20260906.json`、`pf_ipm_singleton_second_exact_maxmin32_20260906.json`、`pf_ipm_singleton_second_exact_exchange4_20260906.json`。

新しいGPU単項等式の15テストは全合格。

---

## 3. 同じ32 LPを複数streamへ分割した実測 (2026-09-06)

**(出典: GPU_BATCH_SPEED_RESULTS_20260906.md, 2026-09-06)**

新規実装: `src/gpu_stream_partition.py`、`scripts/benchmark_gpu_stream_partitions.py`、`tests/test_gpu_stream_partition.py`。独立cuDSS handle/data/buffer/streamを同一host threadのgreenletから協調実行。

| 配置・切替方式 | 全32問題完了（秒） | 認証 |
|---|---:|---:|
| 1x32、直接呼出し | 3.6260 | 32/32 |
| 1x32、factor/solve後に協調切替 | 3.8101 | 32/32 |
| 2x16、factor/solve後に協調切替 | 3.9685 | 32/32 |
| 4x8、factor/solve後に協調切替 | 6.1368 | 32/32 |
| 2x16、factor後のみ切替 | 4.2667 | 32/32 |
| 4x8、factor後のみ切替 | 6.4023 | 32/32 |

各1試行の開発診断。全て同じtrace/step1/maxminの32個の異なるproblem hash。この結果から4 streamを採用せず、stream分割の優先度を下げる。

| 協調shard数 | batch factor呼出し | batch solve呼出し | 環境単位のsolve数 |
|---|---:|---:|---:|
| 1 | 38 | 195 | 6,240 |
| 2 | 70 | 322 | 5,152 |
| 4 | 138 | 597 | 4,776 |

保存: `results/pf_gpu_stream_partition_maxmin32_pilot_20260906.json`、`pf_gpu_stream_partition_maxmin32_direct_20260906.json`、`pf_gpu_stream_partition_maxmin32_factor_only_20260906.json`。

最終回帰試験（43ファイル）: 772 passed・1 skipped（14.03秒）。PPO全体のCPU超えは未達。

---

## 4. 時刻間の数値更新・静的構造再利用と再始動比較 (2026-09-06)

本セクションは複数のファイルを統合している。最初に初期のslack修復方式と内部状態再利用の結果、次に静的構造再利用と再始動比較、最後に現在LPベース再初期化の最新結果を時系列順に示す。

### 4.1 初期の時刻間再利用（slack修復方式）

**(出典: GPU_BATCH_SPEED_RESULTS_20260906.md「続報」、および GPU_NUMERIC_REUSE_RESULTS_20260906.md 前半)**

native workspace/analysisを保持し、現在LPの数値と全段の証明データを更新する処理を実装。

初期のslack再構成方式: 4環境step2/3/4全認証、32環境step4に1問題未認証。CPUにも時刻間basis再利用を許すと、32環境step3はCPU 0.4921秒に対しGPU 3.6467秒、更新・独立確認込み6.8657秒。

### 4.2 旧内部dualを破棄する現在LPベースの再初期化（最新方式）

**(出典: GPU_NUMERIC_REUSE_RESULTS_20260906.md 後半、および GPU_STATIC_UPDATE_RESULTS_20260906.md)**

`src/gpu_ipm_reoptimization.py`と`--restart-mu 1e-4`を追加。旧巨大y/zを捨て、現在costからbound-dual候補を構成する。

32環境×3時刻×3試行の288問題すべてが元認証に合格。CPU側も同一入力288問題が合格・retry 0。

3試行中央値:

| step | GPU解法 (s) | GPU構築/更新・確認込み (s) | CPU16並列の更新・解法・確認込み (s) |
|---:|---:|---:|---:|
| 2（初回構築） | 4.9110 | 16.2989 | 0.7469 |
| 3 | 0.9229 | 4.0956 | 0.5067 |
| 4 | 0.7534 | 3.7944 | 0.3832 |

GPU数値更新単体中央値はstep3で3.1013秒、step4で2.9685秒。

### 4.3 静的構造再利用（GPU数値更新の静的構造）

**(出典: GPU_STATIC_UPDATE_RESULTS_20260906.md)**

`--reuse-numeric-workspace --reuse-static-forest --direct-kkt-payload --restart-mu 0.00001` で有効化。

| 同じ32環境の処理 | 旧GPU 更新等込み | 新GPU（静的再利用）更新等込み | 新GPU解法のみ | CPU16 更新・解法・検証 |
|---|---:|---:|---:|---:|
| step2（初回） | 16.299 | 16.179 | 4.817 | 0.735 |
| step3 | 4.096 | 1.722 | 0.646 | 0.505 |
| step4 | 3.794 | 1.485 | 0.522 | 0.375 |

新GPU step3更新等込み範囲1.683–1.735秒、step4 1.452–1.488秒。対応するCPU範囲は0.498–0.506秒、0.370–0.380秒。旧→新のhot時点中央値比は2.38/2.55倍だが、CPUに対する倍率ではない。

### 4.4 同一GPU状態からの再始動実験

**(出典: GPU_STATIC_UPDATE_RESULTS_20260906.md)**

step2を一度GPUで解き、同じx/y/z/sを全条件で確認し、step3に同じxを渡した。32環境単回対照実験。

| restart μ | step3 factor回数 | step4 factor回数 | 2時点の解法API時間合計 | 全条件の元LP判定 |
|---|---:|---:|---:|---|
| 1e-3 | 20 | 24 | 4.664 | 合格 |
| 1e-4 | 12 | 10 | 1.554 | 合格 |
| 1e-5 | 11 | 9 | 1.375 | 合格 |
| 1e-6 | 12 | 10 | 1.502 | 合格 |

1e-5をsequence比較の実験設定に採用。PPO本番の既定へは昇格していない。

### 4.5 非採用・不合格の記録

**(出典: GPU_NUMERIC_REUSE_RESULTS_20260906.md)**

- GMRES workspace単独有効: 32環境step2で4.9288秒、32/32認証、setup 11.9566秒。
- fresh構築＋plain内部warm: 4環境step2/3/4で2.3105/0.5484/2.1603秒、全合格。
- 数値更新版のstep4: 2/4認証（`pf_ipm_gpu_rebind_interior4_234_20260906.json`）。
- slack修復だけのfresh 4環境: 2.2544/1.2475/1.3902秒、全合格。
- 内部slack/dualのfloor 1e-4: 32環境step3 solve 29.2571秒、step4 solve 61.2613秒で不合格。
- matched-state対照（`probe_ipm_rebind_matched_state.py`）: step4でrebind 2/4認証、fresh 0/4認証。

### 4.6 再現用ファイル

**(出典: GPU_NUMERIC_REUSE_RESULTS_20260906.md、GPU_STATIC_UPDATE_RESULTS_20260906.md)**

- `scripts/benchmark_ipm_sequence.py`
- `scripts/probe_ipm_restart_mu_matched.py`
- `scripts/profile_ipm_numeric_update.py`
- `scripts/summarize_ipm_static_compare.py`
- `results/pf_ipm_static_compare_summary_20260906.json`
- `results/pf_ipm_restart_mu_matched4_20260906.json`、`pf_ipm_restart_mu_matched32_20260906.json`
- `results/pf_ipm_numeric_update_profile32_20260906.json`、`pf_ipm_numeric_update_profile32_static_20260906.json`
- `results/pf_ipm_gpu_bound_restart_mu1e4_32_234_20260906.json` 他repeat版

最終回帰（56ファイル）: 1,314 passed / 4 skipped / 11 warnings、26.93秒。
`results/pf_ipm_static_reuse_regression_20260906.xml`

---

## 5. GPU数値更新のデバイス実装 (2026-09-07)

**(出典: GPU_DEVICE_UPDATE_RESULTS_20260907.md)**

`DeviceNumericUpdatePlan`によりGPU側の反復係数更新を約1秒から約0.05～0.07秒へ短縮。ただし強いCPU 16並列を更新・求解・検証込みで上回ってはいない。

### 5.1 実装

- `src/gpu_ipm_device_update.py`: 条件検査・数値縮約・transactional commit
- `src/gpu_ipm_device_payload.py`: 全数値operatorへの更新先マッピング
- `src/gpu_segmented_linear.py`, `src/gpu_forest_numeric_bounds.py`: 決定的GPU演算
- `scripts/benchmark_ipm_sequence.py --device-numeric-updates`: opt-in検証経路

### 5.2 同一入力・同一精度のCPU比較

RTX 4060 Laptop、32環境、保存済みstep2～8。CPU HiGHS 1.15.1、16 workers、basis再利用。

| 保存時刻 | GPU求解のみ (s) | GPU係数更新 (s) | GPU更新・求解・検証等込み (s) | CPU16更新・求解・検証込み (s) |
|---|---:|---:|---:|---:|
| 2（初回） | 4.758 | — | 20.631 | 0.689 |
| 3 | 0.621 | 0.071 | 0.778 | 0.469 |
| 4 | 0.492 | 0.052 | 0.608 | 0.360 |
| 5 | 0.489 | 0.052 | 0.604 | 0.298 |
| 6 | 0.543 | 0.051 | 0.655 | 0.365 |
| 7 | 0.443 | 0.052 | 0.556 | 0.321 |
| 8 | 0.487 | 0.051 | 0.602 | 0.330 |

全sequence lifecycle: GPU 25.074秒、CPU 2.906秒。1試行で信頼区間は算出していない。

前版（静的再利用、n=3中央値）のstep3/4はGPU 1.722/1.485秒、係数更新0.997/0.881秒だった。今回の短縮は時系列の開発比較。

### 5.3 ボトルネックと追加試験

step3の求解0.621秒の内訳: factor 0.249秒、三角求解と状態更新0.279秒、certificate 0.054秒。11回のfactor、22回の三角求解。係数更新だけをゼロにしてもCPU超えには届かない。

単一目的変数を最適box境界に固定した補助可行性問題（box-face）: 32環境96件元LP認証通過。初回求解1.207秒、後続0.635/0.510秒、更新等込み0.803/0.639秒。通常経路より継続時は速くない。初回構築込み17.387秒。既定不採用。

### 5.4 テストと成果物

最終回帰: 1,526 passed、4 skipped、11 warnings、31.90秒。

Rawと集計:
- `results/pf_ipm_gpu_device_updates32_2to8_20260907.json`
- `results/pf_ipm_cpu1151_device_compare32_2to8_20260907.json`
- `results/pf_ipm_device_compare_summary_20260907.json`
- `results/pf_ipm_gpu_device_updates32_234_20260907_r1.json`
- `results/pf_ipm_box_face4_234_20260907_r1.json`
- `results/pf_ipm_box_face32_234_20260907_r1.json`
- `results/pf_ipm_device_update_regression_20260907.xml`
- `results/pf_ipm_device_update_final_regression_20260907.xml`
- `results/pf_ipm_device_update_edge_tests_20260907.xml`

---

## 6. 行列分解・三角求解の最適化 (2026-09-07)

**(出典: GPU_LINEAR_ALGEBRA_OPTIMIZATION_20260907.md)**

前回のGPU数値更新を維持し、残る反復線形代数を対象とした。

### 6.1 実装した3案

#### 6.1.1 前処理factorの低頻度更新（既定不採用）

`factor_reuse_interval`追加。4環境では精度維持したが求解時間悪化（対照0.282/0.234秒に対し1.036/0.857秒）。32環境試験へ拡大せず。

#### 6.1.2 有限値検査・入出力変換の融合（実験オプション）

`src/gpu_solve_guards.py`に2個のCUDAカーネル。32環境初回試験はhot求解0.642/0.485秒、更新等込み0.808/0.599秒で明確な優位なし。

#### 6.1.3 cuDSSのnative実行レイアウト変更（block diagonal、単独比較）

`factor_layout='block_diagonal'`。論理的には32個の独立LPのまま、cuDSSには非対角ブロック0の1個の大きな疎行列を渡す。

### 6.2 3反復の比較結果

全方式で672/672問題元LP認証通過。

| 保存時刻 | GPU uniform 更新・求解・検証等 (s) | GPU block diagonal 同範囲 (s) | CPU16 更新・求解・検証 (s) | GPU内比較の短縮率 |
|---|---:|---:|---:|---:|
| 2（初回構築込み） | 21.322 | 21.570 | 0.752 | −1.2% |
| 3 | 0.817 | 0.725 | 0.512 | 11.2% |
| 4 | 0.618 | 0.558 | 0.382 | 9.8% |
| 5 | 0.613 | 0.556 | 0.315 | 9.3% |
| 6 | 0.678 | 0.614 | 0.377 | 9.4% |
| 7 | 0.563 | 0.515 | 0.328 | 8.5% |
| 8 | 0.620 | 0.558 | 0.359 | 10.0% |

後続6時刻合計（中央値）: uniform 3.928秒、block diagonal 3.523秒（10.3%短縮）、CPU16 2.249秒。GPUはCPU16の約1.57倍。

**採否:** block diagonalを改善候補として残すが、既定値は変更しない。PPO/全stageへの適用で速度優位を主張しない。

### 6.3 再現コマンド

```sh
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-cuopt-26.8/bin/python scripts/benchmark_ipm_sequence.py \
  --mode gpu --batch 32 --steps 2 3 4 5 6 7 8 \
  --reuse-numeric-workspace --device-numeric-updates --restart-mu 0.00001 \
  --factor-layout block_diagonal --output results/new_block_comparison.json
```

全回帰: 1,555 passed / 4 skipped / 11 warnings、36.46秒。
`results/pf_ipm_linear_algebra_regression_20260907.xml`

- Raw: `results/pf_ipm_factor_lag4_control_20260907.json`、`pf_ipm_factor_lag4_interval2_20260907.json`、`pf_ipm_fused_guards32_234_20260907_r1.json`、`pf_ipm_block_factor32_234_20260907_r1.json`
- 集計: `results/pf_ipm_layout_compare_summary_20260907.json`

---

## 7. 約50%短縮を目標とした構造最適化の結果 (2026-09-07)

**(出典: GPU_HALF_RUNTIME_OPTIMIZATION_20260907.md)**

### 7.1 結論

初回の構築・求解・検証は約半分に短縮できたが、準備後の繰返し計算の半減は未達。CPU 16並列に対する速度優位も未達。

| 計時区間 | 従来GPU block diagonal | 証明再利用GPU | CPU16並列 | GPU変更の効果 |
|---|---:|---:|---:|---|
| 初回準備のみ | 17.095 s | 7.014 s | — | 59.0%短縮 |
| 初回準備＋最初の求解・検証 | 21.541 s | 10.889 s | 0.763 s | 49.4%短縮 |
| 初回準備込み7時刻のlifecycle | 25.563 s | 15.018 s | 3.041 s | 41.3%短縮 |
| 準備後の6時刻合計 | 3.413 s | 3.470 s | 2.229 s | 改善なし（中央値1.7%増） |

### 7.2 実装した構造変更

`--reuse-equality-proofs`で有効化。等式forestの構築証明を同一batch内で再利用、zero-face証明にrebind検査、canonical forestの重複構築を省略。

主要実装: `src/lp_equality_reduction.py`, `src/gpu_forest_ipm.py`, `src/gpu_zero_face_ipm.py`, `src/gpu_forest_map.py`, `src/gpu_ipm_device_update.py`。

### 7.3 試したが採用しなかった数値解法

| 案 | 実測結果・採否 |
|---|---|
| 50% reserve補助LPから前時刻解を再利用 | 4環境×7時刻で再利用0回。毎回の再求解が必要で遅い |
| 等式アフィン空間内のGPU Motzkin投影 | cold 2,000反復で違反約0.0025、warm 5,000反復で約0.045が残り不合格 |
| AMDによる疎行列の並べ替え | 32環境hot 2時刻で1.896/1.560秒。悪化のため既定にしない |
| cuDSS 0.7の別factor algorithm | 0.682/0.525秒。小幅な変化で半減ではない |
| FP32 factor/三角求解＋FP64残差補正 | toyは通過。実GEMの初回Newton検査は正則化1e-6/1e-4とも不合格 |

### 7.4 残る律速と次の判定基準

準備後はFP64疎行列factorと三角求解が残る。採用基準: 同じ32環境・同上精度で、更新・求解・検証込みhot 6時刻を旧目標1.762秒以下、かつCPU16より短くすること。現時点3.470秒では未達。

最終回帰: 1,594 passed / 4 skipped / 11 warnings、34.33秒。
`results/pf_ipm_half_target_final_regression_20260907.xml`

Raw:
- `results/pf_ipm_proof_compare_summary_20260907.json`
- `results/pf_ipm_proof_{baseline,proof_reuse,cpu16}_32_2to8_20260907_r{1,2,3}.json`

---

## 8. 学習中の反復計算を半減する試験 (2026-09-07)

**(出典: GPU_HOT_HALF_EXPERIMENTS_20260907.md)**

### 8.1 結論

半減は未達。初回準備の短縮をこの達成に数えない。

前回基準（GPU反復区間3.470秒、CPU16 2.229秒）に対する目標: 同じ更新・求解・検証込み1.735秒以下。

今回終了時の対照: GPU 3.566秒（単回）。目的上限固定補助問題のGPU方式: 3.679秒（単回）。両者とも各224件元LP認証通過。

### 8.2 律速の直接計測

32環境step3: 求解0.529秒に対し、cuDSS数値分解11回の完了待ちwall時間0.204秒、三角求解22回0.048秒。行列分解だけで約39%を占める。

### 8.3 実装した候補と判断

| 候補 | 観測 | 判断 |
|---|---|---|
| 双対Schur補行列 | 分解次元5,808→2,015。非ゼロ数44,248→53,029。速度改善なし | opt-in、既定不採用 |
| 多重中心性補正 | 4環境で分解11→8回、8→6回。追加求解込み0.496/0.346秒 | 反復数だけでは半減にならない |
| primal/dual独立更新幅 | 共通幅とのmerit比較で保護。認証通過するが半減なし | opt-in |
| 環境間の分解共有＋複数右辺 | 4環境を2環境ずつ共有。次時刻で補正増、元LP認証失敗 | 既定不採用 |
| 固定分解DR＋Anderson | 500反復でも違反が残る | 未認証、不採用 |
| 目的上限固定の実行可能性IPM | 32環境で初回分解42→18回、全224件認証。反復区間3.679秒 | 初回改善と反復改善を区別 |
| 違反二乗の減衰Newton | 30反復で元LP未認証 | 未認証、不採用 |
| 前回フラックスのray縮小／成分別再スケール | 元LP検証不合格。GPU IPMへ戻して全時刻進めても再分解が必要 | 辞書なし再利用案も不十分 |

### 8.4 精度と適用範囲

- 元LP primal違反 ≤1e-5、dual違反 ≤1e-7、相対KKT gap ≤1e-7、独立直接双対gap ≤1e-7。
- GPU経路でCPU LP optimizer呼出し0。現在・未来のCPU参照解はGPUへ与えていない。
- 初回構造解析・入力準備・独立監査にはCPUを使う。「完全GPU内完結」とは呼ばない。

### 8.5 次に必要な改訂

共有factorの拡大や反復数上限の機械的半減は取らない。現在の律速制約行を反応・交換slotへ対応付け、等式を保つ反応組合せ変更を低次元GPU補正空間へ追加する。

最終回帰: 1,627 passed / 4 skipped / 11 warnings、38.58秒。

Raw:
- `results/pf_ipm_hot_half_control32_20260907.json`
- `results/pf_ipm_native_profile32_20260907.json`
- `results/pf_ipm_dual_schur32_20260907.json`
- `results/pf_ipm_mcc2_4_20260907.json`
- `results/pf_ipm_pd_best4_20260907.json`
- `results/pf_ipm_shared2_4_20260907.json`
- `results/pf_gpu_dr4_20260907.json`
- `results/pf_ipm_bound_feasibility_final32_20260907.json`
- `results/pf_feasibility_newton4_20260907.json`
- `results/pf_ipm_scaled_ray_diagnostic4_20260907.json`
- `results/pf_ipm_scaled_ray_fallback4_20260907.json`
- `results/pf_ipm_hot_half_regression_final_20260907.xml`

---

## 構成元ファイル

| ファイル名 | 元の日付 |
|---|---|
| GPU_BATCH_SPEED_RESULTS_20260906.md | 2026-09-06 |
| GPU_NUMERIC_REUSE_RESULTS_20260906.md | 2026-09-06 |
| GPU_STATIC_UPDATE_RESULTS_20260906.md | 2026-09-06 |
| GPU_LINEAR_ALGEBRA_OPTIMIZATION_20260907.md | 2026-09-07 |
| GPU_HALF_RUNTIME_OPTIMIZATION_20260907.md | 2026-09-07 |
| GPU_HOT_HALF_EXPERIMENTS_20260907.md | 2026-09-07 |
| GPU_DEVICE_UPDATE_RESULTS_20260907.md | 2026-09-07 |
