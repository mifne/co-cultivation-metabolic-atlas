# GPU加速dFBA/LPソルバー開発：辞書ベースGPUサロゲート統合ドキュメント

**要約**：本ドキュメントは、3菌種（OR16、NS21、P. freudenreichii）共有培地dFBA向けGPU加速LPソルバーの開発記録を統合したものである。厳密HiGHS解から構築した32,768候補の辞書とニューラル順位付け層により、CPU HiGHS比2.6–2.8倍の高速化を達成したが、24時間PHA相対誤差1%以下という科学的認定基準は複数回の改良を経ても満たしていない。最終的には、未認定artifactはGPUへロードせず自動的にCPU HiGHSへ差し戻す資格ゲートを実装し、RL探索にはGPU近似を、最終評価には厳密CPU HiGHSを使う二段階運用を採用した。

---

## 1. 背景と目的

本プロジェクトでは、3菌種（OR16、NS21、P. freudenreichii）のゲノム規模代謝モデル（GEM）を用いた動的フラックスバランス解析（dFBA）をGPUで高速化する。各タイムステップでLPを解く必要があるdFBAにおいて、RL rolloutの反復計算をGPU上で近似するサロゲート方式を開発した。

**(出典: GPU_SURROGATE_ARCHITECTURE.md)**

当初、各GEMの厳密LPを個別にcuOptへ送る方式を試みたが、RTX 4060上でCPU HiGHSより低速であった。そのため、厳密FBA解の有限辞書をGPU上で一括評価する方式を採用した。

**(出典: GPU_SURROGATE_ARCHITECTURE.md, GPU_BATCH_QP_MIGRATION_REPORT.md)**

---

## 2. システムアーキテクチャ

### 2.1 オンライン計算パス

**(出典: GPU_BATCH_QP_MIGRATION_REPORT.md)**

1. 3-GEMコンテキスト（バイオマス、共有培地供給、反応上下限、目的関数）を構築
2. 33,728件のオフライン厳密協調FBAアンカーをCUDAニューラル決定ヘッドで順位付け
3. 最良128アンカーをCUDA上に保持
4. バッチ化凸包QP投影を適用（全アンカーはブロック対角GEM等式制約を満たすため、凸結合は細胞内定常質量収支を保持）
5. 128アンカーでは不十分な場合、同一行のみ2,048アンカー、6方向菌種別ブロック合成、最大1,200回の主双対反復で再試行
6. フラックスベクトルをdFBA状態更新へ返す（オンライン経路ではSciPy/HiGHSは呼ばない）

### 2.2 アーキテクチャ図

```
SubprocVecEnv workers
  └─ LP parameters [lb, ub, objective, sense]
       └─ single GPU owner / micro-batcher
            ├─ 辞書フラックスを全lb/ubで並列フィルタ
            ├─ feasible候補のみobjectiveを並列評価
            ├─ best feasible fluxを返す
            └─ candidateなし / OOD / residual異常 → CPU HiGHS
```

**(出典: GPU_SURROGATE_ARCHITECTURE.md)**

### 2.3 PPO統合

PPOワーカーは約1 GiBの辞書を個別にロードしない。生成されたマネージャプロセスがCUDAコンテキストを所有し、ワーカー要求をマイクロバッチに結合する。独立GPUは独立したseed/agent/envシャードに適しており、VRAMは共有メモリとして扱わない。

**(出典: GPU_BATCH_QP_MIGRATION_REPORT.md)**

---

## 3. 辞書構築とデータセット

### 3.1 辞書サイズ選定

32,768（2^15）は統計的サンプルサイズとして導出されたものではなく、工学的生産階層として選択された。階層は1,024スモーク状態、8,192パイロット状態、その4倍の生産候補セット。選定理由：
- power-of-two GPUバッチ処理との整合性
- 各フィード/酸素/バイオマス/栄養状態の層別反復
- 8 GB RTX 4060 Laptop VRAMに収まる完全3-GEMフラックス辞書

**(出典: COOPERATIVE_SURROGATE_32768_REPORT.md)**

### 3.2 データセット階層

| 階層 | 整列状態数 | 生フラックス辞書容量 | 目的 |
|---:|---:|---:|---|
| Smoke | 1,024 | 26 MB | 統合・ガードテスト |
| Pilot | 8,192 | 206 MB | RTX 4060上のカバレッジ/フォールバック測定 |
| Production | 32,768 | 823 MB | パイロット合格後のRL rollout高速化 |

**(出典: COOPERATIVE_SURROGATE_SCALING.md)**

注記：上記の反応数・辞書サイズ分析はWCFS1ベースの3菌種構成（OR16, NS21, *L. plantarum*）で測定された。P. freudenreichii（Pf）への変更後（PROJECT_MANAGEMENT.md Phase D参照）もサイジング手法は有効だが、具体的数値はPfモデルで再測定されていない。

**(出典: COOPERATIVE_SURROGATE_SCALING.md, COOPERATIVE_SURROGATE_32768_REPORT.md)**

### 3.3 辞書行の構成

1行は1つの完全なコミュニティ状態を表現し、同一の厳密協調ソルブからの3つの整列フラックスベクトルをすべて含む。コンテキストには以下が最低限含まれる：
- 全反応の上下限とアクティブ目的係数
- 3菌種のバイオマス
- 共有細胞外インベントリと積分区間
- 共通・菌種別フィード制御、酸素移動制御、フェーズ
- 共通成長最適値とPHA/ラバー目的切替

GPU選択はコミュニティ全体の候補をランク付けし、菌種ごとに独立して選択しない。

**(出典: COOPERATIVE_SURROGATE_SCALING.md)**

### 3.4 サンプリング設計

層別サンプリングが必要であり、ランダムポリシーのみから抽出しない：
- 定義済み10共通フィードスケール0–0.02、0.003–0.012を密にサンプリング
- 酸素移動アクション：全範囲0–200
- 初期・生産・枯渇フェーズ
- 初期バイオマス比：少なくとも±20%摂動
- グルコース、イソロイシン、ピリドキサミン、アンモニウム制限境界
- 協調成長とPHA/ラバー生産目的両モード

訓練ポリシーロールアウトは能動学習で追加可能だが、厳密監査状態とheld-outテストセットは辞書候補として再利用しない。

**(出典: COOPERATIVE_SURROGATE_SCALING.md)**

### 3.5 32,768候補辞書の詳細（WCFS1構成）

- 定式化：厳密協調共有培地HiGHS、節約的フラックス交換
- 菌種：OR16, NS21, WCFS1（全行で整列）
- 反応数：合計6,585
- コンテキスト：状態あたり20,200値
- 厳密状態：32,768（全有限、全生フラックス行が一意）
- 可変コンテキスト列：97
- 可変反応フラックス：2,828
- 四捨五入共通成長レベル：6,120
- 厳密ラベル収集：4,333.6秒（72.2分）、8 CPUワーカー
- ホストデータセット：3,511,788,071 bytes
- GPU artifact：1,001.2 MiB

**(出典: COOPERATIVE_SURROGATE_32768_REPORT.md)**

### 3.6 P. freudenreichii再構築辞書（Pf構成）

| 項目 | 値 |
|---|---:|
| 菌構成 | OR16 / NS21 / P. freudenreichii |
| 反応次元 | 6,733 = 2,584 + 2,798 + 1,351 |
| 状態入力次元 | 20,599 |
| 候補数 | 32,768 |
| 検索・学習特徴数 | 74 |
| PHB非ゼロ教師 | 1,642 (5.01%) |
| PHV非ゼロ教師 | 11,214 (34.22%) |
| ニューラル決定対象 | 155反応、重要対象6、PHA目的2 |
| 訓練 / 内部検証 | 26,214 / 6,554 |
| CPU教師生成 | 約76分、16ワーカー |
| GPU学習 | RTX 4060 Laptop、100 epochs、約9.1秒 |
| 辞書容量 | 1,029.19 MiB |

**(出典: PFREUDENREICHII_GPU_REBUILD_20260903.md)**

---

## 4. 性能測定結果

### 4.1 スモークスケーリング（RTX 4060 Laptop, 2026-09-01測定）

等アクション120ステップ（24 h, dt=0.2 h）rollout：

| 候補数 | GPU有効ステップ | End-to-end高速化 | 24h PHA相対誤差 |
|---:|---:|---:|---:|
| 128 | 33/120 (27.5%) | 1.17× | 0.06% |
| 1,024 | 120/120 (100%) | 3.03× | 2.30% |

128候補での低誤差は72.5%のステップでCPUフォールバックした結果であり、小辞書の精度によるものではない。1,024候補でフォールバックがなくなったことで残る近似誤差が顕在化し、事前宣言された1%ゲートを超過した。

**(出典: COOPERATIVE_SURROGATE_SCALING.md)**

### 4.2 アドミッション試験（旧3菌構成）

128環境状態をHiGHSで生成、102解を辞書、26解をholdoutに使用した小規模試験（実装経路の受入試験、本番精度保証用ではない）：

| GEM | CPU HiGHS (LP/s) | GPU安全採用率 | objective MAE | 最大相対Sv残差 |
|---|---:|---:|---:|---:|
| OR16 | 6.85 | 91.4% | 0.00433 | 4.9e-8 |
| NS21 | 6.74 | 96.9% | 0.00388 | 5.5e-8 |
| *L. plantarum* | 26.28 | 98.4% | 0.000166 | 9.8e-8 |

24ステップwarm-start単一環境rollout：HiGHS 12.49秒に対しGPU 2.23秒（5.61倍）、72 FBA中65件をGPU採用、7件をHiGHSへフォールバック。

**(出典: GPU_SURROGATE_ARCHITECTURE.md)**

### 4.3 バッチQP移行後の性能（RTX 4060 Laptop）

**(出典: GPU_BATCH_QP_MIGRATION_REPORT.md)**

| テスト | 結果 |
|---|---:|
| 5 seed, 24h, 120-step CPU HiGHS比較 | 平均高速化2.67× (95% CI 2.61–2.74) |
| GPU feasible in comparison | 600/600 steps |
| オンラインCPU LP呼び出し | 0 |
| 平均/最大終端PHA相対誤差 | 1.24% / 1.71% |
| 最大終端バイオマス絶対誤差 | 0.00116 g/L |

### 4.4 DAgger改良後の性能（旧3菌構成）

**(出典: PHA_GPU_ROLLOUT_ACCURACY_REPORT.md)**

| 指標 | Before | DAgger-2 |
|---|---:|---:|
| PHA相対誤差（平均） | 1.237% | 0.627% |
| PHA相対誤差（最大） | 1.711% | 0.861% |
| 平均高速化 | 2.67× | 2.81× |
| 最大バイオマス絶対誤差 | 0.001161 g/L | 0.000541 g/L |
| GPU feasible requests | 600/600 | 600/600 |
| オンラインCPU LP呼び出し | 0 | 0 |
| Peak allocated VRAM | 900 MiB | 918 MiB |

2026-09-11注記：この節のベンチマークはL. plantarum (WCFS1)を第3菌種としていた当時の3 GEM構成で測定されたものであり、P. freudenreichii（Pf）構成では再測定されていない。アーキテクチャ手法自体は菌種に依存しないが、具体的数値はPfモデルで再測定されていない。

**(出典: GPU_SURROGATE_ARCHITECTURE.md)**

### 4.5 ニューラル–機構ハイブリッドGPU性能（旧3菌構成）

**(出典: NEURAL_MECHANISTIC_GPU_DFBA_REPORT.md)**

| 指標 | 結果 |
|---|---:|
| 条件 | 120 step × 5 seed, dt = 0.2 h |
| CPU HiGHS比速度 | 2.62× |
| 速度向上95% CI | 2.47–2.77× |
| GPU候補採用率（平均） | 96.3% |
| GPU候補採用率（最小） | 95.8% |
| PHA相対誤差（平均） | 0.988% |
| PHA相対誤差（最大） | 2.61% |
| 最大バイオマス絶対誤差 | 0.00154 g/L |
| RTX 4060上のPyTorch peak allocation | 約859 MiB |
| 科学的認定 | 未認定 |

### 4.6 GPU利用率改善（RTX 4060）

**(出典: GPU_UTILIZATION_OPTIMIZATION_RTX4060.md)**

#### 512候補版

| 並列環境数 | CPU HiGHS (transitions/s) | CUDA (transitions/s) | CPU比 | GPU平均使用率 | VRAM最大 | GPU解受理率 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3.79 | 18.64 | 4.92× | 7.4% | 161 MiB | 96.2% |
| 4 | 6.08 | 29.35 | 4.83× | 8.4% | 159 MiB | 96.2% |
| 8 | 8.86 | 40.96 | 4.62× | 11.4% | 181 MiB | 96.5% |
| 16 | 11.97 | 46.56 | 3.89× | 13.0% | 287 MiB | 97.1% |

#### 2048候補版（純粋FBA候補評価飽和試験）

| GEMごとのバッチ | GPU平均使用率 | GPU最大 | VRAM | 処理量（3-GEM予測/s） | 判定 |
|---:|---:|---:|---:|---:|---|
| 8 | 56.9% | 63% | 621 MiB | 3,179 | 小規模 |
| 16 | 69.0% | 73% | 1,131 MiB | 4,156 | 現行CPU上の実用域 |
| 32 | 76.7% | 83% | 2,187 MiB | 4,961 | スケール継続 |
| 64 | 84.3% | 89% | 4,331 MiB | **5,581** | **処理量最適** |
| 128 | **98.3%** | **100%** | **7,935 MiB** | 1,299 | VRAM圧迫で低速化 |

バッチ128ではVRAM圧迫により処理量が低下。現行GPUの推奨点はバッチ64。

16環境・960遷移の長期end-to-end比較：CPU HiGHS 12.24 transitions/sに対しCUDA版 **85.83 transitions/s（7.01倍）**。GPU解受理率99.42%、CPUフォールバック17/2,928 solve attempts。エンドツーエンドのGPU平均使用率は6.8%（GPU計算が短いバーストで終了し、残り時間をCPU処理が占めるため）。

---

## 5. 実装した改良と問題

### 5.1 改善一覧

1. **教師データの整合性修正**：追加学習・DAgger処理に菌種構成と初期NH4を渡し、CPU再ラベル時にPHB/PHVを含む動的目的関数を復元。入力20,599次元の一致を検査。
   
2. **辞書拡充**：32,768件に独立長軌跡960件、GPU到達状態の正解360件、初期低窒素状態境界サンプル256件を追加し、最大34,344件まで拡張。

3. **DAggerによる分布シフト対策**：GPU rolloutコンテキストを取得可能にし、`augment_cooperative_surrogate_dagger.py`で独立GPU-only軌跡を実行、各訪問状態を同一境界・バイオマス・共有培地供給でのCPU HiGHS厳密解で再ラベル。2回のDAgger反復で720件の焦点アンカーを追加（33,728→34,448候補）。

4. **予測器再学習**：長軌跡960件追加で500 epoch学習。サンプル単位保留データのPHAフラックスRMSEが0.630から0.347に低下。

5. **GPU QP投影の拡張**：単独実行可能候補だけでなく、単独実行不能候補を混合するADMM補正を追加。PHA予測値±0.25%の補助制約を設定。

6. **計算削減**：全候補が厳密に満たす制約を反復演算から除外。GPU倍精度Cholesky/Woodbury更新。再順位付けでは全6,733反応ではなく155決定フラックスのみを事前キャッシュ。

7. **検証修正**：QP残差は混合後の実際の返却解で計算。CPU LPは実行箇所で直接計数。実行コードのSHA-256を保存。

**(出典: PFREUDENREICHII_GPU_QUALIFICATION_20260903.md, PHA_GPU_ROLLOUT_ACCURACY_REPORT.md)**

### 5.2 失敗したアプローチと原因

| 方式 | GPU採用率 | PHA相対誤差 | 高速化 | 判定 |
|---:|---:|---:|---:|---|
| 32,768最近傍（再測定） | 97.50% | 2.807% | 2.746× | 不合格 |
| decision-aware強重み | 60.00% | 4.244% | 1.616× | 不合格 |
| 低NH4・高距離512件追加（33,280件） | 99.17% | 6.751% | 2.748× | 不合格 |
| 実行可能2近傍凸補間 | 98.33% | 2.911% | 2.806× | 不合格 |
| 実行可能4近傍凸補間 | 98.33% | 3.515% | 2.714× | 不合格 |
| 実行可能8近傍凸補間 | 98.33% | 4.113% | 2.758× | 不合格 |
| 4-stepごと厳密HiGHS | 100.00% | 3.604% | 1.943× | 不合格 |
| 8-stepごと厳密HiGHS | 100.00% | 2.408% | 2.327× | 不合格 |
| 16-stepごと厳密HiGHS | 100.00% | 2.130% | 2.628× | 不合格 |
| 32-stepごと厳密HiGHS | 99.15% | 2.141% | 2.781× | 不合格 |

候補追加でGPU採用率が上がってもPHA誤差が悪化している。FBAには複数の最適・準最適フラックスが存在し、近傍追加で別の縮退枝が最近傍になると、微小な1-step差が120-stepで累積する。

**(出典: GPU_DICTIONARY_ACTIVE_LEARNING_AND_SAFETY_REPORT.md)**

**その他の失敗事例**：
- 厳密解集合のPCA基底（rank 128）で全6,585フラックスを直接生成するAMN型：基底表現誤差は重要フラックスRMSE 0.0227と小さいが、状態から潜在座標を回帰した予測は上下限制約合格率0%。原因は内部フラックスの非一意性と活性制約切替による不連続性。
- 再順位付けプール512（投影候補128のみ）：1軌跡で8ステップの実行不能を発生。
- 均一8アンカー凸ブレンド：線形実現可能性は保持するが低成長代替最適解を混合、一部seedで大きなrollout誤差。
- 2,048候補プール：ニューラル予測誤差がコンテキスト遠方アンカーを選択し、最悪seed誤差が増加。
- 直接ニューラルPHA目標補間：終端PHAを強く過小評価。

**(出典: PHA_GPU_ROLLOUT_ACCURACY_REPORT.md, NEURAL_MECHANISTIC_GPU_DFBA_REPORT.md)**

### 5.3 使用非推奨の設定

オンラインプールを128→256に増やす試みは、1 seedで改善したが別seedで大きな軌跡乖離を生んだ。オフライン辞書を33,728→34,688に拡大（候補選択の再調整なし）も却下（バッチQP移行報告）。

**(出典: GPU_BATCH_QP_MIGRATION_REPORT.md)**

---

## 6. 科学的認定基準と資格ゲート

### 6.1 認定基準（変更なし）

**(出典: PFREUDENREICHII_GPU_QUALIFICATION_PLAN.md, PFREUDENREICHII_GPU_QUALIFICATION_20260903.md)**

- 独立軌跡あたり120実ステップ、初期NH4 0.05 mM
- 最大終端PHB+PHV質量誤差 ≤ 1%（平均値ではない）
- 最大菌種バイオマス絶対誤差 ≤ 0.01 g/L
- 最大終端3HVモル分率差 ≤ 0.01
- 全オンラインFBA要求をGPU受理、オンラインCPU LPソルブ0回
- CPUとGPUを同一アクション・モデルハッシュ・目的で比較
- 「GPU only」はオンラインFBAソルバーのみを指す

### 6.2 パイロット受入ゲート

**(出典: COOPERATIVE_SURROGATE_SCALING.md)**

パイロットはすべての以下を満たす場合のみ合格：
- 全held-out状態の≥95%がGPUガード通過（本番は99%）
- 受理状態は反応・質量収支・共有培地制約に違反しない
- 共通成長と活性生産目的の相対誤差≤2%
- 24hバイオマス、PHA、ラバー結果の誤差≤1%（厳密積分に対して）
- 厳密ソルバーフォールバック≤5%（パイロット）≤1%（本番）

### 6.3 資格ゲート実装

**(出典: GPU_DICTIONARY_ACTIVE_LEARNING_AND_SAFETY_REPORT.md)**

- artifactごとに小さなvalidation manifest（`xxx.validation.json`）を配置
- `status = qualified`が明示されたartifactのみ科学評価モードでGPUへロード
- manifest欠落・破損・`experimental_not_qualified`はGPU試行前にCPU HiGHSへ差し戻し
- 診断に`surrogate_disabled_reason`を記録
- 近似GPU辞書使用時はコード上で明示的に資格ゲートを解除
- RLの評価軌跡・上位候補・図表・論文値はCPU HiGHSで再計算

統合確認では、未認定32,768辞書を指定しても`backend = scipy`、GPU試行0回、`surrogate_disabled_reason = validation_status:experimental_not_qualified`。

---

## 7. 現状の認定状況（2026-09-11時点）

### 7.1 全体的な状況

**どの構成も24時間PHA相対誤差1%以下を満たしていない。すべてのartifactは`experimental_not_qualified`として保存されている。**

### 7.2 Pf構成での最終診断結果

#### 短期試験A（24 step, seed 20260901）
| 項目 | 値 |
|---|---:|
| PHA終点相対誤差 | 0.6904% |
| GPU採択回数 | 24/24 |
| オンラインCPU LP | 0 |
| 最大菌体量絶対誤差 | 0.03530 g/L（基準外） |
| 3HVモル分率絶対誤差 | 0.01465（基準外） |
| GPU実行時間 | 6.8840秒 |
| 認証 | 不合格 |

**(出典: PFREUDENREICHII_GPU_QUALIFICATION_20260903.md)**

#### 長期試験B（120 step, seed 20260913）
| 項目 | 値 |
|---|---:|
| PHA終点相対誤差 | 13.0530% |
| GPU採択回数 | 120/120 |
| オンラインCPU LP | 0 |
| 最大菌体量絶対誤差 | 0.11613 g/L |
| 3HVモル分率絶対誤差 | 0.01990 |
| 認証 | 不合格 |

**(出典: PFREUDENREICHII_GPU_QUALIFICATION_20260903.md)**

### 7.3 Pf構成・3 seed 24h外部検証（初期NH4 0.05 mM）

| seed | CPU時間(s) | GPU経路時間(s) | PHA相対誤差 | 3HV組成差 | GPU採用率 | CPU差戻 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260911 | 128.46 | 75.54 | 2.75% | 2.73ポイント | 59.17% | 49/120 |
| 20260912 | 129.65 | 68.59 | 2.68% | 4.26ポイント | 65.00% | 42/120 |
| 20260913 | 130.28 | 31.77 | 11.88% | 11.65ポイント | 92.50% | 9/120 |

GPU最大割当VRAM：951.07 MiB

**(出典: PFREUDENREICHII_GPU_REBUILD_20260903.md)**

### 7.4 旧3菌構成での最終改善（最大PHA誤差0.861%）

DAgger-2改良後、24h PHA誤差平均0.627%、最大0.861%、600/600 feasibleを達成した。しかしこれらは旧WCFS1構成の結果であり、Pf構成の精度証明には使用できない。

**(出典: PHA_GPU_ROLLOUT_ACCURACY_REPORT.md)**

---

## 8. 診断と残課題

### 8.1 残存する問題の原因分析

1. **実行可能候補不足（長軌道）**：乳酸、グルコース、プトレシン等の共有供給制約に対し局所候補の凸包が不足。失敗状態と低消費の実行可能アンカーを追加する必要がある。

2. **実行可能性と最適性の分離**：QPが質量収支・境界を満たしても、CPUの「共存増殖率→動的目的→交換フラックス最小化」という三段階最適化との最適性ギャップ保証がない。

3. **状態誤差による生産期切替のずれ**：NH4等の状態誤差が後続の生産期切替タイミングを変化させる。

4. **PHA単一変数補正の限界**：総PHAへの補正だけでは菌体量・交換フラックス誤差が残る。PHB/PHV・菌体量・主要交換フラックスを同時に扱う多出力投影が必要。

**(出典: PFREUDENREICHII_GPU_QUALIFICATION_20260903.md, PFREUDENREICHII_GPU_QUALIFICATION_PLAN.md)**

### 8.2 次の技術候補

厳密性を維持したGPU/ML高速化として、全フラックス辞書置換ではなく、厳密HiGHSへ渡すbasisまたは初期解の予測を試みる。現環境にはhighspy 1.15.1が導入済み。basis warm startの有効性を独立benchmarkで確認し、厳密最適性を保持したまま速度向上が得られた場合のみスクリーニング経路へ採用する。

参考：
- HiGHSはbasisの再利用と`setBasis`を公式Python例で提供
- Fan et al. (ICML 2023)は学習モデルでLPの初期basisを予測し、厳密simplex solverと組み合わせている
- Sambharya et al. (L4DC 2023)も、学習器の出力を反復最適化のwarm startとして使用

**(出典: GPU_DICTIONARY_ACTIVE_LEARNING_AND_SAFETY_REPORT.md)**

### 8.3 今後の実装優先順位

1. **CPU/GPU同一状態比較**：各ステップで増殖、PHB、PHV、NH4、酸素、炭素源・主要分泌物の誤差を分解。時系列のずれと単一状態の解誤差を区別。

2. **多出力投影**：総PHA補正からPHB/PHV・菌体量・主要交換フラックスを同時扱う投影へ移行。CPUの三段階目的を維持する制約または最適性検査を追加。

3. **有限辞書内不足状態の補償**：GPU上の全反応空間LP補正/能動集合補正。追加データは誤差の大きい状態へ限定。精度基準は緩めない。

4. **独立5 seed認定・公平速度測定**：120ステップ診断基準達成後、未使用seed（20286001–20286005）で認定。

**(出典: PFREUDENREICHII_GPU_QUALIFICATION_20260903.md)**

---

## 9. 運用制約と使用指針

### 9.1 科学的運用制約

**(出典: GPU_SURROGATE_ARCHITECTURE.md)**

1. サロゲートはRL探索専用。最終スコア、図、統計検定は `--solver-backend highs`
2. Holdout軌道は訓練辞書と別seed・別方策で作成、辞書内誤差のみを報告しない
3. 本番学習では定期HiGHS監査を有効化。目的値監査失敗時は厳密解へ置換
4. Acceptance、OOD、bound violation、audit failureを結果JSONで監視
5. 辞書増加時はVRAMと`batch × candidates × reactions`の一時tensorを測定

### 9.2 出力時の注記義務

**(出典: GPU_BATCH_QP_MIGRATION_REPORT.md)**
- GPUのみ経路をhigh-throughput PPO探索に使用
- 選択されたポリシーと最終論文結果は厳密CPU HiGHSで評価
- 現在のサロゲートはCPU LPと数値的に同一とは説明しない
- 高いPHA誤差を誘発する状態に対して独立能動学習軌跡を追加

### 9.3 三層構成提案

**(出典: NEURAL_MECHANISTIC_GPU_DFBA_REPORT.md)**
1. gsMOBOで共通流加液・種特異的栄養の候補を絞る（外側探索）
2. GPUニューラル–機構dFBAで多数環境を評価（内側高速化）
3. Pareto候補と最終方策を厳密HiGHSで再評価

---

## 10. 参考文献・先行研究対応

- **Faure et al. (2023)**, Artificial metabolic networks (AMN): 機構制約を学習・推論過程に残すハイブリッドアーキテクチャ。本プロジェクトでは、全フラックス直接回帰ではなく、決定フラックス予測＋辞書再順位付けを採用。
  https://www.nature.com/articles/s41467-023-40380-0

- **Amos and Kolter (2017)**, OptNet: バッチ化QP層のGPU解法。本系の制約規模（6,585反応、3菌種共有培地）への直接適用はVRAM・因子分解コスト大。
  https://proceedings.mlr.press/v70/amos17a.html

- **Agrawal et al. (2019)**, 微分可能凸最適化層: パラメータ化凸最適化の微分可能層化手法。
  https://papers.nips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html

- **Ross et al. (2011)**, DAgger: 学習方策が誘導する状態をラベル化、分布シフトを緩和。本プロジェクトのPHA誤差改善に利用。
  https://proceedings.mlr.press/v15/ross11a.html

- **Boyd et al. (2011)**, ADMM: 分散最適化・統計学習のための交互方向乗数法。GPU凸包QP補正の基盤として利用。
  https://web.stanford.edu/~boyd/papers/admm_distr_stats.html

- **Hallmann et al. (2026)**, gsMOBO: 培地・流加液探索のBayesian Optimization。外側探索器としての利用が適切。
  https://spj.science.org/doi/full/10.34133/csbj.0072

- **Song et al. (2025)**, FBA反復のANN置換: 炭素源ごとに代理モデル構築。本系の3-GEM・20,200次元contextとは設定が異なる。
  https://doi.org/10.1038/s41598-025-89997-9

- **Fan et al. (ICML 2023)**, LP初期basis予測: 学習予測を最終解ではなく厳密解法の加速に使用。
  https://proceedings.mlr.press/v202/fan23d.html

- **Sambharya et al. (L4DC 2023)**, 学習器出力を反復最適化のwarm startとして使用。
  https://proceedings.mlr.press/v211/sambharya23a.html

- **NVIDIA cuOpt**: PDLPは大規模LP向け、小〜中規模にはdual simplex推奨。LP BatchSolveは廃止予定のため、独自並列化として単一GPU所有サービスを実装。
  https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html

**(出典: GPU_BATCH_QP_MIGRATION_REPORT.md, GPU_SURROGATE_ARCHITECTURE.md, NEURAL_MECHANISTIC_GPU_DFBA_REPORT.md, GPU_DICTIONARY_ACTIVE_LEARNING_AND_SAFETY_REPORT.md)**

---

## 11. 構成元ファイル

| ファイル名 | 元の日付 | 備考 |
|---|---|---|
| COOPERATIVE_SURROGATE_32768_REPORT.md | 2026-09-01 | WCFS1構成の32,768候補辞書報告 |
| COOPERATIVE_SURROGATE_SCALING.md | (記載なし) | WCFS1構成のサイズ・要件分析 |
| GPU_BATCH_QP_MIGRATION_REPORT.md | (記載なし) | バッチQP移行後の性能報告 |
| GPU_DICTIONARY_ACTIVE_LEARNING_AND_SAFETY_REPORT.md | (記載なし) | 辞書改良と資格ゲート実装 |
| GPU_SURROGATE_ARCHITECTURE.md | (記載なし) | WCFS1構成のアーキテクチャ設計 |
| NEURAL_MECHANISTIC_GPU_DFBA_REPORT.md | (記載なし) | ニューラル–機構ハイブリッド報告 |
| PHA_GPU_ROLLOUT_ACCURACY_REPORT.md | (記載なし) | DAgger改良後のPHA精度改善報告 |
| PFREUDENREICHII_GPU_QUALIFICATION_20260903.md | 2026-09-03 | Pf構成の認定試験結果 |
| PFREUDENREICHII_GPU_QUALIFICATION_PLAN.md | 2026-09-03 | Pf構成の認定計画 |
| PFREUDENREICHII_GPU_REBUILD_20260903.md | 2026-09-03 | Pf構成の辞書再構築報告 |
| GPU_UTILIZATION_OPTIMIZATION_RTX4060.md | (記載なし) | GPU利用率改善とベンチマーク |
