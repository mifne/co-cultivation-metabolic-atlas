以下の統合ドキュメントは、OR16 + NS21 + P. freudenreichii のdFBAにおけるGPU加速LPソルバー開発の全記録を、提供された4つのファイルから時系列順に再構成したものです。GPUを用いたデータフロー加速、全段LPのGPU化、数値解法の内点法への移行、GNN/GRUによる学習候補器導入などを経て、すべての試行においてCPU超えは未達であり、本稿はその結果と限界を正確に記録するものです。数値は原文から転記し、複数ファイル間の時間経過やベンチマーク条件の相違は注記で明示しています。

---

# GPU加速dFBA/LPソルバー開発：データフロー加速・統合ドキュメント

## フェーズ1: 初期データフロー律速改善 (2026-09-05)

(出典: `DATAFLOW_ACCELERATION_PLAN_20260905.md` Rev 1–5、`DATAFLOW_ACCELERATION_RESULTS_20260905.md`)

### 1.1 ベースラインと律速分析

`pf_coverage96_k4_32x8_20260905.json` を用いた32環境×8step×2組の計測では、CPU単体 **58.9708秒** に対しGPU併用 **61.6208秒**（**4.49%遅い**）であった。CPU対照は、同一辞書/MLP、persistent HiGHS、4 workers×1 threadを使用。全認証・終点ゲートに合格した。

非重複CPU wallの内訳は、後段CPU完了待ち32.044秒（54.3%）が最大であり、GPU自体が主律速ではないことが判明した。

### 1.2 実験1–4（閉ループ比較）

4構成の32環境×8step×2組比較を完了した。全LPでprimal∞≤1e-5、dual違反≤1e-7、relative KKT gap≤1e-7を要求。終点PHA相対≤1%、菌体量最大絶対≤0.01g/L、PHV分率絶対≤0.01を要求。全実験で元LP認証・終点ゲート通過。

| 実験 | 構成 | CPU単体 (s) | GPU併用 (s) | 時間増減 | GPU認証/512 maxmin LP |
|---|---:|---:|---:|---:|---:|
| 1 | K1・必須認証のみ・従来転送/dispatch | 56.3065 | 57.1364 | +1.47% | 138 |
| 2 | K4・GPU内順位・集約転送・GPU投入後にCPU開始 | 59.6479 | 59.7026 | +0.09% | 225 |
| 3 | 実験2 + 転送配列のpackingもCPU開始前へ移動 | 61.9098 | 62.4974 | +0.95% | 225 |
| 4 | K4・GPU認証後、不合格LPだけCPU計算 | 59.9879 | 61.6430 | +2.76% | 225 |

最もCPUに近かった実験2でも合計 **0.0547秒遅く**、CPU超えは未達。GPU認証は全LPの14.6%（225/1536）に留まり、後段LPは全てCPUが担当した。

(食い違いに関する注記: `DATAFLOW_ACCELERATION_PLAN_20260905.md` Rev1のベースライン58.97秒/61.62秒と、`DATAFLOW_ACCELERATION_RESULTS_20260905.md`の各実験CPU時間は異なる。これは実験1–4で実行構成が変更（辞書/MLPは同一だが、dispatch方式や非同期化の有無が異なる）されたことによる変動であり、実験グループ内の直接比較に留める。)

### 1.3 `certificate-only` 経路と内部最適化

入力再生によるmicrobenchmarkでは、K=1で約11%、K=4で約14%短縮。受理mask、受理解、目的値、元LP残差、候補IDはbitwise一致。GPU間の順位をGPU内に保持し、7回の結果downloadをFP64の1回へ集約。GPU投入後のCPU開始方式はopt-inとして維持する。

### 1.4 次の方針

後段CPU処理が依然支配的であるため、元LP入力の一度限りの解析と共有、aggregate/exchange段階の詳細計測とGPU適用を優先する。

---

## フェーズ2: 全3段階GPUバッチ化と辞書再構築 (2026-09-05)

(出典: `DATAFLOW_ACCELERATION_PLAN_20260905.md` Rev 6–8、`GPU_ALL_STAGE_BATCH_RESULTS_20260905.md`)

### 2.1 辞書・ルータ再構築

教師は `pf_neural_basis_actual4x120_20260904`、4軌跡×120step。以下は学習データ上の数であり、独立評価ではない。

| 段階 | 旧候補数→新候補数 | 全候補で認証できる状態 | nearest K4 |
|---|---:|---:|---:|
| aggregate | 52→96 | 61→127 / 480 | 59→117 / 480 |
| exchange | 44→96 | 39→91 / 480 | 37→89 / 480 |

段階別順位ルータを作成。bank SHA変更時は未変更段階のルータだけ全artifact・特徴・候補順・GEM一致検証の上で明示的に再bindingする。

### 2.2 閉ループ結果

32環境×8step×2組

| 指標 | CPU | GPU併用 |
|---|---:|---:|
| オンライン総時間 (s) | **60.6204** | **73.0904** (20.57%減速) |
| 後段GPU認証率 (aggregate) | – | 5.47% (28/512) |
| 後段GPU認証率 (exchange) | – | 3.52% (18/512) |
| CPU差戻し件数 | – | 1295/1536 |
| 全元LP・終点基準 | 合格 | 合格 |

終点最大差はPHA相対6.8204e-8、菌体量2.0159e-8 g/L、PHV分率1.1521e-7。全終点差既定0.01以下。

### 2.3 候補不足診断

8 LP×96候補のCPU sparse LU再因子化でも、元LP認証は **0件**。3候補解釈は特異/不正なbasis。各LPの最良主残差は約0.003–3.73で閾値1e-5を超えた。

### 2.4 次の方針

固定辞書の拡張では限界があり、ユーザーの指摘を受けて「有効制約を探索できる汎用GPU数値解法」へ方針転換。cuDSS・内点法・GNNを検討する。

---

## フェーズ3: GNN/GRU試作と非同期パイプライン (2026-09-05)

(出典: `DATAFLOW_ACCELERATION_PLAN_20260905.md` Rev 9–12、`GPU_PIPELINED_DFBA_20260905.md`)

### 3.1 cuDSS内点法とGNN/GRU試作

- **cuDSS内点法**: 実GEM由来のNewton行列で動作したが、全3段階4環境で元LP認証 **0/12**。Newton系の不安定化が残り、正則化・反復改良では解消しなかった。
- **GNN/GRU**: 全次元主/双対head、DLPack経由の接続を実装。未学習推定は独立4環境のexchangeで同期wall中央値 **2.935ms**（GNNのみ）、**3.626ms**（GNN+GRU）。予測の正確さ・補正短縮の実績ではない。

### 3.2 非同期パイプライン実装

CPUのLP計算中にも別環境の主スレッド処理（状態更新）を進める方式を追加。maxminだけGPUで処理し、後段はCPU直行かつ非同期で発行する。

#### 32環境×8step結果（1組目）

| 指標 | CPU・非同期 | ハイブリッド・非同期 |
|---|---:|---:|
| オンライン総時間 (s) | 39.867 | **38.008** (4.66%短縮) |

この比較では、非同期ハイブリッドがCPU非同期を上回った。

#### 120step検証（32環境）

| 指標 | CPU | GPU/CPU hybrid |
|---|---:|---:|
| オンライン総時間 (s) | **845.541** (14.09分) | 898.378 (14.97分) |
| CPU/Hybrid比 | – | 0.941186 (6.25%増) |
| 完走環境数 | 32/32 | 32/32 |
| CPUへのLP要求数 | 11,520 | 10,873 |

(食い違いに関する注記: 同一実装において8stepではHybridがCPUを上回ったが、120stepではCPUがHybridを上回った。8stepの結果は限定的な条件での結果であり、120stepの結果が実条件をより反映すると判断する。固定された一貫した優位性は本時系列全体を通じて確認されていない。)

### 3.3 Basis Handoff（CPU基底引き継ぎ）

GPUで認証された辞書基底をCPUに引き継ぐ `--cpu-basis-handoff` を実装。4環境×8stepでは反復が29.17%減ったが総時間は短縮せず。32環境×8stepのoff/on比較でもHybrid総時間は短縮せず、**既定offの実験オプションに留める**。

### 3.4 PPO精度設計

報酬・GAE・意思決定・イベントの保存を検証。一律1%や1e-7をPPOの普遍的要件とはしない。拡張20D観測、gamma整合、任意KL停止を実装したが、実GEMの学習成功は未検証。詳細は別途 `PPO_ACCURACY_SPEED_RESULTS_20260906.md`。

---

## フェーズ4: 内点法安定化・Krylov補正 (2026-09-06)

(出典: `DATAFLOW_ACCELERATION_PLAN_20260905.md` Rev 13–26)

### 4.1 正則化とFGMRES

正則化行列を作成し、元Newton系をGPU上の柔軟なGMRES（FGMRES）で補正する経路を実装。小LPのGPU認証に成功したが、実GEM交換段階では元LP認証 **0/4**。正則化1e-9では方向不良、1e-6では100反復安定も主残差約4～5e-4で認証0/4。

### 4.2 等式簡約（HomogeneousEqualityReduction）

森林簡約→ゼロ固定/重複行簡約→境界dual消去→GPU分解・FGMRES→逆順復元→元LP認証、の構成を限定試験。交換段階の変数を **7373→4741** へ縮約。全3段階×4環境は元LP認証 **0/12**、step2 exchangeも **0/4**。

### 4.3 中心化inexact方式と行簡約

中心化inexact方式でexchange4環境のprimal実行可能性はすべて元基準以内へ改善。最適性未達。maxminは解析的row dual y=0で **4/4厳密認証** (GPU 5.729秒、CPU 1 worker 0.538秒)。aggregateは1/4、exchangeは0/4。

binary64の厳密有理数検証で26候補中 **25行** を簡約。exchange4環境はcentered 9.659秒・PC 13.103秒でともに認証 **0/4**。

### 4.4 初期バリア平衡化と小演算集約

内部dualをslackに応じて初期化するbalanced方式でmaxmin 32/32認証（GPU 5.438秒、setup別途7.968秒）。
cuDSS内部の二重補正を削除（内部補正0回＋元残差ゲート維持）でmaxmin 32/32認証（GPU 単回3.909秒、CPU 1 worker中央値4.135秒。CPU 4 workers 1.239秒では未達）。
GMRESの逐次MGS、Givens回転・後退代入を融合。

stream分割は **採用せず**（直接1x32=3.626秒、協調1x32=3.810秒、4x8=6.137秒で悪化）。CPU1.15.1導入（元環境1.14維持）、16 workers中央値 **0.7644秒**。

時刻間数値更新API実装（step3/4更新込みGPU 4.0956/3.7944秒、CPU 16 workers 0.4921秒で及ばず）。内部y/z/sの再構成(functional warm-start)により解法時間は改善（step3/4 中央値0.9229/0.7534秒）も、更新込みでは依然CPU超え未達。

---

## フェーズ5: Device常駐数値更新と半減目標 (2026-09-07)

(出典: `DATAFLOW_ACCELERATION_PLAN_20260905.md` Rev 27–35)

### 5.1 証明再利用とDevice常駐更新

zero-face証明のPython再走査を1.968秒→0.121秒へ削減。二次map構築を1.077秒→0.156秒へ削減。
#### DeviceNumericUpdatePlan

現在のLP数値をGPU上で縮約・operator更新、元LP認証維持。32環境×7時刻の全224件認証、GPUのCPU LP呼出し0回、構造再構築0回。係数更新約0.05～0.07秒。CPU 16 workersは0.298～0.469秒。回帰1555 passed / 4 skipped。

### 5.2 半減目標への挑戦

目標：hot6時刻合計 3.523秒（block diagonal版）→ **1.762秒以下**。
- 証明再利用（initial non-redundancy）で初回準備 17.1→**7.0秒（59.0%短縮）**。
- しかしhot6時刻は 3.47秒と改善なし。lifecycleは25.6秒→15.0秒（41.3%短縮）。

| 方式 | 初回準備 (s) | hot6時刻 (s) | lifecycle (s) |
|---|---:|---:|---:|
| 従来GPU | 17.095 | 3.413 | 25.563 |
| 証明再利用GPU | 7.014 | 3.470 | 15.018 |
| CPU16 workers | 0.812 | 2.229 | 3.041 |

### 5.3 その他の試行と結果

- **Schur補行列**: 不定系2,015次元へ縮約。非ゼロ数が増加し改善なし。
- **多重中心性補正**: 分解回数は減ったが総時間半減には不十分。
- **Douglas–Rachford＋Anderson**: 4環境500反復で違反約1.33が残り不合格。
- **制約変更の逆追跡**: 単一rayを阻んだ元LP行4738は `EX_arg__L_e` の共通供給制約確認。前時刻fluxの単一倍率変更では足りず、反応の組合せ変更が必要。
- **複数半空間投影**: 4環境step3で200反復後も最大違反0.0482、元LP不合格。

---

## フェーズ6: 学習候補器と大規模教師収集 (2026-09-07~08)

(出典: `DATAFLOW_ACCELERATION_PLAN_20260905.md` Rev 36–44)

### 6.1 GNN+GRU教師あり学習

第1回（60epoch）：開発用4環境step3/4のfactorが前解19回、GNN 35回、GNN+GRU 46回。改善せず。
第2回（160epoch）：train-only RMS下限、physics loss（元単位の最大行違反）を追加。

独立seed4環境step2–8、3反復の結果：
| 方式 | hot24 LP/試行 中央値 (s) | 補正反復/LP |
|---|---:|---:|
| 前解warm | **1.645** | 7.46 |
| GNN | 3.350 | 17.38 |
| GNN+GRU | 3.721 | 19.71 |

各方式hot72/72件が元LP認証（CPU LP fallback 0）だが、生学習候補は0/72件しか直接認証されなかった。高速化失敗。次の候補は「認証済み前回flux＋現在残差→修正方向」の学習へ変更。

### 6.2 大規模教師データ収集

512軌道×120時刻×3段階=184,320 LPの収集を予約・実行。

#### 32軌道完了（11520 LP）
2026-09-07 17:29 JST完了。最大 relative KKT gap = 8.994034056675459e-8。収集catalogのSHA256適合確認済み。

#### 強制シャットダウン後の整合性検査（2026-09-08）
旧収集プロセスの消失を確認。学習用192軌道とselection16軌道は完了。193-200軌道は80/120時刻でmanifest保存。`scripts/audit_graph_collection.py` で全74880 LP payloadのSHA256一致確認（28.55秒、約6.90 GB）。診断ディレクトリのroleはdevelopment_diagnostic_not_trainingのまま保持。収集を再開。

直近10完了組は平均807.83秒/組。残り304軌道（38組）の単純外挿では2026-09-08 09:06頃終了見込み（停止・負荷変動を含まない概算）。

### 6.3 最新Status・今後の未実施項目

- 学習用512軌道収集は完了。全軌道を用いたGNN/GRU学習、熱気比較速度検証、閉ループdFBA/PPOへの昇格は**未実施**。
- CPU速度超え、完全GPU内完結、PPO既定の変更は**全て未達**。
- 現時点で本番設定へ接続されたソルバーは存在しない。

---

## 統合結論

本ドキュメントでは、GPU加速dFBA/LPソルバー開発の全過程（初期データフロー最適化、全段GPU化、非同期パイプライン、内点法・Krylov法、GNN/GRU学習）を網羅した。すべてのフェーズにおいて、CPU対照（HiGHS 4 workers）に対する一貫した速度優位は確認されなかった。

数値的不安定性に対する長期的な取り組みにより、maxmin段階ではGPUがCPU単独workerに肉薄あるいは微差で上回る局面もあったが、後段LPを含めた全段階の最適化は未達であり、CPU並列（16 workers）には大きく及ばなかった。

2026-09-08現在、PPO既定は変更されておらず、プロジェクトの主要目標である「CPU速度超え」は未達成である。学習候補器を用いた別アプローチが試行中であるが、その有効性は未検証である。

---

## 参考文献

本プロジェクトの設計・検討にあたり、以下の文献を参考とした(出典: DATAFLOW_ACCELERATION_PLAN_20260905.md Revision 37)。これらの文献の理論・手法を本GEM系にそのまま適用できる保証はなく、各変更の採否は実測に基づいて判断している。GNN+GRUを用いた学習方針(フェーズ6)の直接の出発点となった。

- Faure, L., et al. (2023). Neural initialization combined with mechanistic LP/QP for metabolic models. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10400647/
  （ニューラル初期化と機構論的LP/QPの組合せ。本系における半減保証ではない。）
- Qian, C., et al. (2024). LPのMPNN(Message Passing Neural Network)表現と近似求解. *Proceedings of Machine Learning Research*, 238. https://proceedings.mlr.press/v238/qian24a.html
  （LPのグラフニューラルネットワーク表現と近似求解。本系の厳密LP認証・速度優位を保証しない。）
- Gao, S., et al. (2024). IPM-LSTM: 内点法反復の学習. *NeurIPS 2024*. https://papers.nips.cc/paper_files/paper/2024/hash/de0da9c42ee713f2ceaeed7bc40c522d-Abstract-Conference.html
  （培養時系列ではなくIPM反復自体を学習する手法。本GEM系での転用実績はない。）

なお、GNN+GRU方針の初期設計(Revision 9, 2026-09-05)ではこれに先立ち、[LPのGNN表現(ICLR 2023)](https://arxiv.org/abs/2209.12288)、[FlowGAT(2024)](https://www.nature.com/articles/s41540-024-00348-2)、[学習warm-start(JMLR 2024)](https://www.jmlr.org/beta/papers/v25/23-1174.html)も参照された。

---

## 構成元ファイル

- `docs/DATAFLOW_ACCELERATION_PLAN_20260905.md` (計画: 2026-09-05 ~ 09-08, Rev 1–44)
- `docs/DATAFLOW_ACCELERATION_RESULTS_20260905.md` (実測レポート1: 2026-09-05)
- `docs/GPU_ALL_STAGE_BATCH_RESULTS_20260905.md` (実測レポート2: 2026-09-05)
- `docs/GPU_PIPELINED_DFBA_20260905.md` (実測レポート3: 2026-09-05)
