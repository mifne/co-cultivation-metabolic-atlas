# 精度維持を優先したGPU加速：実装と再評価

## 結論と対象

OR16＋NS21＋P. freudenreichiiの3段階LP（maxmin→aggregate→exchange）を対象に、GPU常駐反復とCPU/GPU並行実行を実装した。完全GPU化を必須条件にせず、同じ精度・同じCPU比較条件の総時間で採否を決める。生物モデル、lexicographic目的、元LPゲート、PPO既定経路は変更していない。

実装計画は `docs/ACCURACY_FIRST_ACCELERATION_PLAN_20260905.md`。元LPの主残差≤1e−5、双対違反≤1e−7、相対KKT gap≤1e−7を維持。終点のPHA相対差≤1%、菌体量最大絶対差≤0.01 g/L、PHV分率絶対差≤0.01を別途確認する。CPUとの数値一致は培養実験での妥当性確認を代替しない。

## 1. GPU上の固定構造・反復buffer・Graph再利用

`GpuPdhgWorkspace` はFP64 CSRと転置、主双対変数、作業bufferをGPU上で保持する。1反復を2 SpMV＋2融合更新kernelとし、固定chunkをCUDA Graphとして再生する。`GpuResidentReducedPdhg` が厳密等式縮約・主双対復元・元LP証明を接続する。

- 同じ疎構造なら係数/RHS/上下限/目的/step尺度だけを既存bufferへ更新する。
- 環境順序・CSR構造の変化は検知して再構築する。等式fingerprintや不正入力を無視しない。
- 最終元LP証明を返却合否の根拠とする。過去checkpointで合格していても最終証明が不合格なら採用しない。
- 返却配列を所有させ、次stepの更新で過去結果が書き換わることを防ぐ。未認証解は次の培養状態やPPOへ使わない。

4環境×3stepのexchange replay（2,048反復、check256、chunk64）は次の結果だった。現在のCPU参照解はGPU初期値へ渡していない。

| 実装 | CPU総時間 / 合格 | GPU cold総時間 / 合格 | GPU GRU総時間 / 合格 |
|---|---:|---:|---:|
| 常駐Graph | 1.0488 s / 12/12 | 1.0831 s / 0/12 | 1.0033 s / 0/12 |
| 融合loop | 1.0545 s / 12/12 | 1.0943 s / 0/12 | 1.0419 s / 0/12 |

定常の反復区間はGraph約0.1517秒/4LP、融合loop約0.1626秒/4LP。起動負荷削減の効果と数学的な収束は別で、**不合格LPの時間からCPUに対する速度倍率は計算しない**。step2は疎構造変更で再構築、step3は動的値更新で再利用できた。host縮約・入力更新・合否判定・互換出力は残るので、完全GPU環境ではない。

データ：`results/pf_resident_graph4x3_20260905.json`、`results/pf_resident_loop4x3_20260905.json`。それぞれsource snapshotを保存。評価用traceは学習に転用していない。

## 2. 成熟したcuOpt-PDLPでも収束・速度を分けて検証

厳密縮約後（3,919行×4,962変数、24,397 nnz/LP）の独立4LPをblock-diagonalでcuOpt Stable3/FP64/PSLPへ投入した。CPU cold HiGHSは4/4合格、0.3905秒。GPUは3秒上限・78,400反復で0/4、動的縮約から元LP証明まで3.3388秒だった。

元LPの主残差は9.67e−4〜1.07e−3、双対違反1.04e−4〜1.34e−4、相対gap2,394〜2,619。主因はbounds側の相補性で、縮約・復元の差ではない。CPUより長い時間を使って未収束なので、同設定を10秒へ延長する追試は行わず、現構成への統合を棄却した。

cuOptのdirect Python APIはhost配列を介する。CPU optimizer/crossoverを使わなくても、この経路を完全GPU内完結とは呼ばない。データ：`results/pf_reduced_cuopt_pdlp_step1_4_20260905.json`。

## 3. GPU判定待ちとCPU厳密計算の重複実行

旧32×120計測では、maxminがCPU68.603秒に対しhybrid85.715秒だった。GPU準備・辞書判定33.933秒を先に払い、失敗分を後でCPUへ戻す直列構造が問題の一つだった。

新しい `--speculative-cpu` は次の順序で動く。

1. 同じoffline辞書初期化・同じpersistent HiGHS・4 workerで、全環境のCPU仕事を投入する。
2. 独立にGPU辞書を評価し、元LP認証を行う。CPU結果はGPU候補へ与えない。
3. GPU認証済み行だけ、まだ始まっていないCPU仕事を取り消す。
4. 実行中のCPU仕事は全件joinし、GPU不合格行には認証済みCPU結果を使う。例外時にも仕事を残さない。

CPU/GPU spanは重複するので足さない。実行CPU件数、採用したCPU解、GPU採用で未使用となったCPU解、未開始取消を分離して記録する。完全GPU化ではなく、厳密CPUを併用する加速候補である。

### 4候補、32環境×8step、2組

両側ともfrozen配列入力・後段非同期化・辞書初期化CPU・各LP1thread/4worker。seed20295201–32はCPU先行、20295233–64はGPU先行。各組768 LP/方式を最後まで計算した。

| 組 | CPU (s) | Hybrid (s) | Hybridの時間増加 | 未開始CPU取消 | 未使用の実行済みCPU解 |
|---|---:|---:|---:|---:|---:|
| 0 | 29.5214 | 30.9177 | 4.73% | 41 | 54 |
| 1 | 34.6370 | 35.5519 | 2.64% | 34 | 52 |

全元LP・終点ゲート合格、PHA・菌体量・PHV分率の終点差はいずれも0。CPU仕事は1,536件から1,461件へ減ったが総時間は短縮しなかった。**CPU処理を減らしたこととCPU単独より速いことを区別し、この設定は不採用とする。** 開発用2組の記述値で、信頼区間や一般的優越性は主張しない。

4環境×3stepのsmokeも全精度合格だがCPU1.7418秒→hybrid1.9064秒。4workerに対して4環境では取消0件だった。データ：`results/pf_speculative_smoke4x3_20260905.json`、`results/pf_speculative32x8_20260905.json`。source snapshotは各実行時点の版を保持している。

## 4. 次の採否判定

4候補での減速を受け、1候補版を同じ32環境・同じ2組の初期条件で測定した。

| 組 | CPU (s) | Hybrid (s) | Hybridの時間変化 |
|---|---:|---:|---:|
| 0 | 29.4746 | 30.6918 | +4.13% |
| 1 | 30.7372 | 30.5330 | −0.66% |

全元LP・終点ゲート合格、終点差0。2組合計はCPU60.2118秒、hybrid61.2248秒（1.68%増）で、再現性のあるCPU超えとは判断しない。GPU採用88/512 maxmin LP、未開始CPU取消44件、未使用CPU解44件。CPU単独を0.66%上回った1組だけを取り出して優越性を主張しない。`results/pf_speculative_k1_32x8_20260905.json`。

この版ではworkerの実start/endも測定し、GPU判定後にjoinした観測spanと、実CPU処理の完了時刻を区別した。基準が緩くなったわけではない。また診断上の安定環境IDとbatch内位置を分けて保存し、非有限・負の残差・不正な候補index/shapeではCPU仕事を取り消さないよう強化した。

学習拡大については、既存候補の単なる全探索ではなく、training-onlyデータにある未収録active setをofflineで追加し、元LP証明のcoverageとGPUメモリ上限に基づき少数の候補を選ぶ準備へ進む。現bankは既存4候補に4軌跡×10時点の40候補を加えた44候補で、480 training stateの残り440時点からの基底は未収録。これを診断データ混入で埋めない。

`basis_coverage_selection.py` に、厳密基底statusの署名・重複集計、連続軌跡/全GEM provenance/禁止seedの検査、認証済みboolean coverageに対する決定的set-coverを実装した。既存候補の保持・候補数・byte上限を扱い、整数の黙示変換やuint64 overflowによる予算逸脱も拒否/防止する。**選定層は実装したが、新しい拡張辞書の作成・router再学習はまだ行っていない。**

次のbuilderでは、すべての入力chunk hashを追加し、480 training stateの基底収集→元LP認証coverage→96/128候補の選定→複数正解候補を教師とするtop-K routerを作る。メモリは同じ最大配列幅の仮定で96/128候補時のpool used約0.71/0.83 GBの概算であり、temporary/cacheを含む実測peakではない。480候補もVRAMだけで即不可能とは断定しない。採否は候補採用率ではなく総時間で決める。

ゼロ補正hybridが不要なfull repair operatorを要求していたCLIも修正した。補正0ならcompact bankだけで起動し、正の補正budgetでは従来どおり検証済みoperatorを要求する。これにより拡張辞書に全逆行列operatorを一律生成する必要がなくなる。修正後4×3 smokeは全精度ゲート合格（`results/pf_speculative_no_repair_smoke4x3_20260905.json`）。初回のCLI拒否は計算開始前に検出・修正し、結果を上書きしていない。

GPU-only PDHG/GRUは収束不十分なのでPPOへ昇格しない。まず有効解を返すmaxmin辞書経路で総時間を改善し、失敗時はCPUへ直接戻す費用制御を検証する。短期・中盤で総費用が有利になった方式だけを120step・独立条件へ進める。

## 5. 最終検証と運用状態

実装変更後に既存CPU/GPU・hybrid・pipeline・取消・coverage選定を含む821テスト、別processで時系列・縮約・診断126テスト、合計**947件（重複なし）**が合格した。前者39.94秒、後者3.14秒。cuOpt/CuPyとTorch/旧conditional captureの既知の依存関係競合を避けるためprocessを分け、依存ライブラリは更新していない。

4件の閉ループ結果ファイルを独立集計で再検査し、全LP証明・完走・CPU実行数/取消数・全worker join・終点差を確認した。4×3 smoke2件、32×8×2比較2件の全てが精度合格。未使用CPU failure warningは0。既定の学習設定は変更せず、未認証fluxを使ったRL学習を開始していない。

全テスト・実測ジョブは終了した。今回の安全性・実行基盤の実装は完了したが、再現性のあるCPU超え、全3段階の完全GPU化、PPO学習全体の高速化は未達/未検証として残す。時間の不利な構成を既定にせず、次の実装はRevision 3に記したtraining-only辞書作成・router比較から再開する。
