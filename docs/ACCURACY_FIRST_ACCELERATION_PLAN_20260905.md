# 精度維持・計算全体のCPU超えを優先する実装計画

## 目的と採用条件

ユーザーの条件は「精度を維持してCPUより速くする。完全GPU化は、それが速い場合に採用する」である。GPU使用率、VRAM消費量、未収束反復の速度を目的指標にしない。GPU-onlyとhybridを候補として残し、既定経路を実測なしに切り替えない。

- 数理モデル：OR16＋NS21＋P. freudenreichii、maxmin→aggregate→exchange、dt=0.2 h、120step=24 h。反応・制約・目的順序を速度のために変更しない。
- 元LP：主残差≤1e−5、双対違反≤1e−7、相対KKT gap≤1e−7。候補を縮約/正規化しても元座標へ復元して独立認証する。
- 軌跡：PHA終点相対誤差≤1%、菌体量最大絶対差≤0.01 g/L、PHV分率絶対差≤0.01。全3段階を同じ経路で解くclosed-loopで確認する。単一段階replayで代替しない。
- 性能：同じ初期条件・流加入力・frozen配列入力・4 CPU worker/各1 thread・利用可能な辞書/前回basisを揃えたCPU対照。変換、転送、補正、認証、状態更新、フォールバックを含む総時間で比較する。初回準備・教師作成・学習費用を別記し、償却可能性も示す。
- 完全GPU：オンラインCPU LP呼出し0だけでは不十分。3段階LPと環境状態更新のdevice常駐、host読戻し境界も監査する。GPU-onlyがhybrid/CPUより遅ければ採用しない。

## 現在の根拠

二項等式縮約はexchangeを7,373→4,962変数にし、元LP復元36/36件を認証した。しかし4環境×41stepの既存GRU＋縮約PDHGは0/164件合格、29.964秒（CPUは164/164件、10.166秒）。残差は残存多項等式と広域bound active setに残り、主双対復元誤差が原因ではない。

固定入力・同じ2,048反復のgraph試作は1.80倍速かったが、元LPは0/4件。数学的な収束と実行コストを分離して改善する必要がある。新規GRUの大型化だけや、同じPDHGの反復数だけを増やす実験は優先しない。

## 実装順序と分岐

| 段階 | 実装・調査 | 採用条件 / 失敗時の変更 |
|---|---|---|
| A1：GPU実行基盤 | 固定FP64 CSR、転置、T、dual復元、反復bufferとCUDA graphを再利用。動的係数/RHS/bounds/目的を同じアドレスに更新。環境ID順・CSR pattern変更は検知して明示再構築 | 同じ反復数で既存x/y・元LP残差と一致、環境別freeze/reset、stale input拒否。まず速度だけでなく準備・更新費用込みで測る |
| A2：収束経路 | 縮約後の元目的LPを成熟したGPU PDLP/barrierと比較。presolve、warm-start、host処理の実態を記録。必要なら残存等式の補完/active-set法を比較 | 元LP証明合格数と総時間で判定。0件なら同じ設定の長期運転やGRU大型化へ進まない |
| B：収支に基づく経路選択 | 段階・状態領域別にGPU候補採用率、probe費用、CPU残余LP費用を測る。採用が見込めない区間は直接CPUへ。fallbackは理由/回数/時間を明示 | `GPU probe費用 + 残余CPU費用 < 全CPU費用` が実測で成立する領域のみ使う。GPU-onlyも同じ総時間で比較 |
| C：学習による補正費用削減 | 収束する補正器に対し縮約primal・残存dualを別尺度/別headで予測、またはactive-set変化を提案。短い補正unroll後の元LP残差・相補性・目的誤差で学習 | 同じ補正器のcold/前回認証解/MLP/GRU対照。教師MSEではなく合格までの時間で採用。効果がなければGRUをオンライン経路から外す |
| D：閉ループ統合 | 全3段階のlexicographic順序、培地/菌体/PHA/体積/酸素等の状態更新を統合。GPU状態更新は一部LPがhostへ戻る場合の転送費も評価 | 短期→中盤→120step、CPUとの終点基準全件。未認証fluxで次時刻へ進めない |
| E：性能確定 | 4/16/32/64環境、実行順交互、複数反復、構成確定後の独立holdout。最後にPPOの学習全体を比較 | CPU/GPU/hybridの合格結果のみ速度比と不確実性を算出。2倍以上は開発目標で保証値ではない。優位性が小さい/不確実なら採用保留 |

完全GPU化の作業をすべて先行させず、A1とA2を並行して進め、まず有効解を返せる高速経路を選ぶ。入力更新がGPU常駐でも、補正が収束しなければDへ進めない。逆にCPUを条件付きで使う方式が最速なら、それを実用版として残してGPU-onlyは研究用比較とする。

## 計画を変更するタイミング

1. 各実装単位のtoy/回帰試験後。
2. 独立短期replay後：合格率0または総費用悪化なら原因を分類し、同じ失敗実験を大規模化しない。
3. step41等の中盤後：制約/目的切替による悪化を検査する。
4. 全120step後：局所高速化が準備・転送・CPU待ちで相殺されていないか再評価する。

過去結果とsource snapshotは上書きせず保存する。診断用traceを学習用に転用せず、train/dev/holdoutとモデル・T・特徴・補正設定のhashを区別する。依存環境は既存cuOpt/conditional graphと新Torchを当面process分離し、無条件のupgradeをしない。

## 一次資料と適用範囲

- [Practical Large-Scale Linear Programming using PDHG, NeurIPS 2021](https://arxiv.org/abs/2106.04756)：実用PDLPは単純PDHGだけでなく前処理、presolve、適応step/restartを組み合わせる。現試作を同等実装とみなさない。
- [Restarted Halpern PDHG](https://arxiv.org/abs/2407.16144)、[cuPDLPx](https://arxiv.org/abs/2507.14051)：LPの識別・収束を改善する研究。一般ベンチマークの倍率を本GEMへ転用しない。
- [NVIDIA cuOpt公式FAQ](https://docs.nvidia.com/cuopt/user-guide/latest/faq.html)：インストール版のAPIと照合し、GPU method/CPU concurrent/crossover、presolve、warm-startの条件を明記する。
- [DC3](https://arxiv.org/abs/2104.12225)、[Learned Warm Starts](https://www.jmlr.org/beta/papers/v25/23-1174.html)：制約補完と補正後損失の設計根拠。GRUが必ず速いという根拠ではない。

## Revision 1：着手

A1の常駐workspaceとA2の収束経路比較を開始。併せて既存hybridの費用を再監査してBの条件を具体化する。現時点のCPU超え/完全GPU化/終点精度の達成状態は変更していない。

## Revision 2：収束経路を保留し、直列待機を取り除く

A1の常駐workspace・融合更新・動的Graph再利用を時系列backendへ接続した。CSR/環境順変更は再構築し、元LPの最終証明を返却合否へ反映する。workspace 36件、resident/backend 34件の試験に合格。実LPでは疎構造が変わるstep2は明示再構築し、step3は値だけ更新した。これは完全GPU環境ではなく、host入力・縮約更新・証明判定・互換出力境界は残る。

A2の縮約cuOpt-PDLP（Stable3/FP64/PSLP、3秒上限）はstep1の独立4LPで0/4、総時間3.3388秒。CPU coldは4/4・0.3905秒。元LP primalは約1e−3、dualは約1e−4、relative gapは2,394–2,619で、縮約座標との残差差は小さい。成熟したPDLPの反復を追加するだけでは現在のCPU超えに繋がらないため、この構成の長期運転は棄却する。

Bの詳細監査では、旧32×120実測のmaxmin累計がCPU68.603秒、hybrid85.715秒。後者はGPU準備8.555秒＋辞書25.378秒の後でCPUへ戻る直列構造だった。次の限定実装は「先にGPUを待つ」方式から、同じ辞書初期化CPU全件とGPU認証を同時開始する方式へ変更する。GPU認証済み行の**未開始CPUタスクだけ**を取り消し、実行中タスクは全件joinしてから次stepへ進む。現在CPU解はGPU入力へ渡さない。

- GPU/CPU spanは重複するので加算しない。総wall、実開始CPU件数、未採用CPU解、取り消した件数を分離する。
- maxmin限定、補正0回、明示opt-in。aggregate/exchangeは従来の厳密CPU。モデル・精度閾値・PPO既定は不変。
- CPU側にも同じ辞書・4worker・frozen入力・非同期後段を使い、短期A/Bで全3段階LPと終点を確認する。
- 冷間起動/重複CPU作業/CPU-GPU資源競合で遅ければ採用しない。観測済みCPU全batch時間と実hybrid時間から経路選択を検討し、reject件数によるCPU時間の線形外挿はしない。

A1の実LP追試（4×3、exchange、2,048反復/check256）も完了した。常駐GraphではCPU1.0488秒・12/12に対してcold1.0831秒・0/12、GRU1.0033秒・0/12。融合loopはcold1.0943秒、GRU1.0419秒で同じく0/12。Graphの定常反復は約0.1517秒/4LP、融合loopは約0.1626秒/4LPとなったが、いずれも未認証なのでCPUに対する速度倍率は掲載しない。初期準備とCSR変更時の再構築も総時間に含む。合格0のため41/120stepへ拡大しない。

Bの4環境×3step closed-loop smokeは元LP・終点とも合格。CPU1.7418秒、speculative hybrid1.9064秒。4CPU workerに対して4環境なのでCPU全件が開始済みで、取り消し0件・実CPU36件（うちGPU採用で未使用3件）。小規模では不採用。次の限定比較は32環境×8step・2組・CPU/GPU順交互とし、未開始仕事を取り消せる場合の実費用を評価する。

## Revision 3：候補数とCPU待機だけでは不十分、認証coverageを改善する

32×8・2組は終了し、4候補版はCPU29.521/34.637秒に対しhybrid30.918/35.552秒、1候補版はCPU29.475/30.737秒に対しhybrid30.692/30.533秒。全元LP・終点に合格、終点差0。ただし4候補は両組で減速、1候補も合計1.68%減速した。1組だけの0.66%短縮を再現性のあるCPU超えとは扱わず、120stepへ自動昇格しない。計画のBを「並行化すれば解決」から以下へ改訂する。

1. **完了した基盤を保持する。** 常駐Graph/融合反復、最終元LP証明、並行CPU取消/join、実行CPUと未使用CPUを区別する計時、安定ID、例外時cleanupをopt-inとして維持。PPO既定は変更しない。
2. **辞書候補の品質を先に改善する。** 現44基底は既存4＋training4軌跡×10時点の40基底。480 training stateの全時刻から元LP認証済み基底を収集し、trainingのみのcoverageで96/128候補を選ぶ。全候補のオンライン全探索や全逆行列repair生成はしない。必要な選定層（署名、重複、trajectory/GEM/seed guard、mandatory/byte予算付きset-cover）は実装・テスト済み。拡張辞書本体とrouterは未作成。
3. **入力由来を先に固定する。** 新builderは旧bank manifest、全training chunk、GEM、選定設定をhash保存する。既存診断traceと今回32×8のseedはtrainingへ転用しない。holdoutを後から学習データに改称しない。新規dirに出力し既存44bankは不変。
4. **少数の有効候補を予測する。** 不合格best候補を正解クラスとする旧単一ラベルではなく、元LP認証合格をmulti-hot教師とするrouterを比較する。nearest/MLP/GRUの必要性を同じonline K=1/4・同じCPU経路で検証し、学習・変換費用を別記する。
5. **費用判定を閉ループ総時間へ戻す。** 小規模・中盤で、同じ完全CPU batchとhybridの実測wallを比較する。候補採用率だけ、partial CPU時間の線形外挿だけではenableしない。利益がないbucketはCPU直接、必要時の再probe/cooldownは別の明示オプションとしてテストする。
6. **精度・速度の合格後に拡張する。** 120step、独立反復、PPOの順。完全GPU化には後段aggregate/exchangeの元LP収束が別に必要で、現PDHG/GRUは未達。GPU辞書でmaxminだけ速くなっても完全GPU化やPPO全体の大幅高速化とは呼ばない。

補正0のhybridがfull repair operatorファイルを必須としていたCLIも修正し、compact bankだけで起動・CPU安全経路を維持できるようにした。正のGPU補正には引き続き検証済みoperatorが必要。詳細結果・再現データ索引は `docs/ACCURACY_FIRST_ACCELERATION_RESULTS_20260905.md`。

最終回帰は別processの821＋126＝947件（重複なし）に合格。全実測は終了。新しい拡張辞書/学習済みrouterやGPU全段階閉ループを完成済みとは扱わない。今回の実測ではCPU超えを安定して示せなかったためPPOの既定経路は切り替えず、上記2–4を次の実装単位とする。
