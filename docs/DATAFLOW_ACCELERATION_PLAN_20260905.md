# データ処理フロー・律速改善計画

## Revision 1 — 実測に基づく優先順位

対象はOR16 + NS21 + P. freudenreichii。GEM・目的関数・境界条件・物理時間刻み・認証閾値は変更しない。辞書/MLPは既存96候補のSHA固定版。新しい方式はopt-inとし、PPO既定を変更しない。

1物理stepでは現在の培地・菌体量・actionからLPを組み立て、maxmin → aggregate → exchangeを順に解き、その後状態を更新する。独立環境は並列化できるが、同一環境の3 LPと次の物理stepには依存関係がある。

### ベースライン

`pf_coverage96_k4_32x8_20260905.json`の32環境×8step×2組ではCPU58.9708秒、GPU併用61.6208秒（4.49%遅い）。CPU対照は同一辞書/MLP、persistent HiGHS、4 workers×1 thread。全認証・終点ゲート合格。

- 非重複CPU wall: maxmin9.407秒（16.0%）、後段CPU完了待ち32.044秒（54.3%）、主thread再開11.504秒（19.5%）、CPU入力準備5.857秒（9.9%）、scheduler0.157秒。
- GPU認証208/512のうちCPU実行を省略できたのは76件。132件はCPUも計算済み。
- GPU入力正規化/準備1.852秒、GPU候補検査区間計5.366秒はCPUと重なるspanであり、wallへ加算しない。

したがって現在は「GPUが主律速」とはいえない。maxminだけを無時間にしても、この計測における理想短縮率の上限は約16%（他の経路が不変の場合）である。

### 修正・検証順

1. **候補順位の不整合**: 独立入力4×120の因果replayで、前回認証候補の無条件先頭挿入はlearned K1を193→147件へ悪化させた。元々top-K内にある場合だけ優先するopt-inを追加。候補集合と元LP証明を維持する。
2. **不要なGPU処理**: zero-repair時に使わない分散・違反個数・修正用候補scoreを省くcertificate-only経路。完全な元LPの主/双対/KKT検査とfamily guardは残す。通常経路との同値テストと実LP microbenchmarkを先に実施。
3. **重複入力処理**: CPU/GPU bounds解析のPython走査、CSR合成の重複、normalize/stack/転送を個別測定。高速化を両比較側へ適用し、CPU対照を意図的に弱めない。
4. **閉ループ計測**: 4×3 smoke → 32×8交互順の同条件対照。GPU認証率だけでなくCPU取消・重複・非重複wall内訳を確認。差が小さければ新seed・長いhorizonで再検証する。
5. **計画更新**: 改善が総時間に効かなければCPU/GPU重複の削減と後段aggregate/exchangeを優先する。5条件最終精度試験を遅い方式のまま走らせない。

### 合格条件

原LP: primal∞≤1e-5、dual違反≤1e-7、relative KKT gap≤1e-7。終点: PHA相対≤1%、菌体量最大絶対≤0.01g/L、PHV分率絶対≤0.01。未認証解で状態を進めず、CPU取消は元LP GPU認証後だけ。全workerをjoinし、未使用のCPU解も実行数に含める。

オフライン学習/構築時間、setup、online wall、microbenchmarkは分けて記載する。固定CPU軌跡の入力再生は閉ループ検証ではない。CPU数値一致は実培養妥当性を保証しない。

設計参考: [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)の転送・同期の削減とbatch化。文献の一般論を本系の速度実績として扱わず、各変更を実測で採否判断する。

## Revision 2 — 同期・CPUとの重複を優先

certificate-onlyは実入力32 LPの単独GPU microbenchmarkでK1約11%、K4約14%短縮し、受理解をbitwise比較した。ただし最初の閉ループ32×8×2では総時間はまだCPUより遅い。単独rank約1.5msに対しCPU並行中rank14–29ms、結果完了/転送/検査34–39msという差が判明した。後者はGPU純演算時間ではない。各`.get()`やscalar同期に伴うGIL再取得・CPU workerとの競合が仮説であり、転送byte量が主因だと断定しない。

次は候補順位をGPU内に保持し、特徴/標準化/logitのfinite条件をGPU boolean maskで最終受理に伝搬する。有限性検査は省略せず、従来のfail-fast APIも維持する。7個の結果downloadをFP64の1配列にまとめ、元LP認証とhost側のshape/有限性/閾値検査は維持する。CPU取消は、この最終検査が終わってからだけ許す。

未使用診断の削減・bounds高速化と同じ基準のCPU対照で再測定する。numeric ndarray boundsの大幅なmicro改善を、list-of-pairsを使うlive環境へ転用しない。raw trace再生と実環境の両形式を別々に測定する。

## Revision 3 — GPU準備を完了してからCPUを起動

第2実験はrank約2ms、normalize約15–19msへ改善したが、転送配列を作るCuPy呼出しがCPU worker開始後に残り、completion区間が約0.13–0.15秒へ悪化した。転送回数を減らすだけでは不十分だった。次の変更はpackingもGPU投入フェーズ内へ移し、その後CPUを開始、最後は既に準備済みの配列を1回downloadする。元LP/finite gateや実行数会計は不変。第2実験の結果を上書きせず、第3実験のsource snapshotと比較を別保存する。

第3実験はK=4を維持する。単独GPU K4は約9msでCPU候補準備（約40–60ms）との重複が期待でき、K1より候補認証率が高い。これは検証すべき仮説であり、K増加自体を速度向上と扱わない。

## Revision 4 — 並行実行を前提にしない

packing前倒し後もcompletion/validation区間のwarm中央値約137msが残り、総時間はCPUより0.95%遅かった。packingのCPU開始後実行だけが主因という仮説は支持されない。単独入力再生との観測差はあるが、入力・実行条件も異なる。GIL、CPUメモリ帯域/スケジューリング、GPU待ち、host validationの寄与は未分離である。使用中highspyは1.14.0で、[同版の公式binding](https://raw.githubusercontent.com/ERGO-Code/HiGHS/v1.14.0/highs/highs_bindings.cpp)は`run`でGILを解放するため、「HiGHS solve全体がGILを保持する」とは説明しない。

第4実験は既存GPU-first経路に今回のtop-K保持と新辞書/MLPを適用し、認証後に不合格LPだけをCPUへ渡す。CPUとの重複計算をゼロにする一方、CPU開始はGPU検査終了まで遅れる。両方の総時間を測り、並行すること自体を目的にしない。こちらではfull diagnostic GPU経路とGPU由来の候補basis initializerを使うため、async K4との差を純粋なdispatch一要因の因果効果とは扱わない。

同時に、speculative例外時のCPU worker joinに加え、投入済みGPU streamのdrainを補強した。この例外専用変更は第3実験の完了後であり、保存済みsourceは変更しない。

## Revision 5 — 第1段階だけの最適化から、共通入力と後段処理へ

4構成の32環境×8step×2組比較を完了した。GPU併用の合計時間は各構成のCPU対照に対して+1.47%、+0.09%、+0.95%、+2.76%で、CPU速度超えは未達。全元LP・終点ゲート合格、PHA・菌体量・PHV分率の終点差は0。回帰テスト1,139件が合格した。速度優位の確認がないためPPO既定は変更せず、完全GPU化や120stepでの成功も主張しない。

第4実験ではCPUとの重複LPをなくし、CPU実行を1,536→1,311件へ減らしたが、wallは短縮しなかった。取消件数やGPU認証率だけで採否を決めない。入力数を増やすだけの辞書拡張も、この残存律速を直接は解消しない。

次の実装優先順位を以下へ更新する。

1. **同じ元LPを一度だけ解析する共通入力層**。全3段階の`PreparedLpRequest`を設計し、CPU比較側にも同じ改善を適用する。c/CSR/rhs/bounds/options/basisの所有権を明確にした不変snapshotからCPU solveとGPU認証を派生させる。enqueue後の元配列変更、reset/close、同一環境の同時solveを試験する。安易なmutable cacheは導入しない。
2. **長い完了区間の原因を分離する**。同一入力・同一warm条件で、GPU event、D2H、host validation、CPU worker内のparser/モデル更新/solveを分けて測る。GILやGPU帯域の単一原因を先に決めない。並行spanをwallへ二重計上しない。
3. **aggregate/exchangeを対象にする**。実験2 CPU wallのmaxminは14.8%に過ぎず、それ以外が約85%を占める。後段のどの処理を減らせるか確認してから、学習用のみの有効basis追加・低候補数のGPU認証を検討する。段階ごとの目的関数・制約変更を同一と仮定せず、全元LP認証を保持する。
4. **短い閉ループで速度を確認してから拡張する**。同一seedでの交互順反復と未使用seedの評価を区別し、初回構築を含むwallとwarm時間を両方報告する。速度改善が確認された方式だけを120stepの閉ループ・総合精度・PPOへ進める。

完了済みの改善と未完了の次段階を区別した[実測レポート](DATAFLOW_ACCELERATION_RESULTS_20260905.md)を併記する。長時間GPUを稼働させること自体を成果にせず、同じ計算・精度で短縮できた時間を最終指標にする。

## Revision 6 — GPUで解決できる仕事量自体を増やす

CPU差戻し件数の微調整では、後段CPU経路が全体時間を制限するという指摘を受け、全3段階をGPU-firstの環境×候補バッチへ通せる経路を優先する。全段でGPU内順位・必須認証のみ・単一結果転送を使い、CPU先行計算をしない。診断用のCPU禁止モードは未認証LPを失敗として返し、成功・速度優位に数えない。host状態更新までGPU化したという意味ではない。

既存aggregate52/exchange44辞書は旧全候補oracleとSHAが同じで、その診断では後段の救済がほとんど無かった。したがってGPU全段flagだけを追加した構成で長時間実験を繰り返さない。GPUへの入力をまとめる構造変更と、有効解を得る能力の改善を別々に検証する。

新たに、同じ元LPからGPU復元した複数候補の凸包/affine空間に元LPの認証可能解が存在するかを、小規模なCPU診断oracleで調べる。これは将来GPUで少数重みのバッチ補正を行う前の存在診断であり、オンラインGPUの成功や速度実績ではない。CPU参照解を候補に混ぜず、診断データを学習へ転用せず、復元した同一主解・候補双対の組に全元LP認証を要求する。この空間に解が無ければ小QPの実装を増やす前に方針を棄却する。

比較CPUには既存4-workerの非同期後段pipelineを維持し、GPU側は全stage barrier batchを使う。これは各実装構成のend-to-end比較で、同一schedulerの単一要因比較ではない。両側の元LP・精度・seed/action・辞書/ルータを固定し、CPU使用量だけでなくオンライン総時間を測る。

## Revision 7 — 後段の有効な解候補と、段階別の順位学習

全3段階GPU-first経路の4環境×3step×2組の機能試験は完走し、全認証・終点ゲート合格。ただしCPU合計3.474秒に対しGPU併用4.944秒、GPU認証15/72 LP、CPU差戻し57/72 LPだった。小バッチの機能確認であり、速度優位や大規模スケーリングを示す結果ではない。

後段候補の凸包/affine補正の存在診断では、行スケーリング後も16試行の元LP認証は0件だった。最適化ソルバが成功と返しても元LPの違反が残るケースがあり、投影層の実装だけを追加する根拠は得られなかった。候補復元の絶対値がexchangeで約1e18に達する問題も確認した。これは候補空間・数値安定性の問題であり、GPU演算量を増やすだけでは解消しない。

次の優先順位を以下に改める。

1. 学習用の4軌跡×120stepだけからaggregate/exchangeのbasisを再構築する。候補自身の元LP認証と全480学習状態のcoverageを確認し、認証不能な候補を辞書へ採用しない。変更しない段階はartifact単位のSHAとmetadataを保持する。
2. 全段階それぞれのcoverageを教師として、小さな順位MLPを学習する。モデルは候補を選ぶだけで、受理は元LPの主/双対/KKT認証が決める。4軌跡の内部交差検証は辞書から独立した汎化検証ではない。
3. 全体bank SHAの変更に伴う古いルータの無条件流用を禁止する。未変更段階に限り全ファイル・モデル・特徴・候補順の同一性を確認し、新しいSHA付きの明示的再bindingを行う。
4. 新しいbank/ルータで未学習seedの短い閉ループを再実行し、CPUと同じbank/ルータを使って比較する。その後32環境へ拡張し、全体wall・段階別認証率・CPU差戻し数を確認する。CPUより遅ければ成功とはせず、可変係数に適応する安定したGPU修正法を次段階とする。

aggregateは旧52候補から96候補へ再構築済み。学習480状態の全候補coverageは61→127、nearest K4は59→117へ改善したが、依然約4分の3を4候補では解けない。交換段階の再構築・段階別順位学習・未学習条件の評価を完了するまで、大幅高速化を主張しない。

## Revision 8 — 辞書の網羅から、構造共有型の汎用GPU解法へ

全段96候補と段階別ルータを作成し、未学習32環境×8step×2組を完了した。CPU60.620秒、GPU73.090秒で20.57%減速。後段GPU認証率はaggregate5.47%、exchange3.52%、CPU差戻し1295/1536件。全精度基準は合格だが、速度・完全GPU化は未達。詳細は [実測レポート](GPU_ALL_STAGE_BATCH_RESULTS_20260905.md)。

ユーザーの固定辞書の網羅性に関する指摘を受け、追加拡張を主方針にするのを止める。未学習8 LP×96候補のCPU再因子化でも元LP認証0だったため、候補選択と固定basisの再展開だけでは足りない。新規の有効制約を探索できるGPU数値解法を主系とし、辞書/NNは任意の初期化へ位置付け直す。

次は固定CSR superpattern＋解析再利用のGPU疎分解を限定診断し、その費用と安定性でバッチ内点法・dynamic revised/dual simplex・高度化PDLPを選定する。過去のcuOpt cold block/PDLP/GPU simplex失敗を未実施として繰り返さない。最新の「検討」依頼に合わせ、新規ソルバ実装へはこのターンで進まず、[代替方式の評価・検証条件](DICTIONARY_FREE_GPU_APPROACH_REVIEW_20260905.md)を整理した。

## Revision 9 — GRU・GNNを数値解法と分離して検証する

続く実装依頼とGNNの提案を受け、辞書非依存のGPU補正層と学習初期化を段階的に実装する。過去のGRUはCPU正解をPCA32へ圧縮・復元するoracle診断でも元LP認証0/12だった。これは当時のcodecに問題がある証拠であり、GRU一般やGNNの不可能性を示す証明ではない。

1. インストール済みcuDSS 0.7を版固定で使い、同一疎構造のバッチ数値分解と右辺solveを作る。symbolic analysisは再利用するが、係数が変われば数値分解を更新する。CPU reorderingを含むsetupとGPU数値演算を区別する。最初の試験でdeterministic modeと内部反復改良の非互換をログから特定し、反復改良を優先する設定へ修正した。
2. 元LPの境界・固定変数・全制約を保持したprimal-dual内点法の限定prototypeを作り、小LP→実GEM行列の順で費用と認証を確認する。CPU最適化へ暗黙に戻さず、未認証なら失敗として返す。前回解やNNなしでも解けるかを先に検査する。小LPやNewton方程式の成功をdFBA成功と呼ばない。
3. GNNは全LPの変数–制約二部グラフを使う。化学量論に加え、共有培地・成長率・parsimony等の補助制約も含め、現在の係数・rhs・上下限・目的関数を入力する。FlowGATのように現在のCPU FBA結果からグラフを構成しない。GNNで構造、GRUで同一環境の過去だけを扱い、全次元の主・双対候補を出す。PCA32による出力制限は引き継がない。
4. GNN/GRUの状態は環境・LP段階・モデル/graph identityに結び付ける。reset・並べ替え・失敗時の状態更新、非有限入力を試験する。未学習ネットワークを既定のPPOへ接続しない。学習時は軌跡単位で分割し、現在の正解や未来状態を入力へ混ぜない。
5. cold、前回解、GRU、GNN、GNN+GRUを、同一の元LP認証に達するまでの総時間で比較する。NN推論時間とGPU補正回数を分離し、GNN追加が補正削減以上に重ければ採用しない。最後に同等CPU対照の閉ループ精度・速度を検証する。

参考: [LPのGNN表現（ICLR 2023）](https://arxiv.org/abs/2209.12288)、[FlowGAT（2024）](https://www.nature.com/articles/s41540-024-00348-2)、[学習warm-start（JMLR 2024）](https://www.jmlr.org/beta/papers/v25/23-1174.html)。表現可能性の理論や他用途の精度は、本系の厳密LP認証・速度優位を保証しない。

## Revision 10 — GNN試作後も、実LPの収束を先に確立する

GNN＋GRU、全次元主/双対head、微分可能なLP残差、DLPack経由のGPU補正接続を実装した。独立4環境のexchange入力では未学習GNN推論2.935ms、GNN＋GRU3.626ms（同期wall中央値、setup別）。予測の正確さ・補正短縮・PPO速度の実績ではない。

cuDSS数値分解は実GEM由来の初期Newton行列でも動作したが、primal-dual内点法の全3段階4環境ではNewton系の不安定化が残り、元LP認証0/12だった。対称スケーリングと反復改良だけでは解消せず、非スケールの正則化Newton系の相対残差が1e-8を超える方向で更新しないguardを追加した。誤った方向で長時間回すことを防いだが、収束改善達成とは区別する。

次の優先順位を、(1) 冗長等式・スケール・正則化を個別に切り分けてGPU数値層を安定化、(2) solverの残差・相補性・補正時間を考慮したGNN/GRU学習、(3) 同一入力・同一元LP認証によるcold/前回解/GRU/GNN/GNN＋GRU比較、へ更新する。固定PCA32へ戻さない。GPU補正が未収束の間は本番への自動接続や大量の学習データ生成を行わない。

限定実装と結果は [GNN＋GRUレポート](GNN_GRU_GPU_IMPLEMENTATION_RESULTS_20260905.md)。本番設定・精度閾値・生物モデルは不変。CPU速度超えと完全GPU内完結は未達。

## Revision 11 — 数値的不安定性の原因を再現してから修正（2026-09-06）

1. 失敗直前のNewton行列・右辺・残差を診断用途だけに保存し、同じ線形方程式のCPU sparse LUとGPU分解を比較する。CPU線形代数の診断はCPU LP fallbackとは区別し、GPUの採用解へ混ぜない。原LPの正解x/yは引き続き入力しない。
2. 全体相対残差だけでなく、主/双対/相補性ブロック、slack/dualの比、正則化由来の方向誤差を記録する。GPU分解の不安定化、行列のスケール、冗長等式、内点法の進行のどこで問題が生じるかを切り分ける。
3. 原因に応じ、可逆スケーリング、等価な制約簡約、適応的なNewton正則化、残差に基づく反復改良・step選択を限定的に実装する。単にLPの精度閾値を緩めたり、生物モデルの境界・目的関数を変えたりしない。
4. 小LPの退化/冗長/スケール試験→同じ実GEM 12 LP→異なる物理時刻の入力で確認する。失敗した反復を続けるのではなく、改善しない要因の修正を止めて計画を改訂する。閉ループや120物理stepの成功とは区別して報告する。

参考: [Altman & Gondzio, regularized symmetric indefinite systems](https://www.research.ed.ac.uk/en/publications/regularized-symmetric-indefinite-systems-in-interior-point-method/)、[Pougkakiotis & Gondzio, 2019](https://arxiv.org/abs/1902.04834)、[Zanetti & Gondzio, 2025](https://arxiv.org/abs/2508.04370)。正則化や安定分解の一般原則を参考にするもので、各論文のアルゴリズムを再現済みという意味ではない。

## Revision 12 — 正則化による安定化と、元Newton系への収束を分離する

失敗直前の検査で、未検査のaffine方向からcorrectorの積を作っていた問題を確認した。affine方向を先に検査し、反復改良は環境ごとに真の残差が減った更新だけを保持するよう変更した。残差判定も全体最大値ではなく主・双対・相補性ブロック別にする。これは不良更新の防止であり、元LP収束達成ではない。

次に境界dualをNewton行列から代数的に消去し、さらにゼロrhs等式と既存境界の符号だけから厳密にゼロと証明できる変数を簡約した。step1の全3段階×4環境では追加190変数、明示的固定11変数、ゼロ行177本、完全重複/符号反転等式82本を除去できる。固定値はすべて正確にゼロであり、元モデル・目的関数は変更しない。証明を逆順に使うGPU dual postsolveと元LP全制約による認証を必須とし、重複行に載ったwarm-startのdualも代表行へ集約する。

しかしexchange4環境の簡約後も、正則化1e-9ではNewton方向の不良が残り、1e-6では100反復を安定に進めても主残差約4〜5e-4で元LP認証0/4だった。強い正則化を維持するだけでは元方程式の誤差が残るため、パラメータ総当たりを主方針にしない。

次は安定な正則化行列K_deltaを**前処理専用**に使い、元のK_0 Newton系をGPU上の小さなflexible GMRESで補正する。保存した前処理済み基底、二度の再直交化、真のブロック別残差、環境ごとの最良解保持とbreakdown検査を用いる。少数のKrylov反復で元残差が落ちなければ未認証として停止し、潜在的な残りの従属関係やLP初期化へ方針を戻す。GPU数値解法とCPU LP fallbackは混同しない。小LP→実GEM→異なる物理stepの順を維持し、本番PPOにはまだ接続しない。

参考: [Saad (1993), Flexible Inner-Outer Preconditioned GMRES](https://epubs.siam.org/doi/10.1137/0914028)、[SIAM/Netlib, Templates for the Solution of Linear Systems](https://www.netlib.org/templates/templates.html)。この補正の収束や本系での速度優位は実測で判断する。

## Revision 13 — 補正回数の増加を止め、既存の等式森林簡約を疎内点法へ再利用

FGMRESは小LPのGPU認証に成功し、元Newton系を直接検査すると初回で止まっていたexchangeでも10反復以上進めた。ただし16→32 Krylov反復への拡張、正則化1e-6→1e-8の変更でも、開発用step1の全3段階×4環境は元LP認証0/12だった。これは初回方向の改善であり、実GEM収束改善を達成したという評価にはしない。CPU LPへは戻していないが、未認証解はdFBAへ採用していない。

次に、既存のHomogeneousEqualityReductionを内点法にも再利用する。2変数の同次等式から作る疎な森林写像で、係数が連動する変数をまとめる。森林簡約→ゼロ固定/重複行簡約→境界dual消去→GPU分解・必要時FGMRES→逆順のGPU主双対復元→**最初の元LP**で認証、という構成を限定試験する。従来の縮約PDHG失敗を無視して同じ実験を再実行するのではなく、今回は疎分解型内点法との組合せを検査する。PCAや学習済み低次元表現による近似ではない。

森林の境界共通部分を決めた元変数へbound normalを割り当てる復元を維持し、単なるTによるdualの平均配分はしない。動的係数に対するplan fingerprint、環境ごとの写像、元LPの有限性・精度ゲートを確認する。収束しなければ未完成として残し、本番の設定や学習を差し替えない。

実装後、exchangeは7373→4741変数、分解行列8374次へ縮約できた。関連212テストは合格したが、step1全3段階×4環境は元LP認証0/12、step2 exchangeも0/4だった。次は保存した1個のNewton系で、全変数系と境界dual消去後の縮約系のKrylov補正を比較し、丸め誤差の再注入と残る従属関係を切り分ける。新方式の本番昇格・CPU速度超えは未達。詳細は [数値安定性の修正・検証記録](GPU_IPM_STABILITY_RESULTS_20260906.md)。

## Revision 14 — PPOの精度、内部反復の精度、元LPの妥当性を分離

ユーザーの精度見直し依頼を受け、[PPO精度・速度の再設計計画](PPO_ACCURACY_SPEED_PLAN_20260906.md)へ具体化した。学習用には報酬・GAE・意思決定・イベントの保存を検証し、一律1%や1e-7をPPOの普遍的要件とはしない。ただし現行gapのmax(1,|objective|)正規化を割合と誤認しない。元LPの主実行可能性とモデル境界は維持する。

実装は(1) 保存checkpointの目的尺度別再採点と初期ログの参照共有バグ修正、(2) 縮約座標Krylov、(3) 元非線形KKT残差を使うinexact中心化方向と下降性確認、(4) 同一入力のCPU実測、の順。縮約Krylovは40対象テストを通ったが、exchange4環境の同じ27反復で0/4のままだったため、単なるベクトル小型化を主対策とせず、内部過剰精度と非線形更新の進展へ重点を移す。

## Revision 15 — 物質収支の改善後は、後半の方向補正と報酬信号の保存を検証

中心化inexact方式でexchange4環境の主実行可能性はすべて元基準以内へ改善した。しかし最適性の最終認証とCPU超えは未達で、内部forcingを0.05→0.1へ変更してもKrylov上限で停止した。元LPの精度を緩めるのではなく、同じ状態で失敗した方向だけに限定した正則化retryを追加検証する。弱い正則化を最初から使った試験は悪化したため既定へ採用しない。

PPO精度は終点PHA割合だけでは決めない。現行Pf3種の3固定action・8step CPU試験では、1%PHA量を報酬換算した例示的尺度がaction間の報酬差より大きかった。観測・生報酬・PHA量/質量・終了・状態閾値を保存する機能を追加し、GAE・順位・明示した誤差予算によるオフライン評価を実装した。学習済みPf方策を使った有効性試験はまだ行っていない。比較用CPUは同じ4件のLPで1/4 workerを実測済みであり、失敗するまでのGPU時間を速度優位として扱わない。

## Revision 16 — 同じ厳密基準でmaxminを認証、残る段階と反復数へ重点を移す

全3段階×4環境の主実行可能性が基準内となった。maxminは既知の成長率上限を根拠に解析的なrow dual y=0を用意し、返却ペアを元LPで再検査して4/4厳密認証へ到達した。ただしGPU5.729秒に対して同一maxminのCPU1 worker中央値0.538秒、4 workers0.152秒であり、CPU速度超えではない。aggregate・exchangeは最適性未達。

正則化retryはexchangeを5.251→8.383秒へ遅くしたため拡大を止める。等式の残存従属・スケール診断と、安全策付きpredictor–correctorで反復数を減らす候補へ進む。PPOはNH4等を追加する版管理済み20D観測、gamma整合、任意KL停止を実装したが、実GEMの学習成功は未検証。詳細とデータは[PPO精度・GPU実装検証記録](PPO_ACCURACY_SPEED_RESULTS_20260906.md)。

## Revision 17 — 反復数ではなく、残存する等式ランク不足を先に扱う

exchange step1/env0のforest・zero-face処理後も、等式1,955本に対する構造ランクは1,940で、独立でない等式が残っている。数値QRでは1,929、26候補行の再構成後退誤差最大9.44e-16、RHS誤差0。CPUの読取り専用線形代数診断であり、LP fallback・行削除・GPU性能測定ではない。

次は候補の依存関係とRHSを検証して元LP認証を維持する安全な簡約を実装し、その後にpredictor–correctorの反復削減を試す。数値的に近い行を無条件に捨てない。285対象テスト合格、maxmin4/4厳密認証は今回の到達点。aggregate・exchangeとCPU超えは未達として明示する。

## Revision 18 — 厳密な行関係の簡約と、分解を共有する予測・補正を実装

ユーザーの継続依頼を受け、現行のPPO/dFBA経路と開発用GPU数値層を[CPU速度改善の実装計画](CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)へ整理した。まずbinary64係数・RHSを有理数として扱う厳密な関係検証を実装し、証明できた従属行だけをGPU作業LPから簡約する。元LPは保持して全行を認証する。

並行して、同じ疎分解をaffine予測とcorrectorで共有し、各方向を元非線形残差・実際のmerit低下で検査する方式を限定実装する。変更は個別にA/B計測し、正則化retryやKrylovの単純増加を主対策にしない。新方式のPPO自動接続や、未認証解を使う長時間学習は行わない。

## Revision 19 — 行簡約・予測補正のA/Bから、補正コストと誤検知の切り分けへ

binary64の厳密有理数検証で26候補中25行を簡約でき、同一等式の証明再利用も実装した。
ただしexchange4環境はcentered 9.659秒・PC 13.103秒でともに認証0/4。
PCでは分解0.477秒、solve・更新12.312秒、三角solve 1,182回だった。
原LP主残差は小さいが最適性未達であり、この時間を有効な解までの速度として使わない。

次はPCのunsafe-affine判定を詳細化し、最大境界ステップでの微小負値を診断する。
必要な安全判定を削除するのではなく、予測用だけにfraction-to-boundaryを追加比較する。
またglobalized経路がfull座標のKrylovを使っていること、残るnear-dependent等式、
正則化と残差スケールを区別して再点検する。既存縮約Krylovの過去の未収束も踏まえ、
同じ実験の単なる反復増量は行わない。本番PPOとCPU基準は維持する。

## Revision 20 — 初期barrierの平衡化は有効、CPU超えは未達として次の律速を限定

内部dualをslackに応じて初期化するbalanced方式と、予測点fraction=.995を限定実装した。
元モデルを変更せず、maxmin4環境は4/4認証・1.633秒となった（従来認証済みGPU5.729秒）。
aggregateは1/4、exchangeは0/4で、全体の安定収束は未達。予測点のunsafe拒否は
旧PC27件から新PC0件へ減ったが、終盤のfull forcing不足は残った。

globalized縮約Krylovは17,192→8,349成分へArnoldiを小さくしたが、今回exchangeでは
単独で高速化しなかった。barrier連動δも固定δより早期失敗となったので既定化しない。
元LPの許容誤差を緩めて成功率を上げる変更は行っていない。

32個の異なる開発用maxmin入力でGPU32/32認証・5.438秒、同一入力CPU3反復中央値は
1 worker4.135秒、4 workers1.223秒。GPU setup7.968秒も別途必要だった。
数値層・比較コード307テストと、重複しないGNN/GRU接続11テストの計318件は合格した。
リポジトリ全件の実行ではなく、テスト合格と実GEM全段収束・PPO成功は分ける。

次は最初に失敗する方向のブロック別精度と前処理を切り分け、認証済みmaxminでは
直交化/小演算/同期の費用を減らす。数値更新・永続cache・段階別バッチの本番接続は、
後段の元LP認証と短い閉ループで検証してから行う。
詳細は[更新計画](CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)と
[今回の実装結果](CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)。

## Revision 21 — GPU小演算の集約、二重補正の削減、元問題の目的差監査

cuDSS solveの前後のfinite検査をdevice上に保ち、内部経路だけ毎回のhost読出しを除いた。
無効laneはNaNで返して既存ゲートが拒否し、公開solveの厳格な契約は変更しない。
GMRESの二度の逐次MGSと、Givens回転・後退代入を独立に融合した。
32環境maxminでは融合だけでは5.596秒に留まったため、cuDSS内部で固定2回実行していた
補正と、外側の元Newton系補正の重複を分離した。内部補正0回＋元残差ゲート維持では
単回3.909秒・32/32認証。旧CPU1 worker中央値4.135秒より僅かに短いが、反復測定・
CPU並列・setup・後段LPを含む速度優位はまだ確認していない。

残る近依存1行の差は有限境界上で9.715e-14以下とFractionで検証した。
これは厳密な同値簡約ではないため、明示的な作業緩和としてのみ実装し、元LPは保持する。
元の全行認証に加え、従来の相補性指標に含まれなかった等式dual×残差と直接双対目的差を
追加ゲートで検査する。この作業緩和のみでもexchange認証0/4であり、既定化しない。

次は最初の失敗方向を保存して前処理の実残差を特定する。認証済みmaxminの速度再測定を
並行の優先課題とし、失敗停止までの短さを高速化と数えない。生物モデル・PPOは変更しない。

## Revision 22 — バッチ並列の実態を点検し、再縮約と反復制御を優先

既に32環境をcuDSS uniform batchへ渡している。GPU3試行中央値3.572876秒（96/96認証）、
CPU fresh-model再計測中央値は1 worker4.220701秒、4 workers1.239123秒。
solve部分はCPU単独に1.181倍となったが、CPU並列・cold setup込み・PPO全体では未達。
現在の主費用はKrylov/三角solve/更新67.2%。単なる小バッチへの分割や分解だけの高速化では
CPU4 workersに届かない。必要なのは同条件で約2.88倍の処理効率改善である。

入力の境界と初回失敗GPU状態を調べ、成長保持条件で必要な正のフラックスが極端に小さい
slackへ押し込まれ、等式方向を修正できなくなる箇所を確認した。ゼロ面簡約後に新しく
2項になった等式74本を2段目forestで座標に組み込む明示オプションを実装し、
元の全行・直接dual gapの認証を維持した。実GEM exchange4はまだ0/4で既定化しない。
等式2冪正規化、native cuOpt barrierへの縮約入力投入も未認証なので性能成功と数えない。

次の実装順は以下。

1. 環境数別の認証済みLP/sを測定。8/16/32の大きさ比較と、同じ総問題数を複数streamで
   分割する配置比較を区別する。各streamは専用handle/bufferを持つ必要がある。
2. 主要67%の制御同期・一時配列を減らす。有限性/active判定はGPU maskに残し、host読出しを
   所定境界へ集約する。計測のためだけの反復同期と実際の依存を分ける。
3. 適格な同一環境・stageの前回GPU状態だけをwarm startに使い、変化した数値だけ更新する。
   入力fingerprintで構造/証明/analysisを再利用し、constructor費用を毎回支払わないAPIへ進む。
4. 全段・実状態変化を伴う閉ループが認証できた段階で初めてPPOへ昇格する。

途中で空になった3実験ファイルを測定JSONのsource snapshotから復元し、関連テストを再実行。
データ破損を無視した継続や、欠損ファイルを正常な実装と扱うことは避けた。詳細と採否は
[バッチ速度の結果](GPU_BATCH_SPEED_RESULTS_20260906.md)へ保存した。

## Revision 23 — 比較条件を是正し、単項等式とstream分割を追加検証

CPU fresh時間はmodel作成を含み、GPU3.57秒はconstructorを除く。従来の1.18倍は
同じライフサイクル範囲の優位ではないため、全体のCPU超えの根拠にしない。
CPU比較wrapperを1/2/4/8/10/16 workersへ拡張した。繰返し測定中のnative異常終了は
成功測定として採用せず、HiGHSの作成・更新・solve・出力取得・破棄を同一owner threadに
固定した後に再測定する。モデル/basis再利用、solver設定、認証基準は維持する。

GMRESの中間host判定を3箇所抑えるopt-inを追加し、無効laneのGPU maskと元残差検査は
残した。32環境maxminは単回3.504秒・32/32認証で、小幅な改善に留まる。
入力構造の監査では、ゼロを跨ぐ境界を持つa*x=0の単項等式が未処理だった。
元入力のゼロRHSと既固定ゼロ列だけでx=0を証明し、GPU上で符号自由なdual q/aを逆順に
復元する。微小係数をゼロへ丸めず、abs(a)<1e-12は縮約せず保持する。

単項等式+second forest+exact equalities+同期削減のmaxmin32は単回3.279秒・32/32認証、
setup12.804秒。exchange4は0/4のままであり、削減した演算量を全段収束やPPO速度と混同しない。
既存near-equality候補の行番号は旧座標なので、この新縮約とは無検証で併用しない。

次の順序を維持する。

1. CPUのowner-lifecycle修正後、広いworker sweepを完了し、失敗試行を除外理由付きで保存する。
2. 同じ32個の異なる入力を1x32 / 2x16 / 4x8に分割するstream比較を実装する。
   各workspace/handle/streamは独立、analysisは順次完了させ、native APIは同一host threadから
   発行する。全stream完了のwall timeを測り、最速stream時間や各stream時間の平均で代用しない。
3. CPUにもGPUと同じ入力限定縮約を使える比較と、変化する状態に対する永続更新を進める。
4. 後段LPの未認証を解消し、因果的な全段dFBA閉ループで初めてPPO用backendへ昇格する。

GPUテストは測定と排他実行する。GPU利用率を上げる目的の重複LPや無意味な演算追加は行わない。

## Revision 24 — stream分割は採用せず、常駐workspaceと反復数削減へ

同じ32個の元入力hashをCPU/GPU全配置で確認した。GPUの独立handle/streamを同じhost threadから
event付きgreenletで動かす実装を追加し、全配置32/32認証を確認した。しかし単回測定では
direct1x32=3.626秒、協調1x32=3.810秒、2x16=3.969秒、4x8=6.137秒で悪化した。
factor後だけ切替える粗いscheduleも2x16=4.267秒、4x8=6.402秒で改善しなかった。
この構成を既定にせず、比較用の実装として残す。

小shardで環境単位の三角solve数は6,240→4,776へ減ったが、hostからのbatch solve呼出しは
195→597へ増えた。起動・同期・小演算の反復費用増加と整合する結果である。ただしkernel trace
なしに個別費用を定量断定しない。QR/縮約はsetup計測に含み、solve悪化の原因と混同しない。

CPU1.15.1は別targetに導入して元環境1.14を維持した。1/2/4/8/10/16 workersの3試行sweepは
exit0、実LP1,152回すべて認証・retry0。16 workers中央値0.7644秒を強い比較候補とする。
1.14のnative異常との同一原因は未確定で、owner-thread固定だけでは解消しなかったことも保存した。

次の優先順位は以下へ修正する。

1. 1x32の固定アドレスworkspaceへrhs、basis、方向、Hessenberg、norm、active maskを常駐させる。
   再利用によるarray alias/以前の解の上書きをテストし、有限値と元Newton残差のゲートを維持する。
2. インストール済みcuDSS0.7.1で、既にfactor済みの固定buffer solve＋真の残差の
   CUDA graph capture/replayが可能かtoy検証する。現行公開文書0.8と同じ機能を仮定しない。
   不可の場合は無理にguardを外さず、融合kernelと常駐bufferに限定する。
3. 固定長2～4反復の内部補正をまとめ、各反復のfinite/residual/best-state判定はGPU内に残す。
   公開GMRESの任意callback契約と、内部の数学的に非負な残差専用経路を分ける。
4. 同じ環境・stageの前回認証済みGPU内部x/y/z/sを因果的に再利用し、cold約38回のfactorも減らす。
   symbolic構造/縮約証明cacheは入力fingerprintで検証する。現CPU解・未来解をGPUへ渡さない。
5. CPUにも変化入力のbasis再利用と同じinput-only縮約を許し、数値更新～元LP認証まで同範囲で比較する。
   後段未認証と全段閉ループの問題を残したまま、PPO全体のCPU超えを主張しない。

今回まとめて実行した関連43テストファイルは772 passed / 1 skipped（14.03秒）。
GPU streamのpending件数はkernel同時実行の証明ではなく、個々のsolver phase時間はyield待ちを
含むので合算しない。すべての実測ジョブは完了し、元モデル・PPO既定は変更していない。

## Revision 25 — 時刻間の数値更新・内部状態再利用を実装して検証

GPUのバッチ演算自体は実装済みであり、単純なstream増加ではCPU並列を超えなかった。
これを踏まえ、同一環境・stageの時刻間で維持できる構造を調べた。開発入力の4環境×3段では
step1→2で固定変数集合が変わる一方、step2→3では全12組で最終等式構造が維持される。
元入力、縮約の座標、境界の証明、CSR patternを現在値で再確認する数値更新APIを追加した。
native handle/analysisと固定配列を維持し、数値因子・旧認証・旧内部状態は必ず無効化する。
更新途中のエラーはworkspaceを使用不能にし、準備段階の非互換だけ明示的なGPU再構築へ送る。

内部x/y/z/sの因果的な引継ぎと、GMRESの6大配列の再利用を実装した。公開解x/yだけから
毎回slack/不等式dualを初期化し直す経路と区別する。以前のCPU解・未来の解は入力しない。
bind時のtarget hashと更新generationを保存し、bind後にLPが変われば提案を拒否する。
slackを現在のh−Gxから復元する実験も追加したが、モデル境界や認証閾値は変更しない。
4環境のstep2/3/4は両方式とも合格したものの、改善は時刻依存なので既定化しない。

cuDSS 0.7.1の固定因子・製作行列4×17ではCUDA Graph再実行、異なるRHS、元残差、NaN拒否を
確認した。ただしこれは能力確認であって、全GEMのGraph化や数値再factorを跨ぐ安全性の実証ではない。

次の検証・実装順を以下とする。

1. 同じ保存入力step2/3/4について、GPUのfresh/rebind・cold/internal warmを比較する。
   準備、数値更新、解法、独立した元LP認証、破棄の時間をすべて保存する。
2. CPU HiGHS 1.15.1にも環境別モデルと前時刻basisの再利用を許し、同じ直接dual gapを追加確認する。
   CPU追加確認は返却済みowned snapshotから行い、余分なnative getter dispatchを課さない。
3. 数値更新層のhost再証明・payload組立・同期読出しを別計測し、支配的なら静的metadata再検証と
   動的値検査を分離して同期を集約する。更新時間を無償とは扱わない。
4. GMRES内の固定長処理をGraphへまとめる前に、pointer寿命・callback・finite判定・factor寿命を
   明示した狭い内部APIを設計する。公開汎用GMRESのエラー検査を単に削除しない。
5. maxminだけの改善で停止せず、未認証のaggregate/exchange、独立した閉ループdFBAへ進める。
   現段階では完全GPU内完結、PPO全体の高速化、CPU並列超えはいずれも未達と記録する。

## Revision 26 — 同一内部状態対照で再利用と数値停滞を分離

数値更新APIは32環境でもnative analysisを1回に維持できた。しかしstep3は更新3.1499秒＋
GPU解法3.6467秒で、前時刻basisを使うCPU16 workersの0.4921秒に及ばない。
step4はslack修復付きでも31/32認証。初期floorを1e-4へ上げるだけの変更はさらに悪化し非採用。
失敗時間や特定の速い時刻をCPU超えの根拠に使わない。全実測は
[時刻間再利用の結果](GPU_NUMERIC_REUSE_RESULTS_20260906.md)へ保存する。

4環境のstep4に同一のGPU内部snapshotを複製し、current matrix/KKT/証明配列が一致することを
確認した対照では、fresh/rebind両方が不合格となった。再利用だけに原因を帰すことはできない。
元LPの出力をanalytic-box dualで認証しても、再利用対象の内部y/zが中心性の良い状態とは限らない。
極小slackと巨大dualの組合せで、前時刻からのわずかな摂動でも更新方向が停滞する。

1. 元出力の認証と内部dual品質をmetadataで区別する修正は完了。
2. 旧巨大dualを捨て、現在の目的係数・bounds・GPU上のwarm xから内部y/z/sを再構成する
   限定オプションを実装した。物理boundsは変更せず、通常Newtonと元認証で処理する。
   4環境の試験後、32環境×3時刻×3試行で288問題すべて認証。GPU step3/4解法の中央値は
   0.9229/0.7534秒へ改善したが、更新込みは4.0956/3.7944秒でCPU並列にはまだ及ばない。
3. 再構成だけで不十分なら、旧最終点だけでなく中心性の良い中間反復点を少数保持し、現在LPの
   摂動を吸収できる候補をGPUで評価する。これはwarm-start候補であって認証済み最終解ではない。
   信頼区分・因果性を明示し、未収束候補を正解として渡さない。
4. 32環境で支配的なhost数値更新を減らす静的/動的分離と、固定factorのNewton処理Graph案を
   並行して設計する。4環境でGMRESが少ない結果を32環境にも一般化しない。

Gondzio & Grotheyの再最適化論文は、旧最適境界点が新LPでは悪い初期点になり得ること、中心経路に
近い適切な非最適点からの再最適化を論じている。今回の候補再構成は完全な論文再実装でも
収束保証でもない。入力・精度・CPU比較条件を維持して実測で採否を決める。

最新の内訳により、次の第一優先は約3秒のhost数値更新の削減とする。同時に新GPU hot経路は
step3で12 factor/24 solve、factor約0.44秒、その他反復約0.37秒なので、旧GMRES支配という
見立てをそのまま当てはめない。固定factor Graphのcapture償却だけでCPU超えになる保証はなく、
より適した初期点・中心性とfactor回数削減を併せて検討する。旧時刻・旧方式・異なる時間範囲を
混ぜた速度比は作らない。PPO既定への昇格条件（全段/閉ループ精度・CPU超え）は変更しない。

## Revision 27 — 数値更新の実測から、同一証明と森林構造の再利用を優先

32環境のstep2→3の数値更新だけをcProfileで分離した。プロファイラ付き総時間4.259秒のうち、
ZeroFace再証明1.968秒（row_proof 82,624回）、二次森林map構築1.077秒、旧・新payload構築
0.783秒が主な費用だった。旧stage_updatesは0.075秒であり、GPU→CPU読出しだけを第一原因と
した見立てを改める。このプロファイラ時間は計測介入を含み、速度比には使わない。

1. 等式係数/RHS、明示固定変数/値、各境界のゼロに対する符号が維持されるときに限り、
   同じ帰納的zero-face証明のPython再走査を省く。条件が変われば従来の全証明を行う。
2. 初回に独立再構築して検証・所有した森林planをfingerprintで確認し、現在入力から縮約と
   全境界witnessを再計算する。静的mapは再利用するが、動的係数・目的関数・witnessは更新する。
   旧host/device配列の破損、型、shape、deviceを検査し、準備失敗時は旧mapを変更しない。
3. 境界集約を事前に固定した群順序のreduceatへ置換する。負の重み・無限境界・witnessの
   tie-break・厳密な不可解判定は維持し、旧scatter版とランダム入力で比較する。
4. 85バッファのGPU照合結果を一度に小配列へ集約するオプションも実装するが、元費用が
   小さいため単独で実測し、逆効果なら採用しない。検査そのものを削除しない。
5. 以上の修正を32環境の同じstep2/3/4、同じ元LP証明基準で測る。数値更新と解法時間を
   分け、強いCPU16 workersの時刻間basis再利用と比較する。次に残るfactor反復数へ進む。

現時点で新規関連テスト270件（CUDAを含む）を通過。速度効果はこの改訂記載時点では未測定。
旧文書の3.57秒対4.22秒/1.24秒だけで判断せず、現在の入力と同じライフサイクル範囲を用いる。

### Revision 27の初回実測と次の絞込み

同じ32環境×3時刻の初回は96/96問題を元LP認証し、step3/4の解法は0.683/0.576秒、
更新・検証等込みは1.936/1.677秒、数値更新は1.152/1.032秒となった。これは単回であり、
旧n=3中央値との比を再現性ある速度比とは扱わない。device-staging集約版の単回は
1.950/1.709秒で、改善を確認できず既定にはしない。元LP・認証閾値・CPU LP呼出し0は維持。

同じ再利用版の再profile（最適化処理を実行しない診断）は904,022 calls/1.411秒。
zero-faceは1.968→0.121秒、二次mapは1.077→0.156秒（いずれもprofile介入あり）。
ただし残るpayload構築0.615秒、host再証明0.460秒はまだ大きい。
旧・新constraint_formの重複生成を4B→2Bへ減らし、証明fingerprintのJSONを同一byteに
保ったままimmutable leafのasdict再帰コピーも除いた。各同値性テストを追加した。

次の小改訂は、canonical CSRのblock diagonalをCOO経由で作らないこと、現在のE/Gと
既存KKT patternの厳密な座標被覆検証によってbmat/unionの再構築を省くこととする。
同時にGPU解法は最初の7反復で全環境が未合格であり、最後の数環境だけが原因ではない。
同じstep2のGPU状態を一度だけ生成し、mu=1e-3/1e-4/1e-5/1e-6の再始動を対照比較する。
中心性を弱めれば必ず改善するとは仮定せず、失敗・遅い試行も保存する。

## Revision 28 — 更新全体をdevice常駐化し、現在の制約違反も予測修正する

Revision27の実装を完了し、同一入力の3試行でGPU hot更新込みを4.096→1.722秒、
3.794→1.485秒へ短縮した。最終回帰は1,314 passed / 4 skipped（2GPU必須検査のみ）。
新GPU288問題、CPU16 workers288問題すべて元LP認証。詳細・rawへの索引は
[静的更新の測定結果](GPU_STATIC_UPDATE_RESULTS_20260906.md)に保存した。
CPU16 workersの同範囲は0.505/0.375秒で依然速い。GPU初回setup込み16.179秒も未改善である。
完全GPU化、全段閉ループ、PPO全体CPU超えの未達を変更しない。

残る数値更新は0.997/0.881秒、GPU解法のみでも0.646/0.522秒でCPU解法0.456/0.326秒より遅い。
したがって「host更新を省けばCPU超え」とも断定しない。次の優先順位を以下へ変更する。

1. **現在のLP数値→縮約→証明buffer更新をGPUへ移す。** 初回に検証した固定構造の
   DeviceNumericUpdatePlanを作り、現在のA.data/RHS/lower/upper/cをまとめて入力する。
   初期段階では生物学的なbounds/cost生成は既存host builderを維持する。Python round(...,6)、
   NH4<0.1や中間体>1e-12の分岐、alias代入規則まで同時に置換して同一性を崩さない。
2. 第1forest、zero-face、第2forest、exact equality、最終E/G/KKT、転置、full/intermediate
   認証buffer、両forestの現在のbound witnessを固定scatter/決定的segmented reductionで更新する。
   係数加算順序によるゼロ相殺・support変化は明示的に検出する。atomic加算の順序に証明を委ねない。
3. 静的等式/RHS・固定集合/値・有限境界mask・縮約後zero-sign・各段support・除去zero-rowの
   可行性・全有限値/overflow・device/stream/dtype/shapeをGPU stagingで一括検査する。
   条件外はfast pathを拒否し、明示的な再構築要求とする。旧解/旧認証で継続しない。
   全検査後のみcommitし、numeric factor・旧acceptanceを無効化する。
4. **GPU解法の初期可行性修正を減らす。** 同一sourceのmu比較は1e-5で各時点1factor減に
   とどまった。前回xは現在の制約に違反し得るため、中心性パラメータの小変更だけを延々探索しない。
   現在制約を使う主変数の予測・修正、または中心性の良い前回中間点からの再始動を小試験する。
   候補はGPU提案であり、新しい元LP証明の前に正解として採用しない。
5. 最初はmaxmin限定とし、GPU常駐更新＋解法＋同等認証の合計を強いCPUと比較する。
   入力生成・GPU更新は別段階として測る。現在数値generationとlayout契約を分離し、
   古いhost problem_hashを現在入力と表示しない。独立検証用には現在の原入力をexportする。
   aggregate/exchangeでは目的関数support/性能制約行が変わるため、同じ構造を無検証に流用しない。

根拠コードはcommunity_solver._community_inequalities（共有代謝物slot係数のbiomass依存）、
frozen_community_inputs.build_step（濃度依存boundsと分岐）、lp_equality_reduction.reduce、
lp_zero_face.rebind、lp_exact_equalitiesのproof reuse、gpu_block_lp.certify_blocks_device。
maxminの等式は固定でも共有行のゼロ相殺によりCSR supportが変わる場合があることを契約へ明記する。
単純なstream細分化、利用率を上げるためのLP複製、CPU比較条件の弱体化は採用しない。

## Revision 29 — GPU数値更新を検証し、反復線形代数へ重点を移す

2026-09-07。Revision28の1～3、5のmaxmin保存入力検証を実装した。
現在の数値をGPU上で縮約・operator更新し、元LP認証を維持したまま係数更新を約0.05～0.07秒へ
短縮した。32環境×7時刻のGPU/CPU各224件をすべて認証、GPUのCPU LP呼出し0回、構造再構築0回。
回帰1,509 passed/4 skippedの後に境界ケース3件と補助LP認証6件も追加・通過した。
比較集計の契約テストも含む最終再実行は1,526 passed/4 skipped、31.90秒で完了した。
[詳細とraw索引](GPU_DEVICE_UPDATE_RESULTS_20260907.md)を参照。

CPU16は更新・求解・検証込み0.298～0.469秒、GPU hotは0.556～0.778秒で、CPU超えは未達。
GPU初回構築費15.822秒も残る。host入力pack/hashと制御があるため完全GPU完結ではない。
全stage/閉ループ/PPOへの未達という判定とPPOのCPU既定を維持する。

1. 完了したGPU更新をopt-in経路として維持する。現在hash/generation、元LP認証、
   等式・境界・supportの条件外拒否を弱めず、過去host縮約を現在値として再利用しない。
2. 境界最適化を補助可行性問題へ変える案を試験したが、32環境のhot時間0.803/0.639秒で
   改善なし。初回解法のみ短縮したためcold-start候補に留め、標準経路へ昇格しない。
3. 主対象をfactorと三角求解へ移す。step3では0.249秒+0.279秒で求解0.621秒の大部分を占める。
   前回xの修正だけでなく、現在Newton系に対する旧factor前処理の再利用を検討する。
   必須条件は現在の全Newton残差検査、現在operator/旧factorの明確な分離、matching scaling保持、
   不十分な場合のGPU再分解、最終元LP認証。factor削減数だけでなく総三角求解数と総時間で採否判断。
4. 固定bufferのCUDA Graph化を別候補とする。同期・allocation・cuDSSデータ寿命を実GEMで確認し、
   toy再生やGPU利用率を速度優位の根拠にしない。上記準Newton案を同時に混ぜず対照比較する。
5. 候補が勝った後、同じ入力・強いCPU16・同じ計時範囲で反復測定する。さらにaggregate/exchange、
   GPU出力が次の状態を決める閉ループへ段階的に拡張する。保存入力の成功をPPO全体に外挿しない。

準Newtonの着想はGondzio・Sobralの著者公開稿に基づくが、論文手法の実装済み・本モデルでの
速度保証とは扱わない: https://www.maths.ed.ac.uk/~gondzio/reports/qnIPM.pdf

## Revision 30 — 前処理factorの低頻度更新を単独評価

GPU数値更新を固定し、warm solve内でのみfactorを2反復以上再利用するopt-in案を追加する。
現在のratio/全Newton系を目標に、旧factorとそのmatching scalingを前処理として使う。
現在の全残差がforcing基準を満たさない場合、同じ状態で現在GPU factorを再構築して再試行する。
LP間のfactor流用は行わず、各solveの初回は必ず現在値で分解する。既定interval=1は旧動作。
まずtoy精度・再分解・不正設定・連続更新を検証し、実GEM4環境→32環境へ進む。
反復数減だけでは採用せず、増えたKrylov/三角求解費、再分解率、総時間を記録する。
この追加時点では効果未測定、PPO既定や元LP認証閾値の変更なし。

### Revision 30の採否とRevision 31 — 実行レイアウトの対照比較

factor低頻度更新は4環境で精度を保ったが、hot求解0.282/0.234秒に対し1.036/0.857秒へ悪化。
factor数は10/8から減らず、追加Krylov・再分解・外側反復が費用を増やした。既定採用を見送る。
GPUガードを2カーネルに融合した別案も実装・テストしたが、32環境の初回試験では明確な総時間短縮なし。

次にcuDSSのnative実行をuniform batchから単一block diagonalへ変える実験を追加した。
論理的LP・環境ごとの元LP認証は変えず、疎な非対角ブロックは厳密に0。
現在値は既存連続bufferで更新し、native CSRも初回証明・更新前の静的検査に含める。
小規模の独立解・係数変更・不正native index拒否を検証し、32環境の初回ではhot求解0.545/0.461秒。
この時点では単回の小改善でありCPU超えではない。uniform/block/CPU16を同一入力・固定ソースで
3回ずつ順次実行し、初回構築・hot総時間・失敗を含めて比較する。新しい3案はすべてopt-inである。

### Revision 31の3反復結果と採用範囲

uniform/block diagonal/CPU16の各672件が元LP認証を通過した。固定ソース・同一入力を照合済み。
hot6時刻合計の中央値は3.928/3.523/2.249秒。block diagonalは従来GPUから10.3%短縮したが、
CPU超えは未達。初回込みsequence lifecycleは25.818/25.748/3.074秒で、初回費の問題も残る。
新方式は32環境maxmin繰返し検証のopt-in候補として残すが、PPO・他stageの既定は変更しない。
最終回帰1,555 passed/4 skipped。詳細は
[反復線形代数の最適化報告](GPU_LINEAR_ALGEBRA_OPTIMIZATION_20260907.md)とraw索引を参照。

次はfactor再利用回数の増加ではなく、同一状態で現在Newton系への補正費を評価する。
旧factorが有効な方向を事前診断できなければ、Krylovを増やす案は棄却する。
CUDA Graph化ではnative allocationとfactor寿命の検証を先に行い、CPU超えを仮定しない。

## Revision 32 — 半減を目標に、求解回数そのものを減らす

ユーザー指定の約50%短縮を、直近block diagonal版の更新・求解・検証込み時間を基準に評価する。
hot6時刻合計3.523秒→1.762秒以下が短期目標。初回込み25.748秒→12.874秒以下は別に判定し、
hotだけの成功を初回込み半減やPPO全体半減と混同しない。元LP精度・CPU16比較条件を維持する。

補助LPで資源制約に余裕を持つ候補を生成し、次の時刻は現在の元LPに対するGPU可行性・最適性の
認証を先に行う。認証を通った場合にだけ再求解を省略する。補助LPは正解の定義ではなく提案器で、
元LPの目的・制約・閾値は変えない。認証失敗時はGPUで候補を更新し、それでも通らなければ
未変更の元LPをGPU求解する。CPU参照解や未来時刻の数値を候補生成へ渡さない。

まず単一目的のmaxminと32個の異なる環境に限定する。全laneが現在元LP認証を通る保守的経路を
小規模で確認し、構築費・全再求解・独立検証も計時する。reserve率は開発用入力で試し、
未知入力や全stageへの網羅性は主張しない。この改訂時点では半減の達成は未確認。

### Revision 32の試験結果 — 高速でも未認証の候補は採用しない

reserve 50%を4環境×7時刻で試したが、前時刻の候補をそのまま再利用できた時刻は0。
全28件を認証したものの補助LPの再求解が毎回必要で、4.59～7.85秒/時刻となった。
資源に余裕を残すだけでは、次時刻に閉じる交換境界などへの対応を代替できない。

GPU SVDで求めた数値的アフィン空間内でMotzkin投影を行う候補器も試作した。
数値SVDは候補生成にだけ使い、元の等式・最適性検査は一切省略しない。
4環境のcoldでは2,000反復後の違反約0.0025、前時刻の認証済みGPU解からのwarmでも
5,000反復後に約0.045の違反が残った。0.134/0.294秒という候補生成時間を高速化の実績には数えない。
着想: De Loera et al., Sampling Kaczmarz–Motzkin, https://arxiv.org/abs/1605.01418
本試作は全行greedy選択であり、論文の完全実装・収束速度保証を主張しない。

cuDSS AMD並べ替えは32環境hotで1.896/1.560秒となり悪化。既定並べ替えの別分解方式は
0.682/0.525秒で小幅な変化に留まった。FP32 factor＋FP64原方程式残差補正はtoyを通ったが、
実GEMでは初回のNewton検査を通らなかった。前処理正則化1e-6/1e-4の双方を棄却。
CPUに戻さず失敗を記録し、FP64既定を維持する。混合精度の参考は
Carson & Higham (2018), https://doi.org/10.1137/17M1140819 であり、本モデルの収束保証ではない。

## Revision 33 — 同一構造の証明の重複を除去し、初回費と反復費を別々に改善

反復求解の半減とは別に、約16～17秒の初回構築を優先的に削減する。既定LP/精度を変更せず、
`--reuse-equality-proofs`で以下をopt-in実装した。

1. 同一batch内の等式A・RHS・形状が完全一致する場合に限り、等式forestの証明を再利用。
   hashは候補検索だけに使い、実配列の完全一致も要求する。各laneのplanはdeep copyで独立所有。
2. zero-faceの既存rebind検証を初回の環境間でも使う。現行の境界の0に対する符号、固定値、
   等式、全proofの整合性を検査し、現在の数値を再構築。不適合時は新しい証明を構築する。
3. 独立のforest再証明でも、現在入力から新規構築したcanonical証明に限って構造を共有する。
   各laneのmap・縮約LP・境界witnessの照合は省略しない。古いplan自身を正当性の根拠にはしない。
4. 新方式/無変更block diagonal/CPU16を32環境×7時刻、3反復、順序をローテーションして比較。
   全laneの元LP認証・入力hash・ソースhash・精度閾値・CPUのbasis/model reuseを照合する。

初回単回試験では初回10.44秒まで短縮し、全224件が合格した。ただしhot6時刻は約3.41秒で、
1.762秒の目標は未達。初回だけの半減を学習全体半減やCPU16超えと呼ばない。
固定ソースの3反復比較と最終回帰後に数値を確定する。次段階はwarm Newtonのfactor/三角求解の
半減であり、未認証のFP32・投影解を通す変更はしない。PPO既定のCPU経路は維持する。

### Revision 33の確定比較（3反復）

同一ソース・同一入力・順序ローテーションで従来GPU/証明再利用GPU/CPU16を比較し、
各方式672件が全て元LP認証を通過した。初回準備中央値17.095→7.014秒（59.0%短縮）、
初回準備＋求解・検証21.541→10.889秒（49.4%短縮）、sequence lifecycle25.563→15.018秒
（41.3%短縮）。**hot6時刻は3.413→3.470秒で改善なし**。CPU16はhot2.229秒、
lifecycle3.041秒で、依然として速い。初回の部分的成功を反復学習の高速化と混同しない。

最適化は厳密な構造準備のopt-inとして残す。次はwarm原Newton残差の停滞方向を調査し、
難しい少数方向のみ倍精度補正するGPU前処理を検討する。構造の仮定を認証なしに固定したり、
不合格GPU解を使う方針へ変更しない。詳細・棄却試験・再現条件は
[半減目標の最適化報告](GPU_HALF_RUNTIME_OPTIMIZATION_20260907.md)を参照。

## Revision 34 — 学習中の反復時間だけを半減する

初回準備の短縮は目標達成に数えない。更新・求解・元LP検証込みhot6時刻3.470秒を、
約1.735秒以下かつ公平なCPU16より短くする。全stage/閉ループ/PPOへの外挿はしない。

1. 境界を消去したNewton系の対角primalブロックも代数消去し、5,808次元の不定系を
   2,015次元のSPD Schur補行列へ変換するGPU実装を追加。LP自体の制約は一切消去しない。
   元の完全Newton残差・globalization・元LP認証を維持する。4/32環境で認証通過したが、
   行列の非ゼロ数は44,248から53,029へ増え、32環境のhot時間は0.696/0.560秒等で改善なし。
   uniform batch版も0.896/0.653秒で改善なし。`--newton-backend dual_schur`はopt-inに留める。
2. 現在のfactorを使う多重中心性補正を追加。新旧方向の合成候補に対して原Newtonのforcingと
   実測merit減少を検査し、更新幅が増える場合のみ採用する。4環境で分解回数11→8、8→6に
   減ったが、追加求解を含む総時間は0.496/0.346秒となり半減には不十分だった。
3. primal/dualの独立更新幅も候補化。独立幅だけでは初回が未認証となったため、同じNewton方向の
   共通幅と独立幅をそれぞれ検査し、meritが小さい方を選ぶopt-inへ変更。4環境で認証通過したが
   hot0.499/0.270秒で、明確な半減ではない。従来共通幅を既定として維持する。
4. 固定行列をLPごとに一度だけ分解するGPU Douglas–Rachford＋Anderson候補器を試作。
   目的座標の箱上限を補助問題で固定し、元LPの可行性・最適性認証を必須にした。
   4環境500反復で違反約1.33が残り不合格。1回のfactorという事実だけで高速化を主張しない。
5. 上記を受け、追加の解法変更より先にnative factor/solveのhost-return、CUDA event区間、
   完了待ちwall時間を分離計測する。計測用の同期は侵襲的なので、このprofile値を速度比較に使わない。

参考（実装・速度の保証ではない）:
- normal equations / augmented system: https://www.maths.ed.ac.uk/~gondzio/reports/ipmXXV.pdf
- 多重中心性補正: https://optimization-online.org/wp-content/uploads/2005/10/1233.pdf
- Anderson加速DR: https://arxiv.org/abs/1908.11482

全案でCPU LPへの隠れたフォールバック、元LPの閾値緩和、現在/未来のCPU参照解の使用はしない。
未認証候補は成功時間に数えず、失敗のrawも保持する。

## Revision 35 — 再分解を減らす案の検証結果と、制約変更への対処

同期付きnative診断では、32環境step3の求解0.529秒のうち分解11回0.204秒、
三角求解22回0.048秒だった。環境間factor共有は元LP未認証となり採用しない。
目的上限固定の補助IPMは32環境全224件が認証されたが、反復6時刻3.679秒。
今回対照の単回3.566秒に対して半減ではなく、従来n=3基準3.470秒→目標1.735秒も未達。

違反二乗Newton、単一ray縮小、等式成分の係数比による再スケールも試した。
rayでは元LP行4738等の現在制約が、他の増殖側制約と両立しない縮小係数を要求した。
GPU fallbackで4環境7時刻を全認証しても、全時刻でIPM再分解が必要だった。
これは単なる旧fluxの振幅調整では足りず、反応の組合せ変更が必要であることを示す。
行と代謝物の対応付け前に生物学的原因や共存不可能性を断定しない。

次は支配制約の入力生成元への逆追跡と、等式を保持する局所反応方向の抽出を優先する。
少数のGPU補正方向で元LP認証に届くかを4環境で検証し、成功してから32環境へ拡張する。
未認証の候補を通す／時間ステップを減らす／CPUを弱い設定へ戻す変更はしない。
PPO既定は維持。全候補の結果と次段階は
[反復半減試験報告](GPU_HOT_HALF_EXPERIMENTS_20260907.md)に記録した。

## Revision 36 — 更新制約をまとめて補正する（実装・検証中）

入力のみの逆追跡で、単一rayを阻んだ元LP行4738はOR16/NS21の
`EX_arg__L_e`（L-アルギニン）の共通供給制約と確認した。現在モデル3種の
fingerprintとtrace manifestの一致、および行の列index一致を照合した。
step2→3でlane0の供給上限は0.998554→0.000424369へ変化しており、
前時刻fluxの単一倍率変更では足りない。これは保存入力内の制約変化であって、
実培養における栄養不足の実証ではない。生物モデルの境界は変更しない。

1. 等式＋目的上限の数値的nullspace内で、現在違反する複数行をGPU上で選択する。
2. 選択行の小規模非負dual QPを解き、フラックス方向を同時補正する候補器を追加。
   既存の1行Motzkinとは区別する。現在/未来のCPU解を使用しない。
3. 元LPの全制約・目的・独立dual検査を維持し、4環境のstep2→3→4から試験。
   不合格なら32環境への拡大はしない。数値nullspaceは認証の代用にしない。
4. 合格した場合のみ32環境・hot6時刻・公平CPU16比較へ進む。SVD等の初回費用、
   入力更新、最終検査、失敗を含む時間を分けて保存する。目標は依然1.735秒以下。

この候補は実験用であり、PPOの既定backend・精度閾値は変更しない。

### Revision 36の結果

複数半空間投影は4環境step3で200反復後も最大違反0.0482、元LP不合格。
初回以外0.927秒だったが成功時間とは扱わない。供給・境界を1%に絞った補助reserveも
step2/3/4全て元GPU LPへの再求解が必要となり採用しない。
次にNewton方向上の全制約の区間を交差させ、終端可行候補が既に最適なら
元LP認証後だけ内部中心性の収束待ちを省くopt-inを実装した。
4環境の補助問題経由step2/3/4は元LP合格、factor17/10/8、hot0.383/0.284秒。
一部laneは候補認証で終了したが、バッチ全体の反復半減は示せていない。
追加機能は既定OFF。`pf_newton_ray_aux4_20260907.json`等に失敗を含む結果を保存。

## Revision 37 — GNN＋GRUを学習した主候補器へ育てる（次段階）

ユーザーの提案を受け、数値解法単独の調整から学習による補正需要の削減を優先する。
現状の`GraphTemporalLP`は全LPの二部グラフ＋nodeごとのGRU、`graph_ipm_bridge`は
diagnostic接続のみ。確認した`probe_graph_ipm`はランダム未学習重みの速度診断である。
別の`train_temporal_lp.py`は低rank latent GRU/MLPであり、GNN＋GRUの学習とは異なる。
過去のGNN＋GRU 4環境約3.6msは推論単体のコストであって、認証済み解の時間ではない。

1. 学習専用traceのmodel fingerprint、LPのstage、seedを監査し、現行モデルと一致する
   trajectoryだけを使用。既存の開発速度比較trace/holdoutを訓練へ混入させない。
2. 入力は現在のLP係数・上下限・目的＋前時刻情報。GNNが制約と変数の関係を処理し、
   GRUが環境・stage別の履歴を処理する。episode reset、失敗、モデル変更で履歴を管理。
3. primal fluxだけでなくdual、slack/制約活性を学習対象として検討。非一意LP解を
   単純MSEで平均しないよう、目的値・元制約残差・必要な交換flux/状態更新の損失を併用。
4. 認証済み学習軌道での教師あり予備学習→短いGPU補正を含む学習へ進む。
   教師の前時刻解を毎回与える評価だけでなく、自身の出力で進む因果rolloutを検証。
5. ランタイムは学習候補→元LP GPU検査→必要laneだけ少数GPU補正→再検査。
   未認証laneは広いGPU求解へ。CPU教師作成はオフラインとして費用を分離する。
6. 前解warm start／MLP／GNNのみ／GRUのみ／GNN＋GRUを同じ入力・精度で比較。
   指標は推論時間ではなく、入力更新＋推論＋補正＋検査の総時間、補正回数、認証率、
   未知軌道でのPHA等の誤差。GRUの必要性はablationで判断する。
7. 最初はstage別の短い学習・検証で補正回数削減を確認し、その後32環境hot6時刻の
   3.470→1.735秒以下を判定。さらに3段階LP・閉ループdFBA/PPOへ拡張する。

参考（いずれも本GEM系での半減保証ではない）：
- Faure et al. (2023), neural initialization＋mechanistic LP/QP:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10400647/
- Qian et al. (2024), LPのMPNN表現と近似求解:
  https://proceedings.mlr.press/v238/qian24a.html
- Gao et al. (2024), IPM-LSTM（培養時系列ではなくIPM内の求解学習）:
  https://papers.nips.cc/paper_files/paper/2024/hash/de0da9c42ee713f2ceaeed7bc40c522d-Abstract-Conference.html

現時点でこのGNN＋GRUの学習・速度認定を完了したとは扱わない。

## Revision 38 — 教師あり学習と実補正回数の比較を実施（進行中）

`pf_gru_training16x60_20260905`のtraining_referenceからmaxminを使用。
現行OR16/NS21/P. freudenreichiiのmodel fingerprint一致を再計算で確認。
12環境720 LPを学習、別4環境240 LPをモデル選択とし、正規化統計は720 LPのみ。
速度比較用traceとはseed・problem hashの重複を排除する。

1. full-space GNN/GNN＋GRUの教師ありtrainerを追加。固定GEM座標のnode embedding、
   train-only中心/scale、primal/dual教師損失と物質収支・目的の損失を使用。
   PCA辞書は使わない。GNNとGNN＋GRUを同じ予算で学習。GRUの状態は因果的に更新し、
   教師の前時刻fluxを入力しない。8時刻で勾配を切るTBPTTを使用する。
2. 初回60epochでは、開発用4環境step3/4のfactorが前解19、GNN35、GNN＋GRU46。
   全解はGPU補正後に元LP認証を通ったが、改善ではなかった。これを成功と扱わない。
   trainで変動ゼロのfluxを1e-3で割る正規化と、稀な制約違反がmean lossに埋もれる
   問題を確認。開発教師lossと物質収支lossの悪化を根拠に再学習方針を変更する。
3. 第2試験はtrain-only RMS（下限1）と、元単位の最大行違反を含むphysics loss。
   160epochでGNN/GNN＋GRUを再学習し、学習に使わない軌道で最終補正数を測る。
4. 学習推論→DLPack→厳密圧縮→current-cost restart→元GPU IPMの経路を追加。
   比較側も同じrestart mu=1e-5。learned restartを「認証済み前回解」と偽らない。
   解が既に正しければ元LP検査で0反復終了、違えばGPU補正。CPU fallbackなし。
5. 生の候補認証率、factor/solve回数、推論＋グラフ準備、更新＋求解＋独立検査の
   時間を保存。3反復の順序を入れ替え、失敗は速度向上に数えない。

これはmaxmin入力リプレイであり、3段階LP・閉ループdFBA/PPOの半減実証ではない。
既存PPOの既定backendは変更しない。学習成果物と失敗試験は上書きせず保持する。

### Revision 38の確定結果

第2学習160epochと独立seed4環境step2–8・順序ローテーション3反復を完了。
hot24 LP/試行の中央値は前解1.645秒、GNN3.350秒、GNN＋GRU3.721秒。
補正反復/LPは7.46→17.38/19.71と増えた。各方式hot72/72件が元LP認証、
CPU LP fallback 0だが、生の学習候補は0/72件しか直接認証されなかった。
これは高速化失敗であり、半減やPPO導入成功とは扱わない。既定は変更しない。

次の候補は、全fluxの再予測から「認証済み前回flux＋現在残差→修正方向」の学習へ変更し、
短いGPU補正後の誤差/反復数も学習・選択指標にすること。まだ実装済みとは主張しない。
詳細は[GNN＋GRU教師あり学習検証報告](GNN_GRU_SUPERVISED_CORRECTION_REPORT_20260907.md)。

## Revision 39 — 独立軌道の拡大と512超の学習曲線（進行中）

720 LP/12軌道はpilotであり、学習規模の十分性を確認していなかった。
32/64/128/256/512/1024軌道×120時刻×3段階の包含的収集を実装。
初期菌体量・窒素量と4種類のaction時系列を仮想範囲内で変え、学習・選択・testを分離する。
固定16環境を前提としないstreaming trainer、manifest/seed/モデル整合性チェック、
独立評価ブロックと学習seedのcluster bootstrap、512超の比較を要求する判定器を追加した。
512を十分な上限とは仮定しない。実測の補正回数と総時間が改善するかを先に確認する。

初回8軌道×120時刻は完了。次の8軌道はCPU並列求解中にnative memory errorで停止。
失敗データは保持・除外した。workers=1でも、モデルcold再構築でもnative crashが再発したため、
教師生成専用のSciPy/HiGHS cold経路を追加し、同じseedを再収集中。
これは既存highspy backendを経由しない回避策であり、根本原因を特定・修正したとは扱わない。
各試行のteacher strategyを記録する。複数最適解の選択が変わり得るため、教師fluxの一致は仮定しない。
全採用解には同じ元LP認証を課す。計測用のCPU baselineはこの収集経路へ変更しない。
小規模4軌道×8時刻のtrain/selectionでGNNとGNN＋GRU各1epochの動作を確認した。
これは性能・収束の検証ではない。既存PPO backendは変更しない。

詳細・再現コマンド・評価入力形式は
[学習曲線プロトコル](GRAPH_LEARNING_CURVE_PROTOCOL_20260907.md)を参照。

最終回帰：1702 passed、4 skipped（47.40秒）。記録は
`results/pf_graph_learning_curve_final_regression_20260907.xml`。
SciPy教師の元LP認証・保存読込、失敗shardの同一seed再試行、排他、
train/selection/test分離、512で自動打切りしない判定を含む。
このテスト合格を大規模学習・高速化の達成とは解釈しない。

## Revision 40 — 教師の精度を維持した再求解と32軌道ジョブの再投入

SciPy教師の111時刻目exchangeで相対KKT gap=2.1336494492511156e-7が発生し、
基準1e-7により停止した。物質収支だけでなく相補性の累積誤差が不合格要因だった。
受入基準（primal 1e-5、dual 1e-7、relative KKT gap 1e-7）は変更していない。

通常のdual simplex、許容誤差1e-10のdual simplex、presolve無効の同設定、
presolve無効のIPM（最適性許容誤差1e-12）の最大4回に限定して再求解する。
合格した時点で終了し、全試行不合格なら停止する。clipや基準の緩和で採用しない。
全試行の設定・残差・時間をLP別に保存する。既存のCPU速度baselineやPPO既定は変更しない。
設定の役割はSciPy公式の[HiGHSドキュメント](https://scipy.github.io/devdocs/reference/optimize.linprog-highs.html)を参照。

### 同一条件での検証

`scripts/verify_teacher_recovery.py`で、失敗traceの認証済み2656 LPだけを読込み、
各時刻の再構成LPのhash一致を確認した後、未保存の224 LPを新しい教師で計算した。
前回不合格の値2.1336494492511156e-7を再現し、追加1回の求解で
8.181944322860705e-8へ改善して合格した。120時刻まで8軌道全て継続できた。
新規224 LPの最大gapも同値、最大primal residual=1.3692680322918704e-9。
これは失敗修正の診断であり、独立の学習データ8軌道として二重計上しない。
診断ディレクトリのroleはdevelopment_diagnostic_not_trainingのまま保持。
記録：`results/pf_teacher_recovery_20260907/recovery_report.json`。

回帰1708 passed、4 skipped（48.22秒）。ジョブ上限・待機状態を含む更新後の
coordinator/retryテスト12件も合格。失敗は記録し、不合格教師の採用は拒否する。

### 投入したジョブの範囲

収集catalogのrequested_jobに上限32軌道、120時刻、training_queued=falseを明示。
既存完了8軌道は再利用、9–16軌道はretry4で再収集、17–24/25–32軌道は順次実行待ち。
32軌道・11520 LPで終了する。64軌道以降、モデル選択/test収集、GNN/GRU学習は投入しない。
既存失敗ディレクトリは上書きしない。未対応の新たな誤差で全再求解が不合格となれば、
今回もfail closedで停止する。過去のnative memory errorの根本原因を解決したとの主張ではない。

## Revision 41 — 学習用32軌道完了、モデル選択用16軌道へ

2026-09-07 17:29 JSTに32軌道×120時刻×3段階=11520 LPの収集完了を確認。
4つの完了manifestのSHA256をcatalogと照合し、32軌道全てcompleted_steps=120、
記録された元LP認証は11520/11520合格。最大relative KKT gap=8.994034056675459e-8。
失敗した過去の試行と診断用replayは件数に含めない。

次のジョブとして、同じcatalogへsplit=selection、16軌道×120時刻×3段階=5760 LPを投入。
学習seed系列と分離した72000000–72000015を使い、同じ精度基準・SciPy教師再求解処理で収集する。
8軌道ずつ2組を順次実行し、16軌道で終了する。これはモデル選択用であり、最終testではない。
GNN/GNN＋GRU学習はまだ未投入。selection完了後に分割整合性を確認して32軌道の学習へ進む。

## Revision 42 — ユーザー指定により512軌道の収集を先に予約

2026-09-07、selection16軌道の正常完了を依存条件として、train512軌道までの拡張を予約。
`scripts/queue_graph_collection.py`はcatalogの書込ロック解放をOS上で待機し、
selectionの完了状態、件数、manifest/hash/seed分離を確認した場合だけ次のcollectorを起動する。
現在のselection catalogを上書きせず、待機状態を別のjob.jsonに保存する。
依存ジョブ失敗時は次を起動しない。途中の精度不合格も既存のfail-closed方針を維持する。

待機記録：`results/pf_graph_train512_queue_20260907/job.json`。
既存32軌道は維持し、追加480軌道を8軌道ずつ収集。目標は512×120×3=184320 LP。
同じ包含的seed順序で32/64/128/256/512を切り出せるため、データ量比較に利用する。
selection16は全サイズに共通で使い、訓練へ混ぜない。学習・速度比較ジョブ自体は未投入。
1024軌道は今回予約しない。512の十分性を結論づける段階では、それを超える比較点が依然必要。

直近の8軌道収集は約797–838秒だった。追加480軌道の単純外挿は約13.3–14.0時間。
これは条件依存・再求解・I/O・マシン停止を含まない概算であり、完了保証ではない。
現時点の空き容量338 GiB、既存8軌道約270 MiBから、追加データは約16 GiBが目安。
新規依存ジョブ・既存coordinatorのテスト10件に合格。

## Revision 43 — 強制シャットダウン後の整合性検査と再開

2026-09-07 23:54 JST、旧収集プロセスの消失を確認。学習用192軌道とselection16軌道は完了、
193–200軌道の組は80/120時刻でmanifest保存されていた。
`scripts/audit_graph_collection.py`で完了shardのmanifest/seed構造と全74880 LP payloadの
SHA256一致を確認した（28.55秒、約6.90 GB）。新たな再求解での認証ではなく保存整合性検査。
記録：`results/pf_graph_shutdown_audit_20260907.json`。

既存collectorの--retry-failedでtrain512軌道まで再開。完了済み192軌道はスキップし、
中断した8軌道のみ同一seedで120時刻を最初から再計算する。
元ディレクトリは保持し、`train_000192_000200_retry1`に新規出力する。
80時刻目からの状態チェックポイント再開ではない。旧queueの状態はinterruptedとし、
再開後の進捗の参照先をcatalog.json requested_jobへ明記した。
精度基準、教師再求解方針、学習/testを自動投入しない範囲は変更していない。

## Revision 44 — 2026-09-08 再中断後の検査と512軌道収集再開

2026-09-08 00:34 JSTに収集を再開。再開前に既存collectorが動いていないことを確認し、
学習用208軌道・selection16軌道のmanifest/role/seed整合性と全80640 LP payloadの
SHA256を検査した。7428045947 bytesを34.69秒で確認し、全件合格。
新規の求解・数値認証ではなく保存整合性検査である。
検査記録：`results/pf_graph_resume_audit_20260908.json`。

既存208軌道は再利用。中断した209–216軌道は同じseedを使い、
`train_000208_000216_retry1`へ120時刻を最初から再収集する。
既存の中断データは保存。`--retry-failed --scipy-teacher --workers 1`で
512軌道まで収集し、学習・速度比較は自動投入しない。
起動PIDは3128、起動記録とログは`results/pf_graph_resume_20260908/`。
進捗の正本は`results/pf_graph_coverage_20260907/catalog.json`のrequested_job。

残り304軌道=38組。直近10完了組は786.96–874.34秒/組、平均807.83秒/組。
00:34再開からの単純外挿で終了は2026-09-08 08:52–09:48 JST、平均では09:06頃。
停止・追加再求解・負荷変動を含まない実績ベースの概算で、統計的信頼区間ではない。
