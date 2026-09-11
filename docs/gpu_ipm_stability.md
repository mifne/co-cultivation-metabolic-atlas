# GPU内点法の数値安定性とCPU超えへ向けた実装：統合ドキュメント

**要約**: 本ドキュメントは、GPU加速dFBA/LPソルバー開発における内点法(IPM)の数値安定性修正、等式簡約・予測子–修正子(PC)・初期平衡化などの実装、およびCPU速度比較の結果を統合したものである。2026-09-06時点で、maxmin段階の固定LPではGPU認証4/4・32/32を達成したが、aggregate/exchange段階の認証は未達であり、CPU速度超え・完全GPU内完結・閉ループdFBA/PPO接続はいずれも未達である。

---

## 1. 背景と実行経路の構造

### 1.1 現在の二つの実行経路 (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)

1. **培養/PPO経路**: 方策action → Python環境step → 流加と状態依存の境界設定 → maxmin → 保持率を課したaggregate → exchange parsimony → フラックスから培地・菌体量・PHAを更新 → 観測/報酬。
2. **新GPU数値層の開発経路**: 保存したSHA確認済みLP入力 → 等式forest簡約 → ゼロ固定/重複等式簡約 → 境界dualを消去した疎Newton系 → cuDSS GPU分解＋必要時Krylov → GPU主双対復元 → 最初の元LPを認証。

新しい疎内点法は、現時点では2の限定診断であり、1の既定PPO経路へ自動接続していない。過去のGPU surrogateによる時間を、新しい厳密解法の時間として使わない。

### 1.2 培養・PPO経路の依存関係 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

```text
action → 流加・酸素移動・状態依存境界
       → maxmin → 成長保持下限を設定
       → aggregate → 目的保持下限を設定
       → exchange parsimony
       → 菌体・培地・PHA・pH更新 → observation / reward → 次step
```

同一環境の段階間・時刻間は独立ではない。GPUバッチ化の基本単位は独立環境の同じ段階であり、1環境の将来stepを先に並列計算する方式ではない。

### 1.3 変更の境界 (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)

- 厳密行関係の検査はhostでのsetupであり、CPU LP fallbackではない。GPU数値層は元の正解x/yを入力しない。
- 元のLPを保持し、GPUで行dualを復元して全行を再検査する。入力の係数を丸めたり、近い行という理由だけで削除しない。
- 初期dualが削除行に乗っている場合も、検証済み関係で残存行へ集約し、warm startの双対情報を捨てない。
- 新方式は明示オプション。失敗・未認証を成功や短い実行時間として扱わない。
- 原LPの主残差1e-5、dual違反1e-7、相補性指標1e-7は維持。PPO学習用の別基準と混同しない。

---

## 2. 数値安定性の修正と検証 (2026-09-06)

### 2.1 結論 (出典: GPU_IPM_STABILITY_RESULTS_20260906.md)

不良なNewton方向を使ってcorrectorを作っていた問題、改善しない反復改良を採用する問題、簡約後にwarm-startの重複等式dualを捨てる問題を修正した。境界dualの代数消去、証明付きゼロ固定・重複等式除去、既存の等式森林簡約、GPU FGMRESによる元Newton系への補正も、元LP認証を維持して実装した。

**新しい実験用内点法は実GEMでまだ収束していない。** 最終構成の開発用step1・全3段階×4環境は0/12、step2のexchangeは0/4認証だった。誤解を返さず停止することと、実用上の安定収束・高速化は別である。CPU速度超え・完全GPU内完結・120物理stepの完走・PHA終点1%を達成したとは主張しない。本番PPO、GEM、培地境界、目的関数、精度閾値は変更していない。

### 2.2 検証条件と「反復」の意味 (出典: GPU_IPM_STABILITY_RESULTS_20260906.md)

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU。FP64 CuPy/cuDSS 0.7、既存`.venv-cuopt-26.8`を使用。
- 入力: `results/pf_coverage_holdout4x120_20260905`のSHA検証済みLP入力のみ。manifest SHA256: `af7b6b92ca27487590c67158294c390072bd89e1c84d0a0aebccdcec64e532bc`。
- 開発用診断であり、学習・最終holdoutの成績ではない。NPZ内のCPU正解x/yは読まず、CPU LPを呼ばない。独立したhost認証は残差計算のみで、最適化ではない。
- ここでの100や600は、**固定された1個のLPを解く内点法の反復上限**。培養の時間を進める120物理stepとは異なる。step1/step2も保存入力を別々に評価しており、新しいGPU閉ループ軌跡ではない。
- 元LPの主残差≤1e-5、dual violation≤1e-7、relative KKT gap≤1e-7を維持。Newton方向のguardは別で、主・双対・相補性ブロックごとの `||residual||∞ / max(1, ||rhs||∞)` の最大値≤1e-8。
- 数値分解・反復・主双対復元はGPU。疎構造の準備、cuDSSの並べ替え、Python制御、コンパクトな認証結果の取得はhostに残る。完全GPU常駐と呼ばない。

### 2.3 見つかった問題と対応 (出典: GPU_IPM_STABILITY_RESULTS_20260906.md)

#### 2.3.1 affine方向の不良をcorrectorの計算前に止める

失敗直前の保存診断では、1環境のcorrector右辺が約2.18e37、GPU線形残差が約1.02e55になった。affine方向が不良でも、検査前に`ds_aff * dz_aff`を計算していたことが増幅経路になっていた。

同じ保存線形方程式のCPU sparse LU診断では、この環境の絶対残差は約4.65e24と大きいが、右辺に対する相対値は約2e-13だった。絶対値だけで「CPUも同様に失敗」とは判断しない。このCPU診断は16回の線形solveであり、LP fallbackにもGPUの採用方向にも使用していない。

対応はaffineの先行検査、有限性検査、ブロック別残差、環境ごとの最良反復保持。不良なら培養状態を進めない。保存診断: [Newton不安定性の切り分け](../results/pf_ipm_instability_audit_20260906.json)。

#### 2.3.2 強い正則化と元LPへの収束は同じではない

Newton行列を安定化する正則化を1e-6へ強めると、簡約なしでも100反復まで進んだが元LP認証は0/4だった。ゼロ固定・重複除去後も100反復で主残差約4〜5e-4が残った。行列スケーリングの単独追加も、この試験では改善を示さなかった。これらの方式・設定は実験用のままであり、既定の本番経路へ昇格していない。

正則化系K_deltaの残差が小さくても、元のK_0には正則化項に由来する誤差が残る。したがってK_deltaを前処理器にし、K_0に対してGPU FGMRESで残差を減らす経路を追加した。小LPでは成功したが、実GEMでは32 Krylov反復の上限でも一部環境の元方向精度に届かない。

FGMRESは実際の前処理解を保存するflexible型、二度の再直交化、Givens回転、真の残差による採否、環境別の停止理由を用いる。ブロック行スケーリング後に改善しても、最後に**非スケールの元K_0**で改善しない方向は採用しない。参考: [Saad, 1993](https://epubs.siam.org/doi/10.1137/0914028)、[SIAM/Netlib Templates](https://www.netlib.org/templates/templates.html)。

#### 2.3.3 境界面と等式構造を使って行列を小さくする

step1の元LPでは、上下限だけでは可動でも、ゼロrhs等式と境界の符号から190変数を厳密にゼロと証明できた。明示的固定11変数もすべて値0だった。ゼロ行177本と完全重複／符号反転等式82本も除去した。許容誤差で小さい係数を消す処理ではない。

追加した最終構成は、既存の2変数同次等式の森林簡約を先に適用する。境界の共通部分、目的関数、残る制約を代数的に変換し、最後に両方の簡約をGPUで逆順に復元する。森林のbound normalは対応する元の境界へ割り当て、ゼロ固定のdualは保存した証明を逆順に使って復元する。認証対象は常に最初の全次元LP。

exchange・step1・4環境に共通する寸法は次のとおり。LPの行数と、内点法が分解するNewton行列の次数は異なる。

| LP表現 | 制約行数 | 変数数 |
|---|---:|---:|
| 元LP | 6,330 | 7,373 |
| 等式森林の適用後 | 3,919 | 4,962 |
| さらにゼロ固定・重複除去後 | 3,633 | 4,741 |

最終exchangeの分解行列は8,374次、54,632非零要素。小さくなったこと自体は元LPの収束やCPU超えを意味しない。

### 2.4 最終構成の実GEM検証 (出典: GPU_IPM_STABILITY_RESULTS_20260906.md)

森林＋zero-face＋境界dual消去、正則化1e-8、元Newton系へのFGMRES上限32、通常の反復改良上限1、内点法上限100。すべて4独立環境を同時処理。表の時間はその**バッチが未認証で停止するまでの同期wall**であり、有効なLP解の計算時間や学習時間ではない。setupは別記し、単発測定に信頼区間は付けない。

| 物理step入力 | LP段階 | 完了した内点反復 | setup秒 | solve秒 | 元LP認証 |
|---|---|---:|---:|---:|---:|
| 1 | maxmin | 20 | 1.444 | 2.782 | 0/4 |
| 1 | aggregate | 21 | 1.452 | 2.858 | 0/4 |
| 1 | exchange | 27 | 1.367 | 4.233 | 0/4 |
| 2 | exchange | 27 | 1.528 | 4.595 | 0/4 |

記録: [maxmin](../results/pf_ipm_forest_fgmres32_maxmin4_20260906.json)、[aggregate](../results/pf_ipm_forest_fgmres32_aggregate4_20260906.json)、[exchange](../results/pf_ipm_forest_fgmres32_exchange4_20260906.json)、[step2 exchange](../results/pf_ipm_forest_fgmres32_exchange_step2_4_20260906.json)。各JSONは設定・入力ID・ソースSHAとスナップショット・各反復の認証値・Krylov停止理由を保存する。過去の失敗記録を上書きしていない。

step1 exchangeでは、数値分解0.160秒に対し、三角solve・Krylov補正・更新・同期を含む区間が3.939秒だった。この区間にはPython制御も含まれるため、すべてをGPUカーネル時間とは解釈しない。分解行列のサイズだけでなく、繰り返す補正とその同期が次の調査対象である。現在のデータから「GPUだけが律速」「新GPUでこの倍率になる」とは言えない。

### 2.5 テストと適用範囲 (出典: GPU_IPM_STABILITY_RESULTS_20260906.md)

数値分解・IPM・簡約・GPU復元・FGMRES・元LP認証まわりの最終対象テストは**114 passed**。小LP、退化・特異な線形系、環境ごとの異なる写像、負の写像係数、重複dual warm-start、非有限値・誤形状・異なる実行context、破損した復元の拒否を含む。リポジトリ全テストの合格とは区別する。

別プロセスで、再利用した等式森林・縮約PDHGとGNN／GRUのモデル・データ・IPM接続も**98 passed**を確認した。重複しない2組の合計は**212 passed**。これらのテスト合格を実GEMの元LP認証成功とは扱わない。

本番GEM／学習済みモデル／辞書は変更していない。これらは明示的に選ぶ実験用ソルバと診断スクリプトであり、CPU fallbackの件数を隠して速度を示す実装ではない。

### 2.6 次の限定診断 (出典: GPU_IPM_STABILITY_RESULTS_20260906.md)

1. 失敗した1つのNewton系を固定し、全変数系FGMRESと境界dualを消去した縮約系FGMRESを同じ反復上限で比較する。毎回の境界dual復元に由来する丸め誤差と、残る等式従属・前処理の限界を切り分ける。
2. どちらも最終的な全K_0残差と元LP認証を使う。内部の小さい残差だけを合格としない。正則化やKrylov上限の無制限な増加は行わない。
3. 実LPでの安定収束を確認してから、前回解／学習したGNN・GRUによる初期化が補正回数を減らすかを評価する。最後に同一の精度条件でCPUとの閉ループ総時間を再測定する。

計画の経緯は [Revision 11–13](DATAFLOW_ACCELERATION_PLAN_20260905.md)。正則化と内点法設計は [Zanetti & Gondzio, 2025](https://arxiv.org/abs/2508.04370) 等を参考にしたが、そのアルゴリズム全体を再現したという主張ではない。

---

## 3. CPU超えへ向けた実装計画と実測結果 (2026-09-06)

### 3.1 律速と優先順位 (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)

| 問題 | 確認できた根拠 | 今回の実装 |
|---|---|---|
| 等式がなお独立でない | exchangeの等式1,955本に対し構造ランク1,940、数値QR1,929 | QRは候補抽出に限定し、binary64として保存された係数・RHSを有理数として検証。厳密に従属すると証明した行だけを簡約 |
| 初期からの反復が多い | 中心化方式は相補性mu<1に約48反復。maxmin認証にも61〜64反復 | 同じ分解をaffine予測とcorrectorで共有。各方向の元非線形残差を検査し、不良時のみ固定中心化へ戻る |
| 補正の増加が遅くなる | exchangeのretry追加で5.251→8.383秒、solve397→666回 | retry/内部反復の総当たりを避け、適格な解に達する反復数・全solve数を比較 |
| setupが高価 | GPU従来setup約1.6〜2秒。QRも毎回行えば更に遅くなる | 等式係数・RHSのfingerprintが完全一致する場合だけ検証済み構造を再利用。変化時は再検証 |
| 実LPとPPOの評価を混同しやすい | 固定LP成功はdFBA軌跡・報酬の成功ではない | まず段階別元LP認証、次に同一actionの閉ループ、最後に方策評価へ進む |

### 3.2 検証順序と計画変更条件 (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)

1. 小LPで、厳密関係・微小な非従属係数・RHS不整合・不変性・主双対復元・初期dualを試験。
2. 開発用step1の3段階×4環境に、まず厳密行簡約だけを適用し、何行が証明できたか/残ったかと元LP認証を計測。
3. 同じ入力でpredictor–correctorをA/B比較。内部solve・fallback・因子分解・認証・setupを含む費用を記録。
4. 失敗する場合は失敗残差を分析して構成を変更。CPU超えに遠い構成で長い検証を続けない。
5. 適格な構成だけ、異なるphysical stepの入力へ拡張。同一入力のCPU HiGHS 1 worker/4 workersを別々に比較。
6. 閉ループへ接続する場合は未認証解を採用せず、同一actionで4環境×8stepから120stepへ進める。GPU内の同期・キャッシュ・異種レイアウトのバッチ化はこの段階で評価。

単一LP・固定入力repeat・独立LPバッチ・閉ループdFBA・PPO学習を明確に区別する。GPUがCPU1 workerのみを上回っても、CPU並列を上回ったとは書かない。

### 3.3 実測後の改訂（同日） (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)

厳密関係25本の簡約だけではexchange4環境は9.659秒、元LP認証0/4。
さらにpredictor–correctorを使うと13.103秒、認証0/4だった。
後者では数値分解0.477秒に対し、三角solve・補正・更新の区間が12.312秒。
したがってGPU分解サイズやVRAM容量だけでなく、反復補正を必要とする数値条件と制御費用が主要課題である。

PCの264 active environment-iterations中、34回が中心化へ戻り、そのうち27回は
affine予測点の安全性判定で止まっていた。最大境界ステップの丸めによる微小負値が
原因かを診断し、予測点だけを境界の99.5%へ置く明示オプションを比較する。
実際の採用ステップ、元LP認証、モデル係数は変更しない。PCを既定へ昇格しない。

厳密な行簡約後も残った1行は約9.7e-17の係数差があるため保持した。
これを等価な簡約と呼んで削除しない。仮に作業緩和を検討する場合も、
有限境界による誤差上界と元LPの全行・双対の最終認証を別々に扱う必要がある。

### 3.4 最終改訂：採用候補と保留する変更 (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)

各変更を別々に測定した。縮約座標Krylovは、全非線形targetノルムを分母として保持し、
最後にbound方向を復元して全成分forcingを検査する実装へ接続できた。しかしexchangeでは
10.123秒・認証0/4で、単独の速度改善は確認できなかった。

balanced初期化は、内部dualのfloorを1から1/sへ変える明示オプションである。
大きい無活性境界による初期muを637.61から1へ下げた。モデルの境界値は不変。
PCの予測点を99.5%境界へ置く設定と組み合わせると、maxmin4/4認証・1.633秒、
aggregate1/4認証・12.659秒、exchange0/4・12.359秒だった。
従来の認証済みGPU maxmin5.729秒より改善したが、同一4入力CPU1 worker0.538秒より遅い。

正則化schedule `min(base, max(1e-12, 0.01*max_active_mu))` も限定実装した。
全成分forcingと元LP認証を変えず、初期はbaseを保持し、終盤に弱める。
この係数と下限は今回の実験設定で、文献の普遍定数や論文アルゴリズムの再現ではない。
balanced centered exchangeでは固定正則化より早く失敗し、残差も悪化したため既定化しない。
数値分解失敗・非有限muは停止し、失敗を高速化として数えない。
動的正則化の一般的根拠は[Pougkakiotis & Gondzio (2019)](https://arxiv.org/abs/1902.04834)、
分解・三角solve費用の区別は[Zanetti & Gondzio (2025)](https://arxiv.org/html/2508.04370v1)を参照した。
今回の対角scheduleは、これらの非対角正則化やpivot制御の実装そのものではない。

別の開発用32環境入力でもmaxmin32/32を認証できた。しかしGPU5.438秒に対し、
同一入力CPUの3反復中央値は1 worker4.135秒、4 workers1.223秒だった。
GPU setup7.968秒は上記solve時間に含まれず、cold全費用ではさらに不利である。
ここから単純な環境増量やGPU購入で全体の速度問題を解消すると推測しない。

次の作業順を以下に更新する。

1. 後段で最初に失敗するNewton方向を固定し、主・等式・不等式・相補性の実残差と
   regularization由来の誤差を分ける。既に停止したlaneの再計算値を原因として使わない。
   near-dependent行、自由方向の前処理、全残差の尺度を検査し、δ総当たりは再開しない。
2. 認証済みmaxminを限定的な速度改善の基準に使い、Krylovの直交化・小演算・同期を
   分解して測る。分解だけの高速化ではなく、反復全体のGPU起動/CPU制御費用を減らす。
3. 後段の短い開発入力で元LPを認証できたら、同一モデル構造の数値更新APIと
   proof/symbolic cacheを独立環境×同一段階のバッチ受付へ接続する。
   3つのLP段階と時間ステップの因果依存は維持する。
4. 最後に同一actionの短い閉ループCPU対照へ進む。精度・速度の両方が確認できた構成だけ
   120物理stepとPPO学習へ広げる。現時点では本番PPOへの自動昇格・再学習はしない。

元LPの認証閾値、生物モデル、学習データは不変。CPU超えと完全GPU内完結は未達。
実測の詳細は[今回の結果](CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)に分離した。

### 3.5 バッチ並列の監査後の追補 (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)

前記の未達状況の後、GPU内部refinementと外側補正の重複を削減し、32環境maxminの
solve中央値3.572876秒を確認した。CPU1 worker4.220701秒より短いが、
CPU4 workers1.239123秒、setup込み、閉ループPPOにはまだ勝っていない。
実行は既に32 LPのuniform GPU batchであり、GPU未並列という診断は誤り。

主費用67.2%の補正/更新を対象に同期集約と常駐bufferを優先し、
同一総問題数でバッチ分割・stream配置を比較する。単なる環境増量やGPU使用率の上昇を
成功条件にしない。ゼロ面処理後の新しい2項等式を再縮約する仕組みも追加したが、
実GEM後段はなお未認証なので本番への自動切替はしない。
現行の結果・次順は[バッチ速度監査](GPU_BATCH_SPEED_RESULTS_20260906.md)と
DATAFLOW計画Revision22を参照する。

---

## 4. CPU超えへ向けたGPU数値層の実装・比較結果 (2026-09-06)

### 4.1 結論 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

厳密な等式簡約、安全策付きpredictor–corrector、初期相補性の平衡化を実装した。maxmin固定入力4環境は元LP認証4/4、GPU solve API時間1.633秒となり、旧GPU構成の5.729秒から短縮した。ただし**3.51倍は単回試験同士の暫定的なGPU内比較**であり、CPUに対する速度向上ではない。

同じ構成のaggregateは1/4、exchangeは0/4で、全3段階の認証は未達である。32独立環境のmaxmin固定入力は32/32で認証できたが、GPU 5.438秒に対してCPU 1 workerは4.135秒、4 workersは1.223秒だった。**CPU超え、全3段階の完全GPU実行、閉ループdFBA/PPOへの接続・精度確認はいずれも未達**である。

### 4.2 実装とその検証結果 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

#### 4.2.1 厳密な等式関係と再利用

exchangeの縮約途中の等式は1,955行で、数値QRは26行を従属候補に挙げた。保存されたbinary64係数・RHSを`Fraction.from_float`で正確な有理数として検査し、25行だけを厳密に簡約した。等式数は1,955→1,930となる。残った1行は約9.7e-17の係数差があり、近いという理由で削除していない。QR閾値だけで削除した行数は0で、完全なランク決定を主張していない。

exchangeの1環境当たりの形状は以下のとおりである。境界dualの消去はNewton線形系の代数操作であり、LPの境界を削除するものではない。

| 段階 | LP変数数 | LP行数 |
|---|---:|---:|
| 元exchange LP | 7,373 | 6,330 |
| forest簡約後 | 4,962 | 3,919 |
| zero-face等の簡約後 | 4,741 | 3,633 |
| 厳密等式25行の簡約後 | 4,741 | 3,608 |

全Newton座標17,192に対して、実際の分解次元は8,349である。maxmin/aggregateでは全Newton座標14,635、分解次元6,431となった。次元の減少率を、そのまま時間短縮率やメモリ削減率とは扱わない。

等式行列とRHSのfingerprintが一致する場合だけ、同一バッチ内で検証済み証明を再利用する。最初のexchange centered試験では、最初の証明構築2.384秒に対し、後続3環境の再利用は各0.00397–0.00419秒だった。これは構造準備部分の費用であり、物理step間の常駐更新まで実装済みという意味ではない。主双対復元・元LP再認証は省略しない。

詳細: [厳密簡約＋centered記録](/home/reiya/co-cultivation/results/pf_ipm_exact25_centered_exchange4_20260906.json)。

#### 4.2.2 predictor–correctorとaffine予測点

同じ数値分解をaffine予測とcorrectorで共有し、各方向の元非線形残差、正値性、実際のmerit下降を確認する。失敗時は同じ状態から固定中心化方向へ1回だけ戻る。

旧affine予測は最大境界ステップを使用し、exchangeでは264 active environment-iterations中34回が中心化へ戻った。このうち27回は予測点の安全性判定によるものだった。予測用ステップだけを最大値の99.5%へ置く明示オプションを追加したところ、259 active environment-iterations中のfallbackは7回となり、同じ安全性理由による棄却は0回になった。予測点を黙ってclipする変更ではなく、最終採用ステップと元LP認証は維持している。

ただし、unsafe affineの解消だけではexchangeを認証できなかった。balanced＋PC995でもfallbackは22回残り、主な内訳は非下降方向18回、後退上限1回、affine方向精度不足3回だった。数値安全性の改善と最適性の達成は分けて評価する。

詳細: [PC](/home/reiya/co-cultivation/results/pf_ipm_exact25_pc_exchange4_20260906.json)、[PC995](/home/reiya/co-cultivation/results/pf_ipm_exact25_pc995_exchange4_20260906.json)、[balanced＋PC995](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_exchange4_20260906.json)。

#### 4.2.3 初期相補性の平衡化

旧初期化は`z >= 1`、`s = max(1, h-Gx)`であり、大きい無活性境界のslackが初期相補性を増大させていた。`balanced`はslackを変えず、正のdual floorだけを`1/s`へ変更する。既存dual・reduced costがfloorを上回る場合は維持するため、一般のwarm startで必ず全要素`s*z=1`になるわけではない。既定は`legacy`のままである。

exchange step1の初期記録では、slack最大値は1e6、初期μは637.6066→1.0となった。元入力1件のCPU配列集計でも、相補性の範囲は1–1e6→1–1と確認した。一方、境界項`D = Σ(z/s)`の最小値は0.002→2e-6、初期stationarityのL2ノルムは51.41→73.50となる。全残差や条件数が必ず改善する操作ではなく、正のinfeasible-start候補として比較した。

centered同士では最終相補性指標が約59–64分の1まで低下したが、認証0/4である。初期化による精度改善を、認証成功やCPU超えと読み替えない。

詳細: [balanced centered](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_centered_exchange4_20260906.json)。

#### 4.2.4 縮約Krylovとbarrier連動正則化

全Newton残差の判定を保ちながら、Krylovの作業座標を縮約するオプションを比較した。しかし今回のexchangeでは、centeredの9.659秒・783 solveに対し10.123秒・816 solveとなり、認証はともに0/4だった。縮約幅だけでは改善を判断できない。

balanced centeredでは、最初に失敗した各環境のμが2.35–2.48e-7まで低下し、選択δは1e-7、forcingは0.1034–0.1087で基準0.1を超えた。毎反復のbase δ=1e-6と1回retryだけでは終盤の尺度に合わない可能性を検討し、`δ = min(base, max(1e-12, 0.01 * max(active_mu)))`という明示的な実験scheduleを追加した。

このscheduleは初期から小δを固定する方式とは異なるが、実測では最終dual違反が3.54e-7–1.33e-6へ悪化し、相補性指標も4.20e-4–1.30e-3となった。5.141秒で停止したことは成功解への高速化ではない。この設定を改善済みの既定方式として採用しない。barrier連動の考え方は文献を参考にしているが、上の係数・floorは限定試験用heuristicであり、論文の正則化法そのものを再実装したとは主張しない。

詳細: [縮約Krylov](/home/reiya/co-cultivation/results/pf_ipm_exact25_centered_condensed_exchange4_20260906.json)、[barrier schedule](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_barrier_centered_exchange4_20260906.json)。

### 4.3 同じexchange固定入力のA/B比較 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

全行で、physical step1、4環境、同じ4つの元LPハッシュ、forest＋厳密等式25行簡約、FP64、forcing=0.1、最大240反復、Krylov上限16、retry1を使用した。既定で全座標Krylov、固定δ=1e-6とし、表に記した項目を変更した。保存されたCPU正解x/yは使用していない。各GPU条件は1回で、統計的有意差は評価していない。

| 条件 | setup秒 | solve API秒 | outer反復 | factor / solve回数 | 最終相補性指標の環境間範囲 | 元LP認証 |
|---|---:|---:|---:|---:|---:|---:|
| legacy＋centered | 4.552 | 9.659 | 118 | 134 / 783 | 5.023e-3–5.788e-3 | 0/4 |
| legacy＋PC | 4.112 | 13.103 | 71 | 92 / 1,182 | 7.169e-3–1.394e-2 | 0/4 |
| legacy＋PC995 | 4.583 | 12.877 | 71 | 86 / 1,123 | 6.843e-3–1.122e-2 | 0/4 |
| legacy＋centered＋縮約Krylov | 4.028 | 10.123 | 119 | 136 / 816 | 3.715e-3–5.775e-3 | 0/4 |
| balanced＋centered | 4.239 | 10.193 | 166 | 180 / 812 | 8.503e-5–9.554e-5 | 0/4 |
| balanced＋PC995 | 3.829 | 12.359 | 70 | 85 / 1,092 | 1.896e-5–3.626e-5 | 0/4 |
| balanced＋centered＋barrier schedule | 3.762 | 5.141 | 144 | 148 / 336 | 4.199e-4–1.297e-3 | 0/4 |

全行が未認証停止であり、表の時間からspeedupを定義しない。factor/solveはバッチ数値処理の呼出回数で、CPU LPの実行回数や成功LP件数ではない。

balanced＋PC995の主残差は最大9.900e-7、dual違反は0まで達したが、相補性は基準1e-7を満たさない。12.359秒の内訳は数値分解0.442秒、三角solve・Krylov・更新11.595秒、認証0.210秒である。反復数を減らしても、多数の内部solveが残れば短時間化には直結しない。

### 4.4 balanced＋PC995の全3段階 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

同じ構成を段階ごとの固定入力に適用した結果を示す。各段階は別の保存LPから開始しており、先行段階の新GPU解から次段階を組み立てた閉ループ実験ではない。

| 段階 | setup秒 | solve API秒 | outer反復 | 元LP認証 | 主残差最大 | dual違反最大 | 相補性指標範囲 |
|---|---:|---:|---:|---:|---:|---:|---:|
| maxmin | 3.581 | 1.633 | 33 | 4/4 | 3.000e-7 | 0 | 3.723e-8–7.548e-8 |
| aggregate | 3.453 | 12.659 | 68 | 1/4 | 1.513e-7 | 0 | 4.837e-8–4.907e-5 |
| exchange | 3.829 | 12.359 | 70 | 0/4 | 9.900e-7 | 0 | 1.896e-5–3.626e-5 |

maxminは30–33反復で認証した。解析的box-bound dualを明示的に許可し、返却する元row dualを実際に構成して再認証している。反復が生成したrow dual自体の相補性指標は0.00126–0.00668であり、これを合格と見なしたわけではない。独立host検査も4/4で合格した。

aggregateは環境0のみ68反復で認証し、主残差1.455e-10、相補性4.837e-8だった。残る3環境は方向精度不足で停止した。exchangeは全環境が同じく方向精度不足で停止した。バッチの一部成功を全段階の成功として扱わない。

初期化を変える前の厳密簡約＋PCではmaxminも0/4、6.790秒で停止していた。この記録を省かず、PC単独が一様に改善するわけではないことを残す。

データ: [maxmin](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_maxmin4_20260906.json)、[aggregate](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_aggregate4_20260906.json)、[exchange](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_exchange4_20260906.json)、[初期化変更前maxmin](/home/reiya/co-cultivation/results/pf_ipm_exact25_pc_maxmin4_20260906.json)。

### 4.5 CPU基準と32環境比較 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

#### 4.5.1 4環境の既存CPU基準

同じ元LPハッシュのCPU記録を参照する。CPUは各worker条件につき3回の中央値、GPUは各1回である。

| 固定LP | 構成 | solve時間秒 | 認証 |
|---|---|---:|---:|
| maxmin、4環境 | CPU HiGHS、1 worker | 0.5381 | 12/12 |
| maxmin、4環境 | CPU HiGHS、4 workers | 0.1522 | 12/12 |
| maxmin、4環境 | 旧GPU centered＋retry1＋解析的dual | 5.7293 | 4/4 |
| maxmin、4環境 | 新GPU balanced＋PC995＋厳密簡約＋解析的dual | 1.6332 | 4/4 |
| exchange、4環境 | CPU HiGHS、1 worker | 1.1604 | 12/12 |
| exchange、4環境 | CPU HiGHS、4 workers | 0.3440 | 12/12 |

旧GPU maxminの採用は61–64反復だった。新GPUとのsolve API比5.7293/1.6332≈3.51は、複数変更を含む暫定的なGPU内比較に限定する。旧setupは1.676秒、新setupは3.581秒であり、setup込みの比は3.51ではない。新GPUはsolve API時間だけを見てもCPUの両worker条件より遅い。

データ: [CPU maxmin4](/home/reiya/co-cultivation/results/pf_ipm_cpu_maxmin4_baseline_20260906.json)、[CPU exchange4](/home/reiya/co-cultivation/results/pf_ipm_cpu_exchange4_baseline_20260906.json)、[旧GPU maxmin4](/home/reiya/co-cultivation/results/pf_ipm_globalized_retry1_box_maxmin4_20260906.json)。

#### 4.5.2 32独立環境のmaxmin

別の32独立環境のphysical step1について、同じ保存入力をGPUとCPUで解いた。GPUは1回、CPUは1/4 workersそれぞれ3回である。同じLPの32複製ではないが、4環境試験とは環境集合が異なるため、両表の比だけで厳密なスケーリング則を推定しない。

| 構成 | solve時間秒 | GPU setup秒 | 認証 |
|---|---:|---:|---:|
| CPU HiGHS、1 worker、3回中央値 | 4.1350 | — | 96/96 |
| CPU HiGHS、4 workers、3回中央値 | 1.2234 | — | 96/96 |
| GPU balanced＋PC995、1回 | 5.4380 | 7.9682 | 32/32 |

GPUの採用反復は31–38、主残差5.487e-8–3.756e-7、dual違反0、相補性1.050e-8–9.925e-8だった。返却主双対の独立host検査も32/32で合格した。GPU数値層のCPU LP callsは0、CPU対照は計192回の実solver実行でretryは0だった。

GPU solve APIの内訳は以下のとおりである。百分率の分母は5.4380秒で、setupは含まない。

| 区間 | 秒 | solve APIに占める割合 |
|---|---:|---:|
| 数値分解 | 0.9469 | 17.41% |
| 三角solve・Krylov・更新 | 4.2570 | 78.28% |
| 元LP認証 | 0.1366 | 2.51% |
| その他の初期化・残差・制御等 | 約0.0976 | 約1.80% |

outer反復38、数値分解38回、solve194回である。主要費用は依然として三角solve・補正・更新区間で、GPUの容量や分解サイズだけが支配的とは言えない。この区間は複数処理をまとめた時間であり、三角solveだけの費用を分離した結果ではない。

GPUはCPU 1 workerの約1.32倍、4 workersの約4.45倍の時間を要した。setup、solve、主双対転送、独立host監査、closeを含む記録済みGPU lifecycleは13.4212秒である。32環境で認証できたことはバッチ解法の適用範囲拡大だが、CPU速度優位の達成ではない。

データ: [GPU maxmin32](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_maxmin32_20260906.json)、[CPU maxmin32](/home/reiya/co-cultivation/results/pf_ipm_cpu_maxmin32_baseline_20260906.json)。

### 4.6 計測範囲と次の実装判断 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

- CPU時間はfresh modelの`solve_batch_wall_seconds`であり、入力検査、モデル作成、HiGHS実行、backendの元LP認証を含む。backend constructor、独立追加認証、closeは別記録。cold process・OS cache除去を主張していない。
- GPUの`ipm.total_seconds`は数値反復、postsolve、元LP認証、診断取得を含む。コンストラクター、元主双対のD2H、独立host監査は別記録。`solver_lifecycle_wall_seconds`も入力ロードやPythonプロセス起動すべてを含む指標ではない。
- CPUとGPUは時間の範囲が完全には同一でない。既存CPU記録と単回GPU記録から信頼区間や統計的有意差は主張しない。ただし今回のGPUがCPUを上回ったとする根拠もない。
- 不合格までの短い停止時間、因子分解だけの時間、同一入力を再認証するiteration0時間を「成功解へのspeedup」と呼ばない。
- 未認証のaggregate/exchangeをPPOに接続する前に、元Newtonの各残差ブロック、境界dual、残った近接依存と正則化の相互作用を切り分ける。失敗試験の反復上限を増やすだけの方針にはしない。
- 認証可能な範囲を増やした後、構造fingerprintに基づく常駐数値更新と段階別バッチアダプターを実装する。現行の反復CPU基底再利用を含む対照と比較する。
- 閉ループでは同一actionの短い軌道から確認し、状態・報酬・終了/打切り・NH4閾値等を比較する。固定LPの合格だけでは、PHA終点やPPO advantageの精度を保証しない。

文献の適用範囲と判断方針は [実装計画](/home/reiya/co-cultivation/docs/CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)を参照。PPOの精度指標と観測の変更は [PPO精度・安定化記録](/home/reiya/co-cultivation/docs/PPO_ACCURACY_SPEED_RESULTS_20260906.md)を参照する。今回の内部初期化・barrier schedule・安全策付きPCは限定的な実装選択であり、引用論文の収束保証をそのまま移したものではない。

### 4.7 回帰検証の範囲 (出典: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md)

最終確認した対象テストは318件である。本体の対象suite 307件に、重複しない`test_graph_ipm_bridge.py`と`test_graph_temporal_lp.py`の11件を加えた集計であり、リポジトリ全件のpytestではない。

μの集計を`sum(comp/ng)`とし、中間総和のoverflowを避ける変更、非有限または非正のactive laneの棄却、δ変更中のfactor例外でも元設定へ戻す検証を含む。32環境試験はこの修正後のsourceを記録している。これらの回帰成功を、未実施の閉ループdFBA/PPOの精度・収束・速度の合格とは扱わない。

---

## 5. 文献の位置付け (出典: CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md, GPU_IPM_STABILITY_RESULTS_20260906.md)

- [Mehrotra (1992)](https://doi.org/10.1137/0802028)、[Wright, Practical Aspects of Primal-Dual Algorithms](https://epubs.siam.org/doi/10.1137/1.9781611971453.ch10)：予測・補正、中心化係数の適応、分解の再利用を考える根拠。今回の安全策付き限定実装に一般的な収束保証をそのまま主張しない。
- [Gleixner, Steffy & Wolter (2016)](https://pubsonline.informs.org/doi/10.1287/ijoc.2016.0692)：浮動小数点候補と正確な有理数計算を分離する根拠。今回の行関係の検証は論文のLP反復改良そのものの実装ではない。
- [Zanetti & Gondzio (2023)](https://arxiv.org/html/2106.16090v3)：内部線形系を過剰に解く負担と外側反復の進展を区別する根拠。
- [Zanetti & Gondzio, 2025](https://arxiv.org/abs/2508.04370)：正則化と内点法設計の参考。ただしそのアルゴリズム全体を再現したという主張ではない。
- [Pougkakiotis & Gondzio (2019)](https://arxiv.org/abs/1902.04834)：動的正則化の一般的根拠。今回の対角scheduleは非対角正則化やpivot制御の実装そのものではない。
- [Saad, 1993](https://epubs.siam.org/doi/10.1137/0914028)、[SIAM/Netlib Templates](https://www.netlib.org/templates/templates.html)：FGMRESの実装参考。

実測に応じて本計画とDATAFLOW_ACCELERATION_PLANを更新する。

---

## 6. 食い違いに関する注記

- **数値の食い違い**: 各元ファイル間で数値や結論に食い違いは確認されなかった。すべての数値・結論は原文のとおり転記している。
- **ステータスの一貫性**: 3ファイルすべてで「CPU超え未達」「全3段階認証未達」「閉ループ/PPO接続未達」が一貫して述べられている。
- **時間計測の範囲**: CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.mdではGPU solve API時間とCPU solve時間を比較しているが、GPU_IPM_STABILITY_RESULTS_20260906.mdではsetupを含まないsolve秒を報告している。両者の「solve」の定義は異なる可能性があるため、比較時は注意が必要。

---

## 構成元ファイル

- `docs/GPU_IPM_STABILITY_RESULTS_20260906.md` (2026-09-06)
- `docs/CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md` (2026-09-06)
- `docs/CPU_SPEED_IMPLEMENTATION_RESULTS_20260906.md` (2026-09-06)
