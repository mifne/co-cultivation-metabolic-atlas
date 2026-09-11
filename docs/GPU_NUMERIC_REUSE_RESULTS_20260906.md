# GPUバッチの時刻間再利用：実装と実測（2026-09-06）

後続の更新処理最適化と同一GPU状態での再始動比較は
[GPU_STATIC_UPDATE_RESULTS_20260906.md](GPU_STATIC_UPDATE_RESULTS_20260906.md)に掲載した。
新しい3試行中央値は32環境のstep3/4で更新込み1.722/1.485秒、CPU16並列は0.505/0.375秒。
以下はその前の実測を保持する履歴であり、最新値と混ぜない。

## 結論と比較の範囲

**GPUは既に独立環境をバッチ処理しているが、公平なCPU並列超えは未達。**
今回、時刻ごとのGPU workspace再構築を省く処理と、過去のGPU内部状態を引き継ぐ処理を
実装した。準備費用は減ったが、現在の数値更新にもhost処理が残り、反復法の停滞もある。
GPU利用率を上げるためだけの重複入力・無意味な演算は追加していない。

最新の現在LPベースdual再初期化では、32環境×3時刻×3試行の288問題すべてが元認証に合格した。
これは下記の旧plain warm/slack修復試験とは異なる新しい実験オプションである。
CPU側も同じ入力288問題が合格・retry0。3試行中央値は次のとおり。

| step | GPU解法 (s) | GPU構築/更新・確認込み (s) | CPU16並列の更新・解法・確認込み (s) |
|---:|---:|---:|---:|
| 2（初回構築） | 4.9110 | 16.2989 | 0.7469 |
| 3 | 0.9229 | 4.0956 | 0.5067 |
| 4 | 0.7534 | 3.7944 | 0.3832 |

GPU数値更新単体の中央値はstep3で3.1013秒、step4で2.9685秒。
元のslack修復方式のstep3 solve3.6467秒からは改善したが、CPUより速いわけではない。
初回構築・最終破棄等を含む3時刻sequenceの中央値はGPU24.8613秒、CPU1.6531秒。
固定した保存入力の繰返し測定であり、異なる条件の汎化検証やPPO全体の性能とは区別する。

GPU3試行の保存名は `pf_ipm_gpu_bound_restart_mu1e4_32_234_20260906.json`、
同名末尾`_repeat.json`、`_repeat3.json`。CPUは`pf_ipm_cpu1151_sequence32_234_20260906.json`、
同名末尾`_repeat2.json`、`_repeat3.json`。全6ファイルの全step入力hash一致・完了・認証を再確認。
繰返しdriverのファイル名展開が空になったため、2回目の保存名は`_repeat`となった。
次の同名起動は上書き拒否で計算前に停止し、第3試行は明示的な別名で正常に実行した。
既存の測定ファイルを上書きしたり、この起動失敗を性能試行として数えたりしていない。

すべて開発用保存入力 `pf_lp_trace_dev32x41_20260905` のmaxmin段、step2/3/4。
同じ環境順・元LP hashをCPU/GPUで確認し、保存済みCPU解ベクトルは読み込まず、GPUへ渡していない。
これはCPU軌道由来の入力を使う段別再最適化試験であり、GPU閉ループdFBAやPPOの性能ではない。
各表は単回診断であり、信頼区間・統計的優位は主張しない。

CPUは別置きHiGHS 1.15.1、環境別single-threadモデル、前時刻basis再利用あり。
4環境では4 workers、32環境では16 workers。同一段の変化する数値入力を順次解く。
元LPのprimal residual≤1e-5、dual violation≤1e-7、relative KKT gap≤1e-7を両者に要求し、
直接計算したdual objectiveとの相対差≤1e-7も独立に確認する。

## 実装

- `src/gpu_ipm_warm_state.py`: 同じ環境順・stageの直前stepだけのx/y/z/sをdevice上で所有コピー。
  座標、device/stream/thread、target hash・numeric generationを照合。新LPを必ず再認証する。
  analytic-box双対で元出力が認証されても内部y/zの認証を意味しないことをmetadataに明記した。
- `src/gpu_newton_krylov.py`: Arnoldi basis、方向、Hessenberg、回転、右辺の6大配列を再利用。
  別呼出しの返却値を上書きせず、alias・再入・device/stream不一致を拒否する。
- `src/lp_zero_face.py`: 旧縮約を現在の境界で再証明する`rebind`。不成立なら拒否し、古い縮約を使わない。
- `src/gpu_ipm_numeric_update.py`: full/forest/zero-face/core全段の係数・RHS・bounds・cost・
  transpose・認証・postsolve witness・KKTを現在値で再構成。native factor handle/analysisと
  固定device bufferを維持し、数値factor・旧内部状態・旧判定を失効させる。
  precommit非互換だけ`NumericRebindRejected`、commit途中の失敗はworkspaceを使用不能にする。
- `scripts/benchmark_ipm_sequence.py`: fresh/rebind、cold/internal warm、slack repairを明示オプション化。
  数値更新・再構築・解析・破棄回数、source snapshot、全環境の元認証、各費用を保存する。
  CPU追加認証には返却済みowned snapshotを使い、余分なnative getter往復を課さない。

全オプションは実験用であり、PPO既定backend・GEM・許容誤差を変更していない。
CSR組立、縮約証明、Python制御がhostに残るため、完全GPU内完結とは呼ばない。

## 実測：内部状態＋現在のslack再構成＋workspace再利用

GPUの「全step費用」は構築/更新、bind、solve、D2H独立認証、export再認証、必要なcloseを含む。
CPUの表値はモデル作成/更新・HiGHS・backend認証。追加独立確認の時間も別列へ示す。
CPU service全体の構築/破棄等は後述のsequence全費用に含む。GPUだけsetupを除いてCPUと比較しない。

| 環境数 | step | GPU solve (s) | GPU全step費用 (s) | GPU認証 | CPU solve (s) | CPU solve＋独立確認 (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 2.2131 | 5.1403 | 4/4 | 0.1897 | 0.1965 |
| 4 | 3 | 1.0089 | 1.4443 | 4/4 | 0.1607 | 0.1673 |
| 4 | 4 | 1.2345 | 1.6094 | 4/4 | 0.1049 | 0.1111 |
| 32 | 2 | 5.1790 | 16.9806 | 32/32 | 0.7129 | 0.7635 |
| 32 | 3 | 3.6467 | 6.8657 | 32/32 | 0.4921 | 0.5436 |
| 32 | 4 | 6.7983 | 9.8104 | 31/32 | 0.3310 | 0.3832 |

CPUは4環境で実LP12回、32環境で実LP96回、すべて認証、数値retry0、process exit0。
GPU側のCPU LP呼出しは0。GPUはいずれもworkspace作成1回、数値更新2回、symbolic analysis1回。
4環境のsequence全費用はGPU8.8929秒、CPU0.4810秒。
32環境はGPU34.3296秒で1問題不合格、CPU1.7236秒で全合格なので、成功速度比として扱わない。
sequence時間には入力読込を含めない。起動済みPython内の測定で、OSのcold-startを意味しない。

保存結果（すべてresults配下、元入力hash・設定・source・独立認証を収録）：

- `pf_ipm_cpu1151_sequence4_234_20260906.json`
- `pf_ipm_cpu1151_sequence32_234_20260906.json`
- `pf_ipm_gpu_rebind_slack_repair4_234_20260906.json`
- `pf_ipm_gpu_rebind_slack_repair32_234_20260906.json`

## 律速を分ける

32環境step3の数値更新3.1499秒は、host再証明1.6500秒、payload組立・旧device照合・staging
1.4851秒、commit0.0148秒。後者のstagingという名称は純GPU転送時間ではない。
旧device値を各配列ごとにhostへ戻して検査する正しさ優先の実装であり、検査・構造再利用の
GPU集約と、hostで毎回行う疎行列再構成の削減が必要である。

同stepのGPU解法3.6467秒には、factor1.0272秒、三角solve/Krylov/更新2.3641秒、
反復内認証0.1870秒がかかる。35 factor・219 solveで、数値更新をゼロにしてもCPU並列を超えない。
4環境のplain内部warmではGMRES補正がほぼ不要な時刻もあるため、GMRESだけの最適化で
全バッチサイズ・全時刻が改善すると仮定しない。

## 非採用・不合格も記録

- GMRES workspace単独を有効にした32環境step2は4.9288秒、32/32認証、setup11.9566秒。
  step1の3.28秒と入力が違うので、workspaceの効果とは比較しない。
- fresh構築＋plain内部warmの4環境step2/3/4は2.3105/0.5484/2.1603秒、すべて合格。
  `pf_ipm_gpu_interior_sequence4_234_20260906.json`（追加host direct audit導入前のrun）。
- 同じ設定の数値更新版はstep3がsolve0.5417秒・全費用0.9773秒で合格したが、step4は2/4。
  `pf_ipm_gpu_rebind_interior4_234_20260906.json`。個別に速い時刻を全体の成功と扱わない。
- slack修復だけのfresh4環境は2.2544/1.2475/1.3902秒、すべて合格。
  `pf_ipm_gpu_interior_slack_repair4_234_20260906.json`。
- 内部slack/dualのfloorを1e-4にした32環境試験はstep3 solve29.2571秒へ悪化し、step4は
  solve61.2613秒で不合格。採用しない。`pf_ipm_gpu_rebind_floor1e4_32_234_20260906.json`。
  floorは初期補助変数だけの変更であり、物理モデルのboundsや精度閾値は一切緩めていない。

旧freshと新rebindのplain warmは、更新前のcold step2の微小な丸め差から既に内部状態が異なる。
旧runのstep4不一致だけで、更新漏れとも更新の正しさとも結論できない。
元出力はanalytic boxで認証される一方、内部dual gapは大きく、次時刻のs≈1e-17、z/s≈1e22など
非中心的な提案が停滞する例を確認した。同一のGPU内部snapshotを両方式へ配る対照で切り分ける。

追加した`probe_ipm_rebind_matched_state.py`のstep4/rebind-first対照では、内部x/y/z/s・
current operator・KKT・postsolve/認証配列の332件がbyte単位で一致した。未因子化のnative
factor.values scratchだけは、次のfactorで全て上書きする未使用領域として別記録した。
同一の初期点でもrebindは2/4認証（4.7257秒、161 factor）、freshは0/4（7.1268秒、240 factor）。
これにより再利用処理だけに原因を帰すことはできないが、両方式の数値挙動が同一ともいえない。
両handleのcloseを確認、CPU LP0、保存済みCPU解なし。失敗した返却ベクトルを正解と扱わない。
保存：`pf_ipm_matched_rebind_first4_step4_20260906.json`。専用CPU契約テスト12件は合格した。

## Graph能力確認と次の計画

`pf_cudss071_graph_replay_probe_20260906.json`：installed cuDSS 0.7.1、製作SPD行列4×17、
固定数値factorのCUDA Graph再生を5回実施。変化RHS、元残差、NaN拒否、その後の回復を確認した。
captureは0.000138秒。ただし実GEMの不定値KKT、numeric refactorを跨ぐ再生、全IPMは未検証。

1. 同一内部snapshotによるfresh/rebind対照で更新の影響を分離する。
2. 不適切な内部dualを無条件に引き継がず、現在LPの残差と中心性からGPU初期点を構成する。
   単純な一律floorは悪化したため採用しない。元LP認証は維持する。
3. 数値更新を、静的構造の検査と動的係数のdevice計算に分ける。同期読出しを集約する。
4. 固定factorのNewton RHS縮約→scale→sanitize→solve→方向復元を限定Graph化する案を検証。
   factor更新ごとにgraphをdrain/destroyし、warmup/capture費用も含めて採否を決める。
5. 強いCPU比較、後段aggregate/exchange、独立した閉ループdFBA/PPOまで同じ基準で検証する。

### 追加実装：旧内部dualを破棄する現在LPベースの再初期化

`src/gpu_ipm_reoptimization.py`と`--restart-mu 1e-4`を追加した。
GPU warm xは変えず、等式dualを0とし、現在costからbound-dual候補
`d=[0, max(c[lower],0), max(-c[upper],0)]`を構成する。
`p=max(d,sqrt(mu)); s=max(h-Gx,mu/p); z=max(d,mu/s)`で内部変数だけを再構成する。
旧巨大y/zは使わない。s/z・z/s・相補性の有限性と正値も検査する。
この点は一般には実行可能でも最適でもなく、mu中心そのものという保証もない。
物理bounds・フラックスx・元LPの受理条件は変えず、以後のNewton解法と元認証を必須とする。
slack修復・一律floorとは別モードであり併用を拒否する。

初回測定で4環境step3/4は11/9 factor、solve0.3181/0.2764秒で全認証。
32環境step3/4は12/10 factor、solve0.9229/0.7652秒で全認証。
32環境の全step費用は4.0956/4.0560秒で、host数値更新が支配的に残る。
CPUのモデル/basis再利用あり比較はsolve0.4921/0.3310秒、独立確認込み0.5436/0.3832秒なので、
まだCPU超えではない。この初期化は開発入力で選んだ実験オプションで、未検証stageへ既定適用しない。
保存：`pf_ipm_gpu_bound_restart_mu1e4_4_234_20260906.json`、
`pf_ipm_gpu_bound_restart_mu1e4_32_234_20260906.json`。

参考：warm startは小さな入力摂動でも境界で停滞し得るため、単なる旧反復点の再利用と、
再中心化・unblockingを区別する。今回の実装は次の論文の完全な再実装ではない。
[Gondzio & Grothey, Re-optimization with the primal-dual interior point method](https://www.maths.ed.ac.uk/~gondzio/reports/crash.pdf)。
CUDA Graphについてはinstalled版の実試験と現行文書を区別する。
[NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/general.html)、
[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

回帰試験：関連44ファイル、845 passed / 1 skipped、13.27秒。
`results/pf_ipm_update_regression_20260906.xml`へ保存。小さなCUDA試験を含むが、実GEM全段認証を
意味しない。CPU比較スクリプトの初回起動はcontext manager非対応で計算前に失敗し、
`closing(RepeatedCpuLP(...))`へ修正・リソース契約テスト追加後に、上記CPU測定を実行した。

最新の再初期化・matched-state検証追加後の最終回帰：関連46ファイル、906 passed / 1 skipped、
13.35秒。`results/pf_ipm_restart_regression_20260906.xml`に保存。上記の845件はその前段の結果である。
本稿に記載した計測・テストのジョブはすべて終了し、GPU workspaceを解放した。
