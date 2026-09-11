# GPUバッチ処理の速度と残る律速

2026-09-06。対象は現行GEMから保存した開発用LP入力。培養閉ループ/PPO全体の速度ではない。

## 同一32入力の測定

入力: `results/pf_lp_trace_dev32x41_20260905`、physical step 1、maxmin段階、異なる32環境。
GPUはRTX 4060 Laptop、cuDSS uniform batch 32。元の参照x/yは読んでいない。
実験条件はbalanced initialization、PC予測fraction .995、globalized full forcing .1、
Krylov16、exact equalities、GPU融合MGS/Givens、device finite check、cuDSS内部refinement0。
外側の元Newton残差と元LP精度閾値は維持した。

| 実装 | 32 LPの秒数（中央値） | 認証済みLP/秒 | 反復 |
|---|---:|---:|---:|
| GPU新方式・setup除外 | 3.572876 | 8.96 | 3 |
| CPU HiGHS 1 worker・fresh model | 4.220701 | 7.58 | 3 |
| CPU HiGHS 4 workers・fresh model | 1.239123 | 25.82 | 3 |

GPUは3.908860/3.457349/3.572876秒で96/96解が元LP基準を通過。
CPUも全試行通過。中央値比でCPU1 workerに対して1.181倍、約15.35%の時間短縮。
CPU4 workersには約2.883倍遅い。n=3の開発測定で、統計的有意性/広い条件への一般化は主張しない。
GPU constructor/setupには別途8.54～9.10秒を要しており、cold総時間ではCPU超えではない。
setupの償却を架空に仮定したPPO全体の速度値は算出しない。

測定ファイル:

- `pf_ipm_microall_device_ir0_maxmin32_20260906.json`
- `pf_ipm_microall_device_ir0_maxmin32_repeat_20260906.json`
- `pf_ipm_microall_device_ir0_maxmin32_repeat3_20260906.json`
- `pf_ipm_cpu_maxmin32_speed_recheck_20260906.json`

上記はすべて `results/` 配下。repeat無番号のファイル名はシェル変数展開の不具合によるもの。
次の同名実行は上書き拒否で停止しており、成功試行を増やして数えていない。

## 並列化済みでも遅い理由

repeat3の3.572876秒のうち、分解0.924743秒（25.9%）、三角solve/Krylov/更新
2.402237秒（67.2%）、元LP認証0.147187秒（4.1%）。外側反復38回、batched solve193回。
受理反復は31～38。factorだけ無限に高速化しても約2.65秒残り、CPU4 workersに届かない。

1回のcuDSS呼出しに32行列を渡している。単一CUDA streamだから32問題をPythonで逐次
解いている、という意味ではない。一方、各LP内部のNewton反復とKrylov依存は逐次で、
各演算のhost分岐/同期も残る。MGS融合は1環境1block、Givensは1環境1threadであり、
バッチを小分割すれば単純にGPUの仕事量を増やせるわけではない。

## 今回の修正と採否

- 内部cuDSS solveのfinite判定をdeviceに残してhost読出しを削減。無効laneはNaNで返し、元のゲートが拒否。公開APIの契約は不変。
- 2-pass逐次MGSとGivens/後退代入を融合。単独では32環境5.596秒に留まった。
- cuDSS内部固定2回の補正と外側の元Newton補正を分離。内部0回が上記3.57秒の主な改善。
- 有理数で有限境界上の差を検証した1行作業緩和を限定実装。元モデルは不変で、厳密同値縮約とは呼ばない。
- 直接双対目的差と等式dual×残差の監査を追加。旧相補性指標だけで大きな目的誤差を見逃さないようにした。
- 等式のみ2冪でlossless正規化する実験を追加したが、exchange4環境は0/4認証。既定化しない。
- 整理済みLPをcuOpt native barrierへ渡した別方式は20.13秒でtime limit、0/4認証。失敗停止を高速化として数えない。

後段の不安定性は、単なる辞書不足ではない。原row616は縮約後
`-.0002*x70 + .01*x486 = 0`。成長保持制約からx70>=.00495、したがってx486>=9.9e-5が必要なのに、
反復中x486≈1.4e-9、境界曲率z/s≈3.8e10となり、必要な等式方向を十分修正できなかった。
小さいslackを機械的に0固定する改善は不適切。

次はゼロ面縮約後に新たに2項になった等式を再度代数的に整理する。
その後、環境数別の認証済みLP/s、setup/solve、CPU1/4 workersを比較し、
最後に状態が実際に変わる閉ループへ進む。GPU使用率を上げるための無意味な計算は加えない。

## 追加測定と実装

| GPUの環境数 | solve秒 | 認証済みLP/秒 | 測定 |
|---|---:|---:|---|
| 8 | 1.787036 | 4.48 | 単回・8/8認証 |
| 16 | 2.712076 | 5.90 | 単回・16/16認証 |
| 32 | 3.572876 | 8.96 | 前述3回中央値・96/96認証 |

追加ファイルは `pf_ipm_batchscale8_maxmin_20260906.json` と
`pf_ipm_batchscale16_isolated_maxmin_20260906.json`。同じtraceの先頭8/16/32環境という
入れ子の部分集合であり、同じ32問題を異なるstream配置で解いた比較ではない。
測定時刻・反復数も異なるため、この表はスケール傾向の開発診断で、統計的な性能保証ではない。
16環境の最初の測定は別GPUテストが重なった可能性があるため除外し、独占状態で再計測した。
8→32で1秒あたりの処理数は約2倍となり、既存のバッチ効果は確認できる。
この傾向を64/128環境や新GPUへ直線外挿しない。

2段目forestを `--second-forest` として接続した。ゼロ面処理で新たに生じた2項等式を
再度座標に組み込み、境界witness付きのGPU主双対写像を段階的に逆適用する。
元モデル・目的・全行認証は保持し、追加の直接dual gap判定も必須にした。
実GEM exchange4の新作業系は16,895次元（従来17,192）となり、7.341秒で停止したが
0/4認証だった。この数値を有効解までの高速化とは扱わず、オプションのまま保持する。
新規toy統合試験は、ゼロ固定後に3項→2項となる等式、cold/warm、主双対復元、
入力不変と非有限拒否を検査した。

次の主要対象は残る67%の補正/更新である。通常のGMRES反復にはGPU値をPython boolへ
戻す複数の判定が残る。GPU側maskを保持した同期集約・常駐作業buffer・同じ総問題数での
stream配置比較を優先し、最後に数値更新APIと前回認証済みGPU解による因果的warm startへ進む。

最終確認: 関連する数値層・写像・比較・GNN/GRU接続の31テストファイルをまとめて実行し、
585件合格・1件skip（12.52秒）。リポジトリ全件のテストではなく、実GEM全段認証/PPO速度優位の代替でもない。
終了時に対象の空ファイルは残らず、GPU計算ジョブも終了済み。

## 再開時の復元

再開時、`gpu_zero_face_ipm.py`、`gpu_forest_ipm.py`、`probe_graph_ipm.py`が空だった。
前2本はnative barrier測定、後1本はrepeat3測定の保存済みsource_snapshotsからapply_patchで復元。
原因は断定していない。復元後の関連テスト304件合格・1件skipを確認した。
途中でJSONに-infが含まれて保存が失敗したnear実験のpartial JSONは結果表に使用せず、
後続出力では非有限診断値を明示文字列に変換し、全体をシリアライズしてから新規作成する。

参考: [NVIDIA cuDSS設定](https://docs.nvidia.com/cuda/cudss/types.html)と
[性能測定・構造再利用の注意](https://docs.nvidia.com/cuda/cudss/tips_and_tricks.html)。
内部反復改良と構造再利用の根拠であり、このGEMでの速度保証を与える資料ではない。

## 追加検証 — CPU比較の強化と単項等式（Revision 23）

CPU4並列を最良値とする前提を取り消し、1/2/4/8/10/16 workersを調べた。
HiGHS1.14の同一process内連続測定はnative allocatorエラーで異常終了した。
owner thread固定後も再発したので、「thread所有を直せば解消した」とは主張しない。
異常終了したsweepには完成JSONがなく、完了した性能比較として採用しない。
個別processのcold1試行は各並列数でexit0・32/32認証だったが、永続運用の安定性の証拠ではない。

既存1.14を上書きせず、`tmp/highspy-1.15.1-comparison`へ1.15.1を別置きした。
PYTHONPATHでこの別版を指定したsweepはprocess exit0、合計1,152回の実LP solveが認証された。
同じ32入力、各設定cold3試行と同一入力hot1回/試行、数値retry0である。

| CPU worker数 | cold3試行中央値（秒） |
|---|---:|
| 1 | 4.4460 |
| 2 | 2.3732 |
| 4 | 1.3933 |
| 8 | 0.9312 |
| 10 | 0.8747 |
| 16 | 0.7644 |

測定範囲は入力検証・model作成・solve・backend元LP認証。process起動、外部独立認証は別。
同一入力hot値は変化するdFBAの時間と解釈しない。worker順sweepは構成選定であり、
ランダム化したCPU/GPU対応試験の代用ではない。元環境のdefaultはまだ1.14のままである。
保存: `results/pf_ipm_cpu_highs1151_owner_maxmin32_sweep_20260906.json`。
公式1.15にはscheduler/cleanup等の修正があるが、今回のmalloc異常と同一原因とは未確定。
[HiGHS 1.15 release](https://github.com/ERGO-Code/HiGHS/releases/tag/v1.15.0)。

GPUの新しいopt-in処理は次の通り。

- GMRES中間host判定3箇所を抑制。残差・finite・active maskを保持。
- ゼロRHS単項等式a*x=0から、ゼロを跨ぐ境界の変数もゼロ固定。
  GPU dual復元では符号自由なq/aを使用。元問題や許容誤差は変えない。
- 2段目forestと厳密等式縮約へ接続。near-equalityの旧座標候補は併用しない。

| GPU構成 | maxmin32 solve（秒） | 認証 | setup（秒） |
|---|---:|---:|---:|
| 中間判定削減のみ | 3.5041 | 32/32 | 8.8331 |
| 単項等式＋2段目forest＋中間判定削減 | 3.2789 | 32/32 | 12.8038 |

どちらも単回の開発診断であり、setupはsolveから除外している。新CPU16 workersより速いとは
言えない。後者のexchange4は8.0747秒で0/4認証、相補性gapが4.63e-6～1.87e-5で基準未達。
計算量削減と全段認証は別課題であり、PPO backendの既定値へ昇格していない。
保存: `pf_ipm_deferred_control_maxmin32_20260906.json`、
`pf_ipm_singleton_second_exact_maxmin32_20260906.json`、
`pf_ipm_singleton_second_exact_exchange4_20260906.json`（全てresults配下）。

新しいGPU単項等式の15テストは全合格。符号・逆順cascade・無限境界・非有限拒否・
Forest/厳密等式との組合せを検査し、CPUの代数的dual復元と元LP認証で照合した。
次に同じ32入力を複数streamに割り当て、全stream完了までのwall timeを測る。

## 同じ32 LPを複数streamへ分割した実測

新規実装: `src/gpu_stream_partition.py`、`scripts/benchmark_gpu_stream_partitions.py`、
`tests/test_gpu_stream_partition.py`。全constructor/analysisを順次完了後、独立cuDSS
handle/data/buffer/streamを同じhost threadのgreenletから協調実行する。
元数値helper、所有権guard、有限値、元Newton残差、元LP認証を変更せず、例外時はdrain後に解放する。

| 配置・切替方式 | 全32問題完了（秒） | 認証 |
|---|---:|---:|
| 1x32、直接呼出し | 3.6260 | 32/32 |
| 1x32、factor/solve後に協調切替 | 3.8101 | 32/32 |
| 2x16、factor/solve後に協調切替 | 3.9685 | 32/32 |
| 4x8、factor/solve後に協調切替 | 6.1368 | 32/32 |
| 2x16、factor後のみ切替 | 4.2667 | 32/32 |
| 4x8、factor後のみ切替 | 6.4023 | 32/32 |

各1試行の開発診断。全て同じtrace/step1/maxminの32個の異なるproblem hashで、CPU1151
sweepの入力hash順とも一致。元精度基準は同一。セットアップ・D2H・外部独立認証・closeは
別計測であり、上表は同期済み全shardのsolve wall timeだけを示す。
singleton/second-forestを使わない統一構成なので、前節3.2789秒とは処理構成が異なる。
この結果から4streamを採用せず、stream分割の優先度を下げる。

| 協調shard数 | batch factor呼出し | batch solve呼出し | 環境単位のsolve数 |
|---|---:|---:|---:|
| 1 | 38 | 195 | 6,240 |
| 2 | 70 | 322 | 5,152 |
| 4 | 138 | 597 | 4,776 |

遅い環境に引きずられる計算量は減ったが、小batchへの呼出し回数が約3倍となった。
host起動/同期と小演算の反復費用が勝つ説明と整合するが、kernel timeline未取得なので
この内訳だけでCPU起動費用の割合を断定しない。setupは8.413/10.536/15.390秒と増えたが、
この増加は上表solve秒に含めていない。event pendingの重なりもkernel並行動作の証明ではない。

保存: `results/pf_gpu_stream_partition_maxmin32_pilot_20260906.json`、
`results/pf_gpu_stream_partition_maxmin32_direct_20260906.json`、
`results/pf_gpu_stream_partition_maxmin32_factor_only_20260906.json`。
入力・source snapshots・設定・全環境の独立認証・lifecycle内訳を含む。

次は1x32の常駐workspace、限定CUDA graph可否検証、因果的な前回GPU状態再利用へ進む。
factor自体にも旧内訳で約0.9秒がかかるため、起動削減だけでCPU16 workers0.764秒を
超える保証はない。反復数削減と、CPUにもbasis再利用を許した変化入力の対称比較が必要。
詳細は計画Revision24と[stream監査](GPU_STREAM_PARTITION_AUDIT_20260906.md)。

最終回帰試験: 関連43ファイル、772 passed・1 skipped（14.03秒）。既存1.14環境で実行。
別置き1.15.1はCPU/owner/benchmarkの40試験も合格。PPO全体のCPU超えは未達である。

## 続報：時刻間の数値更新と内部状態再利用

native workspace/analysisを保持し、現在LPの数値と全段の証明データを更新する処理を実装した。
初期のslack再構成方式では4環境step2/3/4が全認証、32環境ではstep4に1問題未認証だった。
CPUにも時刻間basis再利用を許すと、32環境step3はCPU0.4921秒に対しGPU解法3.6467秒、
更新・独立確認込み6.8657秒であり、CPU並列超えは未達である。
実装、失敗も含む比較、時間範囲、845件合格の回帰試験は
[時刻間再利用の結果](GPU_NUMERIC_REUSE_RESULTS_20260906.md)にまとめた。

その後、旧内部dualを捨てて現在LPから再初期化する方式を追加した。32環境×3時刻×3試行の
288問題がすべて元認証に合格し、step3/4のGPU解法中央値は0.9229/0.7534秒へ改善した。
ただし行列更新・確認込みは4.0956/3.7944秒、CPU16並列の更新・解法・確認込みは
0.5067/0.3832秒であり、CPU超えは未達。新方式・対照試験を含む最終回帰は906 passed/1 skipped。
更新費用約3秒の削減を次の優先課題とし、詳細は上記の時刻間再利用レポートと計画Revision26に保存した。
