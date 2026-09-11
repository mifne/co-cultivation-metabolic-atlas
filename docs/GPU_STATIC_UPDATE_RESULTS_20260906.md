# GPU数値更新の静的構造再利用と再始動比較（2026-09-06）

## 結論と適用範囲

32環境を既にGPUバッチで解いている。今回、変化するLP入力に対するCPU側の重複再証明、
森林map再構築、疎行列組立を削減し、GPU再始動のパラメータを対照比較した。
同じ保存入力のmaxmin段では更新・解法・検証等込みの中央値を、step3で4.096→1.722秒、
step4で3.794→1.485秒へ短縮した。**CPU16並列は0.505/0.375秒であり、依然CPUが速い。**

保存入力はCPU由来の軌道上の状態であり、GPU独立の閉ループdFBA/PPO軌道ではない。
この結果を120ステップ全体、PPO学習速度、aggregate/exchange段、別GPUへ外挿しない。
数値的なLP最適化はGPUで実行しCPU LP fallbackは0だが、入力準備・制御・検査がhostに残る。
「完全GPU内完結」「CPU並列超え」「PPO全体高速化」の達成とは記載しない。

## 同一入力・同一精度での再測定

32環境×step2/3/4、各方式3独立プロセス。新GPUとCPUをGPU1→CPU1→GPU2→CPU2→GPU3→CPU3の
順で排他的に実行した。GPU各試行の実装hashは一致、CPU各試行も一致した。
旧GPUは前の測定記録との時系列比較であり、無作為化交差試験ではない。
値は中央値、単位は秒。step番号は保存された環境更新の時点で、PPO timestepsではない。

| 同じ32環境の処理 | 旧GPU 更新等込み | 新GPU 更新等込み | 新GPU 解法のみ | CPU16 更新・解法・検証 |
|---|---:|---:|---:|---:|
| step2（この試験の初回） | 16.299 | 16.179 | 4.817 | 0.735 |
| step3 | 4.096 | 1.722 | 0.646 | 0.505 |
| step4 | 3.794 | 1.485 | 0.522 | 0.375 |

GPUの「更新等込み」はworkspace生成または数値更新、warm bind、解法、D2Hと独立元LP検証、
次時刻用snapshotの再認証・export、当該時点の破棄を含む。CPUは環境別モデル/basisを再利用し、
更新とnative解法・結果snapshot・独立元LP検証を含む。両方とも入力ファイル読出しは別計測。
初回setupを無償としない。初回を含む3時刻のsequence lifecycleは、旧GPU24.861秒、
新GPU20.084秒、CPU1.643秒。hot時点だけの短縮を全工程の2倍以上の高速化と解釈しない。

新GPU step3更新等込み範囲1.683–1.735秒、step4は1.452–1.488秒。
対応するCPU範囲は0.498–0.506秒、0.370–0.380秒。
旧→新のhot時点中央値比は2.38/2.55倍だが、これはCPUに対する倍率ではない。
GPU/CPUとも288 LPインスタンス全て認証、CPU HiGHS 1.15.1はretry0、GPUのCPU LP呼出し0。
全方式・全試行で96個の元入力hashと環境順序・manifestが一致することを集計時に検査した。

認証条件は元LPのprimal residual≤1e-5、dual violation≤1e-7、relative KKT gap≤1e-7。
独立した直接dual objective gap≤1e-7も確認した。現在または未来のCPU解はGPUへ渡していない。
原モデル、化学量論式、境界、目的関数、判定閾値を高速化のために変更していない。

## 実装と律速の変化

プロファイラはLP最適化をせず、step2のworkspaceをstep3へ更新する処理だけに適用した。
介入付き総時間は4.259→1.411秒、呼出し数5,278,155→904,022。速度表にはこの時間を使わない。

| 処理 | 旧profile | 再利用版profile | 修正 |
|---|---:|---:|---|
| zero-face再証明 | 1.968 | 0.121 | 等式・固定値・ゼロ相対符号が同じ場合の証明再走査省略 |
| 二次森林map | 1.077 | 0.156 | 所有・検証済みの静的map共有、現在の動的witness再計算 |
| 旧・新payload | 0.783 | 0.615 | constraint_form 4B→2B、後続でKKT直接組立も追加 |
| 旧device buffer照合 | 0.075 | 0.074 | この部分だけが主要律速という仮説は非採用 |

上表の再profileは直接KKT/CSR組立を入れる前。最終の非profile数値更新中央値は
step3で3.101→0.997秒、step4で2.969→0.881秒。まだGPU解法より大きい。

- zero-face証明は、係数・等式RHS・明示固定値が一致し、各境界のゼロに対する符号が同じ時だけ
  再利用する。条件が変われば元の証明再走査へ戻る。証明hashのJSON表現は旧版と同一byteを維持。
- forest境界集約は、事前固定順のsegmented reduceatを使用。負の重み、無限境界、最大絶対重み→
  最小元列番号のwitness選択、lower>upperの厳密拒否は保持する。
- 二次mapは初回に独立再構築したcanonical planを所有。hash/shape/native dtype/deviceと旧device値を
  検査し、現在入力の縮約・目的関数・witnessを独立再計算して一致確認してから共有する。
- CSR block diagonalを、canonical入力ではCOO経由にせず直接連結する。非canonicalは全入力検査後
  SciPyへ委ねる。異常ポインタはnative変換前に拒否する。
- opt-inのKKT直接組立は、現在E/Gの両三角と全対角を既存patternへscatterする。全lane unionの
  座標被覆を再検証し、missing/extra座標の両方を拒否。旧bmat+unionとbit一致をテストした。
- GPUで85個のbuffer照合をまとめるオプションも作成したが、単回1.950/1.709秒対
  通常照合1.936/1.677秒で改善せず、最終比較では無効とした。

## 同一GPU状態からの再始動実験

step2を一度だけGPUで解き、元LPと直接dual gateで認証。所有した同一x/y/z/sを全条件で
変更していないことをGPU上で確認し、step3に同じxを渡した。旧y/z/sは初期化方法に従って
再構成する。各条件のstep4は自身のstep3の認証済みGPU状態を使う。
以下は32環境の単回対照実験であり、恒常的な最適μや一般的収束保証ではない。

| restart μ | step3 factor回数 | step4 factor回数 | 2時点の解法API時間合計 | 全条件の元LP判定 |
|---|---:|---:|---:|---|
| 1e-3 | 20 | 24 | 4.664 | 合格 |
| 1e-4 | 12 | 10 | 1.554 | 合格 |
| 1e-5 | 11 | 9 | 1.375 | 合格 |
| 1e-6 | 12 | 10 | 1.502 | 合格 |

4環境smokeも全条件合格。1e-5を続く3回のsequence比較の実験設定に採用し、11/9 factorを
全3回で確認した。PPO本番の既定へは昇格していない。小μだけを選べば改善するわけではなく、
GPU解法0.646/0.522秒だけでもCPUの0.456/0.326秒より遅い。host更新削減だけでも不十分。

## 再現用ファイル

- `scripts/benchmark_ipm_sequence.py`：独立プロセスでのsequence比較。
- `scripts/probe_ipm_restart_mu_matched.py`：同一GPU sourceの対照比較。
- `scripts/profile_ipm_numeric_update.py`：最適化を行わない更新profile。
- `scripts/summarize_ipm_static_compare.py`：hash・認証・実装一致検査を伴う排他的集計。
- `results/pf_ipm_static_compare_summary_20260906.json`：全試行ファイル名・raw時間・集計。
- `results/pf_ipm_restart_mu_matched4_20260906.json`、`pf_ipm_restart_mu_matched32_20260906.json`。
- `results/pf_ipm_numeric_update_profile32_20260906.json`、`pf_ipm_numeric_update_profile32_static_20260906.json`。

最終sequenceのGPU指定は `--reuse-numeric-workspace --reuse-static-forest --direct-kkt-payload
--restart-mu 0.00001`。CPUは別targetのHiGHS 1.15.1と`--workers 16`、basis再利用有効。
すべてRTX 4060 Laptopと現在のCPUで実行した。架空のGPU負荷やLP複製による利用率増加は行わない。

## 最終回帰検査

前の46ファイルに新規・影響範囲10ファイルを追加し、56ファイルをまとめて実行した。
**1,314 passed / 4 skipped / 11 warnings、26.93秒、exit0**。
4件のskipは全て2台目のCUDA GPUが必要なdevice跨ぎ検査であり、現在の単GPUでは実行できない。
異常入力、旧状態変更、precommit失敗、static map破損、同値だが不正なdtype、現在KKT座標不整合、
native workspace再利用、CUDA原LP認証の回帰を含む。記録は
`results/pf_ipm_static_reuse_regression_20260906.xml`。
