# GPU数値更新の実装・検証結果 — 2026-09-07

GPU側の反復係数更新を約1秒から約0.05～0.07秒へ短縮した。ただし、強いCPU16並列を
更新・求解・検証込みで上回ってはいない。今回の測定は保存済みmaxmin LP入力の再生であり、
独立した閉ループdFBA、aggregate/exchange、PPO全体の高速化を示すものではない。

## 実装

`DeviceNumericUpdatePlan`を追加し、初回に検証済みの疎行列構造を固定する。現在の
A.data/RHS/lower/upper/cを入力し、第1forest縮約、固定変数の代入、第2forest、厳密等式の
選択、E/G・転置・KKT・元LP認証バッファ・現在の境界witnessをGPU上で更新する。
決定的なsegmented reductionを使い、相殺で消えていた係数が現れる場合と、その逆も検出する。

静的インデックスを実行前に照合し、数値・構造条件をすべて通過した後だけcommitする。
条件外はfast pathを拒否する。ベンチマークはその場合のGPU再構築を明記・計時するが、
今回の32環境×7時刻では再構築0回、CPU LP呼出し0回だった。旧解の受理状態と数値factorを
無効化し、更新後の元LPを改めて認証する。

現在の元LPハッシュを保存する一方、中間host縮約スナップショットはlayout専用と明記した。
旧host更新APIはこの状態を拒否する。host上の入力生成・pack/hash・制御、初回構造処理と
独立検証のD2Hは残るため、**完全GPU内完結とは表現しない**。

主なコード:

- `src/gpu_ipm_device_update.py`: 条件検査・数値縮約・transactional commit。
- `src/gpu_ipm_device_payload.py`: 全数値operatorへの更新先マッピング。
- `src/gpu_segmented_linear.py`, `src/gpu_forest_numeric_bounds.py`: 決定的GPU演算。
- `scripts/benchmark_ipm_sequence.py --device-numeric-updates`: opt-in検証経路。PPO既定は変更なし。

## 同一入力・同一精度のCPU比較

RTX 4060 Laptop、異なる32環境、保存済みstep2～8。CPUはHiGHS 1.15.1、16 workers、
各workerのsolverスレッド1、model/basis再利用あり。両経路は順次実行し、性能測定中に
他のテスト・GPU試験を重ねていない。入力hash、環境順序、ソースhash、精度閾値の一致を
集計スクリプトで検査した。各backendで224/224問題を認証、CPU側の再試行も0回。

| 保存時刻 | GPU求解のみ (s) | GPU係数更新 (s) | GPU更新・求解・検証等込み (s) | CPU16更新・求解・検証込み (s) |
|---|---:|---:|---:|---:|
| 2（初回） | 4.758 | — | 20.631 | 0.689 |
| 3 | 0.621 | 0.071 | 0.778 | 0.469 |
| 4 | 0.492 | 0.052 | 0.608 | 0.360 |
| 5 | 0.489 | 0.052 | 0.604 | 0.298 |
| 6 | 0.543 | 0.051 | 0.655 | 0.365 |
| 7 | 0.443 | 0.052 | 0.556 | 0.321 |
| 8 | 0.487 | 0.051 | 0.602 | 0.330 |

各行は32件のLPを処理する壁時計時間であって、1件のLPやPPOの1 timestepの時間ではない。
初回GPUの構築費15.822秒は表に含む。全sequence lifecycleはGPU25.074秒、CPU2.906秒。
ファイル読込時間は両者の表・sequence lifecycleから分離されている。
これは1試行で、32環境や7時刻を独立反復nとせず、信頼区間・有意差は算出していない。

前版n=3中央値のstep3/4はGPU1.722/1.485秒、係数更新0.997/0.881秒だった。
今回の短縮は時系列の開発比較であり、ランダム化比較による速度比ではない。
また今回のGPU最初の3時刻だけの別試行は20.902/0.734/0.607秒、全96件が認証された。

精度条件は元単位のprimal residual ≤ 1e-5、dual violation ≤ 1e-7、relative KKT gap ≤ 1e-7、
さらに独立したdirect dual gap ≤ 1e-7を維持。現在時刻のCPU参照解はGPUへ渡していない。

## ボトルネックと追加試験

step3の求解0.621秒の内訳は、factor0.249秒、三角求解と状態更新0.279秒、
certificate0.054秒（残余は他処理）。11回のfactorと22回の三角求解があり、CPU側の
basis再利用に比べてGPU側の反復線形代数が重い。係数更新だけをゼロにしてもCPU超えには届かない。

単一目的変数を最適box境界に固定し、目的関数0の補助可行性問題をGPUで解く試験も実装した。
補助問題の成功だけでは受理せず、元の目的・制約に対してy=0で独立認証する。
補助問題の失敗を元LPの不可能性と混同しない。4環境12件、32環境96件は元LP認証を通過した。

32環境では初回求解1.207秒と短くなったが、後続求解0.635/0.510秒、更新等込み0.803/0.639秒で、
通常の新経路より継続時は速くなっていない。初回構築込みも17.387秒を要した。
既定には採用せず、冷間始動の候補に留める。最初の4環境試行では結果保存時のPath型変換に
不備があり、保存できなかった。修正して再実行したr1だけを保存済み結果として採用した。

## テストと成果物

- 最終再実行（以下の追加検査と集計の比較契約を含む）: **1,526 passed、4 skipped、11 warnings、31.90秒**。
- 既存＋新規の回帰: 1,509 passed、4 skipped、11 warnings。skipは2枚目GPU必須の検査。
- その後追加した相殺support変化・非零固定値・除去zero-row検査を含む更新テスト: 14 passed。
- 補助LP候補の元LP認証、非最適・不可行・NaN・補助未受理の拒否: 6 passed。
- 警告は既存のSciPy/HiGHSオプションと意図的CSR構造変更に由来する。

Rawと集計:

- [GPU 32環境×7時刻](../results/pf_ipm_gpu_device_updates32_2to8_20260907.json)
- [CPU16 同一入力](../results/pf_ipm_cpu1151_device_compare32_2to8_20260907.json)
- [hash照合済み集計](../results/pf_ipm_device_compare_summary_20260907.json)
- [GPU 32環境×3時刻](../results/pf_ipm_gpu_device_updates32_234_20260907_r1.json)
- [補助box-face 4環境](../results/pf_ipm_box_face4_234_20260907_r1.json)
- [補助box-face 32環境](../results/pf_ipm_box_face32_234_20260907_r1.json)
- [回帰XML](../results/pf_ipm_device_update_regression_20260907.xml)
- [最終回帰XML](../results/pf_ipm_device_update_final_regression_20260907.xml)
- [境界ケースXML](../results/pf_ipm_device_update_edge_tests_20260907.xml)

## 次の優先順位

数値更新経路の安全性を維持し、反復線形代数の費用を主対象に移す。
Gondzio・Sobralは準Newton法によってIPMのfactor回数を減らす方向を報告している。
本実装では、旧factorを現在のNewton行列の正解とみなさず、前処理として利用し、
現在の全Newton残差を検証する案を小規模に評価する。再分解を減らしてもKrylov反復が増えれば
逆効果なので、総時間・factor数・三角求解数・失敗を同時に比較する。
[著者公開論文](https://www.maths.ed.ac.uk/~gondzio/reports/qnIPM.pdf)

併行する候補は固定buffer上のCUDA Graph化。ただしcuDSSのanalysisは別扱いであり、
実GEMの現在値更新・寿命・再認証を含めて検証するまで、toyのcapture成功を本番成功としない。
[NVIDIA cuDSS実行仕様](https://docs.nvidia.com/cuda/cudss/general.html)

どちらも本報告時点では未実装・未達。精度閾値を下げる、CPUを単スレッドに制限する、
同じLPを複製して見かけのバッチを増やす変更は行わない。
