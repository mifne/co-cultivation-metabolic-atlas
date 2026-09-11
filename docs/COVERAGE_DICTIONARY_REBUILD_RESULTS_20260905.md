# GPU認証辞書の再構築結果

対象：OR16＋NS21＋P. freudenreichii の3段階FBAのうちmaxmin段階。GEM、反応式、培地境界、目的関数、終点判定基準は変更していない。

## 結論と評価範囲

既存学習データから有効な基底を追加し、複数の正解候補を教師とする小型MLPで候補を選ぶ方式を実装した。辞書には96候補を保持するが、オンラインで証明するのは指定した1～4候補であり、全96候補の総当たりではない。候補順位そのものは全候補について計算する。

新規4軌跡×120stepの**入力trace評価**で、4候補の認証率は旧15.0%から53.96%へ改善した。480行は4本の相関した時系列であり、独立標本480として信頼区間を算出していない。PPO全体の速度、完全GPU化、実培養への予測精度はこの値から主張できない。

| 辞書・選択方式 | 学習480時点：K1 | 学習：K4 | 未使用4軌跡480時点：K1 | 未使用：K4 |
|---|---:|---:|---:|---:|
| 旧44候補・最近傍 | 80/480 (16.67%) | 119/480 (24.79%) | 31/480 (6.46%) | 72/480 (15.00%) |
| 新96候補・最近傍 | 140/480 (29.17%) | 239/480 (49.79%) | 41/480 (8.54%) | 130/480 (27.08%) |
| 新96候補・multi-label MLP | 375/480 (78.13%) | 393/480 (81.88%) | 193/480 (40.21%) | 259/480 (53.96%) |

表はstateless順位であり、閉ループの前回認証候補の優先順位付けを含まない。旧辞書の全候補coverageは学習182/480・未使用132/480、新辞書は学習393/480・未使用292/480。新しい入力で残る失敗は**候補不足188行、K4選択失敗33行**で、今後もこの二つを区別する。

## 再構築した内容

1. 既存4軌跡×120stepのcacheについて、入力chunk・元bank・GEM・collector sourceのSHA、float64 schema、尺度、順序を検査。
2. seedごとに独立したpersistent HiGHSで480 LPを元座標で再解し、元LP証明を全件確認。424種類の完全基底statusを採取。
3. 未被覆状態を補う候補144件を順にcompact投影。各候補は自己の学習状態と全480状態でGPU証明を検査。旧44を含む候補poolで476/480を被覆。
4. 旧44候補はバイト単位で保持し、メモリ上限付き決定的greedyで新52件を選定（総96）。数学的な最小集合の証明ではない。aggregate52候補・exchange44候補はファイルを変更せず継承。
5. 線形/hidden32 MLPを4-foldのtrajectory分割で内部比較（K4 57.29%/61.04%）。MLPを選び、同一設定で全training行へ再fit。全候補不合格の行は全negativeとし、未認証の「一番ましな候補」を正解にしない。
6. online推論はNumPy/CuPyのFP32、元LP証明は従来のFP64。実artifactのCPU/GPU top4は全480行で一致。CPU対照にも同一routerを与え、既存有効warmbasisがあるときは余分な再選択をしない。

主残差≤1e−5、双対違反≤1e−7、相対KKT gap≤1e−7を維持。family guard、全候補出力の有限性、非負の残差も検査する。rankは証明ではなく、認証に落ちれば正確なCPU経路を使う。現在のCPU参照解をGPU提案へ流用していない。

## 費用と速度の区別

再構築155.87秒には元LP再解・投影・GPUによる全training coverage評価・I/Oを含む。投影配列208,378,720 bytes（約198.73 MiB）は保存配列量で、temporary・packed weights・graph cacheを含むGPUピークVRAMではない。全basis逆行列は作成していない。

4環境×3stepのsmokeは全LP/終点ゲート合格、PHA・菌体量・PHV分率の終点差0。CPU2.096秒、hybrid6.131秒で、ここでは高速化していない。小規模・初回captureを含むこの測定を、認証率の改善だけで速度成功と言い換えない。32環境の総wall比較は別記する。

### 32環境の閉ループ総時間

同じ新辞書・MLPを与えたpersistent CPU HiGHS（4 workers、各LP 1 thread）が対照。CPU/GPUで同一のfrozen入力・pipeline・action・seedを使用。各組32環境×8step、2組の実行順をCPU-first/GPU-firstと交互にした。両方式とも環境作成を除いた計算wallで、初回のgraph構築は計算wall内、router初期化時間は別fieldにも記録。

| GPUで検査する候補 | CPU合計 (s) | Hybrid合計 (s) | 時間の増加 | GPU認証 / maxmin LP | 未開始CPU取消 | 重複CPU解 |
|---|---:|---:|---:|---:|---:|---:|
| 4候補 | 58.9708 | 61.6208 | +4.49% | 208/512 | 76 | 132 |
| 1候補 | 57.9005 | 59.1804 | +2.21% | 93/512 | 46 | 47 |

全元LP証明・完走・全worker join・計数整合を独立集計で再確認し、PHA・菌体量・PHV分率の終点誤差は全条件で0。K1の2組目はCPU28.4905秒・hybrid28.5353秒（+0.16%）だが、これを一般的な速度優位としない。2組の記述値で信頼区間は付けない。

K4では208件のGPU認証のうち132件はCPUも計算済みで、実際に省けたCPU LPは76件（全1,536 LPの4.95%）に留まった。GPU提案・証明区間は2組で計5.37秒、うちhost正規化/入力準備の記録値が計1.85秒。これらはCPU実行と重なる区間であり、そのまま総wallへ足してはいけない。

非重複pipeline会計のmaxmin区間はCPU計9.41秒・hybrid計10.44秒。その後のCPU完了待ちはCPU計32.04秒・hybrid計32.52秒で、後段処理が大きく残る。辞書の認証率改善だけではCPU律速の解消や全3段階GPU化にはならない。採否は**速度上の昇格を保留**とし、安全な既定PPOは維持する。

## 再現性と制限

- 辞書・routerの学習seedは20287101–04。旧44の親候補と他stageの由来seedはmanifestで維持。新規入力評価seedは20296001–04で、学習へは混ぜていない。
- cacheには当時の全固定行、row ID/action、環境/LP生成器の完全なsnapshotはない。元collectorが固定行差≤2e−12を省略しており、現在のhashは歴史的なraw LPのbit-exact再現を遡及保証するものではない。
- 学習データは4本のuniform-action軌跡。PPOが訪れる全状態や栄養枯渇境界の網羅は未保証。
- 新規120step入力評価では各時点がCPU軌跡由来。GPUによる状態更新まで連鎖させた120step closed-loopとは区別する。
- 96候補の選定は評価済みpool内のgreedyで、全424基底の最適集合ではない。自己のsource行が既に被覆された候補の一部は未評価であり、未知領域に広く通用する候補を取り逃す余地がある。
- 既定PPOの辞書は変更せず、新artifactはopt-in。過去の結果を削除・上書きしていない。

## 証拠ファイル

- `results/pf_coverage_baseline480_20260905/report.json`
- `results/pf_coverage_rebuilt96_20260905/manifest.json` / `rebuild_report.json` / `collector_report.json`
- `results/pf_coverage_router96_20260905/report.json`
- `results/pf_coverage_holdout4x120_20260905/manifest.json`
- `results/pf_coverage_holdout_comparison_20260905/report.json`
- `results/pf_coverage_router_smoke4x3_20260905.json`
- `results/pf_coverage96_k4_32x8_20260905.json`
- `results/pf_coverage96_k1_32x8_20260905.json`
- `results/pf_coverage_rebuilt96_20260905/provenance_audit_addendum.json`

### 監査後の補記

最初の新bank manifestには、旧bankの`total_seconds`/`cpu_lp_calls`が継承されて残るメタデータ上の不備があった。実費用は`reconstruction_seconds`/`reconstruction_cpu_lp_calls`および`rebuild_report.json`を使う。現在のbuilderは旧値を明示的にrenameし、新規値を保存するよう修正した。評価済みbank/routerのSHAを変えず、補記JSONを新規に保存した。

中間status/poolのhashと内部fold evidenceも監査補記で固定し、継承140候補ファイル・選定96候補・480元LP証明・424status・coverage/MLP集計の一致を確認した。ただし補記の中間hashは監査時点で追加したもので、構築前からの改変検出を遡及保証しない。今後のbuilder/trainは中間hash・候補順・bank/coverage/feature行対応を構築時に検証・保存する。元LPや学習重みの内容は監査で変更していない。

### 最終テストと運用状態

CPU/GPU・元LP証明・従来hybrid・pipeline・取消・新coverage/loader/bindingの884テストが合格。別processで時系列/縮約・router学習の159テストが合格し、重複しない合計**1,043件**を確認した。広域回帰で見つかった任意router未設定時の旧経路互換性2点を修正後、全884件を再実行した。数値閾値を緩めてテストを通したわけではない。

実行済みのsmoke・K4・K1結果はそれぞれ実行時のsource snapshotを保持し、独立集計で再検証した。全計算・テストジョブは終了。今回の成果は、辞書再構築と未知入力での少数候補認証率改善まで。CPUを上回る総速度、120stepの新方式閉ループ、PPO全体の高速化、全3段階のGPU内完結は未達または未検証として明示的に残す。

設計根拠：[SurfinFBA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007786)の有効基底再利用、[Akbari・Barton](https://arxiv.org/abs/1802.02567)の退化・多重最適解を伴うparametric FBA。本系は行列の一部も変わるため、論文の固定行列条件や速度倍率をそのまま転用せず、低rank更新と現在LPの証明を併用した。
