# GNN＋GRU教師あり学習とGPU補正回数の検証

## 結論

教師あり学習、学習済みモデルの保存、別seedのGPU補正比較まで実施した。
今回のfull-vector予測方式ではGPU補正回数は減らず、GNN＋GRUは前回GPU解の再利用より
遅くなった。学習・実行できたことと、高速化できたことを区別する。
本番PPOのbackendは変更せず、現行の前回GPU解再利用を維持する。

## 学習条件

- 対象：OR16＋NS21＋P. freudenreichii、現行GEM fingerprintを実ファイルから再計算して一致確認。
- データ：`results/pf_gru_training16x60_20260905`、明示的な`training_reference`。
- 学習：12環境×60時刻＝720件、モデル選択：別4環境×60時刻＝240件。
- 対象LP：maxmin、6,734変数、5,051行。3段階LP全体の学習ではない。
- 入力：現在の全LPの係数・上下限・目的。変数/制約二部グラフ、hidden=16、message passing 2回。
- GNN：191,890パラメータ。GNN＋GRU：195,154パラメータ。
- 固定GEMのnode座標に束縛したembeddingを含む。異なるGEMや並べ替えにそのまま汎化するとは主張しない。
- train-only中心/scaleから全primal・row dualを直接出力。PCA・候補辞書は使用しない。
- GRUは環境別の入力履歴を使用。教師の前時刻fluxを入力しない。TBPTTは8時刻。
- 教師フラックス・dualへの損失に、物質収支/不等式違反・目的値の損失を追加。
- 教師生成は既存のオフラインCPU計算。今回のオンラインGPU比較にCPU教師解を渡していない。

初回は標準偏差（下限1e-3）で正規化し60epochを学習したが、trainで変動の小さいfluxが
開発教師誤差を過大に支配した。最初の開発速度試験ではstep3/4のバッチ分解ラウンドが
前解19、GNN35、GNN＋GRU46へ増加したため、この結果を保存して方針を改めた。

第2試験ではtrain-only RMS（下限1）で正規化し、元単位の最大行違反を含む損失を使用。
両モデルを160epoch学習した。チェックポイント選択は開発損失のみで、以下の評価seedとは分離。
選択epochはGNN130、GNN＋GRU90。第2試験の学習時間はそれぞれ184.1秒、197.4秒、
前処理を含む総時間395.6秒。PyTorch最大allocatedは約1.37 GiBであり、GPU全VRAM使用量ではない。

## GPU比較方法

評価は`pf_coverage_holdout4x120_20260905`のseed 20296001–20296004。
学習/モデル選択とのseed・LP problem hashの重複は0。評価ファイルからは入力のみを読み、
CPU reference x/yは読み込まない。学習に使ったGEMと評価GEMのfingerprint一致を確認。

各方式でstep2を同じ入力・cold GPU解法で解き、step3–8を比較した。
hot集計は4環境×6時刻＝24 LP/試行。3反復で順序を
前解→GNN→GNN＋GRU、GNN→GNN＋GRU→前解、GNN＋GRU→前解→GNNと変更した。
同じcold入力でも数値実行の反復数には差があるため、完全同一内部状態のpaired試験とは呼ばない。

前解方式と学習方式は、同じcurrent-cost centered restart（mu=1e-5）、同じGPU IPMを使用。
学習方式はprimal候補をDLPackでGPUに渡し、既存の厳密圧縮を通してrestartを生成する。
学習したrow dualも生候補の検査には使用するが、この比較のrestartではリセットする。
したがって「学習したdualを活かす全方式」の否定結果ではない。

最終判定は元LPのprimal残差1e-5、dual違反1e-7、相対KKT gap1e-7、
独立direct-dual gap1e-7を維持。CPU LP fallbackは0。
時間は入力の数値更新、グラフ準備、推論、restart、GPU求解、元LP検査、独立監査、
次時刻用状態のexportを含む。ファイル読み込み、モデル読み込み、初回構造準備、
初回cold solveはhot集計から除外し、rawでは別途記録した。

## 結果

時間と反復は3反復の中央値。補正反復/LPは各laneの認証された`accepted_iteration`から算出。

| 方式 | 補正反復/LP | バッチ分解ラウンド合計 | hot24 LP総時間 | 3反復の範囲 | 前解に対する速度比 |
|---|---:|---:|---:|---:|---:|
| 前回GPU解の再利用 | 7.46 | 48 | 1.645秒 | 1.589–1.662秒 | 1.000倍 |
| 学習済みGNN | 17.38 | 110 | 3.350秒 | 3.295–3.504秒 | 0.491倍 |
| 学習済みGNN＋GRU | 19.71 | 123 | 3.721秒 | 3.710–3.780秒 | 0.442倍 |

- 各方式hot72/72件が元LP認証を通過。初回を含めると各84/84件が合格。
- 生のGNN候補/GNN＋GRU候補はhot0/72件が直接認証。最終合格はGPU補正によるもの。
- hot元LP primal残差最大：前解8.927e-6、GNN9.037e-6、GNN＋GRU9.382e-6。
- GNN＋GRUのグラフ準備＋推論はhot6時刻合計0.092秒。3.721秒の約2.5%にとどまる。
- 補正前のGNN＋GRU候補の元LP primal残差（バッチ最大）は2.949–3.530で、1e-5の認証基準から遠い。
- バッチ分解ラウンドは「4環境をまとめた分解呼び出し」であり、個別LP分解総数とは異なる。

GNN＋GRUはこの条件で約2.26倍の時間を要した。推論コストを0にしても、補正回数の増加を
解消しない限り前解方式に届かない。半減・CPU16超えを達成したとは主張しない。

## 何が分かったか／次の変更候補

1. GPUでニューラル推論する処理そのものより、候補から認証解まで直す反復が律速。
2. この実装は履歴から全fluxを再予測する方式であり、直前の認証済みGPU fluxを直接入力していない。
   そのため、既に良い前回解を使う比較方式の利点を失っている。GRUを足すだけでは補えなかった。
3. 教師LPのフラックス分布への平均的な一致と、求解器にとって良い初期値は別の評価軸。
   maxminの非一意解について、内部fluxを一致させる損失だけを増やすべきではない。
4. 次は全fluxの置換ではなく、認証済み前回flux＋現在の制約残差から小さな修正を学習する
   residual型候補器を検討する。損失も「短いGPU補正後の元LP残差・必要反復」を含める。
   これは次の候補であり、今回実装・高速化済みとは扱わない。
5. この4環境で補正回数が増えたため、32環境・120時刻・PPOへ機械的に拡大しない。
   まず同じ精度の少数環境で前解方式を上回ることが必要。

本検証は保存したmaxmin入力のリプレイであり、予測fluxで培地を更新する閉ループdFBAではない。
PHA終点誤差、共存安定性、PPO報酬、全学習時間の改善は検証していない。

## 成果物

- 初回学習：`results/pf_supervised_gnn_gru_maxmin_20260907/`
- 最終学習済みモデル：`results/pf_supervised_gnn_gru_physics_20260907/gnn.pt`、`gnn_gru.pt`
- 学習履歴・モデルhash：同ディレクトリの`manifest.json`、実行ソース：`sources/`
- 初回比較：`results/pf_supervised_graph_pilot4_20260907.json`
- 最終raw・ソースsnapshot：`results/pf_supervised_graph_heldout4x7_r3_20260907.json`
- 集計：`results/pf_supervised_graph_heldout_summary_20260907.json`
- 回帰テスト：`results/pf_supervised_graph_regression_20260907.xml`

再実行時は別の出力名を指定する。既存成果物を上書きしない。

## 実装の回帰検証

関連する数値求解・グラフ・GRU履歴・新しい学習モデル・GPU restartを含め、
**1664 passed、4 skipped、11 warnings（46.05秒）**。
checkpointの保存/復元、GEM/graph identity不一致の拒否、GNN/GRUへの勾配、
現在LP hash・世代の照合、未認証候補が自動承認されないことも検査した。
これはコードの回帰結果であり、上記の高速化失敗を覆す指標ではない。
