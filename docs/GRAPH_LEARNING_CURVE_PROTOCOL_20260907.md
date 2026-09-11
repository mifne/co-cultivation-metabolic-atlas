# GNN＋GRU：データ拡大と512軌道の十分性評価

## 目的と規模

以前の720件は12軌道×60時刻のmaxmin LPであり、720個の独立した培養条件ではない。
新規データは32、64、128、256、512、1024軌道の包含関係を持つ学習曲線で評価する。
512は上限ではなく比較点であり、1024を超える収集も同じコマンドで指定できる。

| 学習軌道数 | 1段階あたりのLP数（120時刻） | 3段階合計 |
| ---: | ---: | ---: |
| 32 | 3,840 | 11,520 |
| 64 | 7,680 | 23,040 |
| 128 | 15,360 | 46,080 |
| 256 | 30,720 | 92,160 |
| 512 | 61,440 | 184,320 |
| 1024 | 122,880 | 368,640 |

これは計画上の件数であり、収集済み件数ではない。実績は
`results/pf_graph_coverage_20260907/catalog.json`のcompleted shardのみで数える。
失敗・途中軌道、短いsmoke-test、旧データとの重複を加算しない。

## 条件の広がり

現行のOR16、NS21、P. freudenreichiiのGEMを固定する。
初期菌体量は種別に既定値の0.75–1.25倍、初期アンモニウムは0.5–2倍。
既存5次元の正規化actionを0.05–0.95の範囲で変化させる。
一様乱数、10時刻ごとの段階変化、滑らかなランダムウォーク、15時刻ごとのパルスを均等に含める。
これらは仮想スクリーニング範囲であり、実測で校正された生理学的範囲ではない。
GEMの反応・遺伝子や化学量論を都合よく変更しない。
ゼロ供給や極端な初期状態など、この範囲外への保証は別途外挿試験を必要とする。

学習seedは71000000から、モデル選択は72000000から、最終試験は73000000からの
別系列に固定する。同一系列の追加分は既存prefixを変えず増やす。
最初はモデル選択16軌道、最終試験16軌道を固定する計画。
モデル選択の信頼区間が広い場合には、学習軌道数だけでなく評価条件数も増やして比較をやり直す。
各LPはCPUで求解後、元制約の証明条件と保存→読込の整合性を確認する。
native crashを含む未完了shardは採用しない。同じseedを新しいretryディレクトリで再実行し、
元ログと失敗履歴を保持する。都合のよい別seedへの置換はしない。

## 実行例

WSLのプロジェクトルートから実行する。既存の完了shardはハッシュ照合して再利用する。
収集ではCPUを教師生成に使うが、その時間をオンラインGPU高速化の測定へ混ぜない。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-cuopt-26.8/bin/python scripts/collect_graph_learning_curve.py \
  --output results/pf_graph_coverage_20260907 --count 32 --workers 1 --retry-failed --scipy-teacher
```

同じコマンドのcountを64、128、…、1024へ増やせる。モデル選択・最終試験の収集は
`--split selection --count 16`、`--split test --count 16`を指定する。
同一catalogへの書込はロックで排他し、同時起動しない。

CPU highspy経路でnative memory errorが、並列数1およびcold再構築でも発生した。
`--scipy-teacher`は別のSciPy/HiGHS cold経路を直列に使う回避策であり、根本原因の修正ではない。
最初の完了8軌道はpersistent経路、以降は各manifestのteacher_strategyを参照する。
どの経路も同じ元LP認証を必要とするが、非一意最適解の選択まで一致するとは仮定しない。
教師方式に由来する差が見られた場合は、統一した方式による別コレクションで学習曲線を確認する。

```bash
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-cuopt-26.8/bin/python scripts/train_graph_learning_curve.py \
  --collection results/pf_graph_coverage_20260907 \
  --output results/graph_curve_n32_seed1 --train-count 32 --selection-count 16 \
  --batch-size 4 --epochs 160 --seed 20294907 --stage maxmin
```

trainerは軌道バッチ単位でCPU/GPUへ読み込み、全件をVRAMへ常駐させない。
学習データのみから正規化統計を計算し、testラベルへのアクセスは拒否する。
全データを毎epoch読み直すI/O費用は残っている。総学習時間とgraph staging時間を分離記録する。
最初は既存GNN/GNN＋GRUの構造・損失を固定し、データ量の効果を切り分ける。
160 epochは収束の保証ではない。validation曲線を確認し、必要なら学習予算を増やす。
少なくとも3学習seedで再現する。`--max-updates`による更新回数固定の別実験も行い、
データ増加と計算予算増加を区別する。更新回数固定で全条件を訪問できなかった場合は明記する。

## 速度・精度の評価

既存の前解warm start、GNN、GNN＋GRUを同じLP、順序、バッチ幅、精度で比較する。
推論だけではなく、入力更新＋推論＋GPU補正＋元LP検査を含む総時間を測る。
補正反復数、補正不要率、証明成功率、CPUへのLP fallbackも記録する。
同じ独立軌道群を使い、各計測セルで少なくとも3反復し、中央値を比較する。
GPUバッチの秒数を環境数で割って、独立した実測時間を捏造してはならない。
バッチを独立単位として使う場合は、重複しない軌道群を複数用意する。

`scripts/evaluate_graph_learning_curve.py`は学習seedと独立評価ブロックの
二方向cluster bootstrapで95%区間を計算する。入力JSONの形式は以下。
数値は形式説明用であり、測定結果ではない。

```json
{
  "fixed_comparison_id": "architecture-data-split-precision-baseline-version",
  "timing_unit": "independent_trajectory_batch",
  "sizes": [{
    "train_count": 32,
    "converged": false,
    "measurements": [{
      "training_seed": 20294907,
      "block_id": "selection-group-0",
      "total_seconds": 1.0,
      "baseline_seconds": 2.0,
      "timing_repeats": 3,
      "all_accuracy_gates_passed": false
    }]
  }]
}
```

実行には全学習seed×全評価ブロックの完全な組合せと3以上の独立評価ブロックが必要。
全サイズで同じbaseline、学習seedと評価ブロックを使う。失敗条件を除外しない。
`all_accuracy_gates_passed`は実測に基づく申告であり、この集計器がLPを再認証するわけではない。
maxminの入力リプレイと、3段階・120時刻の閉ループdFBAを別のcomparison_idにする。
最終採用には後者でPHA・菌体量・培地状態の誤差と安定性も確認する。

512の十分性候補を認めるには1024も実測する。速度目標は同じ基準方式に対する
2倍以上（時間半減）の95%区間下限、精度維持、学習の収束を要求する。
連続2回のデータ増量で時間短縮率の95%区間上限が2%以下なら、条件付きの飽和候補とする。
この2%は事前に置く工学的判断基準であって、普遍的な必要サンプル数を証明するものではない。
誤差が残る、速度目標未達、区間が広い場合は増量または損失・補正方式を見直す。
最終試験は選択した構成を固定して一度開き、結果を見て調整した場合は新しい最終試験を用意する。

## 現時点の限界

小規模の新trainer動作確認は学習成功や高速化の証拠ではない。
前回のGNN＋GRUは前解より遅かった。データ増量で反復数が減るかは未確定である。
512/1024の収集・学習・速度比較を完了したとは扱わない。
