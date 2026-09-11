# 酸素と窒素配分を内部時間刻みで解くCPU参照計算

`src.resolved_dfba.ResolvedDFBASimulator` を追加した。旧 `dFBASimulator` と保存済みLPトレースの定義は変更していない。新経路の数値バージョンは `resolved_oxygen_nh4_v1`。計算方法に加えてNS21の増殖・蓄積配分の仮定も変更しているため、旧結果との差をすべて数値誤差の修正効果とは解釈しない。

## 資料から取り入れた内容

齋藤氏の進捗報告は、phaCABを導入したRhodococcus sp. RDE2、150 mMピルビン酸、30℃、150 rpmの培養。C/N 40、60、80、100でPHA含有率は10.8、18.7、21.7、22.4%、PHA濃度は444.1、493.3、442.0、433.4 mg/L。窒素制限と増殖・蓄積のトレードオフ、および細胞中含有率と体積あたり生産量を区別する必要性の参考とした。

現在のOR16・NS21・Pf共培養の速度パラメータをこの数値へ合わせてはいない。NH4時系列、kLa、通気量・液量、C/Nの定義が不足し、測定間隔も24時間なので秒〜分の動態やNH4閾値を同定できない。DO図を150 rpmからkLaへ変換していない。資料の定常期は7〜8**日**で、今回の比較は12**時間**である。

## 変更点

- `dt` は制御・観測間隔、`max_internal_dt` は内部の最大積分間隔。一定供給量を内部各刻みに割り当ててLPを再計算する。
- 酸素は `dC/dt = kLa*(Cs-C)-OUR` の定数OUR区間の解析解で更新する。ゴム分解と菌体の消費を同じ酸素予算から引く。酸素予算を溶存濃度として取り込み速度式に渡さない。
- 取り込み速度式に使うDOは無消費区間平均の予測値。この近似と代謝速度・菌体量の区間内変化の誤差は残るため、内部刻みの収束試験は別途必要。
- NH4=0.1で生産を完全にON/OFFする代わりに、まずNS21の増殖可能速度を求め、`NH4/(K_N+NH4)` の割合を下限として確保し、残りの自由度でPHA質量を最大化する。両者の配分を連続化した作業仮説であり、実測された制御則ではない。既定の `K_N=0.1 mmol/L` は旧閾値を参考に置いた暫定値。
- 酸素収支、連続供給量、内部計算回数、制御回数、配分パラメータと未校正フラグを診断情報へ出す。

## 実行

WSL内のプロジェクトディレクトリで、未使用の出力先を指定する。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
.venv-cuopt-26.8/bin/python scripts/simulate_resolved_dfba.py \
  --output results/my_resolved_run --arm three --hours 12 \
  --control-dt 1 --internal-dt 0.025 --feed-rate 0.25
```

この供給は乳酸0.25 mmol/L/hを連続添加する。`--arm two_fixed` は生産菌の接種量一致、`--arm two_equal_total` は総接種量一致の2種対照。

Pythonからは以下のインターフェースを利用する。

```python
sim = ResolvedDFBASimulator(
    models=models, initial_biomass=biomass, initial_metabolites=medium,
    dt=1.0, max_internal_dt=0.025, solver_backend='highs',
)
sim.step({}, {}, dynamic_kla=50.,
         feed_rates_mmol_l_h={'lac__L_e': 0.25})
```

旧 `nutrient_supplementation` の `helper_lactate` 等は1回の制御呼び出し冒頭のボーラス量のまま。その量を内部刻みごとに重複添加しない。既存の `coexistence_feed_rate` は時間あたり定義を保つ。外部から培地濃度へ直接足す操作もボーラスなので、連続供給としての時間刻み比較には使わない。

## 適用範囲と既存データ

このバージョンはHiGHS/GLPKのseparate dFBA参照計算用。cooperative、joint、既存のGPUサロゲート、frozen LP収集には接続していない。これらを指定した場合は例外を出し、旧モデルと新モデルの混在を防ぐ。追加したNS21増殖可能量のLPと、内部刻み増加による計算費用がある。

既存の512軌道収集を自動再開せず、その生物モデル・LPラベル・閾値も変更していない。旧データの個々のLP解の有効性と、新しい動態・状態分布をカバーするかは別問題であり、新しい動態の教師軌道として無検証に混ぜない。

初期菌比・取り込み速度・K_N・kLaの生物学的校正、長時間安定性、3種の生産優位は、この数値計算修正だけでは立証できない。

検証の詳細は `results/resolved_dynamics_20260908/REPORT_JA.md` を参照。
