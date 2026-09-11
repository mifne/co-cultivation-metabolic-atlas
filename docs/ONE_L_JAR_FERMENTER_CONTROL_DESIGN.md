# 1 Lジャーファーメンター・4栄養ポンプ＋酸塩基2ポンプ制御設計

## 結論

サンプルを手動採取するため、4本すべてを給餌用として利用する。ただし4本を常時運転する
のではなく、増殖相だけ使う液、PHA相で止める液、菌体低下時だけ使う液に分ける。pH用2本は
装置PIDの内側ループとし、RLは酸・塩基ポンプを直接操作しない。初回は温度30°C固定、攪拌は
固定またはDO cascadeとし、温度までRLへ渡さない。

## 6ポンプの割当て

| 系統 | 溶液 | 初期使用 | PHA相 | 判断 |
|---|---|:---:|:---:|---|
| 栄養P1 | OR16支援：コハク酸二Na | 使用 | 低流量または停止 | rubber/Lcp活性とOR16菌体で制御 |
| 栄養P2 | NS21支援：glutamate/MSG | 使用 | 停止 | N源を切ってPHAへ移行 |
| 栄養P3 | LP支援：D-mannitol | 使用 | パルス | LP菌体と酸生成で制御 |
| 栄養P4 | 共通：yeast extract | 低流量で使用 | 停止 | AA・vitamin補完、過剰Nを避ける |
| pH base | 2 M NaOHを初期候補 | PID | PID | acid productionを中和 |
| pH acid | 1 M HClを初期候補 | 原則待機 | 原則待機 | overshoot時のみ |

NaOH/HCl濃度は装置メーカー、容器材質、施設SOPに従う。モデルでは48 hに201.9 mmol/Lの
塩基需要が出ており、2 M NaOHなら約101 mL/Lに相当する。実測でも高い場合、液量への影響を
確認したうえで4 M相当へ濃縮するか、初期液量を下げる。高濃度アルカリは局所pH、発熱、腐食、
作業安全上のリスクがあるため、単純に濃くすればよいわけではない。

## 種別液の初期組成

| 液 | Stock例 | 1 L培養液の初期上限 | Stock供給量 |
|---|---:|---:|---:|
| P1 succinate | 無水コハク酸二Na 100 g/L | 10 mM = 1.62 g/L | 16.2 mL |
| P2 glutamate | MSG 100 g/L | 10 mM ≈ 1.87 g/L | 約18.7 mL |
| P3 mannitol | 200 g/L | 10 mM = 1.82 g/L | 9.1 mL |
| P4 yeast extract | 100 g/L | 累積1 g/Lから開始 | 10 mL |

最大候補をすべて入れても栄養feedは約54 mL/Lである。最初は5 mM・YE 0.5 g/Lから始め、
約27 mL/Lに抑える。無機塩、trace metals、phosphate、vitaminsのうち安定な成分、天然ゴムは
initial basal mediumへ入れ、ポンプを使わない。

1 Lが「最大容器容量」か「推奨working volume」かを装置仕様で確認する。最大容器容量なら
初期液量は600–750 mL程度が必要になる場合があり、1 L working volumeが保証される容器なら
850–900 mLから開始し、feed、滴定液、手動採取を液量収支へ入れる。

## 運転相

### Phase A: 立上げ・増殖

- 温度30°C、pH 6.5を初期条件とする。
- P1–P4は低流量またはパルスで供給する。
- P2/P4から過剰窒素を入れず、NH4、glutamate、菌体量を確認する。
- OR16/NS21のrubber分解酵素は酸素を必要とするため、DOを枯渇させない。

### Phase B: PHA蓄積

- NH4 < 0.1 mmol/Lまたは実験で定めた切替値になったらP2とP4を停止する。
- P1はOR16維持に必要な最小量、P3はLPが低下したときだけパルス供給する。
- pH PIDと攪拌/DO制御は継続する。
- NS21菌体、PHA、NH4、rubber fragmentを確認する。

### Phase C: 保持・採取

- 全栄養feedを止める対照も置く。
- サンプルは手動採取し、採取量と時刻を必ずログへ入れる。
- 1回5 mLを6回採ると30 mL減るため、濃度だけでなく液量補正を行う。

## pH制御

- 初回setpointは6.5、dead bandは装置が安定して扱える±0.05–0.1を候補とする。
- L. plantarum標準MRSはpH 6.2–6.5であり、OR16/NS21は30°Cのminimal mediumで報告される。
- acid pumpは通常待機とし、base overshootだけを補正する。酸・塩基を交互に大量投入する
  huntingが起きる場合はPID gainと攪拌を先に調整する。
- RLの観測にはpHだけでなく、累積base/acid量と瞬時滴定速度を入れる。これはacid productionと
  残り液量の代理指標になる。

## 攪拌

最初からRPMを自由なRL操作にせず、次の順で校正する。

1. 水またはbasal mediumで200、300、400、500、600 rpmのkLaをgassing-out法等で測る。
2. rubber添加後に同じ測定を行い、粘度・粒子分散による変化を確認する。
3. DO sensorがある場合はDO 20–40%を初期setpointとし、200–600 rpmのcascadeを使う。
4. DO sensorがない場合は、沈降せず過剰なfoamを生じないRPMを固定し、まず300 rpm付近から
   段階試験する。

現在のプログラムはRL actionを0–200の仮想kLaへ直結しており、実RPMとの校正がない。このまま
では実装置へ転送できないため、`kLa = f(RPM, airflow, volume, rubber load)`を実測して置き換える。

## 温度

初回は30°C固定とする。OR16およびNS21の報告培養温度が30°Cで、L. plantarumも30°Cで
培養可能である。温度は応答が遅く、3種の増殖率・溶存酸素・pH電極挙動を同時に変えるため、
最初のRL actionへは含めない。

30°Cで168 hの対照が成立した後に、28–33°Cの狭い範囲を別実験で調べる。温度をRLへ追加する
場合も、変化速度制限と最低保持時間を設ける。

## RLへ渡す操作

初期RLは5操作に留める。

| action | 実機操作 | Phase Bの制約 |
|---|---|---|
| a0 | P1 succinate flow | 上限制限 |
| a1 | P2 glutamate flow | 強制0 |
| a2 | P3 mannitol flow | LP低下時のみ |
| a3 | P4 yeast extract flow | 強制0 |
| a4 | DO setpointまたは攪拌setpoint | rate limit付き |

pHは装置PID、温度は30°C固定とする。将来はRLがpH setpointを狭い範囲で変更してもよいが、
酸・塩基ポンプの直接duty操作は安全制約の外側へ出さない。

観測には3種菌体、pH、DO、RPM、温度、NH4、glutamate、mannitol、yeast-extract累積量、
base/acid累積量、working volume、PHA、rubber/fragmentを含める。

## プログラム側の未解決点

- feedは現在「濃度を加算」しており、stock流量、希釈、液量増加を計算していない。
- 手動samplingによる液量減少がない。
- pH-statは監査用の理想制御で、NaOH/HCl量とイオン蓄積を物質収支へ入れていない。
- temperature変数と温度依存の増殖・輸送係数がない。
- agitation actionは実RPMではなく未校正kLaである。
- maltotriose/putrescineをsuccinate/glutamateへ変更する実機profileが未実装である。

したがって、実機データでstock濃度、pump mL/min、kLa、滴定量を校正してから、旧モデルを壊さない
`one_l_jar` profileとして実装する。

## コスト

10 mM succinate、10 mM glutamate、10 mM mannitol、YE 1 g/Lで主要栄養原料は概算68円/L、
5 mM・YE 0.5 g/Lなら概算34円/Lである。pH titrant、basal salts、rubber、税、送料は別途。

- コハク酸二Na: https://labchem-wako.fujifilm.com/jp/product/detail/W01W0104-2813.html
- L-glutamic acid: https://labchem-wako.fujifilm.com/jp/product/detail/W01W0107-0050.html
- D-mannitol: https://www.yone-yama.co.jp/shiyaku/search/shosai-04572.html
- Yeast extract: https://labchem-wako.fujifilm.com/jp/product_data/docs/03694014_pamphlet.pdf

## 文献根拠

- OR16/Lcp: https://pmc.ncbi.nlm.nih.gov/articles/PMC7413915/
- NS21/glutamate medium: https://www.tandfonline.com/doi/abs/10.1080/09168451.2016.1263147
- L. plantarum requirements: https://pmc.ncbi.nlm.nih.gov/articles/PMC1287688/
- L. plantarum MRS: https://mediadive.dsmz.de/medium/11?ccno=DSM+20174
- PHA two-stage cultivation: https://pmc.ncbi.nlm.nih.gov/articles/PMC5597195/
