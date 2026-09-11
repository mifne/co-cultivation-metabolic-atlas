from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'results/pfreud_value_audit_20260908'
def readcsv(name):return list(csv.DictReader((O/name).open(encoding='utf-8-sig')))
r=readcsv('summary.csv');c=readcsv('paired_comparisons.csv');integ=json.loads((O/'integrity.json').read_text())
def row(dt,arm,mode='separate'):
    return next(x for x in r if float(x['dt'])==dt and x['arm']==arm and x['mode']==mode and x['group'] in ('time_step','objective_sensitivity'))
def fmt(x):return f'{float(x):.6f}'
lines=['# P. freudenreichii の追加価値：再評価（2026-09-08）','',
'## 結論','',
'P. freudenreichii の乳酸→プロピオン酸変換とNS21による取り込みはモデル内で確認できた。しかし、総PHA量における3種の頑健な優位性は今回の検証では確認できない。生物学的機能の存在と、第三菌を加える工程上の利益を区別する。',
'',
'前回回答の「優位性を支持する結果ではない」は結論として慎重だが、旧20条件のうち乳酸0.25 mmol/L/h・Pf初期0.03 g/Lに約20%増加がある点を省略していた。本監査では旧全条件を同一乳酸供給の2種と照合し、この事後発見した候補を再計算した。成功例だけを残さず、陰性結果も全件保存した。','',
'## 設計とデータの位置づけ','',
'- 現行OR16、NS21、Pfモデルを使用。既存の教師データ、GEM、学習用収集プロセスは変更していない。',
'- 12時間、ゴム初期10 g/L、kLa=50/h、理想pH-stat=7。乳酸は各対照組の中で同じ速度・累積量。初期培地は対照間で完全一致を確認。',
'- 同一生産菌量対照：2種はOR16 0.5＋NS21 0.1 g/L、3種はこれにPf 0.03 g/Lを追加。',
'- 総菌体量一致対照：2種をOR16 0.525＋NS21 0.105 g/Lとし、両構成の総量を0.63 g/Lに揃える。生産菌量固定と総量固定は別の問いなので両方を提示する。',
'- 中心条件：乳酸0.25 mmol/L/h、初期NH4 0.05 mmol/L。dt=1、0.2、0.1 hを比較。',
'- 周辺条件：乳酸0.20/0.30 × 初期NH4 0.025/0.075、dt=0.2 h。統計的標本ではなく局所感度解析。',
'- 現在の協調LP方式でも中心条件の3対照を比較。協調最適化は菌構成によって目的の構成が変わるため、生理学的共生の証明には使わない。',
'- 計24ケース。実行前にdesign.jsonへ条件・主指標・ソースSHA256を保存。候補自体は旧結果からの事後選択であり、事前登録された独立発見とは呼ばない。',
'- 工学的スクリーニングは「両対照よりPHA 5%以上増、ゴム除去低下5%以内、全求解成功」。5%は暫定判断基準で、統計的有意差ではない。','',
'## 旧陽性候補の再現と刻み幅依存','',
'乳酸0.25 mmol/L/h、初期NH4 0.05 mmol/L。全てモデル出力。','',
'| 方式／dt | 2種：生産菌量固定 PHA g/L | 2種：総量一致 PHA g/L | 3種 PHA g/L | 3種の差：固定対照比 | 総量一致対照比 |','|---|---:|---:|---:|---:|---:|']
for mode,dt in [('separate',1.),('separate',.2),('separate',.1),('cooperative',.2)]:
    a,b,t=[row(dt,arm,mode) for arm in ['two_fixed','two_equal_total','three']]
    gain1=100*(float(t['pha_g_l'])/float(a['pha_g_l'])-1);gain2=100*(float(t['pha_g_l'])/float(b['pha_g_l'])-1)
    lines.append(f'| {mode} / {dt} h | {fmt(a["pha_g_l"])} | {fmt(b["pha_g_l"])} | {fmt(t["pha_g_l"])} | {gain1:+.2f}% | {gain2:+.2f}% |')
lines+=['','粗い1時間刻みの約20%増は再現できた。しかし刻みを細かくすると差の符号が変わる。現段階でこの陽性値を3種の優位性として採用できない。dt=0.2→0.1でも絶対PHA量は大幅に変わり、数値収束は未確認。いずれかの刻み幅を「正解」と選んではいけない。',
'',
'時間刻み感度の原因候補には、NH4 0.1 mMでの目的切替、酸素移動と細胞外ゴム分解の更新順序、段階的な流加がある。原因を単独に同定した結果ではない。モデル全体の数値収束評価を優先する必要がある。',
'', '## 周辺条件：全比較','',
'| 乳酸 mmol/L/h | 初期NH4 mmol/L | PHA差：生産菌量固定比 | PHA差：総量一致比 | 両対照に対する暫定基準 |','|---:|---:|---:|---:|---|']
for x in c:
    if x['group']=='local_sensitivity':lines.append(f'| {x["rate"]} | {x["nh4"]} | {float(x["pha_gain_pct_vs_two_fixed"]):+.2f}% | {float(x["pha_gain_pct_vs_two_equal_total"]):+.2f}% | {x["passes_both_controls"]} |')
lines+=['','全結果をsummary.csv、対照との差をpaired_comparisons.csvに保存。決定論的シミュレーションの時刻を独立反復として数えず、p値や有意差の星は付けていない。菌体量・供給・窒素を変えたことは生物学的反復を行ったことにはならない。',
'', '## 表現できるPfの価値','',
'中心条件、逐次dFBA、dt=0.1 hの12時間積算：','',
'| 指標 | mmol/L |','|---|---:|']
t=row(.1,'three')
for label,key in [('Pfの乳酸取り込み','helper_lactate_uptake'),('Pfのプロピオン酸分泌','helper_ppa_secretion'),('NS21のプロピオン酸取り込み','ns21_ppa_uptake')]:lines.append(f'| {label} | {fmt(t[key])} |')
lines+=['',
'これは「乳酸をプロピオン酸へ変換し、NS21へ代謝物を供給する補助機能」の計算上の証拠である。プロピオン酸がPHAの何割に入ったか、PHA炭素がゴム由来かは追跡していない。総PHA量の増加や、材料として望ましい3HV比を達成したという証拠ではない。',
'',
'価値の候補は、乳酸を含む副流の変換と供給組成の調整。ただしNS21はこのモデルで乳酸を直接利用できるため、Pfの添加は乳酸の競合消費にもなる。第3菌の追加価値は、2種への直接乳酸供給と、費用・炭素量を揃えた直接プロピオン酸供給を超えるかで判断すべきである。旧データの直接供給対照も保存しているが、今回24ケースで全供給方法の最適化を完了したわけではない。',
'',
'## 文献とモデルの隔たり','',
'- P. freudenreichiiの乳酸→プロピオン酸・酢酸変換は実験研究に支持される。ただし微好気下では、乳酸枯渇後にプロピオン酸を消費する方向へ移ることも報告されている。現在のPfは専用酸素交換を閉じた嫌気的表現型であり、好気的な同一槽で常にプロピオン酸供給源になるとは言えない。[Dank et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8360058/)',
'- NS21が天然ゴムからPHBVを蓄積し、窒素制限で蓄積が増えることは実験的な根拠がある。一方、今回のGEMにおける絶対速度・3HV mol%をこの論文が検証したわけではない。[Tamamura et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39216800/)',
'- B12分泌能力の可能性と、OR16/NS21がそれを必要とすることは別問題。受容側の必須性が現モデルで十分表現されていないため、B12による共生利益は加点しない。旧静的FVAでもB12分泌の下限は0であり、分泌保証ではない。',
'',
'## 計算の監査','',
f'- 完了ケース {integ["completed"]}/{integ["planned"]}。全ケースのsolver success=1：{integ["all_solver_success"]}。',
f'- 実行前後の対象ソース・GEM SHA256一致：{integ["source_hashes_unchanged"]}。',
'- 時系列行数、12時間到達、累積供給量、PHA重量換算、対照間の培地一致・初期総菌体量を検査。これは実験的妥当性や数値収束の認証ではない。',
'- 既存のPfモデル適合・代謝表現型テスト2件合格。',
'- 別途、逐次/協調×2種/3種の4回の監査で共有培地の負値切り捨て前の不足を計測。診断ファイルは ../pfreud_pool_audit_20260908/。',
'',
'## 次に価値を判定するために必要なこと','',
'1. 時間刻みに対するPHA・ゴム除去・順位の収束を確認し、必要に応じて目的切替・酸素移動等の積分を改善する。教師LPが正しく解けることと、培養動態が収束することを分けて扱う。',
'2. 数値条件を固定してから、2種・3種それぞれの培地供給を同じ資源・探索予算で最適化する。3種だけを最適化した比較で優位性を主張しない。',
'3. 総PHA、ゴム由来炭素の回収、供給費用、希望する3HV範囲を別々に評価する。望ましい組成は用途で決まり、3HVが高いことだけで高価値とは扱わない。',
'4. 機能が必要な条件を絞れた段階で、実測による較正と独立した2種/3種比較へ進む。',
'',
'## 成果物','',
'- design.json：実行条件とソース指紋。',
'- summary.csv：24計算ケースの指標。',
'- paired_comparisons.csv：各3種条件と2つの対照の差。',
'- legacy_all_helper_comparisons.csv：旧20条件中の全第3菌条件を再比較。',
'- pfreud_value_audit.png / .pdf：比較図。',
'- 各ケースのresult.json / trajectory.csv / worker.log：原結果。',
'- integrity.json：整合性検査。','']
(O/'REPORT_JA.md').write_text('\n'.join(lines),encoding='utf-8')
print(O/'REPORT_JA.md')

