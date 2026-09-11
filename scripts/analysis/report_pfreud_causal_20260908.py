"""Read-only aggregation of causal and convergence diagnostics."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/pfreud_causal_20260908'
def read(path): return json.loads(path.read_text())
def writecsv(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
for folder in ['pfreud_causal_20260908','pfreud_refine_20260908']:
    for path in sorted((ROOT/'results'/folder).glob('*/result.json')):
        d=read(path);c=d['case']
        if c.get('force'):continue
        rows.append(dict(dt=c['dt'],arm=c['arm'],profile=c['profile'],pha=d['pha'],phv_fraction=d['phv_fraction'],rubber_removed=d['rubber_removed'],phase_hours=d['phase_hours'],polymer_o2=d['polymer_o2'],o2_transferred=d['o2_transferred'],helper_nh4_uptake=d['helper_nh4_uptake'],helper_ppa_secretion=d['helper_ppa_secretion'],ns21_ppa_uptake=d['ns21_ppa_uptake'],solver_success=d['diagnostics']['solve_success_rate'],source=str(path.relative_to(ROOT))))
old=[]
for dt in [1.,.2,.1]:
    for arm in ['two_fixed','two_equal_total','three']:
        path=ROOT/f'results/pfreud_value_audit_20260908/core_dt{dt}_{arm}/trajectory.csv'
        if path.exists():
            d=list(csv.DictReader(path.open()))[-1]
            old.append(dict(dt=dt,arm=arm,profile='continuous',pha=float(d['pha_g_l']),phv_fraction=float(d['phv_mol_fraction'])))
allrows=old+rows
pairs=[]
for dt,profile in sorted(set((r['dt'],r['profile']) for r in allrows),reverse=True):
    group={r['arm']:r for r in allrows if r['dt']==dt and r['profile']==profile}
    if len(group)!=3:continue
    pairs.append(dict(dt=dt,profile=profile,two_fixed=group['two_fixed']['pha'],two_equal_total=group['two_equal_total']['pha'],three=group['three']['pha'],gain_fixed_pct=100*(group['three']['pha']/group['two_fixed']['pha']-1),gain_equal_pct=100*(group['three']['pha']/group['two_equal_total']['pha']-1),three_phv_fraction=group['three']['phv_fraction']))
writecsv(OUT/'causal_runs.csv',rows);writecsv(OUT/'resolution_and_feed.csv',pairs)
phase=[]
for path in sorted((ROOT/'results/pfreud_causal_phase_retry_20260908').glob('*/result.json')):
    d=read(path);phase.append(dict(arm=d['case']['arm'],schedule=d['case']['force'],pha=d['pha'],phase_hours=d['phase_hours']))
writecsv(OUT/'phase_counterfactual.csv',phase)
v={(d['arm'],d['schedule']):d['pha'] for d in phase}
a,b,c,d=[v[k] for k in [('two_fixed','two_fixed'),('two_fixed','three'),('three','two_fixed'),('three','three')]]
total=d-a;schedule=((b-a)+(d-c))/2;species=((c-a)+(d-b))/2
decomp=dict(total_difference_g_l=total,schedule_component_g_l=schedule,species_component_g_l=species,schedule_share_percent=100*schedule/total,interpretation='Symmetric 2x2 counterfactual decomposition; not a unique biological causal fraction.')
(OUT/'phase_decomposition.json').write_text(json.dumps(decomp,indent=2))
convergence=[]
for arm in ['two_fixed','two_equal_total','three']:
    rs=sorted([r for r in allrows if r['arm']==arm and r['profile']=='continuous'],key=lambda r:r['dt'],reverse=True)
    for coarse,fine in zip(rs,rs[1:]):
        convergence.append(dict(arm=arm,coarse_dt=coarse['dt'],fine_dt=fine['dt'],pha_change_pct=100*(fine['pha']/coarse['pha']-1),phv_change_pp=100*(fine['phv_fraction']-coarse['phv_fraction'])))
writecsv(OUT/'convergence.csv',convergence)
integrity=[]
for folder in ['pfreud_value_audit_20260908','pfreud_causal_20260908','pfreud_causal_phase_retry_20260908']:
    design=read(ROOT/'results'/folder/'design.json')
    for rel,digest in design['sources'].items():
        integrity.append(dict(design=folder,path=rel,unchanged=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest))
(OUT/'source_integrity.json').write_text(json.dumps(integrity,indent=2))
validation=[]
for r in rows:
    path=ROOT/r['source'];case_data=read(path);trajectory=list(csv.DictReader(path.with_name('trajectory.csv').open()))
    expected={'two_fixed':(.5,.1,0.),'two_equal_total':(.525,.105,0.),'three':(.5,.1,.03)}[r['arm']]
    checks=dict(rows=len(trajectory)==round(12/r['dt'])+1,endpoint=abs(float(trajectory[-1]['time_h'])-12)<1e-8,feed=abs(case_data['feed']-3)<1e-9,solver=case_data['diagnostics']['solve_success_rate']==1.,inoculum=all(abs(float(trajectory[0][key])-x)<1e-12 for key,x in zip(['or16_biomass_g_l','ns21_biomass_g_l','helper_biomass_g_l'],expected)),finite=np.isfinite([float(q['pha_g_l']) for q in trajectory]).all().item())
    validation.append(dict(source=r['source'],checks=checks,passed=all(checks.values())))
(OUT/'run_integrity.json').write_text(json.dumps(dict(completed_main_and_refinement=len(rows),expected_main_and_refinement=18,all_planned_complete=len(rows)==18,all_checks_pass=all(r['passed'] for r in validation),all_source_hashes_unchanged=all(r['unchanged'] for r in integrity),checks=validation),indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
ax=axes[0,0];labels=['2 / 2 schedule','2 / 3 schedule','3 / 2 schedule','3 / 3 schedule'];vals=[a,b,c,d]
bars=ax.bar(range(4),vals,color=['#627d98','#7fb3d5','#d59b46','#bf7623']);ax.set_xticks(range(4),labels,rotation=15);ax.set_ylabel('PHA at 12 h (g/L)');ax.set_title('A. Replay the nitrogen-switch schedule (dt = 1 h)');ax.bar_label(bars,fmt='%.4f',padding=3);ax.set_ylim(0,max(vals)*1.18)
ax=axes[0,1]
colors={'two_fixed':'#426f9b','two_equal_total':'#7ba1bf','three':'#cc822d'}
for arm in colors:
    rr=sorted([r for r in allrows if r['arm']==arm and r['profile']=='continuous'],key=lambda r:r['dt'])
    ax.plot([r['dt']*60 for r in rr],[r['pha'] for r in rr],'-o',label=arm,color=colors[arm])
ax.set_xscale('log');ax.invert_xaxis();ax.set_xlabel('Integration step (min), finer to the right');ax.set_ylabel('PHA at 12 h (g/L)');ax.set_title('B. Time-step dependence');ax.legend(fontsize=8)
ax=axes[1,0]
for arm in ['two_fixed','three']:
    path=ROOT/f'results/pfreud_value_audit_20260908/core_dt1.0_{arm}/trajectory.csv';rr=list(csv.DictReader(path.open()))
    ax.plot([float(r['time_h']) for r in rr],[float(r['nh4_mmol_l']) for r in rr],'-o',label=arm,color=colors[arm])
ax.axhline(.1,ls='--',color='#b44b52',label='PHA switch');ax.set_xlabel('Time (h)');ax.set_ylabel('NH4 (mmol/L)');ax.set_title('C. Small NH4 differences toggle entire hours');ax.legend(fontsize=8)
ax=axes[1,1];rr=[r for r in pairs if r['dt']==.05];x=np.arange(len(rr))
ax.bar(x-.18,[r['gain_fixed_pct'] for r in rr],.36,label='vs matched producers',color='#426f9b');ax.bar(x+.18,[r['gain_equal_pct'] for r in rr],.36,label='vs matched total inoculum',color='#cc822d');ax.set_xticks(x,[r['profile'] for r in rr]);ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Three-species PHA advantage (%)');ax.set_title('D. Same lactate budget, different timing (dt = 3 min)');ax.legend(fontsize=8)
fig.suptitle('P. freudenreichii: mechanism and numerical robustness audit',fontsize=15)
fig.savefig(OUT/'causal_audit.png',dpi=180);fig.savefig(OUT/'causal_audit.pdf');plt.close(fig)
table='\n'.join(f"| {r['dt']:g} | {r['profile']} | {r['two_fixed']:.6f} | {r['two_equal_total']:.6f} | {r['three']:.6f} | {r['gain_fixed_pct']:+.2f}% | {r['gain_equal_pct']:+.2f}% |" for r in pairs)
phase_table='\n'.join(f"| {r['arm']} | {r['schedule']} | {r['phase_hours']:g} | {r['pha']:.8f} |" for r in phase)
conv_table='\n'.join(f"| {r['arm']} | {r['coarse_dt']:g} → {r['fine_dt']:g} | {r['pha_change_pct']:+.2f}% | {r['phv_change_pp']:+.2f} |" for r in convergence if r['fine_dt']<=.025)
oxygen_rows=[dict(dt=dt,supply_limit=.25*(1-np.exp(-50*dt))/dt,continuous_fraction=(1-np.exp(-50*dt))/(50*dt)) for dt in [1.,.05,.025,.0125,.00625]]
writecsv(OUT/'oxygen_split_capacity.csv',oxygen_rows)
oxygen_table='\n'.join(f"| {r['dt']:g} | {r['supply_limit']:.4f} | {100*r['continuous_fraction']:.2f}% |" for r in oxygen_rows)
last=min([p for p in pairs if p['profile']=='continuous'],key=lambda p:p['dt'])
decision=dict(verdict='stable_three_species_production_advantage_not_demonstrated',one_hour_schedule_component_percent=100*schedule/total,ppa_transfer_needed_for_large_three_minute_gain=True,latest_complete_continuous_comparison=last,biological_validation=False,statistical_significance_tested=False,scope='12 h deterministic exploratory diagnostics; not global optimization',all_main_refinement_cases_complete=len(rows)==18)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2))
report=f'''# P. freudenreichii：1時間刻みの陽性原因と3種優位の数値検証

作成日：2026-09-08。全てモデル内の決定論的シミュレーション。実測・統計的有意性・長期安定生産を示すものではない。

## 結論

1時間刻みの陽性の主因はNH4による生産スケジュールの変化。3分刻みでの大きな陽性にはPPA経由の動態が必要だった。しかし時間刻みを細かくした比較で大きな優位が保たれず、3種の安定生産優位条件は確定していない。最も細かい完了済みの3群比較（{last['dt']*3600:g}秒刻み）では、3種PHAは{last['three']:.6f} g/L、生産菌接種量一致の2種比{last['gain_fixed_pct']:+.2f}%、総接種量一致の2種比{last['gain_equal_pct']:+.2f}%。

Pfの交差供給が存在することと、追加菌体のコストを含めて最良の2種運転を上回ることは区別する。まず酸素移動とNH4閾値イベントを含む数値計算を整え、その後、同じ資源予算で2種・3種をそれぞれ最適化して未知条件で比較することが必要。

## 比較条件

12時間、ゴム10 g/L、初期NH4 0.05 mmol/L、kLa 50/h、理想pH制御7、HiGHSによるseparate dFBA。乳酸の総添加量は全て3 mmol/L。continuousは0.25 mmol/L/hを12時間、earlyは0.75を最初の4時間、lateは0.75を最後の4時間添加する。体積変化のない濃度供給としての比較。

two_fixedはOR16 0.5 + NS21 0.1 g/L、threeはこれにPf 0.03を追加。two_equal_totalはOR16 0.525 + NS21 0.105で、3種と総接種量を一致させた対照。src/utils.pyの背景ビタミン・核酸塩基・少量アミノ酸は残っているため、乳酸とゴムだけの培地ではない。NH4閾値はモデル上の生産モード判定であり、有機窒素も含む実際の窒素制限の実証ではない。

## 1時間で陽性だった主因：窒素スイッチの1ステップ保持

コードはNH4 < 0.1 mmol/LならNS21のPHA生産を目的にし、それ以外なら生産シンクを閉じて増殖を目的にする。この状態を1ステップ中保持する。Pfの存在でNH4の推移が変わり、生産モードが2種では計5時間、3種では計6時間になった。

8時間時点では2種0.113507、3種0.091346 mmol/Lで、閾値を挟むため2種だけ生産が閉じる。9時間時点では逆になる。単純な一方向の早期窒素枯渇ではなく、閾値付近でのモード切替時刻の差である。

切替スケジュールを2×2で交換し、実際の栄養状態はそのままにした診断：

| 菌種構成 | スケジュールの出所 | 生産モード時間 | PHA g/L |
|---|---|---:|---:|
{phase_table}

元の差は{total:.8f} g/L。2×2の対称分解ではスケジュール成分{schedule:.8f} g/L（{100*schedule/total:.2f}%）、構成成分{species:.8f} g/Lになる。同一スケジュールでの3種効果は+{100*(c/a-1):.2f}%または+{100*(d/b-1):.2f}%。この約90%は**この反実仮想で定義した分解**であり、生物学的寄与の一意な割合ではない。強制スケジュールは培養操作としての提案ではない。

## 酸素の更新も時間刻みに依存する

酸素はステップ冒頭で `C*=0.25` に向けて `1-exp(-50*dt)` の分だけ補給され、その後その在庫内でゴム分解・菌体取り込みを解く。ステップ途中の継続的な移動はLPに入らない。1時間なら溶存量がほぼ飽和しても、次の補給は1時間後である。ゴム分解にはその酸素の25%しか割り当てないため、1時間条件では12時間合計0.75 mmol/Lの上限に達した。細刻みでは補給回数と酸素利用量が増え、PHAの絶対量も変わる。

各ステップで酸素を使い切る極限では、この分割更新の補給速度上限は `0.25*(1-exp(-50*dt))/dt` mmol/L/h。連続移動でDO≈0の上限 `50*0.25=12.5` と比較すると以下になる。初期溶存酸素の一回分を除いた定常的上限の計算であり、各軌道の実消費量そのものではない。

| dt h | 分割更新の補給上限 mmol/L/h | 連続移動上限に対する割合 |
|---:|---:|---:|
{oxygen_table}

この導出から、22.5秒刻みでも酸素在庫が毎回枯渇する条件の補給上限は連続式より約14%低い。PHA終点量の刻み間変化が小さいことだけで、酸素動態まで十分に収束したとは判定できない。

さらに各溶質の在庫制約は `濃度 / (全菌体量 × dt)`。共有酸素を使わない設定のPfも分母に入る。これは保守的な資源割当であり、実際にPfがその酸素を消費したことを意味しない。酸素消費菌だけに割り当てる診断では、3分刻みの2種は0.477472715 g/Lを再現し、3種は0.680879553から0.694208676 g/Lへ変化した。陽性は残ったため、この分母だけでは陽性を説明できない。独立プロセスの結果はresults/pfreud_oxygen_allocation_single_20260908に保存。現行モデルの変更とは区別する。

## 交換代謝物の監査

3分刻み・連続供給の全交換フラックス監査は3種のPHA 0.680879553 g/Lを再現した。12時間合計のPf乳酸取り込み0.071232151、プロピオン酸分泌0.026267964、酢酸分泌0.007702284、NH4取り込み0.008753461 mmol/L。B12分泌は検出されなかった（積算対象は1ステップ1e-15 mmol/L超）。NS21のPPA取り込みは0.026266081 mmol/L。ただし同じ共有プールの積算であり、同位体追跡ではない。

PPAと酢酸の放出炭素は合計約0.09421 mmol-C/L。3種のPHA増分0.20341 g/Lをその炭素だけで直接説明することはできず、他の炭素利用や生産時相が変わる間接効果と切り分ける必要がある。全交換量はresults/pfreud_exchange_audit_20260908/exchange_totals.jsonに保存。

Pfがそのステップで新たに放出したPPAを環境更新後に系外へ除く仮想介入では、PHAは0.474257269 g/Lとなり、3種優位が消失した。PPAと酢酸の両方を除くと0.473982642 g/L。PPA除去だけの場合も生産モード時間は8.75時間で元の3種と同じ、NS21のPPA取り込みは0になった。除去量0.026584181 mmol/Lを別途記録し、Pf自体・初期接種量・共有在庫の配分は残した。これは3分刻みの大きな陽性にPPA経由の動態が必要だったことを支持するが、同じ機序の効果量が細刻みでも保たれる証明ではない。仮想的な選択回収であり、実現可能な培養操作や遺伝子改変の提案ではない。結果はresults/pfreud_transfer_intervention_20260908に保存。

## 時間刻みと供給時期の全比較

| dt h | 供給 | 2種・生産菌一致 | 2種・総接種量一致 | 3種 | 対2種・生産菌一致 | 対2種・総量一致 |
|---:|---|---:|---:|---:|---:|---:|
{table}

時間刻みを細かくした際の変化：

| 構成 | dt h | PHA変化 | 3HVモル割合変化 pp |
|---|---|---:|---:|
{conv_table}

「複数の刻みで陽性」と「数値収束」は異なる。刻み間の誤差が優位幅に対して十分小さくなる必要がある。ここでの変更率は収束診断であり、誤差上限の保証ではない。12時間終点量だけでは長期安定性を判定できない。

## 3種の安定優位を認定する条件

1. 同じ資源・時間・総接種量で、最良の2種運転を超える。3種にだけ最適化した供給を与える比較では不十分。
2. Pfによる乳酸等からの供給がNS21の制限要因を解消し、その利益が炭素・窒素競合と追加菌体のコストを上回る。PPA分泌と取り込みの成立だけでは総PHA増加を保証しない。
3. 生産菌を確保した後の窒素制限と、ゴム分解・PHA生産に必要な酸素・炭素供給が両立する。現行の閾値ちょうどを利用する制御を最適と認定しない。
4. 時間刻み、ソルバーの代替最適解、初期菌比、取り込み速度、kLa、供給誤差を変えても優位が維持される。数値収束と生物学的パラメータ不確実性は別々に確認する。
5. 菌体維持・反復供給・長時間の生産速度と、投入全炭素当たり収率・培地/酸素/pH制御コストを評価する。3HVは高いほど良いと仮定せず、目標組成への適合で評価する。

これらは評価基準であり、今回すべてを満たす培養条件が確定したという意味ではない。Pfの共有酸素取り込みを閉じた設定や汎用取り込み速度は実測校正前であり、実験へそのまま移せる処方ではない。

次のモデル改善では、培地供給の操作間隔と内部積分刻みを分け、酸素移動と消費を整合させ、NH4閾値を跨ぐ時刻を内部で扱う必要がある。操作を1時間ごとにすること自体を禁止する話ではない。制御操作を一定に保ったまま内部計算を細分化して比較する。そのうえで同一評価条件にて2種と3種の供給をそれぞれ最適化し、未知条件での差を確認する。現行の陽性条件だけをRL報酬設計や学習データ選択の根拠にしない。

閾値や増殖/PHAの目的関数は数値計算法とは別の生物学的仮定であり、滑らかな関数に置き換えれば自動的に正しくなるわけではない。交換フラックスの代替最適解にも、一貫した二次目的や不確実性評価が必要。この論点はDFBAlabの一次論文でも扱われている：[DFBAlab: a fast and reliable MATLAB code for dynamic flux balance analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4279678/)。基底の再利用など、積分の精度とLP呼び出し回数を両立させる手法の一次研究もある：[Minimizing the number of optimizations for efficient community dynamic flux balance analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7546477/)。これらの文献は手法の参考であり、本3菌系の優位を示す実験的根拠ではない。

## 再現性と制限

主スクリプトの最初の強制スケジュール4件はsinkのoriginal_bounds参照により失敗し、`.get`を使った別スクリプトで4件を再実行した。自己スケジュール2件は元結果を再現した。失敗ログは保存している。酸素診断の初回逐次実行では計測サブクラスが重複して3種の累積テレメトリが二重計上されたため、独立プロセスで3種を再実行した。初回3種の累積値は解析に使用しない。

既存の本番シミュレータ・GEMは変更していない。source_integrity.jsonに元ソースのハッシュ一致を保存。今回は局所的な機序と数値診断であり、全条件探索、RL最適化、実測検証、完全炭素収支検証は行っていない。結果の反復を独立した生物学的反復としてp値を計算していない。

生成物：causal_runs.csv、resolution_and_feed.csv、convergence.csv、phase_counterfactual.csv、phase_decomposition.json、causal_audit.png/pdf。元軌道とケース条件は各ケースフォルダに保存。
'''
(OUT/'REPORT_JA.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(pairs=pairs,phase_decomposition=decomp,source_integrity=integrity),indent=2))
