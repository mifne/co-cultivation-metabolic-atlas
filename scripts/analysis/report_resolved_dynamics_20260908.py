"""Validation report, keeping numerical accuracy separate from biological fit."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/resolved_dynamics_20260908'
design=json.loads((OUT/'design.json').read_text());rows=[];trajectories={};checks=[]
for case in design['cases']:
    dest=OUT/case['id']
    if not (dest/'result.json').exists():continue
    result=json.loads((dest/'result.json').read_text());cfg=json.loads((dest/'design.json').read_text())
    final=result['final'];diag=result['diagnostics'];oxygen=diag['resolved_dynamics']['oxygen']
    trace=list(csv.DictReader((dest/'trajectory.csv').open()));trajectories[case['id']]=trace
    row=dict(case=case['id'],arm=case['arm'],controller_dt=case['control'],internal_dt=case['internal'],pha=final['pha_g_l'],rubber_removed=10-final['rubber_g_l'],biomass=final['or16_biomass_g_l']+final['ns21_biomass_g_l']+final['helper_biomass_g_l'],nh4=final['nh4_mmol_l'],o2=final['o2_mmol_l'],phv_fraction=final['phv_mol_fraction'],seconds=result['seconds'],internal_steps=diag['steps'],solver_success=diag['solve_success_rate'],oxygen_transferred=oxygen['transferred'],oxygen_consumed=oxygen['polymer_consumed']+oxygen['cellular_consumed'])
    rows.append(row)
    case_checks=dict(time=abs(final['time_h']-12)<1e-10,feed=abs(final['cumulative_feed_mmol_l']-3)<1e-9,
        all_solvers=diag['solve_success_rate']==1.,oxygen_balance=abs(.25+row['oxygen_transferred']-row['oxygen_consumed']-row['o2'])<1e-8,
        finite=bool(np.isfinite([float(r[k]) for r in trace for k in ['pha_g_l','o2_mmol_l','nh4_mmol_l']]).all()),
        nonnegative=all(float(r[k])>=-1e-10 for r in trace for k in ['pha_g_l','o2_mmol_l','nh4_mmol_l']),
        source_hashes=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cfg['source_sha256'].items()))
    checks.append(dict(case=case['id'],passed=all(case_checks.values()),checks=case_checks))
with (OUT/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
refinement=sorted([r for r in rows if r['arm']=='three' and r['controller_dt']==1.],key=lambda r:r['internal_dt'],reverse=True)
changes=[]
for a,b in zip(refinement,refinement[1:]):
    changes.append(dict(coarse=a['internal_dt'],fine=b['internal_dt'],pha_change_pct=100*(b['pha']/a['pha']-1),rubber_change_pct=100*(b['rubber_removed']/a['rubber_removed']-1),biomass_change_pct=100*(b['biomass']/a['biomass']-1),o2_change_mmol_l=b['o2']-a['o2']))
base=next((r for r in rows if r['case']=='three_h0.025_control1'),None)
partition=next((r for r in rows if r['case']=='three_h0.025_control0.2'),None)
partition_diff=None
if base and partition:
    partition_diff={k:partition[k]-base[k] for k in ['pha','rubber_removed','biomass','nh4','o2']}
evidence=json.loads((ROOT/'results/timestep_reference_20260908/evidence.json').read_text())
decision=dict(all_planned_complete=len(rows)==len(design['cases']),all_integrity_pass=all(r['passed'] for r in checks),refinement=changes,controller_partition_difference=partition_diff,biological_calibration=False,reference=evidence,checks=checks)
(OUT/'validation.json').write_text(json.dumps(decision,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for case,color in [('three_h0.025_control1','#b77428'),('three_h0.025_control0.2','#386f99')]:
    if case not in trajectories:continue
    trace=trajectories[case]
    axes[0,0].plot([float(r['time_h']) for r in trace],[float(r['pha_g_l']) for r in trace],label='control '+('1 h' if case.endswith('control1') else '0.2 h'),color=color,ls='-' if case.endswith('control1') else '--')
axes[0,0].set(title='A. Controller partition, same internal step',xlabel='Time (h)',ylabel='PHA (g/L)');axes[0,0].legend()
axes[0,1].plot([r['internal_dt']*60 for r in refinement],[r['pha'] for r in refinement],'-o',color='#386f99')
axes[0,1].invert_xaxis();axes[0,1].set(title='B. Internal-step refinement (three species)',xlabel='Internal step (min), finer to the right',ylabel='PHA at 12 h (g/L)')
for r in refinement:
    trace=trajectories[r['case']];axes[1,0].plot([float(q['time_h']) for q in trace],[float(q['o2_mmol_l'])*32 for q in trace],label=f"{r['internal_dt']*60:g} min")
axes[1,0].set(title='C. Dissolved oxygen, model refinement',xlabel='Time (h)',ylabel='DO (mg/L)');axes[1,0].legend()
ref=evidence['pha_table_transcribed_from_embedded_emf'];axes[1,1].plot([r['cn'] for r in ref],[r['pha_titer_mg_l'] for r in ref],'-o',color='#b77428')
axes[1,1].set(title='D. Supplied RDE2 / pyruvate experiment',xlabel='C/N ratio',ylabel='PHA titer (mg/L)')
axes[1,1].text(.03,.03,'Qualitative evidence; not a three-species fit',transform=axes[1,1].transAxes,fontsize=9)
fig.suptitle('Resolved oxygen and nitrogen allocation: numerical validation',fontsize=14)
fig.savefig(OUT/'validation.png',dpi=180);fig.savefig(OUT/'validation.pdf');plt.close(fig)
table='\n'.join(f"| {r['arm']} | {r['controller_dt']:g} | {r['internal_dt']:g} | {r['pha']:.8f} | {r['rubber_removed']:.6f} | {r['o2']:.6f} | {r['seconds']:.1f} |" for r in rows)
change_text='\n'.join(f"- 内部刻み{r['coarse']*60:g}→{r['fine']*60:g}分：PHA {r['pha_change_pct']:+.3f}%、ゴム分解 {r['rubber_change_pct']:+.3f}%、全菌体量 {r['biomass_change_pct']:+.3f}%、DO差 {r['o2_change_mmol_l']:+.6f} mmol/L。" for r in changes)
partition_text='未完了' if partition_diff is None else '、'.join(f'{k}: {v:+.3e}' for k,v in partition_diff.items())
report=f'''# 酸素とNH4による時間刻み依存の改善

2026-09-08。新規CPU参照経路 `ResolvedDFBASimulator`、数値バージョン `resolved_oxygen_nh4_v1`。旧GEM・旧dFBA・512軌道のLPラベルは変更していない。

## 資料の取り入れ方

提供された齋藤氏の進捗報告はRhodococcus sp. RDE2組換株を150 mMピルビン酸で培養した結果。PHA含有率はC/N 40→100で10.8→22.4%へ増えるが、PHA濃度はC/N 60の493.3 mg/Lで最大となる。増殖と蓄積のトレードオフを扱い、含有率だけを増産指標にしない根拠として用いた。

同じ資料は3菌共培養の速度定数を与えていない。24時間間隔のDO図から秒・分スケールのkLaは同定できず、NH4濃度の時系列もない。150 rpmからkLaを推定していない。図は4検体平均と記載されるが、PHA表の反復数・誤差は明記されていない。定常期7〜8日は今回の12時間比較と異なる。資料の493.3 mg/Lと計算結果が近くてもモデル検証とは呼ばない。

## 実装

制御間隔と内部刻みを分離し、連続供給を `feed_rates_mmol_l_h` で入力する。酸素は区間中一定の消費速度を仮定した `dC/dt=kLa(Cs-C)-OUR` の解析解で更新し、ゴム分解と全菌体の酸素消費を同じ予算で管理する。kLa=0の閉鎖系にも連続的に対応する。酸素を使わないPfを酸素配分の分母に含めない。

酸素の在庫予算を溶存濃度として速度式へ流用しない。速度評価用DOには無消費時の区間平均を予測値として使う。この近似と区間中の代謝・菌体変化の誤差は残るので、内部刻みの収束試験を別途行った。「酸素ODEの定数消費区間を解析的に解いた」ことは「非線形共培養全体を厳密に解いた」ことではない。

NS21は最初に増殖可能量を求め、`NH4/(K_N+NH4)` の割合を確保したうえでPHA質量を最大化する。旧NH4=0.1での完全ON/OFFを除いた。ただしこの配分則は生物学的な作業仮説で、`K_N=0.1 mmol/L`も未校正。旧式との差には数値改善と配分モデル変更の両方が含まれる。

## 同一モデル内の時間刻み比較

12時間、ゴム10 g/L、乳酸0.25 mmol/L/h、初期NH4 0.05 mmol/L、kLa 50/h、理想pH 7、旧比較と同じ背景微量栄養。3種の初期接種量はOR16 0.5、NS21 0.1、Pf 0.03 g/L。2種対照はOR16 0.525、NS21 0.105 g/Lで総接種量を一致させた。

| 構成 | 制御間隔 h | 内部刻み h | PHA g/L | ゴム分解 g/L | 最終DO mmol/L | 実行秒 |
|---|---:|---:|---:|---:|---:|---:|
{table}

内部刻みの細分化：

{change_text}

制御間隔1→0.2時間での差（内部刻み0.025時間固定）：{partition_text}。

制御間隔に対する不変性と、内部刻みの収束は別の検査。ここに挙げた終点量の変化率は誤差上限やあらゆる条件での安定性を保証しない。未評価の供給・初期菌比・kLaへ適用する場合は再度細分化比較が必要。

## 検証

解析解・酸素収支、kLa=0、制御分割、内部刻みで割り切れない制御間隔、連続供給量、ボーラスの重複防止、旧NH4閾値近傍での増殖と蓄積の連続性など9テストが合格。既存のゴム分解・NS21 PHA・Pf・1 Lジャー関連18テストも合格。全比較の完了・LP成功・12時間到達・投入量・酸素収支・有限値・非負濃度・ソースハッシュはvalidation.jsonに保存。

## 適用範囲

新経路はHiGHS/GLPKのseparate dFBA用。cooperative/joint/GPUサロゲート/frozen教師収集にはまだ接続しておらず、誤用は例外で防ぐ。内部細分化と追加のNS21増殖LPの計算費用がある。次の教師生成・高速化では、この数値バージョンを明示し、新しい状態分布で精度を確認する必要がある。

既存の512軌道収集の停止をこの修正で解消したとは主張しない。新モードの軌道と旧教師軌道を混在させず、個々のLP解の再利用可能性と動態教師としての適合性を分けて判断する。

使用方法はdocs/RESOLVED_DYNAMICS_20260908.md。元資料の数値転記と適用限界はresults/timestep_reference_20260908/evidence.json。
'''
(OUT/'REPORT_JA.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(rows=rows,refinement=changes,partition=partition_diff,complete=decision['all_planned_complete'],passed=decision['all_integrity_pass']),indent=2))
