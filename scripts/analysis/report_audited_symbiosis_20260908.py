"""Report every audited case and separate numerical evidence from biology."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/audited_symbiosis_20260908'
PF='Propionibacterium_freudenreichii_shermanii';NS='Rhizobacter_gummiphilus_NS21'
conditions=['base','no_lactate','lactate_rich','nitrogen_staged','oxygen_low']
labels=['Baseline','No lactate','4x lactate','Extra early NH4','Low kLa']
jlabels=['基準','乳酸なし','乳酸4倍','初期4時間NH4追加','低kLa']
def read(path):return json.loads(path.read_text())
def rows(path):
    with path.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def write_csv(name,records):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)

def percent_change(value,reference):
    """A near-zero reference has no reportable relative change."""
    return None if reference<=1e-12 else 100*(value/reference-1)

def percent_text(value):
    return '算定不可' if value is None else f'{value:+.2f}%'

def plot_values(values):
    return [np.nan if value is None else value for value in values]

def passes_screen(pha_gain,rubber_gain):
    return (pha_gain is not None and rubber_gain is not None
            and pha_gain>5 and rubber_gain>=-5)
def main():
    design=read(OUT/'design.json')
    reruns={}
    # Prefer the earliest successful attempt, never the largest output value.
    for manifest in ['numerical_patch_reruns.json','precision_patch_reruns.json']:
        if (OUT/manifest).exists():
            for record in read(OUT/manifest):
                if record['returncode']==0:reruns.setdefault(record['case']['id'],record)
    locations={};patches={}
    for case in design['cases']:
        key=case['id'];location=OUT/key
        if not (location/'result.json').exists():
            rerun=reruns[key]
            assert rerun['returncode']==0 and rerun['case']==case
            location=OUT/rerun['relative_directory']
            assert location.resolve().is_relative_to(OUT.resolve())
            patches[key]=read(location/'numerical_patch.json')
        locations[key]=location
    data={key:read(location/'result.json') for key,location in locations.items()}
    checks={};metrics={}
    for key,d in data.items():
        r=rows(locations[key]/'trajectory.csv');internal=rows(locations[key]/'internal_trajectory.csv')
        diag=d['diagnostics'];aud=diag['audited_cultivation'];acc=aud['accounting'];rd=diag['resolved_dynamics']
        o=rd['oxygen'];feed=rd['continuous_feed'];init_o=d['settings']['initial_medium']['o2_e']
        oxygen_error=d['final']['o2']-init_o-o['transferred']+o['polymer_consumed']+o['cellular_consumed']
        carbon_error=max(abs(x['polymer_carbon_error']) for x in internal)
        checks[key]=dict(valid=aud['valid'],finite_nonnegative=all(math.isfinite(v) and v>=-1e-7 for x in r for v in x.values()),
            final_time=abs(d['final']['time']-12)<1e-8,
            lp_success=acc['lp_requests']==acc['lp_certified_requests']+acc.get('superseded_uncertified_requests',0),
            lactate_feed=abs(feed.get('lac__L_e',0)-12*d['settings']['lactate_rate'])<1e-9,
            nh4_feed=abs(feed.get('nh4_e',0)-(.4 if d['settings']['case']['condition']=='nitrogen_staged' else 0.))<1e-9,
            oxygen_balance=abs(oxygen_error)<1e-8,polymer_carbon=carbon_error<1e-8,
            lp_residual=acc['max_lp_residual']<=1e-7,oxygen_coupling=acc['max_oxygen_mean_residual']<=1e-6,
            storage=acc['max_storage_excess_g_l']<=1e-7,failed_steps=acc['failed_steps']==0)
        metrics[key]=dict(oxygen_balance_error_mmol_l=oxygen_error,polymer_c5_error_mmol_l=carbon_error,
            lp_residual=acc['max_lp_residual'],oxygen_mean_residual=acc['max_oxygen_mean_residual'],
            roundoff_added_mmol_l=acc['roundoff_added_mmol_l'],seconds=d['seconds'],lp_solves=diag['solve_attempts'],
            rejected_trials=len(acc['rejected_lp_trials']),retry_attempts=acc['lp_retry_attempts'],
            superseded_lp_targets=acc.get('superseded_uncertified_requests',0),
            primary_recomputations=acc.get('primary_recomputations',[]),
            max_dual_residual=acc['max_dual_residual'],max_relative_duality_gap=acc['max_relative_duality_gap'])
    checks['source_snapshot_hashes']={f:hashlib.sha256((OUT/'sources'/f.replace('/','__')).read_bytes()).hexdigest()==sha for f,sha in design['sources'].items()}
    execution=read(OUT/'execution_source_verification.json')
    checks['execution_sources']={f:execution['after_run_sha256'].get(f)==sha for f,sha in design['sources'].items()}
    adapter_sha=hashlib.sha256((ROOT/'scripts/analysis/audited_roundoff_adapter_20260908.py').read_bytes()).hexdigest()
    checks['boundary_patch_sources']={}
    for key,p in patches.items():
        if p['patch']=='c30_and_primary_recomputation':
            valid=(p['sources']['src/audited_dfba.py']==design['sources']['src/audited_dfba.py'] and
                   all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha for f,sha in p['sources'].items() if f!='src/audited_dfba.py'))
        else:
            valid=p['source_sha256']==adapter_sha and p['base_source_sha256']==design['sources']['src/audited_dfba.py']
        checks['boundary_patch_sources'][key]=valid and p['threshold_mmol_l']==1e-12
    checks['paired_designs']={}
    for condition in conditions:
        a=data[condition+'_two_equal_total']['settings'];b=data[condition+'_three']['settings']
        checks['paired_designs'][condition]=(a['initial_medium']==b['initial_medium']
            and all(a[k]==b[k] for k in ['kla','lactate_rate','internal_dt','nitrogen_half_saturation','flux_selection'])
            and abs(sum(a['initial_biomass'].values())-.63)<1e-12
            and abs(sum(b['initial_biomass'].values())-.63)<1e-12)
    assert all(all(v.values()) for v in checks.values()),checks
    (OUT/'integrity.json').write_text(json.dumps(dict(checks=checks,metrics=metrics,
        selected_result_directories={k:str(p.relative_to(OUT)) for k,p in locations.items()},boundary_patched_cases=list(patches)),indent=2))
    paired=[]
    for c in conditions:
        a=data[c+'_two_equal_total']['final'];b=data[c+'_three']['final']
        gain=percent_change(b['pha'],a['pha']);rubber_gain=percent_change(10-b['rubber'],10-a['rubber'])
        paired.append(dict(condition=c,two_pha=a['pha'],three_pha=b['pha'],gain_percent=gain,
            pha_difference_g_l=b['pha']-a['pha'],
            two_rubber_removed=10-a['rubber'],three_rubber_removed=10-b['rubber'],rubber_gain_percent=rubber_gain,
            two_hv_percent=100*a['phv_mol_fraction'],three_hv_percent=100*b['phv_mol_fraction'],
            screen_positive=passes_screen(gain,rubber_gain)))
    finer=[]
    for c in ['base','oxygen_low']:
        a=data[c+'_two_equal_total_fine']['final'];b=data[c+'_three_fine']['final']
        coarse=next(r for r in paired if r['condition']==c)
        gain=percent_change(b['pha'],a['pha']);rubber_gain=percent_change(10-b['rubber'],10-a['rubber'])
        finer.append(dict(condition=c,two_pha=a['pha'],three_pha=b['pha'],gain_percent=gain,
            pha_difference_g_l=b['pha']-a['pha'],
            rubber_gain_percent=rubber_gain,two_change_percent=percent_change(a['pha'],coarse['two_pha']),
            three_change_percent=percent_change(b['pha'],coarse['three_pha']),
            gain_change_percentage_points=(gain-coarse['gain_percent'] if gain is not None and coarse['gain_percent'] is not None else None),
            screen_positive=passes_screen(gain,rubber_gain)))
    passes_both_steps=[r['condition'] for r in paired if r['screen_positive'] and any(f['condition']==r['condition'] and f['screen_positive'] for f in finer)]
    need_finer=[r['condition'] for r in paired if r['screen_positive'] and r['condition'] not in ['base','oxygen_low']]
    write_csv('paired_comparison.csv',paired);write_csv('refinement.csv',finer)
    exchange_rows=[]
    for key,d in data.items():
        for species,flux in d['exchanges'].items():
            for met,value in flux.items():exchange_rows.append(dict(case=key,species=species,exchange=met,integrated_mmol_l=value))
    write_csv('integrated_exchanges.csv',exchange_rows)
    (OUT/'summary.json').write_text(json.dumps(dict(paired=paired,refinement=finer,
        conditions_passing_screen_at_both_steps=passes_both_steps,numerical_convergence_established=False,
        require_finer=need_finer,all_integrity_checks_passed=True),indent=2,allow_nan=False))
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');x=np.arange(5)
    ax=axs[0,0]
    ax.bar(x-.18,[r['two_pha'] for r in paired],.36,label='OR16 + NS21',color='#718096')
    ax.bar(x+.18,[r['three_pha'] for r in paired],.36,label='All 3',color='#138a72')
    ax.set(xticks=x,xticklabels=labels,ylabel='PHA at 12 h (g/L)',title='Same total inoculum and feed within each pair');ax.tick_params(axis='x',rotation=15);ax.legend()
    ax=axs[0,1];ax.axhline(0,color='black',lw=.8);ax.axhline(5,color='#999999',ls='--')
    ax.bar(x,plot_values(r['gain_percent'] for r in paired),color=['#138a72' if r['gain_percent'] is not None and r['gain_percent']>0 else '#c26542' for r in paired])
    ax.set(xticks=x,xticklabels=labels,ylabel='3-species PHA difference (%)',title='Exploratory screen, no statistical significance');ax.tick_params(axis='x',rotation=15)
    ax=axs[1,0];arms=['ns21_alone','ns21_pf','two_equal_total','three']
    ax.bar(range(4),[data['base_'+a]['final']['pha'] for a in arms],color=['#4976a8','#7b68a6','#718096','#138a72'])
    ax.set(xticks=range(4),xticklabels=['NS21','NS21 + Pf','OR16 + NS21','All 3'],ylabel='PHA at 12 h (g/L)',title='Alternative allocations, total inoculum 0.63 g/L')
    ax=axs[1,1];x2=np.arange(2)
    ax.bar(x2-.18,plot_values(next(r['gain_percent'] for r in paired if r['condition']==c) for c in ['base','oxygen_low']),.36,label='0.025 h',color='#718096')
    ax.bar(x2+.18,plot_values(f['gain_percent'] for f in finer),.36,label='0.0125 h',color='#138a72')
    ax.axhline(0,color='black',lw=.8);ax.axhline(5,color='#999999',ls='--');ax.legend()
    ax.set(xticks=x2,xticklabels=['Baseline','Low kLa'],ylabel='3-species PHA difference (%)',title='Paired internal-step refinement')
    fig.suptitle('Audited reference v2: exploratory simulations, uncalibrated biology',fontsize=13)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    table='\n'.join(f"|{label}|{r['two_pha']:.6f}|{r['three_pha']:.6f}|{r['pha_difference_g_l']:+.6f}|{percent_text(r['gain_percent'])}|{percent_text(r['rubber_gain_percent'])}|" for label,r in zip(jlabels,paired))
    ftable='\n'.join(f"|{r['condition']}|{r['two_pha']:.6f}|{r['three_pha']:.6f}|{percent_text(r['gain_percent'])}|{percent_text(r['two_change_percent'])}|{percent_text(r['three_change_percent'])}|" for r in finer)
    alttable='\n'.join(f"|{a}|{data['base_'+a]['final']['pha']:.6f}|{10-data['base_'+a]['final']['rubber']:.6f}|" for a in arms+['two_fixed','pf_alone'])
    lowtable='\n'.join(f"|{a}|{data['oxygen_low_'+a]['final']['pha']:.6f}|{100*data['oxygen_low_'+a]['final']['phv_mol_fraction']:.2f}|{10-data['oxygen_low_'+a]['final']['rubber']:.6f}|" for a in ['two_equal_total','three','two_fixed','ns21_alone','inert_pf'])
    transfers=[]
    for c in ['base','oxygen_low']:
        d=data[c+'_three'];pf=d['exchanges'].get(PF,{});ns=d['exchanges'].get(NS,{})
        transfers.append(f"|{c}|{pf.get('lac__L_e_in',0):.6f}|{pf.get('ppa_e_out',0):.6f}|{ns.get('ppa_e_in',0):.6f}|{pf.get('ac_e_out',0):.6f}|{ns.get('lac__L_e_in',0):.6f}|{d['fold_growth'][PF]:.6f}|")
    passed_text='、'.join(jlabels[conditions.index(c)] for c in passes_both_steps) or 'なし'
    pending_text='、'.join(jlabels[conditions.index(c)] for c in need_finer) or 'なし'
    additional=''
    extra_path=OUT/'third_refinement/summary.json'
    if extra_path.exists():
        extra=read(extra_path);latest=extra['paired_gains'][-1]
        final_rows=[r for r in extra['rows'] if r['internal_dt_h']==.00625]
        additional=('\n低kLaは追加の2ケースとして0.00625 hでも計算した（全体で23ケース）。'
            f"この刻みで3種のPHA差は {latest['pha_gain_percent']:+.3f}%。"
            +'一段前からのPHA量変化は、'
            +'、'.join(('2種' if r['arm']=='two_equal_total' else '3種')+f" {r['change_from_double_step_percent']:+.2f}%" for r in final_rows)
            +'。詳細は [追加刻み検証](third_refinement/REPORT_JA.md) と図・CSVを参照。\n')
    report=f'''# 監査後の3種共培養再評価（2026-09-08）

数値不整合を修正した `audited_cultivation_v2` で、基本の{len(data)}ケースを評価した。基準刻みで探索基準を満たすのは{sum(r['screen_positive'] for r in paired)}/5条件、両刻みで候補基準を通過した条件は **{passed_text}**。これは数値収束や優位幅の頑健性を確定する判定ではない。候補の陽性判定のために追加の細刻み確認を要する条件は {pending_text}。
探索基準はPHAが主対照より5%超増え、ゴム除去の悪化が5%以内。これは工学的な候補選択の基準であり、統計的有意差ではない。21条件は実験反復ではない。

## 数値結果

全て12時間、初期ゴム10 g/L、NH4 0.05 mmol/L、pH 7。基準乳酸0.25 mmol/L/h、kLa 50 h⁻¹。3種OR16/NS21/Pfは0.5/0.1/0.03 g/L、主対照OR16/NS21は0.525/0.105 g/Lで総接種量を0.63 g/Lに揃えた。Pfは嫌気的な交換境界を維持した仮定である。

|条件|2種PHA g/L|3種PHA g/L|PHA絶対差 g/L|PHA比率差|ゴム除去比率差|
|---|---:|---:|---:|---:|---:|
{table}

比率の基準値が1e-12以下の場合は「算定不可」とし、候補基準を通過したとは判定しない。PHAの絶対差は表とCSV/JSONに保存した。グラフでは算定できない比率を欠測として扱う。

乳酸4倍は1 mmol/L/h、NH4追加は初期4時間に0.1 mmol/L/h、低kLaは10 h⁻¹。ペア内で投入量を一致させた。条件間の供給量は異なり、増産率を費用効率と同一視できない。NH4追加条件には等量を別時刻に供給する対照がなく、時刻の優位性は判定できない。

内部刻みを0.025 hから0.0125 hへ半減した結果：

|条件|細刻み2種PHA g/L|細刻み3種PHA g/L|細刻みPHA比率差|2種の刻み変更差|3種の刻み変更差|
|---|---:|---:|---:|---:|---:|
{ftable}

{additional}

細刻み間の比較はこの2条件のみ。1段階の半減で一致しても数学的収束の証明ではない。酸素の総収支と速度が整合すること、時間離散化の影響が小さいこと、生理応答が正しいことは別の要件である。

## 反証用対照

基準条件：

|構成|PHA g/L|ゴム除去 g/L|
|---|---:|---:|
{alttable}

NS21単独・NS21+Pf・主対照2種・3種は総菌体量を揃えた運用候補比較であり、NS21への接種量配分は異なる。Pf単独だけはPf量0.03 g/Lを揃えた比較。two_fixedはOR16/NS21の量を0.5/0.1に保ち、Pf追加による総量増加を許す対照である。

低kLa：

|構成|PHA g/L|3HV mol%|ゴム除去 g/L|
|---|---:|---:|---:|
{lowtable}

inert_pfはPfの全反応を停止した数学的な対照。遺伝子ノックアウト、死菌添加、熱殺菌の実験を再現していない。炭素分泌と窒素消費を一緒に停止するため、差をプロピオン酸だけの効果とは呼べない。NS21単独を含む他の構成より優れるかを確認せず、3種の必然性を主張しない。

## Pfの代謝の手掛かり

全内部ステップで積分した交換量（mmol/L）とPfの増殖倍率：

|条件|Pf乳酸取込|Pfプロピオン酸分泌|NS21プロピオン酸取込|Pf酢酸分泌|NS21乳酸直接取込|Pf増殖倍率|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(transfers)}

共有プールへの分泌と取り込みは、同位体による炭素起源追跡や因果効果の測定ではない。NS21が乳酸を直接利用できる場合、Pf経由が必須とは言えない。GEMの交換フラックスは主目的が同じでも複数の解を持つため、今回は主目的を絶対誤差1e-9以内に保ちながら交換絶対値を最小にする規則を使った。この選択規則を実測フラックスと同一視しない。

低kLaでの3HV組成の差は、PHA質量とは別の品質仮説として扱う。現モデルはC30/ODTDの存在でPHV生成を開閉する未検証の条件を含むため、Pfのプロピオン酸供給だけで組成差が決まったとは断定できない。結果を見た後で品質指標を選び直して、当初のPHA増産基準を達成したとは扱わない。

## 修正で保証した範囲と、残る課題

C30がちょうどゼロになる際の微小負値で停止した対照は、境界補正を記録して再実行した。今回使用した補正付き結果は {list(patches)}。元の失敗を保存し、結果の大小で再実行を選んでいない。補正は−1e-12 mmol/L以上の微小負値だけを0にし、受理したステップの補正量を収支に加える。それより負の値は引き続き拒否する。元の成功済みケースは全酸素試行でC30非負を検査済みのため、この追加分岐は一度も発動せず、動態計算は同じになる（C30_NUMERICAL_EQUIVALENCE.md）。ソースの文字列が同一という主張ではなく、限定した非発動分岐の同等性である。教師データのソース契約は別に扱う。

全{len(data)}ケースで有限・非負の記録、12時間到達、指定投入量、酸素収支、細胞外ゴム/C30/ODTDのC5当量収支、LP成功、貯蔵上限、ソース一致を確認した（integrity.json）。酸素平均値の消費との連立残差は最大 {max(v['oxygen_mean_residual'] for v in metrics.values()):.3g} mmol/L、LP最大残差 {max(v['lp_residual'] for v in metrics.values()):.3g}、ゴム分解のC5当量最大誤差 {max(v['polymer_c5_error_mmol_l'] for v in metrics.values()):.3g} mmol/L。

全ソルバー呼出し {sum(v['lp_solves'] for v in metrics.values()):,} 回中、不適合として拒否した試行は {sum(v['rejected_trials'] for v in metrics.values()):,} 回。同じLPの再試行 {sum(v['retry_attempts'] for v in metrics.values()):,} 回に加え、一次目的からの高精度再計算で二次LPの保持目標を作り直した要求が {sum(v['superseded_lp_targets'] for v in metrics.values()):,} 件ある。作り直す前の目標は解けたと数えず、superseded_uncertified_requestsとprimary_recomputationsへ記録した。動態に使用した解は同じ許容誤差で認証した。初回試行が全件成功したという意味ではない。受理解の双対残差最大 {max(v['max_dual_residual'] for v in metrics.values()):.3g}、相対主双対ギャップ最大 {max(v['max_relative_duality_gap'] for v in metrics.values()):.3g}。

一次目的値は一般的な残差1e-7を通過しても、二次目的に課す保持精度1e-9に対しては過大になり得る。二次LPの全解法が失敗した時だけ、一次目的からdual simplex/presolve無効/ソルバー精度1e-10で再計算し、必要ならIPMを試した。物質収支や一次目的の保持精度を緩めていない。この追加処理も元の成功軌道では発動しない。二次LPの数値目標が更新されるため、同一LPへの再試行とは区別した。

GEM全反応の元素収支保証ではない。欠落分子式、仮説反応、疑似菌体、プロトン規約は別に監査記録を残した。未校正のNH4依存配分、酸素応答、酵素速度、Pfの酸素表現型は残る。必須の非ゼロ維持フラックスを今後導入する際は、共通密度/pH倍率の設計変更が必要であり、現参照はそのような入力を拒否する。

背景アミノ酸・核酸成分等が炭素と窒素を供給するため、総PHAをゴム由来PHAと呼べない。NH4は外部供給だけでなく代謝分泌でも増え得る。B12を受け取るOR16/NS21の依存回路は確認できず、このモデルでB12の共生価値は評価できない。維持代謝、死滅、PHA再利用、希釈・継代・長期摂動からの回復も未検証であり、12時間の残存は安定共生の証明ではない。

初期培地の分子式監査では、NH4以外の既知窒素3.01 mmol-N/L、既知有機背景炭素8.89 mmol-C/Lを確認した。NH4 0.05 mmol-N/Lは既知全窒素の1.63%で、外部乳酸の基準12h投入炭素は9 mmol-C/Lである。直接対応しない7成分は未確定として除外した。これは含有量の比較であり、全成分の利用可能性やPHAへの同化量を示さない（results/cultivation_model_audit_20260908/medium_element_inventory_JA.md）。NH4だけで全窒素制限やゴム炭素収率を判定しない。

## 次に進める条件

3種有利な候補が残っても、まず同じ投入炭素・窒素・酸素予算でNS21単独、OR16+NS21、NS21+Pf、3種を比較し、接種比と供給条件をそれぞれ探索する。3HV品質、乳酸除去、ゴム除去、PHA質量、操作費を区別する。生物学的に未確認の依存性を加えて3種を有利にしない。

今回の参照モデルと旧512軌道の教師モデルは異なる。旧教師は保存し、今回のデータと混ぜて学習しない。新しいGNN+GRU/RL教師収集の前に、参照版・制御スキーマ・校正範囲を固定し、小規模な収集と再生の整合性を確認する。

旧計算は results/symbiosis_reassessment_20260908 に保存。数値実装と解選択が変わっているため、新旧の差を生物学的効果と解釈しない。
関連文書: docs/CULTIVATION_MODEL_AUDIT_20260908.md、docs/CULTIVATION_MODEL_REPAIR_20260908.md、docs/AUDITED_SYMBIOSIS_INTERPRETATION_20260908.md。
'''
    (OUT/'REPORT_JA.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(paired=paired,refinement=finer,
        conditions_passing_screen_at_both_steps=passes_both_steps,numerical_convergence_established=False,
        require_finer=need_finer),indent=2,allow_nan=False))
if __name__=='__main__':main()
