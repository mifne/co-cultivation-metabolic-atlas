"""Summarize all preregistered-in-file Pf audit conditions, including failures."""
from pathlib import Path
import csv,hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/pfreud_value_audit_20260908'

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    design=json.loads((OUT/'design.json').read_text())
    progress=json.loads((OUT/'progress.json').read_text())
    if len(progress)!=len(design['cases']):raise RuntimeError('Audit not complete')
    records={};flat=[];failures=[]
    for case in design['cases']:
        f=OUT/case['id']/'result.json'
        if not f.exists():failures.append(case);continue
        d=json.loads(f.read_text());records[case['id']]=d
        s=d['summary'];trajectory=list(csv.DictReader((f.parent/'trajectory.csv').open()))
        assert len(trajectory)==round(12/case['dt'])+1
        assert abs(float(trajectory[-1]['time_h'])-12)<1e-8
        assert abs(s['cumulative_feed_mmol_l']-12*case['rate'])<1e-8
        assert abs(s['final_pha_g_l']-(s['final_phb_mmol_l']*.08609+s['final_phv_mmol_l']*.10012))<1e-10
        if case['arm']=='three':assert abs(sum(d['initial']['initial_biomass'].values())-.63)<1e-10
        elif case['arm']=='two_equal_total':assert abs(sum(d['initial']['initial_biomass'].values())-.63)<1e-10
        else:assert abs(sum(d['initial']['initial_biomass'].values())-.60)<1e-10
        flat.append(dict(**case,pha_g_l=s['final_pha_g_l'],phv_mol_percent=100*s['final_phv_mol_fraction'],rubber_removed_g_l=s['rubber_removed_g_l'],helper_lactate_uptake=s['helper_lactate_uptake_mmol_l'],helper_ppa_secretion=s['helper_propionate_secretion_mmol_l'],ns21_ppa_uptake=s['ns21_propionate_uptake_mmol_l'],base_added=d['diagnostics']['ph_control']['base_added_mmol_l'],acid_added=d['diagnostics']['ph_control']['acid_added_mmol_l'],solver_success=s['solver_success_rate'],seconds=d['seconds']))
    comparisons=[]
    for row in flat:
        if row['arm']!='three':continue
        comp={k:row[k] for k in ['id','group','dt','rate','nh4','mode']}
        for arm in ['two_fixed','two_equal_total']:
            matches=[r for r in flat if r['arm']==arm and all(r[k]==row[k] for k in ['group','dt','rate','nh4','mode'])]
            if not matches:continue
            two=matches[0]
            assert records[two['id']]['initial']['initial_medium']==records[row['id']]['initial']['initial_medium']
            comp[f'pha_gain_pct_vs_{arm}']=100*(row['pha_g_l']/two['pha_g_l']-1) if two['pha_g_l'] else None
            comp[f'rubber_gain_pct_vs_{arm}']=100*(row['rubber_removed_g_l']/two['rubber_removed_g_l']-1) if two['rubber_removed_g_l'] else None
            comp[f'phv_pp_vs_{arm}']=row['phv_mol_percent']-two['phv_mol_percent']
            comp[f'gate_vs_{arm}']=bool(two['solver_success']==row['solver_success']==1. and comp[f'pha_gain_pct_vs_{arm}'] is not None and comp[f'pha_gain_pct_vs_{arm}']>=5 and comp[f'rubber_gain_pct_vs_{arm}']>=-5)
        comp['passes_both_controls']=all(comp.get(f'gate_vs_{arm}',False) for arm in ['two_fixed','two_equal_total'])
        comparisons.append(comp)
    csvwrite(OUT/'summary.csv',flat);csvwrite(OUT/'paired_comparisons.csv',comparisons)
    legacy=list(csv.DictReader((ROOT/'results/third_species_phbv_reassessment_20260903/scenario_summary.csv').open(encoding='utf-8-sig')))
    old=[]
    for row in legacy:
        if row['helper']!='True':continue
        rate=float(row['feed_rate_mmol_l_h']);two=next(r for r in legacy if r['name']==(f'two_lactate_{rate:.2f}' if rate else 'two_rubber_only'))
        old.append(dict(name=row['name'],matched_two=two['name'],gain_percent=100*(float(row['final_pha_g_l'])/float(two['final_pha_g_l'])-1)))
    csvwrite(OUT/'legacy_all_helper_comparisons.csv',old)
    integrity=dict(planned=len(design['cases']),completed=len(flat),failures=failures,all_solver_success=all(r['solver_success']==1 for r in flat),checks='All trajectories reach 12h; expected rows; feed integral; PHA mass conversion; matched media; inoculum totals',source_hashes_unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in design['sources'].items()))
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    a,b,c,d=axes.flat
    colors={'two_fixed':'#0072B2','two_equal_total':'#009E73','three':'#D55E00'}
    a.barh(range(len(old)),[r['gain_percent'] for r in old],color=['#D55E00' if r['gain_percent']>0 else '#777777' for r in old]);a.set_yticks(range(len(old)),[r['name'].replace('three_','') for r in old],fontsize=8);a.axvline(0,color='black',lw=.8);a.set_xlabel('PHA change vs matched two-species feed control (%)');a.set_title('A. All legacy helper conditions (dt = 1 h)')
    for arm in colors:
        rows=sorted([r for r in flat if r['group']=='time_step' and r['arm']==arm],key=lambda x:-x['dt'])
        b.plot(range(len(rows)),[r['pha_g_l'] for r in rows],'o-',label=arm,color=colors[arm]);b.set_xticks(range(len(rows)),[str(r['dt']) for r in rows])
    b.set_xlabel('Integration time step (h)');b.set_ylabel('PHA model output (g/L)');b.set_title('B. Numerical sensitivity of the +20% candidate');b.legend(frameon=False,fontsize=8)
    local=[r for r in comparisons if r['group']=='local_sensitivity']
    for j,arm in enumerate(['two_fixed','two_equal_total']):
        c.bar(np.arange(len(local))+(j-.5)*.35,[r[f'pha_gain_pct_vs_{arm}'] for r in local],width=.35,color=colors[arm],label=f'vs {arm}')
    c.set_xticks(range(len(local)),[f'r={r["rate"]}\nN={r["nh4"]}' for r in local]);c.axhline(0,color='black',lw=.8);c.axhline(5,color='gray',lw=.8,ls='--');c.set_ylabel('PHA change (%)');c.set_title('C. Nearby conditions at dt = 0.2 h');c.legend(frameon=False,fontsize=8)
    for j,(key,label) in enumerate([('helper_lactate_uptake','Pf lactate uptake'),('helper_ppa_secretion','Pf propionate secretion'),('ns21_ppa_uptake','NS21 propionate uptake')]):
        rows=sorted([r for r in flat if r['group']=='time_step' and r['arm']=='three'],key=lambda x:-x['dt'])
        d.bar(np.arange(len(rows))+(j-1)*.25,[r[key] for r in rows],width=.25,label=label)
    d.set_xticks(range(len(rows)),[str(r['dt']) for r in rows]);d.set_xlabel('Integration time step (h)');d.set_ylabel('Integrated exchange (mmol/L)');d.set_title('D. Proposed crossfeeding mechanism');d.legend(frameon=False,fontsize=8)
    fig.suptitle('P. freudenreichii value audit | computational screening, not biological significance',fontsize=13)
    fig.savefig(OUT/'pfreud_value_audit.png',dpi=180);fig.savefig(OUT/'pfreud_value_audit.pdf');plt.close(fig)
    print(json.dumps(dict(integrity=integrity,comparisons=comparisons),indent=2))
if __name__=='__main__':main()

