"""Independent fixed-carbon audit; composition convergence is a separate gate."""
from pathlib import Path
import json,csv,hashlib,math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/direct_propionate_comparison_20260909'
def read(p):return json.loads(p.read_text())
def main():
    d=read(OUT/'design.json');e=read(OUT/'execution_verification.json')
    assert e['all_completed'] and e['count']==len(d['cases'])
    assert e['source_sha256']==d['source_sha256']=={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in d['source_sha256']}
    results={};series={}
    for c in d['cases']:
        p=OUT/c['id'];r=read(p/'result.json');settings=read(p/'settings.json')
        assert r['case']==c==settings['case']
        assert settings['initial_medium']==d['common_initial_medium']
        assert abs(sum(settings['initial_biomass'].values())-.63)<1e-12
        with (p/'trajectory.csv').open() as f:rows=[{k:float(v) for k,v in x.items()} for x in csv.DictReader(f)]
        assert len(rows)==97 and rows[-1]==r['final']
        for i,x in enumerate(rows):
            assert abs(x['time_h']-.25*i)<1e-9
            assert all(math.isfinite(v) for v in x.values())
            assert all(v>=-1e-8 for k,v in x.items() if k.endswith('_g_l') or k.endswith('_mmol_l'))
            t=x['time_h'];pr=c['profile'];fraction=pr['propionate_fraction']
            total=.25*t if pr['lactate']=='uniform' else .5*(min(t,12) if pr['lactate']=='early' else max(0,t-12))
            assert abs(x['lactate_delivered']-total*(1-fraction))<1e-8
            assert abs(x['propionate_delivered']-total*fraction)<1e-8
            assert abs(x['nh4_delivered']-.4*min(1,t/(4 if pr['nitrogen']=='early' else 12)))<1e-8
            assert x['oxygen_transferred']<=c['kla']*t*settings['oxygen_saturation_mmol_l']+1e-8
            assert max(abs(x[k]) for k in ['C_residual','N_residual'])<1e-6
            assert x['oxygen_balance_error']<=1e-8 and x['oxygen_kinetic_error']<=1e-6
            assert x['storage_excess']<=1e-7
        a=r['diagnostics']['audited_cultivation']['accounting']
        assert a['failed_steps']==0 and max(a[k] for k in ['max_lp_residual','max_dual_residual','max_relative_duality_gap'])<=1e-7
        assert a['lp_requests']==a['lp_certified_requests']+a.get('superseded_uncertified_requests',0)
        results[c['id']]=r;series[c['id']]=rows
    refinement=[]
    for c in d['cases']:
        if c['phase']!='refine':continue
        cid=c['id'];coarse=cid.replace('refine_','screen_',1);a=results[coarse]['final'];b=results[cid]['final']
        errors={k:abs(a[k]-b[k])/max(floor,abs(b[k])) for k,floor in [('pha_live_g_l',.001),('phv_live_g_l',.0001)]}
        bio=max(abs(a[k]-v)/max(1e-6,abs(v)) for k,v in b.items() if k.endswith('_live_g_l') and k not in ['pha_live_g_l','phv_live_g_l'])
        hv=abs(a['hv_mol_fraction']-b['hv_mol_fraction'])
        refinement.append(dict(id=cid,pha_error=errors['pha_live_g_l'],phv_error=errors['phv_live_g_l'],biomass_error=bio,hv_fraction_error=hv,
            passed=max(errors.values())<=.02 and bio<=.02 and hv<=.001))
    best=[]
    for kla in [2.,10.]:
        for arm in ['three','two_propionate','two_mixed']:
            rr=[r for r in results.values() if r['case']['phase']=='refine' and r['case']['kla']==kla and r['case']['arm']==arm]
            assert len(rr)==6
            for metric in ['pha_live_g_l','phv_live_g_l']:
                r=max(rr,key=lambda r:r['final'][metric]);best.append(dict(kla=kla,arm=arm,objective=metric,id=r['case']['id'],final=r['final']))
    needs=[r['id'] for r in refinement if not r['passed']]
    summary=dict(completed=len(results),best=best,refinement=refinement,needs_additional_refinement=needs,limitations=d['limitations'],biological_validation=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,k in zip(axs,[2.,10.]):
        for arm in ['three','two_propionate','two_mixed']:
            rr=[r for r in results.values() if r['case']['phase']=='refine' and r['case']['kla']==k and r['case']['arm']==arm]
            ax.scatter([100*r['final']['hv_mol_fraction'] for r in rr],[r['final']['pha_live_g_l'] for r in rr],label=arm)
        ax.set(xlabel='3HV (mol%)',ylabel='Live PHA (g/L)',title=f'kLa={k:g}; all six profiles');ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    table='\n'.join(f"|{r['kla']:g}|{r['arm']}|{r['objective']}|{r['final']['pha_live_g_l']:.6f}|{r['final']['phv_live_g_l']:.6f}|{100*r['final']['hv_mol_fraction']:.3f}|" for r in best)
    (OUT/'REPORT_JA.md').write_text('# プロピオン酸直接添加との比較\n\n全72ケース。各構成6供給条件、kLa 2/10、内部刻み0.025/0.0125 h。\n\n|kLa|構成|選択指標|PHA g/L|3HV単位質量 g/L|3HV mol%|\n|---|---|---|---|---|---|\n'+table+'\n\n追加細分対象：'+str(needs)+'\n\n未達があれば組成・優位性を確定しない。投入炭素は同じだが還元当量・費用は同じではない。阻害・Pf酸素表現型・PHV生成制御は未校正。3HV量はPHA内のHV単位質量であり、別個のPHVホモポリマーの実証ではない。\n',encoding='utf-8')
    print(json.dumps(dict(completed=len(results),needs_additional_refinement=needs,best=best),indent=2))
if __name__=='__main__':main()
