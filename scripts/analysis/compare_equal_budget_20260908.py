"""Equal finite feed search, followed by symmetric refinement. No teacher collection."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,csv,time,hashlib,shutil,traceback,argparse,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/equal_budget_comparison_20260908'
OR='Actinoplanes_sp_OR16_lcp';NS='Rhizobacter_gummiphilus_NS21';PF='Propionibacterium_freudenreichii_shermanii'
KLAS=[2.,10.,50.]
PROFILES=[dict(lactate=l,nitrogen=n) for l in ['uniform','early','late'] for n in ['early','spread']]

def write(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)

def rates(t,profile):
    l=profile['lactate'];n=profile['nitrogen']
    lactate=.25 if l=='uniform' else (.5 if (t<12.-1e-10)==(l=='early') else 0.)
    ammonium=(.1 if t<4.-1e-10 else 0.) if n=='early' else ((.4/12.) if t<12.-1e-10 else 0.)
    return {'lac__L_e':lactate,'nh4_e':ammonium}

def case(kla,profile,arm,phase):
    return dict(id=f'{phase}_k{kla:g}_{profile["lactate"]}_{profile["nitrogen"]}_{arm}',
        kla=kla,profile=profile,arm=arm,phase=phase,dt=.025 if phase=='screen' else .0125,hours=24.)

def source_hashes(paths):return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}

def run_case(c):
    from main import load_requested_models
    from src.physiology_dfba import PhysiologyDFBASimulator
    import math
    p=OUT/c['id'];p.mkdir(exist_ok=True);start=time.monotonic();sim=None;rows=[]
    try:
        design=json.loads((OUT/'design.json').read_text())
        models=load_requested_models(None,'pf-helper3')
        if c['arm']=='two':models.pop(PF)
        biomass={OR:.5,NS:.1,PF:.03} if c['arm']=='three' else {OR:.525,NS:.105}
        config=dict(maintenance={n:dict(reaction='rxn00062_c0' if n==PF else 'ATPM',rate_mmol_g_h=.1) for n in models},
            basal_death_rates={n:.01 for n in models},starvation_death_rates={n:.1 for n in models},
            nitrogen_policy='capacity_ratio',remobilize_pha=True)
        medium=design['common_initial_medium']
        sim=PhysiologyDFBASimulator(models,biomass,medium,initial_rubber=10.,dt=.25,
            max_internal_dt=c['dt'],ph_control_target=7.,**config)
        write(p/'settings.json',dict(case=c,initial_biomass=biomass,initial_medium=medium,config=config,
            oxygen_saturation_mmol_l=sim.oxygen_saturation,gas_transfer_capacity_integral=c['kla']*24.*sim.oxygen_saturation,
            equal_actual_oxygen_consumption_claim=False,biological_parameters_calibrated=False))
        def row():
            a=sim.accounting_audit;e=sim.element_accounting();species=sim.state.species
            hb=sum(s.phb_accumulated for s in species.values());hv=sum(s.phv_accumulated for s in species.values())
            live=hb*sim.PHB_REPEAT_G_PER_MMOL+hv*sim.PHV_REPEAT_G_PER_MMOL
            dead=sum(d['phb_mmol_l']*sim.PHB_REPEAT_G_PER_MMOL+d['phv_mmol_l']*sim.PHV_REPEAT_G_PER_MMOL for d in sim.dead_matter.values())
            r=dict(time_h=sim.state.time,pha_live_g_l=live,pha_dead_g_l=dead,pha_total_g_l=live+dead,
                hv_mol_fraction=hv/(hb+hv) if hb+hv>1e-9 else 0.,rubber_removed_g_l=10.-sim.state.rubber_concentration,
                nh4_mmol_l=sim.state.metabolites.get('nh4_e',0.),o2_mmol_l=sim.state.metabolites.get('o2_e',0.),
                lactate_delivered=sim.cumulative_delivered_mmol_l.get('lac__L_e',0.),
                nh4_delivered=sim.cumulative_delivered_mmol_l.get('nh4_e',0.),
                oxygen_transferred=sim.oxygen_audit['transferred'],oxygen_cellular=sim.oxygen_audit['cellular_consumed'],
                oxygen_polymer=sim.oxygen_audit['polymer_consumed'],
                C_residual=e['known_medium_closure_residual']['C'],N_residual=e['known_medium_closure_residual']['N'],
                oxygen_balance_error=sim.oxygen_audit['max_balance_error'],oxygen_kinetic_error=a['max_oxygen_kinetic_residual'],
                storage_excess=a['max_storage_excess_g_l'],lp_residual=a['max_lp_residual'])
            for n in models:
                short='OR16' if n==OR else 'NS21' if n==NS else 'Pf'
                r[short+'_live_g_l']=species[n].biomass;r[short+'_dead_g_l']=sim.dead_matter[n]['biomass_g_l']
            assert all(math.isfinite(v) for v in r.values())
            assert all(v>=-1e-8 for k,v in r.items() if k.endswith('_g_l') or k.endswith('_mmol_l'))
            assert abs(r['C_residual'])<1e-6 and abs(r['N_residual'])<1e-6
            assert r['storage_excess']<=1e-7 and r['lp_residual']<=1e-7
            assert r['oxygen_kinetic_error']<=1e-6 and r['oxygen_balance_error']<=1e-8
            assert r['lactate_delivered']<=6.+1e-8 and r['nh4_delivered']<=.4+1e-8
            assert r['oxygen_transferred']<=c['kla']*sim.state.time*sim.oxygen_saturation+1e-8
            return r
        rows.append(row())
        with (p/'trajectory.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerow(rows[0]);f.flush()
            while sim.state.time<24.-1e-10:
                sim.step({}, {},dynamic_kla=c['kla'],feed_rates_mmol_l_h=rates(sim.state.time,c['profile']))
                r=row();rows.append(r);writer.writerow(r);f.flush()
                write(p/'progress.json',dict(id=c['id'],status='running',hours=sim.state.time,seconds=time.monotonic()-start,pha_g_l=r['pha_live_g_l']))
        a=sim.accounting_audit
        assert abs(rows[-1]['lactate_delivered']-6.)<1e-8 and abs(rows[-1]['nh4_delivered']-.4)<1e-8
        assert a['failed_steps']==0 and a['max_dual_residual']<=1e-7 and a['max_relative_duality_gap']<=1e-7
        assert a['lp_requests']==a['lp_certified_requests']+a.get('superseded_uncertified_requests',0)
        result=dict(case=c,final=rows[-1],peak_pha_g_l=max(r['pha_live_g_l'] for r in rows),
            seconds=time.monotonic()-start,diagnostics=sim.get_solver_diagnostics())
        write(p/'result.json',result)
        status=dict(id=c['id'],status='complete',hours=sim.state.time,seconds=result['seconds'],pha_g_l=rows[-1]['pha_live_g_l'])
        write(p/'progress.json',status);return status
    except Exception as ex:
        fail=dict(id=c['id'],status='failed',error=repr(ex),traceback=traceback.format_exc(),
            hours=sim.state.time if sim else 0.,seconds=time.monotonic()-start,
            diagnostics=sim.get_solver_diagnostics() if sim else None)
        write(p/'failure.json',fail);status={k:v for k,v in fail.items() if k not in ['traceback','diagnostics']}
        write(p/'progress.json',status);return status

def run_phase(cases,workers,design,phase):
    assert source_hashes(design['source_sha256'])==design['source_sha256']
    statuses=[];pending=[]
    for c in cases:
        p=OUT/c['id']/'progress.json'
        if p.exists():
            s=json.loads(p.read_text())
            if s['status']=='complete':statuses.append(s);continue
            raise RuntimeError(f'Refuse to overwrite retained incomplete case {c["id"]}')
        pending.append(c)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_case,c) for c in pending]):
            s=future.result();statuses.append(s);write(OUT/f'{phase}_progress.json',statuses);print(json.dumps(s),flush=True)
    assert source_hashes(design['source_sha256'])==design['source_sha256']
    write(OUT/f'{phase}_execution.json',dict(source_sha256=design['source_sha256'],results=statuses))
    if any(s['status']!='complete' for s in statuses):raise RuntimeError('Retained failures require investigation')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    screen=[case(k,p,a,'screen') for k in KLAS for p in PROFILES for a in ['two','three']]
    if not (OUT/'design.json').exists():
        from main import load_requested_models
        from src.utils import get_initial_params
        _,medium=get_initial_params(load_requested_models(None,'pf-helper3'))
        medium.update(glc__D_e=0.,yeast_extract_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=.05)
        old=json.loads((ROOT/'results/physiology_long_validation_20260908/design.json').read_text())
        paths=list(old['source_sha256'])+['scripts/analysis/compare_equal_budget_20260908.py']
        design=dict(screen_cases=screen,common_initial_medium=medium,source_sha256=source_hashes(paths),
            budget=dict(lactate_mmol_l=6.,added_nh4_mmol_l=.4,rubber_g_l=10.,initial_total_biomass_g_l=.63,
                oxygen='Same fixed kLa and saturation per stratum: equal transfer capacity, not equal realized consumption'),
            objective='Maximize final live-cell PHA at 24 h independently in each consortium and kLa stratum',
            selection='Union of top 2 profiles and profiles within 4 percent of best in either arm; refine this identical union in both arms',
            criteria=dict(pha_relative_refinement_tolerance=.02,biomass_relative_refinement_tolerance=.02,
                engineering_gain_threshold=.05,maximum_rubber_removal_loss=.05),
            limitations=['Finite six-profile grid; fixed inoculum ratios; no global optimality',
                'Same background medium and pH target; acid/base operating cost not modeled',
                'No calibrated biology, stable coexistence, or rubber-derived PHA claim'])
        write(OUT/'design.json',design);(OUT/'sources').mkdir(exist_ok=True)
        for p in paths:shutil.copy2(ROOT/p,OUT/'sources'/p.replace('/','__'))
    design=json.loads((OUT/'design.json').read_text());assert design['screen_cases']==screen
    run_phase(screen,args.workers,design,'screen')
    selected=[];selection=[]
    for k in KLAS:
        union=set()
        for arm in ['two','three']:
            rr=[json.loads((OUT/c['id']/'result.json').read_text()) for c in screen if c['kla']==k and c['arm']==arm]
            rr.sort(key=lambda r:(-r['final']['pha_live_g_l'],r['case']['id']))
            chosen=[r for i,r in enumerate(rr) if i<2 or r['final']['pha_live_g_l']>=.96*rr[0]['final']['pha_live_g_l']]
            union.update((r['case']['profile']['lactate'],r['case']['profile']['nitrogen']) for r in chosen)
            selection.append(dict(kla=k,arm=arm,ranked_ids=[r['case']['id'] for r in rr],selected_ids=[r['case']['id'] for r in chosen]))
        for l,n in sorted(union):
            for arm in ['two','three']:selected.append(case(k,dict(lactate=l,nitrogen=n),arm,'refine'))
    write(OUT/'selection.json',dict(selection=selection,refinement_cases=selected))
    run_phase(selected,args.workers,design,'refine')
    write(OUT/'execution_verification.json',dict(source_sha256=source_hashes(design['source_sha256']),
        screen_count=len(screen),refinement_count=len(selected),all_completed=True))
    print('Equal-budget search and symmetric refinement complete.',flush=True)

if __name__=='__main__':main()
