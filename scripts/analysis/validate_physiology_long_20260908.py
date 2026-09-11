"""Frozen, matched long-horizon physiology validation. Never collect teachers."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,csv,hashlib,time,traceback,shutil,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/physiology_long_validation_20260908'
OR='Actinoplanes_sp_OR16_lcp';NS='Rhizobacter_gummiphilus_NS21';PF='Propionibacterium_freudenreichii_shermanii'

CASES=[
    dict(id='rich_three',arm='three',dt=.025,hours=24.),
    dict(id='rich_two',arm='two',dt=.025,hours=24.),
    dict(id='rich_three_fine',arm='three',dt=.0125,hours=24.),
    dict(id='rich_two_fine',arm='two',dt=.0125,hours=24.),
    dict(id='rich_three_coarse',arm='three',dt=.05,hours=24.),
    dict(id='rich_three_high_maintenance',arm='three',dt=.025,hours=24.,maintenance=1.),
    dict(id='rich_three_high_death',arm='three',dt=.025,hours=24.,death=.02,starvation=.2),
    dict(id='rich_three_feed_stop',arm='three',dt=.025,hours=24.,feed_stop=12.),
    dict(id='rich_three_nh4_staged',arm='three',dt=.025,hours=24.,nh4_until=4.),
    dict(id='n_depleted_three',arm='three',dt=.025,hours=12.,n_depleted=True),
    dict(id='n_depleted_three_fine',arm='three',dt=.0125,hours=12.,n_depleted=True),
]

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def run_case(case):
    from main import load_requested_models
    from src.utils import get_initial_params
    from src.physiology_dfba import PhysiologyDFBASimulator
    path=OUT/case['id'];path.mkdir(exist_ok=True)
    started=time.monotonic();sim=None;rows=[]
    try:
        models=load_requested_models(None,'pf-helper3')
        if case['arm']=='two':models.pop(PF)
        biomass={OR:.5,NS:.1,PF:.03} if case['arm']=='three' else {OR:.525,NS:.105}
        _,medium=get_initial_params(models)
        medium.update(glc__D_e=0.,yeast_extract_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=.05)
        config=dict(maintenance={n:dict(reaction='rxn00062_c0' if n==PF else 'ATPM',rate_mmol_g_h=case.get('maintenance',.1)) for n in models},
            basal_death_rates={n:case.get('death',.01) for n in models},
            starvation_death_rates={n:case.get('starvation',.1) for n in models},
            nitrogen_policy='capacity_ratio',remobilize_pha=True)
        # Formula audit comes from the actual input GEM mapping. The deliberate
        # N-depleted control also removes unclassified nonzero pools so unknown
        # nitrogen is not silently counted as absent. This is a hypothetical
        # washed-cell condition, not a validated minimal growth medium.
        template=PhysiologyDFBASimulator(models,biomass,medium,initial_rubber=10.,dt=.25,
            max_internal_dt=case['dt'],ph_control_target=7.,**config)
        removed={}
        if case.get('n_depleted'):
            for p,v in list(medium.items()):
                if v and (p not in template.pool_elements or template.pool_elements[p].get('N',0)>0):
                    removed[p]=v;medium[p]=0.
            sim=PhysiologyDFBASimulator(models,biomass,medium,initial_rubber=10.,dt=.25,
                max_internal_dt=case['dt'],ph_control_target=7.,**config)
        else:sim=template
        del template
        write(path/'settings.json',dict(case=case,initial_biomass=biomass,initial_medium=medium,
            config=config,removed_for_n_depleted_control=removed,kla_h_inv=10.,lactate_rate_mmol_l_h=.25,
            carbon_origin_identified=False,biological_parameters_calibrated=False))
        def row():
            e=sim.element_accounting();a=sim.accounting_audit
            live=sum(s.phb_accumulated*sim.PHB_REPEAT_G_PER_MMOL+s.phv_accumulated*sim.PHV_REPEAT_G_PER_MMOL for s in sim.state.species.values())
            dead=sum(d['phb_mmol_l']*sim.PHB_REPEAT_G_PER_MMOL+d['phv_mmol_l']*sim.PHV_REPEAT_G_PER_MMOL for d in sim.dead_matter.values())
            hv=sum(s.phv_accumulated for s in sim.state.species.values());hb=sum(s.phb_accumulated for s in sim.state.species.values())
            result=dict(time_h=sim.state.time,pha_live_g_l=live,pha_dead_g_l=dead,pha_total_g_l=live+dead,
                hv_mol_fraction=hv/(hv+hb) if hv+hb>1e-9 else 0.,rubber_g_l=sim.state.rubber_concentration,
                nh4_mmol_l=sim.state.metabolites.get('nh4_e',0.),o2_mmol_l=sim.state.metabolites.get('o2_e',0.),
                lactate_mmol_l=sim.state.metabolites.get('lac__L_e',0.),propionate_mmol_l=sim.state.metabolites.get('ppa_e',0.),
                known_C_mmol_l=e['current_known']['C'],known_N_mmol_l=e['current_known']['N'],
                C_ledger_residual=e['known_medium_closure_residual']['C'],N_ledger_residual=e['known_medium_closure_residual']['N'],
                oxygen_balance_error=sim.oxygen_audit['max_balance_error'],oxygen_kinetic_error=a['max_oxygen_kinetic_residual'],
                lp_residual=a['max_lp_residual'],storage_excess=a['max_storage_excess_g_l'],
                ns21_growth_fraction=sim.nitrogen_allocation.get(NS,{}).get('growth_fraction',1.),
                lactate_delivered=sim.cumulative_delivered_mmol_l.get('lac__L_e',0.),
                nh4_delivered=sim.cumulative_delivered_mmol_l.get('nh4_e',0.))
            for n in models:
                short='OR16' if n==OR else 'NS21' if n==NS else 'Pf'
                result[short+'_live_g_l']=sim.state.species[n].biomass
                result[short+'_dead_g_l']=sim.dead_matter[n]['biomass_g_l']
                result[short+'_maintenance_deficit']=sim.physiology_trials.get(n,{}).get('deficit_fraction',0.)
                result[short+'_phb_reused_mmol_l']=sim.physiology_totals[n]['phb_remobilized_mmol_l']
            import math
            assert all(math.isfinite(v) for v in result.values())
            assert min(live,dead,sim.state.rubber_concentration)>=-1e-8
            assert max(abs(result[k]) for k in ['C_ledger_residual','N_ledger_residual'])<1e-6
            assert a['max_lp_residual']<=1e-7 and a['max_storage_excess_g_l']<=1e-7
            assert a['max_oxygen_kinetic_residual']<=1e-6 and sim.oxygen_audit['max_balance_error']<=1e-8
            assert min(v for k,v in result.items() if k.endswith('_live_g_l') or k.endswith('_dead_g_l'))>=-1e-8
            return result
        first=row();rows.append(first)
        with (path/'trajectory.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=first);writer.writeheader();writer.writerow(first);f.flush()
            while sim.state.time<case['hours']-1e-10:
                rates={'lac__L_e':.25 if sim.state.time<case.get('feed_stop',1e6)-1e-10 else 0.}
                if sim.state.time<case.get('nh4_until',0.)-1e-10:rates['nh4_e']=.1
                sim.step({}, {},dynamic_kla=10.,feed_rates_mmol_l_h=rates)
                r=row();rows.append(r);writer.writerow(r);f.flush()
                write(path/'progress.json',dict(status='running',hours=sim.state.time,target_hours=case['hours'],
                    seconds=time.monotonic()-started,lp_attempts=sim.solve_attempts,pha_g_l=r['pha_live_g_l']))
        assert abs(sim.cumulative_delivered_mmol_l['lac__L_e']-.25*min(case['hours'],case.get('feed_stop',1e6)))<1e-8
        assert abs(sim.cumulative_delivered_mmol_l.get('nh4_e',0.)-.1*case.get('nh4_until',0.))<1e-8
        result=dict(case=case,final=rows[-1],peak_pha_g_l=max(r['pha_live_g_l'] for r in rows),
            accumulation_onset_h=next((r['time_h'] for r in rows if r['pha_live_g_l']>1e-3),None),
            seconds=time.monotonic()-started,diagnostics=sim.get_solver_diagnostics())
        write(path/'result.json',result)
        write(path/'progress.json',dict(status='complete',hours=sim.state.time,target_hours=case['hours'],seconds=result['seconds'],pha_g_l=result['final']['pha_live_g_l']))
        return dict(id=case['id'],status='complete',peak_pha_g_l=result['peak_pha_g_l'],seconds=result['seconds'])
    except Exception as ex:
        failure=dict(id=case['id'],status='failed',error=repr(ex),traceback=traceback.format_exc(),seconds=time.monotonic()-started,
            hours=sim.state.time if sim else 0.,diagnostics=sim.get_solver_diagnostics() if sim else None)
        write(path/'failure.json',failure);write(path/'progress.json',{k:v for k,v in failure.items() if k not in {'diagnostics','traceback'}})
        return {k:v for k,v in failure.items() if k not in {'diagnostics','traceback'}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if any((OUT/c['id']/'progress.json').exists() for c in CASES):raise RuntimeError('Refuse to overwrite an existing long cohort')
    sources=['src/physiology_dfba.py','src/audited_dfba.py','src/resolved_dfba.py','src/dfba_simulator.py',
        'src/cultivation_numerics.py','src/metabolite_ids.py','src/b12_evidence.py','src/rl_environment.py',
        'src/one_l_jar.py','src/utils.py','main.py','scripts/analysis/validate_physiology_long_20260908.py',
        'models/genome/NS21.faa','models/genome/AP019371.1.faa',
        'models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml',
        'models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml',
        'models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    write(OUT/'design.json',dict(cases=CASES,source_sha256=hashes,
        criteria=dict(pha_accumulation_threshold_g_l=.001,matched_endpoint_pha_relative_tolerance=.02,
            matched_peak_pha_relative_tolerance=.02,relative_pha_denominator_floor_g_l=.001,
            matched_species_biomass_relative_tolerance=.02,hv_absolute_fraction_tolerance=.001,
            carbon_nitrogen_ledger_absolute_tolerance=1e-6,lp_residual_tolerance=1e-7),
        interpretation='Numerical qualification only; physiological assumptions remain uncalibrated.'))
    snapshots=OUT/'sources';snapshots.mkdir(exist_ok=True)
    for p in sources:shutil.copy2(ROOT/p,snapshots/p.replace('/','__'))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(run_case,c) for c in CASES]
        for future in as_completed(futures):
            result=future.result();results.append(result);write(OUT/'progress.json',results)
            print(json.dumps(result),flush=True)
    current={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    assert current==hashes,'Sources changed during execution'
    write(OUT/'execution_verification.json',dict(source_sha256=current,results=results))
    if any(r['status']!='complete' for r in results):raise RuntimeError('Retained case failures require investigation')
    print('All long cases complete.',flush=True)

if __name__=='__main__':main()
