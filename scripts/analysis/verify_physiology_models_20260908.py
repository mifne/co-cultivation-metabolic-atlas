"""Small real-GEM integration and assumption sensitivity checks, not efficacy evidence."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from main import load_requested_models
from src.utils import get_initial_params
from src.physiology_dfba import PhysiologyDFBASimulator

OUT=ROOT/'results/physiology_implementation_20260908';OUT.mkdir(exist_ok=True)
sources=['src/physiology_dfba.py','src/b12_evidence.py','src/audited_dfba.py','src/resolved_dfba.py',
         'src/cultivation_numerics.py','src/rl_environment.py','src/one_l_jar.py','main.py',
         'scripts/analysis/verify_physiology_models_20260908.py']
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
models=load_requested_models(None,'pf-helper3');names=list(models)
or16=next(n for n in names if 'OR16' in n);ns21=next(n for n in names if 'NS21' in n);pf=next(n for n in names if 'freudenreichii' in n)
_,base=get_initial_params(models)
base.update(glc__D_e=0.,yeast_extract_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=.05)
cases=[('three_no_maintenance',0.,.025,.05,True),('three_low_maintenance',.1,.025,.05,True),
       ('three_high_maintenance',1.,.025,.05,True),('three_no_nh4',.1,.025,0.,True),
       ('two_low_maintenance',.1,.025,.05,False),('three_fine',.1,.0125,.05,True),
       ('three_finer',.1,.00625,.05,True),('three_finest',.1,.003125,.05,True)]
summary=[]
for cid,rate,dt,nh4,three in cases:
    selected=models if three else {n:m for n,m in models.items() if n!=pf}
    biomass={or16:.5,ns21:.1,pf:.03} if three else {or16:.525,ns21:.105}
    medium=dict(base,nh4_e=nh4)
    config=dict(nitrogen_policy='capacity_ratio',remobilize_pha=True,
        maintenance={n:dict(reaction='rxn00062_c0' if n==pf else 'ATPM',rate_mmol_g_h=rate) for n in selected} if rate else {},
        basal_death_rates={n:.01 for n in selected},starvation_death_rates={n:.1 for n in selected} if rate else {})
    started=time.monotonic()
    sim=PhysiologyDFBASimulator(selected,biomass,medium,dt=.1,max_internal_dt=dt,
        initial_rubber=10.,ph_control_target=7.,**config)
    sim.step({}, {},dynamic_kla=10.,feed_rates_mmol_l_h={'lac__L_e':.25})
    diag=sim.get_solver_diagnostics();acc=diag['audited_cultivation']['accounting']
    assert abs(sim.state.time-.1)<1e-12 and diag['audited_cultivation']['valid']
    assert acc['max_lp_residual']<=1e-7 and acc['max_oxygen_kinetic_residual']<=1e-6
    assert max(abs(v) for v in sim.element_accounting()['known_medium_closure_residual'].values())<1e-7
    assert all(s.biomass>=0 and s.phb_accumulated>=-1e-9 and s.phv_accumulated>=-1e-9 for s in sim.state.species.values())
    result=dict(id=cid,config=config,internal_dt=dt,horizon_h=.1,seconds=time.monotonic()-started,
        initial_biomass=biomass,initial_medium=medium,diagnostics=diag,
        pha_g_l=sum(s.phb_accumulated*sim.PHB_REPEAT_G_PER_MMOL+s.phv_accumulated*sim.PHV_REPEAT_G_PER_MMOL for s in sim.state.species.values()),
        final_biomass={n:s.biomass for n,s in sim.state.species.items()},
        nitrogen_allocation=sim.nitrogen_allocation,final_medium=sim.state.metabolites)
    (OUT/(cid+'.json')).write_text(json.dumps(result,indent=2))
    summary.append({k:result[k] for k in ['id','seconds','pha_g_l','final_biomass','nitrogen_allocation']})
    print('DONE',cid,round(result['seconds'],2),flush=True)
    if cid=='three_low_maintenance':
        (OUT/'example_uncalibrated_physiology.json').write_text(json.dumps(config,indent=2))
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in hashes.items()),'source changed during validation'
(OUT/'verification_summary.json').write_text(json.dumps(dict(cases=summary,source_sha256=hashes,
    assumptions_calibrated=False,production_or_coexistence_validated=False),indent=2))
print('All eight real-GEM checks passed.',flush=True)
