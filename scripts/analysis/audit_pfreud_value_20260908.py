"""Prespecified numerical and inoculum audit of an exploratory Pf benefit.
Run isolated subprocesses; preserve failures and all conditions, no p-values.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def cases():
    result=[]
    arms=['two_fixed','two_equal_total','three']
    for dt in [1.0,0.2,0.1]:
        for arm in arms:
            result.append(dict(id=f'core_dt{dt}_{arm}',arm=arm,dt=dt,rate=.25,nh4=.05,mode='separate',group='time_step'))
    for rate,nh4 in [(.20,.025),(.20,.075),(.30,.025),(.30,.075)]:
        for arm in arms:
            result.append(dict(id=f'near_r{rate}_n{nh4}_{arm}',arm=arm,dt=.2,rate=rate,nh4=nh4,mode='separate',group='local_sensitivity'))
    for arm in arms:
        result.append(dict(id=f'cooperative_{arm}',arm=arm,dt=.2,rate=.25,nh4=.05,mode='cooperative',group='objective_sensitivity'))
    return result

def worker(case,dest):
    from scripts.analysis import screen_propionibacterium_helper as study
    base=study.dFBASimulator
    captured={}
    class AuditedSimulator(base):
        def __init__(self,*args,**kwargs):
            if case['arm']=='two_equal_total':
                kwargs['initial_biomass'][study.OR16_NAME]=.525
                kwargs['initial_biomass'][study.NS21_NAME]=.105
            captured['initial_biomass']=dict(kwargs['initial_biomass'])
            captured['initial_medium']=dict(kwargs['initial_metabolites'])
            super().__init__(*args,**kwargs)
            captured['sim']=self
    study.dFBASimulator=AuditedSimulator
    scenario=study.Scenario(case['id'],case['arm']=='three','lac__L_e',case['rate'],.03,50.)
    tick=time.perf_counter()
    summary,rows=study.run_scenario(scenario,12.,case['dt'],case['nh4'],case['mode'])
    sim=captured.pop('sim')
    record=dict(case=case,summary=summary,initial=captured,diagnostics=sim.get_solver_diagnostics(),seconds=time.perf_counter()-tick)
    (dest/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    with (dest/'trajectory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(id=case['id'],pha=summary['final_pha_g_l'],success=summary['solver_success_rate'],seconds=record['seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--case');a=p.parse_args()
    if a.case:
        worker(json.loads(a.case),a.output);return
    a.output.mkdir(parents=True,exist_ok=False)
    design=cases()
    files=[Path(__file__),ROOT/'scripts/analysis/screen_propionibacterium_helper.py',ROOT/'src/dfba_simulator.py',ROOT/'src/community_solver.py',ROOT/'src/utils.py']+list((ROOT/'models/sbml/final_consortium').glob('*.xml'))+list((ROOT/'models/sbml/helper_candidates/propionibacterium_panmodel_supplement').rglob('P_sherm_model.xml'))
    manifest=dict(created_unix=time.time(),exploratory_source='results/third_species_phbv_reassessment_20260903/scenario_summary.csv',purpose='Audit post-hoc +20% candidate; not independent biological replication',primary='PHA mass versus both same producer inoculum and equal total inoculum controls',secondary=['rubber removed','3HV fraction (not assumed higher is better)','crossfeeding fluxes'],engineering_screen='>=5% PHA increase with <=5% rubber loss versus BOTH controls, solver success 1.0; not significance',cases=design,sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},limitations=['No biological replicates or p-values','Anaerobic private helper oxygen boundary','No isotope tracing; output PHA is not proven rubber-derived','12 h sensitivity screen, not optimized feeding or experimental validation'])
    (a.output/'design.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    for i,case in enumerate(design):
        dest=a.output/case['id'];dest.mkdir()
        print(f'START {i+1}/{len(design)} {case["id"]}',flush=True)
        tick=time.perf_counter()
        try:
            with (dest/'worker.log').open('w') as log:
                r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(dest),'--case',json.dumps(case)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            record=dict(id=case['id'],returncode=r.returncode,seconds=time.perf_counter()-tick)
        except subprocess.TimeoutExpired:
            record=dict(id=case['id'],status='timeout',seconds=time.perf_counter()-tick)
        records.append(record)
        (a.output/'progress.json').write_text(json.dumps(records,indent=2))
        if (dest/'result.json').exists():
            data=json.loads((dest/'result.json').read_text());print(json.dumps(dict(**record,pha=data['summary']['final_pha_g_l'],success=data['summary']['solver_success_rate'])),flush=True)
        else:print(json.dumps(record),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
