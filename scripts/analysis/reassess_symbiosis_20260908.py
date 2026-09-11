"""Matched resource-budget screen and intracellular-step exchange accounting."""
from pathlib import Path
import argparse,csv,hashlib,json,math,subprocess,sys,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/symbiosis_reassessment_20260908'

def cases():
    result=[]
    for condition in ['base','no_lactate','lactate_rich','nitrogen_staged','oxygen_low']:
        for arm in ['two_equal_total','three']:
            result.append(dict(id=condition+'_'+arm,condition=condition,arm=arm))
    result += [dict(id='base_'+arm,condition='base',arm=arm) for arm in ['two_fixed','pf_alone','ns21_alone','ns21_pf']]
    return result

def worker(case,dest):
    from scripts.analysis import screen_propionibacterium_helper as s
    from src.resolved_dfba import ResolvedDFBASimulator
    exchanges=defaultdict(lambda:defaultdict(float));inactive=defaultdict(float)
    class Audit(ResolvedDFBASimulator):
        def _update_environment(self,solutions,integration_biomass=None):
            for name,fluxes in solutions.items():
                for met,rid in self.exchange_reactions[name].items():
                    delta=fluxes.get(rid,0.)*integration_biomass[name]*self.dt
                    if abs(delta)>1e-15:exchanges[name][met+('_out' if delta>0 else '_in')]+=abs(delta)
                if self.state.species[name].growth_rate < 1e-6:inactive[name]+=self.dt
            return super()._update_environment(solutions,integration_biomass)
    arm=case['arm'];condition=case['condition']
    models=s._models(arm in ['three','pf_alone','ns21_pf'])
    if arm=='pf_alone':models={s.HELPER_NAME:models[s.HELPER_NAME]}
    if arm=='ns21_alone':models={s.NS21_NAME:models[s.NS21_NAME]}
    if arm=='ns21_pf':models.pop(s.OR16_NAME)
    biomass,medium=s.get_initial_params(models)
    composition={s.OR16_NAME:.5,s.NS21_NAME:.1,s.HELPER_NAME:.03}
    biomass={name:composition[name] for name in models}
    if arm in ['two_equal_total','ns21_alone','ns21_pf']:
        total=sum(biomass.values());biomass={name:b*.63/total for name,b in biomass.items()}
    medium.update(yeast_extract_e=0.,glc__D_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=.05)
    initial=biomass.copy()
    settings=dict(case=case,initial_biomass=initial,initial_medium=medium.copy(),hours=12.,internal_dt=case.get('internal_dt',.025),nitrogen_half_saturation=.1,kla=10. if condition=='oxygen_low' else 50.,lactate_rate=0. if condition=='no_lactate' else 1. if condition=='lactate_rich' else .25)
    (dest/'design.json').write_text(json.dumps(settings,indent=2))
    sim=Audit(models=models,initial_biomass=biomass,initial_metabolites=medium,initial_rubber=10.,dt=1.,max_internal_dt=settings['internal_dt'],solver_backend='highs',ph_control_target=7.)
    def row():
        data=dict(time=sim.state.time,pha=sum(v.phb_accumulated*sim.PHB_REPEAT_G_PER_MMOL+v.phv_accumulated*sim.PHV_REPEAT_G_PER_MMOL for v in sim.state.species.values()),phv=sum(v.phv_accumulated for v in sim.state.species.values()),rubber=sim.state.rubber_concentration,o2=sim.state.metabolites['o2_e'],nh4=sim.state.metabolites['nh4_e'],lactate=sim.state.metabolites.get('lac__L_e',0.),propionate=sim.state.metabolites.get('ppa_e',0.))
        data.update({'biomass_'+n:v.biomass for n,v in sim.state.species.items()});return data
    rows=[row()];start=time.perf_counter()
    for hour in range(12):
        rates={'lac__L_e':settings['lactate_rate']}
        if condition=='nitrogen_staged' and hour<4:rates['nh4_e']=.1
        sim.step({}, {},dynamic_kla=settings['kla'],feed_rates_mmol_l_h=rates)
        rows.append(row());(dest/'progress.json').write_text(json.dumps(dict(hours=sim.state.time,seconds=time.perf_counter()-start)))
    result=dict(settings=settings,final=rows[-1],seconds=time.perf_counter()-start,exchanges=exchanges,low_growth_hours=inactive,
        fold_growth={n:v.biomass/initial[n] for n,v in sim.state.species.items()},
        mean_net_growth_h={n:math.log(v.biomass/initial[n])/12 for n,v in sim.state.species.items()},diagnostics=sim.get_solver_diagnostics())
    (dest/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (dest/'trajectory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(case=case,pha=result['final']['pha'],fold_growth=result['fold_growth'],seconds=result['seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.case:worker(json.loads(a.case),a.output);return
    a.output.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'src/resolved_dfba.py',ROOT/'src/dfba_simulator.py',ROOT/'scripts/analysis/screen_propionibacterium_helper.py']
    design=dict(cases=cases(),criterion='Exploratory PHA superiority requires >5% vs total-inoculum-matched OR16+NS21 and no >5% rubber loss; positives need finer-step confirmation. Not a p-value.',scope='12h closed-volume simulations; not proof of stable mutualism or globally optimal feeding',sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.output/'design.json').write_text(json.dumps(design,indent=2))
    def run(c):
        dest=a.output/c['id'];dest.mkdir();print('START',c['id'],flush=True)
        with (dest/'worker.log').open('w') as log:
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',json.dumps(c),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        path=dest/'result.json';d=json.loads(path.read_text()) if path.exists() else {}
        outcome=dict(case=c,returncode=r.returncode,pha=d.get('final',{}).get('pha'));print('DONE',json.dumps(outcome),flush=True);return outcome
    progress=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,c) for c in design['cases']]):
            progress.append(f.result());(a.output/'progress.json').write_text(json.dumps(progress,indent=2))
if __name__=='__main__':main()
