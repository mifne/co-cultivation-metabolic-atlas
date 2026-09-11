"""Causal phase-schedule interventions and time-resolution/feeding checks for Pf.
Forced schedules are numerical counterfactuals, not feasible biological controls.
"""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys,time,os
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/pfreud_causal_phase_retry_20260908'

def design():
    cases=[]
    for arm in ['two_fixed','three']:
        for schedule in ['two_fixed','three']:
            cases.append(dict(id=f'schedule_{arm}_from_{schedule}',arm=arm,dt=1.,profile='continuous',force=schedule))
    for dt in [.05,.025]:
        for arm in ['two_fixed','two_equal_total','three']:
            cases.append(dict(id=f'fine_{dt}_{arm}',arm=arm,dt=dt,profile='continuous'))
    for profile in ['early','late']:
        for arm in ['two_fixed','two_equal_total','three']:
            cases.append(dict(id=f'{profile}_{arm}',arm=arm,dt=.05,profile=profile))
    return cases[:4]

def worker(case,dest):
    import numpy as np
    from scripts.analysis import screen_propionibacterium_helper as s
    base=s.dFBASimulator;capture={};schedule=None
    if case.get('force'):
        path=ROOT/f'results/pfreud_value_audit_20260908/core_dt1.0_{case["force"]}/trajectory.csv'
        schedule=[float(r['nh4_mmol_l'])<.1 for r in list(csv.DictReader(path.open()))[:-1]]
    class Sim(base):
        def __init__(self,*a,**kw):
            if case['arm']=='two_equal_total':kw['initial_biomass'].update({s.OR16_NAME:.525,s.NS21_NAME:.105})
            super().__init__(*a,**kw);self.audit=[];self.polymer_o2=0.;self.o2_transferred=0.;capture['sim']=self
        def step(self,*a,**kw):
            before=float(self.state.metabolites['o2_e'])
            self.o2_transferred+=(.25-before)*(1-np.exp(-50*self.dt))
            return super().step(*a,**kw)
        def solve_fba(self,name):
            if name==s.NS21_NAME:
                model=self.models[name];pha=self.exchange_reactions[name]['pha_c'];phv=self.exchange_reactions[name].get('phv_c')
                phase=self.state.metabolites['nh4_e']<.1
                if schedule is not None:
                    phase=schedule[min(round(self.state.time),len(schedule)-1)]
                    if phase:
                        weights={pha:self.PHB_REPEAT_G_PER_MMOL}
                        for rid in [pha,phv]:
                            if rid:model.reactions.get_by_id(rid).upper_bound=max(0.,self.original_bounds[name].get(rid,(0.,1000.))[1])
                        if phv:weights[phv]=self.PHV_REPEAT_G_PER_MMOL
                        self._set_surrogate_objective_weights(name,model,weights)
                    else:
                        for rid in [pha,phv]:
                            if rid:model.reactions.get_by_id(rid).upper_bound=0.
                        self._set_surrogate_objective(name,model,'R_Growth' if 'R_Growth' in model.reactions else 'Growth')
                self.audit.append(dict(time=self.state.time,nh4=self.state.metabolites['nh4_e'],phase=int(phase),o2=self.state.metabolites['o2_e'],**self.last_polymer_fluxes))
                self.polymer_o2+=self.last_polymer_fluxes['oxygen_mmol_l_step']
            return super().solve_fba(name)
    s.dFBASimulator=Sim
    models=s._models(case['arm']=='three');biomass,medium=s.get_initial_params(models)
    biomass[s.OR16_NAME]=.5;biomass[s.NS21_NAME]=.1
    if case['arm']=='three':biomass[s.HELPER_NAME]=.03
    medium.update(yeast_extract_e=0.,glc__D_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=.05)
    sim=Sim(models=models,initial_biomass=biomass,initial_metabolites=medium,initial_rubber=10.,volume=1.,dt=case['dt'],solver_backend='highs',fba_mode='separate',ph_control_target=7.,polymer_oxygen_fraction=.25,cooperative_optimize_live_objectives=True,cooperative_parsimony=True)
    scenario=s.Scenario(case['id'],case['arm']=='three','lac__L_e',.25,.03,50.)
    rows=[s._row(sim,scenario,0.)];feed=0.;tick=time.perf_counter()
    for i in range(round(12/case['dt'])):
        t=i*case['dt'];rate=.25 if case['profile']=='continuous' else (.75 if (case['profile']=='early' and t<4) or (case['profile']=='late' and t>=8) else 0.)
        feed+=rate*case['dt'];sim.state.metabolites['lac__L_e']+=rate*case['dt'];sim.step({}, {},dynamic_kla=50.)
        row=s._row(sim,scenario,feed);row['helper_nh4_uptake']=sim.state.species[s.HELPER_NAME].metabolite_uptake.get('nh4_e',0.) if case['arm']=='three' else 0.;rows.append(row)
    def integ(key,bkey):return sum(b[key]*a[bkey]*case['dt'] for a,b in zip(rows[:-1],rows[1:]))
    summary=dict(case=case,seconds=time.perf_counter()-tick,pha=rows[-1]['pha_g_l'],phv_fraction=rows[-1]['phv_mol_fraction'],rubber_removed=10-rows[-1]['rubber_g_l'],final_biomass={n:v.biomass for n,v in sim.state.species.items()},feed=feed,phase_hours=sum(x['phase'] for x in sim.audit)*case['dt'],polymer_o2=sim.polymer_o2,o2_transferred=sim.o2_transferred,helper_nh4_uptake=integ('helper_nh4_uptake','helper_biomass_g_l'),helper_ppa_secretion=integ('helper_propionate_secretion','helper_biomass_g_l'),ns21_ppa_uptake=integ('ns21_propionate_uptake','ns21_biomass_g_l'),diagnostics=sim.get_solver_diagnostics())
    (dest/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    for name,data in [('trajectory',rows),('phase_audit',sim.audit)]:
        keys=list(dict.fromkeys(k for r in data for k in r))
        with (dest/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
    print(json.dumps({k:v for k,v in summary.items() if k!='diagnostics'}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.case:worker(json.loads(a.case),a.output);return
    a.output.mkdir(exist_ok=False);cases=design()
    sources=[Path(__file__),ROOT/'src/dfba_simulator.py',ROOT/'src/community_solver.py',ROOT/'scripts/analysis/screen_propionibacterium_helper.py']
    (a.output/'design.json').write_text(json.dumps(dict(cases=cases,created_unix=time.time(),note='Phase swaps are diagnostic. Same 3 mmol/L lactate budget; producer-matched and total-inoculum controls. No biological significance claim.',sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}),indent=2))
    def run(case):
        dest=a.output/case['id'];dest.mkdir();print('START '+case['id'],flush=True)
        try:
            with (dest/'worker.log').open('w') as log:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(dest),'--case',json.dumps(case)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            outcome=dict(case=case,returncode=r.returncode)
        except subprocess.TimeoutExpired:outcome=dict(case=case,status='timeout')
        if (dest/'result.json').exists():
            d=json.loads((dest/'result.json').read_text());outcome.update(pha=d['pha'],phase_hours=d['phase_hours'],seconds=d['seconds'])
        print('DONE '+json.dumps(outcome),flush=True);return outcome
    outcomes=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,c) for c in cases]):
            outcomes.append(future.result());(a.output/'progress.json').write_text(json.dumps(outcomes,indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()

