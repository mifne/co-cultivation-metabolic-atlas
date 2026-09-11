"""Resume the matched symbiosis study after the cultivation-model audit.

All cases use the new reference. Earlier outputs are kept separately and
must never be pooled into the same training/evaluation dataset.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,subprocess,sys,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/audited_symbiosis_20260908'

def designs():
    from scripts.analysis.reassess_symbiosis_20260908 import cases
    result=cases()
    for condition in ['base','oxygen_low']:
        result.extend(dict(id=condition+'_'+arm+'_fine',condition=condition,arm=arm,internal_dt=.0125) for arm in ['two_equal_total','three'])
    result.extend(dict(id='oxygen_low_'+arm,condition='oxygen_low',arm=arm) for arm in ['two_fixed','ns21_alone','inert_pf'])
    priority=['base_two_equal_total','base_three','oxygen_low_two_equal_total','oxygen_low_three',
              'base_two_equal_total_fine','base_three_fine','oxygen_low_two_equal_total_fine','oxygen_low_three_fine']
    result.sort(key=lambda case:priority.index(case['id']) if case['id'] in priority else len(priority))
    return result

def worker(case,dest):
    from scripts.analysis import screen_propionibacterium_helper as s
    from src.audited_dfba import AuditedDFBASimulator
    import numpy as np
    exchanges=defaultdict(lambda:defaultdict(float));internal=[]
    class Audit(AuditedDFBASimulator):
        def _integrate(self,solutions):
            for name,solution in solutions.items():
                for met,rid in self.exchange_reactions[name].items():
                    amount=float(solution.fluxes.get(rid,0.))*self._cell_scale*self._integration_biomass[name]*self.dt
                    if abs(amount)>1e-15:exchanges[name][met+('_out' if amount>0 else '_in')]+=abs(amount)
            super()._integrate(solutions)
            internal.append(dict(time=self.state.time+self.dt,o2=self.state.metabolites['o2_e'],mean_o2=self._oxygen_last_guess,
                nh4=self.state.metabolites['nh4_e'],rubber=self.state.rubber_concentration,
                polymer_carbon_error=self.last_polymer_fluxes['carbon_c5_equivalent_error_mmol_l']))
            if len(internal)%10==0:
                pending=dest/'progress.tmp'
                pending.write_text(json.dumps(dict(hours=self.state.time+self.dt,seconds=time.perf_counter()-start,lp_attempts=self.solve_attempts)))
                pending.replace(dest/'progress.json')
    arm=case['arm'];condition=case['condition']
    models=s._models(arm in ['three','pf_alone','ns21_pf','inert_pf'])
    if arm=='pf_alone':models={s.HELPER_NAME:models[s.HELPER_NAME]}
    elif arm=='ns21_alone':models={s.NS21_NAME:models[s.NS21_NAME]}
    elif arm=='ns21_pf':models.pop(s.OR16_NAME)
    elif arm=='inert_pf':
        for reaction in models[s.HELPER_NAME].reactions:reaction.bounds=(0.,0.)
    biomass,medium=s.get_initial_params(models)
    composition={s.OR16_NAME:.5,s.NS21_NAME:.1,s.HELPER_NAME:.03}
    biomass={name:composition[name] for name in models}
    if arm in ['two_equal_total','ns21_alone','ns21_pf']:
        total=sum(biomass.values());biomass={n:v*.63/total for n,v in biomass.items()}
    medium.update(yeast_extract_e=0.,glc__D_e=0.,lac__L_e=0.,ppa_e=0.,ac_e=0.,nh4_e=.05)
    settings=dict(case=case,initial_biomass=biomass,initial_medium=medium,hours=float(case.get('hours',12)),
        internal_dt=case.get('internal_dt',.025),nitrogen_half_saturation=.1,
        kla=10. if condition=='oxygen_low' else 50.,
        lactate_rate=0. if condition=='no_lactate' else 1. if condition=='lactate_rich' else .25,
        phenotype='Pf oxygen uptake closed, not validated physiology',density_policy='flux_consistent',
        polymer_oxygen_half_saturation=.01,buffer_mmol_l=50.,flux_selection='parsimonious_exchange')
    (dest/'design.json').write_text(json.dumps(settings,indent=2))
    sim=Audit(models=models,initial_biomass=biomass,initial_metabolites=medium,initial_rubber=10.,dt=1.,
        max_internal_dt=settings['internal_dt'],ph_control_target=7.,solver_backend='highs')
    def row():
        phb=sum(v.phb_accumulated for v in sim.state.species.values());phv=sum(v.phv_accumulated for v in sim.state.species.values())
        r=dict(time=sim.state.time,pha=phb*.08609+phv*.10012,phb=phb,phv=phv,phv_mol_fraction=phv/(phb+phv) if phb+phv>0 else 0.,
               rubber=sim.state.rubber_concentration,o2=sim.state.metabolites['o2_e'],nh4=sim.state.metabolites['nh4_e'],
               lactate=sim.state.metabolites.get('lac__L_e',0.),propionate=sim.state.metabolites.get('ppa_e',0.))
        r.update({'biomass_'+n:v.biomass for n,v in sim.state.species.items()});return r
    rows=[row()];start=time.perf_counter()
    for hour in range(int(settings['hours'])):
        rates={'lac__L_e':settings['lactate_rate']}
        if condition=='nitrogen_staged' and hour<4:rates['nh4_e']=.1
        try:
            sim.step({}, {},dynamic_kla=settings['kla'],feed_rates_mmol_l_h=rates)
        except Exception as error:
            (dest/'failure.json').write_text(json.dumps(dict(error=repr(error),last_state=row(),
                diagnostics=sim.get_solver_diagnostics(),completed_hours=hour,partial_step_invalid=True),indent=2,allow_nan=False))
            raise
        rows.append(row());(dest/'progress.json').write_text(json.dumps(dict(hours=sim.state.time,seconds=time.perf_counter()-start,lp_attempts=sim.solve_attempts)))
    result=dict(settings=settings,final=rows[-1],seconds=time.perf_counter()-start,exchanges=exchanges,
                fold_growth={n:v.biomass/biomass[n] for n,v in sim.state.species.items()},diagnostics=sim.get_solver_diagnostics())
    assert all(math.isfinite(value) and value>=-1e-7 for r in rows for value in r.values())
    assert max(abs(r['polymer_carbon_error']) for r in internal)<1e-8
    (dest/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for filename,records in [('trajectory.csv',rows),('internal_trajectory.csv',internal)]:
        with (dest/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    print(json.dumps(dict(case=case,pha=result['final']['pha'],seconds=result['seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.case:worker(json.loads(a.case),a.output);return
    a.output.mkdir(exist_ok=False)
    from scripts.analysis.screen_propionibacterium_helper import _source_model_path
    files=[Path(__file__),ROOT/'scripts/analysis/reassess_symbiosis_20260908.py',ROOT/'src/audited_dfba.py',ROOT/'src/cultivation_numerics.py',ROOT/'src/resolved_dfba.py',ROOT/'src/dfba_simulator.py',ROOT/'src/utils.py',ROOT/'scripts/analysis/screen_propionibacterium_helper.py',_source_model_path(),ROOT/'models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml',ROOT/'models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml']
    files=sorted(set(files+list((ROOT/'src').glob('*.py'))))
    source_dir=a.output/'sources';source_dir.mkdir()
    sources={}
    for file in files:
        rel=file.relative_to(ROOT).as_posix();raw=file.read_bytes();sources[rel]=hashlib.sha256(raw).hexdigest();(source_dir/rel.replace('/','__')).write_bytes(raw)
    design=dict(cases=designs(),sources=sources,model_version='audited_cultivation_v2',
        purpose='post-repair exploratory replication; no statistical significance or stable mutualism claim',
        screen_criterion='PHA >5% against matched-total OR16+NS21 and rubber removal loss <=5%; require paired finer confirmation',
        limits='Uncalibrated phenotype/kinetics; only baseline and low oxygen have finer pairs; not a global feed or inoculum optimum')
    import platform, importlib.metadata
    design['runtime']=dict(python=platform.python_version(),platform=platform.platform(),workers=a.workers,
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','cobra']})
    (a.output/'design.json').write_text(json.dumps(design,indent=2))
    def run(case):
        dest=a.output/case['id'];dest.mkdir();print('START',case['id'],flush=True)
        with (dest/'worker.log').open('w') as log:
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',json.dumps(case),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        path=dest/'result.json';result=json.loads(path.read_text()) if path.exists() else {}
        status=dict(case=case,returncode=child.returncode,pha=result.get('final',{}).get('pha'));print('DONE',json.dumps(status),flush=True);return status
    progress=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,c) for c in design['cases']]):
            progress.append(future.result());(a.output/'progress.json').write_text(json.dumps(progress,indent=2))
    for file,sha in sources.items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source changed during experiment: '+file)
    if any(r['returncode']!=0 for r in progress):raise RuntimeError('Some conditions failed; inspect logs without dropping them')
if __name__=='__main__':main()
