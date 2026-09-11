"""Counterfactual removal of newly secreted Pf products; no gene changes."""
from pathlib import Path
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/pfreud_transfer_intervention_20260908'
def run(mode):
    from scripts.analysis import screen_propionibacterium_helper as s
    from scripts.analysis.pfreud_causal_20260908 import worker
    base=s.dFBASimulator;removed={}
    class Removal(base):
        def _update_environment(self,solutions,integration_biomass=None):
            amount={}
            for met in (['ppa_e'] if mode=='ppa' else ['ppa_e','ac_e']):
                rid=self.exchange_reactions[s.HELPER_NAME].get(met)
                b=integration_biomass[s.HELPER_NAME]
                amount[met]=max(0.,solutions[s.HELPER_NAME].get(rid,0.)*b*self.dt)
            result=super()._update_environment(solutions,integration_biomass)
            for met,quantity in amount.items():
                quantity=min(quantity,max(0.,self.state.metabolites.get(met,0.)))
                self.state.metabolites[met]-=quantity
                removed[met]=removed.get(met,0.)+quantity
            return result
    s.dFBASimulator=Removal
    dest=OUT/mode
    worker(dict(id='remove_'+mode,arm='three',dt=.05,profile='continuous'),dest)
    (dest/'removed.json').write_text(json.dumps(removed,indent=2))
if len(sys.argv)>1:run(sys.argv[1])
else:
    OUT.mkdir(exist_ok=False)
    (OUT/'design.json').write_text(json.dumps(dict(scope='Numerical counterfactual, not a feasible cultivation control or gene prediction',intervention='Remove newly secreted Pf propionate, or propionate plus acetate, after each integration step; track removed carbon; retain Pf biomass, nutrient use and common resource allocation'),indent=2))
    def invoke(mode):
        dest=OUT/mode;dest.mkdir()
        with (dest/'worker.log').open('w') as log:
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),mode],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        print(mode,r.returncode,json.loads((dest/'result.json').read_text())['pha'] if (dest/'result.json').exists() else 'failed',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(invoke,['ppa','ppa_ac']))
