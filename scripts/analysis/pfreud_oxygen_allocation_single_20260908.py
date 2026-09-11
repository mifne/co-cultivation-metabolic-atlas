"""Oxygen-allocation diagnostic; leaves shared models/files unchanged."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis import screen_propionibacterium_helper as s
from scripts.analysis.pfreud_causal_20260908 import worker
base=s.dFBASimulator
class ActiveOxygenAllocation(base):
    def set_uptake_constraints(self,name,medium,kla=50.):
        result=super().set_uptake_constraints(name,medium,kla)
        if name in (s.OR16_NAME,s.NS21_NAME):
            c=float(medium.get('o2_e',0.));rid=self.exchange_reactions[name].get('o2_e')
            if c>0 and rid:
                active=sum(max(1e-6,v.biomass) for n,v in self.state.species.items() if n in (s.OR16_NAME,s.NS21_NAME))
                vmax=self.max_uptake_rate
                limit=min(vmax*c/(.01+c),c/(active*self.dt),vmax*10)
                original=self.original_bounds[name].get(rid,(-1000.,1000.))[0]
                rxn=self.models[name].reactions.get_by_id(rid)
                rxn.lower_bound=self._safe_lb(min(0.,max(-limit,original)),rxn.upper_bound)
        return result
s.dFBASimulator=ActiveOxygenAllocation
out=ROOT/'results/pfreud_oxygen_allocation_single_20260908';out.mkdir(exist_ok=False)
(out/'design.json').write_text(json.dumps(dict(scope='Numerical allocation diagnostic, not biological validation',change='Allocate shared O2 pool only to OR16 and NS21; Pf shared-O2 uptake is disabled in current model',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
for arm in ['three']:
    dest=out/arm;dest.mkdir();print('START',arm,flush=True)
    worker(dict(id=f'active_oxygen_{arm}',arm=arm,dt=.05,profile='continuous'),dest)

