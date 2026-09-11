"""All-metabolite exchange totals for mechanism diagnosis."""
from pathlib import Path
import sys,json
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis import screen_propionibacterium_helper as s
from scripts.analysis.pfreud_causal_20260908 import worker
totals=defaultdict(lambda:defaultdict(float));base=s.dFBASimulator
class Audit(base):
    def _update_environment(self,solutions,integration_biomass=None):
        for name,fluxes in solutions.items():
            b=integration_biomass[name] if integration_biomass is not None else self.state.species[name].biomass
            for met,rid in self.exchange_reactions[name].items():
                v=fluxes.get(rid,0.)*b*self.dt
                if abs(v)>1e-15:totals[name][met+('_out' if v>0 else '_in')]+=abs(v)
        return super()._update_environment(solutions,integration_biomass)
s.dFBASimulator=Audit
out=ROOT/'results/pfreud_exchange_audit_20260908';out.mkdir(exist_ok=False)
worker(dict(id='all_exchange_three',arm='three',dt=.05,profile='continuous'),out)
(out/'exchange_totals.json').write_text(json.dumps(totals,indent=2))
print('EXCHANGES',json.dumps(totals[s.HELPER_NAME]),flush=True)
