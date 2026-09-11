"""Read-only instrumentation of shared-pool clipping in the Pf screen."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis import screen_propionibacterium_helper as s
from collections import defaultdict
base=s.dFBASimulator
class PoolAudited(base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.pool_events=[];self.pool_deficit=defaultdict(float)
    def _update_environment(self,solutions,integration_biomass=None):
        changes=defaultdict(float)
        for name,fluxes in solutions.items():
            b=integration_biomass[name] if integration_biomass is not None else self.state.species[name].biomass
            for met,rid in self.exchange_reactions[name].items():
                if met not in ('h_e','pha_c','phv_c','yeast_extract_e'):
                    changes[met]+=float(fluxes.get(rid,0.))*float(b)*self.dt
        for met,delta in changes.items():
            before=float(self.state.metabolites.get(met,0.))
            if before+delta < -1e-8:
                deficit=-(before+delta);self.pool_deficit[met]+=deficit
                self.pool_events.append(dict(time=float(self.state.time),met=met,before=before,delta=delta,deficit=deficit))
        super()._update_environment(solutions,integration_biomass)
    def get_solver_diagnostics(self):
        d=super().get_solver_diagnostics();d['pool_deficits_mmol_l']=dict(self.pool_deficit);d['pool_events']=self.pool_events;return d
s.dFBASimulator=PoolAudited
out=ROOT/'results/pfreud_pool_audit_20260908';out.mkdir(exist_ok=False)
for mode in ['separate','cooperative']:
    for helper in [False,True]:
        name=f'{mode}_{"three" if helper else "two"}'
        print('START',name,flush=True)
        # Capture the instance without altering any model or numerical settings.
        capture={};original=PoolAudited.__init__
        def init(self,*a,**k):original(self,*a,**k);capture['sim']=self
        PoolAudited.__init__=init
        summary,rows=s.run_scenario(s.Scenario(name,helper,'lac__L_e',.25,.03,50),12,.2,.05,mode)
        PoolAudited.__init__=original
        d=capture['sim'].get_solver_diagnostics()
        (out/f'{name}.json').write_text(json.dumps(dict(summary=summary,diagnostics=d),indent=2))
        print(json.dumps(dict(name=name,deficits=d['pool_deficits_mmol_l'],success=summary['solver_success_rate'])),flush=True)
