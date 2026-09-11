"""Prepare, but do not install, the production form of the conditional repairs."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
text=(ROOT/'src/audited_dfba.py').read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace("    GROWTH_FLOOR_TOLERANCE_H_INV = 1e-9", "    NUMERICAL_REPAIR_REVISION = 'c30_primary_recomputation_20260908'\n    GROWTH_FLOOR_TOLERANCE_H_INV = 1e-9")
replace("        self.state.metabolites['o2_e']=c0\n\n    def _solve_oxygen_coupled", """        self.state.metabolites['o2_e']=c0
        concentration=self.state.metabolites.get('C30_oligo_e',0.)
        if -1e-12<=concentration<0.:
            addition=-concentration
            self.state.metabolites['C30_oligo_e']=0.
            self.last_polymer_fluxes['carbon_c5_equivalent_error_mmol_l']+=6*addition
            self._oxygen_context['c30_roundoff_added']=addition

    def _solve_oxygen_coupled""")
replace("        for attempt,(method,presolve,tol) in enumerate([('highs',True,1e-9),('highs-ds',False,1e-10),('highs-ipm',False,1e-10)]):", """        override=getattr(self,'_lp_algorithm_override',None)
        methods=([override]*3 if override is not None else
                 [('highs',True,1e-9),('highs-ds',False,1e-10),('highs-ipm',False,1e-10)])
        for attempt,(method,presolve,tol) in enumerate(methods):""")
replace("                    return result\n            if digest is None:", """                    if override is not None and stage!='exchange_selection':
                        self._recomputed_primary_values.append(float(result.fun))
                    return result
            if digest is None:""")
replace("    def _solve_lp(self,name,objective,*,growth_lower=None,storage=False,select_fluxes=True):", """    def _solve_lp(self,name,objective,*,growth_lower=None,storage=False,select_fluxes=True):
        # A primary value within the general LP certificate may still exceed
        # the true optimum by more than the tighter lexicographic precision.
        # Recompute the primary only after secondary selection has failed.
        kwargs=dict(growth_lower=growth_lower,storage=storage,select_fluxes=select_fluxes)
        before_requests=self.accounting_audit['lp_requests']
        before_certified=self.accounting_audit['lp_certified_requests']
        solution=self._solve_lp_once(name,objective,**kwargs)
        if solution is not None:return solution
        if self.accounting_audit['rejected_lp_trials'][-1]['stage']!='exchange_selection':return None
        history=[]
        for method in ['highs-ds','highs-ipm']:
            self._lp_algorithm_override=(method,False,1e-10)
            self._recomputed_primary_values=[]
            try:
                solution=self._solve_lp_once(name,objective,**kwargs)
                values=list(self._recomputed_primary_values)
            finally:
                self._lp_algorithm_override=None
                self._recomputed_primary_values=[]
            history.append(dict(method=method,primary_minimization_values=values,recovered=solution is not None))
            if solution is not None:
                abandoned=(self.accounting_audit['lp_requests']-before_requests)-(self.accounting_audit['lp_certified_requests']-before_certified)
                self.accounting_audit['superseded_uncertified_requests']=self.accounting_audit.get('superseded_uncertified_requests',0)+abandoned
                self.accounting_audit.setdefault('primary_recomputations',[]).append(dict(species=name,time=self.state.time,history=history,superseded_requests=abandoned))
                return solution
        return None

    def _solve_lp_once(self,name,objective,*,growth_lower=None,storage=False,select_fluxes=True):""")
replace("    def step(self,rubber_degradation_rates,nutrient_supplementation", """        addition=self._oxygen_context.get('c30_roundoff_added',0.)
        if addition:
            ledger=self.accounting_audit['roundoff_added_mmol_l']
            ledger['C30_oligo_e']=ledger.get('C30_oligo_e',0.)+addition

    def step(self,rubber_degradation_rates,nutrient_supplementation""")
replace("diagnostics['audited_cultivation']=dict(version=self.NUMERICS_VERSION,valid=not self._invalid,",
        "diagnostics['audited_cultivation']=dict(version=self.NUMERICS_VERSION,valid=not self._invalid,\n            numerical_repair_revision=self.NUMERICAL_REPAIR_REVISION,")
dest=ROOT/'results/cultivation_model_audit_20260908/audited_dfba_candidate.py'
dest.write_text(text)
compile(text,str(dest),'exec')
print(dest)
rl=(ROOT/'src/rl_environment.py').read_text()
old="'max_internal_dt', 'density_policy', 'flux_selection', 'GROWTH_FLOOR_TOLERANCE_H_INV'"
assert rl.count(old)==1
rl=rl.replace(old,"'NUMERICAL_REPAIR_REVISION', "+old)
(dest.parent/'rl_environment_candidate.py').write_text(rl)
