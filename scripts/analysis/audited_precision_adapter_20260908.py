"""Retry an unsuccessful lexicographic solve with a more accurate primary LP.

All mathematical inputs and acceptance tolerances remain unchanged. The
fallback is unreachable in previously completed, successful trajectories.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def install():
    import src.audited_dfba as module
    cls=module.AuditedDFBASimulator;original=cls._solve_lp
    def retry_primary(self,name,objective,**kwargs):
        before_requests=self.accounting_audit['lp_requests'];before_certified=self.accounting_audit['lp_certified_requests']
        solution=original(self,name,objective,**kwargs)
        if solution is not None:return solution
        if self.accounting_audit['rejected_lp_trials'][-1]['stage']!='exchange_selection':return None
        history=[]
        for method in ['highs-ds','highs-ipm']:
            backend=module.linprog;offset=len(self.accounting_audit['rejected_lp_trials']);values=[]
            def accurate(c,**options):
                options=dict(options);options['method']=method
                options['options']=dict(options.get('options',{}),presolve=False,
                    primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10,ipm_optimality_tolerance=1e-12)
                result=backend(c,**options)
                if len(c)==len(self.models[name].reactions) and result.success:values.append(float(result.fun))
                return result
            module.linprog=accurate
            try:solution=original(self,name,objective,**kwargs)
            finally:
                module.linprog=backend
                for rejected in self.accounting_audit['rejected_lp_trials'][offset:]:
                    rejected['requested_method']=rejected['method'];rejected.update(method=method,presolve=False,tolerance=1e-10)
            history.append(dict(method=method,primary_minimization_values=values,recovered=solution is not None))
            if solution is not None:
                abandoned=(self.accounting_audit['lp_requests']-before_requests)-(self.accounting_audit['lp_certified_requests']-before_certified)
                self.accounting_audit['superseded_uncertified_requests']=self.accounting_audit.get('superseded_uncertified_requests',0)+abandoned
                self.accounting_audit.setdefault('primary_recomputations',[]).append(dict(species=name,time=self.state.time,history=history,superseded_requests=abandoned))
                return solution
        return None
    cls._solve_lp=retry_primary

def main():
    import argparse
    from scripts.analysis.audited_roundoff_adapter_20260908 import install as install_roundoff
    from scripts.analysis.reassess_audited_symbiosis_20260908 import worker
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),ROOT/'scripts/analysis/audited_roundoff_adapter_20260908.py',ROOT/'src/audited_dfba.py']}
    (a.output/'numerical_patch.json').write_text(json.dumps(dict(patch='c30_and_primary_recomputation',sources=sources,
        threshold_mmol_l=1e-12,scope='only failed branches; unchanged LP/acceptance tolerances; explicit superseded target accounting'),indent=2))
    install_roundoff();install();worker(json.loads(a.case),a.output)

if __name__=='__main__':main()
