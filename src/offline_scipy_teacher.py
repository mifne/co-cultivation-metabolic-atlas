"""Offline-only cold SciPy/HiGHS teacher, isolated from persistent highspy owners.

Deliberately serial: this is a data integrity path, NOT a CPU speed baseline.
"""
from types import SimpleNamespace
import time
import numpy as np
from scipy.optimize import linprog
from .cpu_repeated_lp import _problem,_certificate


class OfflineScipyTeacher:
    def __init__(self,*,n_fluxes=None):
        self.n_fluxes=n_fluxes;self.models={}

    def solve_batch(self,requests):
        from .gpu_compiled_community_backend import stage_key
        self.models.clear();results=[]
        for lane,(objective,kwargs) in enumerate(requests):
            a,rhs,lower,upper,c,neq=_problem(objective,kwargs)
            stage=kwargs.get('_stage') or stage_key(c,a,neq,self.n_fluxes)[0]
            # Only failed original-unit certificates trigger extra solves.
            # Do not relax the acceptance gate or clip residuals/dual values.
            attempts=[];accepted=None
            settings=[('highs-ds',1e-9,True),('highs-ds',1e-10,True),
                      ('highs-ds',1e-10,False),('highs-ipm',1e-10,False)]
            for method,tolerance,presolve in settings:
                options=dict(primal_feasibility_tolerance=tolerance,
                    dual_feasibility_tolerance=tolerance,presolve=presolve)
                if method=='highs-ipm':options['ipm_optimality_tolerance']=1e-12
                tick=time.perf_counter()
                result=linprog(c,A_eq=a[:neq] if neq else None,b_eq=rhs[:neq] if neq else None,
                    A_ub=a[neq:] if neq<len(rhs) else None,b_ub=rhs[neq:] if neq<len(rhs) else None,
                    bounds=np.column_stack((lower,upper)),method=method,options=options)
                record=dict(method=method,options=options,success=bool(result.success),
                    message=str(result.message),seconds=time.perf_counter()-tick)
                attempts.append(record)
                if not result.success:continue
                x=np.array(result.x,copy=True)
                y=np.concatenate((result.eqlin.marginals,result.ineqlin.marginals))
                if x.shape!=c.shape or y.shape!=rhs.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    record['rejection']='nonfinite or malformed solution';continue
                solution=SimpleNamespace(col_value=x,row_dual=y,col_dual=c-a.T@y,value_valid=True,dual_valid=True)
                cert=_certificate(a,rhs,lower,upper,c,neq,solution)
                record['certificate']=cert
                if cert['certificate_passed']:
                    accepted=(x,solution,cert,result.message);break
            if accepted is None:
                raise RuntimeError(f'Offline teacher exhausted {len(attempts)} attempts without original LP certification: {attempts}')
            x,solution,cert,message=accepted
            self.models[lane,stage,*a.shape,neq]={'solver':SimpleNamespace(getSolution=lambda s=solution:s)}
            results.append(SimpleNamespace(success=True,x=x,fun=float(c@x),message=message,
                diagnostics=dict(stage=stage,**cert,teacher_strategy='scipy_cold_serial_certified_retry_v1',
                    cpu_solver_runs=len(attempts),numerical_retry_count=len(attempts)-1,attempts=attempts)))
        return results

    def clear_models(self):self.models.clear()

    def close(self):self.clear_models()
