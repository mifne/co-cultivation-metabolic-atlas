from pathlib import Path
import json,time
import numpy as np
from scipy.sparse import load_npz
from scipy.optimize import linprog
p=Path('results/cultivation_model_audit_20260908/lp_failure');d=np.load(p/'data.npz',allow_pickle=True)
ae=load_npz(p/'a_eq.npz');au=load_npz(p/'a_ub.npz') if (p/'a_ub.npz').exists() else None
rows=[]
for method,presolve,tolerance in [('highs',True,1e-9),('highs-ds',False,1e-10),('highs-ipm',False,1e-10)]:
    start=time.perf_counter();r=linprog(d['c'],A_eq=ae,b_eq=d['b_eq'],A_ub=au,b_ub=d['b_ub'] if au is not None else None,
        bounds=d['bounds'],method=method,options={'presolve':presolve,'primal_feasibility_tolerance':tolerance,'dual_feasibility_tolerance':tolerance,'ipm_optimality_tolerance':1e-12})
    row=dict(method=method,presolve=presolve,tolerance=tolerance,seconds=time.perf_counter()-start,status=r.status,message=r.message)
    if r.success:
        bounds=d['bounds'];station=d['c']-ae.T@r.eqlin.marginals-r.lower.marginals-r.upper.marginals
        dual=d['b_eq']@r.eqlin.marginals+bounds[:,0]@r.lower.marginals+bounds[:,1]@r.upper.marginals
        if au is not None:station-=au.T@r.ineqlin.marginals;dual+=d['b_ub']@r.ineqlin.marginals
        row.update(objective=float(r.fun),eq_residual=float(np.abs(ae@r.x-d['b_eq']).max()),dual_residual=float(abs(station).max()),gap=float(abs(r.fun-dual)))
    rows.append(row)
(p/'retry_diagnosis.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
