"""Capture a genuine failed reference LP for solver diagnosis, without accepting it."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.sparse import save_npz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.audited_dfba as module
from scripts.analysis.reassess_audited_symbiosis_20260908 import worker
DEST=ROOT/'results/cultivation_model_audit_20260908/lp_failure';DEST.mkdir(exist_ok=True)
original=module.linprog
def capture(c,**kwargs):
    result=original(c,**kwargs)
    if result.success:
        eq=np.asarray(kwargs['A_eq']@result.x-kwargs['b_eq'])
        if abs(eq).max()>1e-7:
            save_npz(DEST/'a_eq.npz',kwargs['A_eq'])
            if kwargs.get('A_ub') is not None:save_npz(DEST/'a_ub.npz',kwargs['A_ub'])
            np.savez(DEST/'data.npz',c=c,b_eq=kwargs['b_eq'],b_ub=kwargs.get('b_ub'),bounds=np.array(kwargs['bounds']),x=result.x)
            print(json.dumps(dict(residual=float(abs(eq).max()),row=int(np.argmax(abs(eq))),objective=float(result.fun))),flush=True)
    return result
module.linprog=capture
worker(dict(id='failure',condition='base',arm='two_equal_total',hours=1),DEST)
