"""Capture a fully rejected LP and diagnostics from the no-lactate case."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.sparse import save_npz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.audited_dfba import AuditedDFBASimulator
from scripts.analysis.reassess_audited_symbiosis_20260908 import worker
DEST=ROOT/'results/cultivation_model_audit_20260908/uncertified_lp';DEST.mkdir(exist_ok=True)
original=AuditedDFBASimulator._certified_linprog
def capture(self,name,stage,c,**kwargs):
    result=original(self,name,stage,c,**kwargs)
    if result is None:
        save_npz(DEST/'a_eq.npz',kwargs['A_eq'])
        if kwargs.get('A_ub') is not None:save_npz(DEST/'a_ub.npz',kwargs['A_ub'])
        np.savez(DEST/'data.npz',c=c,b_eq=kwargs['b_eq'],b_ub=kwargs.get('b_ub'),
            bounds=np.array(kwargs['bounds']),primary=kwargs.get('primary'))
        info=dict(name=name,stage=stage,time=self.state.time,diagnostics=self.get_solver_diagnostics())
        (DEST/'diagnostics.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(dict(name=name,stage=stage,time=self.state.time,last_rejected=self.accounting_audit['rejected_lp_trials'][-3:])),flush=True)
    return result
AuditedDFBASimulator._certified_linprog=capture
worker(dict(id='failure',condition='no_lactate',arm='two_equal_total',hours=1),DEST)
