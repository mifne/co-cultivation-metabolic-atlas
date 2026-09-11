"""Offline CPU basis inspection, not an online GPU fallback."""
import json
from pathlib import Path
import time
import numpy as np
import highspy as hp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows=[]
    for stage in (1,2,3):
        d=np.load(ROOT/f"results/pf_seed20286001_cpu_diagnostic/step_016_stage_{stage}.npz")
        a=csr_matrix((d["data"],d["indices"],d["indptr"]),shape=tuple(d["shape"]))
        neq=int(d["neq"])
        lp=hp.HighsLp(); lp.num_col_=a.shape[1]; lp.num_row_=a.shape[0]
        lp.col_cost_=d["c"]; lp.col_lower_=d["lower"]; lp.col_upper_=d["upper"]
        lp.row_lower_=np.r_[d["rhs"][:neq],np.full(a.shape[0]-neq,-np.inf)]; lp.row_upper_=d["rhs"]
        lp.a_matrix_.format_=hp.MatrixFormat.kRowwise
        lp.a_matrix_.start_=a.indptr; lp.a_matrix_.index_=a.indices; lp.a_matrix_.value_=a.data
        solver=hp.Highs()
        for k,v in dict(output_flag=False,threads=1,solver="simplex").items(): solver.setOptionValue(k,v)
        solver.passModel(lp); started=time.perf_counter(); solver.run()
        basis=solver.getBasis(); info=solver.getInfo()
        cols=np.array([i for i,s in enumerate(basis.col_status) if s==hp.HighsBasisStatus.kBasic])
        active=np.array([i for i,s in enumerate(basis.row_status) if s!=hp.HighsBasisStatus.kBasic])
        elapsed=time.perf_counter()-started
        factor=splu(a[active][:,cols].tocsc())
        rows.append(dict(stage=stage,shape=list(a.shape),cpu_solve_seconds=elapsed,
            basic_columns=len(cols),active_rows=len(active),inverse_MiB=len(cols)**2*8/2**20,
            lower_nonbasic=sum(s==hp.HighsBasisStatus.kLower for s in basis.col_status),
            upper_nonbasic=sum(s==hp.HighsBasisStatus.kUpper for s in basis.col_status),
            objective=info.objective_function_value,saved_objective=float(d["cpu_objective"]),
            max_saved_flux_difference=float(np.max(np.abs(np.asarray(solver.getSolution().col_value)-d["cpu_x"])))))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__": main()
