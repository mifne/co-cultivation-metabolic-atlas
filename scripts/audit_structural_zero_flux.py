"""Read-only, query-specific exact sign propagation; NOT global GEM pruning."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compiled_basis_artifact import load_anchor


def forced_zero_columns(a,lower,upper):
    a=a.tocsr();lower=lower.copy();upper=upper.copy()
    zero=(lower==0)&(upper==0);initial=int(zero.sum());passes=0
    while True:
        before=int(zero.sum());passes+=1
        for row in range(a.shape[0]):
            cols=a.indices[a.indptr[row]:a.indptr[row+1]];values=a.data[a.indptr[row]:a.indptr[row+1]]
            cols=cols[values!=0];values=values[values!=0]
            if not len(cols):continue
            minimum=np.where(values>0,lower[cols],-upper[cols])
            maximum=np.where(values>0,upper[cols],-lower[cols])
            # All terms lie on the same side of zero; their zero sum forces
            # each term to zero. Exact comparisons, no sample-zero heuristic.
            if np.all(minimum==0) or np.all(maximum==0):
                if np.any(lower[cols]>0) or np.any(upper[cols]<0):raise ValueError('Inconsistent bounds')
                zero[cols]=True;lower[cols]=upper[cols]=0.
        if int(zero.sum())==before:break
    return zero,dict(initial_fixed_zero=initial,forced_zero=int(zero.sum()),passes=passes,
                     remaining_columns=int((~zero).sum()),remaining_equations=int(np.count_nonzero(a[:,~zero].getnnz(axis=1))))


if __name__=='__main__':
    report={}
    for index in (0,4,8):
        anchor=load_anchor(ROOT/f'results/pf_basis_train20286311_4_compiled/anchor_{index:04d}.npz')
        p=anchor['lp']
        if np.any(p.rhs[:p.neq]!=0):raise ValueError('Requires homogeneous equality rows')
        _,report[str(index)]=forced_zero_columns(p.a[:p.neq],p.lower,p.upper)
    print(json.dumps(dict(scope='Only these cached LP bounds; not safe to prune a GEM globally',results=report),indent=2))
