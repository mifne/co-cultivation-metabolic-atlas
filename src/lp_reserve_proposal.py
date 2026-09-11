"""A stricter auxiliary LP proposes an interior reserve, not a model edit.

Original-LP optimality MUST be independently checked. Auxiliary infeasibility
or optimality never proves the original LP infeasible or optimal.
"""
import math
import numpy as np
from .gpu_pdhg_corrector import _validated_problem


def reserve_proposal(problem, fraction=.5):
    if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0.<fraction<=1.:
        raise ValueError('Finite reserve fraction in (0,1] required')
    a,rhs,lo,hi,c,neq=_validated_problem(problem)
    if np.flatnonzero(c).size!=1:
        raise ValueError('Reserve prototype requires one objective coordinate')
    original_lo,original_hi=lo,hi
    lo,hi,rhs=lo.copy(),hi.copy(),rhs.copy()
    editable=(lo<hi)&(c==0.)
    lower=editable&np.isfinite(lo)&(lo<0.)
    upper=editable&np.isfinite(hi)&(hi>0.)
    proposed_lo=lo.copy();proposed_hi=hi.copy()
    proposed_lo[lower]*=fraction;proposed_hi[upper]*=fraction
    # Do not create a new fixed face or impossible interval merely to obtain
    # a reserve. Keeping an original interval is conservative, not a relaxation.
    valid=proposed_lo<proposed_hi
    lo[valid]=proposed_lo[valid];hi[valid]=proposed_hi[valid]
    positions=np.flatnonzero(rhs[neq:]>0.)+neq
    rhs[positions]*=fraction
    if not (np.all(lo>=original_lo) and np.all(hi<=original_hi)):
        raise ValueError('Auxiliary bounds must be a subset of original bounds')
    return (a.copy(),rhs,lo,hi,c.copy(),neq)
