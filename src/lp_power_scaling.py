"""Reversible power-of-two LP equilibration; no presolve or deleted rows.

x_original = column_scale * x_scaled
y_original = row_scale * objective_scale * y_scaled (HiGHS convention)
Power-of-two scales avoid additional representation error for normal FP64
values. All final feasibility/optimality checks remain in original units.
"""
from dataclasses import dataclass

import numpy as np
from .gpu_pdhg_corrector import _validated_problem


@dataclass(frozen=True)
class ScaledLP:
    original: tuple
    problem: tuple
    row_scale: np.ndarray
    column_scale: np.ndarray
    objective_scale: float


def power_equilibrate(problem, *, rounds=6):
    if type(rounds) is not int or not 0<=rounds<=12:
        raise ValueError('Equilibration rounds must be an integer in 0..12')
    original=_validated_problem(problem)
    a,rhs,lo,hi,c,neq=original
    a=a.copy();m,n=a.shape
    rows=np.repeat(np.arange(m),np.diff(a.indptr))
    re=np.zeros(m,dtype=int);ce=np.zeros(n,dtype=int)
    # Bound cumulative exponents to avoid overflow or underflow from scaling.
    for _ in range(rounds):
        maximum=np.zeros(m);np.maximum.at(maximum,rows,np.abs(a.data))
        delta=np.clip(-np.rint(np.log2(np.where(maximum>0,maximum,1.))/2),-40-re,40-re).astype(int)
        a.data*=np.exp2(delta[rows]);re+=delta
        maximum=np.zeros(n);np.maximum.at(maximum,a.indices,np.abs(a.data))
        delta=np.clip(-np.rint(np.log2(np.where(maximum>0,maximum,1.))/2),-40-ce,40-ce).astype(int)
        a.data*=np.exp2(delta[a.indices]);ce+=delta
    rs,cs=np.exp2(re.astype(float)),np.exp2(ce.astype(float))
    cost=c*cs
    objective_scale=float(np.exp2(np.clip(np.rint(np.log2(max(np.max(np.abs(cost)),1e-300))),-40,40)))
    if not np.any(cost):objective_scale=1.
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        scaled=(a,rhs*rs,lo/cs,hi/cs,cost/objective_scale,neq)
    if (not np.isfinite(a.data).all() or np.any((original[0].data!=0)&(a.data==0))
            or not np.isfinite(scaled[1]).all() or not np.isfinite(scaled[4]).all()
            or not np.array_equal(np.isfinite(lo),np.isfinite(scaled[2]))
            or not np.array_equal(np.isfinite(hi),np.isfinite(scaled[3]))
            or not np.array_equal(lo==hi,scaled[2]==scaled[3])):
        raise ValueError('LP scaling changed finite/fixed domains or lost coefficients')
    return ScaledLP(original,scaled,rs,cs,objective_scale)


from .gpu_batched_ipm import GpuBatchedIPM


class ScaledGpuBatchedIPM(GpuBatchedIPM):
    """GPU iterates in equivalent coordinates, accepts only ORIGINAL LP gates."""
    def __init__(self,problems,**kwargs):
        import time
        before=time.perf_counter()
        self.scaling_plans=[power_equilibrate(p) for p in problems]
        if not self.scaling_plans:raise ValueError('Nonempty LP batch required')
        self.original_problems=[s.original for s in self.scaling_plans]
        super().__init__([s.problem for s in self.scaling_plans],**kwargs)
        from .lp_trace import problem_hash
        from .gpu_block_lp import assemble_blocks
        from cupyx.scipy.sparse import csr_matrix
        cp=self.cp
        self.problem_hashes=tuple(problem_hash(p) for p in self.original_problems)
        self.row_scale=cp.asarray(np.stack([s.row_scale*s.objective_scale for s in self.scaling_plans]))
        self.column_scale=cp.asarray(np.stack([s.column_scale for s in self.scaling_plans]))
        packed=assemble_blocks(self.original_problems)
        self.original_assembled=(csr_matrix(packed[0]),*(cp.asarray(v) for v in packed[1:]))
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds=time.perf_counter()-before

    def certificate(self,x,y):
        from .gpu_block_lp import certify_blocks_device
        return certify_blocks_device(self.original_problems,self.original_assembled,
            (x*self.column_scale).ravel(),(y*self.row_scale).ravel(),cp=self.cp)

    def solve(self,*,initial_x=None,initial_y=None,**kwargs):
        x=self._initial(initial_x,(self.batch,self.n))/self.column_scale
        y=self._initial(initial_y,(self.batch,self.m))/self.row_scale
        result=super().solve(initial_x=x,initial_y=y,**kwargs)
        result['x']*=self.column_scale
        result['y']*=self.row_scale
        result['lp_power_scaling']=True
        return result
