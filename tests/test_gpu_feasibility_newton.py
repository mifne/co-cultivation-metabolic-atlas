import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_feasibility_newton import GpuFeasibilityNewton
from scripts.probe_gpu_bound_feasibility import auxiliary


def test_auxiliary_has_owned_bounds_and_preserves_original():
    a=csr_matrix([[1.,1.]])
    p=(a,np.array([2.]),np.zeros(2),np.ones(2),np.array([-1.,0.]),0)
    q=auxiliary([p])[0]
    assert q[2][0]==q[3][0]==1
    assert p[2][0]==0 and p[4][0]==-1
    assert not np.shares_memory(p[2],q[2])
    with pytest.raises(ValueError):auxiliary([(a,p[1],p[2],p[3],np.ones(2),0)])


def test_gpu_feasibility_newton_proposal_and_model_unchanged():
    import cupy as cp
    problem=(csr_matrix([[1.,1.],[1.,0.]]),np.array([1.,.8]),np.zeros(2),np.ones(2),np.zeros(2),1)
    with GpuCondensedBatchedIPM([problem],newton_backend='dual_schur',globalized=True,
            newton_krylov_iterations=16,device_checked_solves=True) as solver:
        snapshots=[v.copy() for v in (solver.e.data,solver.g.data,solver.b,solver.h,solver.c)]
        x=cp.zeros((1,2),dtype=cp.float64)
        answer,info=GpuFeasibilityNewton(solver).propose(x)
        assert info['original_certificate_required'] and info['cpu_lp_calls']==0
        assert float(cp.max(cp.abs(solver._mv(solver.e,answer,solver.ne)-solver.b)))<=1e-6
        cp.testing.assert_array_equal(x,cp.zeros_like(x))
        for value,old in zip((solver.e.data,solver.g.data,solver.b,solver.h,solver.c),snapshots):
            cp.testing.assert_array_equal(value,old)
        solver.c[0,0]=1.
        with pytest.raises(ValueError,match='Zero-cost'):GpuFeasibilityNewton(solver).propose(x)
