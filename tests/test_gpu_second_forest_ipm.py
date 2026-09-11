import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from src.lp_trace import problem_hash
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem():
    # Zero-face removal of x0 reveals the previously three-term x1=2*x2.
    return (csr_matrix([[1.,0.,0.],[1.,1.,-2.],[0.,0.,-1.]]),
            np.array([0.,0.,-.5]),np.zeros(3),np.full(3,3.),np.array([0.,1.,0.]),2)


@pytest.mark.parametrize('solver_type',[ZeroFaceGpuBatchedIPM,ForestGpuBatchedIPM])
@pytest.mark.parametrize('exact',[False,True])
def test_second_pass_cold_solve_lifts_original_certified_pair(solver_type,exact):
    import cupy as cp
    p=problem()
    original_hash=problem_hash(p)
    with solver_type([p,p],second_forest=True,exact_equalities=exact,
                     globalized=True,newton_krylov_iterations=8,factor_refinements=0) as solver:
        assert solver.n==1
        assert all(len(plan.eliminated_rows)==1 for plan in solver.secondary_forest_plans)
        result=solver.solve(iterations=80)
        assert result['accepted'].all(),result['metrics']
        assert result['second_forest']['enabled'] and result['cpu_lp_calls']==0
        for x,y in zip(result['x'].get(),result['y'].get()):
            np.testing.assert_allclose(x,[0.,1.,.5],atol=1e-6)
            assert paired_certificate(p,x,y)['certificate_passed']
        assert problem_hash(p)==original_hash


@pytest.mark.parametrize('solver_type',[ZeroFaceGpuBatchedIPM,ForestGpuBatchedIPM])
def test_second_pass_compression_and_nonfinite_rejection(solver_type):
    import cupy as cp
    p=problem()
    x=cp.asarray([[0.,1.,.5]])
    y=cp.asarray([[0.,1.,-2.]])
    bx,by=x.copy(),y.copy()
    with solver_type([p],second_forest=True,exact_equalities=True,globalized=True) as solver:
        result=solver.solve(initial_x=x,initial_y=y,iterations=0)
        assert result['accepted'].all(),result['metrics']
        assert result['factor_count']==0
        np.testing.assert_array_equal(x.get(),bx.get())
        np.testing.assert_array_equal(y.get(),by.get())
        bad=cp.full((1,solver.n),cp.nan,dtype=cp.float64)
        assert not solver.certificate(bad,cp.zeros((1,solver.m))) [0]['certificate_passed']
