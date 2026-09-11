import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_batched_ipm import GpuBatchedIPM
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem():
    return (csr_matrix([[1.,1.,0.],[0.,1.,1.]]),np.array([2.,3.]),
            np.array([0.,0.,-np.inf]),np.array([10.,np.inf,4.]),np.array([1.,2.,-.5]),1)


def test_eliminated_bound_directions_satisfy_identical_full_newton_equation():
    import cupy as cp
    ps=[problem()]*2
    with GpuBatchedIPM(ps) as full,GpuCondensedBatchedIPM(ps) as small:
        cp.random.seed(42)
        ratio=cp.exp(cp.random.normal(size=(2,full.ng)))
        rhs=cp.random.normal(size=(2,full.size))
        x=full._solve_newton(rhs,ratio,full._factor_newton(ratio))
        y=small._solve_newton(rhs,ratio,small._factor_newton(ratio))
        np.testing.assert_allclose(y.get(),x.get(),rtol=1e-10,atol=1e-10)
        assert float(cp.max(cp.abs(small._kkt_mv(y,ratio)-rhs)))<1e-10
        assert small.factor.n<full.factor.n


def test_condensed_lp_keeps_full_original_certificate():
    with GpuCondensedBatchedIPM([problem()]*2) as solver:
        r=solver.solve(iterations=40)
        assert r['accepted'].all(),r['metrics']
        for x,y in zip(r['x'].get(),r['y'].get()):
            assert paired_certificate(problem(),x,y)['certificate_passed']


def test_no_explicit_inequality_rows_only_bounds():
    p=(csr_matrix([[1.,1.]]),np.array([1.]),np.zeros(2),np.full(2,np.inf),np.array([1.,2.]),1)
    with GpuCondensedBatchedIPM([p]) as solver:
        r=solver.solve(iterations=40)
        assert r['accepted'].all(),r['metrics']
