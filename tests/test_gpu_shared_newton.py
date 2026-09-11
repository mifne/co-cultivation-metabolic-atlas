import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM


def inputs():
    return [(csr_matrix([[1.,-1.,0.],[0.,1.,1.]]),np.array([0.,2.]),
        np.zeros(3),np.array([2.,2.,1.]),np.array([0.,0.,-1.]),1) for _ in range(4)]


def options():
    return dict(globalized=True,newton_krylov_iterations=16,regularization=1e-6,
        device_checked_solves=True,predictor_corrector=True,shared_factor_group_size=2)


def test_shared_identical_factors_preserve_independent_rhs_and_bad_column():
    import cupy as cp
    with GpuCondensedBatchedIPM(inputs(),**options()) as s:
        s._shared_factor_enabled=True
        ratio=cp.ones((s.batch,s.ng))
        rhs=cp.asarray(np.linspace(-1.,2.,s.batch*s.size).reshape(s.batch,s.size))
        scale=s._factor_newton(ratio)
        result=s._solve_newton(rhs,ratio,scale)
        residual=s._pack_rhs(rhs,ratio)-s._condensed_mv(result[:,:s.condensed_size],ratio,regularized=True)
        assert float(cp.max(cp.abs(residual)))<1e-8
        packed=cp.ones((4,s.factor.n,1))
        good=s._shared_newton.solve(packed)
        saved=good.copy()
        packed[0]=cp.nan
        bad=s._shared_newton.solve(packed)
        assert bool(cp.isnan(bad[0]).all())
        cp.testing.assert_allclose(bad[1:],saved[1:])
        cp.testing.assert_array_equal(good,saved)


def test_inactive_members_do_not_change_shared_factor():
    import cupy as cp
    with GpuCondensedBatchedIPM(inputs(),**options()) as s:
        s._shared_factor_enabled=True
        s._current_factor_active=cp.asarray([True,False,True,False])
        ratio=cp.ones((4,s.ng));ratio[1::2]*=100
        scale=s._factor_newton(ratio)
        rhs=cp.ones((4,s.size))
        answer=s._solve_newton(rhs,ratio,scale)
        residual=s._pack_rhs(rhs,ratio)-s._condensed_mv(answer[:,:s.condensed_size],ratio,regularized=True)
        assert float(cp.max(cp.abs(residual[::2])))<1e-8


def test_cold_solve_keeps_independent_certified_factors():
    with GpuCondensedBatchedIPM(inputs(),**options()) as s:
        result=s.solve(iterations=100)
        assert result['accepted'].all()
        assert not result['shared_factor_active']
        assert s._shared_newton.factor.factor_count==0


def test_invalid_cluster_size():
    opts=options();opts['shared_factor_group_size']=3
    with pytest.raises(ValueError,match='divide'):GpuCondensedBatchedIPM(inputs(),**opts)
