"""Lagged factors are proposals; original and full Newton gates are unchanged."""
import numpy as np
import pytest
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from tests.test_gpu_ipm_device_update import options as base_options
from tests.test_gpu_ipm_numeric_update import _problems
from scripts.probe_ipm_restart_mu_matched import _verify


def options():
    return dict(base_options(),krylov_coordinates='condensed')


@pytest.mark.parametrize('interval',[1,2,4])
def test_cuda_factor_lag_preserves_original_certificate(interval):
    with ForestGpuBatchedIPM(_problems(),**options()) as solver:
        result=solver.solve(iterations=120,factor_reuse_interval=interval)
        assert _verify(_problems(),result,2)['qualified']
        assert result['factor_reuse_interval']==interval
        assert result['cpu_lp_calls']==0
        if interval==1:
            assert result['factor_reuse_count']==result['factor_refresh_count']==0
        else:assert result['factor_reuse_count']>0


@pytest.mark.parametrize('interval',[0,-1,9,True,2.0,None])
def test_cuda_invalid_lag_rejected_before_factorization(interval):
    with ForestGpuBatchedIPM(_problems(),**options()) as solver:
        before=solver.factor.factor_count
        with pytest.raises(ValueError,match='Factor reuse'):
            solver.solve(iterations=3,factor_reuse_interval=interval)
        assert solver.factor.factor_count==before


def test_cuda_lagged_factor_failure_gets_same_state_current_refresh(monkeypatch):
    import src.gpu_globalized_condensed as module
    with ForestGpuBatchedIPM(_problems(),**options()) as solver:
        original_factor=solver._factor_newton
        original_solve=solver._solve_newton
        original_gmres=module.globalized_condensed_gmres
        factored_ratio=None
        stale_calls=[]
        def factor(ratio):
            nonlocal factored_ratio
            factored_ratio=ratio.copy()
            return original_factor(ratio)
        def solve(rhs,ratio,scaling):
            answer=original_solve(rhs,ratio,scaling)
            if not bool(solver.cp.array_equal(ratio,factored_ratio)):
                stale_calls.append(ratio.copy())
                answer[:]=0.
            return answer
        def gmres(solver_arg,rhs,answer,ratio,scaling,*args,**kwargs):
            proposal,diagnostic,invalid=original_gmres(solver_arg,rhs,answer,ratio,scaling,*args,**kwargs)
            if not bool(solver.cp.array_equal(ratio,factored_ratio)):proposal[:]=0.
            return proposal,diagnostic,invalid
        monkeypatch.setattr(solver,'_factor_newton',factor)
        monkeypatch.setattr(solver,'_solve_newton',solve)
        monkeypatch.setattr(module,'globalized_condensed_gmres',gmres)
        result=solver.solve(iterations=120,factor_reuse_interval=2)
        assert stale_calls and result['factor_refresh_count']>0
        assert result['factor_refresh_count']==result['factor_reuse_count']
        assert _verify(_problems(),result,2)['qualified']
        assert solver.regularization==1e-6
