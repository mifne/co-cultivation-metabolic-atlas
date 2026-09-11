"""A bounded barrier schedule must not change the LP or leak factor state."""
import numpy as np
import pytest

from src.gpu_globalized_ipm import barrier_regularization,retry_regularizations
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.lp_trace import problem_hash
from tests.test_gpu_globalized_options import problem


def test_schedule_keeps_early_base_then_shrinks_with_largest_active_mu():
    active=np.array([True,True])
    for mu,expected in (([1.,2.],1e-6),([1e-6,2e-6],2e-8),([1e-14,2e-14],1e-12)):
        assert barrier_regularization(1e-6,np.array(mu),active,xp=np)==pytest.approx(expected)
    assert barrier_regularization(1e-6,np.array([1e-7,np.nan]),
        np.array([True,False]),xp=np)==pytest.approx(1e-9)
    assert barrier_regularization(1e-6,np.array([np.nan,np.inf]),
        np.array([False,False]),xp=np)==1e-6
    # Never raise a caller's smaller configured delta up to the floor.
    assert barrier_regularization(1e-14,np.array([1.]),np.array([True]),xp=np)==1e-14


@pytest.mark.parametrize('mu',[np.array([0.]),np.array([-1.]),np.array([np.nan]),np.array([np.inf])])
def test_active_invalid_complementarity_cannot_select_a_factor(mu):
    with pytest.raises(ValueError):
        barrier_regularization(1e-6,mu,np.array([True]),xp=np)


def test_schedule_retry_uses_explicit_lower_floor_but_legacy_is_unchanged():
    assert retry_regularizations(1e-8,2)==[]
    np.testing.assert_allclose(retry_regularizations(1e-8,2,floor=1e-12),[1e-9,1e-10])
    assert retry_regularizations(1e-12,2,floor=1e-12)==[]


def test_cuda_scheduled_factor_and_original_certificate_restore_configuration():
    p=problem()
    original_hash=problem_hash(p)
    with GpuCondensedBatchedIPM([p],globalized=True,ipm_initialization='balanced',
            regularization=1e-4,regularization_schedule='barrier',
            newton_krylov_iterations=8) as solver:
        r=solver.solve(iterations=100)
        assert r['accepted'].all(),r['metrics']
        values=[v['delta'] for v in r['regularization_schedule_history']]
        assert values[0]==1e-4 and min(values)<1e-4
        assert solver.regularization==1e-4 and problem_hash(p)==original_hash
        assert r['cpu_lp_calls']==0


def test_cuda_failed_scheduled_factor_cannot_leak_delta(monkeypatch):
    from src.gpu_sparse_factor import CudssError
    import src.gpu_globalized_ipm as module
    monkeypatch.setattr(module,'barrier_regularization',lambda *args,**kwargs:1e-12)
    with GpuCondensedBatchedIPM([problem()],globalized=True,
            ipm_initialization='balanced',regularization_schedule='barrier') as solver:
        calls=[]
        def fail(ratio):
            calls.append(solver.regularization)
            raise CudssError('manufactured scheduled factor failure')
        monkeypatch.setattr(solver,'_factor_newton',fail)
        r=solver.solve(iterations=1)
        assert calls==[1e-12] and r['status']=='gpu_numeric_failure'
        assert solver.regularization==1e-9
        assert not r['accepted'].any()
