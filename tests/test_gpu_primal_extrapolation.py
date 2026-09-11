import numpy as np
from scipy.sparse import csr_matrix
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_primal_extrapolation import certified_extrapolation


def problems(cost=0.):
    return [(csr_matrix([[1.,-1.]]),np.array([0.]),np.zeros(2),np.ones(2),
             np.array([cost,0.]),1)]


def test_ray_certificate_does_not_mutate_input():
    import cupy as cp
    with GpuCondensedBatchedIPM(problems()) as s:
        old=cp.array([[-1.,-1.]]);direction=cp.array([[3.,3.]])
        candidate,passed,metrics=certified_extrapolation(s,old,direction,cp.array([True]))
        assert passed.all() and metrics[0]['certificate_passed']
        assert np.allclose(candidate.get(),.5)
        assert np.array_equal(old.get(),[[-1.,-1.]])
        assert np.array_equal(direction.get(),[[3.,3.]])
        assert s.factor.factor_count==0


def test_primal_feasible_is_not_optimality_certificate():
    import cupy as cp
    with GpuCondensedBatchedIPM(problems(-1.)) as s:
        _,passed,_=certified_extrapolation(s,cp.zeros((1,2)),cp.ones((1,2)),cp.array([True]))
        assert not passed.any()


def test_ineligible_direction_cannot_accept():
    import cupy as cp
    with GpuCondensedBatchedIPM(problems()) as s:
        _,passed,metrics=certified_extrapolation(s,cp.zeros((1,2)),cp.ones((1,2)),cp.array([False]))
        assert not passed.any() and metrics is None


def test_no_feasible_ray_cannot_accept():
    import cupy as cp
    with GpuCondensedBatchedIPM(problems()) as s:
        _,passed,metrics=certified_extrapolation(s,cp.array([[-1.,-1.]]),-cp.ones((1,2)),cp.array([True]))
        assert not passed.any() and metrics is None


def test_opt_in_solver_keeps_certified_output_and_positive_warm_proposal():
    import cupy as cp
    with GpuCondensedBatchedIPM(problems(),globalized=True,predictor_corrector=True,
            certified_primal_extrapolation=True,retain_internal_state=True) as s:
        r=s.solve(initial_x=cp.array([[-1.,-1.]]),iterations=60)
        assert r['accepted'].all()
        assert s.certificate(r['x'],r['y'])[0]['certificate_passed']
        assert r['certified_primal_extrapolation']
        assert cp.all(s._last_internal_state[2]>0.) and cp.all(s._last_internal_state[3]>0.)
