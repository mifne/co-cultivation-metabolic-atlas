import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_batched_ipm import GpuBatchedIPM, constraint_form
from scripts.probe_downstream_gpu_coverage import paired_certificate


def example(rhs=1., cost=(1., 2.)):
    return (csr_matrix([[1., 1.]]), np.array([rhs]), np.zeros(2), np.full(2, np.inf),
            np.array(cost), 1)


def test_dictionary_free_independent_gpu_lp_solutions_and_original_certificates():
    ps = [example(), example(2., (3., 1.)), example(4., (1., 2.))]
    with GpuBatchedIPM(ps) as solver:
        result = solver.solve(iterations=30)
        assert result['accepted'].all(), result['metrics']
        np.testing.assert_allclose(result['x'].get(), [[1., 0.], [0., 2.], [4., 0.]], atol=1e-6)
        for p, x, y in zip(ps, result['x'].get(), result['y'].get()):
            assert paired_certificate(p, x, y)['certificate_passed']
        assert result['cpu_lp_calls'] == 0 and result['analysis_count'] == 1
        assert 2*result['factor_count'] <= result['solve_count'] <= 2*(1+result['newton_refinements'])*result['factor_count']
        assert [r['objective'] for r in result['checkpoints'][0]['metrics']]==[0.,0.,0.]
        assert not any(r['certificate_passed'] for r in result['checkpoints'][0]['metrics'])


def test_fixed_variable_and_free_variable_and_inequality_are_retained():
    p = (csr_matrix([[1., 1., 0.], [0., 0., -1.]]), np.array([3., -2.]),
         np.array([-np.inf, 1., -np.inf]), np.array([np.inf, 1., np.inf]),
         np.array([0., 0., 1.]), 1)
    e, b, g, h, fixed, il, iu = constraint_form(p)
    assert e.shape == (2, 3) and g.shape == (1, 3)
    with GpuBatchedIPM([p]) as solver:
        r = solver.solve(iterations=35)
        assert r['accepted'].all(), r['metrics']
        np.testing.assert_allclose(r['x'].get(), [[2., 1., 2.]], atol=1e-6)
        assert paired_certificate(p, r['x'].get()[0], r['y'].get()[0])['certificate_passed']


def test_infeasible_problem_never_reported_as_certified():
    p = example(-1.)
    with GpuBatchedIPM([p]) as solver:
        r = solver.solve(iterations=6)
        assert not r['accepted'].any() and r['cpu_lp_calls'] == 0


def test_warm_certified_pair_is_not_changed_and_input_device_required():
    import cupy as cp
    with GpuBatchedIPM([example()]) as solver:
        x, y = cp.asarray([[1., 0.]]), cp.asarray([[1.]])
        r = solver.solve(initial_x=x, initial_y=y, iterations=0)
        assert r['accepted'].all() and r['factor_count'] == 0
        np.testing.assert_array_equal(r['y'].get(), [[1.]])
        with pytest.raises(ValueError):solver.solve(initial_x=np.ones((1, 2)))


def test_different_bound_masks_rejected_before_gpu_factorization():
    p = example()
    other = (*p[:2], np.array([0., 1.]), np.array([np.inf, 1.]), *p[4:])
    with pytest.raises(ValueError):GpuBatchedIPM([p, other])


def test_inaccurate_newton_direction_cannot_advance_state(monkeypatch):
    import cupy as cp
    with GpuBatchedIPM([example()]) as solver:
        monkeypatch.setattr(solver.factor, 'solve', lambda rhs: cp.zeros_like(rhs))
        r=solver.solve(iterations=3)
        assert r['status']=='unreliable_affine_direction'
        assert r['iterations']==0 and not r['accepted'].any()
        np.testing.assert_array_equal(r['x'].get(),np.zeros((1,2)))


def test_uncertified_zero_budget_and_repeated_workspace_solve_do_not_claim_success():
    with GpuBatchedIPM([example()]) as solver:
        r=solver.solve(iterations=0)
        assert not r['accepted'].any() and r['factor_count']==0
        first=solver.solve(iterations=30)
        second=solver.solve(iterations=30)
        assert first['accepted'].all() and second['accepted'].all()
        assert solver.factor.analysis_count==1


def test_regularized_residual_cannot_hide_original_newton_defect():
    import cupy as cp
    with GpuBatchedIPM([example()],regularization=1e-6) as solver:
        d=cp.zeros((1,solver.size),dtype=cp.float64);d[:,solver.n]=100.
        ratio=cp.ones((1,solver.ng))
        rhs=solver._kkt_mv(d,ratio)
        original=rhs-solver._kkt_mv(d,ratio,regularized=False)
        assert float(solver._direction_error(rhs,original)[0])==pytest.approx(1e-4)
        assert float(solver._direction_error(rhs,rhs-solver._kkt_mv(d,ratio))[0])==0.


def test_bad_affine_direction_is_rejected_before_corrector_product(monkeypatch):
    import cupy as cp
    with GpuBatchedIPM([example()]) as solver:
        calls=[]
        def wrong(rhs,ratio,scaling):
            calls.append(rhs.copy())
            return cp.ones_like(rhs)*1e30
        monkeypatch.setattr(solver,'_solve_newton',wrong)
        r=solver.solve(iterations=3)
        # One first solve and at most bounded refinement; no corrector RHS.
        assert len(calls)<=1+solver.newton_refinements
        assert r['iterations']==0 and r['status']=='unreliable_affine_direction'
