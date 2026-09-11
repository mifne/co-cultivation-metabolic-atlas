"""CUDA singleton postsolve regression; unchanged original-LP acceptance gates.

All solution pairs below are manufactured analytically. No CPU optimizer,
reference trace, or relaxed certificate is used. GPU execution belongs to the
coordinating agent so these tests do not compete with performance measurements.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.probe_downstream_gpu_coverage import paired_certificate
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from src.lp_trace import problem_hash
from src.lp_zero_face import ZeroFaceReduction


def _problem(a, rhs, lo, hi, c, neq):
    return (csr_matrix(a, dtype=float), *(np.asarray(value, dtype=float)
        for value in (rhs, lo, hi, c)), neq)


def _singleton(coefficient=1., cost=7., bounds=(-3., 4.)):
    return _problem([[coefficient, 0.], [0., -1.]], [0., -1.],
        [bounds[0], 0.], [bounds[1], 2.], [cost, 1.], 1)


def _check_original(problem, x, y):
    cert = paired_certificate(problem, x, y)
    assert cert['certificate_passed'], cert
    # Retain explicit limits here so a accidentally weakened helper cannot
    # silently turn this regression into a different accuracy target.
    assert cert['primal_residual'] <= 1e-5, cert
    assert cert['dual_violation'] <= 1e-7, cert
    assert cert['relative_kkt_gap'] <= 1e-7, cert


def test_cuda_signed_singleton_duals_match_cpu_for_both_coefficient_and_cost_signs():
    import cupy as cp
    ps = [_singleton(a, c) for a, c in ((1., 7.), (1., -7.), (-2., 7.), (-2., -7.))]
    hashes = [problem_hash(p) for p in ps]
    with ZeroFaceGpuBatchedIPM(ps, fix_singleton_equalities=True) as solver:
        assert solver.n == 1 and solver.m == 1
        x = cp.ones((4, 1), dtype=cp.float64)
        y = -cp.ones((4, 1), dtype=cp.float64)
        xx, yy = solver.lift_device(x, y)
        host_x, host_y = xx.get(), yy.get()
        for i, p in enumerate(ps):
            cpu = ZeroFaceReduction(p, fix_singleton_equalities=True)
            assert cpu.witnesses[0].orientation == 'equality'
            cx, cy = cpu.lift([1.], [-1.])
            np.testing.assert_allclose(host_x[i], cx, atol=1e-13, rtol=0.)
            np.testing.assert_allclose(host_y[i], cy, atol=1e-13, rtol=0.)
            np.testing.assert_allclose(p[4] - p[0].T @ host_y[i], 0., atol=1e-13, rtol=0.)
            _check_original(p, host_x[i], host_y[i])
        np.testing.assert_array_equal(x.get(), np.ones((4, 1)))
        np.testing.assert_array_equal(y.get(), -np.ones((4, 1)))
    assert hashes == [problem_hash(p) for p in ps]


def test_cuda_min_max_and_signed_singleton_reverse_cascade_matches_cpu():
    import cupy as cp
    # row0 becomes a singleton only after rows1/2/3 fix x0/x2/x3. Its
    # reverse dual repair changes all three previously fixed reduced costs.
    p = _problem([[-1., 1., -1., 1., 0.], [2., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., -1.]],
        [0., 0., 0., 0., -1.], [-2., -2., 0., -3., 0.], [2., 2., 3., 0., 2.],
        [-9., 7., -20., 20., 1.], 4)
    cpu = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert [(w.row, w.orientation) for w in cpu.witnesses] == [
        (1, 'equality'), (2, 'min'), (3, 'max'), (0, 'equality')]
    cx, cy = cpu.lift([1.], [-1.])
    assert cy[1] < 0. and cy[2] < 0. and cy[3] > 0.
    with ZeroFaceGpuBatchedIPM([p, p], fix_singleton_equalities=True) as solver:
        x, y = solver.lift_device(cp.ones((2, 1), dtype=cp.float64),
                                 -cp.ones((2, 1), dtype=cp.float64))
        for xx, yy in zip(x.get(), y.get()):
            np.testing.assert_allclose(xx, cx, atol=1e-13, rtol=0.)
            np.testing.assert_allclose(yy, cy, atol=1e-13, rtol=0.)
            _check_original(p, xx, yy)


def test_cuda_singleton_reduction_remains_disabled_by_default():
    import cupy as cp
    p = _singleton()
    with ZeroFaceGpuBatchedIPM([p]) as solver:
        assert solver.n == 2 and solver.m == 2
        assert not solver.face_plans[0].witnesses
        x = cp.asarray([[0., 1.]], dtype=cp.float64)
        y = cp.asarray([[7., -1.]], dtype=cp.float64)
        xx, yy = solver.lift_device(x, y)
        np.testing.assert_array_equal(xx.get(), x.get())
        np.testing.assert_array_equal(yy.get(), y.get())
        _check_original(p, xx.get()[0], yy.get()[0])


@pytest.mark.parametrize('bounds', [(-3., 4.), (-np.inf, 4.), (-3., np.inf), (-np.inf, np.inf)])
def test_cuda_singleton_zero_crossing_and_free_bounds(bounds):
    import cupy as cp
    p = _singleton(coefficient=-2., cost=7., bounds=bounds)
    with ZeroFaceGpuBatchedIPM([p], fix_singleton_equalities=True) as solver:
        x, y = solver.lift_device(cp.ones((1, 1), dtype=cp.float64),
                                 -cp.ones((1, 1), dtype=cp.float64))
        cpu_y = ZeroFaceReduction(p, fix_singleton_equalities=True).lift_dual([-1.])
        np.testing.assert_allclose(y.get()[0], cpu_y, atol=1e-13, rtol=0.)
        _check_original(p, x.get()[0], y.get()[0])


def test_cuda_tiny_singleton_is_retained_without_tightening_original_bounds():
    import cupy as cp
    p = _singleton(coefficient=1e-16, cost=0.)
    with ZeroFaceGpuBatchedIPM([p], fix_singleton_equalities=True) as solver:
        assert solver.n == 2 and solver.m == 2
        assert not solver.face_plans[0].witnesses
        assert solver.face_plans[0].skipped_small_singleton_rows.tolist() == [0]
        x = cp.asarray([[1., 1.]], dtype=cp.float64)
        y = cp.asarray([[0., -1.]], dtype=cp.float64)
        xx, yy = solver.lift_device(x, y)
        np.testing.assert_array_equal(xx.get(), x.get())
        np.testing.assert_array_equal(yy.get(), y.get())
        # The unchanged original residual gate permits this tiny nonzero
        # residual. The reduction must not replace x0 with an exact zero.
        _check_original(p, xx.get()[0], yy.get()[0])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_cuda_nonfinite_singleton_proposals_fail_original_gate(bad):
    import cupy as cp
    p = _singleton()
    with ZeroFaceGpuBatchedIPM([p], fix_singleton_equalities=True) as solver:
        x = cp.asarray([[1.]], dtype=cp.float64)
        y = cp.asarray([[bad]], dtype=cp.float64)
        assert not solver.certificate(x, y)[0]['certificate_passed']


def _composed_problem():
    a = np.array([[2., 0., 0., 0., 0., 0., 0.],
                  [0., 1., -1., 0., 0., 0., 0.],
                  [0., 0., 0., 1., 1., 1., 0.],
                  [0., 0., 0., 0., 1., -1., 1.],
                  [0., 0., 0., 1., 3., -1., 2.]])
    y = np.array([-3.5, 0., 1., 2., 3.])
    p = _problem(a, [0., 0., 3., 1., 5.], [-3., 0., 0., 0., 0., 0., 0.],
        [3., 10., 10., 10., 10., 10., 10.], a.T @ y, 5)
    return p, np.array([0., 1., 1., 1., 1., 1., 1.]), y


@pytest.mark.parametrize('solver_type', [ZeroFaceGpuBatchedIPM, ForestGpuBatchedIPM])
@pytest.mark.parametrize('exact_equalities', [False, True])
def test_cuda_singletons_compose_with_forest_and_exact_rows_without_cpu_optimizer(solver_type, exact_equalities):
    import cupy as cp
    p, x, y = _composed_problem()
    _check_original(p, x, y)
    with solver_type([p], fix_singleton_equalities=True,
                     exact_equalities=exact_equalities) as solver:
        assert any(w.orientation == 'equality' for w in solver.face_plans[0].witnesses)
        if exact_equalities:
            assert len(solver.equality_plans[0].removed_rows) == 1
        initial_x = cp.asarray(x[None, :], dtype=cp.float64)
        initial_y = cp.asarray(y[None, :], dtype=cp.float64)
        # Test compression/postsolve, not iterative convergence or timing.
        result = solver.solve(initial_x=initial_x, initial_y=initial_y, iterations=0)
        assert result['accepted'].all(), result['metrics']
        assert result['factor_count'] == 0 and result['cpu_lp_calls'] == 0
        xx, yy = result['x'].get()[0], result['y'].get()[0]
        _check_original(p, xx, yy)
        # Independently replay the original-space singleton proof even when
        # the GPU route also performed a forest/rank-exact row reduction.
        cpu = ZeroFaceReduction(p, fix_singleton_equalities=True)
        cx, cy = cpu.lift(xx[cpu.columns], yy[cpu.rows])
        np.testing.assert_allclose(xx, cx, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(yy, cy, atol=1e-12, rtol=0.)
        np.testing.assert_array_equal(initial_x.get()[0], x)
        np.testing.assert_array_equal(initial_y.get()[0], y)
