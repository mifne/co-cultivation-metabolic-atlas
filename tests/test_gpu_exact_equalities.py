"""Exact row reduction through zero-face/forest and original GPU certificates."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem(order=None):
    a = csr_matrix([[1., 1., 1., 0.], [0., 1., -1., 1.], [1., 3., -1., 2.]])
    b = np.array([3., 1., 5.])
    c = np.asarray(a.T @ np.array([1., 2., 3.]))
    if order is not None:
        a, b = a[order].tocsr(), b[order]
    return a, b, np.zeros(4), np.full(4, 10.), c, 3


@pytest.mark.parametrize('solver_type', [ZeroFaceGpuBatchedIPM, ForestGpuBatchedIPM])
def test_cuda_exact_row_warm_compression_and_original_dual_lift(solver_type):
    import cupy as cp
    order = np.array([2, 0, 1])
    ps = [problem(), problem(order)]
    xx = cp.ones((2, 4), dtype=cp.float64)
    yy = cp.asarray([[1., 2., 3.], *[np.array([1., 2., 3.])[order].tolist()]])
    original_y = yy.copy()
    with solver_type(ps, exact_equalities=True) as solver:
        assert len(solver.equality_plans) == 2
        assert all(len(p.removed_rows) == 1 for p in solver.equality_plans)
        assert solver.m == 2
        result = solver.solve(initial_x=xx, initial_y=yy, iterations=0)
        assert result['accepted'].all(), result['metrics']
        assert result['factor_count'] == 0 and result['cpu_lp_calls'] == 0
        np.testing.assert_array_equal(yy.get(), original_y.get())
        assert result['x'].shape == (2, 4) and result['y'].shape == (2, 3)
        for p, x, y in zip(ps, result['x'].get(), result['y'].get()):
            assert paired_certificate(p, x, y)['certificate_passed']


@pytest.mark.parametrize('solver_type', [ZeroFaceGpuBatchedIPM, ForestGpuBatchedIPM])
def test_cuda_reduced_cold_solve_keeps_original_all_row_gate(solver_type):
    with solver_type([problem()], exact_equalities=True, globalized=True) as solver:
        result = solver.solve(iterations=80)
        assert result['accepted'].all(), result['metrics']
        assert paired_certificate(problem(), result['x'].get()[0], result['y'].get()[0])['certificate_passed']


def test_cuda_bad_original_state_never_hidden_by_row_reduction():
    import cupy as cp
    with ZeroFaceGpuBatchedIPM([problem()], exact_equalities=True) as solver:
        bad_x = cp.full((1, solver.n), np.nan, dtype=cp.float64)
        y = cp.zeros((1, solver.m), dtype=cp.float64)
        assert not solver.certificate(bad_x, y)[0]['certificate_passed']
        bad_x[:] = 0.
        assert not solver.certificate(bad_x, y)[0]['certificate_passed']


def test_cuda_default_does_not_enable_qr_or_remove_extra_rows():
    with ZeroFaceGpuBatchedIPM([problem()]) as solver:
        assert solver.equality_plans == [] and solver.m == 3


def test_invalid_option_fails_before_loading_gpu():
    with pytest.raises(ValueError, match='boolean'):
        ZeroFaceGpuBatchedIPM([problem()], exact_equalities=1)
