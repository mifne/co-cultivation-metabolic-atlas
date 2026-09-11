"""Lossless equality-only scaling and physical dual/certificate integration.

Use ``-k 'not cuda'`` for independent CPU checks. CUDA integration tests are
explicitly named so they can be run separately from benchmark experiments.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_batched_ipm import GpuBatchedIPM, constraint_form
from src.gpu_ipm_equality_scaling import scale_equality_form
from src.lp_trace import problem_hash


def _form(e=None, b=None):
    e = csr_matrix([[-.0002, .01], [0., 2.], [0., 0.]], dtype=np.float64) if e is None else e
    b = np.zeros(e.shape[0], dtype=np.float64) if b is None else b
    n = e.shape[1]
    return (e, b, csr_matrix(np.eye(n)), np.ones(n),
            np.zeros(n, dtype=bool), np.arange(n), np.arange(n))


def test_weak_row_gets_power_of_two_scale_64_and_exact_roundtrip():
    original = _form()
    scaled, scale = scale_equality_form(original)
    np.testing.assert_array_equal(scale, [64., .25, 1.])
    np.testing.assert_array_equal(scaled[0].toarray()[0], [-.0128, .64])
    rows = np.repeat(np.arange(len(scale)), np.diff(original[0].indptr))
    exponents = np.rint(np.log2(scale)).astype(np.int32)
    np.testing.assert_array_equal(np.ldexp(scaled[0].data, -exponents[rows]), original[0].data)
    np.testing.assert_array_equal(np.ldexp(scaled[1], -exponents), original[1])
    assert scale.dtype == np.float64 and scaled[0].dtype == scaled[1].dtype == np.float64


def test_rhs_participates_and_inconsistent_empty_row_is_not_deleted():
    original = _form(csr_matrix([[.01, 0.], [0., 0.], [0., 0.]]), np.array([4., -2., 0.]))
    scaled, scale = scale_equality_form(original)
    np.testing.assert_array_equal(scale, [.125, .25, 1.])
    np.testing.assert_array_equal(scaled[1], [.5, -.5, 0.])
    assert scaled[0].shape == original[0].shape
    assert scaled[0].nnz == original[0].nnz == 1


def test_inputs_not_mutated_equality_outputs_independent_other_members_identical():
    original = _form()
    e_data, e_indices, e_indptr, b = (original[0].data.copy(), original[0].indices.copy(),
                                    original[0].indptr.copy(), original[1].copy())
    scaled, scale = scale_equality_form(original)
    for index in range(2, 7):
        assert scaled[index] is original[index]
    assert not np.shares_memory(scaled[0].data, original[0].data)
    assert not np.shares_memory(scaled[1], original[1])
    scaled[0].data[:] = 123.
    scaled[1][:] = 456.
    scale[:] = 789.
    for actual, expected in ((original[0].data, e_data), (original[0].indices, e_indices),
                             (original[0].indptr, e_indptr), (original[1], b)):
        np.testing.assert_array_equal(actual, expected)


def test_noncanonical_sparse_storage_is_neither_summed_nor_sorted():
    # Two entries for column 1 deliberately remain separate (no hidden row edit).
    e = csr_matrix((np.array([.01, -.0002, .005]), np.array([1, 0, 1]),
                    np.array([0, 3])), shape=(1, 2))
    scaled, scale = scale_equality_form(_form(e))
    np.testing.assert_array_equal(scale, [64.])
    np.testing.assert_array_equal(scaled[0].indices, e.indices)
    np.testing.assert_array_equal(scaled[0].indptr, e.indptr)
    np.testing.assert_array_equal(scaled[0].data, e.data * 64.)
    assert scaled[0].nnz == 3


@pytest.mark.parametrize('magnitude,expected', [(2.**-41, 2.**40), (2.**39, 2.**-40)])
def test_exact_exponent_limits_are_accepted(magnitude, expected):
    _, scale = scale_equality_form(_form(csr_matrix([[magnitude]])))
    np.testing.assert_array_equal(scale, [expected])


@pytest.mark.parametrize('magnitude', [2.**-42, 2.**40])
def test_exponent_outside_limits_is_rejected_not_clipped(magnitude):
    with pytest.raises(ValueError, match='bounded range'):
        scale_equality_form(_form(csr_matrix([[magnitude]])))


@pytest.mark.parametrize('location', ['coefficient', 'rhs'])
def test_subnormal_loss_is_rejected_instead_of_silently_changing_equation(location):
    tiny = np.nextafter(0., 1.)
    e = csr_matrix([[1., tiny if location == 'coefficient' else 0.]])
    b = np.array([tiny if location == 'rhs' else 0.])
    with pytest.raises(ValueError, match='lossless'):
        scale_equality_form(_form(e, b))
    assert (e.data[-1] if location == 'coefficient' else b[0]) == tiny


@pytest.mark.parametrize('location', ['coefficient', 'rhs'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_data_fails_closed(location, bad):
    original = list(_form())
    if location == 'coefficient':
        original[0].data[0] = bad
    else:
        original[1][0] = bad
    with pytest.raises(ValueError, match='Finite'):
        scale_equality_form(original)


@pytest.mark.parametrize('kind', ['fp32_e', 'fp32_b', 'dense_e', 'rhs_shape', 'wrong_width',
                                 'fixed_shape', 'negative_index', 'large_index', 'float_index'])
def test_invalid_metadata_rejected(kind):
    original = list(_form())
    if kind == 'fp32_e':
        original[0] = original[0].astype(np.float32)
    elif kind == 'fp32_b':
        original[1] = original[1].astype(np.float32)
    elif kind == 'dense_e':
        original[0] = original[0].toarray()
    elif kind == 'rhs_shape':
        original[1] = original[1][:, None]
    elif kind == 'wrong_width':
        original[2] = csr_matrix(np.eye(3))
        original[3] = np.ones(3)
    elif kind == 'fixed_shape':
        original[4] = np.zeros(3, dtype=bool)
    elif kind == 'negative_index':
        original[5] = np.array([-1])
    elif kind == 'large_index':
        original[6] = np.array([2])
    else:
        original[5] = np.array([0.])
    with pytest.raises(ValueError):
        scale_equality_form(original)


def test_no_equalities_is_valid_and_keeps_other_form_members():
    original = _form(csr_matrix((0, 2), dtype=np.float64))
    scaled, scale = scale_equality_form(original)
    assert scale.shape == (0,) and scaled[0].shape == (0, 2)
    assert scaled[1].shape == (0,)
    assert all(scaled[i] is original[i] for i in range(2, 7))


def test_original_problem_objective_bounds_and_inequality_remain_unchanged():
    p = (csr_matrix([[-.0002, .01], [1., -2.]]), np.array([.0098, 4.]),
         np.array([0., 1.]), np.array([10., 1.]), np.array([2., 3.]), 1)
    before = problem_hash(p)
    original = constraint_form(p)
    scaled, scale = scale_equality_form(original)
    assert scale.tolist() == [64., .5]  # Includes the exact fixed-bound equality.
    assert scaled[0].shape == (2, 2)
    assert all(scaled[i] is original[i] for i in range(2, 7))
    assert problem_hash(p) == before


@pytest.mark.parametrize('enabled', [False, True])
def test_row_dual_map_uses_batch_scales_and_preserves_inequality_sign(enabled):
    solver = object.__new__(GpuBatchedIPM)
    solver.cp, solver.neq, solver.m = np, 2, 3
    solver.equality_row_scaling = enabled
    solver.equality_scale = np.array([[64., .25, .5], [32., .125, .25]])
    internal = np.array([[1., -2., 5.], [3., -4., 7.]])
    z = np.array([[.2, 2., 3.], [.4, 4., 5.]])
    actual = solver._row_dual(internal, z)
    physical = internal * solver.equality_scale if enabled else internal
    expected = np.column_stack((-physical[:, :2], -z[:, 0]))
    np.testing.assert_array_equal(actual, expected)
    # The appended fixed equality is implicit in bound duals, not a new LP row.
    assert actual.shape == (2, 3)
    np.testing.assert_array_equal(internal, [[1., -2., 5.], [3., -4., 7.]])


@pytest.mark.parametrize('bad', [1, None, 'true', np.bool_(True)])
def test_scaling_flag_requires_explicit_boolean_before_gpu_allocation(bad, monkeypatch):
    import src.gpu_batched_ipm as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid scaling flag must not construct a GPU factor')
    monkeypatch.setattr(module, 'UniformCudssFactor', forbidden)
    with pytest.raises(ValueError, match='equality_row_scaling must be boolean'):
        GpuBatchedIPM([_warm_problem()], equality_row_scaling=bad)


def _warm_problem():
    a = csr_matrix([[-.0002, .01], [1., 0.]])
    y = np.array([2., 0.])
    return (a, np.array([.0098, 4.]), np.zeros(2), np.full(2, 10.),
            np.asarray(a.T @ y), 1)


@pytest.mark.parametrize('globalized', [False, True])
def test_cuda_certified_warm_pair_is_preserved_and_original_lp_is_unchanged(globalized):
    import cupy as cp
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    p = _warm_problem()
    original_hash = problem_hash(p)
    x, y = cp.asarray([[1., 1.]]), cp.asarray([[2., 0.]])
    with GpuBatchedIPM([p], equality_row_scaling=True, globalized=globalized) as solver:
        np.testing.assert_array_equal(solver.equality_scale.get(), [[64.]])
        result = solver.solve(initial_x=x, initial_y=y, iterations=0)
        assert result['accepted'].tolist() == [True]
        assert result['factor_count'] == result['cpu_lp_calls'] == 0
        np.testing.assert_array_equal(result['x'].get(), x.get())
        np.testing.assert_array_equal(result['y'].get(), y.get())
        assert paired_certificate(p, result['x'][0].get(), result['y'][0].get())['certificate_passed']
        assert solver.problem_hashes == (original_hash,)
    assert problem_hash(p) == original_hash


@pytest.mark.parametrize('globalized', [False, True])
def test_cuda_initial_dual_inverse_map_includes_fixed_bound_rows(globalized, monkeypatch):
    import cupy as cp
    p = (csr_matrix([[.0002, .01, 0.], [0., 0., 1.]]), np.array([.0102, 3.]),
         np.array([0., 0., 2.]), np.array([5., 5., 2.]), np.array([2., 3., 4.]), 1)
    initial_y = cp.asarray([[.75, -.2]])
    class InitialDualObserved(Exception):
        pass
    with GpuBatchedIPM([p], equality_row_scaling=True, globalized=globalized) as solver:
        captured = []
        original_mv = solver._mv
        def observe(matrix, value, columns):
            if matrix is solver.et:
                captured.append(value.copy())
                raise InitialDualObserved
            return original_mv(matrix, value, columns)
        monkeypatch.setattr(solver, '_mv', observe)
        with pytest.raises(InitialDualObserved):
            solver.solve(initial_y=initial_y, iterations=1)
        # Equality multiplier -0.75 and fixed-row multiplier -(c-A.T*y)[2].
        np.testing.assert_array_equal(solver.equality_scale.get(), [[64., .25]])
        np.testing.assert_allclose(captured[0].get(), [[-.75/64., -4.2/.25]], rtol=0., atol=0.)
        np.testing.assert_array_equal(initial_y.get(), [[.75, -.2]])
        assert solver.factor.factor_count == 0


def test_cuda_certificate_uses_unscaled_original_residual_not_small_working_row():
    import cupy as cp
    # Working residual is about 1e-6 after scaling, but original residual is 1.
    p = (csr_matrix([[2.**20, 0.]]), np.array([2.**20]),
         np.zeros(2), np.full(2, 10.), np.zeros(2), 1)
    with GpuBatchedIPM([p], equality_row_scaling=True) as solver:
        bad_x = cp.asarray([[1. + 2.**-20, 0.]])
        y = cp.zeros((1, 1), dtype=cp.float64)
        working_residual = solver._mv(solver.e, bad_x, solver.ne)-solver.b
        assert float(cp.max(cp.abs(working_residual))) < 1e-5
        metrics = solver.certificate(bad_x, y)
        assert not metrics[0]['certificate_passed']
        result = solver.solve(initial_x=bad_x, initial_y=y, iterations=0)
        assert not result['accepted'].any()


@pytest.mark.parametrize('globalized', [False, True])
def test_cuda_scaled_cold_solution_is_certified_on_original_problem(globalized):
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    p = (csr_matrix([[.0002, .01]]), np.array([.01]), np.zeros(2),
         np.full(2, 10.), np.array([2., 1.]), 1)
    with GpuBatchedIPM([p], equality_row_scaling=True, globalized=globalized) as solver:
        result = solver.solve(iterations=90)
        assert result['accepted'].tolist() == [True], result['metrics']
        assert result['cpu_lp_calls'] == 0
        assert paired_certificate(p, result['x'][0].get(), result['y'][0].get())['certificate_passed']
        np.testing.assert_allclose(result['x'].get(), [[0., 1.]], atol=2e-6)
