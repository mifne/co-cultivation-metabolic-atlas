"""Small CPU references plus root-selected CUDA restart correctness tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix

from src.gpu_ipm_reoptimization import centered_bound_restart


def _solver(*, q=1, lower=(0, 1), upper=(0, 2), batch=2, ne=1):
    n, ng = 3, q+len(lower)+len(upper)
    original = np.array([[1., 1., 0.]])[:q]
    g = np.vstack((original, -np.eye(n)[list(lower)], np.eye(n)[list(upper)]))
    solver = SimpleNamespace(cp=np, batch=batch, n=n, ne=ne, ng=ng, q=q,
        il=np.asarray(lower, dtype=np.int64), iu=np.asarray(upper, dtype=np.int64),
        g=block_diag([csr_matrix(g)]*batch, format='csr'),
        h=np.tile(np.concatenate((np.ones(q), np.zeros(len(lower)),
                                 np.full(len(upper), 3.))), (batch, 1)),
        c=np.tile(np.array([2., -3., -4.]), (batch, 1)),
        factor=SimpleNamespace(_context=lambda: None))
    solver._mv = lambda matrix, value, width: (matrix @ value.ravel()).reshape(batch, width)
    return solver


def _x(batch=2): return np.tile(np.array([.25, .5, 1.]), (batch, 1))


def test_reference_formula_current_cost_signs_and_nonunit_complementarity():
    solver = _solver()
    (x, y, z, s), metadata = centered_bound_restart(solver, _x(), mu=1e-4)
    np.testing.assert_array_equal(x, _x())
    np.testing.assert_array_equal(y, np.zeros((2, 1)))
    np.testing.assert_allclose(s, [[.25, .25, .5, 2.75, 2.]]*2, rtol=0., atol=0.)
    np.testing.assert_allclose(z, [[4e-4, 2., 2e-4, 1e-4/2.75, 4.]]*2, rtol=1e-15)
    assert np.max(s*z) > metadata['mu']
    assert metadata['old_duals_discarded'] and metadata['old_slacks_discarded']
    assert metadata['current_LP_requires_new_certificate']
    assert metadata['cpu_lp_calls'] == metadata['vector_downloads'] == 0


def test_infeasible_x_is_unchanged_and_only_internal_slack_is_positive():
    solver = _solver()
    supplied = np.array([[-2., 5., 8.], [-3., 6., 9.]])
    (x, _y, z, s), _metadata = centered_bound_restart(solver, supplied, mu=1e-4)
    np.testing.assert_array_equal(x, supplied)
    assert np.any(solver.h-solver._mv(solver.g, x, solver.ng) < 0.)
    assert np.all(s > 0.) and np.all(z > 0.)
    # Original infeasibility is not hidden by returning an acceptance flag.
    assert 'accepted' not in _metadata


def test_old_state_is_never_read_and_all_returned_arrays_are_owned():
    solver = _solver()
    solver._last_internal_state = (None, np.full((2, 1), np.nan),
        np.full((2, solver.ng), np.inf), np.zeros((2, solver.ng)))
    supplied = _x()
    saved = (supplied.copy(), solver.c.copy(), solver.h.copy(), solver.g.copy())
    arrays, _metadata = centered_bound_restart(solver, supplied, mu=1e-4)
    for value in arrays:
        value[:] = 99.
    np.testing.assert_array_equal(supplied, saved[0])
    np.testing.assert_array_equal(solver.c, saved[1])
    np.testing.assert_array_equal(solver.h, saved[2])
    np.testing.assert_array_equal(solver.g.toarray(), saved[3].toarray())


@pytest.mark.parametrize('q,lower,upper', [(0, (0,), ()), (0, (), (2,)), (1, (), ())])
def test_empty_proposal_blocks_and_no_equality_rows(q, lower, upper):
    solver = _solver(q=q, lower=lower, upper=upper, ne=0)
    (_xcopy, y, z, s), _ = centered_bound_restart(solver, _x(), mu=1e-4)
    assert y.shape == (2, 0) and z.shape == s.shape == (2, solver.ng)
    assert np.all(np.isfinite(z)) and np.all(s > 0.)


def test_fixed_variables_are_not_projected_and_equality_multipliers_restart_at_zero():
    solver = _solver(lower=(1,), upper=(2,), ne=2)
    solver.fixed = np.array([0], dtype=np.int64)
    supplied = _x()
    (x, y, _z, _s), _ = centered_bound_restart(solver, supplied, mu=1e-4)
    np.testing.assert_array_equal(x, supplied)
    np.testing.assert_array_equal(y, np.zeros((2, 2)))


@pytest.mark.parametrize('mu', [0., 1e-9, 1e-1, -1., np.nan, np.inf, True, None,
                              np.array(1e-4), np.array([1e-4]), np.float64(1e-4)])
def test_invalid_or_nonexplicit_barrier_scalar_fails(mu):
    with pytest.raises(ValueError, match='scalar mu'):
        centered_bound_restart(_solver(), _x(), mu=mu)


@pytest.mark.parametrize('mu', [1e-8, 1e-2])
def test_closed_barrier_range_endpoints_are_supported(mu):
    arrays, metadata = centered_bound_restart(_solver(), _x(), mu=mu)
    assert metadata['mu'] == mu and all(np.isfinite(value).all() for value in arrays)


@pytest.mark.parametrize('field', ['x', 'c', 'h'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_current_data_fails_closed(field, bad):
    solver, supplied = _solver(), _x()
    (supplied if field == 'x' else getattr(solver, field))[0, 0] = bad
    with pytest.raises(ValueError, match='Finite current'):
        centered_bound_restart(solver, supplied, mu=1e-4)


@pytest.mark.parametrize('field', ['x', 'c', 'h'])
def test_fp32_current_vectors_are_not_silently_promoted(field):
    solver, supplied = _solver(), _x()
    if field == 'x': supplied = supplied.astype(np.float32)
    else: setattr(solver, field, getattr(solver, field).astype(np.float32))
    with pytest.raises(ValueError, match='FP64'):
        centered_bound_restart(solver, supplied, mu=1e-4)


@pytest.mark.parametrize('change', ['shape', 'index_type', 'out_of_range', 'duplicate', 'partition',
                                   'matrix_shape', 'matrix_type', 'dimension', 'backend', 'context'])
def test_invalid_current_solver_metadata_is_not_broadcast_or_repaired(change):
    solver, supplied = _solver(), _x()
    if change == 'shape': supplied = supplied[:, :2]
    elif change == 'index_type': solver.il = solver.il.astype(np.float64)
    elif change == 'out_of_range': solver.il[0] = solver.n
    elif change == 'duplicate': solver.il[1] = solver.il[0]
    elif change == 'partition': solver.q += 1
    elif change == 'matrix_shape': solver.g = solver.g[:-1]
    elif change == 'matrix_type': solver.g = solver.g.astype(np.float32)
    elif change == 'dimension': solver.n = True
    elif change == 'backend': solver.cp = SimpleNamespace(__name__='unknown')
    else:
        def rejected(): raise RuntimeError('Wrong stream or closed workspace')
        solver.factor._context = rejected
    with pytest.raises((ValueError, RuntimeError)):
        centered_bound_restart(solver, supplied, mu=1e-4)


@pytest.mark.parametrize('failure', ['activity', 'slack_subtraction', 'complementarity', 'ratio'])
def test_computed_overflow_fails_without_clipping_or_input_mutation(failure):
    solver, supplied = _solver(), _x()
    if failure == 'activity': solver._mv = lambda *_args: np.full((2, solver.ng), np.inf)
    elif failure == 'slack_subtraction':
        solver.h[:] = np.finfo(np.float64).max
        solver._mv = lambda *_args: np.full((2, solver.ng), -np.finfo(np.float64).max)
    elif failure == 'complementarity':
        solver.h[:] = 1e200
        solver.c[:, 0] = 1e200
    else: solver.h[:] = 1e200
    saved = supplied.copy()
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='representable'):
            centered_bound_restart(solver, supplied, mu=1e-4)
    np.testing.assert_array_equal(supplied, saved)


def test_cuda_current_restart_matches_numpy_and_does_not_download_vectors(monkeypatch):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1: pytest.skip('CUDA required')
    from cupyx.scipy.sparse import csr_matrix as device_csr
    reference = _solver()
    expected, _ = centered_bound_restart(reference, _x(), mu=1e-4)
    solver = _solver()
    solver.cp = cp
    for name in ('c', 'h', 'il', 'iu'): setattr(solver, name, cp.asarray(getattr(solver, name)))
    solver.g = device_csr(solver.g)
    solver.factor.device = cp.cuda.runtime.getDevice()
    solver.factor.stream = cp.cuda.get_current_stream()
    def reject_download(*_args, **_kwargs): raise AssertionError('No vector download allowed')
    monkeypatch.setattr(cp, 'asnumpy', reject_download)
    arrays, metadata = centered_bound_restart(solver, cp.asarray(_x()), mu=1e-4)
    assert metadata['backend'] == 'cupy'
    for value, wanted in zip(arrays, expected):
        np.testing.assert_allclose(value.get(), wanted, rtol=1e-14, atol=1e-15)


def test_cuda_restart_rejects_wrong_current_stream_before_arithmetic():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1: pytest.skip('CUDA required')
    from cupyx.scipy.sparse import csr_matrix as device_csr
    solver = _solver()
    solver.cp = cp
    for name in ('c', 'h', 'il', 'iu'): setattr(solver, name, cp.asarray(getattr(solver, name)))
    solver.g = device_csr(solver.g)
    solver.factor.device = cp.cuda.runtime.getDevice()
    solver.factor.stream = cp.cuda.get_current_stream()
    supplied = cp.asarray(_x())
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Stream(non_blocking=True):
        with pytest.raises(ValueError, match='bound current CUDA device/stream'):
            centered_bound_restart(solver, supplied, mu=1e-4)
