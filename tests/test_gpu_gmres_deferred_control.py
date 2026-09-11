"""Deferred GPU lane checks keep original-system acceptance and rejection.

CPU tests never query CUDA. GPU-owner tests are selected explicitly by ``cuda``.
"""

import numpy as np
import pytest

from src.gpu_newton_krylov import (
    GMRES_BREAKDOWN, GMRES_CONVERGED, GMRES_NONFINITE, batched_gmres,
)


def _error(xp):
    def relative(rhs, residual):
        scale = xp.max(xp.abs(rhs), axis=1)
        return xp.max(xp.abs(residual), axis=1) / xp.where(scale > 0., scale, 1.)
    return relative


def _host(value, xp):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _compare(matrices, rhs, *, xp, microkernels, initial=None, precondition=None,
             tolerance=1e-12, max_iterations=12):
    matrices, rhs = xp.asarray(matrices), xp.asarray(rhs)
    initial = xp.zeros_like(rhs) if initial is None else xp.asarray(initial)
    matvec = lambda x: xp.einsum('bij,bj->bi', matrices, x)
    precondition = (lambda x: x.copy()) if precondition is None else precondition
    results = []
    for deferred in (False, True):
        result = batched_gmres(rhs, initial, matvec, precondition, _error(xp), xp=xp,
            tolerance=tolerance, max_iterations=max_iterations,
            microkernels=microkernels, defer_lane_checks=deferred)
        results.append(result)
    # Only redundant masked work differs; live-lane arithmetic is identical.
    np.testing.assert_array_equal(_host(results[0][0], xp), _host(results[1][0], xp))
    for key in results[0][1]:
        np.testing.assert_array_equal(_host(results[0][1][key], xp),
                                      _host(results[1][1][key], xp))
    answer, info = results[1]
    independently_measured = _error(xp)(rhs, rhs - matvec(answer))
    np.testing.assert_array_equal(_host(info['error'], xp), _host(independently_measured, xp))
    return _host(answer, xp), {key: _host(value, xp) for key, value in info.items()}


@pytest.mark.parametrize('microkernels', ['none', 'mgs', 'all'])
def test_numpy_mixed_convergence_times_are_unchanged(microkernels):
    matrices = np.stack([np.eye(4), np.diag([1., 2., 3., 4.]),
                         np.diag([1., 1., 2., 2.]), np.eye(4)])
    rhs = np.array([[1., 0., 0., 0.], [1., -2., 3., -4.],
                    [1., -2., 3., -4.], [0., 0., 0., 0.]])
    answer, info = _compare(matrices, rhs, xp=np, microkernels=microkernels)
    np.testing.assert_array_equal(info['iterations'], [1, 4, 2, 0])
    np.testing.assert_array_equal(info['termination'], GMRES_CONVERGED)
    np.testing.assert_allclose(np.einsum('bij,bj->bi', matrices, answer), rhs, atol=1e-12)


@pytest.mark.parametrize('microkernels', ['none', 'all'])
@pytest.mark.parametrize('scale', [1e-250, 1e-150, 1., 1e150, 1e250])
def test_numpy_scaled_rhs_keep_true_relative_error(microkernels, scale):
    matrices = np.stack([np.diag([1., 2., 4.]), np.eye(3)])
    rhs = scale * np.array([[1., -2., 4.], [2., 0., -1.]])
    _, info = _compare(matrices, rhs, xp=np, microkernels=microkernels)
    np.testing.assert_array_equal(info['termination'], GMRES_CONVERGED)
    assert np.max(info['error']) < 1e-12


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_numpy_all_preconditioner_lanes_invalid_stay_rejected(microkernels):
    matrices = np.stack([np.eye(3), np.eye(3)])
    rhs = np.array([[1., 2., 3.], [-1., 0., 2.]])
    answer, info = _compare(matrices, rhs, xp=np, microkernels=microkernels,
                            precondition=lambda x: np.full_like(x, np.nan))
    np.testing.assert_array_equal(answer, 0.)
    np.testing.assert_array_equal(info['iterations'], 1)
    np.testing.assert_array_equal(info['termination'], GMRES_NONFINITE)


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_numpy_all_singular_lanes_stay_rejected(microkernels):
    matrices = np.zeros((2, 3, 3))
    rhs = np.array([[1., 2., 3.], [-1., 0., 2.]])
    initial = np.array([[4., 5., 6.], [2., 3., 4.]])
    answer, info = _compare(matrices, rhs, initial=initial, xp=np, microkernels=microkernels)
    np.testing.assert_array_equal(answer, initial)
    np.testing.assert_array_equal(info['iterations'], 1)
    np.testing.assert_array_equal(info['termination'], GMRES_BREAKDOWN)


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_numpy_failed_lane_does_not_poison_live_lane(microkernels):
    matrices = np.stack([np.eye(3), np.diag([1., 2., 4.])])
    rhs = np.array([[1., 2., 3.], [1., 2., 4.]])
    def precondition(x):
        output = x.copy()
        output[0] = np.nan
        return output
    answer, info = _compare(matrices, rhs, xp=np, microkernels=microkernels,
                            precondition=precondition)
    np.testing.assert_array_equal(answer[0], 0.)
    np.testing.assert_allclose(answer[1], 1.)
    np.testing.assert_array_equal(info['iterations'], [1, 3])
    np.testing.assert_array_equal(info['termination'], [GMRES_NONFINITE, GMRES_CONVERGED])


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_numpy_nonfinite_candidate_cannot_be_accepted(microkernels):
    # Finite directions and projected RHS can still overflow when multiplied.
    rhs = np.array([[1e308]])
    output = []
    for deferred in (False, True):
        with np.errstate(over='ignore', invalid='ignore'):
            answer, info = batched_gmres(rhs, np.zeros_like(rhs), lambda x: x * 1e-308,
                lambda x: np.full_like(x, 1e308), _error(np), xp=np,
                microkernels=microkernels, defer_lane_checks=deferred)
        np.testing.assert_array_equal(answer, 0.)
        np.testing.assert_array_equal(info['termination'], GMRES_NONFINITE)
        np.testing.assert_array_equal(info['iterations'], 1)
        output.append(info)
    for key in output[0]:
        np.testing.assert_array_equal(output[0][key], output[1][key])


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_deferred_preserves_previous_best_when_true_error_increases(microkernels):
    matrix = np.array([[-4., -3., -4.], [0., -2., -2.], [-4., 4., 0.]])
    rhs = np.array([[1., 4., 4.]])
    result = []
    for deferred in (False, True):
        seen = []
        def measurement(b, residual):
            error = np.max(np.abs(residual) * np.array([[100., 1., 1.]]), axis=1)
            seen.append(error.copy())
            return error
        answer, info = batched_gmres(rhs, np.zeros_like(rhs), lambda x: x @ matrix.T,
            lambda x: x.copy(), measurement, xp=np, max_iterations=2, tolerance=1e-14,
            microkernels=microkernels, defer_lane_checks=deferred)
        assert seen[1][0] < seen[0][0] < seen[2][0]
        np.testing.assert_array_equal(info['error'], np.min(np.stack(seen), axis=0))
        np.testing.assert_array_equal(info['error'], measurement(rhs, rhs - answer @ matrix.T))
        result.append((answer, info))
    np.testing.assert_array_equal(result[0][0], result[1][0])
    for key in result[0][1]:
        np.testing.assert_array_equal(result[0][1][key], result[1][1][key])


def _all_nonfinite_matvec(xp, microkernels):
    rhs = xp.ones((2, 3), dtype=xp.float64)
    def matvec(x):
        # Zero initial and inactive inputs remain safe, all live Arnoldi inputs fail.
        return xp.where(xp.any(x != 0., axis=1)[:, None], xp.nan, 0.) * xp.ones_like(x)
    for deferred in (False, True):
        answer, info = batched_gmres(rhs, xp.zeros_like(rhs), matvec,
            lambda x: x.copy(), _error(xp), xp=xp, microkernels=microkernels,
            defer_lane_checks=deferred)
        np.testing.assert_array_equal(_host(answer, xp), 0.)
        np.testing.assert_array_equal(_host(info['iterations'], xp), 1)
        np.testing.assert_array_equal(_host(info['termination'], xp), GMRES_NONFINITE)
        np.testing.assert_array_equal(_host(info['error'], xp), 1.)


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_numpy_all_arnoldi_matvec_lanes_nonfinite(microkernels):
    _all_nonfinite_matvec(np, microkernels)


@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.parametrize('negative_at_initial', [False, True])
def test_negative_error_validation_is_not_deferred_or_removed(deferred, negative_at_initial):
    # A pure measurement which is negative at the requested residual value.
    def measurement(rhs, residual):
        zero = np.max(np.abs(residual), axis=1) == 0.
        return np.where(zero != negative_at_initial, -1., 1.)
    with pytest.raises(ValueError, match='nonnegative'):
        batched_gmres(np.array([[1., 0.]]), np.zeros((1, 2)), lambda x: x.copy(),
            lambda x: x.copy(), measurement, xp=np, defer_lane_checks=deferred)


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_only_zero_inputs_are_sent_after_all_lanes_fail(microkernels):
    seen = []
    preconditions = []
    def matvec(x):
        seen.append(x.copy())
        return x.copy()
    def precondition(x):
        preconditions.append(x.copy())
        return np.full_like(x, np.nan)
    answer, info = batched_gmres(np.ones((2, 3)), np.zeros((2, 3)), matvec,
        precondition, _error(np), xp=np, microkernels=microkernels, defer_lane_checks=True)
    assert len(seen) == 3  # Initial residual, masked Arnoldi, masked true residual.
    assert len(preconditions) == 1
    assert all(not value.any() for value in seen)
    np.testing.assert_array_equal(answer, 0.)
    assert info['nonfinite'].all()


def test_deferred_control_eliminates_three_host_active_queries_per_live_iteration():
    class CountingNumpy:
        def __init__(self): self.queries = 0
        def __getattr__(self, name): return getattr(np, name)
        def any(self, value, **kwargs):
            self.queries += 1
            return np.any(value, **kwargs)
    counts = []
    for deferred in (False, True):
        xp = CountingNumpy()
        _, info = batched_gmres(np.array([[1., 2., 3.]]), np.zeros((1, 3)),
            lambda x: x * np.array([[1., 2., 4.]]), lambda x: x.copy(),
            _error(np), xp=xp, defer_lane_checks=deferred, max_iterations=3)
        assert info['iterations'][0] == 3
        counts.append(xp.queries)
    assert counts[0] - counts[1] == 9


@pytest.mark.parametrize('invalid', [None, 0, 1, 'yes', np.bool_(True)])
def test_deferred_option_requires_explicit_python_bool(invalid):
    with pytest.raises(ValueError, match='explicit bool'):
        batched_gmres(np.ones((1, 2)), np.zeros((1, 2)), lambda x: x,
            lambda x: x, _error(np), xp=np, defer_lane_checks=invalid)


@pytest.mark.parametrize('microkernels', ['none', 'all'])
@pytest.mark.parametrize('case', ['mixed', 'scaled', 'all_invalid', 'all_singular', 'one_invalid'])
def test_cuda_deferred_matches_existing_lane_control(microkernels, case):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA device required')
    matrices = np.stack([np.eye(3), np.diag([1., 2., 4.]), np.eye(3)])
    rhs = np.array([[1., 2., 3.], [1., 2., 4.], [0., 0., 0.]])
    precondition = None
    if case == 'scaled':
        rhs[0] *= 1e-250
        rhs[1] *= 1e250
    elif case == 'all_invalid':
        precondition = lambda x: cp.full_like(x, cp.nan)
    elif case == 'all_singular':
        matrices[:] = 0.
    elif case == 'one_invalid':
        def precondition(x):
            output = x.copy()
            output[0] = cp.nan
            return output
    _, info = _compare(matrices, rhs, xp=cp, microkernels=microkernels,
                       precondition=precondition)
    assert info['converged'][2]
    if case in ('mixed', 'scaled'):
        assert info['converged'].all()
    elif case == 'all_invalid':
        np.testing.assert_array_equal(info['nonfinite'], [True, True, False])
    elif case == 'all_singular':
        np.testing.assert_array_equal(info['breakdown'], [True, True, False])
    else:
        np.testing.assert_array_equal(info['nonfinite'], [True, False, False])


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_cuda_all_arnoldi_matvec_lanes_nonfinite(microkernels):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA device required')
    _all_nonfinite_matvec(cp, microkernels)
