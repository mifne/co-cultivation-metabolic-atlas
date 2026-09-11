"""GMRES-level invariants for independently selected fused microkernel modes.

The CUDA parameter IDs allow the GPU owner to select them explicitly. Merely
collecting this file or selecting ``not cuda`` never initializes a GPU.
"""

import numpy as np
import pytest

from src import gpu_krylov_microkernels as kernels
from src.gpu_newton_krylov import (
    GMRES_CONVERGED, GMRES_ITERATION_LIMIT, GMRES_NONFINITE, batched_gmres,
)


@pytest.fixture(params=['numpy', 'cuda'], ids=['cpu', 'cuda'])
def xp(request):
    if request.param == 'numpy':
        return np
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA device required')
    return cp


def _host(value, xp):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _error(rhs, residual, xp):
    return xp.max(xp.abs(residual), axis=1) / xp.maximum(1., xp.max(xp.abs(rhs), axis=1))


@pytest.mark.parametrize('mode', ['none', 'mgs', 'all'])
def test_modes_solve_original_system_and_call_only_selected_hooks(xp, mode, monkeypatch):
    calls = dict(mgs=0, givens=0)
    original_mgs, original_givens = kernels.mgs2_inplace, kernels.givens_backsolve_inplace

    def mgs(*args, **kwargs):
        calls['mgs'] += 1
        return original_mgs(*args, **kwargs)

    def givens(*args, **kwargs):
        calls['givens'] += 1
        return original_givens(*args, **kwargs)

    monkeypatch.setattr(kernels, 'mgs2_inplace', mgs)
    monkeypatch.setattr(kernels, 'givens_backsolve_inplace', givens)
    matrices = xp.asarray([[[0., 2., -1.], [1., -3., 0.], [2., 1., 4.]],
                           [[2., 1., 0.], [1., -3., 2.], [0., 2., 0.]]], dtype=xp.float64)
    expected = xp.asarray([[1., -2., .5], [-3., .25, 2.]], dtype=xp.float64)
    regularized = matrices + .2 * xp.eye(3)[None]
    matvec = lambda x: xp.einsum('bij,bj->bi', matrices, x)
    rhs = matvec(expected)
    answer, info = batched_gmres(rhs, xp.zeros_like(rhs), matvec,
        lambda x: xp.linalg.solve(regularized, x[..., None])[..., 0],
        lambda b, r: _error(b, r, xp), xp=xp, tolerance=1e-12, microkernels=mode)
    np.testing.assert_allclose(_host(answer, xp), _host(expected, xp), atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(_host(info['termination'], xp), GMRES_CONVERGED)
    np.testing.assert_allclose(_host(info['error'], xp), _host(_error(rhs, rhs - matvec(answer), xp), xp),
                               atol=2e-15, rtol=2e-13)
    assert np.max(np.abs(_host(xp.einsum('bij,bj->bi', regularized, answer) - rhs, xp))) > .1
    assert (calls['mgs'] > 0) == (mode != 'none')
    assert (calls['givens'] > 0) == (mode == 'all')


@pytest.mark.parametrize('mode', ['none', 'mgs', 'all'])
def test_true_weighted_residual_retains_better_previous_candidate(xp, mode):
    matrix = xp.asarray([[-4., -3., -4.], [0., -2., -2.], [-4., 4., 0.]], dtype=xp.float64)
    rhs = xp.asarray([[1., 4., 4.]], dtype=xp.float64)
    weights = xp.asarray([[100., 1., 1.]], dtype=xp.float64)
    seen = []

    def measured(_, residual):
        error = xp.max(xp.abs(residual) * weights, axis=1)
        seen.append(error.copy())
        return error

    answer, info = batched_gmres(rhs, xp.zeros_like(rhs), lambda x: x @ matrix.T,
        lambda x: x.copy(), measured, xp=xp, max_iterations=2, tolerance=1e-14,
        microkernels=mode)
    recorded = _host(xp.stack(seen), xp)
    assert recorded.shape == (3, 1)  # An explicit residual for both proposals.
    assert recorded[1, 0] < recorded[0, 0] < recorded[2, 0]
    np.testing.assert_array_equal(_host(info['termination'], xp), GMRES_ITERATION_LIMIT)
    np.testing.assert_allclose(_host(info['error'], xp), recorded.min(axis=0))
    actual_error = xp.max(xp.abs(rhs - answer @ matrix.T) * weights, axis=1)
    np.testing.assert_allclose(_host(info['error'], xp), _host(actual_error, xp), atol=1e-12)


@pytest.mark.parametrize('mode', ['none', 'mgs', 'all'])
@pytest.mark.parametrize('failure', ['preconditioner', 'matvec', 'measurement'])
def test_bad_lane_is_rejected_without_poisoning_good_or_inactive_lane(xp, mode, failure):
    rhs = xp.asarray([[1., -2.], [3., 4.], [0., 0.]], dtype=xp.float64)
    initial = xp.zeros_like(rhs)
    counts = dict(matvec=0, measurement=0)
    precondition_inputs = []

    def matvec(x):
        counts['matvec'] += 1
        value = x.copy()
        if failure == 'matvec' and counts['matvec'] == 2:
            value[0] = xp.nan
        return value

    def precondition(x):
        precondition_inputs.append(x.copy())
        value = x.copy()
        if failure == 'preconditioner':
            value[0] = xp.nan
        return value

    def measured(b, residual):
        counts['measurement'] += 1
        value = _error(b, residual, xp)
        if failure == 'measurement' and counts['measurement'] == 2:
            value[0] = xp.inf
        return value

    answer, info = batched_gmres(rhs, initial, matvec, precondition, measured,
                                xp=xp, tolerance=1e-12, microkernels=mode)
    host_answer = _host(answer, xp)
    np.testing.assert_array_equal(host_answer[0], 0.)
    np.testing.assert_allclose(host_answer[1], [3., 4.], atol=1e-12)
    np.testing.assert_array_equal(host_answer[2], 0.)
    np.testing.assert_array_equal(_host(info['termination'], xp),
                                 [GMRES_NONFINITE, GMRES_CONVERGED, GMRES_CONVERGED])
    np.testing.assert_array_equal(_host(info['iterations'], xp), [1, 1, 0])
    np.testing.assert_array_equal(_host(info['nonfinite'], xp), [True, False, False])
    np.testing.assert_allclose(_host(info['error'], xp),
                               _host(_error(rhs, rhs - answer, xp), xp), atol=1e-12)
    assert all(np.array_equal(_host(value[2], xp), [0., 0.]) for value in precondition_inputs)


@pytest.mark.parametrize('mode', ['none', 'mgs', 'all'])
@pytest.mark.parametrize('already_converged', [False, True])
def test_zero_work_does_not_call_microkernels(xp, mode, already_converged, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('No Arnoldi or microkernel work is required')

    monkeypatch.setattr(kernels, 'mgs2_inplace', forbidden)
    monkeypatch.setattr(kernels, 'givens_backsolve_inplace', forbidden)
    rhs = xp.asarray([[1., -2.]], dtype=xp.float64)
    initial = rhs.copy() if already_converged else xp.zeros_like(rhs)
    answer, info = batched_gmres(rhs, initial, lambda x: x.copy(), forbidden,
        lambda b, r: _error(b, r, xp), xp=xp, max_iterations=4 if already_converged else 0,
        microkernels=mode)
    np.testing.assert_array_equal(_host(answer, xp), _host(initial, xp))
    np.testing.assert_array_equal(_host(info['iterations'], xp), 0)
    np.testing.assert_array_equal(_host(info['termination'], xp),
                                 GMRES_CONVERGED if already_converged else GMRES_ITERATION_LIMIT)


def test_finite_but_bad_fused_coefficients_cannot_bypass_true_residual(xp, monkeypatch):
    original = kernels.givens_backsolve_inplace

    def incorrect_coefficients(*args, **kwargs):
        coefficients, diagonal = original(*args, **kwargs)
        coefficients[...] = 0.
        return coefficients, diagonal

    monkeypatch.setattr(kernels, 'givens_backsolve_inplace', incorrect_coefficients)
    rhs = xp.asarray([[1., -2.]], dtype=xp.float64)
    initial = xp.zeros_like(rhs)
    answer, info = batched_gmres(rhs, initial, lambda x: x.copy(), lambda x: x.copy(),
        lambda b, r: _error(b, r, xp), xp=xp, microkernels='all')
    np.testing.assert_array_equal(_host(answer, xp), 0.)
    assert not _host(info['converged'], xp).any()
    np.testing.assert_array_equal(_host(info['error'], xp), 1.)


@pytest.mark.parametrize('invalid', ['automatic', 'givens', True, None])
def test_mode_requires_an_explicit_supported_choice(invalid):
    rhs = np.ones((1, 2))
    with pytest.raises(ValueError, match='microkernels'):
        batched_gmres(rhs, np.zeros_like(rhs), lambda x: x.copy(), lambda x: x.copy(),
                     lambda b, r: _error(b, r, np), xp=np, microkernels=invalid)
