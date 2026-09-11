"""Reusable GMRES storage ownership and unchanged original-residual decisions."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.gpu_newton_krylov import GmresWorkspace, batched_gmres


def _error(rhs, residual):
    return np.max(np.abs(residual), axis=1) / np.maximum(1., np.max(np.abs(rhs), axis=1))


def _solve(rhs, *, workspace=None, max_iterations=4, initial=None,
           matvec=None, precondition=None, error_measure=None, **kwargs):
    return batched_gmres(rhs, np.zeros_like(rhs) if initial is None else initial,
        (lambda x: x*np.arange(1., rhs.shape[1]+1.)) if matvec is None else matvec,
        (lambda x: x.copy()) if precondition is None else precondition,
        _error if error_measure is None else error_measure, xp=np,
        max_iterations=max_iterations, tolerance=1e-13, workspace=workspace, **kwargs)


@pytest.mark.parametrize('microkernels', ['none', 'mgs', 'all'])
@pytest.mark.parametrize('defer_lane_checks', [False, True])
def test_cpu_workspace_reuses_real_arrays_and_matches_fresh_results(microkernels, defer_lane_checks):
    ws = GmresWorkspace((2, 4), xp=np, max_iterations=4)
    arrays = {name: (id(value), value.ctypes.data) for name, value in ws.buffers.items()}
    expected_bytes = 8*(2*5*4+2*4*4+2*5*4+2*4+2*4+2*5)
    assert ws.nbytes == expected_bytes
    for rhs in (np.array([[1., -3., 2., 4.], [0., 0., 0., 0.]]),
                np.array([[2., 5., -3., 6.], [-4., 2., -3., 7.]])):
        expected, expected_info = _solve(rhs, microkernels=microkernels,
                                        defer_lane_checks=defer_lane_checks)
        actual, actual_info = _solve(rhs, workspace=ws, microkernels=microkernels,
                                    defer_lane_checks=defer_lane_checks)
        np.testing.assert_array_equal(actual, expected)
        for name in expected_info:
            np.testing.assert_array_equal(actual_info[name], expected_info[name])
        assert arrays == {name: (id(v), v.ctypes.data) for name, v in ws.buffers.items()}
    assert ws.use_count == 2 and ws.budget == 4 and ws.max_iterations == 4


def test_cpu_every_workspace_region_is_reset_after_poisoned_previous_call():
    ws = GmresWorkspace((2, 4), xp=np, max_iterations=4)
    _solve(np.ones((2, 4)), workspace=ws)
    for array in ws.buffers.values():
        array.fill(np.nan)

    def forbidden(_):
        raise AssertionError('Converged zero lanes must not reach a preconditioner')

    answer, info = _solve(np.zeros((2, 4)), workspace=ws, precondition=forbidden)
    np.testing.assert_array_equal(answer, 0.)
    assert info['converged'].all() and not info['nonfinite'].any()
    for array in ws.buffers.values():
        np.testing.assert_array_equal(array, 0.)


def test_cpu_returned_answer_and_every_diagnostic_survive_later_reuse():
    ws = GmresWorkspace((2, 4), xp=np, max_iterations=4)
    rhs = np.arange(1., 9.).reshape(2, 4)
    answer, info = _solve(rhs, workspace=ws)
    saved_answer, saved_info = answer.copy(), {name: value.copy() for name, value in info.items()}
    for result in (answer, *info.values()):
        assert all(not np.may_share_memory(result, array) for array in ws.buffers.values())
    other, other_info = _solve(-rhs, workspace=ws)
    other[:] = 99.
    for value in other_info.values():
        value[:] = 0
    for array in ws.buffers.values():
        array.fill(44.)
    np.testing.assert_array_equal(answer, saved_answer)
    for name in info:
        np.testing.assert_array_equal(info[name], saved_info[name])


@pytest.mark.parametrize('shape,budget', [((0, 2), 2), ((1, 0), 2), ((True, 2), 2),
    ((1, 2, 3), 2), ((1, 2), -1), ((1, 2), 129), ((1, 2), True)])
def test_cpu_invalid_workspace_geometry_is_rejected(shape, budget):
    with pytest.raises(ValueError):
        GmresWorkspace(shape, xp=np, max_iterations=budget)


def test_cpu_requested_budget_must_match_even_when_capped_width_is_equal():
    ws = GmresWorkspace((1, 4), xp=np, max_iterations=12)
    assert ws.budget == 4
    with pytest.raises(ValueError, match='budget'):
        _solve(np.ones((1, 4)), workspace=ws, max_iterations=4)
    _solve(np.ones((1, 4)), workspace=ws, max_iterations=12)


@pytest.mark.parametrize('rhs,initial', [
    (np.ones((1, 4)), np.zeros((1, 4))),
    (np.ones((2, 3)), np.zeros((2, 3))),
    (np.ones((2, 4), dtype=np.float32), np.zeros((2, 4))),
    (np.ones((2, 4)), np.zeros((2, 4), dtype=np.float32))])
def test_cpu_workspace_requires_exact_input_shape_and_dtype(rhs, initial):
    ws = GmresWorkspace((2, 4), xp=np, max_iterations=4)
    with pytest.raises(ValueError, match='matching FP64 shape'):
        _solve(rhs, initial=initial, workspace=ws)


def test_cpu_workspace_rejects_other_namespace_and_invalid_workspace_object():
    ws = GmresWorkspace((1, 4), xp=np, max_iterations=4)
    with pytest.raises(ValueError, match='namespace'):
        batched_gmres(np.ones((1, 4)), np.zeros((1, 4)), lambda x:x, lambda x:x,
            _error, xp=SimpleNamespace(__name__='numpy'), max_iterations=4, workspace=ws)
    with pytest.raises(ValueError, match='GmresWorkspace'):
        _solve(np.ones((1, 4)), workspace=object())


@pytest.mark.parametrize('aliased', ['rhs', 'initial'])
def test_cpu_inputs_must_not_alias_workspace_regions(aliased):
    ws = GmresWorkspace((2, 4), xp=np, max_iterations=4)
    view = ws.buffers['basis'][:, 0, :]
    view[:] = 1.
    rhs, initial = np.ones((2, 4)), np.zeros((2, 4))
    if aliased == 'rhs':
        rhs = view
    else:
        initial = view
    with pytest.raises(ValueError, match='alias'):
        _solve(rhs, initial=initial, workspace=ws)


def test_cpu_buffer_mapping_is_readonly_and_corruption_fails_before_reset():
    ws = GmresWorkspace((2, 4), xp=np, max_iterations=4)
    with pytest.raises(TypeError):
        ws.buffers['basis'] = np.zeros((2, 5, 4))
    original = ws._buffers['basis']
    ws._buffers['basis'] = np.zeros_like(original)
    with pytest.raises(ValueError, match='metadata or address'):
        _solve(np.ones((2, 4)), workspace=ws)
    ws._buffers['basis'] = original
    _solve(np.ones((2, 4)), workspace=ws)


def test_cpu_owned_buffer_must_remain_writeable_and_keep_shape():
    ws = GmresWorkspace((1, 4), xp=np, max_iterations=4)
    array = ws.buffers['basis']
    array.flags.writeable = False
    with pytest.raises(ValueError, match='metadata or address'):
        _solve(np.ones((1, 4)), workspace=ws)
    array.flags.writeable = True
    original_shape = array.shape
    array.shape = (5, 4)
    with pytest.raises(ValueError, match='metadata or address'):
        _solve(np.ones((1, 4)), workspace=ws)
    array.shape = original_shape
    _solve(np.ones((1, 4)), workspace=ws)


@pytest.mark.parametrize('which', ['matvec', 'precondition', 'error_measure'])
def test_cpu_callback_exceptions_release_workspace(which):
    ws = GmresWorkspace((1, 4), xp=np, max_iterations=4)

    def failure(*_):
        raise ArithmeticError('manufactured callback failure')

    with pytest.raises(ArithmeticError, match='callback failure'):
        _solve(np.ones((1, 4)), workspace=ws, **{which:failure})
    result, info = _solve(np.ones((1, 4)), workspace=ws)
    assert info['converged'].all()
    np.testing.assert_allclose(result, 1./np.arange(1., 5.)[None], atol=1e-13)


def test_cpu_reentrant_workspace_use_is_rejected_without_poisoning_outer_call():
    ws = GmresWorkspace((1, 4), xp=np, max_iterations=4)
    nested_attempts = []

    def precondition(x):
        with pytest.raises(RuntimeError, match='concurrent/reentrant'):
            _solve(np.ones((1, 4)), workspace=ws)
        nested_attempts.append(True)
        return x.copy()

    _, info = _solve(np.ones((1, 4)), workspace=ws, precondition=precondition)
    assert info['converged'].all() and nested_attempts and ws.use_count == 1


def test_cpu_concurrent_workspace_use_fails_fast_and_releases_afterward():
    ws = GmresWorkspace((1, 4), xp=np, max_iterations=4)
    entered, release = threading.Event(), threading.Event()

    def precondition(x):
        entered.set()
        if not release.wait(3.):
            raise AssertionError('Test synchronization deadline expired')
        return x.copy()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_solve, np.ones((1, 4)), workspace=ws, precondition=precondition)
        try:
            assert entered.wait(3.)
            with pytest.raises(RuntimeError, match='concurrent/reentrant'):
                _solve(np.ones((1, 4)), workspace=ws)
        finally:
            release.set()
        _, info = future.result(timeout=3.)
    assert info['converged'].all()
    _solve(np.ones((1, 4)), workspace=ws)


def test_cpu_nonfinite_validation_and_lane_termination_are_unchanged():
    ws = GmresWorkspace((2, 4), xp=np, max_iterations=4)
    rhs = np.ones((2, 4))
    with pytest.raises(ValueError, match='Finite right-hand'):
        _solve(rhs, initial=np.full_like(rhs, np.nan), workspace=ws)

    def invalid_lane(x):
        value = x.copy()
        value[0] = np.nan
        return value

    expected, expected_info = _solve(rhs, precondition=invalid_lane)
    actual, actual_info = _solve(rhs, precondition=invalid_lane, workspace=ws)
    np.testing.assert_array_equal(actual, expected)
    for name in expected_info:
        np.testing.assert_array_equal(actual_info[name], expected_info[name])


def test_cpu_zero_budget_workspace_preserves_initial_diagnostics():
    ws = GmresWorkspace((1, 4), xp=np, max_iterations=0)
    expected, expected_info = _solve(np.ones((1, 4)), max_iterations=0)
    actual, actual_info = _solve(np.ones((1, 4)), max_iterations=0, workspace=ws)
    np.testing.assert_array_equal(actual, expected)
    for name in expected_info:
        np.testing.assert_array_equal(actual_info[name], expected_info[name])


@pytest.mark.parametrize('microkernels', ['none', 'all'])
def test_cuda_workspace_reuse_matches_fresh_and_rejects_wrong_stream(microkernels, monkeypatch):
    import cupy as cp
    ws = GmresWorkspace((2, 4), xp=cp, max_iterations=4)
    addresses = {name:v.data.ptr for name, v in ws.buffers.items()}
    scale = cp.arange(1., 5., dtype=cp.float64)[None]

    def error(rhs, residual):
        return cp.max(cp.abs(residual), axis=1)/cp.maximum(1., cp.max(cp.abs(rhs), axis=1))

    def run(rhs, workspace=None):
        return batched_gmres(rhs, cp.zeros_like(rhs), lambda x:x*scale, lambda x:x.copy(),
            error, xp=cp, max_iterations=4, tolerance=1e-12,
            microkernels=microkernels, defer_lane_checks=True, workspace=workspace)

    rhs = cp.arange(1., 9., dtype=cp.float64).reshape(2, 4)
    expected, expected_info = run(rhs)
    actual, actual_info = run(rhs, ws)
    saved = actual.get()
    np.testing.assert_allclose(saved, expected.get(), atol=1e-12, rtol=0.)
    for name in expected_info:
        np.testing.assert_allclose(actual_info[name].get(), expected_info[name].get(), atol=1e-12, rtol=0.)
    with cp.cuda.Stream(non_blocking=True):
        with pytest.raises(ValueError, match='stream'):
            run(rhs, ws)
    initial = cp.zeros_like(rhs)
    with monkeypatch.context() as patch:
        # Exercise the current-device metadata guard without allocating on a
        # second physical GPU, which need not exist on this workstation.
        patch.setattr(cp.cuda.runtime, 'getDevice', lambda: ws._device + 1)
        with pytest.raises(ValueError, match='device'):
            ws._validate(rhs, initial, xp=cp, max_iterations=4)
    for value in ws.buffers.values():
        value.fill(cp.nan)
    other, info = run(-rhs, ws)
    assert bool(info['converged'].all())
    np.testing.assert_allclose(other.get(), -expected.get(), atol=1e-12, rtol=0.)
    np.testing.assert_array_equal(actual.get(), saved)
    assert addresses == {name:v.data.ptr for name, v in ws.buffers.items()}
    for value in (actual, *actual_info.values()):
        assert all(not cp.may_share_memory(value, buffer) for buffer in ws.buffers.values())
