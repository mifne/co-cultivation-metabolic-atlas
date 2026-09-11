"""CPU-only strict matched-array and diagnostic ownership contracts."""
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.probe_ipm_rebind_matched_state import (
    MatchedStateMismatch, _own_solver, array_difference, array_fingerprint,
    assert_exact_array, solver_options,
)


def test_exact_initial_arrays_include_signed_zero_and_own_no_data():
    report = []
    value = np.array([[1., 0., 2.]])
    assert_exact_array(value, value.copy(), 'initial.x', report, require_fp64=True, require_finite=True)
    assert report[0]['exact_equal']
    original = report[0]['left']['sha256']
    value[:] = 9.
    assert report[0]['left']['sha256'] == original
    with pytest.raises(MatchedStateMismatch):
        assert_exact_array(np.array([[0.]]), np.array([[-0.]]), 'signed_zero', report)
    assert not report[-1]['exact_equal']


@pytest.mark.parametrize('left,right', [(np.ones((1, 2)), np.ones((2, 1))),
    (np.ones((1, 2)), np.ones((1, 2), dtype=np.float32)),
    (np.array([[1., 2.]]), np.array([[1., 3.]])),
    (np.array([[np.nan]]), np.array([[np.nan]])),
    (np.array([[np.inf]]), np.array([[np.inf]]))])
def test_initial_state_mismatch_and_nonfinite_values_fail_closed(left, right):
    report = []
    with pytest.raises(MatchedStateMismatch):
        assert_exact_array(left, right, 'initial', report, require_fp64=True, require_finite=True)
    assert len(report) == 1 and not report[0]['exact_equal']


def test_identical_infinite_bounds_are_legal_operator_payloads():
    report = []
    bounds = np.array([-np.inf, 3., np.inf])
    assert_exact_array(bounds, bounds.copy(), 'bounds', report)
    assert report[0]['exact_equal']


def test_array_fingerprint_preserves_shape_and_dtype_not_just_bytes():
    value = np.arange(4., dtype=np.float64)
    assert array_fingerprint(value) != array_fingerprint(value.reshape(2, 2))
    assert array_fingerprint(value) != array_fingerprint(value.view(np.uint64))
    assert array_fingerprint(value.reshape(2, 2).T)['sha256'] == array_fingerprint(
        np.ascontiguousarray(value.reshape(2, 2).T))['sha256']


def test_state_difference_reports_per_environment_without_becoming_a_gate():
    a, b = np.array([[1., 2.], [4., 5.]]), np.array([[1., 3.], [2., 8.]])
    result = array_difference(a, b)
    assert result['max_absolute_difference'] == 3.
    assert result['per_environment_max_absolute_difference'] == [1., 3.]
    assert not result['bitwise_equal'] and result['all_finite']
    assert 'accepted' not in result


def test_all_owned_solver_handles_close_on_later_exception():
    closed = []
    def factory(problems, **options):
        label = problems
        return SimpleNamespace(close=lambda: closed.append(label))
    record = dict(created_solver_labels=[], close_attempted_solver_labels=[])
    with pytest.raises(RuntimeError, match='deliberate'):
        with ExitStack() as stack:
            _own_solver(stack, factory, 'source', {}, record, 'source')
            _own_solver(stack, factory, 'fresh', {}, record, 'fresh')
            raise RuntimeError('deliberate failure after both handles exist')
    assert closed == ['fresh', 'source']
    assert record['close_attempted_solver_labels'] == ['fresh', 'source']


def test_close_failure_does_not_prevent_other_handle_cleanup():
    closed = []
    def factory(label, **_):
        def close():
            closed.append(label)
            if label == 'fresh':
                raise RuntimeError('deliberate cleanup failure')
        return SimpleNamespace(close=close)
    record = dict(created_solver_labels=[], close_attempted_solver_labels=[])
    with pytest.raises(RuntimeError, match='cleanup'):
        with ExitStack() as stack:
            _own_solver(stack, factory, 'source', {}, record, 'source')
            _own_solver(stack, factory, 'fresh', {}, record, 'fresh')
    assert closed == ['fresh', 'source']


def test_both_target_options_preserve_strict_algorithm_and_retained_gpu_state():
    options = solver_options(SimpleNamespace(second_forest=True, fix_singleton_equalities=True,
                                             reuse_gmres_workspace=True))
    assert options['retain_internal_state'] and options['globalized']
    assert options['exact_equalities'] and options['newton_krylov_iterations'] == 16
    assert options['predictor_affine_fraction'] == .995
    assert 'initial_x' not in options and 'initial_y' not in options
