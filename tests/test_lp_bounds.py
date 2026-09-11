"""Bounds acceleration preserves the two existing host-adapter contracts."""

from decimal import Decimal

import numpy as np
import pytest

from src.cpu_repeated_lp import _problem
from src.gpu_compiled_community_backend import lp_arrays
from src.lp_bounds import split_variable_bounds


def _cpu(bounds, n=3):
    return _problem(np.ones(n), {'bounds': bounds})[2:4]


def _gpu(bounds, n=3):
    return lp_arrays(np.ones(n), bounds=bounds)[2:4]


@pytest.mark.parametrize('bounds, expected_lower, expected_upper', [
    (None, [0., 0., 0.], [np.inf, np.inf, np.inf]),
    ((0., None), [0., 0., 0.], [np.inf, np.inf, np.inf]),
    (np.array([-2., 4.]), [-2., -2., -2.], [4., 4., 4.]),
    (np.array([[0., np.inf], [-np.inf, 2.], [-1., 3.]]),
     [0., -np.inf, -1.], [np.inf, 2., 3.]),
    (np.array([[Decimal('0.25'), None], [None, Decimal('2.5')],
               [Decimal('-1.5'), Decimal('3.5')]], dtype=object),
     [.25, -np.inf, -1.5], [np.inf, 2.5, 3.5]),
])
def test_checked_cpu_bounds_keep_default_broadcast_numeric_and_object_semantics(
        bounds, expected_lower, expected_upper):
    lower, upper = _cpu(bounds)
    np.testing.assert_array_equal(lower, expected_lower)
    np.testing.assert_array_equal(upper, expected_upper)
    assert lower.dtype == upper.dtype == np.float64


@pytest.mark.parametrize('bounds, message', [
    (np.zeros((3, 1)), 'Bounds must be a pair'),
    (np.zeros((2, 2)), 'Bounds must be a pair'),
    (np.zeros((1, 2, 3)), 'Bounds must be a pair'),
    (np.array([[0., 1.], [np.nan, 2.], [-1., 3.]]), 'Invalid variable bounds'),
    (np.array([[2., 1.], [0., 2.], [-1., 3.]]), 'Invalid variable bounds'),
])
def test_checked_cpu_bounds_keep_shape_nan_and_order_rejections(bounds, message):
    with pytest.raises(ValueError, match=message):
        _cpu(bounds)


def test_checked_cpu_bounds_keep_infinite_sides_and_do_not_alias_input():
    bounds = np.array([[-np.inf, np.inf], [np.inf, np.inf],
                       [-np.inf, -np.inf]])
    snapshot = bounds.copy()
    lower, upper = _cpu(bounds)
    lower[:] = 7.; upper[:] = 8.
    np.testing.assert_array_equal(bounds, snapshot)


def test_unchecked_gpu_bounds_keep_default_dtype_and_no_pair_broadcast():
    lower, upper = _gpu(None)
    np.testing.assert_array_equal(lower, [0, 0, 0])
    np.testing.assert_array_equal(upper, [np.inf, np.inf, np.inf])
    assert lower.dtype == np.dtype(np.int_)
    assert upper.dtype == np.float64
    for pair in ((0., None), np.array([0., 1.])):
        with pytest.raises(TypeError):
            _gpu(pair)


def test_unchecked_gpu_bounds_keep_object_none_nan_inf_and_reverse_values():
    bounds = np.array([[0., None], [np.nan, 2.], [-np.inf, -2.]], dtype=object)
    lower, upper = _gpu(bounds)
    np.testing.assert_array_equal(lower, [0., np.nan, -np.inf])
    np.testing.assert_array_equal(upper, [np.inf, 2., -2.])
    reverse = np.array([[2., 1.], [0., 2.], [-1., 3.]])
    lo, hi = _gpu(reverse)
    np.testing.assert_array_equal(lo, reverse[:, 0])
    np.testing.assert_array_equal(hi, reverse[:, 1])


def test_unchecked_gpu_bounds_keep_permissive_row_count_and_bad_column_errors():
    lower, upper = _gpu(np.array([[0., 1.], [1., 2.]]))
    np.testing.assert_array_equal(lower, [0., 1.])
    np.testing.assert_array_equal(upper, [1., 2.])
    with pytest.raises(ValueError, match='not enough values to unpack'):
        _gpu(np.zeros((3, 1)))
    with pytest.raises(ValueError, match='too many values to unpack'):
        _gpu(np.zeros((3, 3)))


@pytest.mark.parametrize('dtype', [np.bool_, np.int32, np.int64,
                                    np.float32, np.float64])
def test_numeric_helper_matches_legacy_values_and_owns_columns(dtype):
    bounds = np.array([[0, 1], [-2, 3], [4, 5]], dtype=dtype)
    expected_cpu = (
        np.array([value for value in np.asarray(bounds, dtype=object)[:, 0]], dtype=float),
        np.array([value for value in np.asarray(bounds, dtype=object)[:, 1]], dtype=float),
    )
    expected_gpu = (
        np.array([lo for lo, hi in bounds]),
        np.array([hi for lo, hi in bounds]),
    )
    for checked, expected in ((True, expected_cpu), (False, expected_gpu)):
        lower, upper = split_variable_bounds(bounds, 3, checked=checked)
        np.testing.assert_array_equal(lower, expected[0])
        np.testing.assert_array_equal(upper, expected[1])
        lower[:] = 9; upper[:] = 10
        np.testing.assert_array_equal(bounds, np.array([[0, 1], [-2, 3], [4, 5]], dtype=dtype))


def test_unchecked_list_and_non_native_array_keep_legacy_column_dtypes():
    mixed = [(1, 2.5), (3, 4.5)]
    lower, upper = split_variable_bounds(mixed, 2, checked=False)
    assert lower.dtype == np.dtype(np.int_)
    assert upper.dtype == np.float64
    big_endian = np.array([[1., 2.], [3., 4.]], dtype='>f8')
    lower, upper = split_variable_bounds(big_endian, 2, checked=False)
    assert lower.dtype == upper.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(lower, [1., 3.])
    np.testing.assert_array_equal(upper, [2., 4.])


def test_numeric_fast_path_never_requests_an_object_array(monkeypatch):
    import src.lp_bounds as module

    original = np.asarray
    requested_dtypes = []

    def tracked(*args, **kwargs):
        requested_dtypes.append(kwargs.get('dtype'))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.np, 'asarray', tracked)
    bounds = np.column_stack((np.arange(16.), np.arange(16.) + 1.))
    split_variable_bounds(bounds, 16, checked=True)
    split_variable_bounds(bounds, 16, checked=False)
    assert object not in requested_dtypes
