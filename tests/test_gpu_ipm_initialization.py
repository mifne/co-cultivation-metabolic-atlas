"""CPU-only initialization algebra and validation; no LP or CUDA execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.gpu_ipm_initialization import initialize_inequality_dual


def _inputs():
    return (np.array([[1., 1e6, 2., 10., 100., 1.],
                      [4., 8., 16., 32., 64., 128.]]),
            np.array([[-2., 3.], [0., -4.]]),
            np.array([[3., -5.], [-7., 8.]]),
            np.array([[6., -7.], [-9., 10.]]))


def test_default_and_legacy_are_exactly_the_previous_initialization():
    s, row, lower, upper = _inputs()
    expected = np.concatenate((np.maximum(1., -row), np.maximum(1., lower),
                               np.maximum(1., -upper)), axis=1)
    for kwargs in ({}, {'mode': 'legacy'}):
        actual = initialize_inequality_dual(s, row, lower, upper, xp=np, **kwargs)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64


def test_balanced_mixed_cost_signs_and_warm_duals_preserve_the_right_sign():
    s, row, lower, upper = _inputs()
    actual = initialize_inequality_dual(s, row, lower, upper, mode='balanced', xp=np)
    np.testing.assert_array_equal(actual,
        [[2., 1e-6, 3., .1, .01, 7.],
         [.25, 4., .0625, 8., 9., .0078125]])
    # Proposals above the floor remain untouched, not rescaled to force mu=1.
    assert (s * actual).max() > 1.


def test_balanced_cold_zero_duals_have_unit_complementarity():
    s = np.array([[1., 2., 1e6], [4., 8., 1e3]])
    zero = np.zeros((2, 1))
    actual = initialize_inequality_dual(s, zero, zero, zero, mode='balanced', xp=np)
    np.testing.assert_allclose(s * actual, np.ones_like(s), rtol=2e-16, atol=0.)
    assert np.mean(s * actual) == pytest.approx(1.)
    assert np.mean(s) > 1e5


@pytest.mark.parametrize('active_block', range(3))
def test_individual_empty_blocks_are_supported(active_block):
    parts = [np.empty((2, 0)) for _ in range(3)]
    parts[active_block] = np.zeros((2, 2))
    s = np.array([[1., 2.], [4., 8.]])
    actual = initialize_inequality_dual(s, *parts, mode='balanced', xp=np)
    np.testing.assert_array_equal(actual, 1. / s)


@pytest.mark.parametrize('mode', ['legacy', 'balanced'])
def test_inputs_are_unchanged_and_result_is_independent(mode):
    args = _inputs()
    saved = tuple(value.copy() for value in args)
    actual = initialize_inequality_dual(*args, mode=mode, xp=np)
    actual[:] = 100.
    for value, expected in zip(args, saved):
        np.testing.assert_array_equal(value, expected)


def test_largest_finite_slack_keeps_a_positive_finite_floor():
    s = np.array([[np.finfo(np.float64).max]])
    zero, empty = np.zeros((1, 1)), np.empty((1, 0))
    actual = initialize_inequality_dual(s, zero, empty, empty, mode='balanced', xp=np)
    assert np.isfinite(actual).all() and (actual > 0.).all()
    np.testing.assert_allclose(s * actual, [[1.]], rtol=2e-15, atol=0.)


def test_unrepresentable_balanced_reciprocal_fails_without_clipping():
    s = np.array([[np.nextafter(0., 1.)]])
    zero, empty = np.zeros((1, 1)), np.empty((1, 0))
    with pytest.raises(ValueError, match='reciprocal'):
        initialize_inequality_dual(s, zero, empty, empty, mode='balanced', xp=np)
    np.testing.assert_array_equal(
        initialize_inequality_dual(s, zero, empty, empty, xp=np), [[1.]])


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_input_fails_closed(index, bad):
    args = list(_inputs())
    args[index][0, 0] = bad
    with pytest.raises(ValueError, match='Finite'):
        initialize_inequality_dual(*args, mode='balanced', xp=np)


@pytest.mark.parametrize('bad', [0., -0., -1.])
def test_nonpositive_slack_fails_closed(bad):
    args = list(_inputs())
    args[0][0, 0] = bad
    with pytest.raises(ValueError, match='positive'):
        initialize_inequality_dual(*args, xp=np)


@pytest.mark.parametrize('index', range(4))
def test_fp32_is_not_silently_promoted(index):
    args = list(_inputs())
    args[index] = args[index].astype(np.float32)
    with pytest.raises(ValueError, match='FP64'):
        initialize_inequality_dual(*args, xp=np)


@pytest.mark.parametrize('kind', ['rank', 'batch', 'split_width', 'empty_batch', 'empty_width', 'list'])
def test_shape_and_backend_errors_are_not_broadcast(kind):
    args = list(_inputs())
    if kind == 'rank':
        args[0] = args[0].ravel()
    elif kind == 'batch':
        args[1] = args[1][:1]
    elif kind == 'split_width':
        args[1] = args[1][:, :1]
    elif kind == 'empty_batch':
        args = [value[:0] for value in args]
    elif kind == 'empty_width':
        args = [value[:, :0] for value in args]
    else:
        args[1] = args[1].tolist()
    with pytest.raises(ValueError):
        initialize_inequality_dual(*args, xp=np)


@pytest.mark.parametrize('mode', ['', 'adaptive', None, True, ['balanced']])
def test_unknown_modes_fail_closed(mode):
    with pytest.raises(ValueError, match='initializer'):
        initialize_inequality_dual(*_inputs(), mode=mode, xp=np)


class _DeviceArray(np.ndarray):
    """Metadata-only fake for device validation; never invokes CUDA."""
    @property
    def device(self):
        return SimpleNamespace(id=self.device_id)


@pytest.mark.parametrize('other_index,current_device', [(1, 0), (None, 1)])
def test_mixed_or_noncurrent_device_is_rejected_before_arithmetic(other_index, current_device):
    args = []
    for index, value in enumerate(_inputs()):
        fake = value.view(_DeviceArray)
        fake.device_id = 1 if index == other_index else 0
        args.append(fake)
    backend = SimpleNamespace(ndarray=_DeviceArray, float64=np.float64,
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: current_device)))
    with pytest.raises(ValueError, match='current CUDA device'):
        initialize_inequality_dual(*args, xp=backend)
