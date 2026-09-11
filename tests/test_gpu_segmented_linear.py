"""Ordered static-map arithmetic; CUDA tests are run only by the parent."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.gpu_segmented_linear import DeviceSegmentedLinear, _CUDA


def maps():
    return (np.array([0, 3, 6, 6], dtype=np.int64),
            np.array([0, 1, 2, 0, 2, 1], dtype=np.int64),
            np.ones(6, dtype=np.float64))


def reference(indptr, indices, weights, values):
    result = np.empty(len(indptr)-1, dtype=np.float64)
    with np.errstate(all='ignore'):
        for row in range(len(result)):
            total = np.float64(0.)
            for entry in range(indptr[row], indptr[row+1]):
                product = np.float64(weights[entry] * values[indices[entry]])
                total = np.float64(total + product)
            result[row] = total
    return result


def assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_cpu_declared_order_empty_segment_and_independent_outputs():
    supplied = maps()
    operation = DeviceSegmentedLinear(*supplied, 3, cp=np)
    values = np.array([1e16, 1., -1e16])
    before = values.copy()
    first = operation.apply(values)
    assert_bits_equal(first, np.array([0., 1., 0.]))
    second = operation.apply(values)
    assert not np.shares_memory(first, second)
    first[:] = 99.
    assert_bits_equal(second, np.array([0., 1., 0.]))
    assert_bits_equal(values, before)
    assert operation.validate_integrity()


def test_cpu_multiply_rounding_is_not_fused_into_add():
    epsilon = np.float64(2.**-27)
    operation = DeviceSegmentedLinear(np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64), np.array([-1., 1.+epsilon]), 2, cp=np)
    actual = operation.apply(np.array([1., 1.-epsilon]))
    # Separate rounding gives zero; fma(1+eps,1-eps,-1) gives -2**-54.
    assert_bits_equal(actual, np.array([0.]))


@pytest.mark.parametrize('seed', range(12))
def test_cpu_random_maps_preserve_duplicates_and_unsorted_indices(seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 9, size=19, dtype=np.int64)
    indptr = np.r_[np.int64(0), np.cumsum(counts)]
    indices = rng.integers(0, 13, size=int(indptr[-1]), dtype=np.int64)
    weights = rng.normal(size=len(indices))
    values = rng.normal(size=13)
    operation = DeviceSegmentedLinear(indptr, indices, weights, 13, cp=np)
    assert_bits_equal(operation.apply(values), reference(indptr, indices, weights, values))


@pytest.mark.parametrize('output_size', [0, 1, 7])
def test_cpu_no_entries_and_empty_outputs(output_size):
    operation = DeviceSegmentedLinear(np.zeros(output_size+1, dtype=np.int64),
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 1, cp=np)
    result = operation.apply(np.array([np.nan]))
    assert result.shape == (output_size,)
    assert_bits_equal(result, np.zeros(output_size))


def test_cpu_nonfinite_values_overflow_and_underflow_propagate():
    operation = DeviceSegmentedLinear(np.arange(7, dtype=np.int64),
        np.arange(6, dtype=np.int64), np.array([2., 1., 0., .5, .5, -1.]), 6, cp=np)
    values = np.array([np.finfo(float).max, np.nan, np.inf,
                       np.finfo(float).tiny, np.nextafter(0., 1.), np.inf])
    actual = operation.apply(values)
    assert np.isposinf(actual[0]) and np.isnan(actual[1]) and np.isnan(actual[2])
    assert actual[3] == np.finfo(float).tiny*.5
    assert_bits_equal(actual[4:5], np.array([0.]))
    assert np.isneginf(actual[5])


def test_cpu_owns_immutable_host_and_separate_expected_snapshots():
    supplied = maps()
    operation = DeviceSegmentedLinear(*supplied, 3, cp=np)
    proof = operation.proof_hash
    supplied[0][-1] = 0
    supplied[1][:] = 2
    supplied[2][:] = 99.
    for (label, target, expected), (host_label, host) in zip(
            operation.static_targets, operation.host_snapshots):
        assert label == host_label
        assert not np.shares_memory(target, expected)
        assert not np.shares_memory(target, host)
        assert not np.shares_memory(expected, host)
        for array in (target, expected, host):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flags.writeable = True
    entries = operation.static_targets
    entries.clear()
    assert len(operation.static_targets) == 3
    assert operation.proof_hash == proof
    assert operation.validate_integrity()


@pytest.mark.parametrize('input_size', [0, -1, True, np.int64(3), 3., 2**63])
def test_cpu_input_size_requires_positive_builtin_int64_range(input_size):
    with pytest.raises(ValueError, match='input_size'):
        DeviceSegmentedLinear(*maps(), input_size, cp=np)


@pytest.mark.parametrize('field,value', [
    (0, np.array([], dtype=np.int64)),
    (0, np.array([1, 6], dtype=np.int64)),
    (0, np.array([0, 7], dtype=np.int64)),
    (0, np.array([0, -1, 6], dtype=np.int64)),
    (0, np.array([0, 4, 3, 6], dtype=np.int64)),
    (0, np.array([0, 2**63-1, 6], dtype=np.int64)),
    (0, np.array([0, 6], dtype=np.uint64)),
    (0, np.array([0., 6.])),
    (0, [0, 6]),
    (1, np.array([0, 1, 3, 0, 2, 1], dtype=np.int64)),
    (1, np.array([0, 1, -1, 0, 2, 1], dtype=np.int64)),
    (1, np.arange(6, dtype=np.int32)),
    (1, np.arange(6, dtype=np.float64)),
    (2, np.ones(5)),
    (2, np.ones(6, dtype=np.float32)),
    (2, np.array([1., 1., np.inf, 1., 1., 1.])),
    (2, np.array([1., 1., np.nan, 1., 1., 1.])),
    (2, np.ones((2, 3))),
])
def test_cpu_malformed_maps_rejected_before_backend(field, value):
    supplied = list(maps())
    supplied[field] = value
    with pytest.raises(ValueError):
        DeviceSegmentedLinear(*supplied, 3, cp=np)


def test_cpu_noncontiguous_host_maps_are_copied_contiguously():
    indptr = np.array([0, 99, 3, 99], dtype=np.int64)[::2]
    indices = np.array([0, 99, 1, 99, 2, 99], dtype=np.int64)[::2]
    weights = np.arange(6, dtype=np.float64)[::2]
    operation = DeviceSegmentedLinear(indptr, indices, weights, 3, cp=np)
    assert all(array.flags.c_contiguous for _, array in operation.host_snapshots)
    assert_bits_equal(operation.apply(np.ones(3)), np.array([6.]))


@pytest.mark.parametrize('values', [np.ones(2), np.ones((1, 3)),
    np.ones(3, dtype=np.float32), np.ones(3, dtype=np.int64),
    np.ones(6)[::2], [1., 2., 3.]])
def test_cpu_apply_shape_dtype_and_contiguous_contract(values):
    operation = DeviceSegmentedLinear(*maps(), 3, cp=np)
    with pytest.raises(ValueError, match='Flat contiguous FP64'):
        operation.apply(values)


@pytest.mark.parametrize('field', ['_targets', '_expected_targets', '_snapshots', '_input_size', '_output_size'])
def test_cpu_owned_metadata_replacement_rejected(field):
    operation = DeviceSegmentedLinear(*maps(), 3, cp=np)
    if field in ('_input_size', '_output_size'):
        setattr(operation, field, getattr(operation, field)+1)
    else:
        original = getattr(operation, field)
        setattr(operation, field, (original[0].copy(), *original[1:]))
    with pytest.raises(ValueError, match='changed'):
        operation.apply(np.ones(3))


def test_cpu_thread_and_backend_binding():
    operation = DeviceSegmentedLinear(*maps(), 3, cp=np)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation.apply, np.ones(3))
        with pytest.raises(RuntimeError, match='thread'):
            future.result()
    operation.cp = object()
    with pytest.raises(RuntimeError, match='backend'):
        operation.apply(np.ones(3))


def test_cpu_apply_has_no_finite_scan_and_integrity_is_explicit(monkeypatch):
    operation = DeviceSegmentedLinear(*maps(), 3, cp=np)
    monkeypatch.setattr(np, 'isfinite', lambda *args: pytest.fail('No per-call finite scan'))
    monkeypatch.setattr(operation, 'validate_integrity', lambda: pytest.fail('No implicit value audit'))
    actual = operation.apply(np.array([np.inf, 1., -np.inf]))
    assert np.isnan(actual[:2]).all()


def test_cpu_kernel_requires_separate_ordered_rounding():
    assert '__dmul_rn' in _CUDA and '__dadd_rn' in _CUDA
    assert 'atomic' not in _CUDA and '__syncthreads' not in _CUDA
    assert 'entry = indptr[segment]' in _CUDA
    assert '++entry' in _CUDA and 'double total = 0.0' in _CUDA


def test_cuda_matches_cpu_bitwise_and_uses_device_expected_snapshots(monkeypatch):
    import cupy as cp
    supplied = maps()
    cuda = DeviceSegmentedLinear(*supplied, 3, cp=cp)
    cpu = DeviceSegmentedLinear(*supplied, 3, cp=np)
    assert cuda.validate_integrity()
    values = np.array([1e16, 1., -1e16])
    device_values = cp.asarray(values)
    for (label, target, expected), (host_label, host) in zip(cuda.static_targets, cuda.host_snapshots):
        assert label == host_label
        assert isinstance(expected, cp.ndarray)
        assert target.data.ptr != expected.data.ptr
        np.testing.assert_array_equal(target.get(), host)
        np.testing.assert_array_equal(expected.get(), host)
    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', lambda *args: pytest.fail('No apply D2H'))
        patch.setattr(cp, 'asarray', lambda *args, **kwargs: pytest.fail('No apply map upload'))
        patch.setattr(cp, 'isfinite', lambda *args: pytest.fail('No apply finite scan'))
        first = cuda.apply(device_values)
        second = cuda.apply(device_values)
    assert first.data.ptr != second.data.ptr
    assert_bits_equal(first.get(), cpu.apply(values))
    assert_bits_equal(second.get(), cpu.apply(values))


def test_cuda_separate_rounding_subnormal_and_overflow():
    import cupy as cp
    epsilon = 2.**-27
    indptr = np.array([0, 2, 3, 4, 5, 5], dtype=np.int64)
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    weights = np.array([-1., 1.+epsilon, .5, 2., 0.])
    values = np.array([1., 1.-epsilon, np.finfo(float).tiny, np.finfo(float).max, np.inf])
    actual = DeviceSegmentedLinear(indptr, indices, weights, 5, cp=cp).apply(cp.asarray(values)).get()
    expected = DeviceSegmentedLinear(indptr, indices, weights, 5, cp=np).apply(values)
    assert_bits_equal(actual[:3], expected[:3])
    assert np.isposinf(actual[2]) and np.isnan(actual[3])
    assert_bits_equal(actual[4:], np.array([0.]))


def test_cuda_random_maps_match_declared_order_bitwise():
    import cupy as cp
    rng = np.random.default_rng(991)
    counts = rng.integers(0, 31, size=43, dtype=np.int64)
    indptr = np.r_[np.int64(0), np.cumsum(counts)]
    indices = rng.integers(0, 71, size=int(indptr[-1]), dtype=np.int64)
    weights, values = rng.normal(size=len(indices)), rng.normal(size=71)
    operation = DeviceSegmentedLinear(indptr, indices, weights, 71, cp=cp)
    assert_bits_equal(operation.apply(cp.asarray(values)).get(), reference(indptr, indices, weights, values))


def test_cuda_stream_and_static_content_audit():
    import cupy as cp
    operation = DeviceSegmentedLinear(*maps(), 3, cp=cp)
    values = cp.ones(3, dtype=cp.float64)
    with cp.cuda.Stream(non_blocking=True):
        with pytest.raises(RuntimeError, match='stream'):
            operation.apply(values)
    operation.static_targets[2][1][0] = 2.
    with pytest.raises(ValueError, match='differs'):
        operation.validate_integrity()


def test_cuda_rejects_host_or_foreign_device_values():
    import cupy as cp
    operation = DeviceSegmentedLinear(*maps(), 3, cp=cp)
    with pytest.raises(ValueError, match='FP64'):
        operation.apply(np.ones(3))
    if cp.cuda.runtime.getDeviceCount() > 1:
        with cp.cuda.Device((operation.device+1) % cp.cuda.runtime.getDeviceCount()):
            foreign = cp.ones(3, dtype=cp.float64)
        with pytest.raises(ValueError, match='device'):
            operation.apply(foreign)
