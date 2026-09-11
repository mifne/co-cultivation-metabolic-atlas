"""Selected-before-encoding equivalence; GPU cases require an explicit run."""

import builtins

import numpy as np
import pytest

from src.selected_lp_features import FIELDS, SelectedLPFeatures


def _inputs(dtype=np.float64):
    shapes = ((3,), (4,), (4,), (4,), (2, 4), (4,), (3,))
    result = {}
    for index, (field, shape) in enumerate(zip(FIELDS, shapes)):
        values = np.arange(np.prod(shape), dtype=np.float64) - 2.375 + index * 0.125
        result[field] = values.reshape(shape).astype(dtype)
    if np.dtype(dtype).kind == 'f':
        result['rhs'][:] = [np.nan, np.inf, -np.inf]
    return result


def _legacy_cpu(inputs, indices):
    joined = np.concatenate([np.asarray(inputs[field]).ravel() for field in FIELDS])
    joined = np.nan_to_num(joined, nan=0., posinf=1e13, neginf=-1e13)
    return (np.sign(joined) * np.log1p(np.abs(joined))).astype(np.float32)[indices]


def _indices(inputs):
    starts = np.cumsum([0] + [np.asarray(inputs[field]).size for field in FIELDS])
    return np.array([starts[-1] - 1, *starts[:-1][::-1], 2, 1, 0, 2, 5, 5], dtype=np.int64)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int8, np.int32, np.int64, np.uint64])
def test_cpu_order_duplicates_dtype_and_finite_encoding_match_original(dtype):
    inputs = _inputs(dtype)
    indices = _indices(inputs)
    snapshots = {field: value.copy() for field, value in inputs.items()}
    plan = SelectedLPFeatures.from_inputs(inputs, indices)
    with np.errstate(over='ignore', invalid='ignore'):
        expected = _legacy_cpu(inputs, indices)
        result = plan.cpu_features(inputs)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32 and result.shape == indices.shape
    for field in FIELDS:
        np.testing.assert_array_equal(inputs[field], snapshots[field])


@pytest.mark.parametrize('unselected_dtype', [np.float64, np.int64, np.uint64])
def test_cpu_unselected_field_participates_in_common_dtype_promotion(unselected_dtype):
    inputs = _inputs(np.float32)
    inputs['rhs'] = np.array([0.27, 1.37, 2.125], dtype=np.float32)
    inputs['delta'] = inputs['delta'].astype(unselected_dtype)
    indices = np.array([2, 0, 1, 0], dtype=np.int32)  # only float32 rhs entries
    plan = SelectedLPFeatures.from_inputs(inputs, indices)
    np.testing.assert_array_equal(plan.cpu_features(inputs), _legacy_cpu(inputs, indices))
    # A dtype change between calls must not reuse stale promotion metadata.
    inputs['row_scale'] = inputs['row_scale'].astype(np.float64)
    np.testing.assert_array_equal(plan.cpu_features(inputs), _legacy_cpu(inputs, indices))


def test_cpu_signed_and_unsigned_integer_promotion_matches_concatenate():
    inputs = _inputs(np.int64)
    inputs['lower'] = np.array([0, 2**63, 2**64 - 1, 1], dtype=np.uint64)
    inputs['upper'] = np.array([np.iinfo(np.int64).min, -1, 0, 4], dtype=np.int64)
    indices = np.arange(sum(value.size for value in inputs.values()))[::-1]
    plan = SelectedLPFeatures.from_inputs(inputs, indices)
    np.testing.assert_array_equal(plan.cpu_features(inputs), _legacy_cpu(inputs, indices))


@pytest.mark.parametrize('view', ['transpose', 'negative_stride', 'strided'])
def test_cpu_c_order_selection_of_noncontiguous_arrays(view):
    inputs = _inputs()
    source = np.arange(24., dtype=np.float64).reshape(4, 6)
    inputs['delta'] = {'transpose': source.T, 'negative_stride': source[::-1, ::-1], 'strided': source[::2, ::2]}[view]
    indices = np.arange(sum(value.size for value in inputs.values()))[::-1]
    plan = SelectedLPFeatures.from_inputs(inputs, indices)
    np.testing.assert_array_equal(plan.cpu_features(inputs), _legacy_cpu(inputs, indices))


def test_cpu_scalar_and_zero_width_fields_empty_selection():
    inputs = _inputs()
    inputs['delta'] = np.empty((0, 4), dtype=np.float64)
    inputs['row_scale'] = np.array(-3.25)
    indices = np.array([sum(value.size for value in inputs.values()) - 1] * 3)
    plan = SelectedLPFeatures.from_inputs(inputs, indices)
    np.testing.assert_array_equal(plan.cpu_features(inputs), _legacy_cpu(inputs, indices))
    empty = SelectedLPFeatures.from_inputs(inputs, np.empty(0, dtype=np.int64))
    assert empty.cpu_features(inputs).shape == (0,)
    assert empty.cpu_features(inputs).dtype == np.float32


def test_cpu_encoding_touches_only_selected_width_and_never_imports_gpu(monkeypatch):
    inputs = _inputs()
    inputs['delta'] = np.arange(512., dtype=np.float64).reshape(16, 32)[:, ::2]
    indices = np.array([0, 1, 5, 5])
    expected = _legacy_cpu(inputs, indices)
    plan = SelectedLPFeatures.from_inputs(inputs, indices)
    original_import = builtins.__import__
    original_nan_to_num = np.nan_to_num
    encoded_shapes = []

    def cpu_only_import(name, *args, **kwargs):
        if name.split('.')[0] in {'cupy', 'cupyx', 'cuda'}:
            raise AssertionError('GPU imported on the CPU feature path')
        return original_import(name, *args, **kwargs)

    def track_encoding(value, *args, **kwargs):
        encoded_shapes.append(value.shape)
        return original_nan_to_num(value, *args, **kwargs)

    def forbid_concatenate(*args, **kwargs):
        raise AssertionError('Full LP fields must not be concatenated')

    monkeypatch.setattr(builtins, '__import__', cpu_only_import)
    monkeypatch.setattr(np, 'nan_to_num', track_encoding)
    monkeypatch.setattr(np, 'concatenate', forbid_concatenate)
    np.testing.assert_array_equal(plan.cpu_features(inputs), expected)
    assert encoded_shapes == [(len(indices),)]


@pytest.mark.parametrize('indices', [np.array([-1]), np.array([30]), np.array([2**64 - 1], dtype=np.uint64),
    np.array([[0]]), np.array([0.0]), np.array([True]), np.array(0)])
def test_invalid_indices_rejected_at_setup(indices):
    with pytest.raises(ValueError):
        SelectedLPFeatures.from_inputs(_inputs(), indices)


@pytest.mark.parametrize('shape', [(-1, 4), (2.0, 4), (True, 4), 4, (2**63, 4)])
def test_invalid_shapes_rejected_at_setup(shape):
    shapes = {field: value.shape for field, value in _inputs().items()}
    shapes['delta'] = shape
    with pytest.raises(ValueError):
        SelectedLPFeatures(np.array([0]), shapes)


def test_setup_copies_index_metadata_and_rejects_device_indices():
    inputs = _inputs()
    indices = np.array([3, 2, 0, 3])
    expected = _legacy_cpu(inputs, indices)
    plan = SelectedLPFeatures.from_inputs(inputs, indices)
    indices[:] = 999
    np.testing.assert_array_equal(plan.cpu_features(inputs), expected)
    assert not plan.feature_indices.flags.writeable
    with pytest.raises(TypeError):
        plan.field_shapes['rhs'] = (77,)

    class DeviceIndices:
        __cuda_array_interface__ = {}

        def __array__(self, *args, **kwargs):
            raise AssertionError('Device indices must never be converted implicitly')

    with pytest.raises(TypeError, match='host integer'):
        SelectedLPFeatures.from_inputs(inputs, DeviceIndices())
    inputs['rhs'] = DeviceIndices()
    with pytest.raises(TypeError, match='device array'):
        plan.cpu_features(inputs)


@pytest.mark.parametrize('mutation', ['missing', 'unselected_shape', 'same_size_shape'])
def test_cpu_validates_every_dynamic_field_shape(mutation):
    inputs = _inputs()
    plan = SelectedLPFeatures.from_inputs(inputs, np.array([0]))
    if mutation == 'missing':
        del inputs['delta']
    elif mutation == 'unselected_shape':
        inputs['delta'] = np.zeros((3, 4))
    else:
        inputs['delta'] = inputs['delta'].reshape(4, 2)
    with pytest.raises(ValueError):
        plan.cpu_features(inputs)


def test_from_inputs_batched_metadata_and_validation_without_device_runtime():
    inputs = {field: np.repeat(value[None], 3, axis=0) for field, value in _inputs().items()}
    plan = SelectedLPFeatures.from_inputs(inputs, np.array([0]), batched=True)
    assert plan.field_shapes['delta'] == (2, 4)
    inputs['delta'] = inputs['delta'][:2]
    with pytest.raises(ValueError, match='batch sizes'):
        SelectedLPFeatures.from_inputs(inputs, np.array([0]), batched=True)
    inputs['rhs'] = np.array(0.)
    with pytest.raises(ValueError, match='batch dimension'):
        SelectedLPFeatures.from_inputs(inputs, np.array([0]), batched=True)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int64])
def test_gpu_features_match_original_batch_order_and_encoding(dtype):
    cp = pytest.importorskip('cupy')
    from src.gpu_neural_basis_proposal import features

    host = _inputs(dtype)
    indices = _indices(host)
    inputs = {field: cp.asarray(np.repeat(value[None], 3, axis=0)) for field, value in host.items()}
    plan = SelectedLPFeatures.from_inputs(inputs, indices, batched=True)
    before = {field: value.copy() for field, value in inputs.items()}
    result = plan.gpu_features(inputs)
    expected = features(inputs)[:, cp.asarray(indices)]
    cp.testing.assert_array_equal(result, expected)
    assert result.dtype == cp.float32 and result.shape == (3, len(indices))
    for field in FIELDS:
        cp.testing.assert_array_equal(inputs[field], before[field])
    # Different batch sizes and values reuse the index plan.
    changed = {field: value[:1].copy() for field, value in inputs.items()}
    changed['c'] += 1
    cp.testing.assert_array_equal(plan.gpu_features(changed), features(changed)[:, cp.asarray(indices)])


def test_gpu_noncontiguous_mixed_dtypes_no_download_and_only_selected_encoding(monkeypatch):
    cp = pytest.importorskip('cupy')
    from src.gpu_neural_basis_proposal import features

    inputs = {field: cp.asarray(np.repeat(value[None], 2, axis=0)) for field, value in _inputs(np.float32).items()}
    inputs['delta'] = cp.arange(48., dtype=cp.float64).reshape(2, 4, 6)[:, ::-1, ::2]
    indices = np.array([0, 1, 6, 6, 2, 27, 30])
    plan = SelectedLPFeatures.from_inputs(inputs, indices, batched=True)
    expected = features(inputs)[:, cp.asarray(indices)]
    plan.gpu_features(inputs)  # setup before forbidding metadata uploads
    original_encoding = cp.nan_to_num
    encoded_shapes = []

    def forbidden(*args, **kwargs):
        raise AssertionError('GPU feature routing must not transfer data or concatenate all fields')

    def track_encoding(value, *args, **kwargs):
        encoded_shapes.append(value.shape)
        return original_encoding(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', forbidden)
        patch.setattr(cp, 'asarray', forbidden)
        patch.setattr(cp, 'concatenate', forbidden)
        patch.setattr(cp, 'nan_to_num', track_encoding)
        result = plan.gpu_features(inputs)
    cp.testing.assert_array_equal(result, expected)
    assert encoded_shapes == [(2, len(indices))]


def test_gpu_scalar_zero_width_empty_selection_and_shape_guards():
    cp = pytest.importorskip('cupy')
    from src.gpu_neural_basis_proposal import features

    host = _inputs()
    host['delta'] = np.empty((0, 4))
    host['row_scale'] = np.array(1.75)
    inputs = {field: cp.asarray(np.repeat(value[None], 2, axis=0)) for field, value in host.items()}
    width = sum(value.size for value in host.values())
    indices = np.array([width - 1, 0, width - 1])
    plan = SelectedLPFeatures.from_inputs(inputs, indices, batched=True)
    cp.testing.assert_array_equal(plan.gpu_features(inputs), features(inputs)[:, cp.asarray(indices)])
    empty = SelectedLPFeatures.from_inputs(inputs, np.empty(0, dtype=np.int64), batched=True)
    assert empty.gpu_features(inputs).shape == (2, 0)
    with pytest.raises(ValueError, match='batch sizes'):
        plan.gpu_features(dict(inputs, c=inputs['c'][:1]))
    with pytest.raises(ValueError, match='shape'):
        plan.gpu_features(dict(inputs, delta=cp.zeros((2, 1, 4))))
    with pytest.raises(TypeError, match='CuPy'):
        plan.gpu_features(dict(inputs, rhs=np.zeros((2, 3))))
    with pytest.raises(ValueError, match='Missing'):
        plan.gpu_features({field: value for field, value in inputs.items() if field != 'c'})

