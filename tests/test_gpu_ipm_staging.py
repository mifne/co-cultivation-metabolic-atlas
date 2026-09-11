"""CPU-mock tests of compact GPU validation; CUDA tests run only by root."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_ipm_numeric_update import NumericRebindRejected
from src.gpu_ipm_staging import stage_numeric_updates


class _DeviceArray(np.ndarray):
    @property
    def device(self): return SimpleNamespace(id=getattr(self, 'device_id', 0))


class _MockGPU:
    """NumPy computes the mock device ops; only asnumpy counts downloads."""
    __name__ = 'cupy'
    ndarray = _DeviceArray
    def __init__(self):
        self.downloads = []
        self.asarray_calls = 0
        self.fail_allocation_at = None
        self.fail_download = False
        self.stream = SimpleNamespace(ptr=7, synchronize=self.synchronize)
        self.synchronizations = 0
        self.cuda = SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0),
                                    get_current_stream=lambda: self.stream)
    def synchronize(self): self.synchronizations += 1
    def asarray(self, value):
        self.asarray_calls += 1
        if self.asarray_calls == self.fail_allocation_at: raise MemoryError('Mock allocation failed')
        return np.array(value, copy=True).view(_DeviceArray)
    def all(self, value): return np.asarray(np.all(value)).view(_DeviceArray)
    def stack(self, values): return np.stack(values).view(_DeviceArray)
    def asnumpy(self, value):
        if self.fail_download: raise RuntimeError('Mock transfer failed')
        self.downloads.append((value.shape, value.dtype, value.nbytes))
        return np.array(value, copy=True)


def _fixture(*, xp=None, vectors=2):
    if xp is None: xp = _MockGPU()
    old_matrix = csr_matrix(np.array([[1., 0., 2.], [0., 3., 4.]]))
    new_matrix = old_matrix.copy()
    new_matrix.data[1] = 5.
    old_vectors = {('vectors', index): np.array([float(index), np.inf, -np.inf])
                   for index in range(vectors)}
    new_vectors = {path: value.copy() for path, value in old_vectors.items()}
    if vectors: new_vectors[('vectors', 0)][0] += 1.
    old_vectors[('values',)] = np.array([[1., 2.]])
    old_vectors[('factor', 'values')] = np.array([[1., 2.]])
    new_vectors[('values',)] = np.array([[1., 3.]])
    new_vectors[('factor', 'values')] = np.array([[1., 3.]])
    convert = xp.asarray
    current_matrix = SimpleNamespace(shape=old_matrix.shape,
        data=convert(old_matrix.data), indices=convert(old_matrix.indices),
        indptr=convert(old_matrix.indptr))
    if xp is np:
        stream = SimpleNamespace(synchronize=lambda: None)
    else: stream = xp.stream
    solver = SimpleNamespace(cp=xp, matrix=current_matrix,
        vectors=[convert(old_vectors[('vectors', index)]) for index in range(vectors)],
        values=convert(np.array([[900., 901.]])),
        factor=SimpleNamespace(device=0, stream=stream, _context=lambda: None,
                               values=convert(np.array([[800., 801.]]))))
    if xp is not np: xp.asarray_calls = 0
    return solver, ({('matrix',): old_matrix}, old_vectors), ({('matrix',): new_matrix}, new_vectors)


def _targets(solver):
    return [solver.matrix.data, solver.matrix.indices, solver.matrix.indptr,
            *solver.vectors, solver.values, solver.factor.values]


def _assert_unmutated(solver, old):
    for value, expected in zip(_targets(solver), old):
        np.testing.assert_array_equal(value, expected)


def test_single_compact_download_checks_every_static_buffer_and_returns_only_changes():
    solver, old, new = _fixture(vectors=82)
    before = [value.copy() for value in _targets(solver)]
    copies, skipped = stage_numeric_updates(solver, old, new)
    assert solver.cp.downloads == [((85,), np.dtype(bool), 85)]
    assert len(copies) == 4 and skipped == 81
    assert [name for name, _target, _source in copies] == ["('matrix',).data",
        "('vectors', 0)", "('values',)", "('factor', 'values')"]
    _assert_unmutated(solver, before)
    for name, target, source in copies:
        assert source.shape == target.shape and source.dtype == target.dtype
        assert not np.shares_memory(target, source), name
    assert solver.cp.synchronizations == 1


def test_numpy_reference_matches_mock_gpu_and_sources_are_independent():
    reference, old, new = _fixture(xp=np)
    device, device_old, device_new = _fixture()
    copies, skipped = stage_numeric_updates(reference, old, new)
    device_copies, device_skipped = stage_numeric_updates(device, device_old, device_new)
    assert skipped == device_skipped
    for (name, _target, source), (device_name, _dt, device_source) in zip(copies, device_copies):
        assert name == device_name
        np.testing.assert_array_equal(source, device_source)
    before = [source.copy() for _name, _target, source in copies]
    new[0][('matrix',)].data[:] = 100.
    for value in new[1].values(): value[:] = 100.
    for (_name, _target, source), expected in zip(copies, before):
        np.testing.assert_array_equal(source, expected)


@pytest.mark.parametrize('part', ['indptr', 'indices', 'data', 'vector', 'vector_nan', 'data_nan'])
def test_stale_device_content_is_rejected_by_compact_status_without_mutation(part):
    solver, old, new = _fixture()
    if part.startswith('vector'):
        solver.vectors[1][0] = np.nan if part.endswith('nan') else 100.
    else:
        getattr(solver.matrix, 'data' if part == 'data_nan' else part)[0] = (
            np.nan if part.endswith('nan') else 100)
    before = [value.copy() for value in _targets(solver)]
    with pytest.raises(NumericRebindRejected, match='Stale current'):
        stage_numeric_updates(solver, old, new)
    assert len(solver.cp.downloads) == 1
    _assert_unmutated(solver, before)


def test_matching_signed_infinities_and_signed_zero_follow_array_equal_semantics():
    solver, old, new = _fixture()
    solver.vectors[0][0] = -0.
    copies, skipped = stage_numeric_updates(solver, old, new)
    assert skipped == 1 and len(copies) == 4


@pytest.mark.parametrize('location', ['old_vector', 'new_vector', 'old_sparse', 'new_sparse', 'new_kkt'])
def test_host_nan_or_nonfinite_sparse_kkt_input_is_rejected_before_download(location):
    solver, old, new = _fixture()
    if location == 'old_vector': old[1][('vectors', 0)][0] = np.nan
    elif location == 'new_vector': new[1][('vectors', 0)][0] = np.nan
    elif location == 'old_sparse': old[0][('matrix',)].data[0] = np.inf
    elif location == 'new_sparse': new[0][('matrix',)].data[0] = np.inf
    else: new[1][('values',)][0, 0] = np.inf
    before = [value.copy() for value in _targets(solver)]
    with pytest.raises(NumericRebindRejected, match='nonfinite'):
        stage_numeric_updates(solver, old, new)
    assert not solver.cp.downloads
    _assert_unmutated(solver, before)


@pytest.mark.parametrize('change', ['keys', 'csr_pattern', 'sparse_shape', 'vector_shape',
                                   'fp32', 'sparse_device', 'vector_device', 'stream', 'device'])
def test_metadata_errors_reject_before_device_work(change):
    solver, old, new = _fixture()
    if change == 'keys': new[1][('extra',)] = np.zeros(1)
    elif change == 'csr_pattern': new[0][('matrix',)].indices[0] = 1
    elif change == 'sparse_shape': solver.matrix.shape = (2, 4)
    elif change == 'vector_shape': solver.vectors[0] = solver.vectors[0][:2]
    elif change == 'fp32': solver.matrix.data = solver.matrix.data.astype(np.float32)
    elif change == 'sparse_device': solver.matrix.indices.device_id = 1
    elif change == 'vector_device': solver.vectors[0].device_id = 1
    elif change == 'stream': solver.factor.stream = SimpleNamespace(ptr=99)
    else: solver.cp.cuda.runtime.getDevice = lambda: 1
    before = [value.copy() for value in _targets(solver)]
    with pytest.raises(NumericRebindRejected): stage_numeric_updates(solver, old, new)
    assert solver.cp.asarray_calls == 0 and not solver.cp.downloads
    _assert_unmutated(solver, before)


def test_int32_target_does_not_hide_out_of_range_old_int64_reference():
    solver, old, new = _fixture()
    # The unchanged new coordinates would overflow if silently cast to int32.
    huge = np.array([2**32, 1, 2], dtype=np.int64)
    old[1][('vectors', 0)] = huge.copy()
    new[1][('vectors', 0)] = huge.copy()
    solver.vectors[0] = solver.cp.asarray(np.array([0, 1, 2], dtype=np.int32))
    with pytest.raises(NumericRebindRejected, match='overflow'):
        stage_numeric_updates(solver, old, new)
    assert not solver.cp.downloads


def test_old_reference_is_not_cast_down_when_only_new_values_fit_target():
    solver, old, new = _fixture()
    old[1][('vectors', 0)] = np.array([2**32, 1, 2], dtype=np.int64)
    new[1][('vectors', 0)] = np.array([0, 1, 2], dtype=np.int64)
    solver.vectors[0] = solver.cp.asarray(np.array([0, 1, 2], dtype=np.int32))
    with pytest.raises(NumericRebindRejected, match='Stale current'):
        stage_numeric_updates(solver, old, new)
    assert len(solver.cp.downloads) == 1


@pytest.mark.parametrize('where', ['reference_allocation', 'replacement_allocation', 'download'])
def test_resource_runtime_failures_drain_and_preserve_original_exception_and_targets(where):
    solver, old, new = _fixture()
    before = [value.copy() for value in _targets(solver)]
    if where == 'reference_allocation': solver.cp.fail_allocation_at = 2
    elif where == 'replacement_allocation': solver.cp.fail_allocation_at = 6
    else: solver.cp.fail_download = True
    with pytest.raises(RuntimeError if where == 'download' else MemoryError):
        stage_numeric_updates(solver, old, new)
    assert solver.cp.synchronizations == 1
    _assert_unmutated(solver, before)


def test_no_static_payloads_need_no_status_download_but_dynamic_values_are_staged():
    solver, old, new = _fixture(vectors=0)
    copies, skipped = stage_numeric_updates(solver, ({}, old[1]), ({}, new[1]))
    assert not solver.cp.downloads and len(copies) == 2 and skipped == 0


def test_cuda_compact_staging_validates_and_stages_without_vector_download(monkeypatch):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1: pytest.skip('CUDA required')
    solver, old, new = _fixture(xp=np)
    solver.cp = cp
    for attr in ('data', 'indices', 'indptr'):
        setattr(solver.matrix, attr, cp.asarray(getattr(solver.matrix, attr)))
    solver.vectors = [cp.asarray(value) for value in solver.vectors]
    solver.values, solver.factor.values = cp.asarray(solver.values), cp.asarray(solver.factor.values)
    solver.factor.device = cp.cuda.runtime.getDevice()
    solver.factor.stream = cp.cuda.get_current_stream()
    real_download, shapes = cp.asnumpy, []
    def status_only(value, *args, **kwargs):
        assert value.dtype == cp.bool_ and value.shape == (5,)
        shapes.append(value.shape)
        return real_download(value, *args, **kwargs)
    monkeypatch.setattr(cp, 'asnumpy', status_only)
    before = [value.copy() for value in _targets(solver)]
    copies, skipped = stage_numeric_updates(solver, old, new)
    assert shapes == [(5,)] and skipped == 1 and len(copies) == 4
    for target, expected in zip(_targets(solver), before):
        assert bool(cp.array_equal(target, expected))
    for _name, target, source in copies:
        assert target.data.ptr != source.data.ptr
