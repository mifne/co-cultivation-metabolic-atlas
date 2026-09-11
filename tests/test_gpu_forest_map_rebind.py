"""Verified static map reuse must reproduce fresh current-input postsolve."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_forest_map import GpuForestMap, _prepare_host_maps, _plan_key
from src.lp_equality_reduction import HomogeneousEqualityReduction


def inputs(revision=0):
    return [(csr_matrix([[1., -sign, 0.], [0., 0., 1.+revision*.2]]),
        np.array([0., 1.-revision*.1]), np.array([0., 0., -2.]),
        np.array([2.-revision*.5, 2., 2.]), np.array([-1., -sign, 1.+revision]), 1)
        for sign in [1., -1.]]


def setup(p):
    plans=[HomogeneousEqualityReduction.from_problem(v) for v in p]
    return plans,[a.reduce(v) for a,v in zip(plans,p)]


def test_cpu_host_optional_verified_plans_are_canonical_and_owned():
    plans,red=setup(inputs())
    host,owned=_prepare_host_maps(plans,red,_return_verified=True)
    assert all(a is not b for a,b in zip(plans,owned))
    assert [_plan_key(a) for a in plans]==[_plan_key(a) for a in owned]
    plans[0].weights[0]=22.
    assert owned[0].weights[0]==1.


def test_cuda_rebind_copy_avoids_tree_search_and_matches_fresh(monkeypatch):
    import cupy as cp
    plans,red=setup(inputs())
    old=GpuForestMap(plans,red)
    current=[a.reduce(v) for a,v in zip(plans,inputs(1))]
    fresh=GpuForestMap(plans,current)
    old_cost=old._objective.copy()
    def forbidden(*a,**k): raise AssertionError('Must reuse verified trees')
    monkeypatch.setattr(HomogeneousEqualityReduction,'from_problem',forbidden)
    new=old.rebind_copy(current)
    assert new is not old and new.static_maps_reused
    assert new._device_dtypes == old._device_dtypes
    assert new._device_dtypes_key == old._device_dtypes_key
    for name in ('transform','compression','dual_lift','transform_t'):
        assert getattr(new,'_'+name) is getattr(old,'_'+name)
    assert new._objective.data.ptr != old._objective.data.ptr
    np.testing.assert_array_equal(old._objective.get(),old_cost.get())
    z=cp.asarray([[.3,.5],[0.,.2]],dtype=cp.float64)
    y=cp.asarray([[.1],[-.1]],dtype=cp.float64)
    for got,expected in zip(new.expand(z,y),fresh.expand(z,y)):
        np.testing.assert_array_equal(got.get(),expected.get())
    # A second numeric rebind also avoids construction and leaves new intact.
    old_hash=new._host_key
    old_again=new.rebind_copy(red)
    assert new._host_key==old_hash
    for got,expected in zip(old_again.expand(z,y),old.expand(z,y)):
        np.testing.assert_array_equal(got.get(),expected.get())


@pytest.mark.parametrize('tamper',['witness','objective','matrix','plan','device','host','batch_rows','dimension'])
def test_cuda_rebind_rejects_stale_or_mixed_state_without_mutation(tamper):
    plans,red=setup(inputs())
    old=GpuForestMap(plans,red)
    current=[a.reduce(v) for a,v in zip(plans,inputs(1))]
    if tamper=='witness': current[0].upper_witness[0]=99
    elif tamper=='objective': current[0].problem[4][0]+=1.
    elif tamper=='matrix': current[0].original_problem[0].data[0]+=1.
    elif tamper=='plan': old._verified_plans[0].T.data[0]+=1.
    elif tamper=='device': old._transform.data[0]+=1.
    elif tamper=='host':
        old._host.weights.flags.writeable=True
        old._host.weights[0,0]+=1.
    elif tamper=='batch_rows': old._batch_rows[0,0]=99
    else: old.original_n+=1
    before=old._objective.get().copy()
    key=old._host_key
    with pytest.raises(ValueError): old.rebind_copy(current)
    assert old._host_key==key
    np.testing.assert_array_equal(old._objective.get(),before)


def device_part(mapper, path):
    names = path.split('.')
    owner = mapper if len(names) == 1 else getattr(mapper, '_'+names[0])
    name = '_'+names[0] if len(names) == 1 else names[1]
    return owner, name, getattr(owner, name)


_SPARSE_NAMES = ('transform', 'compression', 'dual_lift', 'original_at', 'transform_t')
_DENSE_NAMES = ('objective', 'weights', 'kept_rows', 'eliminated_rows',
                'lower_witness', 'upper_witness', 'fallback_witness', 'batch_rows')


@pytest.mark.parametrize('path',
    [name+'.'+part for name in _SPARSE_NAMES for part in ('data', 'indices', 'indptr')]
    + list(_DENSE_NAMES))
def test_cuda_identical_values_with_wrong_dtype_are_rejected(path):
    import cupy as cp
    plans, reductions = setup(inputs())
    mapper = GpuForestMap(plans, reductions)
    current = [plan.reduce(value) for plan, value in zip(plans, inputs(1))]
    owner, name, original = device_part(mapper, path)
    wrong_dtype = cp.float64 if original.dtype.kind in 'iu' else cp.float32
    replacement = original.astype(wrong_dtype)
    # These small manufactured coefficients are exactly representable, so
    # the historical value-only array_equal check could not detect this.
    assert bool(cp.array_equal(original, replacement))
    setattr(owner, name, replacement)
    before_key = mapper._host_key
    before_objective = cp.asnumpy(mapper._objective).copy()
    with pytest.raises(ValueError, match='dtype/device'):
        mapper.rebind_copy(current)
    assert getattr(owner, name) is replacement
    assert mapper._host_key == before_key
    np.testing.assert_array_equal(cp.asnumpy(mapper._objective), before_objective)


@pytest.mark.parametrize('path',
    [name+'.'+part for name in _SPARSE_NAMES for part in ('indices', 'indptr')]
    + ['kept_rows', 'eliminated_rows', 'batch_rows'])
def test_cuda_index_width_must_match_verified_native_layout(path):
    import cupy as cp
    plans, reductions = setup(inputs())
    mapper = GpuForestMap(plans, reductions)
    owner, name, original = device_part(mapper, path)
    replacement = original.astype(cp.int64 if original.dtype == cp.int32 else cp.int32)
    assert bool(cp.array_equal(original, replacement))
    setattr(owner, name, replacement)
    with pytest.raises(ValueError, match='dtype/device'):
        mapper.rebind_copy(reductions)
    assert getattr(owner, name) is replacement


@pytest.mark.parametrize('path', ['transform.data', 'transform.indices', 'transform.indptr',
                                  'kept_rows', 'batch_rows'])
def test_cuda_same_valued_host_arrays_cannot_replace_device_payload(path):
    plans, reductions = setup(inputs())
    mapper = GpuForestMap(plans, reductions)
    owner, name, original = device_part(mapper, path)
    setattr(owner, name, original.get())
    with pytest.raises(ValueError, match='dtype/device'):
        mapper.rebind_copy(reductions)


def test_cuda_native_dtype_snapshot_tampering_is_rejected():
    plans, reductions = setup(inputs())
    mapper = GpuForestMap(plans, reductions)
    mapper._device_dtypes = tuple((name, '<f8' if name == 'batch_rows' else dtype)
                                 for name, dtype in mapper._device_dtypes)
    with pytest.raises(ValueError, match='snapshot/map'):
        mapper.rebind_copy(reductions)


@pytest.mark.parametrize('path', ['transform.indices', 'transform.indptr', 'batch_rows'])
def test_cuda_same_valued_foreign_device_array_is_rejected(path):
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip('A second CUDA device is required for the foreign-device guard')
    plans, reductions = setup(inputs())
    mapper = GpuForestMap(plans, reductions)
    owner, name, original = device_part(mapper, path)
    expected = original.get()
    with cp.cuda.Device((mapper.device+1) % cp.cuda.runtime.getDeviceCount()):
        replacement = cp.asarray(expected)
    setattr(owner, name, replacement)
    with pytest.raises(ValueError, match='dtype/device'):
        mapper.rebind_copy(reductions)
    assert getattr(owner, name) is replacement
