"""CPU-only layout/arithmetic/launch contracts, not a CUDA execution test.

The fake RawKernel is an independent small serial interpreter. Real CUDA
compilation and arithmetic equivalence must additionally be checked on GPU.
"""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_forest_numeric_bounds import DeviceForestBounds, _snapshot_layout, _CUDA
from src.lp_equality_reduction import HomogeneousEqualityReduction


def problem(weights, assignments=None):
    weights = np.asarray(weights, dtype=float)
    if assignments is None:
        assignments = np.zeros(len(weights), dtype=int)
    assignments = np.asarray(assignments)
    rows = []
    for group in np.unique(assignments):
        members = np.flatnonzero(assignments == group)
        for column in members[1:]:
            row = np.zeros(len(weights))
            row[members[0]], row[column] = weights[column], -weights[members[0]]
            rows.append(row)
    a = csr_matrix(np.asarray(rows).reshape((-1, len(weights))))
    return (a, np.zeros(a.shape[0]), np.full(len(weights), -10.),
            np.full(len(weights), 10.), np.zeros(len(weights)), a.shape[0])


class DeviceArray:
    def __init__(self, value, device=0):
        self._array = np.asarray(value)
        self.device = SimpleNamespace(id=device)
        self.data = SimpleNamespace(ptr=self._array.__array_interface__['data'][0])

    @property
    def shape(self): return self._array.shape

    @property
    def dtype(self): return self._array.dtype

    @property
    def strides(self): return self._array.strides

    def copy(self): return DeviceArray(self._array.copy(), self.device.id)

    def get(self): raise AssertionError('D2H is forbidden')

    def __bool__(self): raise AssertionError('Device scalar readback is forbidden')

    def __array__(self, *args, **kwargs): raise AssertionError('Host conversion is forbidden')


class FakeCP:
    ndarray, float64, int64, bool_ = DeviceArray, np.float64, np.int64, np.bool_

    def __init__(self):
        self.current_device, self.current_stream = 0, 71
        self.uploads, self.launches = 0, []
        self.cuda = SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: self.current_device),
            get_current_stream=lambda: SimpleNamespace(ptr=self.current_stream))

    def array(self, value, copy):
        assert copy
        self.uploads += 1
        return DeviceArray(np.array(value, copy=True))

    def empty(self, shape, dtype):
        return DeviceArray(np.empty(shape, dtype=dtype))

    def RawKernel(self, source, name, options):
        assert source == _CUDA and '--fmad=false' in options and '--ftz=false' in options
        def launch(grid, block, args):
            self.launches.append((name, grid, block))
            if name == 'forest_lane_valid':
                groups, output, batch, count = args
                output._array[:] = groups._array.all(axis=1)
                return
            arrays = [value._array for value in args[:15]]
            (lower, upper, cost, weights, starts, priority, cost_order, reps,
             new_lo, new_hi, new_cost, lw_out, uw_out, fallback, good) = arrays
            batch, n, groups = map(int, args[15:18])
            assert tuple(args[18:]) == tuple(s // 8 for v in args[:3] for s in v.strides)
            for lane in range(batch):
                for group in range(groups):
                    flat = lane * groups + group
                    begin, end = starts[flat:flat+2]
                    lo, hi, lw, uw, valid = -np.inf, np.inf, -1, -1, True
                    with np.errstate(all='ignore'):
                        for column in priority[begin:end]:
                            w, l, u = weights[lane, column], lower[lane, column], upper[lane, column]
                            a, b = (l, u) if w > 0 else (u, l)
                            dl, du = np.float64(a / w), np.float64(b / w)
                            valid = bool(valid and not np.isnan(l) and not np.isnan(u)
                                and l != np.inf and u != -np.inf and l <= u
                                and np.isfinite(w) and w != 0 and not np.isnan(dl) and not np.isnan(du)
                                and (not np.isfinite(a) or np.isfinite(dl))
                                and (not np.isfinite(b) or np.isfinite(du)))
                            if dl > lo: lo, lw = dl, column if np.isfinite(dl) else -1
                            elif dl == lo: lo = dl
                            if du < hi: hi, uw = du, column if np.isfinite(du) else -1
                            elif du == hi: hi = du
                        total = np.float64(0.)
                        for column in cost_order[begin:end]:
                            c = cost[lane, column]
                            product = np.float64(weights[lane, column] * c)
                            total = np.float64(total + product)
                            valid = bool(valid and np.isfinite(c) and np.isfinite(product) and np.isfinite(total))
                    new_lo[lane, group] = lo if valid else np.nan
                    new_hi[lane, group] = hi if valid else np.nan
                    new_cost[lane, group] = total if valid else np.nan
                    lw_out[lane, group], uw_out[lane, group] = (lw, uw) if valid else (-1, -1)
                    fallback[lane, group] = reps[lane, group]
                    good[lane, group] = valid and lo <= hi
        return launch


def apply_fake(plans, lower, upper, cost):
    cp = FakeCP()
    transform = DeviceForestBounds(plans, cp=cp)
    actual = transform.apply(*(DeviceArray(v) for v in (lower, upper, cost)))
    assert cp.uploads == 5 and len(cp.launches) == 2
    return tuple(v._array for v in actual), transform, cp


@pytest.mark.parametrize('seed', range(24))
def test_independent_lane_maps_dynamic_endpoints_witnesses_and_cost_match_cpu(seed):
    rng = np.random.default_rng(seed)
    n, lanes = 15, 3
    original, plans = [], []
    for lane in range(lanes):
        assignment = rng.permutation(np.repeat(np.arange(5), 3))
        weights = rng.choice([-1., 1.], n) * np.exp2(rng.integers(-4, 1, n))
        p = problem(weights, assignment)
        original.append(p)
        plans.append(HomogeneousEqualityReduction(p))
    for update in range(3):
        lower, upper, cost = [], [], []
        for plan in plans:
            w = plan.weights
            il, iu = -rng.integers(0, 8, n), rng.integers(0, 8, n)
            lo, hi = np.minimum(w*il, w*iu), np.maximum(w*il, w*iu)
            lo[rng.random(n) < .15], hi[rng.random(n) < .15] = -np.inf, np.inf
            lower.append(lo); upper.append(hi)
            cost.append(rng.integers(-10, 11, n).astype(float))
        lower, upper, cost = map(np.asarray, (lower, upper, cost))
        before = [v.copy() for v in (lower, upper, cost)]
        actual, _, _ = apply_fake(plans, lower, upper, cost)
        for lane, (plan, p) in enumerate(zip(plans, original)):
            reduced = plan.reduce((p[0], p[1], lower[lane], upper[lane], cost[lane], p[-1]))
            expected = (*reduced.problem[2:5], reduced.lower_witness, reduced.upper_witness,
                        reduced.fallback_witness)
            for result, reference in zip(actual[:6], expected):
                np.testing.assert_array_equal(result[lane], reference)
        assert actual[-1].tolist() == [True]*lanes
        for current, old in zip((lower, upper, cost), before):
            np.testing.assert_array_equal(current, old)


def test_tie_priority_is_not_cost_accumulation_order_and_free_witness_is_minus_one():
    plan = HomogeneousEqualityReduction(problem([.5, 1., -1., 1.], [0, 0, 0, 1]))
    w = plan.weights
    lower, upper = np.minimum(-2*w, 2*w), np.maximum(-2*w, 2*w)
    lower[-1], upper[-1] = -np.inf, np.inf
    cost = np.array([2e16, 1., 1e16, 3.])
    actual, transform, _ = apply_fake([plan], lower[None], upper[None], cost[None])
    assert actual[3].tolist() == actual[4].tolist() == [[1, -1]]
    # Original column sum is (1e16 + 1) - 1e16 = 0; priority order is different.
    assert actual[2].tolist() == [[0., 3.]]
    assert transform._layout.priority_columns.tolist() == [1, 2, 0, 3]
    assert transform._layout.cost_columns.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize('field,value', [('lower', np.nan), ('upper', np.nan),
    ('cost', np.nan), ('cost', np.inf), ('cost', -np.inf), ('lower', np.inf), ('upper', -np.inf)])
def test_nonfinite_invalid_input_flags_only_affected_lane_without_clipping(field, value):
    plan = HomogeneousEqualityReduction(problem([1., -1.]))
    values = dict(lower=np.full((2, 2), -1.), upper=np.ones((2, 2)), cost=np.zeros((2, 2)))
    values[field][0, 0] = value
    actual, _, _ = apply_fake([plan, plan], values['lower'], values['upper'], values['cost'])
    assert actual[-1].tolist() == [False, True]
    assert np.isnan(actual[0][0, 0]) and np.isnan(actual[2][0, 0])


def test_finite_endpoint_division_overflow_is_invalid_not_an_infinite_free_bound():
    plan = HomogeneousEqualityReduction(problem([.5, 1.]))
    actual, _, _ = apply_fake([plan], np.zeros((1, 2)),
        np.full((1, 2), np.finfo(float).max), np.zeros((1, 2)))
    assert not actual[-1][0] and np.isnan(actual[1][0, 0])


def test_objective_intermediate_overflow_is_invalid_even_if_inputs_finite():
    plan = HomogeneousEqualityReduction(problem([1., 1., 1.]))
    actual, _, _ = apply_fake([plan], np.zeros((1, 3)), np.ones((1, 3)),
        np.array([[1e308, 1e308, -1e308]]))
    assert not actual[-1][0] and np.isnan(actual[2][0, 0])


@pytest.mark.parametrize('delta', [np.nextafter(0., 1.), 1e-15, 1.])
def test_infeasible_intersection_is_not_tolerance_clipped(delta):
    plan = HomogeneousEqualityReduction(problem([1., 1.]))
    actual, _, _ = apply_fake([plan], np.array([[delta, -1.]]),
                            np.array([[2., 0.]]), np.zeros((1, 2)))
    assert actual[0][0, 0] == delta and actual[1][0, 0] == 0.
    assert not actual[-1][0]


def test_empty_original_bound_is_invalid_even_before_intersection():
    plan = HomogeneousEqualityReduction(problem([1.]))
    actual, _, _ = apply_fake([plan], np.array([[2.]]), np.array([[1.]]), np.array([[0.]]))
    assert not actual[-1][0]


def test_snapshots_are_owned_and_immutable_and_caller_plan_mutation_is_isolated():
    plan = HomogeneousEqualityReduction(problem([.5, 1., -1.]))
    cp = FakeCP()
    transform = DeviceForestBounds([plan], cp=cp)
    for name in ('weights', 'starts', 'priority_columns', 'cost_columns', 'representatives'):
        with pytest.raises(ValueError): getattr(transform._layout, name).flags.writeable = True
    with pytest.raises(FrozenInstanceError): transform._layout.batch = 9
    for label, actual, snapshot in transform.static_targets:
        assert label.startswith('forest_bounds_') and actual is not snapshot
        assert not np.shares_memory(actual._array, snapshot._array)
        np.testing.assert_array_equal(actual._array, snapshot._array)
    plan.weights[:] = np.nan
    plan.original_to_reduced[:] = 999
    plan.representatives[:] = 999
    result = transform.apply(DeviceArray(np.full((1, 3), -1.)),
        DeviceArray(np.ones((1, 3))), DeviceArray(np.ones((1, 3))))
    assert result[-1]._array.tolist() == [True]
    assert cp.uploads == 5


@pytest.mark.parametrize('mutation', ['weight', 'group', 'transform', 'layout', 'representative'])
def test_mutated_or_inconsistent_plan_is_rejected_at_construction(mutation):
    plan = HomogeneousEqualityReduction(problem([.5, 1., -1.]))
    if mutation == 'weight': plan.weights[0] = np.nan
    elif mutation == 'group': plan.original_to_reduced[0] = 9
    elif mutation == 'transform': plan.transform.data[0] *= -1.
    elif mutation == 'layout': plan.weights[0] = .125
    else: plan.representatives[0] = 0
    with pytest.raises(ValueError): DeviceForestBounds([plan], cp=FakeCP())


def test_constructor_rejects_unverified_empty_and_mixed_dimension_batches():
    plan = HomogeneousEqualityReduction(problem([1., 1.]))
    for plans in ([], [SimpleNamespace()], [plan, HomogeneousEqualityReduction(problem([1.]))]):
        with pytest.raises(ValueError): _snapshot_layout(plans)


@pytest.mark.parametrize('change', ['dtype', 'shape', 'input_device', 'current_device', 'stream', 'thread', 'host'])
def test_metadata_mismatch_rejected_before_any_launch(monkeypatch, change):
    import src.gpu_forest_numeric_bounds as module
    cp = FakeCP()
    transform = DeviceForestBounds([HomogeneousEqualityReduction(problem([1., 1.]))], cp=cp)
    arrays = [DeviceArray(np.ones((1, 2))) for _ in range(3)]
    if change == 'dtype': arrays[0] = DeviceArray(np.ones((1, 2), dtype=np.float32))
    elif change == 'shape': arrays[0] = DeviceArray(np.ones((2, 1)))
    elif change == 'input_device': arrays[0].device.id = 1
    elif change == 'current_device': cp.current_device = 1
    elif change == 'stream': cp.current_stream += 1
    elif change == 'thread': monkeypatch.setattr(module.threading, 'get_ident', lambda: -1)
    else: arrays[0] = np.ones((1, 2))
    with pytest.raises(ValueError): transform.apply(*arrays)
    assert not cp.launches


def test_strided_inputs_no_per_call_host_upload_or_scalar_download_and_owned_outputs():
    cp = FakeCP()
    plan = HomogeneousEqualityReduction(problem([1., -1., .5]))
    transform = DeviceForestBounds([plan, plan], cp=cp)
    lower = np.full((4, 6), -1.)[::2, ::-2]
    upper = np.asfortranarray(np.ones((2, 3)))
    cost = np.broadcast_to(np.array([1., 2., 3.]), (2, 3))
    inputs = tuple(DeviceArray(v) for v in (lower, upper, cost))
    first = transform.apply(*inputs)
    second = transform.apply(*inputs)
    assert cp.uploads == 5 and len(cp.launches) == 4
    assert first[-1]._array.all()
    for a, b in zip(first, second):
        assert not np.shares_memory(a._array, b._array)
    assert '__dmul_rn' in _CUDA and '__dadd_rn' in _CUDA and '__ddiv_rn' in _CUDA
    assert 'atomicAdd(' not in _CUDA


@pytest.mark.parametrize('change', ['layout_tuple', 'targets_tuple', 'pointer', 'shape',
    'dtype', 'stride', 'device', 'snapshot_pointer', 'snapshot_shape', 'batch', 'n', 'groups'])
def test_static_layout_preflight_rejects_replacements_before_comparison_or_launch(change):
    cp = FakeCP()
    transform = DeviceForestBounds([HomogeneousEqualityReduction(problem([1., 1., .5]))], cp=cp)
    value, snapshot = transform.static_targets[0][1:]
    if change == 'layout_tuple':
        original = transform._device_layout
        transform._device_layout = (*original[:2], original[2].copy(), *original[3:])
    elif change == 'targets_tuple': transform.static_targets = tuple(list(transform.static_targets))
    elif change == 'pointer': value.data.ptr += 8
    elif change == 'shape': value._array = value._array.reshape((3, 1))
    elif change == 'dtype': value._array = value._array.astype(np.int64)
    elif change == 'stride': value._array = value._array[:, ::-1]
    elif change == 'device': value.device.id = 1
    elif change == 'snapshot_pointer': snapshot.data.ptr += 8
    elif change == 'snapshot_shape': snapshot._array = snapshot._array.reshape((3, 1))
    elif change == 'batch': transform.batch += 1
    elif change == 'n': transform.original_variables += 1
    else: transform.reduced_variables += 1
    with pytest.raises(ValueError, match='[Ss]tatic forest'):
        transform._check_layout()
    arrays = [DeviceArray(np.ones((1, 3))) for _ in range(3)]
    with pytest.raises(ValueError): transform.apply(*arrays)
    assert not cp.launches and cp.uploads == 5


def _cuda():
    # Deliberately called only inside test_cuda_*; CPU-only selection cannot
    # load CUDA or initialize a context through collection of this module.
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('CUDA device required')
    return cp


@pytest.mark.parametrize('seed', [0, 17, 41])
@pytest.mark.parametrize('strided', [False, True])
def test_cuda_raw_kernel_matches_cpu_cost_bounds_and_witnesses(seed, strided):
    cp = _cuda()
    rng = np.random.default_rng(seed)
    batch, n = 3, 12
    originals, plans, lower, upper, costs = [], [], [], [], []
    for lane in range(batch):
        assignment = rng.permutation(np.repeat(np.arange(4), 3))
        p = problem(rng.choice([-1., 1.], n) * np.exp2(rng.integers(-4, 1, n)), assignment)
        plan = HomogeneousEqualityReduction(p)
        w = plan.weights
        lo, hi = np.minimum(-2*w, 2*w), np.maximum(-2*w, 2*w)
        lo[plan.original_to_reduced == 0], hi[plan.original_to_reduced == 0] = -np.inf, np.inf
        originals.append(p); plans.append(plan); lower.append(lo); upper.append(hi)
        costs.append(rng.integers(-10, 11, n).astype(float))
    lower, upper, costs = map(np.asarray, (lower, upper, costs))
    transform = DeviceForestBounds(plans, cp=cp)
    if strided:
        lower_device = cp.asarray(np.repeat(lower, 2, axis=1))[:, ::2]
        upper_device = cp.asarray(np.asfortranarray(upper))
        cost_device = cp.asarray(costs[:, ::-1].copy())[:, ::-1]
    else:
        lower_device, upper_device, cost_device = map(cp.asarray, (lower, upper, costs))
    transform._check_layout()
    result = transform.apply(lower_device, upper_device, cost_device)
    actual = [value.get() for value in result]
    assert actual[-1].tolist() == [True] * batch
    for lane, (p, plan) in enumerate(zip(originals, plans)):
        reduced = plan.reduce((p[0], p[1], lower[lane], upper[lane], costs[lane], p[-1]))
        reference = (*reduced.problem[2:5], reduced.lower_witness, reduced.upper_witness,
                     reduced.fallback_witness)
        for value, expected in zip(actual[:6], reference):
            np.testing.assert_array_equal(value[lane], expected)
    for _, value, snapshot in transform.static_targets:
        assert bool(cp.array_equal(value, snapshot))


def test_cuda_nonfinite_overflow_and_empty_intersection_return_lane_flags():
    cp = _cuda()
    plan = HomogeneousEqualityReduction(problem([.5, 1., 1.]))
    batch = 7
    lo, hi, costs = np.full((batch, 3), -1.), np.ones((batch, 3)), np.zeros((batch, 3))
    lo[1, 0] = np.nan
    hi[2, 0] = np.finfo(float).max  # Finite endpoint / 0.5 overflows.
    costs[3] = [1e308, 1e308, 1e308]
    lo[4, 0] = np.inf
    hi[5, 0] = -np.inf
    lo[6, 0], hi[6, 1] = np.nextafter(0., 1.), 0.
    transform = DeviceForestBounds([plan] * batch, cp=cp)
    actual = [value.get() for value in transform.apply(*map(cp.asarray, (lo, hi, costs)))]
    assert actual[-1].tolist() == [True, False, False, False, False, False, False]
    assert actual[0][6, 0] > actual[1][6, 0] == 0.
    assert np.isnan(actual[0][1:6]).all()


def test_cuda_replaced_static_array_is_rejected_before_kernel_launch():
    cp = _cuda()
    transform = DeviceForestBounds([HomogeneousEqualityReduction(problem([1., 1.]))], cp=cp)
    values = transform._device_layout
    transform._device_layout = (*values[:2], cp.asarray([999, 999], dtype=cp.int64), *values[3:])
    with pytest.raises(ValueError, match='layout'):
        transform._check_layout()
    with pytest.raises(ValueError):
        transform.apply(cp.zeros((1, 2)), cp.ones((1, 2)), cp.zeros((1, 2)))
