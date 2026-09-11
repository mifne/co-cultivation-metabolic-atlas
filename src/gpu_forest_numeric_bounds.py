"""Deterministic FP64 forest box/cost transforms, entirely on device per call.

Only an already-verified homogeneous forest coordinate map is snapshotted.
This does not validate changed equality operators or certify an LP solution;
the caller must maintain those contracts and consume the returned lane flags.
No input is clipped and no CPU optimizer/fallback or vector download is used.
"""
from dataclasses import dataclass
import threading

import numpy as np

from .lp_equality_reduction import HomogeneousEqualityReduction, _immutable_array


_CUDA = r'''
extern "C" __global__ void forest_bounds_cost(
    const double* lower, const double* upper, const double* cost,
    const double* weights, const long long* starts,
    const long long* priority_columns, const long long* cost_columns,
    const long long* representatives,
    double* new_lower, double* new_upper, double* new_cost,
    long long* lower_witness, long long* upper_witness,
    long long* fallback_witness, bool* group_valid,
    const long long batch, const long long n, const long long groups,
    const long long lower_s0, const long long lower_s1,
    const long long upper_s0, const long long upper_s1,
    const long long cost_s0, const long long cost_s1)
{
    const long long group = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= batch * groups) return;
    const long long lane = group / groups;
    const long long begin = starts[group], end = starts[group + 1];
    const double infinity = __longlong_as_double(0x7ff0000000000000LL);
    const double nan_value = __longlong_as_double(0x7ff8000000000000LL);
    double lo = -infinity, hi = infinity;
    long long lw = -1, uw = -1;
    bool valid = true;
    // Precompiled order: largest |weight|, then smallest original column.
    // Equal finite endpoints retain the FIRST witness in that order.
    for (long long entry = begin; entry < end; ++entry) {
        const long long column = priority_columns[entry];
        const double w = weights[lane * n + column];
        const double l = lower[lane * lower_s0 + column * lower_s1];
        const double u = upper[lane * upper_s0 + column * upper_s1];
        const bool input_valid = !isnan(l) && !isnan(u) && l != infinity
            && u != -infinity && l <= u && isfinite(w) && w != 0.0;
        const double a = w > 0.0 ? l : u;
        const double b = w > 0.0 ? u : l;
        const double induced_lo = __ddiv_rn(a, w);
        const double induced_hi = __ddiv_rn(b, w);
        // Finite endpoints must NEVER overflow into an apparently free box.
        valid = valid && input_valid && !isnan(induced_lo) && !isnan(induced_hi)
            && (!isfinite(a) || isfinite(induced_lo))
            && (!isfinite(b) || isfinite(induced_hi));
        if (induced_lo > lo) {
            lo = induced_lo;
            lw = isfinite(lo) ? column : -1;
        } else if (induced_lo == lo) {
            lo = induced_lo; // Preserve endpoint arithmetic, not an epsilon tie.
        }
        if (induced_hi < hi) {
            hi = induced_hi;
            uw = isfinite(hi) ? column : -1;
        } else if (induced_hi == hi) {
            hi = induced_hi;
        }
    }
    double total = 0.0;
    // NOT the witness priority order: T.T @ c sums in original-column order.
    // Explicit round-to-nearest multiply/add prohibits fused or atomic sums.
    for (long long entry = begin; entry < end; ++entry) {
        const long long column = cost_columns[entry];
        const double c = cost[lane * cost_s0 + column * cost_s1];
        const double product = __dmul_rn(weights[lane * n + column], c);
        total = __dadd_rn(total, product);
        valid = valid && isfinite(c) && isfinite(product) && isfinite(total);
    }
    const bool arithmetic_valid = valid;
    valid = valid && lo <= hi;
    // Poison invalid arithmetic instead of hiding NaN/overflow behind max/min.
    // For a finite but empty intersection, preserve the actual lo > hi pair.
    new_lower[group] = arithmetic_valid ? lo : nan_value;
    new_upper[group] = arithmetic_valid ? hi : nan_value;
    new_cost[group] = arithmetic_valid ? total : nan_value;
    lower_witness[group] = arithmetic_valid ? lw : -1;
    upper_witness[group] = arithmetic_valid ? uw : -1;
    fallback_witness[group] = representatives[group];
    group_valid[group] = valid;
}

extern "C" __global__ void forest_lane_valid(
    const bool* group_valid, bool* valid, const long long batch, const long long groups)
{
    const long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batch) return;
    bool good = true;
    for (long long group = 0; group < groups; ++group)
        good = good && group_valid[lane * groups + group];
    valid[lane] = good;
}
'''


@dataclass(frozen=True)
class _Layout:
    batch: int
    original_variables: int
    reduced_variables: int
    weights: np.ndarray
    starts: np.ndarray
    priority_columns: np.ndarray
    cost_columns: np.ndarray
    representatives: np.ndarray


@dataclass(frozen=True)
class _Context:
    cp: object
    device: int
    stream: int
    owner_thread: int
    batch: int
    original_variables: int
    reduced_variables: int


def _snapshot_layout(plans):
    """Constructor-only host validation; never retain mutable caller plans."""
    plans = tuple(plans)
    if not plans or any(not isinstance(plan, HomogeneousEqualityReduction) for plan in plans):
        raise ValueError('A nonempty batch of verified homogeneous forest plans is required')
    n, groups = plans[0].original_variables, plans[0].reduced_variables
    if (type(n) is not int or type(groups) is not int or not 0 < groups <= n
            or len(plans) * n >= 2**31):
        raise ValueError('Unsupported forest batch dimensions')
    all_weights, starts, priority, cost_order, representatives = [], [0], [], [], []
    positions = np.arange(n, dtype=np.int64)
    for plan in plans:
        if (plan.original_variables != n or plan.reduced_variables != groups
                or plan.original_shape[1] != n):
            raise ValueError('All lanes require the same original/reduced dimensions')
        group, weight = np.asarray(plan.original_to_reduced), np.asarray(plan.weights)
        if (group.shape != (n,) or group.dtype.kind not in 'iu'
                or weight.shape != (n,) or weight.dtype != np.float64
                or np.any(group < 0) or np.any(group >= groups)
                or not np.isfinite(weight).all() or np.any(weight == 0.)):
            raise ValueError('Invalid forest groups or finite nonzero FP64 weights')
        order = np.lexsort((positions, -np.abs(weight), group))
        ordered_group = group[order]
        offsets = np.r_[0, np.flatnonzero(np.diff(ordered_group)) + 1].astype(np.int64)
        if not np.array_equal(ordered_group[offsets], np.arange(groups)):
            raise ValueError('Each reduced coordinate must have original members')
        layout = plan._bound_layout
        for actual, expected in ((layout.groups, group), (layout.weights, weight),
            (layout.order, order), (layout.ordered_groups, ordered_group),
            (layout.starts, offsets), (layout.positions, positions), (layout.positive, weight > 0.)):
            if not np.array_equal(actual, expected):
                raise ValueError('The saved CPU bound layout does not match the current plan')
        for transform in (plan.T, plan.transform):
            if (getattr(transform, 'format', None) != 'csr' or transform.shape != (n, groups)
                    or transform.data.dtype != np.float64
                    or not np.array_equal(transform.indptr, np.arange(n + 1))
                    or not np.array_equal(transform.indices, group)
                    or not np.array_equal(transform.data, weight)):
                raise ValueError('Forest T must exactly match the one-weight-per-original-column map')
        reps = np.asarray(plan.representatives)
        expected_reps = np.full(groups, n, dtype=np.int64)
        unit = weight == 1.
        np.minimum.at(expected_reps, group[unit], positions[unit])
        if (reps.shape != (groups,) or reps.dtype.kind not in 'iu'
                or np.any(expected_reps == n) or not np.array_equal(reps, expected_reps)):
            raise ValueError('Forest fallback representatives do not match the verified unit weights')
        all_weights.append(weight.copy())
        lane_base = len(priority)
        starts.extend((offsets[1:] + lane_base).tolist())
        starts.append(lane_base + n)
        priority.extend(order.tolist())
        cost_order.extend(np.lexsort((positions, group)).tolist())
        representatives.append(reps.copy())
    return _Layout(len(plans), n, groups,
        _immutable_array(np.asarray(all_weights, dtype=np.float64)),
        _immutable_array(np.asarray(starts, dtype=np.int64)),
        _immutable_array(np.asarray(priority, dtype=np.int64)),
        _immutable_array(np.asarray(cost_order, dtype=np.int64)),
        _immutable_array(np.asarray(representatives, dtype=np.int64)))


class DeviceForestBounds:
    """Snapshot plans once, then apply GPU-only box/objective transformations.

    ``apply`` accepts exact FP64 [batch, original_variables] CUDA arrays,
    including strided views, on the constructor's device/stream/owner thread.
    It returns owned [batch, reduced_variables] lower/upper/cost and int64
    lower/upper/fallback witnesses, plus bool [batch] validity flags. Flags
    MUST gate subsequent use; they are not original-LP certificates.

    ``static_targets`` exposes (label, actual_device_array, owned_snapshot)
    triples for a caller's GPU-only integrity gate. No repeated H2D is needed.
    Changing caller plans later cannot change this object's private layout.
    """

    def __init__(self, plans, cp=None):
        self._layout = _snapshot_layout(plans)
        if cp is None:
            import cupy as cp
        self.cp = cp
        self.device = int(cp.cuda.runtime.getDevice())
        self.stream = int(cp.cuda.get_current_stream().ptr)
        self.owner_thread = threading.get_ident()
        self.batch = self._layout.batch
        self.original_variables = self._layout.original_variables
        self.reduced_variables = self._layout.reduced_variables
        self._context = _Context(cp, self.device, self.stream, self.owner_thread,
            self.batch, self.original_variables, self.reduced_variables)
        names = ('weights', 'starts', 'priority_columns', 'cost_columns', 'representatives')
        self._device_layout = tuple(cp.array(getattr(self._layout, name), copy=True) for name in names)
        self.static_targets = tuple((f'forest_bounds_{name}', value, value.copy())
                                    for name, value in zip(names, self._device_layout))
        self._original_device_layout = self._device_layout
        self._original_static_targets = self.static_targets
        self._static_metadata = tuple((self._array_metadata(actual), self._array_metadata(snapshot))
                                      for _, actual, snapshot in self.static_targets)
        options = ('--std=c++11', '--fmad=false', '--ftz=false')
        self._kernel = cp.RawKernel(_CUDA, 'forest_bounds_cost', options=options)
        self._flags_kernel = cp.RawKernel(_CUDA, 'forest_lane_valid', options=options)

    def _array_metadata(self, value):
        """Pointer/shape/stride inspection only; never examine device values."""
        if not isinstance(value, self._context.cp.ndarray):
            raise ValueError('Static forest layout requires its original CUDA arrays')
        try:
            pointer = int(value.data.ptr)
            metadata = (pointer, tuple(value.shape), value.dtype.str,
                        tuple(value.strides), int(value.device.id))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError('Malformed static forest CUDA array metadata') from error
        if pointer <= 0 or pointer % value.dtype.itemsize:
            raise ValueError('Invalid static forest CUDA allocation pointer')
        return metadata

    def _check_layout(self):
        """Preflight before a caller's static GPU comparison or kernel launch.

        This checks identities and metadata, NOT contents. A caller accepting
        mutation of private static arrays must additionally compare the actual
        arrays against ``static_targets`` snapshots on device before use.
        """
        context = self._context
        if (self.cp is not context.cp
                or (self.device, self.stream, self.owner_thread, self.batch,
                    self.original_variables, self.reduced_variables) !=
                   (context.device, context.stream, context.owner_thread, context.batch,
                    context.original_variables, context.reduced_variables)):
            raise ValueError('Static forest context or dimensions were replaced')
        if (threading.get_ident() != context.owner_thread
                or int(context.cp.cuda.runtime.getDevice()) != context.device
                or int(context.cp.cuda.get_current_stream().ptr) != context.stream):
            raise ValueError('Forest transform requires its original device, stream and owner thread')
        if (self._device_layout is not self._original_device_layout
                or self.static_targets is not self._original_static_targets
                or len(self._device_layout) != 5 or len(self.static_targets) != 5):
            raise ValueError('Static forest layout or integrity-target tuple was replaced')
        for value, target, expected in zip(self._device_layout, self.static_targets, self._static_metadata):
            _, actual, snapshot = target
            if (actual is not value or self._array_metadata(actual) != expected[0]
                    or self._array_metadata(snapshot) != expected[1]):
                raise ValueError('Static forest array identity, pointer, shape, dtype, stride or device changed')

    def apply(self, lower, upper, c):
        self._check_layout()
        cp = self.cp
        shape = (self.batch, self.original_variables)
        arrays = (lower, upper, c)
        for value in arrays:
            if (not isinstance(value, cp.ndarray) or value.shape != shape
                    or value.dtype != cp.float64 or value.device.id != self.device
                    or any(stride % 8 for stride in value.strides)):
                raise ValueError('Exact-shape FP64 CUDA arrays on the bound device are required')
        reduced_shape = (self.batch, self.reduced_variables)
        values = tuple(cp.empty(reduced_shape, dtype=cp.float64) for _ in range(3))
        witnesses = tuple(cp.empty(reduced_shape, dtype=cp.int64) for _ in range(3))
        group_valid = cp.empty(reduced_shape, dtype=cp.bool_)
        valid = cp.empty((self.batch,), dtype=cp.bool_)
        dimensions = tuple(np.int64(v) for v in (self.batch, self.original_variables, self.reduced_variables))
        strides = tuple(np.int64(stride // 8) for array in arrays for stride in array.strides)
        width = self.batch * self.reduced_variables
        self._kernel(((width + 127) // 128,), (128,),
            (*arrays, *self._device_layout, *values, *witnesses, group_valid, *dimensions, *strides))
        self._flags_kernel(((self.batch + 127) // 128,), (128,),
            (group_valid, valid, np.int64(self.batch), np.int64(self.reduced_variables)))
        return (*values, *witnesses, valid)
