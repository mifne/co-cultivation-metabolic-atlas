"""Reusable batched CUDA primal/dual maps for homogeneous equality forests.

Host construction validates and snapshots the algebraic reduction only; it
does not optimize an LP. Numeric compression and expansion stay on the bound
CUDA stream. Expansion is a proposal, not a certificate: the caller must check
the resulting pair against the unreduced original LP.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import threading
import time

import numpy as np
from scipy.sparse import csr_matrix
from .csr_block_assembly import block_diag

from .lp_equality_reduction import HomogeneousEqualityReduction, ReducedEqualityLP


def _readonly(value):
    if isinstance(value, csr_matrix):
        result = value.copy()
        for array in (result.data, result.indices, result.indptr):
            array.flags.writeable = False
    else:
        result = np.array(value, copy=True)
        result.flags.writeable = False
    return result


def _same_matrix(left, right):
    left = csr_matrix(left, copy=True)
    right = csr_matrix(right, copy=True)
    for value in (left, right):
        value.sum_duplicates()
        value.eliminate_zeros()
        value.sort_indices()
    return (left.shape == right.shape
            and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices)
            and np.array_equal(left.data, right.data))


def _same_problem(left, right):
    return (isinstance(left, (tuple, list)) and len(left) == 6
            and left[-1] == right[-1] and _same_matrix(left[0], right[0])
            and all(np.array_equal(a, b) for a, b in zip(left[1:5], right[1:5])))


@dataclass(frozen=True)
class _HostForestMaps:
    batch: int
    original_n: int
    original_m: int
    reduced_n: int
    reduced_m: int
    transform: csr_matrix
    compression: csr_matrix
    dual_lift: csr_matrix
    original_at: csr_matrix
    objective: np.ndarray
    kept_rows: np.ndarray
    eliminated_rows: np.ndarray
    lower_witness: np.ndarray
    upper_witness: np.ndarray
    fallback_witness: np.ndarray
    weights: np.ndarray


def _fingerprint(value):
    """Hash owned numerical metadata without rebuilding trees or Python lists."""
    digest = hashlib.sha256()
    def add(part):
        if isinstance(part, csr_matrix):
            add(('csr', part.shape, part.indptr, part.indices, part.data))
        elif isinstance(part, np.ndarray):
            digest.update(str((part.shape, part.dtype.str)).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        elif is_dataclass(part):
            for field in fields(part):
                add(field.name); add(getattr(part, field.name))
        elif isinstance(part, (tuple, list)):
            digest.update(str(len(part)).encode())
            for item in part: add(item)
        else:
            digest.update(repr(part).encode())
    add(value)
    return digest.hexdigest()


def _plan_key(plan):
    names = ('original_shape', 'original_variables', 'original_equalities',
        'reduced_variables', 'reduced_equalities', 'equality_fingerprint', 'max_scale_ratio',
        'T', 'transform', 'compression', 'dual_lift_map', 'kept_rows', 'eliminated_rows',
        'weights', 'representatives', 'original_to_reduced', '_eliminated_matrix')
    values = [(name, getattr(plan, name)) for name in names]
    if hasattr(plan, '_bound_layout'): values.append(('_bound_layout', plan._bound_layout))
    return _fingerprint(values)


def _prepare_host_maps(plans, reductions, *, _return_verified=False, reuse_equality_proofs=False):
    """Validate every environment before touching CUDA; own all stored data.

Plans/reductions are mutable legacy objects. Rebuilding the inexpensive tree
maps catches corrupted maps, mixed plans and stale bound witnesses. A plan
created for an earlier dynamic LP is valid when its equality map is unchanged;
the reduction must nevertheless match the *current* objective and bounds.
"""
    plans, reductions = tuple(plans), tuple(reductions)
    if not plans or len(plans) != len(reductions):
        raise ValueError('Nonempty equally sized plan and reduction batches required')
    if type(reuse_equality_proofs) is not bool:raise ValueError('Boolean proof reuse required')
    canonical={}
    if reuse_equality_proofs:
        from .lp_equality_reduction import prepare_independent_forest_plans
        groups={}
        for index,(plan,reduction) in enumerate(zip(plans,reductions)):
            if (not isinstance(plan,HomogeneousEqualityReduction)
                    or not isinstance(reduction,ReducedEqualityLP) or reduction.plan is not plan):
                raise ValueError('Each reduction must belong to its supplied forest plan')
            ratio=plan.max_scale_ratio
            if isinstance(ratio,(bool,np.bool_)) or not np.isscalar(ratio) or not np.isfinite(ratio) or ratio<1.:
                raise ValueError('Invalid forest scale ratio')
            groups.setdefault(float(ratio),[]).append(index)
        for ratio,indices in groups.items():
            fresh=prepare_independent_forest_plans([reductions[i].original_problem for i in indices],
                max_scale_ratio=ratio)
            canonical.update(zip(indices,fresh))
    verified = []
    for index,(plan, reduction) in enumerate(zip(plans, reductions)):
        if (not isinstance(plan, HomogeneousEqualityReduction)
                or not isinstance(reduction, ReducedEqualityLP)
                or reduction.plan is not plan):
            raise ValueError('Each reduction must belong to its supplied forest plan')
        fresh = canonical[index] if reuse_equality_proofs else HomogeneousEqualityReduction.from_problem(
            reduction.original_problem, max_scale_ratio=plan.max_scale_ratio)
        for name in ('original_shape', 'original_variables', 'original_equalities',
                     'reduced_variables', 'reduced_equalities', 'equality_fingerprint'):
            if getattr(plan, name) != getattr(fresh, name):
                raise ValueError(f'Forest plan metadata changed: {name}')
        for name in ('T', 'transform', 'compression', 'dual_lift_map'):
            if not _same_matrix(getattr(plan, name), getattr(fresh, name)):
                raise ValueError(f'Forest plan map changed: {name}')
        for name in ('kept_rows', 'eliminated_rows', 'weights', 'representatives',
                     'original_to_reduced'):
            if not np.array_equal(getattr(plan, name), getattr(fresh, name)):
                raise ValueError(f'Forest plan indices/weights changed: {name}')
        current = fresh.reduce(reduction.original_problem)
        if not _same_problem(reduction.problem, current.problem):
            raise ValueError('Reduced LP does not match its original LP and forest')
        if not _same_matrix(reduction.kept_matrix, current.kept_matrix):
            raise ValueError('Retained original rows do not match the reduction')
        for name in ('lower_witness', 'upper_witness', 'fallback_witness'):
            if not np.array_equal(getattr(reduction, name), getattr(current, name)):
                raise ValueError(f'Stale or invalid forest bound witness: {name}')
        verified.append(current)
    first = verified[0]
    original_shape, reduced_shape = first.original_problem[0].shape, first.problem[0].shape
    original_neq, reduced_neq = first.original_problem[-1], first.problem[-1]
    if any(r.original_problem[0].shape != original_shape
           or r.problem[0].shape != reduced_shape
           or r.original_problem[-1] != original_neq or r.problem[-1] != reduced_neq
           for r in verified):
        raise ValueError('Regroup forests with different original/reduced batch dimensions')
    original_m, original_n = original_shape
    reduced_m, reduced_n = reduced_shape
    trees = [r.plan for r in verified]
    sparse = {
        'transform': block_diag([p.T for p in trees], format='csr'),
        'compression': block_diag([p.compression for p in trees], format='csr'),
        'dual_lift': block_diag([p.dual_lift_map for p in trees], format='csr'),
        'original_at': block_diag([r.original_problem[0] for r in verified], format='csr').T.tocsr(),
    }
    arrays = {
        'objective': np.stack([r.original_problem[4] for r in verified]),
        'kept_rows': np.stack([p.kept_rows for p in trees]),
        'eliminated_rows': np.stack([p.eliminated_rows for p in trees]),
        'weights': np.stack([p.weights for p in trees]),
        **{name: np.stack([getattr(r, name) for r in verified])
           for name in ('lower_witness', 'upper_witness', 'fallback_witness')},
    }
    host = _HostForestMaps(len(plans), original_n, original_m, reduced_n, reduced_m,
                          **{name: _readonly(value) for name, value in {**sparse, **arrays}.items()})
    return (host, tuple(trees)) if _return_verified else host


class GpuForestMap:
    """Independent environment forests, reusable without an IPM workspace.

``compress`` returns reduced-coordinate warm-start proposals. ``expand`` lifts
primal and row-dual proposals, including signed bound witnesses. Inputs must
be exact-shape FP64 CUDA arrays; no input is modified. Nonfinite proposals
propagate to the caller's strict original-LP certificate (no host finite scan).
"""

    def __init__(self, plans, reductions, cp=None, *, reuse_equality_proofs=False):
        before = time.perf_counter()
        host, self._verified_plans = _prepare_host_maps(plans, reductions, _return_verified=True,
            reuse_equality_proofs=reuse_equality_proofs)
        self._verified_plan_keys = tuple(_plan_key(p) for p in self._verified_plans)
        self._host_key = _fingerprint(host)
        self.host_setup_seconds = time.perf_counter() - before
        if cp is None:
            import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as device_csr
        self.cp = cp
        self.device = cp.cuda.runtime.getDevice()
        self.stream = cp.cuda.get_current_stream()
        self.thread = threading.get_ident()
        self._host = host
        for name in ('batch', 'original_n', 'original_m', 'reduced_n', 'reduced_m'):
            setattr(self, name, getattr(host, name))
        for name in ('transform', 'compression', 'dual_lift', 'original_at'):
            setattr(self, '_' + name, device_csr(getattr(host, name)))
        self._transform_t = self._transform.T.tocsr()
        for name in ('objective', 'kept_rows', 'eliminated_rows', 'weights',
                     'lower_witness', 'upper_witness', 'fallback_witness'):
            setattr(self, '_' + name, cp.asarray(getattr(host, name)))
        self._batch_rows = cp.arange(self.batch)[:, None]
        # Record the native device dtypes once. CSR construction/transpose can
        # legitimately choose an index width different from a host CSR, so a
        # later update must compare with this verified device layout rather
        # than infer the expected width from the array being checked.
        self._device_dtypes = tuple(
            (name+'.'+part, getattr(getattr(self, '_'+name), part).dtype.str)
            for name in ('transform', 'compression', 'dual_lift', 'original_at', 'transform_t')
            for part in ('data', 'indices', 'indptr')) + tuple(
                (name, getattr(self, '_'+name).dtype.str)
                for name in ('objective', 'kept_rows', 'eliminated_rows', 'weights',
                             'lower_witness', 'upper_witness', 'fallback_witness', 'batch_rows'))
        self._device_dtypes_key = _fingerprint(self._device_dtypes)
        # Setup timing includes all device allocation/copy, not a hidden
        # synchronization repeated during numeric postsolve.
        self.stream.synchronize()
        self.setup_seconds = time.perf_counter() - before

    def rebind_copy(self, reductions):
        """Stage a new numeric map while sharing already verified static maps.

        The first constructor validated fresh canonical forests. Reuse their
        owned plans only if every map/coordinate fingerprint is unchanged.
        Recompute current reductions from their current original inputs and
        check the supplied reduced LP/witnesses independently. No tree search,
        CPU optimization, old numerical witness or old acceptance is reused.
        The old map is untouched, including on any preparation failure.
        """
        self._context()
        before = time.perf_counter()
        reductions = tuple(reductions)
        if any(getattr(self, name) != getattr(self._host, name) for name in
               ('batch', 'original_n', 'original_m', 'reduced_n', 'reduced_m')):
            raise ValueError('Forest map dimensions changed from their snapshot')
        if (len(reductions) != self.batch or _fingerprint(self._host) != self._host_key
                or _fingerprint(self._device_dtypes) != self._device_dtypes_key
                or tuple(_plan_key(p) for p in self._verified_plans) != self._verified_plan_keys):
            raise ValueError('Verified forest snapshot/map changed or batch is incomplete')
        verified = []
        for plan, reduction in zip(self._verified_plans, reductions):
            if not isinstance(reduction, ReducedEqualityLP):
                raise ValueError('Current forest reduction required')
            current = plan.reduce(reduction.original_problem)
            if (not _same_problem(current.problem, reduction.problem)
                    or not _same_matrix(current.kept_matrix, reduction.kept_matrix)):
                raise ValueError('Current reduced LP does not match the verified forest')
            for name in ('lower_witness', 'upper_witness', 'fallback_witness'):
                if not np.array_equal(getattr(current, name), getattr(reduction, name)):
                    raise ValueError('Current forest witness does not match: '+name)
            verified.append(current)
        host = replace(self._host,
            original_at=_readonly(block_diag([r.original_problem[0] for r in verified], format='csr').T.tocsr()),
            objective=_readonly(np.stack([r.original_problem[4] for r in verified])),
            **{name:_readonly(np.stack([getattr(r,name) for r in verified]))
               for name in ('lower_witness', 'upper_witness', 'fallback_witness')})
        # Detect altered device arrays before sharing them with another map.
        # Aggregate equality flags on device; only one scalar is downloaded.
        cp = self.cp
        from cupyx.scipy.sparse import csr_matrix as device_csr
        native_dtypes = dict(self._device_dtypes)
        checks = []

        def check_array(target, expected, name, *, index=False):
            native_dtype = np.dtype(native_dtypes[name])
            if (not isinstance(target, cp.ndarray)
                    or target.shape != expected.shape or target.dtype != native_dtype
                    or target.device.id != self.device
                    or (index and (native_dtype.kind != 'i' or native_dtype.itemsize not in (4, 8)))):
                raise ValueError('Forest device array type/shape/dtype/device changed: '+name)
            # In particular, array_equal by itself would accept floating-point
            # indices or an int32/int64 replacement with identical values.
            # Cast only the expected host data to the original native dtype.
            checks.append(cp.array_equal(target, cp.asarray(expected, dtype=native_dtype)))

        check_array(self._batch_rows, np.arange(self.batch, dtype=np.int64)[:, None],
                    'batch_rows', index=True)
        for name in ('transform', 'compression', 'dual_lift', 'original_at', 'transform_t'):
            target = getattr(self, '_'+name)
            expected = self._host.transform.T.tocsr() if name == 'transform_t' else getattr(self._host, name)
            if not isinstance(target, device_csr) or target.shape != expected.shape:
                raise ValueError('Forest device map type/shape changed: '+name)
            for part in ('data', 'indices', 'indptr'):
                check_array(getattr(target, part), getattr(expected, part), name+'.'+part,
                            index=part != 'data')
        for name in ('objective', 'kept_rows', 'eliminated_rows', 'weights',
                     'lower_witness', 'upper_witness', 'fallback_witness'):
            check_array(getattr(self, '_'+name), getattr(self._host, name), name,
                        index=name not in ('objective', 'weights'))
        if not bool(cp.all(cp.stack(checks))):
            raise ValueError('Forest device map data changed from its verified snapshot')
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result._host = host
        result._host_key = _fingerprint(host)
        result._original_at = device_csr(host.original_at)
        for name in ('objective', 'lower_witness', 'upper_witness', 'fallback_witness'):
            setattr(result, '_'+name, cp.asarray(getattr(host, name)))
        # Newly owned numeric arrays receive their actual construction dtype;
        # static arrays retain the already checked native layout. In normal
        # same-pattern updates these are identical to the previous snapshot.
        for part in ('data', 'indices', 'indptr'):
            native_dtypes['original_at.'+part] = getattr(result._original_at, part).dtype.str
        for name in ('objective', 'lower_witness', 'upper_witness', 'fallback_witness'):
            native_dtypes[name] = getattr(result, '_'+name).dtype.str
        result._device_dtypes = tuple(native_dtypes.items())
        result._device_dtypes_key = _fingerprint(result._device_dtypes)
        result.stream.synchronize()
        result.numeric_rebind_seconds = time.perf_counter()-before
        result.static_maps_reused = True
        return result

    def _context(self):
        if (threading.get_ident() != self.thread
                or self.cp.cuda.runtime.getDevice() != self.device
                or self.cp.cuda.get_current_stream().ptr != self.stream.ptr):
            raise RuntimeError('Forest map cannot change thread, CUDA device, or stream')

    def _array(self, value, width):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.shape != (self.batch, width)
                or value.dtype != cp.float64 or value.device.id != self.device):
            raise ValueError('Exact-shape FP64 arrays on the bound CUDA device required')

    def compress(self, full_x, full_y):
        """Original pair -> reduced warm-start proposals, CUDA operations only."""
        self._context()
        self._array(full_x, self.original_n)
        self._array(full_y, self.original_m)
        z = (self._compression @ full_x.ravel()).reshape(self.batch, self.reduced_n)
        y = full_y[self._batch_rows, self._kept_rows]
        return z, y

    def expand(self, reduced_x, reduced_y):
        """Reduced pair -> original pair; acceptance remains the caller's job."""
        self._context()
        self._array(reduced_x, self.reduced_n)
        self._array(reduced_y, self.reduced_m)
        cp = self.cp
        full_x = (self._transform @ reduced_x.ravel()).reshape(self.batch, self.original_n)
        full_y = cp.zeros((self.batch, self.original_m), dtype=cp.float64)
        full_y[self._batch_rows, self._kept_rows] = reduced_y
        q = self._objective.ravel() - self._original_at @ full_y.ravel()
        reduced_cost = (self._transform_t @ q).reshape(self.batch, self.reduced_n)
        witness = cp.where(reduced_cost >= 0., self._lower_witness, self._upper_witness)
        witness = cp.where(witness >= 0, witness, self._fallback_witness)
        normal = cp.zeros((self.batch, self.original_n), dtype=cp.float64)
        normal[self._batch_rows, witness] = (
            reduced_cost / self._weights[self._batch_rows, witness])
        if self._eliminated_rows.shape[1]:
            removed_y = self._dual_lift @ (q - normal.ravel())
            full_y[self._batch_rows, self._eliminated_rows] = removed_y.reshape(self.batch, -1)
        return full_x, full_y


__all__ = ['GpuForestMap']
