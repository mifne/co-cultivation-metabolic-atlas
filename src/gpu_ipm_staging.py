"""Opt-in numeric-update staging with one compact device validation read.

All old numerical payloads are checked, including unchanged payloads. The
replacement is staged, never committed here. This exchanges many D2H vector
reads for H2D reference copies and GPU comparisons; it is not a proven speedup.
"""

import numpy as np


_DYNAMIC = {('values',), ('factor', 'values')}
_DTYPES = tuple(np.dtype(value) for value in (np.bool_, np.int32, np.int64, np.float64))


def _target(solver, path):
    value = solver
    for part in path:
        value = value[part] if isinstance(part, int) else getattr(value, part)
    return value


def _host_array(value, name, *, finite=False):
    from .gpu_ipm_numeric_update import NumericRebindRejected
    value = np.asarray(value)
    if value.dtype not in _DTYPES:
        raise NumericRebindRejected('Unsupported numeric payload dtype: '+name)
    if value.dtype.kind == 'f' and (np.isnan(value).any()
            or (finite and not np.isfinite(value).all())):
        raise NumericRebindRejected('Invalid nonfinite numeric payload: '+name)
    return value


def _host_cast_exact(value, dtype, name):
    """New values must not overflow/truncate when staged to target storage."""
    from .gpu_ipm_numeric_update import NumericRebindRejected
    dtype = np.dtype(dtype)
    if dtype not in _DTYPES or dtype.kind != value.dtype.kind:
        raise NumericRebindRejected('Changed/incompatible payload dtype: '+name)
    if dtype.kind == 'i' and value.size:
        limit = np.iinfo(dtype)
        if value.min() < limit.min or value.max() > limit.max:
            raise NumericRebindRejected('Payload integer cast would overflow: '+name)
    converted = value.astype(dtype, copy=False)
    if not np.array_equal(converted, value):
        raise NumericRebindRejected('Numeric payload is not exactly representable: '+name)
    return converted


def stage_numeric_updates(solver, old_payload, new_payload):
    """Return ``(copies, skipped)`` without changing any existing buffer.

    Each copy is ``(name, target_array, staged_source_array)``, matching the
    existing numeric-update commit loop. The solver must retain its prepared
    coordinate maps and own the current device/stream/thread for the complete
    operation. Caller must not mutate the host payloads or device targets
    concurrently, and must retain these returned objects until commit drains.

    CSR shape/index/data and every static vector are checked against the old
    host payload. Only the two dynamic Newton/factor values arrays skip old
    value equality, as they contain the preceding factorization. Their shape,
    dtype/device and replacement values are still checked. Vector +/-inf is
    legal (e.g. certificate bounds); NaN never compares equal. CSR values and
    replacement KKT values must be finite. Int32/int64 index conversion is
    allowed only when exact, and floating payloads must remain FP64.

    CuPy performs all comparisons on device and reads one boolean status
    vector. NumPy is an explicit test/reference path, never a CUDA fallback.
    Value errors are precommit NumericRebindRejected; runtime/resource errors
    retain their type. Pending device work is drained before local references
    can be released on failure. No LP is solved and no acceptance is implied.
    """
    from .gpu_ipm_numeric_update import NumericRebindRejected
    try:
        return _stage_numeric_updates(solver, old_payload, new_payload)
    except NumericRebindRejected:
        raise
    except ValueError as error:
        raise NumericRebindRejected(str(error)) from error


def _stage_numeric_updates(solver, old_payload, new_payload):
    from .gpu_ipm_numeric_update import NumericRebindRejected
    xp = solver.cp
    if getattr(xp, '__name__', None) not in ('numpy', 'cupy'):
        raise NumericRebindRejected('Only NumPy reference or CuPy staging is supported')
    solver.factor._context()
    cuda = xp.__name__ == 'cupy'
    if cuda and (xp.cuda.runtime.getDevice() != solver.factor.device
            or xp.cuda.get_current_stream().ptr != solver.factor.stream.ptr):
        raise NumericRebindRejected('Numeric staging requires the bound CUDA device/stream')
    old_sparse, old_arrays = old_payload
    new_sparse, new_arrays = new_payload
    if old_sparse.keys() != new_sparse.keys() or old_arrays.keys() != new_arrays.keys():
        raise NumericRebindRejected('Prepared operator/certificate payload set changed')

    checks, replacements = [], []
    skipped = 0

    def validate_target(target, expected, name):
        if (not isinstance(target, xp.ndarray) or target.shape != expected.shape
                or np.dtype(target.dtype) not in _DTYPES
                or np.dtype(target.dtype).kind != expected.dtype.kind
                or (cuda and target.device.id != solver.factor.device)):
            raise NumericRebindRejected('Device payload shape/dtype/device changed: '+name)

    # Complete host metadata/representability checks before enqueueing work.
    for path, new_matrix in new_sparse.items():
        old_matrix, target = old_sparse[path], _target(solver, path)
        name = str(path)
        if old_matrix.shape != new_matrix.shape or target.shape != new_matrix.shape:
            raise NumericRebindRejected('Sparse operator shape changed: '+name)
        for attr in ('indptr', 'indices', 'data'):
            old = _host_array(getattr(old_matrix, attr), name+'.'+attr, finite=True)
            new = _host_array(getattr(new_matrix, attr), name+'.'+attr, finite=True)
            buffer = getattr(target, attr)
            validate_target(buffer, old, name+'.'+attr)
            if old.shape != new.shape or (attr != 'data' and not np.array_equal(old, new)):
                raise NumericRebindRejected('Sparse CSR coordinates changed: '+name+'.'+attr)
            _host_cast_exact(new, buffer.dtype, name+'.'+attr)
            checks.append((name+'.'+attr, buffer, old))
            if attr == 'data':
                if np.array_equal(old, new): skipped += 1
                else: replacements.append((name+'.data', buffer, new))
    for path, value in new_arrays.items():
        name, target = str(path), _target(solver, path)
        old = _host_array(old_arrays[path], name, finite=path in _DYNAMIC)
        new = _host_array(value, name, finite=path in _DYNAMIC)
        validate_target(target, old, name)
        if old.shape != new.shape:
            raise NumericRebindRejected('Prepared vector shape changed: '+name)
        _host_cast_exact(new, target.dtype, name)
        if path not in _DYNAMIC:
            checks.append((name, target, old))
        if path not in _DYNAMIC and np.array_equal(old, new): skipped += 1
        else: replacements.append((name, target, new))

    reference_buffers, statuses, copies = [], [], []
    enqueued = False
    try:
        for _name, target, old in checks:
            enqueued = True
            # Preserve reference precision, not a lossy cast to target dtype.
            reference = xp.asarray(old)
            reference_buffers.append(reference)
            statuses.append(xp.all(target == reference))
        if statuses:
            status = xp.stack(statuses)
            observed = xp.asnumpy(status) if cuda else np.array(status, copy=True)
            bad = np.flatnonzero(~observed)
            if len(bad):
                raise NumericRebindRejected('Stale current operator/certificate payload: '+checks[int(bad[0])][0])
        for name, target, new in replacements:
            enqueued = True
            source = xp.asarray(_host_cast_exact(new, target.dtype, name))
            if not cuda:
                source = source.copy()
            if (source.shape != target.shape or source.dtype != target.dtype
                    or (cuda and source.device.id != solver.factor.device)):
                raise NumericRebindRejected('Staging mismatch: '+name)
            copies.append((name, target, source))
        solver.factor.stream.synchronize()
        solver.factor._context()
        return copies, skipped
    except BaseException as error:
        if enqueued:
            try: solver.factor.stream.synchronize()
            except BaseException as draining:
                error.add_note(f'Numeric staging stream drain also failed: {draining}')
        if isinstance(error, ValueError) and not isinstance(error, NumericRebindRejected):
            raise NumericRebindRejected(str(error)) from error
        raise
