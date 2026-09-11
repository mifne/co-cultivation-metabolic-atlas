"""Declared-order FP64 segmented sums for static GPU linear maps.

Each output is a serial left fold starting at positive zero. Multiplication
and addition round separately, in the supplied entry order: no atomics,
parallel reduction, sorting, duplicate merging, or FMA contraction occurs.
This is an arithmetic primitive, not an LP solver or a certificate. Callers
must reject nonfinite results through their normal numerical guards.

``cp=np`` is an explicit CPU reference implementation for small tests. CUDA
inputs never fall back to NumPy. ``apply`` launches on the bound stream and
does not download data, synchronize, or scan input/output finiteness on host.
"""

import hashlib
import threading

import numpy as np


_CUDA = r'''
extern "C" __global__ void ordered_segmented_linear(
    const long long* indptr, const long long* indices,
    const double* weights, const double* values,
    double* output, const long long output_size)
{
    const long long segment = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= output_size) return;
    double total = 0.0;
    for (long long entry = indptr[segment]; entry < indptr[segment + 1]; ++entry) {
        const double product = __dmul_rn(weights[entry], values[indices[entry]]);
        total = __dadd_rn(total, product);
    }
    output[segment] = total;
}
'''


def _immutable(value):
    """Own contiguous, bytes-backed data that cannot be made writeable."""
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


def _host_map(value, dtype, name):
    if (not isinstance(value, np.ndarray) or value.ndim != 1
            or value.dtype != np.dtype(dtype)):
        raise ValueError(name+' must be a one-dimensional host '+np.dtype(dtype).name+' array')
    return _immutable(value)


def _fingerprint(input_size, snapshots):
    digest = hashlib.sha256()
    digest.update(np.asarray([input_size], dtype='<i8').tobytes())
    for label, array in zip(('indptr', 'indices', 'weights'), snapshots):
        digest.update(label.encode('ascii'))
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


class DeviceSegmentedLinear:
    """Own a static declared-order linear map and its inspectable snapshots.

    ``indptr`` and ``indices`` are flat host int64 arrays; ``weights`` is a
    finite flat host float64 array. Empty segments and repeated/unsorted input
    indices are legal and remain in exactly the declared order. ``input_size``
    must be a positive built-in int within signed int64 range.

    Host snapshots are physically immutable. CuPy does not expose a read-only
    device-memory guarantee: device maps are private, independently owned and
    logically immutable. ``static_targets`` exposes them only for the parent
    transaction's integrity audit, never for mutation. Each target has a
    separately owned expected snapshot on the same backend, so that audit
    needs no map upload. ``validate_integrity``
    explicitly verifies their values (one CUDA scalar read). ``apply`` checks
    metadata/ownership only, not map contents; it adds no hidden device read.
    """

    def __init__(self, indptr, indices, weights, input_size, cp=None):
        if type(input_size) is not int or not 0 < input_size <= np.iinfo(np.int64).max:
            raise ValueError('input_size must be a positive built-in int within int64 range')
        indptr = _host_map(indptr, np.int64, 'indptr')
        indices = _host_map(indices, np.int64, 'indices')
        weights = _host_map(weights, np.float64, 'weights')
        nnz = len(indices)
        if (len(indptr) < 1 or len(weights) != nnz
                or indptr[0] != 0 or indptr[-1] != nnz
                or np.any(indptr < 0) or np.any(indptr > nnz)
                or np.any(indptr[1:] < indptr[:-1])):
            raise ValueError('indptr must be valid nondecreasing segment offsets ending at nnz')
        if np.any(indices < 0) or np.any(indices >= input_size):
            raise ValueError('Segment input index outside input_size')
        if not np.isfinite(weights).all():
            raise ValueError('Segment weights must be finite FP64 values')
        if cp is None:
            import cupy as cp
        if cp is not np:
            if getattr(cp, '__name__', None) != 'cupy':
                raise ValueError('Only CuPy or the explicit NumPy reference backend is supported')
            import cupy
            if cp is not cupy:
                raise ValueError('The actual CuPy module is required')
        self.cp = cp
        self._backend = cp
        self._input_size = input_size
        self._output_size = len(indptr)-1
        self._sizes = (self._input_size, self._output_size)
        self._snapshots = (indptr, indices, weights)
        self._host_ids = tuple(id(value) for value in self._snapshots)
        self._proof_hash = _fingerprint(input_size, self._snapshots)
        self.thread = threading.get_ident()
        self._cuda = cp is not np
        if self._cuda:
            self.device = cp.cuda.runtime.getDevice()
            self.stream = cp.cuda.get_current_stream()
            self._targets = tuple(cp.asarray(value) for value in self._snapshots)
            self._expected_targets = tuple(cp.asarray(value) for value in self._snapshots)
            self._kernel = cp.RawKernel(_CUDA, 'ordered_segmented_linear',
                                       options=('--fmad=false', '--ftz=false'))
        else:
            self.device = None
            self.stream = None
            self._targets = tuple(_immutable(value) for value in self._snapshots)
            self._expected_targets = tuple(_immutable(value) for value in self._snapshots)
            self._kernel = None
        self._target_ids = tuple(id(value) for value in self._targets)
        self._target_pointers = tuple(self._pointer(value) for value in self._targets)
        self._expected_ids = tuple(id(value) for value in self._expected_targets)
        self._expected_pointers = tuple(self._pointer(value) for value in self._expected_targets)

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def proof_hash(self):
        return self._proof_hash

    @property
    def static_targets(self):
        """Fresh (label, target, owned same-backend expected snapshot) list.

        CuPy target/snapshot pairs can be compared entirely on-device. Both
        are logically immutable; the snapshot never aliases its target.
        """
        return list(zip(('indptr', 'indices', 'weights'), self._targets, self._expected_targets))

    @property
    def host_snapshots(self):
        """Immutable host metadata for constructor/integrity diagnostics."""
        return list(zip(('indptr', 'indices', 'weights'), self._snapshots))

    def _pointer(self, value):
        return int(value.data.ptr if self._cuda else value.__array_interface__['data'][0])

    def _context(self):
        if self.cp is not self._backend or threading.get_ident() != self.thread:
            raise RuntimeError('Segmented linear map cannot change backend or thread')
        if self._cuda and (self.cp.cuda.runtime.getDevice() != self.device
                or self.cp.cuda.get_current_stream().ptr != self.stream.ptr):
            raise RuntimeError('Segmented linear map cannot change CUDA device or stream')
        if (type(self._input_size) is not int or type(self._output_size) is not int
                or (self._input_size, self._output_size) != self._sizes
                or self._output_size != len(self._snapshots[0])-1
                or tuple(id(value) for value in self._snapshots) != self._host_ids
                or len(self._targets) != 3 or len(self._expected_targets) != 3):
            raise ValueError('Segmented linear host map metadata changed')
        for targets, identities, pointers in (
                (self._targets, self._target_ids, self._target_pointers),
                (self._expected_targets, self._expected_ids, self._expected_pointers)):
            for index, (target, snapshot) in enumerate(zip(targets, self._snapshots)):
                if (not isinstance(target, self.cp.ndarray) or target.shape != snapshot.shape
                        or target.dtype != snapshot.dtype or not target.flags.c_contiguous
                        or id(target) != identities[index]
                        or self._pointer(target) != pointers[index]
                        or (self._cuda and target.device.id != self.device)):
                    raise ValueError('Segmented linear target shape/dtype/device/ownership changed')

    def validate_integrity(self):
        """Explicit audit; unlike apply, this may read one device scalar."""
        self._context()
        if _fingerprint(self._input_size, self._snapshots) != self._proof_hash:
            raise ValueError('Segmented linear immutable host snapshot/hash changed')
        if self._cuda:
            flags = [self.cp.array_equal(target, expected)
                     & self.cp.array_equal(expected, self.cp.asarray(snapshot))
                     for target, expected, snapshot in
                     zip(self._targets, self._expected_targets, self._snapshots)]
            valid = bool(self.cp.all(self.cp.stack(flags)))
        else:
            valid = all(np.array_equal(target, snapshot) and np.array_equal(expected, snapshot)
                        for target, expected, snapshot in
                        zip(self._targets, self._expected_targets, self._snapshots))
        if not valid:
            raise ValueError('Segmented linear device map differs from its owned snapshot')
        return True

    def apply(self, values):
        """Return a new flat FP64 output; nonfinite arithmetic propagates."""
        self._context()
        cp = self.cp
        if (not isinstance(values, cp.ndarray) or values.shape != (self._input_size,)
                or values.dtype != np.dtype(np.float64) or not values.flags.c_contiguous
                or (self._cuda and values.device.id != self.device)):
            raise ValueError('Flat contiguous FP64 values on the bound backend/device required')
        output = cp.empty(self._output_size, dtype=cp.float64)
        if self._cuda:
            if self._output_size:
                self._kernel(((self._output_size+127)//128,), (128,),
                    (*self._targets, values, output, np.int64(self._output_size)))
        else:
            indptr, indices, weights = self._targets
            # NumPy scalar operations provide a separate binary64 rounding at
            # each multiplication and addition, not BLAS/np.sum accumulation.
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                for segment in range(self._output_size):
                    total = np.float64(0.)
                    for entry in range(int(indptr[segment]), int(indptr[segment+1])):
                        product = np.multiply(weights[entry], values[indices[entry]], dtype=np.float64)
                        total = np.add(total, product, dtype=np.float64)
                    output[segment] = total
        return output


__all__ = ['DeviceSegmentedLinear']
