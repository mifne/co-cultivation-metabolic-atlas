"""Optional launch-fused FP64 microkernels for flexible GMRES.

This module does not select a Krylov candidate or change a stopping rule. It
only replaces scalar/vector work *within* the existing Arnoldi iteration.
Two sequential modified Gram--Schmidt passes are retained (not CGS2). The
caller must retain finite checks, explicit original-system residuals and its
best-candidate rule. The NumPy path is a small arithmetic reference, not a
CPU fallback for a CUDA input.
"""

from functools import lru_cache


_MGS2_CUDA = r'''
extern "C" __global__ void mgs2_inplace(
    const double* basis, double* work, double* h,
    const long long width, const int capacity, const int column)
{
    // A single cooperative block owns one environment. This permits a
    // sequential MGS pass without global synchronization between columns.
    const int lane = blockIdx.x;
    const int thread = threadIdx.x;
    const long long woff = (long long)lane * width;
    const long long boff = (long long)lane * capacity * width;
    const long long hoff = (long long)lane * capacity * (capacity - 1);
    __shared__ double partial[256];
    for (int pass = 0; pass < 2; ++pass) {
        for (int row = 0; row <= column; ++row) {
            const long long bstart = boff + (long long)row * width;
            double total = 0.0;
            for (long long index = thread; index < width; index += blockDim.x) {
                total += basis[bstart + index] * work[woff + index];
            }
            partial[thread] = total;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride; stride /= 2) {
                if (thread < stride) partial[thread] += partial[thread + stride];
                __syncthreads();
            }
            const double coefficient = partial[0];
            if (thread == 0) h[hoff + (long long)row * (capacity - 1) + column] += coefficient;
            for (long long index = thread; index < width; index += blockDim.x) {
                work[woff + index] -= coefficient * basis[bstart + index];
            }
            // The next dot product must see the just-updated vector; omitting
            // this barrier would incorrectly turn the method into a race.
            __syncthreads();
        }
    }
}
'''


_GIVENS_CUDA = r'''
// NVRTC provides device hypot/isfinite as builtins without host math headers.
extern "C" __global__ void givens_backsolve_inplace(
    double* h, double* cosine, double* sine, double* projected_rhs,
    const bool* active, double* coefficients, double* diagonal,
    const int batch, const int capacity, const int column)
{
    // Each environment owns only <= 128 scalar coordinates. The dependent
    // rotations/back substitution remain serial, with no block reductions.
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batch) return;
    const int budget = capacity - 1;
    const long long hoff = (long long)lane * capacity * budget;
    const long long roff = (long long)lane * budget;
    const long long poff = (long long)lane * capacity;
    const long long coff = (long long)lane * (column + 1);
    for (int row = 0; row < column; ++row) {
        const long long top = hoff + (long long)row * budget + column;
        const double upper = h[top];
        const double lower = h[top + budget];
        const double c = cosine[roff + row];
        const double s = sine[roff + row];
        h[top] = c * upper + s * lower;
        h[top + budget] = -s * upper + c * lower;
    }
    const long long top = hoff + (long long)column * budget + column;
    const double upper = h[top];
    const double lower = h[top + budget];
    const double d = hypot(upper, lower);
    diagonal[lane] = d;
    const bool valid = active[lane] && isfinite(d) && d != 0.0;
    const double c = valid ? upper / d : 0.0;
    const double s = valid ? lower / d : 0.0;
    cosine[roff + column] = c;
    sine[roff + column] = s;
    h[top] = d;
    h[top + budget] = 0.0;
    const double old_rhs = projected_rhs[poff + column];
    projected_rhs[poff + column] = c * old_rhs;
    projected_rhs[poff + column + 1] = -s * old_rhs;
    for (int row = column; row >= 0; --row) {
        double value = 0.0;
        if (valid) {
            double tail = 0.0;
            for (int j = row + 1; j <= column; ++j) {
                tail += h[hoff + (long long)row * budget + j] * coefficients[coff + j];
            }
            // A nonfinite earlier diagonal propagates to coefficients and
            // must be rejected by the caller's finite-candidate guard.
            value = (projected_rhs[poff + row] - tail)
                    / h[hoff + (long long)row * budget + row];
        }
        coefficients[coff + row] = value;
    }
}
'''


@lru_cache(maxsize=1)
def _mgs2_kernel():
    import cupy as cp
    # Separate multiply/subtract rounding matches the unfused vector updates.
    # Dot reduction ordering can differ from CuPy; this is not bitwise mode.
    return cp.RawKernel(_MGS2_CUDA, 'mgs2_inplace', options=('--fmad=false',))


@lru_cache(maxsize=1)
def _givens_kernel():
    import cupy as cp
    return cp.RawKernel(_GIVENS_CUDA, 'givens_backsolve_inplace',
                        options=('--fmad=false',))


def _array(value, shape, xp):
    if (not isinstance(value, xp.ndarray) or value.shape != shape
            or value.dtype != xp.float64 or not value.flags.c_contiguous):
        raise ValueError('Matching C-contiguous FP64 Krylov arrays required')


def mgs2_inplace(basis, work, triangular, column, *, xp):
    """Apply two *sequential* MGS passes, updating work and one H column.

    basis: [B, K+1, N]; work: [B, N]; triangular: [B, K+1, K]. The
    column is a Python int in 0..K-1. Work must already be masked for inactive
    lanes by the caller. No finiteness scan/synchronization is added here:
    nonfinite arithmetic propagates and must fail the caller's usual guards.

    CUDA uses one 256-thread block per environment and one kernel launch for
    all 2*(column+1) projections. This trades within-vector parallelism for
    fewer launches and therefore MUST be benchmarked at the actual B and N.
    It does not guarantee a speedup, especially for very wide vectors with a
    tiny batch. No basis approximation, dictionary or normal equations occur.
    """
    if (not isinstance(basis, xp.ndarray) or basis.ndim != 3
            or basis.shape[0] < 1 or basis.shape[1] < 2 or basis.shape[2] < 1):
        raise ValueError('Nonempty basis [B, K+1, N] with K >= 1 required')
    batch, capacity, width = basis.shape
    if (type(column) is not int or not 0 <= column < capacity - 1
            or capacity > 129):
        raise ValueError('An integer column within a Krylov budget <= 128 required')
    _array(basis, (batch, capacity, width), xp)
    _array(work, (batch, width), xp)
    _array(triangular, (batch, capacity, capacity - 1), xp)
    if (xp.may_share_memory(basis, work)
            or xp.may_share_memory(basis, triangular)
            or xp.may_share_memory(work, triangular)):
        raise ValueError('Basis, work and triangular storage must not overlap')
    if xp.__name__ == 'numpy':
        for _ in range(2):
            for row in range(column + 1):
                coefficient = xp.sum(basis[:, row] * work, axis=1)
                triangular[:, row, column] += coefficient
                work -= coefficient[:, None] * basis[:, row]
    elif xp.__name__ == 'cupy':
        device = work.device.id
        if (basis.device.id != device or triangular.device.id != device
                or xp.cuda.runtime.getDevice() != device):
            raise ValueError('Krylov arrays must share the current CUDA device')
        # RawKernel scalar arguments have explicit C-compatible widths.
        import numpy as np
        _mgs2_kernel()((batch,), (256,), (
            basis, work, triangular, np.int64(width),
            np.int32(capacity), np.int32(column)))
    else:
        raise ValueError('Only NumPy reference or CuPy device arithmetic is supported')


def givens_backsolve_inplace(triangular, cosine, sine, projected_rhs,
                             active, column, *, xp):
    """Return (coefficients, diagonal), fusing Givens and back substitution.

    The array shapes are H=[B,K+1,K], cosine/sine=[B,K], projected_rhs=[B,K+1],
    active=[B] bool. H, cosine, sine and projected_rhs are updated in place;
    active is read-only. Returned coefficients are FP64 [B,column+1], and
    diagonal is the FP64 [B] current hypot, including nonfinite/zero values.

    The caller MUST reproduce its existing nonfinite/singular termination
    using the returned diagonal before forming a candidate. Only initially
    active lanes with finite nonzero current diagonal are back-substituted;
    others return zero coefficients. Nonfinite earlier diagonals/overflow
    propagate and remain subject to the existing finite-candidate check.

    The hypot-based Givens sequence and triangular solve are unchanged; no
    normal equations or small-residual acceptance is introduced. Inactive
    scratch rotations may change, but inactive candidates must not be used.
    No callback, original-system residual or convergence decision occurs here.
    """
    if (not isinstance(triangular, xp.ndarray) or triangular.ndim != 3
            or triangular.shape[0] < 1 or triangular.shape[1] < 2
            or triangular.shape[2] != triangular.shape[1] - 1):
        raise ValueError('Nonempty triangular [B,K+1,K] storage required')
    batch, capacity, budget = triangular.shape
    if (type(column) is not int or not 0 <= column < budget or budget > 128):
        raise ValueError('An integer column within a Krylov budget <= 128 required')
    _array(triangular, (batch, capacity, budget), xp)
    _array(cosine, (batch, budget), xp)
    _array(sine, (batch, budget), xp)
    _array(projected_rhs, (batch, capacity), xp)
    if (not isinstance(active, xp.ndarray) or active.shape != (batch,)
            or active.dtype != xp.bool_ or not active.flags.c_contiguous):
        raise ValueError('C-contiguous boolean active mask [B] required')
    storage = (triangular, cosine, sine, projected_rhs, active)
    for index, value in enumerate(storage):
        if any(xp.may_share_memory(value, other) for other in storage[index + 1:]):
            raise ValueError('Givens scratch storage must not overlap')
    coefficients = xp.empty((batch, column + 1), dtype=xp.float64)
    if xp.__name__ == 'numpy':
        for row in range(column):
            upper = triangular[:, row, column].copy()
            lower = triangular[:, row + 1, column].copy()
            triangular[:, row, column] = cosine[:, row] * upper + sine[:, row] * lower
            triangular[:, row + 1, column] = -sine[:, row] * upper + cosine[:, row] * lower
        upper = triangular[:, column, column].copy()
        lower = triangular[:, column + 1, column].copy()
        diagonal = xp.hypot(upper, lower)
        valid = active & xp.isfinite(diagonal) & (diagonal != 0.)
        denominator = xp.where(valid, diagonal, 1.)
        cosine[:, column] = xp.where(valid, upper / denominator, 0.)
        sine[:, column] = xp.where(valid, lower / denominator, 0.)
        triangular[:, column, column] = diagonal
        triangular[:, column + 1, column] = 0.
        old_rhs = projected_rhs[:, column].copy()
        projected_rhs[:, column] = cosine[:, column] * old_rhs
        projected_rhs[:, column + 1] = -sine[:, column] * old_rhs
        for row in range(column, -1, -1):
            tail = xp.sum(triangular[:, row, row + 1:column + 1]
                          * coefficients[:, row + 1:column + 1], axis=1)
            denominator = xp.where(valid, triangular[:, row, row], 1.)
            coefficients[:, row] = xp.where(
                valid, (projected_rhs[:, row] - tail) / denominator, 0.)
    elif xp.__name__ == 'cupy':
        device = triangular.device.id
        if (any(value.device.id != device for value in storage)
                or xp.cuda.runtime.getDevice() != device):
            raise ValueError('Givens arrays must share the current CUDA device')
        import numpy as np
        diagonal = xp.empty(batch, dtype=xp.float64)
        _givens_kernel()(((batch + 127) // 128,), (128,), (
            triangular, cosine, sine, projected_rhs, active, coefficients, diagonal,
            np.int32(batch), np.int32(capacity), np.int32(column)))
    else:
        raise ValueError('Only NumPy reference or CuPy device arithmetic is supported')
    return coefficients, diagonal
