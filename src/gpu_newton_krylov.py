"""Bounded FP64 flexible right-preconditioned GMRES for independent systems.

The operator always represents the original Newton equations. A regularized
factorization may be supplied as ``precondition``; it never changes the target
equations or the residual used to accept a direction. NumPy is supported for
small deterministic tests, and CuPy runs the same arithmetic on the device.
"""

import math
from contextlib import contextmanager
import threading
from types import MappingProxyType


GMRES_CONVERGED = 0
GMRES_ITERATION_LIMIT = 1
GMRES_BREAKDOWN = 2
GMRES_NONFINITE = 3


class GmresWorkspace:
    """Reusable owned FP64 Arnoldi storage for one exact batch/width/budget.

    This reuses the large basis, preconditioned directions and Hessenberg
    allocations, not final answers or diagnostics. Every lease clears all six
    owned buffers before use; an exception still releases the non-reentrant
    lease. CuPy storage is bound to the creating CUDA device and stream. A
    later call on that same stream is ordered after earlier device work even
    if it was asynchronous when the previous Python call returned.

    ``buffers`` is a read-only mapping for inspection, not caller scratch.
    Callbacks must not mutate its arrays. Inputs must not alias these arrays.
    Fixed addresses alone do NOT imply CUDA graph capture support: the solver
    still performs its existing host checks and temporary allocations.
    """

    def __init__(self, shape, *, xp, max_iterations=12):
        if (not isinstance(shape, (tuple, list)) or len(shape) != 2
                or any(type(v) is not int or v < 1 for v in shape)):
            raise ValueError('Workspace shape must contain positive integer batch and width')
        if type(max_iterations) is not int or not 0 <= max_iterations <= 128:
            raise ValueError('Workspace integer budget in 0..128 required')
        if getattr(xp, '__name__', None) not in ('numpy', 'cupy'):
            raise ValueError('Workspace supports the NumPy or CuPy namespace only')
        self._xp = xp
        self._shape = tuple(shape)
        self._max_iterations = max_iterations
        self._budget = min(max_iterations, self._shape[1])
        self._device = self._stream = None
        if xp.__name__ == 'cupy':
            self._device = xp.cuda.runtime.getDevice()
            self._stream = xp.cuda.get_current_stream().ptr
        self._lock = threading.Lock()
        self._use_count = 0
        batch, width = self._shape
        k = self._budget
        self._shapes = dict(basis=(batch, k + 1, width), directions=(batch, k, width),
            triangular=(batch, k + 1, k), cosine=(batch, k), sine=(batch, k),
            projected_rhs=(batch, k + 1))
        self._buffers = {name: xp.empty(dimensions, dtype=xp.float64)
                         for name, dimensions in self._shapes.items()}
        self._owners = dict(self._buffers)
        self._addresses = {name: self._address(value) for name, value in self._buffers.items()}

    def _address(self, value):
        return value.data.ptr if self._xp.__name__ == 'cupy' else value.ctypes.data

    @property
    def shape(self):
        return self._shape

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def budget(self):
        return self._budget

    @property
    def buffers(self):
        return MappingProxyType(self._buffers)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self._buffers.values())

    @property
    def use_count(self):
        return self._use_count

    def _validate(self, rhs, initial, *, xp, max_iterations):
        if xp is not self._xp:
            raise ValueError('GMRES workspace array namespace must match exactly')
        if type(max_iterations) is not int or max_iterations != self._max_iterations:
            raise ValueError('GMRES workspace requested budget must match exactly')
        if self._device is not None:
            if xp.cuda.runtime.getDevice() != self._device:
                raise ValueError('GMRES workspace must use its bound CUDA device')
            if xp.cuda.get_current_stream().ptr != self._stream:
                raise ValueError('GMRES workspace must use its bound CUDA stream')
        if tuple(self._buffers) != tuple(self._shapes):
            raise ValueError('GMRES workspace buffers were modified')
        for name, value in self._buffers.items():
            if (value is not self._owners[name] or not isinstance(value, xp.ndarray)
                    or value.shape != self._shapes[name] or value.dtype != xp.float64
                    or not value.flags.c_contiguous or self._address(value) != self._addresses[name]
                    or (self._device is None and not value.flags.writeable)
                    or (self._device is not None and value.device.id != self._device)):
                raise ValueError('GMRES workspace owned buffer metadata or address changed')
        for value in (rhs, initial):
            if (not isinstance(value, xp.ndarray) or value.shape != self._shape
                    or value.dtype != xp.float64
                    or (self._device is not None and value.device.id != self._device)):
                raise ValueError('GMRES workspace inputs require matching FP64 shape/device')
            if any(xp.may_share_memory(value, buffer) for buffer in self._buffers.values()):
                raise ValueError('GMRES inputs must not alias workspace buffers')

    @contextmanager
    def _lease(self, rhs, initial, *, xp, max_iterations):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('GMRES workspace is already in use; concurrent/reentrant reuse is forbidden')
        try:
            self._validate(rhs, initial, xp=xp, max_iterations=max_iterations)
            for value in self._buffers.values():
                value.fill(0.)
            self._use_count += 1
            yield self._buffers
        finally:
            self._lock.release()


def _norm2(vectors, xp):
    """Scaled row norms avoid squaring large or underflowing small entries."""
    scale = xp.max(xp.abs(vectors), axis=1)
    denominator = xp.where(scale > 0., scale, 1.)
    return scale * xp.sqrt(xp.sum((vectors / denominator[:, None]) ** 2, axis=1))


def _batched_gmres_impl(rhs, initial, matvec, precondition, error_measure, *, xp,
                  max_iterations=12, tolerance=1e-8,microkernels='none',
                  defer_lane_checks=False, _workspace_buffers=None):
    """Improve ``initial`` for ``matvec(x) = rhs`` independently per batch row.

    All vectors are finite FP64 ``xp.ndarray`` objects of shape ``[B, N]``.
    ``matvec`` and ``precondition`` take and return vectors of that shape and
    must operate independently on each row. Inactive rows receive zero input.
    ``error_measure(rhs, residual)`` returns nonnegative FP64 errors ``[B]``;
    acceptance uses that measure on an explicitly recomputed original-system
    residual, never the small least-squares residual. The preconditioner can
    vary between calls: the actual preconditioned vectors are retained.

    Return ``(best, diagnostics)``. ``best`` is the initial vector or the finite
    candidate with the smallest measured error, separately for each row. All
    diagnostic values remain arrays on the selected device: ``error`` is its
    measured error; ``iterations`` counts Arnoldi attempts; ``termination``
    contains the GMRES_* constants above; ``converged``, ``breakdown``, and
    ``nonfinite`` are boolean masks. Nonfinite callback results explicitly
    terminate the affected row and preserve its previous best vector.

    The unrestarted budget is 0..128 and is capped at N. Two-pass modified
    Gram-Schmidt and incremental Givens rotations avoid normal equations.
    Vector data never move to the host; only scalar validation and active-row
    checks synchronize. No optimizer or alternate solve is called.

    With ``defer_lane_checks=True``, lane rejection stays on the device until
    the next Arnoldi-loop boundary. This removes three intermediate active-row
    host checks per attempted iteration; finite-value guards, true residuals,
    negative-error validation, acceptance and iteration counts are unchanged.
    Callbacks must be independent, pure per-row numerical operations; vector
    callbacks must accept zero input for inactive rows. The remaining matvec
    calls in an iteration can receive all-zero batches after the last active
    row fails. Callback call counts and side effects must not define their
    mathematical result. No further preconditioner call is made after all
    rows have stopped.
    """
    if (type(max_iterations) is not int or not 0 <= max_iterations <= 128
            or not math.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError('Finite positive tolerance and an integer budget in 0..128 required')
    if microkernels not in ('none','mgs','all'):
        raise ValueError('Choose explicit none, mgs or all Krylov microkernels')
    if type(defer_lane_checks) is not bool:
        raise ValueError('defer_lane_checks must be an explicit bool')
    if (not isinstance(rhs, xp.ndarray) or rhs.ndim != 2
            or rhs.shape[0] < 1 or rhs.shape[1] < 1 or rhs.dtype != xp.float64
            or not isinstance(initial, xp.ndarray) or initial.shape != rhs.shape
            or initial.dtype != xp.float64):
        raise ValueError('Matching nonempty FP64 arrays of shape [B, N] required')
    if not bool(xp.all(xp.isfinite(rhs)) & xp.all(xp.isfinite(initial))):
        raise ValueError('Finite right-hand sides and initial vectors required')

    batch, width = rhs.shape
    budget = min(max_iterations, width)

    def vector(value):
        if (not isinstance(value, xp.ndarray) or value.shape != rhs.shape
                or value.dtype != xp.float64):
            raise ValueError('Operator callbacks must return FP64 arrays of shape [B, N]')
        return value

    def measured(residual):
        value = error_measure(rhs, residual)
        if (not isinstance(value, xp.ndarray) or value.shape != (batch,)
                or value.dtype != xp.float64):
            raise ValueError('error_measure must return an FP64 array of shape [B]')
        return value

    best = initial.copy()
    residual = rhs - vector(matvec(initial))
    best_error = measured(residual).copy()
    if bool(xp.any(best_error < 0.)):
        raise ValueError('error_measure must be nonnegative')
    valid = xp.all(xp.isfinite(residual), axis=1) & xp.isfinite(best_error)
    converged = valid & (best_error <= tolerance)
    termination = xp.full(batch, GMRES_ITERATION_LIMIT, dtype=xp.int32)
    termination = xp.where(~valid, GMRES_NONFINITE, termination)
    termination = xp.where(converged, GMRES_CONVERGED, termination)
    iterations = xp.zeros(batch, dtype=xp.int32)
    active = valid & ~converged

    # Mask only rows already marked inactive; invalid results are never repaired
    # or accepted. This prevents inactive environments from poisoning callbacks.
    residual = xp.where(active[:, None], residual, 0.)
    beta = _norm2(residual, xp)
    bad = active & ~xp.isfinite(beta)
    termination = xp.where(bad, GMRES_NONFINITE, termination)
    active &= ~bad
    zero = active & (beta == 0.)
    termination = xp.where(zero, GMRES_BREAKDOWN, termination)
    active &= ~zero

    if _workspace_buffers is None:
        basis = xp.zeros((batch, budget + 1, width), dtype=xp.float64)
        directions = xp.zeros((batch, budget, width), dtype=xp.float64)
        triangular = xp.zeros((batch, budget + 1, budget), dtype=xp.float64)
        cosine = xp.zeros((batch, budget), dtype=xp.float64)
        sine = xp.zeros_like(cosine)
        projected_rhs = xp.zeros((batch, budget + 1), dtype=xp.float64)
    else:
        basis = _workspace_buffers['basis']
        directions = _workspace_buffers['directions']
        triangular = _workspace_buffers['triangular']
        cosine = _workspace_buffers['cosine']
        sine = _workspace_buffers['sine']
        projected_rhs = _workspace_buffers['projected_rhs']
    basis[:, 0] = residual / xp.where(active, beta, 1.)[:, None]
    projected_rhs[:, 0] = xp.where(active, beta, 0.)
    epsilon = xp.finfo(xp.float64).eps

    for column in range(budget):
        if not bool(xp.any(active)):
            break
        iterations += active.astype(xp.int32)
        z = vector(precondition(xp.where(active[:, None], basis[:, column], 0.)))
        bad = active & ~xp.all(xp.isfinite(z), axis=1)
        termination = xp.where(bad, GMRES_NONFINITE, termination)
        active &= ~bad
        directions[:, column] = xp.where(active[:, None], z, 0.)
        if not defer_lane_checks and not bool(xp.any(active)):
            break
        work = vector(matvec(directions[:, column])).copy()
        bad = active & ~xp.all(xp.isfinite(work), axis=1)
        termination = xp.where(bad, GMRES_NONFINITE, termination)
        active &= ~bad
        work = xp.where(active[:, None], work, 0.)
        original_norm = _norm2(work, xp)

        # A second MGS pass is necessary when nearly dependent Newton rows
        # cause the first pass to lose orthogonality through cancellation.
        if microkernels!='none':
            from .gpu_krylov_microkernels import mgs2_inplace
            mgs2_inplace(basis,work,triangular,column,xp=xp)
        else:
            for _ in range(2):
                for row in range(column + 1):
                    coefficient = xp.sum(basis[:, row] * work, axis=1)
                    triangular[:, row, column] += coefficient
                    work -= coefficient[:, None] * basis[:, row]
        next_norm = _norm2(work, xp)
        bad = active & (~xp.isfinite(original_norm) | ~xp.isfinite(next_norm)
                        | ~xp.all(xp.isfinite(triangular[:, :column + 1, column]), axis=1))
        termination = xp.where(bad, GMRES_NONFINITE, termination)
        active &= ~bad
        arnoldi_breakdown = next_norm <= epsilon * original_norm
        triangular[:, column + 1, column] = xp.where(active, next_norm, 0.)

        # Apply the previous rotations to this Hessenberg column, then append
        # one stable rotation. This solves least squares without H.T @ H.
        if microkernels == 'all':
            from .gpu_krylov_microkernels import givens_backsolve_inplace
            coefficients, diagonal = givens_backsolve_inplace(
                triangular, cosine, sine, projected_rhs, active, column, xp=xp)
        else:
            for row in range(column):
                upper = triangular[:, row, column].copy()
                lower = triangular[:, row + 1, column].copy()
                triangular[:, row, column] = cosine[:, row] * upper + sine[:, row] * lower
                triangular[:, row + 1, column] = -sine[:, row] * upper + cosine[:, row] * lower
            upper = triangular[:, column, column].copy()
            lower = triangular[:, column + 1, column].copy()
            diagonal = xp.hypot(upper, lower)
        bad = active & ~xp.isfinite(diagonal)
        termination = xp.where(bad, GMRES_NONFINITE, termination)
        active &= ~bad
        singular = active & (diagonal == 0.)
        termination = xp.where(singular, GMRES_BREAKDOWN, termination)
        active &= ~singular
        if not defer_lane_checks and not bool(xp.any(active)):
            break
        if microkernels != 'all':
            denominator = xp.where(active, diagonal, 1.)
            cosine[:, column] = xp.where(active, upper / denominator, 0.)
            sine[:, column] = xp.where(active, lower / denominator, 0.)
            triangular[:, column, column] = diagonal
            triangular[:, column + 1, column] = 0.
            old_rhs = projected_rhs[:, column].copy()
            projected_rhs[:, column] = cosine[:, column] * old_rhs
            projected_rhs[:, column + 1] = -sine[:, column] * old_rhs

            coefficients = xp.zeros((batch, column + 1), dtype=xp.float64)
            for row in range(column, -1, -1):
                tail = xp.sum(triangular[:, row, row + 1:column + 1]
                              * coefficients[:, row + 1:column + 1], axis=1)
                denominator = xp.where(active, triangular[:, row, row], 1.)
                coefficients[:, row] = xp.where(
                    active, (projected_rhs[:, row] - tail) / denominator, 0.)
        candidate = initial + xp.einsum('bkn,bk->bn', directions[:, :column + 1], coefficients)
        bad = active & ~xp.all(xp.isfinite(candidate), axis=1)
        termination = xp.where(bad, GMRES_NONFINITE, termination)
        active &= ~bad
        if not defer_lane_checks and not bool(xp.any(active)):
            break
        candidate_residual = rhs - vector(matvec(xp.where(active[:, None], candidate, 0.)))
        candidate_error = measured(candidate_residual)
        if bool(xp.any(active & (candidate_error < 0.))):
            raise ValueError('error_measure must be nonnegative')
        bad = active & (~xp.all(xp.isfinite(candidate_residual), axis=1)
                        | ~xp.isfinite(candidate_error))
        termination = xp.where(bad, GMRES_NONFINITE, termination)
        active &= ~bad
        improved = active & (candidate_error < best_error)
        best = xp.where(improved[:, None], candidate, best)
        best_error = xp.where(improved, candidate_error, best_error)
        converged = active & (candidate_error <= tolerance)
        termination = xp.where(converged, GMRES_CONVERGED, termination)
        active &= ~converged
        stopped = active & arnoldi_breakdown
        termination = xp.where(stopped, GMRES_BREAKDOWN, termination)
        active &= ~stopped
        basis[:, column + 1] = xp.where(active[:, None], work, 0.) / xp.where(
            active, next_norm, 1.)[:, None]

    return best, {
        'error': best_error,
        'iterations': iterations,
        'termination': termination,
        'converged': termination == GMRES_CONVERGED,
        'breakdown': termination == GMRES_BREAKDOWN,
        'nonfinite': termination == GMRES_NONFINITE,
    }


def batched_gmres(rhs, initial, matvec, precondition, error_measure, *, xp,
                  max_iterations=12, tolerance=1e-8, microkernels='none',
                  defer_lane_checks=False, workspace=None):
    """Batched original-residual GMRES with optional reusable Arnoldi storage.

    ``workspace=None`` retains the previous allocation/arithmetic path. Supply
    a :class:`GmresWorkspace` with exactly matching shape, requested iteration
    budget, array namespace, and CUDA device/stream to reuse its six arrays.
    Workspace use does not change tolerances, masks, residual evaluations or
    best-candidate acceptance. Answers and all diagnostics remain independent
    of its storage and of later calls. It does not enable CUDA graph capture.
    """
    options = dict(xp=xp, max_iterations=max_iterations, tolerance=tolerance,
                   microkernels=microkernels, defer_lane_checks=defer_lane_checks)
    if workspace is None:
        return _batched_gmres_impl(rhs, initial, matvec, precondition, error_measure, **options)
    if not isinstance(workspace, GmresWorkspace):
        raise ValueError('workspace must be a GmresWorkspace or None')
    with workspace._lease(rhs, initial, xp=xp, max_iterations=max_iterations) as buffers:
        return _batched_gmres_impl(rhs, initial, matvec, precondition, error_measure,
                                  _workspace_buffers=buffers, **options)


batched_gmres.__doc__ += '\n' + _batched_gmres_impl.__doc__
