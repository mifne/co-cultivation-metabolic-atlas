"""Reusable, fixed-pattern FP64 PDHG device buffers and ordinary CUDA Graphs.

This is an iteration engine, not an LP solver or acceptance gate. Callers must
certify lifted answers against the unchanged original LP. Updating a problem
invalidates its iterate; ``reset`` is mandatory before another run. No CPU
optimizer, current-reference warm start, or automatic algorithm fallback exists.

CSR structure and environment order are immutable. Matrix values, vectors and
diagonal metrics are copied to stable device addresses outside graph replay.
Public validation synchronizes and may transfer CSR *indices* to the host;
iteration arithmetic and graph replay remain on the selected GPU. Instances are
serialized, single-owner services, not thread-safe concurrent submission queues.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


_SPMV_SOURCE = r'''
extern "C" __global__ void csr_warp_spmv(
    const int rows, const int* rowptr, const int* columns,
    const double* values, const double* x, double* y) {
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    double sum = 0.0;
    for (int index = rowptr[row] + lane; index < rowptr[row+1]; index += 32)
        sum += values[index] * x[columns[index]];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
}
'''


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ids(values, count):
    ids = tuple(range(count)) if values is None else tuple(values)
    if len(ids) != count or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (str, int, np.integer)) for v in ids):
        raise ValueError("env_ids must contain one integer/string identifier per environment")
    if len(set(ids)) != count:
        raise ValueError("env_ids must be unique")
    return ids


class GpuPdhgWorkspace:
    """Fixed-pattern block-CSR service; all numeric device buffers are owned.

    ``a`` is a canonical, sorted, FP64/int32 CuPy CSR block-diagonal matrix.
    Vectors accept flat or ``[batch, per_environment]`` NumPy/CuPy inputs.
    A supplied CUDA stream must support ordinary stream capture; unsupported
    capture raises, without switching routes or modifying the environment.
    ``use_graph=False`` explicitly selects the same arithmetic's Python loop.
    """

    def __init__(self, a, *, batch_size, n_variables, n_constraints,
                 rhs, c, lower, upper, tau, sigma, inequality_mask,
                 theta=1.0, env_ids=None, chunk_size=64, use_graph=True,
                 stream=None):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix

        started = time.perf_counter()
        self.cp = cp
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.batch = _integer(batch_size, "batch_size")
        self.n = _integer(n_variables, "n_variables")
        self.m = _integer(n_constraints, "n_constraints", minimum=0)
        self.chunk_size = _integer(chunk_size, "chunk_size")
        if max(self.batch*self.n, self.batch*self.m) >= np.iinfo(np.int32).max:
            raise ValueError("Block dimensions exceed int32 CSR capacity")
        if any(value is None for value in (rhs, c, lower, upper, tau, sigma, inequality_mask, theta)):
            raise ValueError("Initial problem vectors, metrics, mask and theta are required")
        if not isinstance(use_graph, (bool, np.bool_)):
            raise ValueError("use_graph must be boolean")
        self.use_graph = bool(use_graph)
        self.env_ids = _ids(env_ids, self.batch)
        self.stream = stream if stream is not None else cp.cuda.Stream(non_blocking=True)
        self._ready = False
        self.graph = None
        self.capture_seconds = 0.0
        self.graph_upload_seconds = 0.0
        self.iterations_run = 0
        self.update_timing = {}
        self.run_stats = {}
        self._wait_for_inputs()
        with self.stream:
            self._pattern_indptr, self._pattern_indices = self._matrix_pattern(a)
            self.a = a.copy()
            rows = np.repeat(np.arange(self.batch*self.m, dtype=np.int32), np.diff(self._pattern_indptr))
            order = np.lexsort((rows, self._pattern_indices))
            at_indptr = np.r_[0, np.cumsum(np.bincount(self._pattern_indices, minlength=self.batch*self.n))].astype(np.int32)
            self._transpose_order = cp.asarray(order, dtype=cp.int64)
            self.at = csr_matrix((cp.empty_like(self.a.data), cp.asarray(rows[order]), cp.asarray(at_indptr)), shape=(self.batch*self.n, self.batch*self.m))
            self.rhs = cp.empty(self.batch*self.m, dtype=cp.float64)
            self.c = cp.empty(self.batch*self.n, dtype=cp.float64)
            self.lower, self.upper = cp.empty_like(self.c), cp.empty_like(self.c)
            self.tau, self.sigma = cp.empty_like(self.c), cp.empty_like(self.rhs)
            self.inequality_mask = cp.empty(self.rhs.shape, dtype=cp.bool_)
            self.dual_upper = cp.empty_like(self.rhs)
            self.theta = cp.empty((), dtype=cp.float64)
            self.x, self.x_bar, self.x_next = [cp.empty_like(self.c) for _ in range(3)]
            self.y, self.y_next, self.ax, self.row_work = [cp.empty_like(self.rhs) for _ in range(4)]
            self.aty, self.col_work = cp.empty_like(self.c), cp.empty_like(self.c)
            self.accepted = cp.zeros(self.batch, dtype=cp.bool_)
            self._col_accepted = cp.zeros(self.c.shape, dtype=cp.bool_)
            self._row_accepted = cp.zeros(self.rhs.shape, dtype=cp.bool_)
            self._col_env = cp.repeat(cp.arange(self.batch), self.n)
            self._row_env = cp.repeat(cp.arange(self.batch), self.m)
            self._kernel = cp.RawKernel(_SPMV_SOURCE, "csr_warp_spmv", options=("--std=c++11",))
            # Disable fused multiply-add in these fused launch kernels to keep
            # the previous separate FP64 arithmetic's operation boundaries.
            self._row_update = cp.ElementwiseKernel(
                "float64 old_y, float64 rhs, float64 ax, float64 sigma, float64 dual_upper, bool done",
                "float64 out_y, float64 next_y",
                "double delta = sigma * (rhs - ax); double candidate = old_y + delta; "
                "out_y = done ? old_y : (isnan(candidate) ? candidate : fmin(candidate, dual_upper)); next_y = out_y;",
                "pdhg_workspace_row_update_fp64", options=("--fmad=false",))
            self._col_update = cp.ElementwiseKernel(
                "float64 old_x, float64 aty, float64 c, float64 tau, float64 lower, float64 upper, float64 theta, bool done",
                "float64 out_x, float64 xbar",
                "double step = tau * (c - aty); double candidate = old_x - step; "
                "double next = done ? old_x : (isnan(candidate) ? candidate : fmin(fmax(candidate, lower), upper)); "
                "double extrapolation = theta * (next - old_x); xbar = next + extrapolation; out_x = next;",
                "pdhg_workspace_col_update_fp64", options=("--fmad=false",))
            self._forward_args = (np.int32(self.a.shape[0]), self.a.indptr, self.a.indices, self.a.data, self.x_bar, self.ax)
            self._transpose_args = (np.int32(self.at.shape[0]), self.at.indptr, self.at.indices, self.at.data, self.y_next, self.aty)
        self.update_problem(a=a, rhs=rhs, c=c, lower=lower, upper=upper,
                            tau=tau, sigma=sigma, inequality_mask=inequality_mask,
                            theta=theta, env_ids=self.env_ids)
        self.reset()
        # Compile every used operator before capture. Warmup is never returned
        # as an actual iterate; reset also clears any extrapolation history.
        with self.stream:
            self._iterate(1)
        self.synchronize()
        self.reset()
        if self.use_graph:
            self._capture()
        self.setup_seconds = time.perf_counter()-started
        self.setup_timing = dict(setup_total_seconds=self.setup_seconds,
                                graph_capture_seconds=self.capture_seconds,
                                graph_upload_seconds=self.graph_upload_seconds,
                                chunk_size=self.chunk_size, graph_enabled=self.use_graph,
                                kernels_per_iteration=4 if self.m else 2,
                                spmv_backend="csr_warp_fp64")

    def _device(self):
        if int(self.cp.cuda.runtime.getDevice()) != self.device_id:
            raise ValueError("Workspace used from a different CUDA device")

    def _wait_for_inputs(self):
        self._device()
        producer = self.cp.cuda.get_current_stream()
        if producer.ptr != self.stream.ptr:
            event = self.cp.cuda.Event()
            event.record(producer)
            self.stream.wait_event(event)

    def _matrix_pattern(self, a):
        cp = self.cp
        if getattr(a, "format", None) != "csr" or a.shape != (self.batch*self.m, self.batch*self.n):
            raise ValueError("a must be a block CSR matrix of the configured shape")
        for value, dtype in ((a.data, cp.float64), (a.indices, cp.int32), (a.indptr, cp.int32)):
            if not isinstance(value, cp.ndarray) or value.device.id != self.device_id or value.dtype != dtype:
                raise ValueError("a requires FP64 values/int32 CSR indices on the workspace device")
        ptr, cols = cp.asnumpy(a.indptr), cp.asnumpy(a.indices)
        if ptr.shape != (a.shape[0]+1,) or ptr[0] != 0 or ptr[-1] != len(cols) or np.any(np.diff(ptr) < 0):
            raise ValueError("Invalid CSR row pointers")
        if a.data.size != cols.size or np.any(cols < 0) or np.any(cols >= a.shape[1]):
            raise ValueError("Invalid CSR column indices")
        rows = np.repeat(np.arange(a.shape[0]), np.diff(ptr))
        if len(cols) > 1 and np.any((rows[1:] == rows[:-1]) & (cols[1:] <= cols[:-1])):
            raise ValueError("CSR rows must have sorted, unique column indices")
        if self.m and np.any(rows//self.m != cols//self.n):
            raise ValueError("CSR must not couple different environments")
        if not bool(cp.all(cp.isfinite(a.data)).item()):
            raise ValueError("Matrix values must be finite")
        return ptr, cols

    def _vector(self, value, name, size, *, boolean=False, bounds=False):
        cp = self.cp
        if isinstance(value, cp.ndarray) and value.device.id != self.device_id:
            raise ValueError(f"{name} is on another CUDA device")
        result = cp.asarray(value)
        if result.shape not in ((size,), (self.batch, size//self.batch)):
            raise ValueError(f"{name} has the wrong shape")
        if boolean:
            if result.dtype != cp.bool_:
                raise ValueError(f"{name} must be boolean")
        else:
            if result.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be real numeric")
            result = result.astype(cp.float64, copy=False)
            invalid = cp.isnan(result) if bounds else ~cp.isfinite(result)
            if bool(cp.any(invalid).item()):
                raise ValueError(f"{name} contains non-finite values")
        return result.reshape(size)

    def update_problem(self, *, a=None, rhs=None, c=None, lower=None, upper=None,
                       tau=None, sigma=None, inequality_mask=None, theta=None,
                       env_ids=None):
        """Validate a complete update before copying; preserve captured pointers.

        Omitting ``env_ids`` retains the configured order. Explicitly supplying
        a different order is an error, not an implicit reassignment of states.
        Metrics must be supplied again whenever matrix coefficients change;
        their convergence properties remain the caller's responsibility.
        """
        started = time.perf_counter()
        self._wait_for_inputs()
        if env_ids is not None and _ids(env_ids, self.batch) != self.env_ids:
            raise ValueError("env_ids/order changed; explicitly rebuild the workspace")
        cp = self.cp
        with self.stream:
            if a is not None:
                ptr, cols = self._matrix_pattern(a)
                if not np.array_equal(ptr, self._pattern_indptr) or not np.array_equal(cols, self._pattern_indices):
                    raise ValueError("CSR pattern changed; explicitly rebuild the workspace")
                if tau is None or sigma is None:
                    raise ValueError("Matrix updates require explicit tau and sigma metrics")
            updates = {}
            for name, supplied, size in (("rhs", rhs, self.batch*self.m), ("c", c, self.batch*self.n),
                                        ("lower", lower, self.batch*self.n), ("upper", upper, self.batch*self.n),
                                        ("tau", tau, self.batch*self.n), ("sigma", sigma, self.batch*self.m),
                                        ("inequality_mask", inequality_mask, self.batch*self.m)):
                if supplied is not None:
                    updates[name] = self._vector(supplied, name, size, boolean=name == "inequality_mask", bounds=name in ("lower", "upper"))
            lo, hi = updates.get("lower", self.lower), updates.get("upper", self.upper)
            if bool(cp.any((lo > hi) | cp.isposinf(lo) | cp.isneginf(hi)).item()):
                raise ValueError("Bounds require lower <= upper and finite feasible points")
            for name in ("tau", "sigma"):
                if name in updates and bool(cp.any(updates[name] <= 0).item()):
                    raise ValueError(f"{name} metrics must be strictly positive")
            if theta is not None:
                theta = float(theta)
                if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
                    raise ValueError("theta must be finite and between zero and one")
            validated_at = time.perf_counter()
            # No persistent data have changed before all validation succeeds.
            self._ready = False
            if a is not None:
                cp.copyto(self.a.data, a.data)
                cp.take(self.a.data, self._transpose_order, out=self.at.data)
            for name, value in updates.items():
                cp.copyto(getattr(self, name), value)
            if "inequality_mask" in updates:
                self.dual_upper.fill(cp.inf)
                cp.copyto(self.dual_upper, 0., where=self.inequality_mask)
            if theta is not None:
                self.theta.fill(theta)
        self.synchronize()
        self.update_timing = dict(update_total_seconds=time.perf_counter()-started,
                                 validation_seconds=validated_at-started,
                                 copy_and_synchronize_seconds=time.perf_counter()-validated_at,
                                 matrix_values_updated=a is not None,
                                 copied_vector_names=list(updates))
        return dict(self.update_timing)

    def reset(self, x=None, y=None):
        """Cold or arbitrary warm start, projected to current box/dual signs."""
        self._wait_for_inputs()
        cp = self.cp
        with self.stream:
            warm_x = None if x is None else self._vector(x, "x", self.batch*self.n)
            warm_y = None if y is None else self._vector(y, "y", self.batch*self.m)
            if warm_x is None:
                self.x.fill(0.)
            else:
                cp.copyto(self.x, warm_x)
            cp.maximum(self.x, self.lower, out=self.x)
            cp.minimum(self.x, self.upper, out=self.x)
            cp.copyto(self.x_bar, self.x)
            if warm_y is None:
                self.y.fill(0.)
            else:
                cp.copyto(self.y, warm_y)
            cp.minimum(self.y, self.dual_upper, out=self.y)
            self.accepted.fill(False)
            self._row_accepted.fill(False)
            self._col_accepted.fill(False)
        self.iterations_run = 0
        self._ready = True
        self.synchronize()

    def set_accepted(self, mask):
        """Freeze certified environments; only reset may unfreeze them."""
        self._wait_for_inputs()
        if not self._ready:
            raise RuntimeError("reset is required after a problem update")
        cp = self.cp
        with self.stream:
            value = self._vector(mask, "accepted mask", self.batch, boolean=True)
            if bool(cp.any(self.accepted & ~value).item()):
                raise ValueError("Cannot unfreeze accepted environments without reset")
            cp.copyto(self.accepted, value)
            cp.take(self.accepted, self._col_env, out=self._col_accepted)
            cp.take(self.accepted, self._row_env, out=self._row_accepted)
            cp.copyto(self.x_bar, self.x, where=self._col_accepted)

    def _iterate(self, count):
        for _ in range(count):
            if self.m:
                self._kernel(((self.a.shape[0]+3)//4,), (128,), self._forward_args)
                self._row_update(self.y, self.rhs, self.ax, self.sigma, self.dual_upper,
                                 self._row_accepted, self.y, self.y_next)
            self._kernel(((self.at.shape[0]+3)//4,), (128,), self._transpose_args)
            self._col_update(self.x, self.aty, self.c, self.tau, self.lower, self.upper,
                             self.theta, self._col_accepted, self.x, self.x_bar)

    def _capture(self):
        started = time.perf_counter()
        with self.stream:
            self.stream.begin_capture()
            try:
                self._iterate(self.chunk_size)
                self.graph = self.stream.end_capture()
            except BaseException:
                try:
                    self.stream.end_capture()
                except BaseException:
                    pass
                raise
        self.synchronize()
        self.capture_seconds = time.perf_counter()-started
        started = time.perf_counter()
        self.graph.upload(stream=self.stream)
        self.synchronize()
        self.graph_upload_seconds = time.perf_counter()-started

    def run(self, iterations, *, use_graph=None):
        """Enqueue fixed work; full chunks use graph, tail uses same loop."""
        self._device()
        iterations = _integer(iterations, "iterations", minimum=0)
        if not self._ready:
            raise RuntimeError("reset is required after a problem update")
        if use_graph is not None and not isinstance(use_graph, (bool, np.bool_)):
            raise ValueError("use_graph must be boolean")
        graph_route = self.use_graph if use_graph is None else bool(use_graph)
        if graph_route and self.graph is None:
            raise RuntimeError("Workspace has no captured graph; rebuild with use_graph=True")
        started = time.perf_counter()
        chunks, tail = 0, iterations
        with self.stream:
            if graph_route:
                chunks, tail = divmod(iterations, self.chunk_size)
                for _ in range(chunks):
                    self.graph.launch(stream=self.stream)
                self._iterate(tail)
            else:
                self._iterate(iterations)
        self.iterations_run += iterations
        self.run_stats = dict(iterations=iterations, total_iterations=self.iterations_run,
                              graph_chunks=chunks, python_tail_iterations=tail,
                              graph_route=graph_route,
                              enqueue_seconds=time.perf_counter()-started,
                              timing_scope="Host enqueue only; caller must synchronize for elapsed wall time")
        return dict(self.run_stats)

    def state(self, *, synchronize=True):
        """Return borrowed device x/y views; copy if retaining across updates.

        To avoid the default synchronization, consume ``state(synchronize=False)``
        inside ``with workspace.stream``. Do not mutate borrowed buffers.
        """
        self._device()
        if not self._ready:
            raise RuntimeError("reset is required after a problem update")
        if synchronize:
            self.synchronize()
        return self.x.reshape(self.batch, self.n), self.y.reshape(self.batch, self.m)

    def synchronize(self):
        self._device()
        self.stream.synchronize()
