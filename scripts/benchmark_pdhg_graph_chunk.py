"""Fixed-work CUDA Graph launch diagnostic, NOT a qualified LP speedup.

Both routes execute identical FP64 PDHG iterations with a cached CSR transpose
and preallocated workspaces. Current CPU reference solutions are never warm
starts. Original lifted x/y and original-LP residuals are compared after equal
work, independently of whether either output meets the unchanged certificate.
The reduced LP is fixed across chunk replays: no dFBA trajectory or new-state
input update is represented by this kernel-launch microbenchmark.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lp_trace import load_trace_lp


SCOPE = ("Fixed-input, fixed-iteration GPU launch microbenchmark. Cached-transpose "
         "Python loop and ordinary CUDA Graph run identical arithmetic and buffers. "
         "No current CPU reference warm starts, no neural inference, no CPU LP, "
         "no dFBA state updates, no certificate-based early stopping. Timing gain "
         "does not establish LP convergence, accepted-solution speedup, or end-to-end GPU superiority.")


class PreallocatedSpmv:
    """Keep all cuSPARSE descriptor/scalar/scratch lifetimes beyond capture.

    Mirrors the installed CuPy ``cupyx.cusparse.spmv`` wrapper, but allocates
    descriptors and workspace once. This is a version-recorded diagnostic,
    not a promise that CuPy's lower-level wrapper is a stable public API.
    """
    def __init__(self, matrix, x, y):
        import cupy as cp
        import cupyx.cusparse as cs
        self.cp, self.cs = cp, cs
        self.matrix, self.x, self.y = matrix, x, y
        self.a_desc = cs.SpMatDescriptor.create(matrix)
        self.x_desc = cs.DnVecDescriptor.create(x)
        self.y_desc = cs.DnVecDescriptor.create(y)
        self.handle = cp.cuda.device.get_cusparse_handle()
        self.operation = cs._cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        self.dtype = cs._dtype.to_cuda_dtype(matrix.dtype)
        self.algorithm = cs._cusparse.CUSPARSE_MV_ALG_DEFAULT
        self.alpha, self.beta = np.array(1., dtype=np.float64), np.array(0., dtype=np.float64)
        self.arguments = (self.handle, self.operation, self.alpha.ctypes.data, self.a_desc.desc,
                          self.x_desc.desc, self.beta.ctypes.data, self.y_desc.desc,
                          self.dtype, self.algorithm)
        size = cs._cusparse.spMV_bufferSize(*self.arguments)
        self.workspace = cp.empty(size, dtype=cp.int8)

    def __call__(self):
        self.cs._cusparse.spMV(*self.arguments, self.workspace.data.ptr)


class CsrWarpSpmv:
    """Standalone FP64 CSR operator, not a cuSPARSE capture restriction patch.

    One warp owns each CSR row. NVRTC uses ordinary device compilation with
    no relocatable-device-code, dynamic parallelism or external CUDA library.
    The same operator is used for Python-loop and CUDA-Graph measurements.
    """
    source = r'''
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
    _kernel = None

    def __init__(self, matrix, x, y):
        import cupy as cp
        if matrix.format != "csr" or matrix.dtype != cp.float64:
            raise ValueError("CSR kernel requires FP64 CSR")
        if matrix.indptr.dtype != cp.int32 or matrix.indices.dtype != cp.int32:
            raise ValueError("CSR kernel requires int32 sparse indices")
        if CsrWarpSpmv._kernel is None:
            CsrWarpSpmv._kernel = cp.RawKernel(self.source, "csr_warp_spmv", options=("--std=c++11",))
        self.matrix, self.x, self.y = matrix, x, y
        self.workspace = cp.empty(0, dtype=cp.int8)
        self.arguments = (np.int32(matrix.shape[0]), matrix.indptr, matrix.indices, matrix.data, x, y)
        self.grid, self.block = ((matrix.shape[0]+3)//4,), (128,)

    def __call__(self):
        self._kernel(self.grid, self.block, self.arguments)


class FixedPdhgChunk:
    """A static allocation schedule shared by both loop and graph routes."""
    def __init__(self, corrector, *, spmv_backend="cusparse"):
        cp = corrector.cp
        self.cp, self.corrector = cp, corrector
        self.stream = cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.a = corrector.a
            # Never create/transcode the transpose in the iteration loop.
            self.at = corrector.a.T.tocsr()
            self.x = cp.empty(corrector.batch*corrector.n, dtype=cp.float64)
            self.x_bar = cp.empty_like(self.x)
            self.x_next = cp.empty_like(self.x)
            self.y = cp.empty(corrector.batch*corrector.m, dtype=cp.float64)
            self.y_next = cp.empty_like(self.y)
            self.ax = cp.empty_like(self.y)
            self.aty = cp.empty_like(self.x)
            self.row_work = cp.empty_like(self.y)
            self.col_work = cp.empty_like(self.x)
            self.dual_upper = cp.where(corrector.inequality_mask, 0., cp.inf)
            # Cold reduced primal is projected to the same transformed box.
            self.initial_x = cp.minimum(cp.maximum(cp.zeros_like(self.x), corrector.lower), corrector.upper)
            operator = CsrWarpSpmv if spmv_backend == "csr-kernel" else PreallocatedSpmv
            self.forward = operator(self.a, self.x_bar, self.ax)
            self.transpose = operator(self.at, self.y_next, self.aty)
        self.stream.synchronize()
        self.reset()

    def reset(self):
        before = time.perf_counter()
        with self.stream:
            self.cp.copyto(self.x, self.initial_x)
            self.cp.copyto(self.x_bar, self.initial_x)
            self.y.fill(0.)
        self.stream.synchronize()
        return time.perf_counter()-before

    def iterate(self, count):
        cp, solver = self.cp, self.corrector
        for _ in range(count):
            self.forward()
            cp.subtract(solver.rhs, self.ax, out=self.row_work)
            cp.multiply(solver.sigma, self.row_work, out=self.row_work)
            cp.add(self.y, self.row_work, out=self.y_next)
            cp.minimum(self.y_next, self.dual_upper, out=self.y_next)
            self.transpose()
            cp.subtract(solver.c, self.aty, out=self.col_work)
            cp.multiply(solver.tau, self.col_work, out=self.col_work)
            cp.subtract(self.x, self.col_work, out=self.x_next)
            cp.maximum(self.x_next, solver.lower, out=self.x_next)
            cp.minimum(self.x_next, solver.upper, out=self.x_next)
            cp.subtract(self.x_next, self.x, out=self.x_bar)
            cp.multiply(self.x_bar, solver.theta, out=self.x_bar)
            cp.add(self.x_next, self.x_bar, out=self.x_bar)
            cp.copyto(self.x, self.x_next)
            cp.copyto(self.y, self.y_next)

    def capture(self, count):
        before = time.perf_counter()
        with self.stream:
            self.stream.begin_capture()
            try:
                self.iterate(count)
                graph = self.stream.end_capture()
            except BaseException:
                # Pair begin/end even when capture is invalidated. Never change
                # the CUDA environment or quietly switch algorithms on failure.
                try:
                    self.stream.end_capture()
                except BaseException:
                    pass
                raise
        self.stream.synchronize()
        return graph, time.perf_counter()-before

    def snapshot(self):
        self.stream.synchronize()
        solver, cp = self.corrector, self.cp
        z, y = self.x.reshape(solver.batch, solver.n), self.y.reshape(solver.batch, solver.m)
        # Lift and unchanged certificate occur OUTSIDE the chunk timing.
        metrics = solver._certificate(z, y)
        original_x, original_y = solver._last_lift
        return dict(reduced_x=cp.asnumpy(z), reduced_y=cp.asnumpy(y),
                    original_x=cp.asnumpy(original_x), original_y=cp.asnumpy(original_y), metrics=metrics)


def compare_snapshots(reference, actual, *, atol=1e-8, rtol=1e-10):
    comparison = {}
    for name in ("reduced_x", "reduced_y", "original_x", "original_y"):
        left, right = reference[name], actual[name]
        passed = bool(np.isfinite(left).all() and np.isfinite(right).all()
                      and np.allclose(left, right, rtol=rtol, atol=atol))
        comparison[name] = dict(max_abs_difference=float(np.max(np.abs(left-right), initial=0)),
                                exactly_equal=bool(np.array_equal(left, right)),
                                equivalent=passed)
        if not passed:
            raise AssertionError(f"Equal-iteration route mismatch: {name}")
    metric_names = ("primal_residual", "dual_violation", "relative_kkt_gap", "objective")
    comparison["original_certificate_metrics"] = {}
    for name in metric_names:
        left = np.asarray([r[name] for r in reference["metrics"]])
        right = np.asarray([r[name] for r in actual["metrics"]])
        passed = bool(np.isfinite(left).all() and np.isfinite(right).all()
                      and np.allclose(left, right, rtol=rtol, atol=atol))
        comparison["original_certificate_metrics"][name] = dict(
            max_abs_difference=float(np.max(np.abs(left-right), initial=0)), equivalent=passed)
        if not passed:
            raise AssertionError(f"Original-LP residual mismatch: {name}")
    if [r["certificate_passed"] for r in reference["metrics"]] != [r["certificate_passed"] for r in actual["metrics"]]:
        raise AssertionError("Original-LP acceptance differs between routes")
    return comparison


def timed_run(chunk, graph, count, launches, *, route):
    cp = chunk.cp
    reset_seconds = chunk.reset()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    with chunk.stream:
        start.record(chunk.stream)
        before = time.perf_counter()
        for _ in range(launches):
            if route == "graph":
                graph.launch(stream=chunk.stream)
            else:
                chunk.iterate(count)
        stop.record(chunk.stream)
    stop.synchronize()
    wall = time.perf_counter()-before
    return dict(route=route, iterations=count*launches, launches=launches,
                synchronized_wall_seconds=wall, cuda_event_seconds=cp.cuda.get_elapsed_time(start, stop)/1000.,
                reset_seconds=reset_seconds, reset_plus_wall_seconds=reset_seconds+wall), chunk.snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("results/pf_lp_trace_dev4x60_20260905"))
    parser.add_argument("--stage", choices=["maxmin", "aggregate", "exchange"], default="exchange")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--environments", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--primal-weight", type=float, default=1.)
    parser.add_argument("--spmv-backend", choices=["cusparse", "csr-kernel"], default="cusparse")
    parser.add_argument("--equivalence-atol", type=float, default=1e-8)
    parser.add_argument("--equivalence-rtol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=Path("results/pf_pdhg_graph_chunk4x64_20260905.json"))
    args = parser.parse_args()
    if min(args.step, args.environments, args.chunk, args.launches, args.repeats) < 1:
        raise ValueError("Positive dimensions and budgets required")
    if not np.isfinite([args.equivalence_atol, args.equivalence_rtol]).all() or min(args.equivalence_atol, args.equivalence_rtol) <= 0:
        raise ValueError("Finite positive numerical-equivalence tolerances required")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(status="initializing", scope=SCOPE,
                  configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  primary_sources=[
                      "https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html",
                      "https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Graph.html",
                      "https://docs.nvidia.com/cuda/cusparse/index.html",
                      "https://github.com/cupy/cupy/blob/v14.2.0/cupyx/cusparse.py"],
                  pairs=[])
    report["numerical_equivalence_tolerances"] = dict(atol=args.equivalence_atol, rtol=args.equivalence_rtol,
        warning="Comparing two fixed-work numerical implementations only; original LP acceptance thresholds are unchanged and pass flags must match")
    report["operator"] = dict(backend=args.spmv_backend,
        standalone_kernel_sha256=hashlib.sha256(CsrWarpSpmv.source.encode()).hexdigest() if args.spmv_backend == "csr-kernel" else None,
        rawkernel_options=["--std=c++11"] if args.spmv_backend == "csr-kernel" else None,
        no_device_c_no_external_libraries=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    def save():
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(args.output)
    try:
        save()
        manifest_path = args.trace/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "completed" or len(manifest["seeds"]) < args.environments:
            raise ValueError("Completed trace with sufficient environments required")
        entries = [entry for entry in manifest["entries"] if entry["stage"] == args.stage
                   and entry["step"] == args.step and entry["environment_id"] < args.environments]
        entries.sort(key=lambda entry: entry["environment_id"])
        if [entry["environment_id"] for entry in entries] != list(range(args.environments)):
            raise ValueError("Missing or duplicate fixed-input LP cohort")
        problems = [load_trace_lp(args.trace, entry)[0] for entry in entries]
        report.update(trace_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      trace_role=manifest.get("role"), model_fingerprints=manifest["model_fingerprints"],
                      seeds=manifest["seeds"][:args.environments], entries=entries,
                      warm_start="Cold zero projected into reduced bounds; no CPU-reference x/y used")
        source_dir = args.output.with_suffix(".sources")
        source_dir.mkdir(exist_ok=False)
        report["source_hashes"] = {}
        for name in ("scripts/benchmark_pdhg_graph_chunk.py", "src/gpu_reduced_pdhg.py",
                     "src/gpu_pdhg_corrector.py", "src/lp_equality_reduction.py",
                     "src/gpu_block_lp.py", "src/lp_trace.py", "src/cpu_repeated_lp.py"):
            target = source_dir/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
            report["source_hashes"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
        import cupy as cp
        import cupyx.cusparse as cs
        from src.gpu_reduced_pdhg import GpuReducedPdhgCorrector
        report["runtime"] = dict(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                                 cuda_driver=cp.cuda.runtime.driverGetVersion(), device=int(cp.cuda.Device().id),
                                 gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                                 lower_level_spmv_wrapper_sha256=hashlib.sha256(Path(cs.__file__).read_bytes()).hexdigest())
        before = time.perf_counter()
        solver = GpuReducedPdhgCorrector(problems, primal_weight=args.primal_weight)
        chunk = FixedPdhgChunk(solver, spmv_backend=args.spmv_backend)
        cp.cuda.runtime.deviceSynchronize()
        allocation_seconds = time.perf_counter()-before
        before = time.perf_counter()
        with chunk.stream:
            chunk.iterate(args.chunk)
        chunk.stream.synchronize()
        warmup_seconds = time.perf_counter()-before
        chunk.reset()
        report["setup"] = dict(common_allocation_and_reduction_seconds=allocation_seconds,
                               common_loop_warmup_seconds=warmup_seconds,
                               solver=solver.setup_timing,
                               workspace_bytes=chunk.forward.workspace.nbytes+chunk.transpose.workspace.nbytes,
                               cached_transpose_nnz=int(chunk.at.nnz))
        report["capture_status"] = "attempting"
        save()
        before = time.perf_counter()
        try:
            graph, capture_seconds = chunk.capture(args.chunk)
        except BaseException:
            report["setup"]["failed_capture_attempt_seconds"] = time.perf_counter()-before
            report["capture_status"] = "unsupported_or_failed"
            raise
        report["capture_status"] = "completed"
        before = time.perf_counter()
        graph.upload(stream=chunk.stream)
        graph.launch(stream=chunk.stream)
        chunk.stream.synchronize()
        graph_first_launch_seconds = time.perf_counter()-before
        report["setup"].update(graph_capture_and_instantiation_seconds=capture_seconds,
                               graph_upload_and_first_launch_seconds=graph_first_launch_seconds)
        # Validate one chunk first, then every timed equal-work trial pair.
        first_loop, loop_state = timed_run(chunk, graph, args.chunk, 1, route="loop")
        first_graph, graph_state = timed_run(chunk, graph, args.chunk, 1, route="graph")
        report["single_chunk_equivalence"] = compare_snapshots(loop_state, graph_state,
            atol=args.equivalence_atol, rtol=args.equivalence_rtol)
        report["single_chunk_metrics"] = dict(loop=loop_state["metrics"], graph=graph_state["metrics"])
        if args.spmv_backend == "csr-kernel":
            # Existing cuSPARSE-based corrector is a numerical reference only:
            # same cold state and exactly chunk*launches updates. It is not a
            # CPU reference solution and is not included in the graph timings.
            before = time.perf_counter()
            legacy = solver.solve(iterations=args.chunk*args.launches,
                                  check_interval=args.chunk*args.launches)
            legacy_seconds = time.perf_counter()-before
            if legacy["iterations_run"] != args.chunk*args.launches:
                raise ValueError("Legacy corrector stopped early; equal-work comparison is unavailable")
            legacy_state = dict(reduced_x=cp.asnumpy(legacy["reduced_x"]), reduced_y=cp.asnumpy(legacy["reduced_y"]),
                                original_x=cp.asnumpy(legacy["x"]), original_y=cp.asnumpy(legacy["y"]), metrics=legacy["metrics"])
            _, candidate_state = timed_run(chunk, graph, args.chunk, args.launches, route="graph")
            report["existing_cusparse_equal_work_validation"] = dict(
                iterations=legacy["iterations_run"], diagnostic_seconds=legacy_seconds,
                equivalence=compare_snapshots(legacy_state, candidate_state,
                    atol=args.equivalence_atol, rtol=args.equivalence_rtol),
                existing_metrics=legacy["metrics"], candidate_metrics=candidate_state["metrics"],
                warning="Numerical reference only, not one route of the matched fixed-work timing comparison")
        report["status"] = "running"
        save()
        for repeat in range(args.repeats):
            order = ("loop", "graph") if repeat % 2 == 0 else ("graph", "loop")
            timings, states = {}, {}
            for route in order:
                timings[route], states[route] = timed_run(chunk, graph, args.chunk, args.launches, route=route)
            equivalence = compare_snapshots(states["loop"], states["graph"],
                atol=args.equivalence_atol, rtol=args.equivalence_rtol)
            pair = dict(repeat=repeat, execution_order=list(order), timings=timings,
                        equivalence=equivalence, final_original_metrics=states["graph"]["metrics"],
                        loop_over_graph_wall_ratio=timings["loop"]["synchronized_wall_seconds"]/timings["graph"]["synchronized_wall_seconds"])
            report["pairs"].append(pair)
            print(f'pair {repeat}: loop {timings["loop"]["synchronized_wall_seconds"]:.6f}s, '
                  f'graph {timings["graph"]["synchronized_wall_seconds"]:.6f}s, '
                  f'ratio {pair["loop_over_graph_wall_ratio"]:.3f}; exact original x/y '
                  f'{equivalence["original_x"]["exactly_equal"]}/{equivalence["original_y"]["exactly_equal"]}', flush=True)
            save()
        loop_per_chunk = np.mean([p["timings"]["loop"]["synchronized_wall_seconds"]/args.launches for p in report["pairs"]])
        graph_per_chunk = np.mean([p["timings"]["graph"]["synchronized_wall_seconds"]/args.launches for p in report["pairs"]])
        reset_loop = np.mean([p["timings"]["loop"]["reset_seconds"] for p in report["pairs"]])
        reset_graph = np.mean([p["timings"]["graph"]["reset_seconds"] for p in report["pairs"]])
        extra = capture_seconds+graph_first_launch_seconds
        common = allocation_seconds+warmup_seconds
        saving = loop_per_chunk-graph_per_chunk
        counts = sorted(set([1, args.launches, args.launches*args.repeats, 120]))
        report["summary"] = dict(
            loop_mean_seconds_per_chunk=float(loop_per_chunk), graph_mean_seconds_per_chunk=float(graph_per_chunk),
            mean_loop_over_graph_ratio=float(loop_per_chunk/graph_per_chunk),
            per_pair_ratios=[p["loop_over_graph_wall_ratio"] for p in report["pairs"]],
            graph_incremental_setup_seconds=extra,
            graph_extra_setup_break_even_chunks=math.ceil(extra/saving) if saving > 0 else None,
            amortization_warning="Arithmetic estimates using measured average fixed-input chunks; not measured dFBA trajectories or scaling forecasts",
            setup_inclusive_estimates=[dict(chunks=count,
                loop_seconds=float(common+reset_loop+count*loop_per_chunk),
                graph_seconds=float(common+extra+reset_graph+count*graph_per_chunk),
                graph_amortized_seconds_per_chunk=float(graph_per_chunk+extra/count)) for count in counts],
            all_equal_iteration_equivalence_checks_passed=True,
            final_graph_certificate_pass_count=sum(r["certificate_passed"] for r in report["pairs"][-1]["final_original_metrics"]),
            LP_acceptance_warning="A faster rejected iterate is not an accepted-LP speedup")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      failure_policy="No solver/environment modification or silent fallback")
        raise
    finally:
        report["diagnostic_total_seconds_including_IO_setup_validation"] = time.perf_counter()-total_started
        save()


if __name__ == "__main__":
    main()
