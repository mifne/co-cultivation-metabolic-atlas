"""Bounded cuDSS 0.7.x CUDA-graph capability probe, not an LP speedup claim.

Manufactured SPD systems, one numeric factorization, fixed retained buffers,
and distinct right-hand sides test actual replay rather than frozen answers.
No production solver is changed. Graph failure is saved explicitly; it never
falls back to an ordinary solve while claiming graph success.
"""

import argparse
import gc
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


_SANITIZE = r'''
extern "C" __global__ void prepare_rhs(
    const double* input, double* rhs, int* valid, const int batch, const int n) {
    const int lane=blockIdx.x*blockDim.x+threadIdx.x;
    if(lane>=batch)return;
    bool finite=true;
    for(int j=0;j<n;++j)finite=finite&&isfinite(input[(long long)lane*n+j]);
    valid[lane]=finite;
    for(int j=0;j<n;++j)rhs[(long long)lane*n+j]=finite?input[(long long)lane*n+j]:0.0;
}
'''

_CHECK = r'''
extern "C" __global__ void check_original_residual(
    const double* matrix,const double* input,const double* solution,const int* valid,
    double* output,double* residual,double* metrics,const int batch,const int n) {
    const int lane=blockIdx.x*blockDim.x+threadIdx.x;
    if(lane>=batch)return;
    const long long v=(long long)lane*n, a=(long long)lane*n*n;
    const double invalid=__longlong_as_double(0x7ff8000000000000LL);
    const double infinite=__longlong_as_double(0x7ff0000000000000LL);
    bool finite=valid[lane]!=0;
    double error=0.0,scale=0.0;
    for(int i=0;i<n;++i)finite=finite&&isfinite(solution[v+i]);
    for(int i=0;i<n;++i){
        double product=0.0;
        for(int j=0;j<n;++j)product+=matrix[a+(long long)i*n+j]*solution[v+j];
        const double defect=product-input[v+i];
        finite=finite&&isfinite(product)&&isfinite(defect);
        residual[v+i]=defect;
        error=fmax(error,fabs(defect));
        scale=fmax(scale,fabs(input[v+i]));
    }
    for(int i=0;i<n;++i){
        output[v+i]=finite?solution[v+i]:invalid;
        if(!finite)residual[v+i]=invalid;
    }
    metrics[(long long)lane*3]=finite?error:infinite;
    metrics[(long long)lane*3+1]=finite?error/fmax(1.0,scale):infinite;
    metrics[(long long)lane*3+2]=finite?1.0:0.0;
}
'''


class GraphCaptureFailure(RuntimeError):
    pass


def manufactured_system(batch, size, revision=0):
    """SPD by positive diagonal + outer product, no numerical solve/oracle."""
    if (type(batch) is not int or not 1 <= batch <= 32 or type(size) is not int
            or not 2 <= size <= 128 or type(revision) is not int or revision not in (0, 1)):
        raise ValueError('Manufactured probe bounds: batch1..32, size2..128, revision0/1')
    vector = np.linspace(.2, .8, size)
    return np.stack([np.diag(2.+.1*np.arange(size)+.05*i+.25*revision)
        + np.outer(vector, vector)*(.1+.02*revision) for i in range(batch)])


def manufactured_rhs(matrices, index):
    if type(index) is not int or index < 0:
        raise ValueError('Nonnegative manufactured RHS index required')
    batch, size, _ = matrices.shape
    x = np.cos(np.arange(size)[None, :]*.23 + index*.37 + np.arange(batch)[:, None]*.11)
    x += (index+1)*.15
    return np.einsum('bij,bj->bi', matrices, x), x


def capture_once(stream, enqueue):
    """Always pair begin/end, including failed capture; no ordinary fallback."""
    begun = False
    try:
        stream.begin_capture()
        begun = True
        enqueue()
    except Exception as error:
        if begun:
            try:
                abandoned = stream.end_capture()
                del abandoned
            except Exception as ending:
                error.add_note(f'Ending failed capture also reported: {ending}')
        raise GraphCaptureFailure(f'CUDA graph capture failed: {error}') from error
    try:
        return stream.end_capture()
    except Exception as error:
        raise GraphCaptureFailure(f'CUDA graph finalization failed: {error}') from error


def evaluate_snapshot(matrix, rhs, expected, output, residual, device_metrics,
                      *, tolerance=1e-10, invalid_lanes=()):
    """Independent host arithmetic checks only; never an optimizer or solve."""
    batch, size, _ = matrix.shape
    if (output.shape != (batch, size) or residual.shape != (batch, size)
            or device_metrics.shape != (batch, 3) or expected.shape != output.shape):
        raise ValueError('Full returned vectors/metrics required')
    rows = []
    for lane in range(batch):
        if lane in invalid_lanes:
            rejected = (not np.isfinite(output[lane]).any()
                and not np.isfinite(residual[lane]).any()
                and device_metrics[lane, 2] == 0.
                and np.isinf(device_metrics[lane, :2]).all())
            rows.append(dict(lane=lane, intentional_invalid_rhs=True, passed=bool(rejected)))
            continue
        host_residual = matrix[lane]@output[lane] - rhs[lane]
        rhs_scale = max(1., float(np.max(np.abs(rhs[lane]))))
        x_scale = max(1., float(np.max(np.abs(expected[lane]))))
        relative_residual = float(np.max(np.abs(host_residual))/rhs_scale)
        solution_error = float(np.max(np.abs(output[lane]-expected[lane]))/x_scale)
        residual_agrees = bool(np.allclose(residual[lane], host_residual, atol=1e-12*rhs_scale, rtol=1e-10))
        metrics_agree = bool(np.isclose(device_metrics[lane, 0], np.max(np.abs(residual[lane])),
            atol=1e-14*rhs_scale, rtol=1e-10) and np.isclose(device_metrics[lane, 1],
            device_metrics[lane, 0]/rhs_scale, atol=1e-14, rtol=1e-10))
        passed = bool(np.isfinite(output[lane]).all() and np.isfinite(residual[lane]).all()
            and np.isfinite(device_metrics[lane]).all() and device_metrics[lane, 2] == 1.
            and relative_residual <= tolerance and solution_error <= tolerance
            and residual_agrees and metrics_agree)
        rows.append(dict(lane=lane, intentional_invalid_rhs=False, passed=passed,
            host_relative_residual=relative_residual, known_solution_relative_error=solution_error,
            device_absolute_residual=float(device_metrics[lane, 0]),
            device_relative_residual=float(device_metrics[lane, 1]),
            residual_agrees=residual_agrees, metrics_agree=metrics_agree))
    return dict(passed=all(r['passed'] for r in rows), rows=rows)


class FixedFactorGraphProbe:
    """Own the factor, fixed buffers and graph until all work is drained."""
    def __init__(self, matrices, *, refinement_steps=0):
        import cupy as cp
        from src.gpu_sparse_factor import UniformCudssFactor
        self.cp = cp
        self.owner_thread = threading.get_ident()
        self.stream = cp.cuda.get_current_stream()
        self.device = cp.cuda.runtime.getDevice()
        self.matrices = np.asarray(matrices, dtype=np.float64).copy()
        if (self.matrices.ndim != 3 or self.matrices.shape[1] != self.matrices.shape[2]
                or not np.isfinite(self.matrices).all()):
            raise ValueError('Finite square SPD manufactured batch required')
        self.batch, self.size, _ = self.matrices.shape
        self.factor = None
        self.graph = None
        self.closed = False
        self.graph_launches = 0
        self.direct_enqueues = 0
        try:
            pattern = csr_matrix(np.ones((self.size, self.size)))
            self.factor = UniformCudssFactor(pattern, batch_size=self.batch, matrix_type='spd',
                                              refinement_steps=refinement_steps)
            self.matrix = cp.asarray(self.matrices)
            self.factor.factor(self.matrix.reshape(self.batch, -1))
            self.input = cp.zeros((self.batch, self.size), dtype=cp.float64)
            self.valid = cp.zeros(self.batch, dtype=cp.int32)
            self.output = cp.zeros_like(self.input)
            self.residual = cp.zeros_like(self.input)
            self.metrics = cp.zeros((self.batch, 3), dtype=cp.float64)
            self.prepare_kernel = cp.RawKernel(_SANITIZE, 'prepare_rhs', options=('--fmad=false',))
            self.check_kernel = cp.RawKernel(_CHECK, 'check_original_residual', options=('--fmad=false',))
            self.pointers = self._pointers()
            self.stream.synchronize()
        except BaseException as error:
            try: self.close()
            except BaseException as cleanup:
                error.add_note(f'Constructor cleanup also failed: {cleanup}')
            raise

    def _pointers(self):
        return tuple(v.data.ptr for v in (self.matrix, self.input, self.valid, self.output,
            self.residual, self.metrics, self.factor.values, self.factor.rhs, self.factor.solution))

    def _context(self):
        if self.closed or threading.get_ident() != self.owner_thread:
            raise RuntimeError('Graph probe closed or owner thread changed')
        self.factor._context()
        if self._pointers() != self.pointers:
            raise RuntimeError('A captured device-buffer address was replaced')
        for value in (self.matrix, self.input, self.valid, self.output, self.residual, self.metrics):
            if value.device.id != self.device:
                raise RuntimeError('Graph buffer belongs to another CUDA device')

    def set_rhs(self, rhs, *, allow_intentional_nonfinite=False):
        self._context()
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (self.batch, self.size) or (
                not allow_intentional_nonfinite and not np.isfinite(rhs).all()):
            raise ValueError('Matching finite RHS required outside explicit rejection test')
        self.cp.copyto(self.input, self.cp.asarray(rhs))
        # Deliberately outside capture/timed launch: input preparation is measured separately.
        self.stream.synchronize()

    def enqueue_fixed(self):
        """Only device kernels and native SOLVE; no host numerical materialization."""
        self._context()
        grid = ((self.batch+127)//128,)
        arguments = (np.int32(self.batch), np.int32(self.size))
        self.prepare_kernel(grid, (128,), (self.input, self.factor.rhs, self.valid, *arguments))
        # Already analyzed/factorized, hybrid modes disabled by existing adapter.
        # Capture is explicitly tested on this ABI instead of assumed supported.
        self.factor._execute(1008)
        self.check_kernel(grid, (128,), (self.matrix, self.input, self.factor.solution,
            self.valid, self.output, self.residual, self.metrics, *arguments))

    def warmup(self):
        self._context()
        for _ in range(2):
            self.enqueue_fixed()
            self.direct_enqueues += 1
        self.stream.synchronize()

    def capture(self):
        self._context()
        if self.graph is not None:
            raise RuntimeError('Destroy previous graph before recapture')
        self.graph = capture_once(self.stream, self.enqueue_fixed)

    def replay(self):
        self._context()
        if self.graph is None:
            raise RuntimeError('No successfully captured graph; no direct fallback')
        self.graph.launch(stream=self.stream)
        self.graph_launches += 1

    def snapshot(self):
        self._context()
        self.stream.synchronize()
        return self.output.get(), self.residual.get(), self.metrics.get()

    def destroy_graph(self):
        self._context()
        self.stream.synchronize()
        self.graph = None
        gc.collect()  # Graph destruction precedes any factor/buffer destruction.

    def close(self):
        if self.closed: return
        if (threading.get_ident() != self.owner_thread
                or self.cp.cuda.runtime.getDevice() != self.device
                or self.cp.cuda.get_current_stream().ptr != self.stream.ptr):
            raise RuntimeError('Close must use the original owner thread, device and stream')
        errors = []
        try: self.stream.synchronize()
        except BaseException as error: errors.append(error)
        self.graph = None
        gc.collect()
        if self.factor is not None:
            try: self.factor.close()
            except BaseException as error: errors.append(error)
        self.closed = True
        if errors: raise BaseExceptionGroup('Graph probe cleanup failed', errors)


def run_probe(*, batch=4, size=17, replays=5, refinement_steps=0):
    if type(replays) is not int or not 2 <= replays <= 100:
        raise ValueError('At least two distinct replay RHSs, at most100')
    matrix = manufactured_system(batch, size)
    import cupy as cp
    stream = cp.cuda.Stream(non_blocking=True)
    probe = None
    record = dict(status='not_started', graph_capture_succeeded=False,
        graph_replay_qualified=False, batch=batch, size=size, replays=replays,
        refinement_steps=refinement_steps, cpu_optimizer_calls=0, cpu_numerical_solve_calls=0,
        cpu_matvec_used_for_manufactured_rhs=True, graph_trials=[], direct_control_trials=[],
        timing_scope='micro-capability probe; launch+completion only, input preparation/checks separate; not LP/PPO speedup',
        factor_refresh_after_capture='not_attempted: captured internal factor-address lifetime across refactor unproven',
        factor_refresh_safe_next_step='drain/destroy old graph, refactor, warmup and recapture; never silently reuse stale graph',
        installed_package_version=version('nvidia-cudss-cu12'), cupy_version=cp.__version__,
        reference_urls=['https://docs.nvidia.com/cuda/cudss/general.html#cuda-graphs-support',
            'https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html'],
        version_caveat='Official live cuDSS documentation includes0.8 migration; adapter/header0.7.x is tested explicitly')
    try:
        with stream:
            started = time.perf_counter()
            probe = FixedFactorGraphProbe(matrix, refinement_steps=refinement_steps)
            record['setup_seconds'] = time.perf_counter()-started
            record['runtime_cudss_version'] = probe.factor.version
            record['device_name'] = cp.cuda.runtime.getDeviceProperties(probe.device)['name'].decode()
            record['buffer_pointers'] = list(probe.pointers)
            rhs, expected = manufactured_rhs(matrix, 0)
            probe.set_rhs(rhs)
            probe.warmup()
            record['warmup_validation'] = evaluate_snapshot(matrix, rhs, expected, *probe.snapshot())
            if not record['warmup_validation']['passed']:
                record['status'] = 'ordinary_warmup_numerically_failed'
                return record
            started = time.perf_counter()
            probe.capture()
            record['capture_seconds'] = time.perf_counter()-started
            record['graph_capture_succeeded'] = True
            previous = None
            outputs_changed = []
            for index in range(replays):
                rhs, expected = manufactured_rhs(matrix, index+1)
                for kind in (('graph', 'direct') if index % 2 == 0 else ('direct', 'graph')):
                    started = time.perf_counter()
                    probe.set_rhs(rhs)
                    preparation = time.perf_counter()-started
                    started = time.perf_counter()
                    if kind == 'graph': probe.replay()
                    else:
                        probe.enqueue_fixed()
                        probe.direct_enqueues += 1
                    stream.synchronize()
                    seconds = time.perf_counter()-started
                    snapshot = probe.snapshot()
                    result = evaluate_snapshot(matrix, rhs, expected, *snapshot)
                    result.update(index=index, rhs_sha256=hashlib.sha256(rhs.tobytes()).hexdigest(),
                        input_preparation_seconds=preparation, launch_completion_seconds=seconds,
                        solution_sha256=hashlib.sha256(snapshot[0].tobytes()).hexdigest())
                    record['graph_trials' if kind == 'graph' else 'direct_control_trials'].append(result)
                    if kind == 'graph':
                        if previous is not None:
                            outputs_changed.append(not np.array_equal(snapshot[0], previous))
                        previous = snapshot[0].copy()
            # Deliberate bad lane must stay rejected, while other graph lanes solve.
            rhs, expected = manufactured_rhs(matrix, replays+1)
            rhs[0, 0] = np.nan
            probe.set_rhs(rhs, allow_intentional_nonfinite=True)
            probe.replay()
            record['nonfinite_lane_rejection'] = evaluate_snapshot(matrix, rhs, expected,
                *probe.snapshot(), invalid_lanes=(0,))
            # Restore a valid RHS to prove one bad lane did not poison later replay.
            rhs, expected = manufactured_rhs(matrix, replays+2)
            probe.set_rhs(rhs)
            probe.replay()
            record['recovery_validation'] = evaluate_snapshot(matrix, rhs, expected, *probe.snapshot())
            record['distinct_rhs_proved'] = len({r['rhs_sha256'] for r in record['graph_trials']}) == replays
            record['outputs_changed_between_replays'] = bool(all(outputs_changed))
            record['graph_replay_qualified'] = bool(
                all(t['passed'] for t in record['graph_trials']+record['direct_control_trials'])
                and record['distinct_rhs_proved'] and record['outputs_changed_between_replays']
                and record['nonfinite_lane_rejection']['passed'] and record['recovery_validation']['passed'])
            record['status'] = ('qualified_manufactured_fixed_factor_graph_replay'
                if record['graph_replay_qualified'] else 'graph_replay_numerical_validation_failed')
            record['graph_launches'] = probe.graph_launches
            record['direct_enqueues_including_warmup'] = probe.direct_enqueues
            record['factor_count'] = probe.factor.factor_count
            record['analysis_count'] = probe.factor.analysis_count
            record['buffer_addresses_unchanged'] = probe._pointers() == probe.pointers
    except GraphCaptureFailure as error:
        record.update(status='graph_capture_unsupported_or_failed', error=str(error),
            error_type=type(error).__name__, ordinary_fallback_used=False)
    except Exception as error:
        record.update(status='probe_runtime_failed', error=str(error),
            error_type=type(error).__name__, ordinary_fallback_used=False,
            graph_replay_qualified=False)
    finally:
        if probe is not None:
            try:
                with stream: probe.close()
                record['cleanup_succeeded'] = True
            except Exception as error:
                record.update(cleanup_succeeded=False, cleanup_error=str(error),
                    graph_replay_qualified=False, status='probe_cleanup_failed')
    return record


def _json_safe(value):
    if isinstance(value, dict): return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--size', type=int, default=17)
    parser.add_argument('--replays', type=int, default=5)
    parser.add_argument('--refinement-steps', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Never overwrite graph capability results')
    record = run_probe(batch=args.batch, size=args.size, replays=args.replays,
                       refinement_steps=args.refinement_steps)
    sources = ('scripts/probe_cudss_graph_replay.py', 'src/gpu_sparse_factor.py')
    record['sources'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    record['source_snapshots'] = {p: (ROOT/p).read_text() for p in sources}
    dist = distribution('nvidia-cudss-cu12')
    headers = [dist.locate_file(p) for p in dist.files if str(p).endswith('/include/cudss.h')]
    record['installed_header'] = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                    for p in headers]
    serialized = json.dumps(_json_safe(record), indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output: output.write(serialized)
    print(json.dumps(dict(status=record['status'], qualified=record['graph_replay_qualified'],
                         output=str(args.output))), flush=True)


if __name__ == '__main__': main()
