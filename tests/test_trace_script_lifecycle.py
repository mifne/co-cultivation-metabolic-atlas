import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import scripts.benchmark_block_lp_trace as benchmark
import scripts.capture_dfba_lp_trace as capture


def completed_manifest(path, *, entries=None):
    path.mkdir()
    (path / 'manifest.json').write_text(json.dumps(dict(
        status='completed', role='development_diagnostic_not_training',
        completed_steps=[1], model_fingerprints={'toy': 'hash'}, seeds=[101],
        entries=[] if entries is None else entries)))


def benchmark_args(monkeypatch, trace, output):
    monkeypatch.setattr('sys.argv', ['benchmark_block_lp_trace.py',
        '--trace', str(trace), '--output', str(output), '--steps', '1',
        '--stages', 'maxmin'])


def test_benchmark_saves_gpu_startup_failure(tmp_path, monkeypatch):
    trace, output = tmp_path / 'trace', tmp_path / 'report.json'
    completed_manifest(trace)
    benchmark_args(monkeypatch, trace, output)

    class FailedGpu:
        def __init__(self, **kwargs):
            raise RuntimeError('gpu startup failed')

    monkeypatch.setattr(benchmark, 'GpuBlockLP', FailedGpu)
    with pytest.raises(RuntimeError, match='gpu startup failed'):
        benchmark.main()
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert report['error_phase'] == 'gpu_initialization'
    assert report['error_type'] == 'RuntimeError'
    assert report['timing_scope']['gpu_library_setup_in_ratio'] is False
    assert report['timing_scope']['cpu_worker_threads_start_lazily_inside_cpu_seconds'] is True


def test_benchmark_close_does_not_mask_solve_error_and_final_status_is_saved(
        tmp_path, monkeypatch):
    trace, output = tmp_path / 'trace', tmp_path / 'report.json'
    entry = dict(step=1, stage='maxmin', environment_id=0,
                 problem_sha256='fixed-input')
    completed_manifest(trace, entries=[entry])
    benchmark_args(monkeypatch, trace, output)
    problem = (csr_matrix([[1.]]), np.array([1.]), np.array([0.]),
               np.array([1.]), np.array([-1.]), 0)
    monkeypatch.setattr(benchmark, 'load_trace_lp',
                        lambda directory, item: (problem, np.array([1.]), np.array([0.])))

    class FakeRuntime:
        @staticmethod
        def deviceSynchronize():
            return None

    class FakeGpu:
        cp = SimpleNamespace(cuda=SimpleNamespace(runtime=FakeRuntime()))

        def __init__(self, **kwargs):
            self.history = []

    class FailedCpu:
        def __init__(self, workers):
            pass

        def solve_batch(self, requests):
            raise ValueError('cpu solve is the primary failure')

        def close(self):
            raise RuntimeError('cpu close is secondary')

    monkeypatch.setattr(benchmark, 'GpuBlockLP', FakeGpu)
    monkeypatch.setattr(benchmark, 'RepeatedCpuLP', FailedCpu)
    with pytest.raises(ValueError, match='primary failure'):
        benchmark.main()
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert report['error_type'] == 'ValueError'
    assert report['error'] == 'cpu solve is the primary failure'
    assert report['cleanup_errors'] == [dict(
        phase='cpu_close_step_1_maxmin', error_type='RuntimeError',
        error='cpu close is secondary')]


def capture_args(monkeypatch, output):
    monkeypatch.setattr('sys.argv', ['capture_dfba_lp_trace.py',
        '--output', str(output), '--steps', '1', '--environments', '1',
        '--workers', '1'])


def test_capture_saves_environment_startup_failure(tmp_path, monkeypatch):
    output = tmp_path / 'capture'
    capture_args(monkeypatch, output)
    monkeypatch.setattr(capture, 'environment',
                        lambda seed: (_ for _ in ()).throw(RuntimeError('environment failed')))
    with pytest.raises(RuntimeError, match='environment failed'):
        capture.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['error_phase'] == 'environment_setup'
    assert manifest['error_type'] == 'RuntimeError'
    assert manifest['lifecycle_seconds'] >= 0


class ToyEnvironment:
    def __init__(self):
        self.simulator = SimpleNamespace(models={'toy': object()},
            _cooperative_solver=SimpleNamespace(_linprog_options={}))

    def reset(self, *, seed):
        return None


def configured_capture(monkeypatch, *, drive, cpu_type):
    monkeypatch.setattr(capture, 'describe_teacher_environment', lambda env: {'test_double': True})
    monkeypatch.setattr(capture, 'snapshot', lambda env: {'toy_initial_state': True})
    monkeypatch.setattr(capture, 'environment',
        lambda seed: (ToyEnvironment(), SimpleNamespace(n_fluxes=1)))
    monkeypatch.setattr(capture, 'model_fingerprint', lambda model: 'fixed')
    monkeypatch.setattr(capture, 'matched_actions',
                        lambda seeds, steps: np.zeros((len(seeds), steps, 1)))
    monkeypatch.setattr(capture, 'RepeatedCpuLP', cpu_type)
    monkeypatch.setattr(capture, 'drive_microbatch', drive)


def test_capture_close_does_not_mask_primary_and_save_is_attempted(tmp_path, monkeypatch):
    output = tmp_path / 'capture'
    capture_args(monkeypatch, output)

    class FailedCloseCpu:
        def __init__(self, workers, *, n_fluxes):
            pass

        def close(self):
            raise RuntimeError('cpu close is secondary')

    def failed_drive(*args):
        raise ValueError('capture is the primary failure')

    configured_capture(monkeypatch, drive=failed_drive, cpu_type=FailedCloseCpu)
    with pytest.raises(ValueError, match='primary failure'):
        capture.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['error_type'] == 'ValueError'
    assert manifest['error'] == 'capture is the primary failure'
    assert manifest['cleanup_errors'] == [dict(
        phase='cpu_close', error_type='RuntimeError', error='cpu close is secondary')]


def test_capture_close_failure_after_success_is_saved_and_raised(tmp_path, monkeypatch):
    output = tmp_path / 'capture'
    capture_args(monkeypatch, output)

    class FailedCloseCpu:
        def __init__(self, workers, *, n_fluxes):
            pass

        def close(self):
            raise RuntimeError('cpu close failed')

    def successful_drive(environments, actions, service, snapshot):
        service.manifest['entries'].extend([{}, {}, {}])
        return [{'endpoint': 1.}]

    configured_capture(monkeypatch, drive=successful_drive, cpu_type=FailedCloseCpu)
    with pytest.raises(RuntimeError, match='cpu close failed'):
        capture.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['error_phase'] == 'cpu_close'
    assert manifest['completed_steps'] == [1]
    assert manifest['cleanup_errors'][0]['error'] == 'cpu close failed'
