"""CPU-only manufactured/algebra/lifecycle checks; root explicitly runs CUDA."""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.probe_cudss_graph_replay import (
    FixedFactorGraphProbe, GraphCaptureFailure, _json_safe, capture_once,
    evaluate_snapshot, manufactured_rhs, manufactured_system, run_probe,
)


@pytest.mark.parametrize('batch,size', [(1, 2), (4, 17), (8, 33)])
def test_manufactured_matrix_is_spd_and_rhs_solutions_are_distinct(batch, size):
    matrix = manufactured_system(batch, size)
    np.testing.assert_array_equal(matrix, matrix.transpose(0, 2, 1))
    assert np.all(np.linalg.eigvalsh(matrix) > 0.)
    rhs0, x0 = manufactured_rhs(matrix, 0)
    rhs1, x1 = manufactured_rhs(matrix, 1)
    assert not np.array_equal(rhs0, rhs1)
    assert not np.array_equal(x0, x1)
    np.testing.assert_allclose(np.einsum('bij,bj->bi', matrix, x1), rhs1)


@pytest.mark.parametrize('batch,size,revision', [(0, 17, 0), (33, 17, 0), (4, 1, 0),
                                               (4, 129, 0), (4, 17, 2), (True, 17, 0)])
def test_bounded_manufactured_configuration(batch, size, revision):
    with pytest.raises(ValueError): manufactured_system(batch, size, revision)


def _correct_snapshot():
    matrix = manufactured_system(3, 7)
    rhs, expected = manufactured_rhs(matrix, 1)
    residual = np.stack([a@x-b for a, x, b in zip(matrix, expected, rhs)])
    absolute = np.max(np.abs(residual), axis=1)
    metrics = np.stack((absolute, absolute/np.maximum(1., np.max(np.abs(rhs), axis=1)),
                        np.ones(3)), axis=1)
    return matrix, rhs, expected, expected.copy(), residual, metrics


def test_independent_host_check_accepts_correct_original_equations():
    assert evaluate_snapshot(*_correct_snapshot())['passed']


@pytest.mark.parametrize('corruption', ['old_answer', 'residual', 'metrics', 'nan', 'false_finite'])
def test_independent_checks_reject_frozen_or_invalid_outputs(corruption):
    matrix, rhs, expected, output, residual, metrics = _correct_snapshot()
    if corruption == 'old_answer': output = manufactured_rhs(matrix, 0)[1]
    elif corruption == 'residual': residual[1, 0] = 1.
    elif corruption == 'metrics': metrics[1, 0] = 1.
    elif corruption == 'nan': output[1, 0] = np.nan
    else: metrics[1, 2] = 0.
    assert not evaluate_snapshot(matrix, rhs, expected, output, residual, metrics)['passed']


def test_intentional_invalid_lane_requires_explicit_nan_output_not_sanitized_zero():
    matrix, rhs, expected, output, residual, metrics = _correct_snapshot()
    rhs[0, 0] = np.nan
    output[0] = np.nan
    residual[0] = np.nan
    metrics[0] = [np.inf, np.inf, 0.]
    assert evaluate_snapshot(matrix, rhs, expected, output, residual, metrics, invalid_lanes=(0,))['passed']
    output[0] = 0.
    assert not evaluate_snapshot(matrix, rhs, expected, output, residual, metrics, invalid_lanes=(0,))['passed']


class _Stream:
    def __init__(self, fail_begin=False, fail_end=False):
        self.fail_begin, self.fail_end = fail_begin, fail_end
        self.log = []
        self.graph = object()
    def begin_capture(self):
        self.log.append('begin')
        if self.fail_begin: raise RuntimeError('unsupported begin')
    def end_capture(self):
        self.log.append('end')
        if self.fail_end: raise RuntimeError('invalidated end')
        return self.graph


def test_capture_success_pairs_begin_end_exactly_once():
    stream = _Stream()
    assert capture_once(stream, lambda: stream.log.append('enqueue')) is stream.graph
    assert stream.log == ['begin', 'enqueue', 'end']


@pytest.mark.parametrize('location', ['begin', 'enqueue', 'end', 'enqueue_and_end'])
def test_capture_failure_explicit_without_ordinary_fallback(location):
    stream = _Stream(fail_begin=location == 'begin', fail_end='end' in location)
    def enqueue():
        stream.log.append('enqueue')
        if 'enqueue' in location: raise RuntimeError('unsupported CUDA operation')
    with pytest.raises(GraphCaptureFailure): capture_once(stream, enqueue)
    assert stream.log.count('begin') == 1
    assert stream.log.count('end') == (0 if location == 'begin' else 1)
    assert stream.log.count('enqueue') == (0 if location == 'begin' else 1)


def test_graph_probe_close_drains_before_destroying_graph_then_factor():
    log = []
    class Graph:
        def __del__(self): log.append('graph_destroy')
    class Factor:
        def close(self): log.append('factor_close')
    probe = object.__new__(FixedFactorGraphProbe)
    probe.closed = False
    probe.owner_thread = threading.get_ident()
    probe.device = 0
    probe.stream = SimpleNamespace(ptr=123, synchronize=lambda: log.append('drain'))
    probe.cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: 0), get_current_stream=lambda: probe.stream))
    probe.graph = Graph()
    probe.factor = Factor()
    probe.close()
    assert log == ['drain', 'graph_destroy', 'factor_close']
    probe.close()
    assert len(log) == 3


def test_graph_probe_close_rejects_wrong_stream_without_destroying_live_buffers():
    probe = object.__new__(FixedFactorGraphProbe)
    probe.closed = False
    probe.owner_thread = threading.get_ident()
    probe.device = 0
    probe.stream = SimpleNamespace(ptr=123)
    probe.cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: 0),
        get_current_stream=lambda: SimpleNamespace(ptr=456)))
    with pytest.raises(RuntimeError, match='original owner'):
        probe.close()
    assert not probe.closed


def test_no_graph_means_replay_raises_instead_of_running_direct():
    probe = object.__new__(FixedFactorGraphProbe)
    probe._context = lambda: None
    probe.graph = None
    probe.enqueue_fixed = lambda: pytest.fail('No implicit ordinary fallback')
    with pytest.raises(RuntimeError, match='No successfully captured graph'): probe.replay()


def test_capture_body_has_no_materializing_validation_or_allocator_calls():
    path = Path(__file__).resolve().parents[1]/'scripts/probe_cudss_graph_replay.py'
    tree = ast.parse(path.read_text())
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    forbidden = {'get', 'asnumpy', 'synchronize', 'bool', 'empty', 'empty_like', 'zeros',
                 'zeros_like', 'ones', 'ones_like', 'asarray', 'copy', 'copyto'}
    for name in ('enqueue_fixed', '_context', '_pointers'):
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Call):
                function = node.func
                call_name = function.id if isinstance(function, ast.Name) else (
                    function.attr if isinstance(function, ast.Attribute) else None)
                assert call_name not in forbidden, (name, call_name)


def test_context_rejects_owner_thread_migration_before_native_calls():
    probe = object.__new__(FixedFactorGraphProbe)
    probe.closed = False
    probe.owner_thread = threading.get_ident()+1
    with pytest.raises(RuntimeError, match='owner thread'): probe._context()


def test_nonfinite_json_values_are_not_silently_zeroed():
    assert _json_safe([np.inf, np.nan, -np.inf, np.bool_(False)]) == ['Infinity', 'NaN', '-Infinity', False]


def test_cuda_installed_fixed_factor_capture_replay():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1: pytest.skip('CUDA device required')
    result = run_probe(batch=2, size=7, replays=3)
    # Capability refusal is an explicit, saved outcome rather than success.
    if result['status'] == 'graph_capture_unsupported_or_failed':
        assert not result['graph_capture_succeeded']
        assert not result['graph_replay_qualified']
        assert not result['ordinary_fallback_used']
        pytest.skip('Installed runtime explicitly rejected graph capture: '+result['error'])
    assert result['status'] == 'qualified_manufactured_fixed_factor_graph_replay', result
    assert result['graph_replay_qualified'] and result['cleanup_succeeded']
    assert result['factor_count'] == result['analysis_count'] == 1
    assert result['graph_launches'] == 5  # Three distinct inputs, invalid lane, recovery.
    assert result['buffer_addresses_unchanged']
