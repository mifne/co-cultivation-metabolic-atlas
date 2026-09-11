"""CPU-only scheduler contracts plus explicitly selected tiny CUDA tests."""

import threading

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_stream_partition import cooperative_hooks, partition_indices, solve_cooperatively
from scripts.benchmark_gpu_stream_partitions import _independent_direct_gate, _json_safe, _ordered_rows


class _Stream:
    current = None
    def __init__(self, ptr):
        self.ptr = ptr
        self.stack = []
        self.drains = 0
    def __enter__(self):
        self.stack.append(_Stream.current)
        _Stream.current = self
        return self
    def __exit__(self, *args):
        _Stream.current = self.stack.pop()
    def synchronize(self): self.drains += 1


class _Factor:
    def __init__(self, stream):
        self.stream = stream
        self.thread = threading.get_ident()
    def _context(self):
        assert threading.get_ident() == self.thread
        assert _Stream.current is self.stream


class _Event:
    instances = []
    def __init__(self):
        self.ready = True
        self.records = 0
        self.waits = 0
        self.stream = None
        _Event.instances.append(self)
    def record(self, stream):
        assert self.ready, 'A pending event was overwritten'
        assert _Stream.current is stream
        self.ready = False
        self.records += 1
        self.stream = stream
    @property
    def done(self): return self.ready
    def synchronize(self):
        self.waits += 1
        self.ready = True


class _Solver:
    def __init__(self, stream, identity, rounds=2, fail=False):
        self.factor = _Factor(stream)
        self.identity, self.rounds, self.fail = identity, rounds, fail
        self.trace = []
    def _factor_newton(self, ratio):
        self.factor._context()
        self.trace.append(('factor', ratio))
        return ratio + 1
    def _factor_solve(self, rhs):
        self.factor._context()
        self.trace.append(('solve', rhs))
        return rhs * 2
    def solve(self, **kwargs):
        answer = self.identity
        for iteration in range(self.rounds):
            answer = self._factor_newton(answer)
            if self.fail:
                raise ArithmeticError('Deliberate numerical workspace exception')
            answer = self._factor_solve(answer)
        self.factor._context()
        return {'identity': self.identity, 'answer': answer, 'kwargs': kwargs}


def _mock_workspaces(rounds=(2, 2)):
    streams = [_Stream(i+101) for i in range(len(rounds))]
    return [_Solver(s, i, r) for i, (s, r) in enumerate(zip(streams, rounds))], streams


@pytest.mark.parametrize('partitions', [1, 2, 4])
def test_partition_covers_exact_same_32_inputs_once(partitions):
    groups = partition_indices(32, partitions)
    assert all(len(group) == 32//partitions for group in groups)
    assert sorted(i for group in groups for i in group) == list(range(32))
    assert groups[0] == tuple(range(0, 32, partitions))


@pytest.mark.parametrize('batch,partitions', [(0, 1), (32, 0), (3, 2), (2, 4), (True, 1), (32, 2.)])
def test_invalid_partition_is_rejected(batch, partitions):
    with pytest.raises(ValueError): partition_indices(batch, partitions)


@pytest.mark.parametrize('mode', ['factor-only', 'factor-and-solve'])
def test_mixed_finish_times_restore_stream_and_original_helpers(mode):
    _Event.instances = []
    solvers, streams = _mock_workspaces((0, 1, 3, 2))
    results, stats = solve_cooperatively(solvers, streams,
        solve_kwargs={'iterations': 7}, yield_mode=mode, event_factory=_Event)
    assert [r['identity'] for r in results] == list(range(4))
    assert [r['answer'] for r in results] == [0, 4, 30, 18]
    assert all(r['kwargs'] == {'iterations': 7} for r in results)
    assert all('_factor_newton' not in s.__dict__ and '_factor_solve' not in s.__dict__ for s in solvers)
    assert _Stream.current is None
    assert all(stream.drains == 1 for stream in streams)
    assert stats['max_pending_shards'] == 3
    assert stats['event_waits'] == sum(s.rounds for s in solvers)*(1 if mode == 'factor-only' else 2)
    assert all(event.records == event.waits for event in _Event.instances)
    assert stats['yield_counts'][2]['_factor_newton'] == 3
    assert stats['yield_counts'][2]['_factor_solve'] == (3 if mode == 'factor-and-solve' else 0)


def test_ready_events_never_require_synchronize_wait():
    class ReadyEvent(_Event):
        @property
        def done(self):
            self.ready = True
            return True
    solvers, streams = _mock_workspaces()
    _, stats = solve_cooperatively(solvers, streams, event_factory=ReadyEvent)
    assert stats['event_waits'] == 0
    assert stats['max_pending_shards'] == 2


def test_exception_unwinds_paused_workers_and_restores_all_helpers():
    solvers, streams = _mock_workspaces((4, 2, 3))
    solvers[1].fail = True
    with pytest.raises(ArithmeticError, match='Deliberate'):
        solve_cooperatively(solvers, streams, event_factory=_Event)
    assert all('_factor_newton' not in s.__dict__ and '_factor_solve' not in s.__dict__ for s in solvers)
    assert all(stream.drains == 2 for stream in streams)
    assert _Stream.current is None
    assert len(solvers[0].trace) < 8  # Cancelled, not silently run to completion.


def test_hooks_restore_preexisting_instance_callable_by_identity():
    solver, stream = _mock_workspaces((1,))
    solver = solver[0]
    original = lambda value: value + 123
    solver._factor_newton = original
    with cooperative_hooks(solver, lambda phase: None):
        assert solver._factor_newton(1) == 124
    assert solver.__dict__['_factor_newton'] is original
    assert '_factor_solve' not in solver.__dict__


@pytest.mark.parametrize('case', ['duplicate_solver', 'duplicate_factor', 'duplicate_stream',
                                   'wrong_thread', 'wrong_stream', 'empty', 'count'])
def test_invalid_owner_or_shared_workspace_rejected(case):
    solvers, streams = _mock_workspaces()
    if case == 'duplicate_solver': solvers[1] = solvers[0]
    elif case == 'duplicate_factor': solvers[1].factor = solvers[0].factor
    elif case == 'duplicate_stream': streams[1] = streams[0]
    elif case == 'wrong_thread': solvers[1].factor.thread = -1
    elif case == 'wrong_stream': streams[1] = _Stream(999)
    elif case == 'empty': solvers, streams = [], []
    else: streams = streams[:1]
    with pytest.raises(ValueError):
        solve_cooperatively(solvers, streams, event_factory=_Event)


def test_original_environment_order_and_exactly_once_are_checked():
    rows = [[dict(environment_id=i) for i in group] for group in partition_indices(8, 4)]
    assert [r['environment_id'] for r in _ordered_rows(rows, 4, 8)] == list(range(8))
    rows[1][0]['environment_id'] = 0
    with pytest.raises(ValueError, match='environment'):
        _ordered_rows(rows, 4, 8)


def test_independent_direct_gate_matches_primal_scaled_backend_contract():
    assert _independent_direct_gate(dict(primal_objective=[1.], dual_objective=[1.], signed_gap=[1e-8]))
    assert not _independent_direct_gate(dict(primal_objective=[1.], dual_objective=[1e9], signed_gap=[1.]))
    assert not _independent_direct_gate(dict(primal_objective=[1.], dual_objective=[-np.inf], signed_gap=[np.inf]))


def test_nonfinite_diagnostics_remain_explicit_not_zero():
    assert _json_safe(dict(x=np.array([np.inf, np.nan, -np.inf]), accepted=np.bool_(False))) == {
        'x': ['Infinity', 'NaN', '-Infinity'], 'accepted': False}


@pytest.mark.parametrize('partitions', [1, 2, 4])
def test_cuda_cooperative_tiny_original_lp_certificates(partitions):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA device required')
    from src.gpu_batched_ipm import GpuBatchedIPM
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    problems = [(csr_matrix(np.array([[1., 1.]])), np.array([1.+.1*i]),
                 np.zeros(2), np.full(2, 2.), np.array([-1., -.2]), 0) for i in range(4)]
    groups = partition_indices(4, partitions)
    streams, solvers = [], []
    try:
        for group in groups:
            stream = cp.cuda.Stream(non_blocking=True)
            streams.append(stream)
            with stream:
                solvers.append(GpuBatchedIPM([problems[i] for i in group], regularization=1e-9))
        results, stats = solve_cooperatively(solvers, streams,
            solve_kwargs=dict(iterations=60), yield_mode='factor-and-solve')
        assert stats['owner_thread_id'] == threading.get_ident()
        for group, result, stream, solver in zip(groups, results, streams, solvers):
            assert all(result['accepted'])
            with stream:
                solver.factor._context()
                x, y = result['x'].get(), result['y'].get()
            for i, xx, yy in zip(group, x, y):
                assert paired_certificate(problems[i], xx, yy)['certificate_passed']
                np.testing.assert_allclose(xx, [1.+.1*i, 0.], atol=2e-6)
            assert '_factor_newton' not in solver.__dict__ and '_factor_solve' not in solver.__dict__
    finally:
        for solver, stream in zip(solvers, streams):
            with stream:
                solver.close()
