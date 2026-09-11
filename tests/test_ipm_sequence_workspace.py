"""GPU workspace lifecycle orchestration with CPU-only fake resources."""
from types import SimpleNamespace
import hashlib
import inspect
import json

import numpy as np
import pytest

import scripts.benchmark_ipm_sequence as module


class DeviceArray:
    def __init__(self, values):
        self.values = np.asarray(values)

    def get(self):
        return self.values.copy()


class Harness:
    def __init__(self, monkeypatch, *, reuse=True, warm=True, failure=None):
        self.events, self.solvers = [], []
        self.failure = failure
        self.args = SimpleNamespace(steps=[2, 3, 4], stage='maxmin', batch=1, iterations=9,
            warm=warm, repair_slacks=True, interior_floor=1e-6, second_forest=True,
            fix_singleton_equalities=True, reuse_gmres_workspace=True, reuse_numeric_workspace=reuse)
        self.inputs = [([f'h{step}'], dict(problem_sha256=[f'h{step}'])) for step in self.args.steps]
        self.record = dict(steps=[])
        self.cp = SimpleNamespace(ndarray=DeviceArray)
        monkeypatch.setattr(module, 'paired_certificate', self.certificate)
        monkeypatch.setattr(module, 'audit_direct_dual', self.direct)
        monkeypatch.setattr(module, '_independent_direct_gate', lambda row: row['ok'])

    def certificate(self, p, x, y):
        self.events.append(('certificate', p))
        if self.failure == 'certificate' and p == 'h3':
            raise RuntimeError('certificate failure')
        return dict(certificate_passed=True)

    def direct(self, p, x, y, xp):
        return dict(ok=not (self.failure == 'direct_reject' and p == 'h3'))

    def factory(self, problems, **options):
        if self.failure == 'constructor' and problems == ['h3']:
            raise RuntimeError('constructor failure')
        generation = len(self.solvers) + 1
        def buffer(offset):
            return SimpleNamespace(data=SimpleNamespace(ptr=generation * 100 + offset))
        factor = SimpleNamespace(analysis_count=1, factor_count=0, failed=False,
            objects={'handle': SimpleNamespace(value=generation)},
            values=buffer(1), rhs=buffer(2), solution=buffer(3))
        solver = SimpleNamespace(factor=factor, values=buffer(4), problem_hashes=tuple(problems),
                                 closed=0, options=options)
        self.solvers.append(solver)
        self.events.append(('create', problems[0]))
        def close():
            solver.closed += 1
            self.events.append(('close', solver.problem_hashes[0]))
        solver.close = close
        def solve(**kwargs):
            current = solver.problem_hashes[0]
            self.events.append(('solve', current, kwargs['internal_warm_start'] is not None))
            if self.failure == 'solve' and current == 'h3':
                raise RuntimeError('solve failure')
            factor.factor_count += 1
            return dict(total_seconds=.01, accepted=np.array([True]),
                factor_count=factor.factor_count, x=DeviceArray([[0.]]), y=DeviceArray([[0.]]))
        solver.solve = solve
        def export_internal_state(**kwargs):
            current = solver.problem_hashes[0]
            self.events.append(('export', current))
            if self.failure == 'export' and current == 'h3':
                raise RuntimeError('export failure')
            old_step = kwargs['step']
            def bind(target, **binding):
                assert binding['step'] == old_step+1
                assert binding['repair_slacks'] and binding['interior_floor'] == 1e-6
                assert binding['environment_ids'] == [0]
                assert binding['stage'] == 'maxmin'
                assert target.problem_hashes == (f'h{binding["step"]}',)
                self.events.append(('bind', target.problem_hashes[0]))
                if self.failure == 'bind_reject':
                    raise ValueError('warm topology mismatch')
                return object()
            return SimpleNamespace(bind=bind)
        solver.export_internal_state = export_internal_state
        return solver

    def rebind(self, solver, problems):
        self.events.append(('update', problems[0]))
        if problems == ['h3']:
            if self.failure == 'rebind_reject':
                raise module.NumericRebindRejected('changed CSR topology')
            if self.failure == 'runtime': raise RuntimeError('native failure')
            if self.failure == 'memory': raise MemoryError('allocation failure')
            if self.failure == 'plain_value': raise ValueError('commit failure')
        if self.failure != 'stale_hash':
            solver.problem_hashes = tuple(problems)
        if self.failure == 'changed_pointer':
            solver.factor.values.data.ptr += 1
        return dict(cpu_lp_calls=0, numerical_factor_invalidated=True,
                    old_internal_state_cleared=True, new_problem_hashes=solver.problem_hashes)

    def run(self):
        return module._run_gpu_sequence(self.record, self.args, self.inputs, cp=self.cp,
            solver_factory=self.factory, rebind_workspace=self.rebind)


def test_compatible_updates_keep_native_pointer_analysis_and_prior_step_warm_bundle(monkeypatch):
    h = Harness(monkeypatch)
    h.run()
    assert len(h.solvers) == 1 and h.solvers[0].closed == 1
    assert [event for event in h.events if event[0] == 'solve'] == [
        ('solve', 'h2', False), ('solve', 'h3', True), ('solve', 'h4', True)]
    assert h.record['gpu_workspace_creations'] == h.record['actual_symbolic_analysis_count'] == 1
    assert h.record['numeric_update_successes'] == 2
    assert not h.record['GPU_workspace_rebuilt_each_step']
    for trial in h.record['steps'][1:]:
        assert not trial['workspace_created'] and trial['constructor_seconds'] == 0.
        assert all(trial['native_workspace_identity_preserved'].values())
        assert trial['numeric_update']['cpu_lp_calls'] == 0
    assert h.events.index(('export', 'h2')) < h.events.index(('update', 'h3')) < h.events.index(('bind', 'h3'))


def test_default_fresh_policy_closes_every_step_and_counts_actual_analysis(monkeypatch):
    h = Harness(monkeypatch, reuse=False)
    h.run()
    assert len(h.solvers) == 3 and all(s.closed == 1 for s in h.solvers)
    assert h.record['GPU_workspace_rebuilt_each_step']
    assert h.record['gpu_workspace_creations'] == h.record['actual_symbolic_analysis_count'] == 3
    assert h.record['gpu_workspace_rebuilds'] == 2
    assert h.record['numeric_update_attempts'] == 0
    assert not any(e[0] == 'update' for e in h.events)
    assert h.events.index(('close', 'h2')) < h.events.index(('create', 'h3'))


def test_explicit_update_rejection_closes_old_then_rebuilds_GPU_with_reason(monkeypatch):
    h = Harness(monkeypatch, failure='rebind_reject')
    h.run()
    assert len(h.solvers) == 2 and all(s.closed == 1 for s in h.solvers)
    trial = h.record['steps'][1]
    assert trial['numeric_update_rejected'] == 'changed CSR topology'
    assert trial['workspace_rebuilt'] and trial['workspace_created']
    assert not trial['numeric_update_succeeded']
    assert h.record['numeric_update_rejections'] == 1
    assert h.record['numeric_update_successes'] == 1
    assert h.record['actual_symbolic_analysis_count'] == 2
    assert h.events.index(('close', 'h2')) < h.events.index(('create', 'h3'))
    assert h.record['cpu_lp_calls'] == 0


@pytest.mark.parametrize('failure,exception', [
    ('solve', RuntimeError), ('certificate', RuntimeError), ('export', RuntimeError),
    ('runtime', RuntimeError), ('memory', MemoryError), ('plain_value', ValueError),
    ('stale_hash', RuntimeError), ('changed_pointer', RuntimeError)])
def test_failure_closes_native_resource_and_never_silently_rebuilds(monkeypatch, failure, exception):
    h = Harness(monkeypatch, failure=failure)
    with pytest.raises(exception): h.run()
    assert len(h.solvers) == 1 and h.solvers[0].closed == 1
    assert len(h.record['steps']) == 1


def test_failed_later_constructor_cannot_leak_previous_workspace(monkeypatch):
    h = Harness(monkeypatch, reuse=False, failure='constructor')
    with pytest.raises(RuntimeError, match='constructor'): h.run()
    assert len(h.solvers) == 1 and h.solvers[0].closed == 1


def test_host_direct_rejection_stops_sequence_and_closes_without_export(monkeypatch):
    h = Harness(monkeypatch, failure='direct_reject')
    h.run()
    assert len(h.record['steps']) == 2
    assert not h.record['steps'][-1]['all_original_certificates_passed']
    assert ('export', 'h3') not in h.events
    assert h.solvers[0].closed == 1


def test_warm_binding_rejection_uses_same_current_GPU_without_CPU_fallback(monkeypatch):
    h = Harness(monkeypatch, failure='bind_reject')
    h.run()
    assert len(h.solvers) == 1
    assert all(not e[2] for e in h.events if e[0] == 'solve')
    assert h.record['steps'][1]['warm_binding_rejected'] == 'warm topology mismatch'


def test_numeric_reuse_is_independent_of_warm_start_enablement(monkeypatch):
    h = Harness(monkeypatch, warm=False)
    h.run()
    assert h.record['numeric_update_successes'] == 2
    assert not any(e[0] in ('export', 'bind') for e in h.events)
    assert not h.solvers[0].options['retain_internal_state']


def test_metadata_failure_after_factory_return_still_closes_resource(monkeypatch):
    solver = SimpleNamespace(closed=0)
    def close(): solver.closed += 1
    solver.close = close
    with module._GpuSequenceWorkspace(lambda *a, **k: solver, {}, reuse=True, rebind=None) as workspace:
        with pytest.raises(AttributeError): workspace.prepare(['h2'])
    assert solver.closed == 1


def test_close_still_releases_native_handle_when_analysis_counter_is_corrupt(monkeypatch):
    h = Harness(monkeypatch)
    workspace = module._GpuSequenceWorkspace(h.factory, {}, reuse=True, rebind=h.rebind)
    solver, _ = workspace.prepare(['h2'])
    solver.factor.analysis_count = -1
    with pytest.raises(RuntimeError, match='decreased'): workspace.close()
    assert solver.closed == 1


def test_reuse_flag_is_opt_in_and_numeric_module_is_snapshotted():
    source = inspect.getsource(module.main)
    assert "'--reuse-numeric-workspace', action=argparse.BooleanOptionalAction, default=False" in source
    assert "'src/gpu_ipm_numeric_update.py'" in source


@pytest.mark.parametrize('fail_second', [False, True])
def test_CPU_branch_closes_non_contextmanager_service_on_success_and_failure(monkeypatch, tmp_path, fail_second):
    """Exercise the actual CLI CPU branch, but never instantiate native HiGHS."""
    import src.cpu_repeated_lp as cpu_module
    trace = tmp_path / 'trace'
    trace.mkdir()
    raw = json.dumps(dict(seeds=[101], completed_steps=[3])).encode()
    (trace / 'manifest.json').write_bytes(raw)
    output = tmp_path / 'record.json'
    events = []
    # This service deliberately has NO __enter__/__exit__ methods.
    service = SimpleNamespace(history=[])
    service.close = lambda: events.append('close')
    def factory(**kwargs):
        assert kwargs == dict(workers=1, reuse_basis=True, n_fluxes=1, exchange_support_updates=True)
        events.append('create')
        return service
    monkeypatch.setattr(cpu_module, 'RepeatedCpuLP', factory)
    monkeypatch.setattr(module, 'version', lambda name: 'fake-no-native-LP')
    def inputs(directory, stage, step, batch):
        assert stage == 'maxmin' and batch == 1
        digest = f'h{step}'
        p = (None, None, None, None, np.array([0., -1.]), 0)
        return [p], dict(trace_directory=str(trace), input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            problem_sha256=[digest], entries=[dict(environment_id=0, step=step, stage=stage,
                                                  problem_sha256=digest)])
    monkeypatch.setattr(module, 'load_inputs', inputs)
    monkeypatch.setattr(module, 'problem_request', lambda *a, **k: (np.array([0., -1.]), {}))
    def timed(current, problems, requests, stage, kind, step, **kwargs):
        assert current is service and kwargs['require_direct_dual']
        events.append(f'step{step}')
        if fail_second and step == 3:
            raise RuntimeError('mock CPU solve failure')
        current.history.append(step)
        return dict(solve_batch_wall_seconds=.01, all_original_certificates_passed=True)
    monkeypatch.setattr(module, '_timed_batch', timed)
    monkeypatch.setattr(module, 'history_counts', lambda history: dict(actual_cpu_optimizer_runs=len(history)))
    # Isolate source snapshots from repository I/O while preserving real JSON
    # output and the actual closing(...) branch under test.
    fake_file = SimpleNamespace(read_text=lambda: 'mock source', read_bytes=lambda: b'mock source')
    class FakeRoot:
        def __truediv__(self, path): return fake_file
    monkeypatch.setattr(module, 'ROOT', FakeRoot())
    monkeypatch.setattr(module.sys, 'argv', ['benchmark_ipm_sequence.py', '--mode', 'cpu',
        '--trace', str(trace), '--output', str(output), '--batch', '1', '--workers', '1', '--steps', '2', '3'])
    if fail_second:
        with pytest.raises(RuntimeError, match='mock CPU solve'): module.main()
        assert not output.exists()
    else:
        module.main()
        record = json.loads(output.read_text())
        assert record['completed'] and record['all_requested_steps_qualified']
        assert record['actual_cpu_optimizer_runs'] == 2
        assert record['gpu_calls'] == 0
    assert events == ['create', 'step2', 'step3', 'close']
