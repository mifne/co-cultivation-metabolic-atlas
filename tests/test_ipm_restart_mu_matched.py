"""Matched restart orchestration contracts; no CUDA or native LP execution."""
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest

import scripts.probe_ipm_restart_mu_matched as module


class DeviceArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def copy(self):
        return DeviceArray(self.values.copy())

    def get(self):
        return self.values.copy()


class Harness:
    def __init__(self, monkeypatch, failure=None):
        self.events, self.solvers, self.states = [], [], []
        self.failure, self.syncs = failure, 0
        self.args = SimpleNamespace(batch=4, stage='maxmin', iterations=12,
            mus=[1e-3, 1e-4, 1e-5, 1e-6], device_staging=True, direct_kkt_payload=False)
        self.inputs = []
        for step in (2, 3, 4):
            problem = (np.zeros((1, 1)), np.array([step]), np.zeros(1),
                       np.ones(1), np.ones(1), 0)
            self.inputs.append(([problem] * 4, dict(problem_sha256=[f'h{step}'] * 4)))
        self.cp = SimpleNamespace(ndarray=DeviceArray,
            array_equal=lambda a, b: np.array_equal(a.values, b.values))
        self.record = {}
        monkeypatch.setattr(module, 'paired_certificate', self.certificate)
        monkeypatch.setattr(module, 'audit_direct_dual', self.direct)
        monkeypatch.setattr(module, '_independent_direct_gate', lambda row: row['pass'])

    def certificate(self, problem, x, y):
        step = int(problem[1][0])
        if self.failure == f'verify{step}':
            raise RuntimeError('independent verification failure')
        return dict(certificate_passed=self.failure != f'original{step}')

    def direct(self, problem, x, y, xp):
        return {'pass': self.failure != f'direct{int(problem[1][0])}'}

    def synchronize(self):
        self.syncs += 1

    def factory(self, problems, **options):
        step = int(problems[0][1][0])
        if self.failure == f'constructor{step}':
            raise RuntimeError('constructor failure before native acquisition')
        generation = len(self.solvers) + 1
        def buffer(offset):
            return SimpleNamespace(data=SimpleNamespace(ptr=100 * generation + offset))
        factor = SimpleNamespace(analysis_count=1, factor_count=0,
            objects={'handle': SimpleNamespace(value=generation)},
            values=buffer(1), rhs=buffer(2), solution=buffer(3))
        solver = SimpleNamespace(step=step, closed=0, factor=factor, values=buffer(4),
            problem_hashes=tuple(f'h{step}' for p in problems), options=options,
            previous=None, generation=generation)
        self.solvers.append(solver)
        self.events.append(('create', generation, step))
        def close():
            solver.closed += 1
            self.events.append(('close', generation, solver.step))
        solver.close = close
        def solve(*, iterations, internal_warm_start):
            assert iterations == self.args.iterations
            self.events.append(('solve', generation, solver.step, internal_warm_start))
            if self.failure == f'solve{solver.step}':
                raise RuntimeError('native solve failure')
            if self.failure == 'mutate_solve' and solver.step == 3:
                self.states[0]._arrays[0].values[0, 0] += 1.
            factor.factor_count += 1
            return dict(total_seconds=.01, factor_count=1, solve_count=2,
                accepted=np.full(4, self.failure != f'accept{solver.step}'),
                x=DeviceArray(np.full((3 if self.failure == 'shape' else 4, 1), solver.step)),
                y=DeviceArray(np.zeros((4, 1))), cpu_lp_calls=0)
        solver.solve = solve
        def export_internal_state(*, environment_ids, stage, step):
            assert environment_ids == [0, 1, 2, 3] and stage == 'maxmin'
            assert step == solver.step
            if self.failure == f'export{step}':
                raise RuntimeError('export failure')
            state = SimpleNamespace(_arrays=tuple(DeviceArray(np.full((4, 1), step + i))
                for i in range(4)), signature='same-coordinates', environment_ids=tuple(environment_ids),
                step=step, stage=stage, source_problem_hashes=solver.problem_hashes,
                source_certificate_basis=('analytic_box_bound',) * 4)
            self.states.append(state)
            self.events.append(('export', generation, step, state))
            def bind(target, **kwargs):
                assert kwargs['step'] == step + 1 == target.step
                assert kwargs['environment_ids'] == environment_ids
                assert kwargs['stage'] == stage
                assert kwargs['interior_floor'] == 0. and not kwargs['repair_slacks']
                assert kwargs['restart_mu'] in self.args.mus
                self.events.append(('bind', target.generation, target.step, state, kwargs['restart_mu']))
                if self.failure == 'bind':
                    raise ValueError('warm map rejected')
                if self.failure == 'mutate_bind':
                    state._arrays[0].values[0, 0] += 1.
                return SimpleNamespace(metadata=dict(source_step=step, target_step=target.step,
                    restart_mu=kwargs['restart_mu'], source_problem_sha256=state.source_problem_hashes))
            state.bind = bind
            return state
        solver.export_internal_state = export_internal_state
        return solver

    def rebind(self, solver, problems, **options):
        assert solver.step == 3
        assert options['reuse_static_forest'] and options['device_staging']
        assert options.get('direct_kkt_payload', False) == self.args.direct_kkt_payload
        self.events.append(('update', solver.generation, options))
        if self.failure == 'update_runtime':
            raise RuntimeError('native update failure')
        if self.failure == 'update_plain_value':
            raise ValueError('unexpected commit failure')
        if self.failure == 'update_reject':
            from src.gpu_ipm_numeric_update import NumericRebindRejected
            raise NumericRebindRejected('different topology')
        solver.step = 4
        if self.failure != 'stale_hash':
            solver.problem_hashes = ('h4',) * 4
        if self.failure == 'pointer_change':
            solver.factor.values.data.ptr += 1
        return dict(cpu_lp_calls=0, old_internal_state_cleared=True)

    def run(self):
        module._run_matched(self.record, self.args, self.inputs, cp=self.cp,
            solver_factory=self.factory, rebind_workspace=self.rebind, synchronize=self.synchronize)
        return self.record


def test_every_mu_gets_identical_source_and_own_qualified_step3_state(monkeypatch):
    h = Harness(monkeypatch)
    record = h.run()
    assert record['completed'] and record['all_requested_arms_qualified']
    assert record['source_solve_calls'] == 1 and record['cpu_lp_calls'] == 0
    assert len(h.solvers) == 5 and all(s.closed == 1 for s in h.solvers)
    assert record['matched_source_preservation_verified']
    step3_binds = [event for event in h.events if event[0] == 'bind' and event[2] == 3]
    step4_binds = [event for event in h.events if event[0] == 'bind' and event[2] == 4]
    assert len(step3_binds) == len(step4_binds) == 4
    assert all(event[3] is h.states[0] for event in step3_binds)
    assert len({id(event[3]) for event in step4_binds}) == 4
    assert all(event[3] is not h.states[0] for event in step4_binds)
    assert [event[4] for event in step3_binds] == h.args.mus
    assert all(s.options['retain_internal_state'] for s in h.solvers)
    assert all(s.options['regularization'] == 1e-6 for s in h.solvers)
    for arm in record['arms']:
        summary = arm['workspace_summary']
        assert summary['workspace_creations'] == summary['actual_symbolic_analysis_count'] == 1
        assert summary['workspace_close_calls'] == summary['numeric_update_successes'] == 1
        assert arm['steps'][0]['workspace_created'] and not arm['steps'][1]['workspace_created']
        assert all(arm['steps'][1]['native_workspace_identity_preserved'].values())
        assert arm['steps'][0]['previous_bundle_object_id'] == id(h.states[0])
        assert arm['steps'][1]['warm_binding']['source_step'] == 3
        assert arm['successful_sequence_solve_seconds'] >= 0.
        for step in arm['steps']:
            assert step['source_after_solve']['x_unchanged']
            assert step['source_after_export']['vector_D2H_copies'] == 0
            assert all(step[name] >= 0. for name in ('binding_seconds', 'solve_wall_seconds',
                'D2H_and_independent_verification_seconds', 'source_guard_seconds',
                'workspace_prepare_wall_seconds', 'numeric_update_wall_seconds', 'constructor_seconds'))
    assert h.syncs > 20


@pytest.mark.parametrize('failure', ['original2', 'direct2', 'accept2'])
def test_source_must_pass_all_independent_gates_before_any_arm(monkeypatch, failure):
    h = Harness(monkeypatch, failure)
    record = h.run()
    assert not record['completed'] and not record['arms']
    assert record['source']['status'] == 'unqualified'
    assert record['source']['successful_solve_seconds'] is None
    assert not h.states and h.solvers[0].closed == 1


@pytest.mark.parametrize('failure', ['original3', 'direct3', 'accept3'])
def test_unqualified_target_is_not_exported_or_counted_as_speed_success(monkeypatch, failure):
    h = Harness(monkeypatch, failure)
    record = h.run()
    assert record['completed'] and not record['all_requested_arms_qualified']
    assert len(record['arms']) == 4 and len(h.states) == 1
    assert not any(e[0] == 'update' for e in h.events)
    for arm in record['arms']:
        assert arm['status'] == 'unqualified' and len(arm['steps']) == 1
        assert arm['successful_sequence_solve_seconds'] is None
        assert arm['steps'][0]['successful_solve_seconds'] is None
        assert arm['steps'][0]['solver_result']['factor_count'] == 1
    assert all(s.closed == 1 for s in h.solvers)


@pytest.mark.parametrize('failure', ['constructor2', 'constructor3', 'solve2', 'solve3', 'solve4',
    'verify2', 'verify3', 'export2', 'export3', 'bind', 'update_runtime',
    'update_plain_value', 'stale_hash', 'pointer_change', 'shape', 'mutate_bind', 'mutate_solve'])
def test_exceptions_close_all_handles_abort_remaining_and_never_cold_fallback(monkeypatch, failure):
    h = Harness(monkeypatch, failure)
    record = h.run()
    assert 'error' in record and not record['completed']
    assert not record['all_requested_arms_qualified']
    assert not record['matched_source_preservation_verified']
    assert all(s.closed == 1 for s in h.solvers)
    assert len(record['arms']) <= 1 and len(h.solvers) <= 2
    if record['arms']:
        assert record['arms'][-1]['status'] == 'error'
        assert record['arms'][-1]['successful_sequence_solve_seconds'] is None
    if failure in ('bind', 'mutate_bind'):
        assert not any(event[0] == 'solve' and event[2] == 3 for event in h.events)
    if failure == 'constructor2':
        assert record['source_solve_calls'] == 0


def test_numeric_rejection_has_explicit_GPU_rebuild_but_no_cpu_solve(monkeypatch):
    h = Harness(monkeypatch, 'update_reject')
    record = h.run()
    assert record['completed'] and record['all_requested_arms_qualified']
    assert len(h.solvers) == 9 and all(s.closed == 1 for s in h.solvers)
    for arm in record['arms']:
        step = arm['steps'][1]
        assert step['numeric_update_rejected'] == 'different topology'
        assert step['workspace_created'] and step['workspace_rebuilt']
        assert arm['workspace_summary']['workspace_creations'] == 2
    assert record['cpu_lp_calls'] == 0


def test_direct_payload_opt_in_is_same_for_every_arm(monkeypatch):
    h = Harness(monkeypatch)
    h.args.direct_kkt_payload = True
    assert h.run()['all_requested_arms_qualified']
    updates = [e for e in h.events if e[0] == 'update']
    assert len(updates) == 4 and all(e[2]['direct_kkt_payload'] for e in updates)


@pytest.mark.parametrize('mus,iterations', [([], 10), ([1e-4, 1e-4], 10),
    ([float('nan')], 10), ([float('inf')], 10), ([0.], 10), ([1e-9], 10),
    ([.1], 10), ([1e-4], 0), ([1e-4], 1001), ([1e-4], True)])
def test_invalid_conditions_are_rejected_before_gpu_import(mus, iterations):
    with pytest.raises(ValueError):
        module.validate_args(SimpleNamespace(mus=mus, iterations=iterations))


def test_parser_limits_scope_and_preserves_mu_order():
    args = module.make_parser().parse_args(['--output', 'unused.json'])
    assert args.batch == 4 and args.mus == [1e-3, 1e-4, 1e-5, 1e-6]
    assert args.device_staging and not args.direct_kkt_payload
    module.validate_args(args)
    with pytest.raises(SystemExit):
        module.make_parser().parse_args(['--output', 'unused.json', '--batch', '8'])


def test_exclusive_json_preserves_failures_nonfinite_diagnostics_and_existing_file(tmp_path):
    output = tmp_path/'record.json'
    record = dict(completed=False, error=dict(type='RuntimeError', message='failed'),
                  diagnostic=float('inf'), successful_solve_seconds=None)
    module._write_exclusive(output, record)
    actual = json.loads(output.read_text())
    assert not actual['completed'] and actual['diagnostic'] == 'Infinity'
    assert actual['successful_solve_seconds'] is None
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        module._write_exclusive(output, dict(completed=True))
    assert output.read_bytes() == before


def test_manifest_contract_rejects_mixed_or_reordered_target_inputs():
    raw = json.dumps(dict(seeds=[101, 102, 103, 104])).encode()
    provenances = []
    for step in (2, 3, 4):
        provenances.append(dict(trace_directory='/same', input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            problem_sha256=[f'{step}-{env}' for env in range(4)],
            entries=[dict(step=step, stage='maxmin', environment_id=env,
                          problem_sha256=f'{step}-{env}') for env in range(4)]))
    assert module.validate_sequence_provenance(provenances, [2, 3, 4], 'maxmin', 4, raw)
    provenances[-1]['entries'].reverse()
    with pytest.raises(ValueError, match='Environment'):
        module.validate_sequence_provenance(provenances, [2, 3, 4], 'maxmin', 4, raw)


def test_new_numeric_and_assembly_sources_are_snapshotted():
    assert {'src/csr_block_assembly.py', 'src/gpu_ipm_kkt_payload.py',
            'src/gpu_ipm_staging.py', 'src/gpu_ipm_reoptimization.py'} <= set(module.SOURCE_PATHS)


@pytest.mark.parametrize('failure', [None, 'solve3', 'direct3'])
def test_actual_cli_records_success_and_failure_exclusively_without_native_backend(
        monkeypatch, tmp_path, failure):
    h = Harness(monkeypatch, failure)
    trace = tmp_path/'trace'
    trace.mkdir()
    raw = json.dumps(dict(seeds=[101, 102, 103, 104])).encode()
    (trace/'manifest.json').write_bytes(raw)
    def load_inputs(directory, stage, step, batch):
        assert directory == trace and stage == 'maxmin' and batch == 4
        problems, provenance = h.inputs[step - 2]
        return problems, dict(provenance, trace_directory=str(trace),
            input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            entries=[dict(environment_id=env, stage=stage, step=step,
                          problem_sha256=f'h{step}') for env in range(4)])
    monkeypatch.setattr(module, 'load_inputs', load_inputs)
    h.cp.cuda = SimpleNamespace(runtime=SimpleNamespace(getDeviceProperties=lambda device: {'name': b'fake'}),
        get_current_stream=lambda: SimpleNamespace(synchronize=h.synchronize))
    monkeypatch.setitem(sys.modules, 'cupy', h.cp)
    monkeypatch.setitem(sys.modules, 'src.gpu_forest_ipm', SimpleNamespace(ForestGpuBatchedIPM=h.factory))
    monkeypatch.setitem(sys.modules, 'src.gpu_ipm_numeric_update', SimpleNamespace(rebind_forest_ipm=h.rebind))
    monkeypatch.setattr(module, 'SOURCE_PATHS', ('scripts/probe_ipm_restart_mu_matched.py',))
    output = tmp_path/'record.json'
    monkeypatch.setattr(module.sys, 'argv', ['probe_ipm_restart_mu_matched.py', '--trace', str(trace),
        '--output', str(output), '--batch', '4', '--iterations', '12'])
    assert module.main() == (0 if failure is None else 1)
    record = json.loads(output.read_text())
    assert record['current_reference_vectors_loaded'] is False
    assert record['current_CPU_solutions_passed_to_GPU'] is False
    assert record['CPU_optimizer_fallback_allowed'] is False
    assert record['original_certificate_limits'] == dict(
        primal_residual=1e-5, dual_violation=1e-7, relative_kkt_gap=1e-7)
    assert record['additional_direct_dual_relative_gap_limit'] == 1e-7
    assert record['all_requested_arms_qualified'] == (failure is None)
    assert record['completed'] == (failure != 'solve3')
    assert all(s.closed == 1 for s in h.solvers)
    path = 'scripts/probe_ipm_restart_mu_matched.py'
    assert record['source_sha256'][path] == hashlib.sha256((module.ROOT/path).read_bytes()).hexdigest()
    if failure == 'solve3':
        assert record['error']['type'] == 'RuntimeError'
        assert record['arms'][0]['successful_sequence_solve_seconds'] is None
    previous = output.read_bytes()
    count = len(h.solvers)
    with pytest.raises(FileExistsError):
        module.main()
    assert output.read_bytes() == previous and len(h.solvers) == count
