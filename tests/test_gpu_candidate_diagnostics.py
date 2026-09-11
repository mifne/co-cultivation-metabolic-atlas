"""Host-only diagnostic capture: no CUDA kernels or optimizer calls."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import src.gpu_hybrid_lp as module
from tests.test_gpu_hybrid_cpu_dispatch import host_service, device


def _slots():
    return dict(indices=device([[0, 1]]*4),
        input_family_valid=device([[True, True], [True, True], [True, True], [False, True]]),
        accepted=device([[True, False], [False, False], [False, False], [False, False]]),
        primal_residual=device([[0., 1.], [2e-5, 0.], [np.inf, 0.], [np.nan, 0.]]),
        dual_violation=device([[0., 0.], [0., 2e-7], [0., 0.], [0., 0.]]),
        relative_kkt_gap=device([[0., 0.], [0., 0.], [0., 2e-7], [0., 0.]]),
        basis_dual_violation=device([[0., 2e-8]]*4),
        objective=device([[1., 2.]]*4), primal_violation_count=device([[0, 1]]*4),
        primal_violation_l1=device([[0., 1.]]*4))


def _attach_engine(service, result, slots):
    key = next(iter(service.banks))
    service.banks[key].evaluators = [None, None]
    engine = SimpleNamespace(last_candidates=slots,
        evaluate=lambda inputs, order:result, close=lambda:None)
    service.heterogeneous_banks[key] = engine
    return engine


def test_snapshot_preserves_pre_cpu_values_and_does_not_change_routes(host_service):
    service, requests, calls, failed, templates, bank_result = host_service
    slots = _slots()
    _attach_engine(service, bank_result, slots)
    service.candidate_diagnostics = True
    results = service.solve_batch(requests)
    record = service.history[-1]
    snapshot = record['candidate_diagnostics']
    assert snapshot['indices'] == [[0, 1]]*4
    assert snapshot['selection']['accepted'] == [True, False, False, False]
    assert snapshot['selection']['candidate_index'] == [0, -1, -1, -1]
    assert snapshot['selection']['best_candidate_index'] == [0, 1, 0, 1]
    assert snapshot['accepted'] == [[True, False], [False, False], [False, False], [False, False]]
    assert snapshot['primal_residual'][1][0] == 2e-5
    # CPU successes subsequently replace the stage-level metrics, never the
    # captured GPU candidates or their true original certificate flags.
    assert all(record['accepted']) and all(row.success for row in results)
    assert record['primal_residual'] == [0.]*4
    assert calls[0][0] == [1, 2, 3]
    assert record['routes'] == ['gpu_dictionary', 'cpu_gpu_budget_exhausted',
        'cpu_high_dispersion', 'cpu_unknown_dispersion']
    assert snapshot['stage_call'] == 1
    assert record['candidate_diagnostic_seconds'] >= 0.
    slots['primal_residual'][1, 0] = 77.
    slots['accepted'][:] = True
    assert snapshot['primal_residual'][1][0] == 2e-5
    assert not snapshot['accepted'][1][0]


def test_default_path_never_calls_candidate_snapshot_or_downloads_slot_fields(host_service, monkeypatch):
    service, requests, calls, failed, templates, bank_result = host_service
    class ForbiddenDownload:
        def get(self):
            raise AssertionError('Diagnostic download in default path')
    _attach_engine(service, bank_result, dict(primal_residual=ForbiddenDownload()))
    service.candidate_diagnostics = False
    def forbidden(*args, **kwargs):
        raise AssertionError('Diagnostic helper in default path')
    monkeypatch.setattr(module, 'candidate_diagnostic_snapshot', forbidden)
    assert all(row.success for row in service.solve_batch(requests))
    assert 'candidate_diagnostics' not in service.history[-1]
    assert 'candidate_diagnostic_seconds' not in service.history[-1]


def test_nonfinite_values_are_null_and_scalar_gates_fail_closed():
    slots = _slots()
    slots['indices'][3] = [-1, 99]
    snapshot = module.candidate_diagnostic_snapshot(slots, {}, [[0, 1]]*4, 2)
    assert snapshot['primal_residual'][2][0] is None
    assert snapshot['primal_residual'][3][0] is None
    assert snapshot['metric_finite']['primal_residual'][2][0] is False
    assert snapshot['certificate_metrics_finite'][3][0] is False
    assert snapshot['scalar_gate_pass']['primal_residual'][1][0] is False
    assert snapshot['scalar_gate_pass']['dual_violation'][1][1] is False
    assert snapshot['scalar_gate_pass']['relative_kkt_gap'][2][1] is False
    assert snapshot['scalar_gate_pass']['primal_residual'][2][0] is False
    assert snapshot['index_in_range'][3] == [False, False]
    assert snapshot['basis_dual_feasible_for_ranking'][0] == [True, False]
    assert snapshot['scalar_gate_thresholds'] == dict(
        primal_residual=1e-5, dual_violation=1e-7, relative_kkt_gap=1e-7)
    assert snapshot['raw_values_finite'] is None and snapshot['raw_duals_finite'] is None
    json.dumps(snapshot, allow_nan=False)


@pytest.mark.parametrize('source', [None, [], {}])
def test_missing_candidate_metadata_is_explicitly_unknown_not_a_passing_zero(source):
    snapshot = module.candidate_diagnostic_snapshot(source, {}, [[0, 1]], 2)
    assert not snapshot['available']
    assert snapshot['indices'] is None and snapshot['accepted'] is None
    assert snapshot['input_family_valid'] is None
    assert snapshot['primal_residual'] is None
    assert snapshot['certificate_metrics_finite'] is None
    assert all(value is None for value in snapshot['scalar_gate_pass'].values())
    assert all(value is None for value in snapshot['selection'].values())
    assert snapshot['unavailable_fields']['primal_residual'] == 'not_recorded'
    json.dumps(snapshot, allow_nan=False)


def test_mismatched_shapes_and_non_boolean_flags_are_not_inferred():
    snapshot = module.candidate_diagnostic_snapshot(dict(indices=device([[0, 1]]),
        accepted=device([[np.nan, 1.]]), primal_residual=device([0., 0.])), {}, [[0, 1]], 2)
    assert snapshot['accepted'] is None and snapshot['primal_residual'] is None
    assert snapshot['scalar_gate_pass']['primal_residual'] is None
    assert snapshot['unavailable_fields']['accepted'] == 'unexpected_shape_or_dtype'
    with pytest.raises(ValueError, match='B×K'):
        module.candidate_diagnostic_snapshot({}, {}, [0, 1], 2)


def test_cli_rejects_candidate_diagnostics_without_heterogeneous_hybrid_before_gpu(monkeypatch, tmp_path):
    from scripts import benchmark_compact_gpu as benchmark
    monkeypatch.setattr(sys, 'argv', ['benchmark_compact_gpu.py', '--bank', str(tmp_path/'unused'),
        '--output', str(tmp_path/'unused.json'), '--candidate-diagnostics'])
    with pytest.raises(ValueError, match='heterogeneous hybrid'):
        benchmark.main()
    assert not (tmp_path/'unused.json').exists()


@pytest.mark.parametrize('oracle_fails', [False, True])
def test_oracle_restores_metadata_and_never_changes_cpu_routes_or_initial_basis(host_service, oracle_fails):
    service, requests, calls, failed, templates, bank_result = host_service
    slots = _slots()
    engine = _attach_engine(service, bank_result, slots)
    bank = next(iter(service.banks.values()))
    bank.evaluators = [None]*3
    bank.centers = np.zeros((3, 1))
    service.candidate_limit = 2
    service.candidate_diagnostics = service.candidate_oracle = True
    original_scores, original_dual = object(), object()
    engine.last_candidate_scores, engine.last_candidate_dual = original_scores, original_dual
    evaluated = []

    def evaluate(inputs, order):
        evaluated.append(np.array(order))
        if order.shape[1] == 2:
            return bank_result
        engine.last_candidate_scores = 'oracle scores'
        engine.last_candidate_dual = 'oracle duals'
        oracle_slots = {name:device(np.pad(value.get(), ((0, 0), (0, 1)))) for name, value in slots.items()}
        oracle_slots['indices'] = device([[0, 1, 2]]*4)
        oracle_slots['accepted'] = device([[True, False, False], [False, False, True],
                                          [True, False, False], [False, False, False]])
        engine.last_candidates = oracle_slots
        if oracle_fails:
            raise RuntimeError('deliberate oracle failure after metadata write')
        return dict(bank_result, accepted=device([True, True, True, False]),
            candidate_index=device([0, 2, 0, -1]), best_candidate_index=device([2]*4),
            values=device([[333.]]*4))

    engine.evaluate = evaluate
    results = service.solve_batch(requests)
    assert [value.shape for value in evaluated] == [(4, 2), (4, 3)]
    assert evaluated[1].tolist() == [[0, 1, 2]]*4
    assert engine.last_candidates is slots
    assert engine.last_candidate_scores is original_scores
    assert engine.last_candidate_dual is original_dual
    # The oracle would accept rows 1/2, but the real K2 trajectory still
    # dispatches all original misses with the original CPU basis proposals.
    assert calls[0][0] == [1, 2, 3]
    for request, basis in zip(calls[0][1], (templates[1], templates[0], templates[1])):
        assert request[1]['_initial_basis'] is basis
    np.testing.assert_array_equal([row.x[0] for row in results], [99., 11., 12., 13.])
    record = service.history[-1]
    assert record['candidate_diagnostics']['selection']['accepted'] == [True, False, False, False]
    assert record['candidate_evaluations'] == 8
    oracle = record['candidate_oracle']
    assert oracle['normal_k_accepted'] == [True, False, False, False]
    assert oracle['seconds'] >= 0. and oracle['stage_call'] == 1
    if oracle_fails:
        assert oracle['status'] == 'failed' and oracle['snapshot'] is None
        assert oracle['additional_accepted_environment_ids'] is None
        assert oracle['outside_k_rescued_count'] is None
    else:
        assert oracle['status'] == 'completed'
        assert oracle['additional_accepted_environment_ids'] == [1, 2]
        # Row 2 differs inside K; do not misreport it as K-outside coverage.
        assert oracle['outside_k_rescued_environment_ids'] == [1]
        assert oracle['additional_accepted_count'] == 2 and oracle['outside_k_rescued_count'] == 1
        assert oracle['snapshot']['selection']['best_candidate_index'] == [2]*4


def test_oracle_off_never_runs_counterfactual_even_with_diagnostics(host_service, monkeypatch):
    service, requests, calls, failed, templates, bank_result = host_service
    _attach_engine(service, bank_result, _slots())
    service.candidate_diagnostics = True
    service.candidate_oracle = False
    def forbidden(*args, **kwargs):
        raise AssertionError('Oracle invoked when disabled')
    monkeypatch.setattr(module, 'candidate_oracle_snapshot', forbidden)
    assert all(row.success for row in service.solve_batch(requests))
    assert 'candidate_oracle' not in service.history[-1]


def test_oracle_restores_absent_attributes_and_missing_metrics_stay_unknown():
    engine = SimpleNamespace()
    def evaluate(inputs, order):
        engine.last_candidates = None
        engine.last_candidate_scores = 'temporary scores'
        engine.last_candidate_dual = 'temporary duals'
        return {}
    engine.evaluate = evaluate
    result = module.candidate_oracle_snapshot(engine, {}, [[0]], 2, [False])
    assert result['additional_accepted_count'] is None
    assert result['outside_k_rescued_count'] is None
    assert not hasattr(engine, 'last_candidates')
    assert not hasattr(engine, 'last_candidate_scores')
    assert not hasattr(engine, 'last_candidate_dual')
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('flags', [[], ['--hybrid', '--heterogeneous-candidates',
    '--tie-policy', 'original3', '--repair-operators', 'unused']])
def test_cli_oracle_requires_diagnostics_before_gpu(monkeypatch, tmp_path, flags):
    from scripts import benchmark_compact_gpu as benchmark
    monkeypatch.setattr(sys, 'argv', ['benchmark_compact_gpu.py', '--bank', str(tmp_path/'unused'),
        '--output', str(tmp_path/'unused.json'), '--candidate-oracle', *flags])
    with pytest.raises(ValueError, match='Candidate oracle requires'):
        benchmark.main()


def test_oracle_report_cannot_contain_a_speed_ratio_even_if_accuracy_passes():
    from scripts.benchmark_compact_gpu import record_performance_comparison
    ordinary = dict(cpu_seconds=2., gpu_seconds=1., all_endpoint_gates_passed=True)
    record_performance_comparison(ordinary, True)
    assert ordinary['cpu_over_gpu_ratio'] == 2.
    record_performance_comparison(ordinary, True, diagnostic_only=True)
    assert 'cpu_over_gpu_ratio' not in ordinary
    assert ordinary['performance_comparison_valid'] is False
    assert 'oracle' in ordinary['performance_comparison_note']
    incomplete = dict(cpu_seconds=2., gpu_seconds=1., all_endpoint_gates_passed=False)
    record_performance_comparison(incomplete, False)
    assert 'cpu_over_gpu_ratio' not in incomplete
