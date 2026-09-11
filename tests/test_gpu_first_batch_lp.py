"""GPU-first original-stage tests, including explicit no-CPU failure policy."""
import numpy as np
import pytest

from tests.test_gpu_hybrid_backend import tiny_hybrid_factory, _request, _frozen_bank, _assert_bank_unchanged


def test_all_certified_batch_never_invokes_cpu_and_reuses_device_graph(tiny_hybrid_factory, monkeypatch):
    service, bank, p, _ = tiny_hybrid_factory(gpu_first_batch=True,
        heterogeneous_candidates=True, heterogeneous_replay=True, repair_rounds=0,
        candidate_limit=2, temporal_candidate_policy='within-budget')
    def forbidden(*args, **kwargs):
        raise AssertionError('A certified GPU batch must not execute a CPU LP')
    monkeypatch.setattr(service.direct_cpu, 'solve_batch', forbidden)
    frozen = _frozen_bank(bank)
    for ids in ([13, 7], [7, 13]):
        rows = service.solve_batch([_request(p), _request(p)], environment_ids=ids)
        assert all(row.success and row.message == 'gpu_dictionary_batch' for row in rows)
        assert [row.diagnostics['environment_id'] for row in rows] == ids
        record = service.history[-1]
        assert record['bank_accepts'] == 2 and record['cpu_lp_calls'] == 0
        assert record['cpu_solver_runs'] == 0
        assert sum(record['gpu_first_phases'].values()) == pytest.approx(record['seconds'])
        assert record['certificate_only'] and record['device_routing'] and record['packed_result_transfer']
    assert len(service.heterogeneous_banks[next(iter(service.banks))].graph_cache) == 1
    _assert_bank_unchanged(bank, frozen)
    assert service._gpu_first_previous
    service.reset_trajectory()
    assert not service._gpu_first_previous


def test_only_uncertified_rows_go_to_same_dictionary_cpu(tiny_hybrid_factory, monkeypatch):
    service, _, p, q = tiny_hybrid_factory(gpu_first_batch=True,
        heterogeneous_candidates=True, repair_rounds=0, candidate_limit=2)
    original = service.direct_cpu.solve_batch
    calls = []
    def exact(requests, *, environment_ids):
        calls.append((requests, environment_ids))
        return original(requests, environment_ids=environment_ids)
    monkeypatch.setattr(service.direct_cpu, 'solve_batch', exact)
    rows = service.solve_batch([_request(p), _request(q)], environment_ids=[19, 53])
    assert all(row.success for row in rows)
    assert len(calls) == 1 and len(calls[0][0]) == 1 and calls[0][1] == [53]
    record = service.history[-1]
    assert record['bank_accepts'] == 1 and record['cpu_lp_calls'] == 1
    assert record['cpu_speculative_unused'] == 0
    assert record['groups'][0]['ids'] == [1]
    assert record['groups'][0]['environment_ids'] == [53]
    assert record['routes'] == ['gpu_dictionary_batch', 'cpu_after_gpu_certificate']


def test_reject_policy_returns_failure_without_cpu_or_fabricated_solution(tiny_hybrid_factory, monkeypatch):
    service, _, p, q = tiny_hybrid_factory(gpu_first_batch=True, gpu_cpu_fallback='reject',
        heterogeneous_candidates=True, repair_rounds=0, candidate_limit=2)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reject policy must prohibit CPU optimization')
    monkeypatch.setattr(service.direct_cpu, 'solve_batch', forbidden)
    rows = service.solve_batch([_request(p), _request(q)])
    assert rows[0].success and not rows[1].success
    assert rows[1].x is None and rows[1].fun is None
    assert rows[1].message == 'gpu_uncertified_no_cpu'
    assert service.history[-1]['cpu_lp_calls'] == 0
    assert list(service._gpu_first_previous.values())[1] == -1


@pytest.mark.parametrize('field,value', [('values', np.nan), ('objective', np.inf),
    ('primal_residual', -1.), ('primal_residual', 1e-4),
    ('dual_violation', 1e-6), ('relative_kkt_gap', 1e-6), ('candidate_index', -1)])
def test_contradictory_gpu_acceptance_fails_closed(tiny_hybrid_factory, monkeypatch, field, value):
    service, _, p, _ = tiny_hybrid_factory(gpu_first_batch=True, gpu_cpu_fallback='reject',
        heterogeneous_candidates=True, repair_rounds=0, candidate_limit=2)
    engine = next(iter(service.heterogeneous_banks.values()))
    original = engine.evaluate
    def malformed(inputs, order):
        result = original(inputs, order)
        result[field][0] = value
        return result
    monkeypatch.setattr(engine, 'evaluate', malformed)
    row = service.solve_batch([_request(p)])[0]
    assert not row.success and row.x is None
    assert service.history[-1]['bank_accepts'] == 0


@pytest.mark.parametrize('stage', ['maxmin', 'aggregate', 'exchange'])
def test_each_original_stage_reaches_same_gpu_batch_engine(tiny_hybrid_factory, monkeypatch, stage):
    import src.gpu_hybrid_lp as module
    service, bank, p, _ = tiny_hybrid_factory(gpu_first_batch=True, gpu_cpu_fallback='reject',
        heterogeneous_candidates=True, repair_rounds=0, candidate_limit=2)
    old_key = next(iter(service.banks)); key = (stage, *old_key[1:])
    engine = service.heterogeneous_banks.pop(old_key)
    service.heterogeneous_banks[key] = engine
    service.banks = {key: bank}
    monkeypatch.setattr(module, 'stage_key', lambda *args: key)
    result = service.solve_batch([_request(p)])
    assert result[0].success and service.history[-1]['stage'] == stage
    assert service.history[-1]['bank_accepts'] == 1


def test_invalid_device_routing_row_cannot_be_accepted(tiny_hybrid_factory, monkeypatch):
    service, _, p, _ = tiny_hybrid_factory(gpu_first_batch=True, gpu_cpu_fallback='reject',
        heterogeneous_candidates=True, repair_rounds=0, candidate_limit=2)
    import src.gpu_first_batch_lp as module
    original = module.device_candidate_order
    def invalid(cp, bank, inputs, router):
        order, valid, label = original(cp, bank, inputs, router)
        valid[:] = False
        return order, valid, label
    monkeypatch.setattr(module, 'device_candidate_order', invalid)
    assert not service.solve_batch([_request(p)])[0].success


def test_gpu_error_drains_stream_before_returning(tiny_hybrid_factory, monkeypatch):
    service, _, p, _ = tiny_hybrid_factory(gpu_first_batch=True,
        heterogeneous_candidates=True, repair_rounds=0)
    engine = next(iter(service.heterogeneous_banks.values()))
    def fail(*args):
        raise RuntimeError('injected kernel failure')
    monkeypatch.setattr(engine, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='injected kernel failure'):
        service.solve_batch([_request(p)])
    assert not service.history and not service.cpu.models


@pytest.mark.parametrize('options', [dict(speculative_cpu=True), dict(repair_rounds=1),
    dict(candidate_diagnostics=True), dict(heterogeneous_candidates=False)])
def test_incompatible_gpu_first_modes_rejected_before_setup(options):
    from types import SimpleNamespace
    from src.gpu_hybrid_lp import HybridCertifiedBackend
    kw = dict(gpu_first_batch=True, heterogeneous_candidates=True, repair_rounds=0)
    kw.update(options)
    with pytest.raises(ValueError, match='GPU-first batch'):
        HybridCertifiedBackend(SimpleNamespace(n_fluxes=2), {}, **kw)


def test_reject_policy_cannot_leave_a_cpu_only_stage():
    from types import SimpleNamespace
    from src.gpu_hybrid_lp import HybridCertifiedBackend
    with pytest.raises(ValueError, match='all original stages'):
        HybridCertifiedBackend(SimpleNamespace(n_fluxes=2), {}, gpu_first_batch=True,
            gpu_cpu_fallback='reject', gpu_stages=('maxmin',),
            heterogeneous_candidates=True, repair_rounds=0)


def test_integer_request_cannot_receive_continuous_gpu_certificate(tiny_hybrid_factory):
    service, _, p, _ = tiny_hybrid_factory(gpu_first_batch=True, gpu_cpu_fallback='reject',
        heterogeneous_candidates=True, repair_rounds=0)
    c, kwargs = _request(p)
    kwargs['integrality'] = np.ones(len(c))
    with pytest.raises(ValueError, match='continuous LPs only'):
        service.solve_batch([(c, kwargs)])
    assert not service.history
