"""Host-only dispatch tests; NumPy stubs deliberately do not invoke CUDA/LPs."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import src.gpu_hybrid_lp as module


class DeviceArray(np.ndarray):
    def get(self): return np.array(self, copy=True)


def device(value, dtype=None): return np.asarray(value, dtype=dtype).view(DeviceArray)


@pytest.fixture
def host_service(monkeypatch):
    cp = SimpleNamespace(ndarray=DeviceArray, int32=np.int32, int64=np.int64, inf=np.inf,
        asarray=device, zeros=lambda *args, **kwargs:device(np.zeros(*args, **kwargs)),
        full=lambda *args, **kwargs:device(np.full(*args, **kwargs)),
        sum=lambda *args, **kwargs:device(np.sum(*args, **kwargs)),
        argsort=lambda *args, **kwargs:device(np.argsort(*args, **kwargs)),
        cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None),
            runtime=SimpleNamespace(memGetInfo=lambda:(100, 200))))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    import src.gpu_neural_basis_proposal as neural
    monkeypatch.setattr(neural, 'features', lambda inputs:device(np.zeros((4, 1))))
    key = ('aggregate', 0, 1, 0)
    monkeypatch.setattr(module, 'stage_key', lambda *args:key)
    bank_result = dict(accepted=device([True, False, False, False]), candidate_index=device([0, -1, -1, -1]),
        best_candidate_index=device([0, 1, 0, 1]), candidate_count=device([2, 2, 2, 1]),
        observable_dispersion=device([0., .01, 10., np.inf]), candidate_evaluations=8,
        values=device([[99.], [np.nan], [np.nan], [np.nan]]), objective=device([99., np.inf, np.inf, np.inf]),
        primal_residual=device([0., 1., 1., 1.]), dual_violation=device([0., 1., 1., 1.]),
        relative_kkt_gap=device([0., 1., 1., 1.]))
    bank = SimpleNamespace(feature_indices=np.array([0]), centers=np.zeros((2, 1)), feature_scale=np.ones(1),
        candidate_ranking='residual', prepare_host=lambda problems:{},
        evaluate_device=lambda inputs, order:bank_result)
    calls, failed = [], set()
    class Cpu:
        def solve_batch(self, requests, *, environment_ids=None):
            calls.append((list(environment_ids), requests))
            return [SimpleNamespace(success=i not in failed, x=np.array([10.+i]), fun=10.+i,
                diagnostics=dict(environment_id=i, primal_residual=1. if i in failed else 0.,
                    dual_violation=0., relative_kkt_gap=0.)) for i in environment_ids]
        def close(self): pass
    # Only the routing control flow is real; neither __init__ nor a GPU
    # evaluator/CPU optimizer is invoked by these host wiring tests.
    service = module.HybridCertifiedBackend.__new__(module.HybridCertifiedBackend)
    service.coordinates = SimpleNamespace(n_fluxes=1, normalize=lambda *args:args)
    service.banks = {key:bank}; service.previous_candidates = {}; service.candidate_limit = 0
    service.cohort_candidates = False; service.dispersion_threshold = 1.; service.repair_rounds = 0
    service.heterogeneous_banks = {}; service.heterogeneous_replay = False
    service.repair_columns = 1; service.repair_pivots = 1; service.max_repair_batch = 4; service.bucket = False
    service.cpu_basis_proposals = True
    templates = [dict(col_status=np.array([index], dtype=np.int32), row_status=np.array([], dtype=np.int32))
                 for index in (0, 3)]
    service.host_bases = {key:templates}; service.cpu = Cpu(); service.dispatch = ThreadPoolExecutor(max_workers=1)
    reducer = SimpleNamespace(nonbasic0=np.array([0]), run_device=lambda **kwargs:
        dict(accepted=device([False]), restricted_pivots=device([0]), warm_state={}))
    solver = SimpleNamespace(restricted_solver=reducer, prepare_host=lambda problems:{})
    service.operators = SimpleNamespace(repairs={}, solver=lambda *args:solver)
    service.history = []
    requests = [(np.array([1.]), dict(bounds=[(0., 100.)], request_marker=i)) for i in range(4)]
    yield service, requests, calls, failed, templates, bank_result
    service.close()


def test_zero_gpu_budget_dispatches_all_misses_once_with_original_ids_bases_and_labels(host_service):
    service, requests, calls, failed, templates, _ = host_service
    result = service.solve_batch(requests)
    assert len(calls) == 1 and calls[0][0] == [1, 2, 3]
    assert [request[1]['request_marker'] for request in calls[0][1]] == [1, 2, 3]
    for request, basis in zip(calls[0][1], (templates[1], templates[0], templates[1])):
        assert request[1]['_initial_basis'] is basis
    assert all('_initial_basis' not in kwargs for _, kwargs in requests)
    assert service.history[-1]['routes'] == ['gpu_dictionary', 'cpu_gpu_budget_exhausted',
        'cpu_high_dispersion', 'cpu_unknown_dispersion']
    assert service.history[-1]['cpu_lp_calls'] == 3
    assert len(service.history[-1]['groups']) == 1
    np.testing.assert_array_equal([row.x[0] for row in result], [99., 11., 12., 13.])


def test_zero_gpu_budget_does_not_resend_failed_low_dispersion_cpu_row(host_service):
    service, requests, calls, failed, _, _ = host_service
    failed.add(1)
    result = service.solve_batch(requests)
    assert len(calls) == 1 and calls[0][0] == [1, 2, 3]
    assert not result[1].success and result[1].x is None and result[1].fun is None
    assert service.history[-1]['cpu_lp_calls'] == 3


def test_nonzero_gpu_budget_preserves_early_cpu_and_later_failed_repair_waves(host_service):
    service, requests, calls, failed, _, _ = host_service
    service.repair_rounds = 1
    result = service.solve_batch(requests)
    assert [ids for ids, _ in calls] == [[2, 3], [1]]
    assert all(row.success for row in result)
    assert service.history[-1]['cpu_lp_calls'] == 3
    assert any(group['route'] == 'gpu_restricted_repair' for group in service.history[-1]['groups'])


def test_all_dictionary_acceptances_never_launch_a_cpu_batch(host_service):
    service, requests, calls, failed, _, bank_result = host_service
    bank_result['accepted'] = device([True]*4)
    bank_result['values'] = device([[99.]]*4); bank_result['objective'] = device([99.]*4)
    assert all(row.success for row in service.solve_batch(requests))
    assert not calls and service.history[-1]['cpu_lp_calls'] == 0


def test_missing_candidate_never_selects_last_basis_by_negative_index(host_service):
    service, requests, calls, failed, templates, bank_result = host_service
    bank_result['best_candidate_index'][1] = -1
    result = service.solve_batch(requests)
    assert all(row.success for row in result)
    assert '_initial_basis' not in calls[0][1][0][1]
    assert calls[0][1][1][1]['_initial_basis'] is templates[0]


@pytest.mark.parametrize('repair_passes',[False,True])
def test_near_dual_policy_can_repair_high_dispersion_but_never_uncertified_rows(host_service,monkeypatch,repair_passes):
    service,requests,calls,failed,templates,bank_result=host_service
    old_key=next(iter(service.banks));key=('maxmin',*old_key[1:])
    bank=service.banks.pop(old_key);service.banks[key]=bank
    service.host_bases[key]=service.host_bases.pop(old_key)
    monkeypatch.setattr(module,'stage_key',lambda *args:key)
    service.repair_policy='maxmin-dual';service.repair_rounds=1
    service.compact_repair_workspace=True
    bank.candidate_ranking='count'  # Must not replace the dual-qualified best candidate.
    bank_result.update(input_family_valid=device([True]*4),basis_dual_violation=device([0.]*4),
        primal_violation_count=device([0,1,2,1]),dual_violation=device([0.]*4),
        relative_kkt_gap=device([0.,0.,0.,1.]))
    repairs=[]
    def repair(**kwargs):
        repairs.append(kwargs)
        return dict(accepted=device([repair_passes]),restricted_pivots=device([1]),warm_state={},
            values=device([[77.]]),objective=device([77.]),primal_residual=device([0.]),
            dual_violation=device([0.]),relative_kkt_gap=device([0.]))
    service.operators.solver(None,None).restricted_solver.run_device=repair
    result=service.solve_batch(requests)
    assert len(repairs)==2 and all(call['compute_expansion'] is False for call in repairs)
    assert service.history[-1]['repair_eligible']==[False,True,True,False]
    assert calls[0][0]==[3]
    assert service.history[-1]['routes'][3]=='cpu_not_near_dual_maxmin'
    if repair_passes:
        assert len(calls)==1
        assert service.history[-1]['routes'][1:3]==['gpu_restricted_repair']*2
        assert result[1].x[0]==77. and result[2].x[0]==77.
    else:
        assert calls[1][0]==[1,2]
        assert service.history[-1]['routes'][1:3]==['cpu_gpu_budget_exhausted']*2
    assert all(row.success for row in result)
