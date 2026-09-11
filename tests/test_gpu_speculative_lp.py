"""Host-only speculative routing tests. No CUDA or CPU LP execution."""

from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest

from src.gpu_speculative_lp import solve_speculative_maxmin


class DeviceArray(np.ndarray):
    def get(self):
        return np.array(self, copy=True)


def device(value):
    return np.asarray(value).view(DeviceArray)


@pytest.fixture
def harness(monkeypatch):
    import src.speculative_cpu_batch as cpu_module
    import src.gpu_neural_basis_proposal as neural
    cp = SimpleNamespace(asarray=device,
        sum=lambda *args, **kwargs: device(np.sum(*args, **kwargs)),
        argsort=lambda *args, **kwargs: device(np.argsort(*args, **kwargs)),
        cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(neural, "features", lambda inputs: inputs["probe"])
    h = SimpleNamespace(events=[], cancelled_indices={0}, instances=[],
                        gpu_exception=None, resolve_exception=None, cpu_failures=set(),
                        evaluated_orders=[], prepared_problems=[])
    h.key = ("maxmin", 1, 1, 0)
    h.requests = [(np.array([-.1]), {"marker": i}) for i in range(4)]
    h.arrays = [(np.array([float(i)]),) for i in range(4)]
    h.candidate = dict(accepted=device([True, True, False, False]),
        candidate_index=device([1, 0, -1, -1]), values=device([[91.], [92.], [93.], [94.]]),
        objective=device([9., 9., 9., 9.]), primal_residual=device(np.zeros(4)),
        dual_violation=device(np.zeros(4)), relative_kkt_gap=device(np.zeros(4)),
        candidate_evaluations=8)
    def prepare(problems):
        h.events.append("prepare_gpu")
        h.prepared_problems.append(problems)
        return {"probe": device(np.zeros((len(problems), 1)))}
    def evaluate(inputs, order):
        h.events.append("evaluate_gpu")
        h.evaluated_orders.append(np.array(order, copy=True))
        if h.gpu_exception is not None:
            raise h.gpu_exception
        return h.candidate
    h.bank = SimpleNamespace(prepare_host=prepare, proposal_features=lambda inputs:inputs["probe"],
        centers=device([[0.], [5.]]), feature_scale=device([1.]), evaluators=[object(), object()],
        evaluate_device=evaluate)
    h.service = SimpleNamespace(repair_rounds=0, direct_cpu=object(), banks={h.key:h.bank},
        coordinates=SimpleNamespace(normalize=lambda *args:args), _speculative_previous={},
        candidate_limit=0, heterogeneous_banks={}, heterogeneous_replay=False,
        cohort_candidates=False, history=[], previous_candidates={},
        cpu=SimpleNamespace(models={"old":object()}))

    class Pending:
        def __init__(self, direct_cpu, requests, ids):
            assert direct_cpu is h.service.direct_cpu
            h.events.append("dispatch_cpu")
            self.requests = requests
            self.ids = list(ids)
            self._finished = False
            self.telemetry = None
            self.drained = False
            h.instances.append(self)

        def resolve(self, accepted):
            h.events.append("resolve_cpu")
            self.accepted = np.asarray(accepted).copy()
            if h.resolve_exception is not None:
                self._finished = True
                self.telemetry = dict(joined_all=True)
                raise h.resolve_exception
            rows = []
            for index, env in enumerate(self.ids):
                if accepted[index] and index in h.cancelled_indices:
                    rows.append(None)
                    continue
                success = index not in h.cpu_failures
                rows.append(SimpleNamespace(success=success,
                    x=np.array([1000.+index]) if success else None,
                    fun=2000.+index if success else None,
                    diagnostics=dict(environment_id=env, stage="maxmin", success=success,
                        cpu_lp_calls=1, cpu_solver_runs=1, numerical_retry_count=0,
                        primal_residual=0. if success else 1., dual_violation=0., relative_kkt_gap=0.,
                        message="unused worker failed" if not success else "CPU exact")))
            started = sum(row is not None for row in rows)
            unused = sum(bool(accepted[i]) and row is not None for i, row in enumerate(rows))
            self._finished = True
            self.telemetry = dict(cpu_jobs_actual_started=started,
                cpu_jobs_cancelled=len(rows)-started, joined_all=True, outstanding_jobs=0,
                unused_cpu_results_gpu_accepted=unused,
                cpu_results_used=int((~accepted).sum()))
            return rows, self.telemetry

        def drain(self):
            if self._finished:
                raise RuntimeError("single use already finished")
            self.drained = True
            self._finished = True
            h.events.append("drain_cpu")

    monkeypatch.setattr(cpu_module, "SpeculativeCpuBatch", Pending)
    h.run = lambda ids=None: solve_speculative_maxmin(h.service, h.requests, h.key, h.arrays,
        [11, 29, 31, 47] if ids is None else ids, time.perf_counter())
    return h


def test_cpu_results_never_enter_gpu_inputs_and_only_certified_rows_can_cancel(harness):
    h = harness
    rows = h.run()
    assert h.events == ["dispatch_cpu", "prepare_gpu", "evaluate_gpu", "resolve_cpu"]
    assert h.instances[0].requests is h.requests
    assert h.prepared_problems[0] == h.arrays
    assert all("_initial_basis" not in kwargs for _, kwargs in h.requests)
    np.testing.assert_array_equal(h.instances[0].accepted, [True, True, False, False])
    np.testing.assert_array_equal([row.x[0] for row in rows], [91., 92., 1002., 1003.])
    assert rows[0].diagnostics["cpu_lp_calls"] == 0
    assert rows[1].diagnostics["cpu_lp_calls"] == 1
    assert rows[1].diagnostics["cpu_result_used"] is False
    assert rows[2].diagnostics["cpu_result_used"] is True


def test_actual_started_unused_cancelled_and_cpu_adoption_counts_are_distinct(harness):
    h = harness
    h.run()
    record = h.service.history[-1]
    assert record["bank_accepts"] == 2
    assert record["cpu_lp_calls"] == 3
    assert record["cpu_results_used"] == 2
    assert record["cpu_speculative_unused"] == 1
    assert record["cpu_speculative_cancelled"] == 1
    assert record["bank_accepts"] + record["cpu_lp_calls"] - record["cpu_speculative_unused"] == record["batch"]
    assert record["cpu_solver_runs"] == 3
    assert record["groups"][0]["ids"] == [1, 2, 3]
    assert record["groups"][0]["environment_ids"] == [29, 31, 47]
    assert [row["cpu_result_used"] for row in record["groups"][0]["rows"]] == [False, True, True]


@pytest.mark.parametrize("field,value", [("values", np.nan), ("objective", np.inf),
    ("primal_residual", np.nan), ("primal_residual", 1e-4),
    ("dual_violation", 1e-6), ("relative_kkt_gap", np.inf),
    ("primal_residual", -1.), ("dual_violation", -1.), ("relative_kkt_gap", -1.),
    ("candidate_index", -1), ("candidate_index", 2)])
def test_gpu_nonfinite_or_violated_gate_uses_cpu_and_cannot_cancel(harness, field, value):
    h = harness
    h.candidate[field][0] = value
    rows = h.run()
    assert h.instances[0].accepted.tolist() == [False, True, False, False]
    assert rows[0].diagnostics["cpu_result_used"] is True
    assert rows[0].x[0] == 1000.
    assert h.service.history[-1]["cpu_speculative_cancelled"] == 0


@pytest.mark.parametrize('field,value',[
    ('values',np.zeros((4,2))),('objective',np.zeros((4,1))),
    ('primal_residual',np.zeros((4,1))),('candidate_index',np.zeros(4,dtype=float))])
def test_malformed_gpu_shapes_or_index_type_drain_and_raise(harness,field,value):
    h=harness
    h.candidate[field]=device(value)
    with pytest.raises(ValueError,match='Malformed'):
        h.run()
    assert h.instances[0].drained
    assert not h.service.history


def test_false_gpu_flag_is_not_overridden_by_finite_residuals(harness):
    h = harness
    h.candidate["accepted"][:] = False
    rows = h.run()
    assert all(row.message == "cpu_speculative_exact" for row in rows)
    assert h.service.history[-1]["bank_accepts"] == 0
    assert h.service.history[-1]["cpu_lp_calls"] == 4


def test_gpu_exception_drains_every_cpu_job_and_preserves_exception(harness):
    h = harness
    h.gpu_exception = RuntimeError("original GPU failure")
    with pytest.raises(RuntimeError, match="original GPU failure"):
        h.run()
    assert h.instances[0].drained
    assert h.service.history == []
    assert h.service._speculative_previous == {}


def test_cpu_resolution_failure_does_not_redrain_or_hide_original_exception(harness):
    h = harness
    h.resolve_exception = RuntimeError("original CPU joined failure")
    with pytest.raises(RuntimeError, match="original CPU joined failure"):
        h.run()
    assert not h.instances[0].drained
    assert h.service.history == []


def test_unused_failed_cpu_result_remains_in_diagnostics_not_final_gpu_gate(harness):
    h = harness
    h.cpu_failures.add(1)
    rows = h.run()
    assert rows[1].success
    assert rows[1].x[0] == 92.
    diagnostic = h.service.history[-1]["groups"][0]["rows"][0]
    assert diagnostic["success"] is False
    assert diagnostic["cpu_result_used"] is False
    assert diagnostic["message"] == "unused worker failed"
    assert h.service.history[-1]["accepted"] == [True]*4


def test_required_cpu_failure_returns_no_solution_and_clears_previous_choice(harness):
    h = harness
    h.cpu_failures.add(2)
    h.service._speculative_previous[(h.key, 31)] = 1
    rows = h.run()
    assert not rows[2].success and rows[2].x is None and rows[2].fun is None
    assert h.service._speculative_previous[(h.key, 31)] == -1


def test_previous_candidate_mapping_tracks_ids_after_reorder_and_reset(harness):
    h = harness
    h.run()
    h.run(ids=[29, 11, 47, 31])
    order = h.evaluated_orders[-1]
    assert order[0, 0] == 0  # env29 previously accepted candidate0
    assert order[1, 0] == 1  # env11 previously accepted candidate1
    from src.gpu_hybrid_lp import HybridCertifiedBackend
    HybridCertifiedBackend.reset_trajectory(h.service)
    assert h.service._speculative_previous == {}
    assert h.service.previous_candidates == {}
    assert h.service.cpu.models == {}


def _three_candidate_learned_router(h):
    """Give the host-only harness a complete learned order [0, 1, 2]."""
    h.bank.evaluators = [object(), object(), object()]
    h.bank.centers = device([[0.], [5.], [10.]])

    class Router:
        candidate_count = 3
        input_dim = 1

        def rank(self, features, k=None):
            assert k == 3
            return device(np.tile([0, 1, 2], (len(features), 1)))

    h.service.coverage_routers = {h.key: Router()}
    h.service.candidate_limit = 2


def test_within_budget_temporal_order_never_displaces_learned_top_k(harness):
    h = harness
    _three_candidate_learned_router(h)
    h.service.temporal_candidate_policy = 'within-budget'
    # Candidate 2 is stale/outside the learned top-2 for env 11; candidate 1
    # is still in-budget for env 29 and may therefore be promoted safely.
    h.service._speculative_previous[(h.key, 11)] = 2
    h.service._speculative_previous[(h.key, 29)] = 1
    h.run()
    assert h.evaluated_orders[-1].tolist() == [
        [0, 1],  # out-of-budget previous candidate cannot evict candidate 1
        [1, 0],  # in-budget previous candidate is reordered to the first slot
        [0, 1],
        [0, 1],
    ]
    record = h.service.history[-1]
    assert record['temporal_candidate_policy'] == 'within-budget'
    assert record['candidate_router'] == 'learned_coverage'


def test_default_temporal_policy_preserves_legacy_prepend_semantics(harness):
    h = harness
    _three_candidate_learned_router(h)
    # The service fixture intentionally has no temporal_candidate_policy;
    # getattr(..., 'prepend') is the backward-compatible PPO/default path.
    h.service._speculative_previous[(h.key, 11)] = 2
    h.candidate['candidate_index'][0] = 2
    h.run()
    assert h.evaluated_orders[-1][0].tolist() == [2, 0]
    assert h.service.history[-1]['temporal_candidate_policy'] == 'prepend'


def test_coverage_router_changes_only_speculative_gpu_order_and_is_recorded(harness):
    h = harness
    class Router:
        candidate_count = 2
        input_dim = 1
        def rank(self, features, k=None):
            assert k == 2
            return device(np.tile([1, 0], (len(features), 1)))
    h.service.coverage_routers = {h.key:Router()}
    h.run()
    assert h.evaluated_orders[-1].tolist() == [[1, 0]]*4
    assert h.service.history[-1]['candidate_router'] == 'learned_coverage'


def test_device_validity_gate_and_packed_transfer_cannot_cancel_invalid_router_row(harness):
    h=harness;cp=sys.modules['cupy']
    cp.float64=np.float64
    cp.where=lambda *a:device(np.where(*a))
    cp.arange=lambda *a:device(np.arange(*a))
    cp.take_along_axis=lambda *a,**kw:device(np.take_along_axis(*a,**kw))
    cp.concatenate=lambda *a,**kw:device(np.concatenate(*a,**kw))
    class Router:
        def rank_with_validity(self,features,k=None):
            return device(np.tile([1,0],(4,1))),device([False,True,True,True])
        def rank(self,*args,**kwargs):raise AssertionError('Host rank path forbidden')
    h.service.coverage_routers={h.key:Router()}
    h.service.device_routing=True;h.service.packed_result_transfer=True
    h.service.temporal_candidate_policy='within-budget';h.service.candidate_limit=1
    rows=h.run()
    assert h.instances[0].accepted.tolist()==[False,True,False,False]
    assert rows[0].x[0]==1000. and rows[0].diagnostics['cpu_result_used']
    assert h.evaluated_orders[-1].tolist()==[[1]]*4
    assert h.service.history[-1]['device_routing']
    assert h.service.history[-1]['packed_result_transfer']


def test_delayed_cpu_dispatch_preserves_gate_but_avoids_gpu_prepare_contention(harness):
    h=harness;h.service.speculative_dispatch_phase='after-submit'
    rows=h.run()
    assert h.events==['prepare_gpu','evaluate_gpu','dispatch_cpu','resolve_cpu']
    assert h.instances[0].accepted.tolist()==[True,True,False,False]
    assert [row.x[0] for row in rows]==[91.,92.,1002.,1003.]
    record=h.service.history[-1]
    assert record['speculative_dispatch_phase']=='after-submit'
    phases=record['gpu_host_observed_phases']
    assert sum(v for k,v in phases.items() if k.endswith('_seconds'))==pytest.approx(record['gpu_probe_span_seconds'])


def test_gpu_failure_before_delayed_cpu_dispatch_creates_no_pending_worker(harness):
    h=harness;h.service.speculative_dispatch_phase='after-submit'
    h.gpu_exception=RuntimeError('probe failed before CPU dispatch')
    with pytest.raises(RuntimeError,match='probe failed'):h.run()
    assert not h.instances and not h.service.history


def test_delayed_cpu_constructor_failure_drains_already_enqueued_gpu(harness,monkeypatch):
    h=harness;h.service.speculative_dispatch_phase='after-submit'
    import src.speculative_cpu_batch as cpu_module
    def fail(*args):raise RuntimeError('CPU proposal setup failed')
    monkeypatch.setattr(cpu_module,'SpeculativeCpuBatch',fail)
    calls=[]
    sys.modules['cupy'].cuda.get_current_stream=lambda:SimpleNamespace(synchronize=lambda:calls.append('drained'))
    with pytest.raises(RuntimeError,match='CPU proposal setup failed'):h.run()
    assert calls==['drained'] and not h.instances


def test_gpu_stream_cleanup_failure_does_not_hide_original_error(harness):
    h=harness;h.service.speculative_dispatch_phase='after-submit'
    h.gpu_exception=RuntimeError('probe broke')
    def failed_sync():raise RuntimeError('stream broke')
    sys.modules['cupy'].cuda.get_current_stream=lambda:SimpleNamespace(synchronize=failed_sync)
    with pytest.raises(BaseExceptionGroup) as found:h.run()
    assert [str(e) for e in found.value.exceptions]==['probe broke','stream broke']


@pytest.mark.parametrize("accepted", [[True], [1, 1, 0, 0]])
def test_malformed_gpu_acceptance_drains_cpu_batch(harness, accepted):
    h = harness
    h.candidate["accepted"] = device(accepted)
    with pytest.raises(ValueError, match="boolean vector"):
        h.run()
    assert h.instances[0].drained


def test_repair_mode_not_silently_enabled_by_speculation(harness):
    h = harness
    h.service.repair_rounds = 1
    with pytest.raises(ValueError, match="without repair"):
        h.run()
    assert h.instances == []


@pytest.mark.parametrize('options', [
    dict(certificate_only=1, speculative_cpu=True, heterogeneous_candidates=True),
    dict(certificate_only=True, speculative_cpu=False, heterogeneous_candidates=True),
    dict(certificate_only=True, speculative_cpu=True, heterogeneous_candidates=False),
])
def test_certificate_only_rejects_nonboolean_or_non_speculative_combinations(options):
    from src.gpu_hybrid_lp import HybridCertifiedBackend

    with pytest.raises(ValueError, match='Certificate-only probes require speculative heterogeneous mode'):
        HybridCertifiedBackend(None, {}, repair_rounds=0,
            gpu_stages=('maxmin',), **options)
