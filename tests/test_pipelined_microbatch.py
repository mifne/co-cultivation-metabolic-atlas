from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
from greenlet import getcurrent

from scripts.pipelined_microbatch import drive_pipelined


def request(stage, environment_id, value):
    if stage == "maxmin":
        objective = np.array([0.0, 0.0, -1.0])
    elif stage == "aggregate":
        objective = np.array([-1.0, -1.0, 0.0])
    elif stage == "exchange":
        objective = np.array([0.0, 0.0, 0.0, 1.0])
    else:  # pragma: no cover - test helper contract
        raise ValueError(stage)
    return objective, dict(
        A_eq=np.zeros((1, len(objective))), b_eq=np.zeros(1),
        bounds=[(0.0, None)] * len(objective), method="highs-ds",
        _stage=stage, _environment_id=environment_id, _value=float(value))


def solved(environment_id, stage, value):
    diagnostics = dict(
        environment_id=environment_id, stage=stage, cpu_lp_calls=1,
        cpu_solver_runs=1, numerical_retry_count=0,
        initial_basis_used=False, primal_residual=0.0,
        dual_violation=0.0, relative_kkt_gap=0.0)
    return SimpleNamespace(success=True, x=np.array([value]), fun=-value,
                           diagnostics=diagnostics, message="optimal")


class FakeExactCpu:
    def __init__(self, hooks=None):
        self.hooks = hooks or {}
        self.calls = []
        self.lock = threading.Lock()

    def _solve(self, job):
        environment_id, (_, kwargs) = job
        stage = kwargs["_stage"]
        assert threading.current_thread() is not threading.main_thread()
        with self.lock:
            self.calls.append((environment_id, stage, "start"))
        hook = self.hooks.get((environment_id, stage))
        if hook is not None:
            hook()
        with self.lock:
            self.calls.append((environment_id, stage, "finish"))
        return solved(environment_id, stage, kwargs["_value"])


class FakeDictionary:
    def __init__(self, coordinates, exact=None):
        self.coordinates = coordinates
        self.cpu = exact or FakeExactCpu()
        self.history = []
        self.pipeline_history = []
        self.prepared = []
        self.main_thread = threading.get_ident()

    def prepare_request(self, objective, kwargs, environment_id):
        assert threading.get_ident() == self.main_thread
        assert environment_id == kwargs["_environment_id"]
        self.prepared.append((environment_id, kwargs["_stage"]))
        return (objective, dict(kwargs)), dict(
            stage=kwargs["_stage"], dictionary_reason="fake",
            dictionary_selection_seconds=0.0)

    def solve_batch(self, requests, *, environment_ids=None):
        assert threading.get_ident() == self.main_thread
        assert environment_ids is None
        environment_ids = list(range(len(requests)))
        results = [solved(i, "maxmin", row[1]["_value"])
                   for i, row in zip(environment_ids, requests)]
        self.history.append(dict(
            batch=len(results), cpu_lp_calls=len(results), seconds=0.0,
            rows=[dict(result.diagnostics) for result in results]))
        return results


class FakeHybrid:
    def __init__(self, dictionary):
        self.coordinates = dictionary.coordinates
        self.direct_cpu = dictionary
        self.cpu = dictionary.cpu
        self.history = []
        self.pipeline_history = []
        self.maxmin_batches = []

    def solve_batch(self, requests):
        ids = [kwargs["_environment_id"] for _, kwargs in requests]
        assert ids == list(range(len(requests)))
        assert all(kwargs["_stage"] == "maxmin" for _, kwargs in requests)
        self.maxmin_batches.append(ids)
        results = [solved(i, "maxmin", kwargs["_value"])
                   for i, (_, kwargs) in enumerate(requests)]
        self.history.append(dict(
            stage="maxmin", batch=len(results), bank_accepts=len(results),
            candidate_engine="fake_gpu", candidate_evaluations=len(results),
            candidate_count=[1] * len(results), observable_dispersion=[0.0] * len(results),
            preparation_seconds=0.0, bank_seconds=0.0, seconds=0.0,
            cpu_lp_calls=0, routes=["gpu_dictionary"] * len(results), groups=[],
            accepted=[True] * len(results), primal_residual=[0.0] * len(results),
            dual_violation=[0.0] * len(results), relative_kkt_gap=[0.0] * len(results)))
        return results


class ToyEnvironment:
    def __init__(self, environment_id, parent, overlap=None, initial_delay=0.0):
        self.environment_id = environment_id
        self.parent = parent
        self.overlap = overlap
        self.initial_delay = initial_delay
        self.value = 0.0
        self.events = []
        self.model = SimpleNamespace(stats=SimpleNamespace(status="optimal;"))
        self.thread = threading.get_ident()

    def step(self, action):
        assert threading.get_ident() == self.thread
        if self.initial_delay:
            time.sleep(self.initial_delay)
        maxmin = self.parent.switch(request(
            "maxmin", self.environment_id, action + self.environment_id))
        self.events.append("maxmin")
        if self.overlap is not None and self.environment_id == 1:
            assert self.overlap["started"].wait(1.0)
            self.overlap["seen"] = True
            self.overlap["release"].set()
        aggregate = self.parent.switch(request(
            "aggregate", self.environment_id, action + 10 + self.environment_id))
        self.events.append("aggregate")
        exchange = self.parent.switch(request(
            "exchange", self.environment_id, action + 20 + self.environment_id))
        self.events.append("exchange")
        self.value += maxmin.x[0] + aggregate.x[0] + exchange.x[0]


def make_environments(count, overlap=None, initial_delay=0.0):
    parent = getcurrent()
    rows = []
    for environment_id in range(count):
        env = ToyEnvironment(environment_id, parent, overlap, initial_delay)
        rows.append((env, env.model))
    return rows


def test_pipeline_preserves_stage_order_stable_ids_and_nonoverlap_timing():
    coordinates = SimpleNamespace(n_fluxes=2)
    dictionary = FakeDictionary(coordinates)
    service = FakeHybrid(dictionary)
    environments = make_environments(3)
    progress = [0, 0, 0]
    outputs = drive_pipelined(
        environments, [[1, 2], [3, 4], [5, 6]], service,
        lambda env: env.value, progress, workers=2)

    assert progress == [2, 2, 2]
    assert outputs == [69.0, 87.0, 105.0]
    assert service.maxmin_batches == [[0, 1, 2], [0, 1, 2]]
    assert [row["stage"] for row in service.history] == [
        "maxmin", "aggregate", "exchange", "maxmin", "aggregate", "exchange"]
    assert all(row["seconds"] == 0.0 for row in service.history
               if row["stage"] != "maxmin")
    assert all(row["groups"][0]["ids"] == [0, 1, 2] for row in service.history
               if row["stage"] != "maxmin")
    for environment_id in range(3):
        stages = [stage for i, stage, event in dictionary.cpu.calls
                  if i == environment_id and event == "start"]
        assert stages == ["aggregate", "exchange", "aggregate", "exchange"]
    assert len(service.pipeline_history) == 2
    for row in service.pipeline_history:
        accounted = sum(row[name] for name in (
            "maxmin_service_seconds", "main_thread_resume_seconds",
            "cpu_preparation_seconds", "cpu_wait_seconds", "scheduler_overhead_seconds"))
        assert accounted == pytest.approx(row["total_seconds"], abs=1e-9)
        assert row["async_span_seconds"]["aggregate"] >= 0
        assert row["async_span_seconds"]["exchange"] >= 0


def test_host_work_continues_while_exact_cpu_job_is_pending():
    started = threading.Event()
    release = threading.Event()
    overlap = dict(started=started, release=release, seen=False)

    def block_first_aggregate():
        started.set()
        assert release.wait(1.0)

    exact = FakeExactCpu({(0, "aggregate"): block_first_aggregate})
    dictionary = FakeDictionary(SimpleNamespace(n_fluxes=2), exact)
    service = FakeHybrid(dictionary)
    outputs = drive_pipelined(
        make_environments(2, overlap), [[1], [2]], service,
        lambda env: env.value, workers=2)

    assert overlap["seen"]
    assert outputs == [33.0, 39.0]
    assert all(thread == dictionary.main_thread for thread in [
        env.thread for env, _ in make_environments(1)])


def test_worker_failure_joins_every_pending_cpu_solve():
    second_started = threading.Event()
    second_finished = threading.Event()

    def fail_after_peer_starts():
        assert second_started.wait(1.0)
        raise ValueError("deliberate exact failure")

    def finish_peer():
        second_started.set()
        time.sleep(0.05)
        second_finished.set()

    exact = FakeExactCpu({
        (0, "aggregate"): fail_after_peer_starts,
        (1, "aggregate"): finish_peer,
    })
    dictionary = FakeDictionary(SimpleNamespace(n_fluxes=2), exact)
    service = FakeHybrid(dictionary)

    with pytest.raises(ValueError, match="deliberate exact failure"):
        drive_pipelined(
            make_environments(2), [[1], [2]], service,
            lambda env: env.value, workers=2)
    assert second_finished.is_set()
    failure = service.pipeline_history[-1]
    assert failure["failed"] and failure["pending_joined"]
    assert len(service.pipeline_failed_requests) == 1
    assert len(service.pipeline_failed_diagnostics) == 1
    failed_request = service.pipeline_failed_requests[0]
    failed_diagnostics = service.pipeline_failed_diagnostics[0]
    assert failed_request[1]["_environment_id"] == failed_diagnostics["environment_id"] == 0
    assert failed_request[1]["_stage"] == failed_diagnostics["stage"] == "aggregate"
    assert failed_diagnostics["success"] is False


def test_cpu_comparator_and_hybrid_pipeline_produce_identical_snapshots():
    actions = [[0.5, 1.5], [2.5, 3.5]]
    coordinates = SimpleNamespace(n_fluxes=2)
    cpu = FakeDictionary(coordinates)
    hybrid_dictionary = FakeDictionary(coordinates)
    hybrid = FakeHybrid(hybrid_dictionary)

    cpu_rows = drive_pipelined(
        make_environments(2), actions, cpu, lambda env: env.value, workers=2)
    hybrid_rows = drive_pipelined(
        make_environments(2), actions, hybrid, lambda env: env.value, workers=2)

    np.testing.assert_allclose(cpu_rows, hybrid_rows, rtol=0, atol=0)
    assert [row["stage"] for row in cpu.history] == [
        "maxmin", "aggregate", "exchange", "maxmin", "aggregate", "exchange"]
    assert [row["stage"] for row in hybrid.history] == [
        "maxmin", "aggregate", "exchange", "maxmin", "aggregate", "exchange"]


def test_pipeline_rejects_unequal_action_counts_before_starting_workers():
    service = FakeHybrid(FakeDictionary(SimpleNamespace(n_fluxes=2)))
    with pytest.raises(ValueError, match="equal action counts"):
        drive_pipelined(
            make_environments(2), [[1], [2, 3]], service,
            lambda env: env.value, workers=2)


def test_cycle_zero_accounts_initial_host_request_construction_as_resume_time():
    service = FakeHybrid(FakeDictionary(SimpleNamespace(n_fluxes=2)))
    drive_pipelined(
        make_environments(1, initial_delay=0.02), [[1]], service,
        lambda env: env.value, workers=1)

    timing = service.pipeline_history[0]
    assert timing["main_thread_resume_seconds"] >= 0.018
