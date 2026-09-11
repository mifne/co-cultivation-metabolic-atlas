from concurrent.futures import CancelledError
from types import SimpleNamespace

import numpy as np
import pytest

from src.cpu_repeated_lp import RepeatedCpuLP
from src.speculative_cpu_batch import SpeculativeCpuBatch


class _Future:
    def __init__(self, *, running=False, value=None, error=None):
        self.running = running
        self.value = value
        self.error = error
        self.was_cancelled = False
        self.result_calls = 0

    def cancel(self):
        if self.running or self.was_cancelled:
            return False
        self.was_cancelled = True
        return True

    def cancelled(self):
        return self.was_cancelled

    def done(self):
        return self.was_cancelled or self.running

    def result(self):
        self.result_calls += 1
        if self.was_cancelled:
            raise CancelledError()
        self.running = True
        if self.error is not None:
            raise self.error
        return self.value


class _Pool:
    def __init__(self, futures, fail_at=None):
        self.futures = list(futures)
        self.fail_at = fail_at
        self.submissions = []

    def submit(self, function, *args):
        if self.fail_at is not None and len(self.submissions) == self.fail_at:
            raise RuntimeError("submit failed")
        self.submissions.append((function, args))
        return self.futures[len(self.submissions) - 1]


class _Dictionary:
    def __init__(self, futures, fail_at=None):
        self.cpu = SimpleNamespace(
            closed=False,
            pool=_Pool(futures, fail_at=fail_at),
            _solve=lambda job: None,
        )
        self.prepared_ids = []

    def prepare_request(self, objective, kwargs, environment_id):
        self.prepared_ids.append(environment_id)
        updated = dict(kwargs, prepared_for=environment_id)
        return (objective, updated), {"proposal": environment_id}


def _result(index, runs=1):
    return SimpleNamespace(
        success=True,
        diagnostics={"cpu_solver_runs": runs, "row": index},
    )


def _requests(count):
    return [(np.array([float(index)]), {}) for index in range(count)]


def test_resolve_cancels_only_certified_pending_jobs_and_joins_everything():
    futures = [
        _Future(running=False, value=_result(0)),
        _Future(running=False, value=_result(1)),
        _Future(running=True, value=_result(2, runs=2)),
        _Future(running=True, value=_result(3)),
    ]
    dictionary = _Dictionary(futures)
    batch = SpeculativeCpuBatch(dictionary, _requests(4), [10, 20, 30, 40])
    results, telemetry = batch.resolve(np.array([True, False, True, False]))

    assert results[0] is None
    assert results[1].diagnostics["proposal"] == 20
    assert results[2].diagnostics["proposal"] == 30
    assert results[3].diagnostics["proposal"] == 40
    assert futures[0].was_cancelled
    assert not futures[1].was_cancelled  # Uncertified rows are never cancelled.
    assert not futures[2].was_cancelled  # Already-running certified work is joined.
    assert all(future.result_calls == 1 for future in futures)
    assert telemetry is batch.telemetry
    assert telemetry["gpu_certified_rows"] == 2
    assert telemetry["cpu_jobs_cancelled"] == 1
    assert telemetry["cpu_jobs_actual_started"] == 3
    assert telemetry["cpu_jobs_completed"] == 3
    assert telemetry["cpu_solver_runs"] == 4
    assert telemetry["cpu_results_used"] == 2
    assert telemetry["unused_cpu_results_gpu_accepted"] == 1
    assert telemetry["cancelled_environment_ids"] == [10]
    assert telemetry["joined_all"] and telemetry["outstanding_jobs"] == 0
    assert dictionary.cpu._speculative_batch_active is None
    with pytest.raises(RuntimeError, match="single use"):
        batch.abort()


def test_abort_cancels_nothing_and_drains_all_jobs_before_reporting_all_errors():
    futures = [
        _Future(running=True, error=ValueError("first")),
        _Future(running=True, value=_result(1)),
        _Future(running=True, error=RuntimeError("second")),
    ]
    dictionary = _Dictionary(futures)
    batch = SpeculativeCpuBatch(dictionary, _requests(3), [1, 2, 3])
    with pytest.raises(BaseExceptionGroup) as captured:
        batch.abort()
    assert len(captured.value.exceptions) == 2
    assert all(not future.was_cancelled for future in futures)
    assert all(future.result_calls == 1 for future in futures)
    assert batch.telemetry["cpu_jobs_failed"] == 2
    assert batch.telemetry["gpu_certified_rows"] == 0
    assert batch.telemetry["joined_all"] and batch.telemetry["outstanding_jobs"] == 0
    assert dictionary.cpu._speculative_batch_active is None


def test_submission_failure_cancels_and_drains_every_previously_submitted_job():
    futures = [
        _Future(running=True, value=_result(0)),
        _Future(running=False, value=_result(1)),
        _Future(running=False, value=_result(2)),
    ]
    dictionary = _Dictionary(futures, fail_at=2)
    with pytest.raises(RuntimeError, match="submit failed"):
        SpeculativeCpuBatch(dictionary, _requests(3), [4, 5, 6])
    assert len(dictionary.cpu.pool.submissions) == 2
    assert not futures[0].was_cancelled and futures[0].result_calls == 1
    assert futures[1].was_cancelled and futures[1].result_calls == 1
    assert futures[2].result_calls == 0
    assert dictionary.cpu._speculative_batch_active is None


@pytest.mark.parametrize(
    "requests, ids, message",
    [
        (_requests(2), [1], "One stable"),
        (_requests(2), [1, 1], "unique"),
        ([(np.array([1.]), [])], [1], "kwargs dictionary"),
    ],
)
def test_validation_finishes_before_any_dispatch(requests, ids, message):
    dictionary = _Dictionary([_Future(running=True, value=_result(0)) for _ in requests])
    with pytest.raises((TypeError, ValueError), match=message):
        SpeculativeCpuBatch(dictionary, requests, ids)
    assert dictionary.cpu.pool.submissions == []
    assert getattr(dictionary.cpu, "_speculative_batch_active", None) is None


def test_closed_service_and_overlapping_batch_are_rejected():
    dictionary = _Dictionary([_Future(running=True, value=_result(0))])
    dictionary.cpu.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        SpeculativeCpuBatch(dictionary, _requests(1), [1])
    dictionary.cpu.closed = False
    first = SpeculativeCpuBatch(dictionary, _requests(1), [1])
    with pytest.raises(RuntimeError, match="Only one"):
        SpeculativeCpuBatch(dictionary, _requests(1), [1])
    first.abort()


def test_invalid_gpu_mask_drains_without_cancelling_any_cpu_row():
    futures = [
        _Future(running=True, value=_result(0)),
        _Future(running=False, value=_result(1)),
    ]
    dictionary = _Dictionary(futures)
    batch = SpeculativeCpuBatch(dictionary, _requests(2), [7, 8])
    with pytest.raises(ValueError, match="one boolean"):
        batch.resolve(np.array([1, 0], dtype=np.int8))
    assert all(not future.was_cancelled for future in futures)
    assert all(future.result_calls == 1 for future in futures)
    assert batch.telemetry["mode"] == "aborted"
    assert batch.telemetry["outstanding_jobs"] == 0


class _RealDictionary:
    def __init__(self):
        self.cpu = RepeatedCpuLP(workers=1)

    def prepare_request(self, objective, kwargs, environment_id):
        return (objective, dict(kwargs)), {
            "dictionary_reason": "tiny_real_test",
            "environment_from_proposal": environment_id,
        }


def test_real_cpu_service_returns_exact_uncancelled_results_and_leaves_no_jobs():
    dictionary = _RealDictionary()
    requests = [
        ([-1.0], {"A_ub": [[1.0]], "b_ub": [limit], "bounds": [(0.0, None)]})
        for limit in (2.0, 3.0, 4.0)
    ]
    try:
        batch = SpeculativeCpuBatch(dictionary, requests, [101, 102, 103])
        results, telemetry = batch.resolve(np.array([True, False, False]))
        # The accepted row may be cancelled or may already be running.  Every
        # uncertified row must have its exact, original-LP CPU result.
        assert results[1].success and results[1].fun == pytest.approx(-3.0)
        assert results[2].success and results[2].fun == pytest.approx(-4.0)
        for index in (1, 2):
            assert results[index].diagnostics["certificate_passed"]
            assert results[index].diagnostics["dictionary_reason"] == "tiny_real_test"
        if results[0] is not None:
            assert results[0].success and results[0].fun == pytest.approx(-2.0)
        assert telemetry is batch.telemetry and telemetry["joined_all"]
        assert telemetry["outstanding_jobs"] == 0
        assert (telemetry["cpu_jobs_actual_started"]
                + telemetry["cpu_jobs_cancelled"] == 3)
        assert telemetry["cpu_jobs_with_observed_start"] == telemetry["started"]
        assert telemetry["actual_cpu_work_span_seconds"] >= 0.0
        assert telemetry["actual_cpu_completion_since_dispatch_seconds"] >= 0.0
        assert telemetry["summed_cpu_worker_seconds"] >= 0.0
        assert len(telemetry["cpu_execution_timing"]) == 3
    finally:
        dictionary.cpu.close()
