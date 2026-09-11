"""Cancellable exact-CPU batch launched alongside a certified GPU probe.

The GPU caller remains responsible for solving and certifying its proposals.
This helper only overlaps the *same* dictionary-initialized CPU requests with
that work.  A CPU job may be cancelled solely after the caller reports that
the corresponding original LP has passed the GPU certificate.  Jobs which
are already running are always joined, so no HiGHS worker survives a stage.

The class is intentionally single use.  Construct, run the GPU probe on the
calling thread, then call :meth:`resolve`; call :meth:`abort` if the GPU probe
raises.  ``results[i] is None`` means that the pending CPU job was cancelled,
never that an unverified LP was accepted.
"""

from concurrent.futures import CancelledError
import threading
import time

import numpy as np


class SpeculativeCpuBatch:
    """Submit one stable batch to a ``CpuDictionaryLP`` CPU service.

    Request preparation is completed on the main thread before the first job
    is dispatched.  This preserves the existing boundary between host LP
    preparation and the worker-only :class:`RepeatedCpuLP` solve.
    """

    def __init__(self, cpu_dictionary, requests, environment_ids):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Speculative CPU batch preparation requires the main thread")
        if not callable(getattr(cpu_dictionary, "prepare_request", None)):
            raise TypeError("cpu_dictionary.prepare_request is required")
        cpu = getattr(cpu_dictionary, "cpu", None)
        if cpu is None or not callable(getattr(cpu, "_solve", None)):
            raise TypeError("cpu_dictionary.cpu._solve is required")
        if getattr(cpu, "closed", True):
            raise RuntimeError("The CPU dictionary backend is closed")
        pool = getattr(cpu, "pool", None)
        if pool is None or not callable(getattr(pool, "submit", None)):
            raise TypeError("cpu_dictionary.cpu.pool.submit is required")

        self.cpu_dictionary = cpu_dictionary
        self.cpu = cpu
        self.requests = list(requests)
        self.environment_ids = list(environment_ids)
        if len(self.environment_ids) != len(self.requests):
            raise ValueError("One stable environment ID is required per request")
        try:
            unique = len(set(self.environment_ids))
        except TypeError as error:
            raise ValueError("Stable environment IDs must be hashable and unique") from error
        if unique != len(self.environment_ids):
            raise ValueError("Stable environment IDs must be unique")
        if getattr(cpu, "_speculative_batch_active", None) is not None:
            raise RuntimeError("Only one speculative batch may use a CPU service at a time")

        self._token = object()
        cpu._speculative_batch_active = self._token
        self._finished = False
        self._futures = []
        self._prepared = []
        self._metadata = []
        self._cpu_started_at = [None] * len(self.requests)
        self._cpu_finished_at = [None] * len(self.requests)
        self.results = None
        self.history = []
        self.telemetry = None
        self._created_at = time.perf_counter()

        proposal_started = time.perf_counter()
        try:
            # Complete validation/proposal selection before dispatch.  If one
            # row is malformed, no worker can observe a partially prepared
            # batch or mutate a persistent HiGHS model.
            for environment_id, request in zip(self.environment_ids, self.requests):
                if not isinstance(request, (tuple, list)) or len(request) != 2:
                    raise TypeError("Each CPU request must be an (objective, kwargs) pair")
                objective, kwargs = request
                if not isinstance(kwargs, dict):
                    raise TypeError("Each CPU request requires a kwargs dictionary")
                prepared, metadata = cpu_dictionary.prepare_request(
                    objective, kwargs, environment_id
                )
                if not isinstance(metadata, dict):
                    raise TypeError("CPU dictionary metadata must be a dictionary")
                self._prepared.append(prepared)
                self._metadata.append(dict(metadata))
        except BaseException:
            self._release_service()
            raise
        self._proposal_seconds = time.perf_counter() - proposal_started

        self._dispatch_started = time.perf_counter()
        try:
            for index, (environment_id, prepared) in enumerate(
                    zip(self.environment_ids, self._prepared)):
                self._futures.append(
                    pool.submit(self._run_cpu, index, (environment_id, prepared))
                )
        except BaseException as submit_error:
            # A partially submitted batch cannot be returned.  Cancel whatever
            # has not started and drain every other future before releasing the
            # shared persistent service.  Cleanup errors never prevent later
            # futures from being joined.
            cleanup_errors = self._cancel_and_drain_after_submit_failure()
            self._release_service()
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "CPU batch submission and cleanup failed",
                    [submit_error, *cleanup_errors],
                )
            raise
        self._dispatch_finished = time.perf_counter()

    def _run_cpu(self, index, job):
        """Record one worker's actual execution interval, even on failure."""

        self._cpu_started_at[index] = time.perf_counter()
        try:
            return self.cpu._solve(job)
        finally:
            self._cpu_finished_at[index] = time.perf_counter()

    def _release_service(self):
        if getattr(self.cpu, "_speculative_batch_active", None) is self._token:
            self.cpu._speculative_batch_active = None

    def _cancel_and_drain_after_submit_failure(self):
        errors = []
        for future in self._futures:
            future.cancel()
        for future in self._futures:
            try:
                future.result()
            except CancelledError:
                pass
            except BaseException as error:
                errors.append(error)
        return errors

    def _check_open(self):
        if self._finished:
            raise RuntimeError("A speculative CPU batch is single use")

    def resolve(self, gpu_accepted):
        """Cancel only certified pending rows, then join every CPU future.

        Returns ``(results, telemetry)``.  The result list has the original
        batch order and a cancelled row is represented by ``None``.  A
        GPU-accepted row whose CPU job was already running contains its
        now-unused exact CPU result.
        """

        self._check_open()
        accepted = np.asarray(gpu_accepted)
        if accepted.shape != (len(self._futures),) or accepted.dtype.kind != "b":
            validation_error = ValueError(
                "gpu_accepted must be one boolean per CPU request"
            )
            try:
                self.abort()
            except BaseException as drain_error:
                raise BaseExceptionGroup(
                    "GPU acceptance validation and CPU drain failed",
                    [validation_error, drain_error],
                )
            raise validation_error
        return self._finish(accepted.copy(), cancel_certified=True, mode="resolved")

    def abort(self):
        """GPU-failure path: cancel nothing and drain the complete CPU batch."""

        self._check_open()
        accepted = np.zeros(len(self._futures), dtype=bool)
        return self._finish(accepted, cancel_certified=False, mode="aborted")

    def drain(self):
        """Alias for :meth:`abort`, emphasizing that every job is joined."""

        return self.abort()

    def _finish(self, accepted, *, cancel_certified, mode):
        wait_started = time.perf_counter()
        cancelled = np.zeros(len(self._futures), dtype=bool)
        if cancel_certified:
            # Calling Future.cancel only for certified rows is the central
            # safety invariant.  A false return means running/done, and that
            # worker is joined below rather than interrupted.
            for index in np.flatnonzero(accepted):
                cancelled[index] = bool(self._futures[index].cancel())

        results = [None] * len(self._futures)
        errors = []
        for index, (future, metadata) in enumerate(zip(self._futures, self._metadata)):
            try:
                result = future.result()
            except CancelledError:
                if not cancelled[index]:
                    errors.append(RuntimeError("An uncertified CPU job was cancelled"))
                continue
            except BaseException as error:
                errors.append(error)
                continue
            diagnostics = getattr(result, "diagnostics", None)
            if not isinstance(diagnostics, dict):
                errors.append(RuntimeError("An exact CPU result requires mutable diagnostics"))
                continue
            diagnostics.update(metadata)
            results[index] = result

        finished_at = time.perf_counter()
        failed = len(errors)
        actual_started = len(self._futures) - int(cancelled.sum())
        completed = sum(
            not future.cancelled() and future.done() for future in self._futures
        )
        cpu_solver_runs = sum(
            int(result.diagnostics.get("cpu_solver_runs", 1))
            for result in results
            if result is not None
        )
        unused = sum(
            bool(accepted[index]) and result is not None
            for index, result in enumerate(results)
        )
        execution_rows = []
        for index, (worker_started, worker_finished) in enumerate(
                zip(self._cpu_started_at, self._cpu_finished_at)):
            execution_rows.append(dict(
                environment_id=self.environment_ids[index],
                started_since_dispatch_seconds=(None if worker_started is None else
                    float(worker_started - self._dispatch_started)),
                finished_since_dispatch_seconds=(None if worker_finished is None else
                    float(worker_finished - self._dispatch_started)),
                worker_seconds=(None if worker_started is None or worker_finished is None else
                    float(worker_finished - worker_started)),
                cancelled=bool(cancelled[index]),
            ))
        starts = [value for value in self._cpu_started_at if value is not None]
        finishes = [value for value in self._cpu_finished_at if value is not None]
        timed_started = len(starts)
        if starts and len(finishes) == len(starts):
            actual_work_span = max(finishes) - min(starts)
            actual_completion = max(finishes) - self._dispatch_started
            summed_worker = sum(
                finished - worker_started
                for worker_started, finished in zip(
                    self._cpu_started_at, self._cpu_finished_at
                )
                if worker_started is not None and finished is not None
            )
        elif not actual_started:
            actual_work_span = actual_completion = summed_worker = 0.0
        else:
            # This is reachable only for a nonstandard executor which does not
            # invoke the submitted wrapper (for example a unit-test double).
            actual_work_span = actual_completion = summed_worker = None
        record = dict(
            kind="speculative_exact_cpu_batch",
            mode=mode,
            batch=len(self._futures),
            environment_ids=list(self.environment_ids),
            gpu_certified_rows=int(accepted.sum()),
            proposal_seconds=float(self._proposal_seconds),
            dispatch_seconds=float(self._dispatch_finished - self._dispatch_started),
            wait_seconds=float(finished_at - wait_started),
            cpu_batch_span_seconds=float(finished_at - self._dispatch_started),
            cpu_batch_span_scope=(
                "Dispatch-to-join observation; includes time spent running the "
                "concurrent GPU probe after CPU work may already have finished."
            ),
            actual_cpu_work_span_seconds=(None if actual_work_span is None else
                float(actual_work_span)),
            actual_cpu_completion_since_dispatch_seconds=(
                None if actual_completion is None else float(actual_completion)
            ),
            summed_cpu_worker_seconds=(None if summed_worker is None else
                float(summed_worker)),
            cpu_execution_timing=execution_rows,
            cpu_jobs_with_observed_start=timed_started,
            total_seconds=float(finished_at - self._created_at),
            cpu_jobs_submitted=len(self._futures),
            cpu_jobs_actual_started=actual_started,
            cpu_jobs_cancelled=int(cancelled.sum()),
            cpu_jobs_completed=int(completed),
            cpu_jobs_failed=failed,
            cpu_solver_runs=cpu_solver_runs,
            cpu_results_used=int(sum((not accepted[index]) and result is not None
                                     for index, result in enumerate(results))),
            unused_cpu_results_gpu_accepted=int(unused),
            cancelled_environment_ids=[
                self.environment_ids[index] for index in np.flatnonzero(cancelled)
            ],
            joined_all=all(future.done() for future in self._futures),
            outstanding_jobs=sum(not future.done() for future in self._futures),
            rows=[None if result is None else dict(result.diagnostics)
                  for result in results],
        )
        # Short aliases keep the integration record compact while the explicit
        # names above document precisely what each counter means.
        record.update(
            submitted=record["cpu_jobs_submitted"],
            started=record["cpu_jobs_actual_started"],
            cancelled=record["cpu_jobs_cancelled"],
            completed=record["cpu_jobs_completed"],
            unused_completed=record["unused_cpu_results_gpu_accepted"],
            submission_seconds=record["dispatch_seconds"],
            observed_cpu_batch_span_seconds=record["cpu_batch_span_seconds"],
        )
        self.results = results
        self.telemetry = record
        self.history.append(record)
        self._finished = True
        self._release_service()
        if errors:
            raise BaseExceptionGroup("Exact CPU speculative jobs failed", errors)
        return results, record
