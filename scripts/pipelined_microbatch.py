"""Pipelined three-stage dFBA scheduling for matched CPU/GPU benchmarks.

The environment and every greenlet run on the calling Python main thread.
Only already-assembled, host-only LP requests enter the private executor, and
the executor invokes :meth:`RepeatedCpuLP._solve` directly.  ``maxmin`` remains
a full-environment barrier handled by ``service.solve_batch``; rejected GPU
stages are not silently accepted because ``aggregate`` and ``exchange`` are
always solved and certified by the exact CPU backend.

The synthetic asynchronous stage records deliberately set ``seconds`` to zero.
Their wall spans overlap each other and main-thread work, so summing them would
double count time.  Non-overlapping workflow intervals are recorded separately
in ``service.pipeline_history``.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
import time
from types import SimpleNamespace

import numpy as np
from greenlet import greenlet

from src.gpu_compiled_community_backend import stage_key


_PROOF_FIELDS = ("primal_residual", "dual_violation", "relative_kkt_gap")


def _request_stage(request, n_fluxes):
    """Classify a host LP without a third normalization/bounds-copy pass."""
    if not isinstance(request, (tuple, list)) or len(request) != 2:
        raise TypeError("A yielded LP request must be an (objective, kwargs) pair")
    objective, kwargs = request
    if not isinstance(kwargs, dict):
        raise TypeError("LP keyword arguments must be a dictionary")
    objective = np.asarray(objective, dtype=float)
    if objective.ndim != 1 or not len(objective):
        raise ValueError("LP objective must be a nonempty one-dimensional vector")

    def constraint_shape(name):
        value = kwargs.get(name)
        if value is None:
            return 0, len(objective)
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = np.shape(value)
        if len(shape) != 2:
            raise ValueError(f"{name} must be a two-dimensional constraint matrix")
        rows, columns = map(int, shape)
        if rows < 0 or columns != len(objective):
            raise ValueError(f"{name} columns must match the objective")
        return rows, columns

    eq_rows, columns = constraint_shape("A_eq")
    ub_rows, _ = constraint_shape("A_ub")
    structural_matrix = SimpleNamespace(shape=(eq_rows + ub_rows, columns))
    stage = stage_key(objective, structural_matrix, eq_rows, n_fluxes)[0]
    if stage not in {"maxmin", "aggregate", "exchange"}:
        raise RuntimeError("The pipeline supports exactly the original three LP stages")
    return stage


def _result_diagnostics(result):
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        raise RuntimeError("An exact CPU result requires mutable diagnostics")
    return diagnostics


def _proof_values(results, field):
    values = []
    for result in results:
        value = _result_diagnostics(result).get(field)
        values.append(float(value) if value is not None else None)
    return values


def _cpu_stage_record(stage, results, submitted, observed, cycle):
    """Create one ordered record without pretending overlapping spans add."""
    ids = list(range(len(results)))
    diagnostics = [dict(_result_diagnostics(result)) for result in results]
    first = min(submitted.values())
    last = max(observed.values())
    return dict(
        stage=stage,
        pipeline_cycle=cycle,
        batch=len(results),
        bank_accepts=0,
        candidate_engine="pipelined_exact_cpu",
        candidate_evaluations=0,
        candidate_count=[0] * len(results),
        observable_dispersion=[None] * len(results),
        preparation_seconds=0.0,
        bank_seconds=0.0,
        seconds=0.0,
        async_span_seconds=max(0.0, last - first),
        async_timing_scope=(
            "Submission-to-observation span; overlaps the other CPU stage and "
            "main-thread environment work. It must not be summed as wall time."
        ),
        cpu_lp_calls=len(results),
        cpu_solver_runs=sum(int(row.get("cpu_solver_runs", 1)) for row in diagnostics),
        numerical_retry_count=sum(int(row.get("numerical_retry_count", 0)) for row in diagnostics),
        routes=["cpu_pipeline_fallback"] * len(results),
        groups=[dict(ids=ids, route="cpu_fallback", rows=diagnostics)],
        accepted=[bool(result.success) for result in results],
        repair_policy="pipelined_exact_cpu_stage",
        repair_eligible=[False] * len(results),
        cpu_initial_basis_proposals=any(row.get("initial_basis_used", False) for row in diagnostics),
        **{field: _proof_values(results, field) for field in _PROOF_FIELDS},
    )


def _ensure_maxmin_history(service, before, results, elapsed, cycle):
    """Keep one maxmin record whether the service records itself or is a toy."""
    history = service.history
    added = len(history) - before
    if added == 0:
        diagnostics = [dict(_result_diagnostics(result)) for result in results]
        history.append(dict(
            stage="maxmin", pipeline_cycle=cycle, batch=len(results), bank_accepts=0,
            candidate_engine="service_without_native_history", candidate_evaluations=0,
            candidate_count=[0] * len(results), observable_dispersion=[None] * len(results),
            preparation_seconds=0.0, bank_seconds=0.0, seconds=elapsed,
            cpu_lp_calls=len(results), routes=["service_maxmin"] * len(results),
            groups=[dict(ids=list(range(len(results))), route="cpu_fallback", rows=diagnostics)],
            accepted=[bool(result.success) for result in results],
            **{field: _proof_values(results, field) for field in _PROOF_FIELDS}))
        return
    if added != 1:
        raise RuntimeError("A maxmin batch must append exactly one service history record")
    record = history[-1]
    if not isinstance(record, dict):
        raise RuntimeError("Service history records must be dictionaries")
    inferred = {row.get("stage") for row in record.get("rows", []) if isinstance(row, dict)}
    if record.get("stage") not in (None, "maxmin") or inferred - {"maxmin"}:
        raise RuntimeError("The full barrier produced a non-maxmin history record")
    # CpuDictionaryLP has a generic record; enrich it without replacing its
    # measured dictionary and exact-solver timings.
    diagnostics = [dict(_result_diagnostics(result)) for result in results]
    record.setdefault("stage", "maxmin")
    record.setdefault("pipeline_cycle", cycle)
    record.setdefault("bank_accepts", 0)
    record.setdefault("candidate_engine", "pipelined_exact_cpu")
    record.setdefault("candidate_evaluations", 0)
    record.setdefault("candidate_count", [0] * len(results))
    record.setdefault("observable_dispersion", [None] * len(results))
    record.setdefault("preparation_seconds", record.get("dictionary_seconds", 0.0))
    record.setdefault("bank_seconds", 0.0)
    record.setdefault("routes", ["cpu_pipeline_maxmin"] * len(results))
    record.setdefault("groups", [dict(ids=list(range(len(results))),
                                      route="cpu_fallback", rows=diagnostics)])
    record.setdefault("accepted", [bool(result.success) for result in results])
    for field in _PROOF_FIELDS:
        if field not in record:
            record[field] = _proof_values(results, field)


def drive_pipelined(environments, actions, service, snapshot, progress=None, workers=4):
    """Drive independent environments through a maxmin/CPU/CPU pipeline.

    Parameters follow :func:`scripts.microbatch_comparison_support.drive_microbatch`.
    Stable environment IDs are their positions in ``environments``.  Every
    environment has at most one asynchronous exact CPU request outstanding.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("dFBA greenlets and COBRA objects must remain on the main thread")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    environments = list(environments)
    action_rows = [tuple(row) for row in actions]
    count = len(environments)
    if len(action_rows) != count:
        raise ValueError("One action sequence is required per environment")
    if not count:
        if progress is not None and len(progress):
            raise ValueError("Progress length must match environments")
        return []
    steps = len(action_rows[0])
    if any(len(row) != steps for row in action_rows):
        raise ValueError("All environments must have equal action counts")
    if progress is not None and len(progress) != count:
        raise ValueError("Progress length must match environments")
    if not callable(snapshot):
        raise TypeError("snapshot must be callable")
    if not hasattr(service, "coordinates") or not hasattr(service.coordinates, "n_fluxes"):
        raise TypeError("service.coordinates.n_fluxes is required for stable stage identity")
    if not hasattr(service, "history"):
        service.history = []
    if not hasattr(service, "pipeline_history"):
        service.pipeline_history = []
    if not isinstance(service.history, list) or not isinstance(service.pipeline_history, list):
        raise TypeError("service history containers must be lists")
    # These are diagnostic-only copies for the benchmark failure-artifact
    # writer.  Reset them per drive so stale failures cannot be attributed to
    # a later run.
    service.pipeline_failed_requests = []
    service.pipeline_failed_diagnostics = []

    is_hybrid = hasattr(service, "direct_cpu")
    dictionary = service.direct_cpu if is_hybrid else service
    if dictionary is None or not callable(getattr(dictionary, "prepare_request", None)):
        raise RuntimeError("The pipeline requires a CPU dictionary request preparer")
    exact_cpu = getattr(dictionary, "cpu", None)
    if exact_cpu is None or not callable(getattr(exact_cpu, "_solve", None)):
        raise RuntimeError("The pipeline requires dictionary.cpu._solve")
    n_fluxes = int(service.coordinates.n_fluxes)

    if not steps:
        return [snapshot(env) for env, _ in environments]

    outputs = [None] * count
    main_resume_seconds = 0.0
    cpu_preparation_seconds = 0.0
    maxmin_service_seconds = 0.0
    cpu_wait_seconds = 0.0
    current_cycle = 0
    pending = {}
    cycle_started = time.perf_counter()

    def counters():
        return (main_resume_seconds, cpu_preparation_seconds,
                maxmin_service_seconds, cpu_wait_seconds)

    def resume(environment_id, value=None, *, initial=False):
        nonlocal main_resume_seconds
        before = time.perf_counter()
        yielded = greenlets[environment_id].switch() if initial else greenlets[environment_id].switch(value)
        main_resume_seconds += time.perf_counter() - before
        return yielded

    def worker(environment_id):
        env, model = environments[environment_id]
        for action in action_rows[environment_id]:
            env.step(action)
            status = model.stats.status
            if not isinstance(status, str) or not status.startswith("optimal;"):
                raise RuntimeError(status)
            if progress is not None:
                progress[environment_id] += 1
        return snapshot(env)

    greenlets = [greenlet(lambda environment_id=i: worker(environment_id)) for i in range(count)]
    maxmin_requests = {}
    # Count the initial host work that constructs the first maxmin requests in
    # cycle zero.  Later cycles start after the previous exchange has already
    # constructed their maxmin requests, so their counter baseline is refreshed
    # at the top of the loop.
    cycle_counter_start = counters()

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exact-lp-pipeline")
    failure_join_seconds = 0.0
    try:
        for environment_id in range(count):
            request = resume(environment_id, initial=True)
            if greenlets[environment_id].dead or _request_stage(request, n_fluxes) != "maxmin":
                raise RuntimeError("Every nonempty trajectory must first yield maxmin")
            maxmin_requests[environment_id] = request

        for current_cycle in range(steps):
            if current_cycle:
                cycle_counter_start = counters()
            ordered_maxmin = [maxmin_requests[i] for i in range(count)]
            before_history = len(service.history)
            before = time.perf_counter()
            # Both services assign default IDs positionally.  Because this is
            # always the full 0..B-1 cohort, omission gives the same stable IDs
            # and keeps the CPU/GPU call signature identical.
            maxmin_results = service.solve_batch(ordered_maxmin)
            maxmin_elapsed = time.perf_counter() - before
            maxmin_service_seconds += maxmin_elapsed
            if len(maxmin_results) != count or not all(result.success for result in maxmin_results):
                raise RuntimeError("The full maxmin batch was not certified")
            _ensure_maxmin_history(service, before_history, maxmin_results,
                                   maxmin_elapsed, current_cycle)

            stage_results = {"aggregate": {}, "exchange": {}}
            submitted = {"aggregate": {}, "exchange": {}}
            observed = {"aggregate": {}, "exchange": {}}
            next_maxmin = {}

            def submit_cpu(environment_id, stage, request):
                nonlocal cpu_preparation_seconds
                if stage not in {"aggregate", "exchange"}:
                    raise RuntimeError("Only aggregate/exchange enter the asynchronous CPU executor")
                before_prepare = time.perf_counter()
                objective, kwargs = request
                updated, metadata = dictionary.prepare_request(objective, kwargs, environment_id)
                cpu_preparation_seconds += time.perf_counter() - before_prepare
                metadata = dict(metadata)
                if metadata.get("stage", stage) != stage:
                    raise RuntimeError("CPU dictionary metadata changed the LP stage")
                submitted[stage][environment_id] = time.perf_counter()
                future = executor.submit(exact_cpu._solve, (environment_id, updated))
                pending[future] = dict(environment_id=environment_id, stage=stage,
                                       metadata=metadata, request=request)

            def accept_future(future):
                info = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    diagnostics = dict(
                        environment_id=info["environment_id"], stage=info["stage"],
                        success=False, cpu_lp_calls=1,
                        error_type=type(error).__name__, error=str(error))
                    diagnostics.update(info["metadata"])
                    service.pipeline_failed_requests = [info["request"]]
                    service.pipeline_failed_diagnostics = [diagnostics]
                    raise
                observed[info["stage"]][info["environment_id"]] = time.perf_counter()
                diagnostics = _result_diagnostics(result)
                diagnostics.update(info["metadata"])
                if not result.success:
                    service.pipeline_failed_requests = [info["request"]]
                    service.pipeline_failed_diagnostics = [dict(diagnostics)]
                    raise RuntimeError(
                        f"Exact CPU {info['stage']} failed for environment {info['environment_id']}")
                stage_results[info["stage"]][info["environment_id"]] = result
                request = resume(info["environment_id"], result)
                if info["stage"] == "aggregate":
                    if greenlets[info["environment_id"]].dead:
                        raise RuntimeError("Environment ended before exchange")
                    stage = _request_stage(request, n_fluxes)
                    if stage != "exchange":
                        raise RuntimeError("aggregate must be followed by exchange")
                    submit_cpu(info["environment_id"], stage, request)
                elif current_cycle + 1 < steps:
                    if greenlets[info["environment_id"]].dead:
                        raise RuntimeError("Environment ended before the next maxmin barrier")
                    if _request_stage(request, n_fluxes) != "maxmin":
                        raise RuntimeError("exchange must be followed by the next maxmin")
                    next_maxmin[info["environment_id"]] = request
                else:
                    if not greenlets[info["environment_id"]].dead:
                        raise RuntimeError("Environment yielded an unexpected LP after final exchange")
                    outputs[info["environment_id"]] = request

            def process_ready(block):
                nonlocal cpu_wait_seconds
                if not pending:
                    return False
                before_wait = time.perf_counter()
                done, _ = wait(tuple(pending), timeout=None if block else 0,
                               return_when=FIRST_COMPLETED)
                cpu_wait_seconds += time.perf_counter() - before_wait
                for future in sorted(done, key=lambda item: (
                        0 if pending[item]["stage"] == "aggregate" else 1,
                        pending[item]["environment_id"])):
                    accept_future(future)
                return bool(done)

            # Resume each environment as soon as its full-batch maxmin result
            # is available. A completed CPU job is serviced between resumes so
            # its exchange can enter the executor before all later work queues.
            for environment_id, result in enumerate(maxmin_results):
                request = resume(environment_id, result)
                if greenlets[environment_id].dead:
                    raise RuntimeError("Environment ended before aggregate")
                stage = _request_stage(request, n_fluxes)
                if stage != "aggregate":
                    raise RuntimeError("maxmin must be followed by aggregate")
                submit_cpu(environment_id, stage, request)
                while process_ready(False):
                    pass

            while pending:
                process_ready(True)
                while process_ready(False):
                    pass

            if any(len(stage_results[stage]) != count for stage in stage_results):
                raise RuntimeError("An asynchronous CPU stage lost an environment result")
            for stage in ("aggregate", "exchange"):
                ordered = [stage_results[stage][i] for i in range(count)]
                service.history.append(_cpu_stage_record(
                    stage, ordered, submitted[stage], observed[stage], current_cycle))

            if current_cycle + 1 < steps and len(next_maxmin) != count:
                raise RuntimeError("The next full maxmin barrier is incomplete")
            maxmin_requests = next_maxmin

            cycle_finished = time.perf_counter()
            current = counters()
            deltas = [value - old for value, old in zip(current, cycle_counter_start)]
            cycle_total = cycle_finished - cycle_started
            accounted = sum(deltas)
            service.pipeline_history.append(dict(
                cycle=current_cycle,
                batch=count,
                maxmin_service_seconds=deltas[2],
                main_thread_resume_seconds=deltas[0],
                cpu_preparation_seconds=deltas[1],
                cpu_wait_seconds=deltas[3],
                scheduler_overhead_seconds=max(0.0, cycle_total - accounted),
                total_seconds=cycle_total,
                async_span_seconds={stage:max(observed[stage].values()) - min(submitted[stage].values())
                                    for stage in ("aggregate", "exchange")},
                nonoverlap_accounting=(
                    "maxmin_service + main_thread_resume + cpu_preparation + cpu_wait + "
                    "scheduler_overhead equals cycle wall time; async spans overlap and are excluded."
                )))
            cycle_started = cycle_finished
        return outputs
    except BaseException as error:
        # Every submitted solve is joined, even when a peer solve, a greenlet,
        # or stage validation fails. Never leave a HiGHS model live in a worker.
        before_join = time.perf_counter()
        joined_errors = []
        for future in list(pending):
            try:
                future.result()
            except BaseException as joined_error:
                joined_errors.append(type(joined_error).__name__)
        failure_join_seconds = time.perf_counter() - before_join
        service.pipeline_history.append(dict(
            cycle=current_cycle, batch=count, failed=True,
            error_type=type(error).__name__, error=str(error),
            pending_joined=True, joined_worker_error_types=joined_errors,
            failure_join_seconds=failure_join_seconds,
            timing_scope="Failure diagnostic only; not a completed benchmark interval."))
        raise
    finally:
        executor.shutdown(wait=True)
