"""Validate and summarize completed maxmin-GPU/CPU-stage pipeline reports.

The module is deliberately read-only.  It does not repair incomplete reports
or infer missing diagnostics; a summary is emitted only after every configured
run passes the same structural, endpoint, timing, and original-LP checks.
"""

import argparse
import json
import math
from pathlib import Path


NONOVERLAP_FIELDS = (
    "maxmin_service_seconds",
    "main_thread_resume_seconds",
    "cpu_preparation_seconds",
    "cpu_wait_seconds",
    "scheduler_overhead_seconds",
)
ENDPOINT_FIELDS = ("pha_relative", "biomass_g_l", "phv_fraction")
CERTIFICATE_LIMITS = {
    "primal_residual": 1e-5,
    "dual_violation": 1e-7,
    "relative_kkt_gap": 1e-7,
}
STAGES = ("maxmin", "aggregate", "exchange")
COUNT_FIELDS = ("bank_accepts", "gpu_nonbank_accepts", "cpu_lp_calls",
                "cpu_solver_runs", "numerical_retry_count", "cpu_results_used",
                "cpu_speculative_unused", "cpu_speculative_cancelled",
                "cpu_speculative_unused_failures")


class PipelineSummaryError(ValueError):
    """The input cannot support a valid performance/accuracy summary."""


def _mapping(value, label):
    if not isinstance(value, dict):
        raise PipelineSummaryError(f"{label} must be an object")
    return value


def _sequence(value, label, length=None):
    if not isinstance(value, list):
        raise PipelineSummaryError(f"{label} must be an array")
    if length is not None and len(value) != length:
        raise PipelineSummaryError(f"{label} must contain {length} entries")
    return value


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PipelineSummaryError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineSummaryError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "finite and positive" if positive else "finite and nonnegative"
        raise PipelineSummaryError(f"{label} must be {qualifier}")
    return value


def _pipeline_blocks(history, *, side, steps, environments, block_size, wall):
    rows = _sequence(history, f"{side}_pipeline_history", steps)
    blocks = []
    current = None
    cycle_total = 0.0
    for cycle, row in enumerate(rows):
        row = _mapping(row, f"{side}_pipeline_history[{cycle}]")
        if row.get("failed") is True:
            raise PipelineSummaryError(f"{side} pipeline cycle {cycle} is marked failed")
        if _integer(row.get("cycle"), f"{side} cycle ID") != cycle:
            raise PipelineSummaryError(f"{side} pipeline cycle IDs are not sequential")
        if _integer(row.get("batch"), f"{side} cycle {cycle} batch", minimum=1) != environments:
            raise PipelineSummaryError(f"{side} cycle {cycle} has the wrong batch size")
        parts = {
            field: _number(row.get(field), f"{side} cycle {cycle} {field}")
            for field in NONOVERLAP_FIELDS
        }
        total = _number(row.get("total_seconds"), f"{side} cycle {cycle} total_seconds")
        if not math.isclose(sum(parts.values()), total, rel_tol=1e-6, abs_tol=1e-6):
            raise PipelineSummaryError(
                f"{side} cycle {cycle} nonoverlap timers do not sum to total_seconds")
        spans = _mapping(row.get("async_span_seconds"),
                         f"{side} cycle {cycle} async_span_seconds")
        for stage in ("aggregate", "exchange"):
            _number(spans.get(stage), f"{side} cycle {cycle} {stage} async span")

        block_index = cycle // block_size
        if current is None or current["block_index"] != block_index:
            current = dict(
                block_index=block_index,
                start_step=cycle + 1,
                end_step=cycle + 1,
                steps=0,
                total_seconds=0.0,
                **{field: 0.0 for field in NONOVERLAP_FIELDS},
            )
            blocks.append(current)
        current["end_step"] = cycle + 1
        current["steps"] += 1
        current["total_seconds"] += total
        for field, value in parts.items():
            current[field] += value
        cycle_total += total

    # The benchmark timer includes small setup/teardown intervals outside the
    # cycle accounting.  Permit only a small clock/accounting discrepancy.
    wall_tolerance = max(0.01, wall * 1e-3)
    if cycle_total > wall + wall_tolerance:
        raise PipelineSummaryError(
            f"{side} pipeline cycle totals exceed wall time beyond tolerance")
    return blocks, cycle_total, max(0.0, wall - cycle_total), wall_tolerance


def _record_rows(record):
    direct = record.get("rows")
    if isinstance(direct, list):
        return direct
    rows = []
    groups = record.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("rows"), list):
                rows.extend(group["rows"])
    return rows


def _cpu_rows(record):
    """Return executed exact-CPU diagnostics, including unused speculation."""
    direct = record.get("rows")
    if isinstance(direct, list):
        return direct
    rows = []
    groups = record.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if (isinstance(group, dict) and group.get("route") == "cpu_fallback"
                    and isinstance(group.get("rows"), list)):
                rows.extend(group["rows"])
    return rows


def _warning_diagnostic(value):
    """Retain unused-worker failures without emitting nonstandard JSON NaNs."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _warning_diagnostic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_warning_diagnostic(item) for item in value]
    return value


def _record_stage(record, label):
    stage = record.get("stage")
    if stage is None:
        inferred = {
            row.get("stage") for row in _record_rows(record)
            if isinstance(row, dict) and row.get("stage") is not None
        }
        if len(inferred) == 1:
            stage = inferred.pop()
    if stage not in STAGES:
        raise PipelineSummaryError(f"{label} has no unambiguous original3 stage")
    return stage


def _certificate_values(record, field, environments, label):
    values = record.get(field)
    if not isinstance(values, list):
        rows = _record_rows(record)
        if len(rows) == environments and all(isinstance(row, dict) for row in rows):
            values = [row.get(field) for row in rows]
    values = _sequence(values, f"{label} {field}", environments)
    checked = [_number(value, f"{label} {field}[{index}]")
               for index, value in enumerate(values)]
    limit = CERTIFICATE_LIMITS[field]
    if max(checked, default=0.0) > limit:
        raise PipelineSummaryError(f"{label} {field} exceeds {limit:g}")
    return checked


def _validate_lp_history(history, *, side, steps, environments):
    records = _sequence(history, f"{side}_history", steps * len(STAGES))
    maxima = {stage: {field: 0.0 for field in CERTIFICATE_LIMITS} for stage in STAGES}
    counts = {
        stage: {field: 0 for field in COUNT_FIELDS}
        for stage in STAGES
    }
    warnings = []
    for index, record in enumerate(records):
        label = f"{side}_history[{index}]"
        record = _mapping(record, label)
        expected = STAGES[index % len(STAGES)]
        stage = _record_stage(record, label)
        if stage != expected:
            raise PipelineSummaryError(
                f"{side}_history is not ordered maxmin/aggregate/exchange")
        if _integer(record.get("batch"), f"{label} batch", minimum=1) != environments:
            raise PipelineSummaryError(f"{label} has the wrong batch size")

        accepted = record.get("accepted")
        if isinstance(accepted, list):
            accepted = _sequence(accepted, f"{label} accepted", environments)
            if any(value is not True for value in accepted):
                raise PipelineSummaryError(f"{label} contains an unaccepted LP")
        else:
            rows = _record_rows(record)
            if len(rows) != environments or any(
                    not isinstance(row, dict) or row.get("success") is not True
                    for row in rows):
                raise PipelineSummaryError(f"{label} lacks complete successful LP diagnostics")

        for field in CERTIFICATE_LIMITS:
            values = _certificate_values(record, field, environments, label)
            maxima[stage][field] = max(maxima[stage][field], max(values, default=0.0))

        cpu = _integer(record.get("cpu_lp_calls"), f"{label} cpu_lp_calls")
        bank = (_integer(record.get("bank_accepts"), f"{label} bank_accepts")
                if side == "hybrid" else 0)
        speculative = ("cpu_speculative_unused" in record or "cpu_speculative_cancelled" in record
                       or str(record.get("candidate_engine", "")).startswith("speculative_"))
        unused = cancelled = 0
        used = cpu
        if speculative:
            if side != "hybrid" or stage != "maxmin":
                raise PipelineSummaryError(f"{label} speculation requires hybrid maxmin")
            used = _integer(record.get("cpu_results_used"), f"{label} cpu_results_used")
            unused = _integer(record.get("cpu_speculative_unused"), f"{label} cpu_speculative_unused")
            cancelled = _integer(record.get("cpu_speculative_cancelled"), f"{label} cpu_speculative_cancelled")
            # Actual work may overlap a GPU adoption. Do not mislabel every
            # executed CPU task as a fallback or count cancelled work as run.
            if (used + unused != cpu or cpu + cancelled != environments
                    or bank + cpu - unused != environments or bank != unused + cancelled):
                raise PipelineSummaryError(f"{label} speculative route counts are inconsistent with batch")
            gpu_nonbank = 0
            if _integer(record.get("gpu_nonbank_accepts", 0), f"{label} gpu_nonbank_accepts") != 0:
                raise PipelineSummaryError(f"{label} speculative maxmin does not support GPU repair")
            telemetry = record.get("speculative_cpu")
            if telemetry is not None:
                telemetry = _mapping(telemetry, f"{label} speculative_cpu")
                for name, expected_count in (("cpu_jobs_actual_started", cpu),
                                             ("cpu_jobs_cancelled", cancelled),
                                             ("cpu_results_used", used),
                                             ("unused_cpu_results_gpu_accepted", unused)):
                    if name in telemetry and _integer(telemetry[name], f"{label} {name}") != expected_count:
                        raise PipelineSummaryError(f"{label} speculative CPU telemetry disagrees with record")
                if telemetry.get("joined_all") is not True or telemetry.get("outstanding_jobs") != 0:
                    raise PipelineSummaryError(f"{label} speculative CPU workers were not all joined")
        else:
            if bank + cpu > environments or (side == "cpu" and cpu != environments):
                raise PipelineSummaryError(f"{label} route counts are inconsistent with batch")
            gpu_nonbank = environments - bank - cpu if side == "hybrid" else 0
        cpu_rows = _cpu_rows(record)
        if len(cpu_rows) != cpu:
            raise PipelineSummaryError(f"{label} has incomplete exact-CPU row diagnostics")
        row_lp_calls = 0
        solver_runs = 0
        retry_count = 0
        row_used = row_unused = unused_failures = 0
        for row_index, row in enumerate(cpu_rows):
            row = _mapping(row, f"{label} CPU row {row_index}")
            used_for_output = row.get("cpu_result_used") if speculative else True
            if not isinstance(used_for_output, bool):
                raise PipelineSummaryError(f"{label} speculative CPU row lacks cpu_result_used boolean")
            row_used += int(used_for_output)
            row_unused += int(not used_for_output)
            if row.get("success") is not True:
                if used_for_output or row.get("success") is not False:
                    raise PipelineSummaryError(f"{label} contains a failed exact-CPU row used for output")
                unused_failures += 1
                warnings.append(dict(kind="unused_speculative_cpu_failure", record=label,
                    stage=stage, cpu_row_index=row_index,
                    message="Unused CPU worker failed; the final output passed the separate original-LP certificate.",
                    diagnostic=_warning_diagnostic(row)))
            row_lp_calls += _integer(
                row.get("cpu_lp_calls"), f"{label} CPU row {row_index} cpu_lp_calls",
                minimum=1)
            solver_runs += _integer(
                row.get("cpu_solver_runs"), f"{label} CPU row {row_index} cpu_solver_runs")
            retry_count += _integer(
                row.get("numerical_retry_count"),
                f"{label} CPU row {row_index} numerical_retry_count")
        if row_lp_calls != cpu:
            raise PipelineSummaryError(f"{label} CPU row LP counts disagree with record")
        if row_used != used or row_unused != unused:
            raise PipelineSummaryError(f"{label} CPU result adoption counts disagree with rows")
        if "cpu_solver_runs" in record and _integer(
                record["cpu_solver_runs"], f"{label} cpu_solver_runs") != solver_runs:
            raise PipelineSummaryError(f"{label} CPU solver-run count disagrees with rows")
        if "numerical_retry_count" in record and _integer(
                record["numerical_retry_count"],
                f"{label} numerical_retry_count") != retry_count:
            raise PipelineSummaryError(f"{label} numerical-retry count disagrees with rows")
        counts[stage]["bank_accepts"] += bank
        counts[stage]["gpu_nonbank_accepts"] += gpu_nonbank
        counts[stage]["cpu_lp_calls"] += cpu
        counts[stage]["cpu_solver_runs"] += solver_runs
        counts[stage]["numerical_retry_count"] += retry_count
        counts[stage]["cpu_results_used"] += used
        counts[stage]["cpu_speculative_unused"] += unused
        counts[stage]["cpu_speculative_cancelled"] += cancelled
        counts[stage]["cpu_speculative_unused_failures"] += unused_failures
    return maxima, counts, warnings


def _endpoint_maxima(errors, cpu_rows, gpu_rows, environments, label):
    rows = _sequence(errors, f"{label} errors", environments)
    cpu_rows = _sequence(cpu_rows, f"{label}.cpu_rows", environments)
    gpu_rows = _sequence(gpu_rows, f"{label}.gpu_rows", environments)
    maxima = {field: 0.0 for field in ENDPOINT_FIELDS}
    for environment_id, (row, cpu, gpu) in enumerate(zip(rows, cpu_rows, gpu_rows)):
        row = _mapping(row, f"{label} errors[{environment_id}]")
        cpu = _mapping(cpu, f"{label}.cpu_rows[{environment_id}]")
        gpu = _mapping(gpu, f"{label}.gpu_rows[{environment_id}]")
        cpu_pha = _number(cpu.get("pha"), f"{label} CPU PHA")
        gpu_pha = _number(gpu.get("pha"), f"{label} hybrid PHA")
        cpu_phv = _number(cpu.get("phv_fraction"), f"{label} CPU PHV fraction")
        gpu_phv = _number(gpu.get("phv_fraction"), f"{label} hybrid PHV fraction")
        if cpu_phv > 1 or gpu_phv > 1:
            raise PipelineSummaryError(f"{label} PHV fraction must be <= 1")
        cpu_biomass = _mapping(cpu.get("biomass"), f"{label} CPU biomass")
        gpu_biomass = _mapping(gpu.get("biomass"), f"{label} hybrid biomass")
        if not cpu_biomass or set(cpu_biomass) != set(gpu_biomass):
            raise PipelineSummaryError(f"{label} endpoint biomass species do not match")
        biomass_error = 0.0
        for species in cpu_biomass:
            cpu_value = _number(cpu_biomass[species], f"{label} CPU biomass {species}")
            gpu_value = _number(gpu_biomass[species], f"{label} hybrid biomass {species}")
            biomass_error = max(biomass_error, abs(cpu_value - gpu_value))
        recomputed = dict(
            pha_relative=abs(cpu_pha - gpu_pha) / max(abs(cpu_pha), 1e-9),
            biomass_g_l=biomass_error,
            phv_fraction=abs(cpu_phv - gpu_phv),
        )
        for field in ENDPOINT_FIELDS:
            value = _number(row.get(field), f"{label} errors[{environment_id}].{field}")
            if value > 0.01:
                raise PipelineSummaryError(
                    f"{label} environment {environment_id} {field} exceeds 0.01")
            if not math.isclose(value, recomputed[field], rel_tol=1e-9, abs_tol=1e-12):
                raise PipelineSummaryError(
                    f"{label} environment {environment_id} reported {field} disagrees with endpoints")
            maxima[field] = max(maxima[field], value)
    return maxima


def summarize_pipeline_report(report, *, block_size=20):
    """Return a strict, JSON-serializable summary of one parsed report."""
    report = _mapping(report, "report")
    block_size = _integer(block_size, "block_size", minimum=1)
    if report.get("status") != "completed":
        raise PipelineSummaryError("report status must be completed")
    configuration = _mapping(report.get("configuration"), "configuration")
    required = {
        "pipeline_cpu_stages": True,
        "hybrid": True,
        "cpu_backend": "dictionary",
        "tie_policy": "original3",
        "candidate_oracle": False,
    }
    for field, expected in required.items():
        if configuration.get(field) != expected:
            raise PipelineSummaryError(f"configuration.{field} must be {expected!r}")
    if not isinstance(configuration.get("speculative_cpu", False), bool):
        raise PipelineSummaryError("configuration.speculative_cpu must be boolean")
    gpu_stages = _sequence(configuration.get("gpu_stages"), "configuration.gpu_stages")
    if gpu_stages != ["maxmin"]:
        raise PipelineSummaryError("pipeline summary requires maxmin-only GPU stages")
    steps = _integer(configuration.get("steps"), "configuration.steps", minimum=1)
    environments = _integer(
        configuration.get("environments"), "configuration.environments", minimum=1)
    repeats = _integer(configuration.get("repeats"), "configuration.repeats", minimum=1)
    runs = _sequence(report.get("runs"), "runs", repeats)

    summaries = []
    overall_walls = {"cpu": 0.0, "hybrid": 0.0}
    overall_errors = {field: 0.0 for field in ENDPOINT_FIELDS}
    overall_certificates = {
        side: {stage: {field: 0.0 for field in CERTIFICATE_LIMITS} for stage in STAGES}
        for side in ("cpu", "hybrid")
    }
    overall_gpu_counts = {
        stage: {field: 0 for field in COUNT_FIELDS} for stage in STAGES
    }
    maximum_wall_tolerance = 0.0
    overall_warnings = []

    for repeat, run in enumerate(runs):
        label = f"run[{repeat}]"
        run = _mapping(run, label)
        if _integer(run.get("repeat"), f"{label}.repeat") != repeat:
            raise PipelineSummaryError("run repeat IDs are not sequential")
        if "failure" not in run or run["failure"] is not None:
            raise PipelineSummaryError(f"{label} contains or omits a failure diagnostic")
        if run.get("all_endpoint_gates_passed") is not True:
            raise PipelineSummaryError(f"{label} endpoint gate did not pass")
        execution_order = run.get("execution_order")
        if execution_order not in {"cpu-first", "gpu-first"}:
            raise PipelineSummaryError(f"{label}.execution_order is invalid")
        if not isinstance(run.get("cold_first_use_included"), bool):
            raise PipelineSummaryError(f"{label}.cold_first_use_included must be boolean")
        _sequence(run.get("seeds"), f"{label}.seeds", environments)
        for side in ("cpu", "gpu"):
            completed = _sequence(
                run.get(f"{side}_completed_steps"),
                f"{label}.{side}_completed_steps", environments)
            if any(_integer(value, f"{label}.{side}_completed_steps") != steps
                   for value in completed):
                raise PipelineSummaryError(f"{label} {side} did not complete every step")
            _sequence(run.get(f"{side}_rows"), f"{label}.{side}_rows", environments)

        cpu_wall = _number(run.get("cpu_seconds"), f"{label}.cpu_seconds", positive=True)
        hybrid_wall = _number(run.get("gpu_seconds"), f"{label}.gpu_seconds", positive=True)
        ratio = cpu_wall / hybrid_wall
        reported_ratio = _number(
            run.get("cpu_over_gpu_ratio"), f"{label}.cpu_over_gpu_ratio", positive=True)
        if not math.isclose(ratio, reported_ratio, rel_tol=1e-9, abs_tol=1e-12):
            raise PipelineSummaryError(f"{label} reported speed ratio is inconsistent")

        cpu_blocks, cpu_cycles, cpu_outside, cpu_tol = _pipeline_blocks(
            run.get("cpu_pipeline_history"), side=f"{label}.cpu", steps=steps,
            environments=environments, block_size=block_size, wall=cpu_wall)
        hybrid_blocks, hybrid_cycles, hybrid_outside, hybrid_tol = _pipeline_blocks(
            run.get("gpu_pipeline_history"), side=f"{label}.hybrid", steps=steps,
            environments=environments, block_size=block_size, wall=hybrid_wall)
        maximum_wall_tolerance = max(maximum_wall_tolerance, cpu_tol, hybrid_tol)

        cpu_certificates, cpu_counts, cpu_warnings = _validate_lp_history(
            run.get("cpu_history"), side="cpu", steps=steps, environments=environments)
        hybrid_certificates, gpu_counts, hybrid_warnings = _validate_lp_history(
            run.get("gpu_history"), side="hybrid", steps=steps,
            environments=environments)
        run_warnings = [dict(warning, repeat=repeat) for warning in cpu_warnings + hybrid_warnings]
        overall_warnings.extend(run_warnings)
        reported_cpu_calls = _integer(
            run.get("online_cpu_lp_calls"), f"{label}.online_cpu_lp_calls")
        history_cpu_calls = sum(row["cpu_lp_calls"] for row in gpu_counts.values())
        if reported_cpu_calls != history_cpu_calls:
            raise PipelineSummaryError(
                f"{label} online_cpu_lp_calls disagrees with gpu_history")
        reported_cpu_solver_runs = _integer(
            run.get("cpu_solver_runs"), f"{label}.cpu_solver_runs")
        cpu_history_solver_runs = sum(
            row["cpu_solver_runs"] for row in cpu_counts.values())
        if reported_cpu_solver_runs != cpu_history_solver_runs:
            raise PipelineSummaryError(f"{label} cpu_solver_runs disagrees with cpu_history")
        reported_online_solver_runs = _integer(
            run.get("online_cpu_solver_runs"), f"{label}.online_cpu_solver_runs")
        gpu_history_solver_runs = sum(
            row["cpu_solver_runs"] for row in gpu_counts.values())
        if reported_online_solver_runs != gpu_history_solver_runs:
            raise PipelineSummaryError(
                f"{label} online_cpu_solver_runs disagrees with gpu_history")
        endpoint_maxima = _endpoint_maxima(
            run.get("errors"), run.get("cpu_rows"), run.get("gpu_rows"),
            environments, label)

        for field, value in endpoint_maxima.items():
            overall_errors[field] = max(overall_errors[field], value)
        for side, source in (("cpu", cpu_certificates),
                             ("hybrid", hybrid_certificates)):
            for stage in STAGES:
                for field, value in source[stage].items():
                    overall_certificates[side][stage][field] = max(
                        overall_certificates[side][stage][field], value)
        for stage in STAGES:
            for field in COUNT_FIELDS:
                overall_gpu_counts[stage][field] += gpu_counts[stage][field]
        overall_walls["cpu"] += cpu_wall
        overall_walls["hybrid"] += hybrid_wall

        summaries.append(dict(
            repeat=repeat,
            execution_order=execution_order,
            cold_first_use_included=run["cold_first_use_included"],
            wall_seconds=dict(cpu=cpu_wall, hybrid=hybrid_wall),
            cpu_over_hybrid_ratio=ratio,
            endpoint_error_maxima=endpoint_maxima,
            step_blocks={"cpu": cpu_blocks, "hybrid": hybrid_blocks},
            pipeline_cycle_seconds={"cpu": cpu_cycles, "hybrid": hybrid_cycles},
            wall_outside_pipeline_cycles_seconds={
                "cpu": cpu_outside, "hybrid": hybrid_outside},
            cpu_reference_history_counts=cpu_counts,
            gpu_history_counts=gpu_counts,
            original_lp_certificate_maxima={
                "cpu": cpu_certificates, "hybrid": hybrid_certificates},
            warnings=run_warnings,
        ))

    overall_cpu_counts = {
        stage: {
            field: sum(run["cpu_reference_history_counts"][stage][field]
                       for run in summaries)
            for field in COUNT_FIELDS
        }
        for stage in STAGES
    }
    return dict(
        status="validated_completed_pipeline",
        configuration=dict(
            steps=steps, environments=environments, repeats=repeats,
            block_size_steps=block_size, cpu_backend="dictionary",
            gpu_stages=["maxmin"], tie_policy="original3",
            speculative_cpu=bool(configuration.get("speculative_cpu", False))),
        count_semantics=dict(cpu_lp_calls="Actual started CPU LP requests, including unused speculative work",
            cpu_results_used="Final outputs adopted from CPU",
            cpu_speculative_unused="Started CPU requests superseded by independently certified GPU output",
            cpu_speculative_cancelled="Queued CPU jobs cancelled before execution",
            no_repair_identity="bank_accepts + cpu_lp_calls - cpu_speculative_unused == batch",
            unused_failure_policy="Preserved as warnings; never substituted for final output certificates"),
        validation_limits=dict(
            endpoint_error_maximum=0.01,
            original_lp_certificate=CERTIFICATE_LIMITS,
            cycle_accounting_relative_tolerance=1e-6,
            cycle_accounting_absolute_tolerance_seconds=1e-6,
            maximum_applied_cycle_sum_vs_wall_tolerance_seconds=maximum_wall_tolerance),
        runs=summaries,
        overall=dict(
            wall_seconds=overall_walls,
            cpu_over_hybrid_ratio=(overall_walls["cpu"] / overall_walls["hybrid"]),
            cpu_over_hybrid_ratio_scope=(
                "Descriptive ratio of summed validated wall times only; no confidence "
                "interval or general performance superiority is claimed."),
            endpoint_error_maxima=overall_errors,
            cpu_reference_history_counts=overall_cpu_counts,
            cpu_reference_total_lp_calls=sum(
                row["cpu_lp_calls"] for row in overall_cpu_counts.values()),
            cpu_reference_total_solver_runs=sum(
                row["cpu_solver_runs"] for row in overall_cpu_counts.values()),
            cpu_reference_total_numerical_retries=sum(
                row["numerical_retry_count"] for row in overall_cpu_counts.values()),
            gpu_history_counts=overall_gpu_counts,
            gpu_maxmin_bank_accepts=overall_gpu_counts["maxmin"]["bank_accepts"],
            gpu_maxmin_nonbank_accepts=overall_gpu_counts["maxmin"][
                "gpu_nonbank_accepts"],
            gpu_maxmin_cpu_lp_calls=overall_gpu_counts["maxmin"]["cpu_lp_calls"],
            gpu_total_cpu_lp_calls=sum(
                row["cpu_lp_calls"] for row in overall_gpu_counts.values()),
            gpu_total_cpu_results_used=sum(row["cpu_results_used"] for row in overall_gpu_counts.values()),
            gpu_total_cpu_speculative_unused=sum(row["cpu_speculative_unused"] for row in overall_gpu_counts.values()),
            gpu_total_cpu_speculative_cancelled=sum(row["cpu_speculative_cancelled"] for row in overall_gpu_counts.values()),
            gpu_total_cpu_speculative_unused_failures=sum(row["cpu_speculative_unused_failures"] for row in overall_gpu_counts.values()),
            gpu_total_cpu_solver_runs=sum(
                row["cpu_solver_runs"] for row in overall_gpu_counts.values()),
            gpu_total_numerical_retries=sum(
                row["numerical_retry_count"] for row in overall_gpu_counts.values()),
            original_lp_certificate_maxima=overall_certificates,
            warnings=overall_warnings,
        ),
    )


def summarize_pipeline_file(path, *, block_size=20):
    """Read *path* without modifying it and return its validated summary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return summarize_pipeline_report(report, block_size=block_size)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Strictly validate and summarize a completed pipeline benchmark JSON")
    parser.add_argument("report", type=Path)
    parser.add_argument("--block-size", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        summary = summarize_pipeline_file(args.report, block_size=args.block_size)
    except (OSError, json.JSONDecodeError, PipelineSummaryError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
