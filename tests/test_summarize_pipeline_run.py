import copy
import json

import pytest

from scripts.summarize_pipeline_run import (
    PipelineSummaryError,
    main,
    summarize_pipeline_report,
)


def lp_record(stage, environments, *, hybrid):
    bank = 1 if hybrid and stage == "maxmin" else 0
    cpu = environments - bank
    cpu_rows = [dict(
        stage=stage, success=True, cpu_lp_calls=1, cpu_solver_runs=1,
        numerical_retry_count=0, primal_residual=1e-8,
        dual_violation=1e-10, relative_kkt_gap=1e-9,
    ) for _ in range(cpu)]
    record = dict(
        stage=stage,
        batch=environments,
        accepted=[True] * environments,
        primal_residual=[1e-8] * environments,
        dual_violation=[1e-10] * environments,
        relative_kkt_gap=[1e-9] * environments,
        cpu_lp_calls=cpu,
        groups=[dict(route="cpu_fallback", rows=cpu_rows)] if cpu_rows else [],
    )
    if hybrid:
        record.update(bank_accepts=bank)
    else:
        record["rows"] = cpu_rows
        record.pop("groups")
    return record


def pipeline_cycles(steps, environments, scale):
    rows = []
    for cycle in range(steps):
        fields = dict(
            maxmin_service_seconds=0.10 * scale,
            main_thread_resume_seconds=0.20 * scale,
            cpu_preparation_seconds=0.05 * scale,
            cpu_wait_seconds=0.14 * scale,
            scheduler_overhead_seconds=0.01 * scale,
        )
        rows.append(dict(
            cycle=cycle, batch=environments, **fields,
            total_seconds=sum(fields.values()),
            async_span_seconds={"aggregate": 0.2 * scale, "exchange": 0.3 * scale},
        ))
    return rows


def complete_report(steps=21, environments=2):
    cpu_pipeline = pipeline_cycles(steps, environments, 1.0)
    gpu_pipeline = pipeline_cycles(steps, environments, 0.5)
    cpu_wall = sum(row["total_seconds"] for row in cpu_pipeline) + 0.1
    gpu_wall = sum(row["total_seconds"] for row in gpu_pipeline) + 0.1
    cpu_rows = [dict(pha=2.0, phv_fraction=0.5,
                     biomass={"species_a": 1.0, "species_b": 2.0})
                for _ in range(environments)]
    gpu_rows = [dict(pha=1.998, phv_fraction=0.497,
                     biomass={"species_a": 0.998, "species_b": 2.0})
                for _ in range(environments)]
    run = dict(
        repeat=0,
        seeds=list(range(environments)),
        execution_order="cpu-first",
        cold_first_use_included=True,
        failure=None,
        all_endpoint_gates_passed=True,
        cpu_completed_steps=[steps] * environments,
        gpu_completed_steps=[steps] * environments,
        cpu_rows=cpu_rows,
        gpu_rows=gpu_rows,
        cpu_seconds=cpu_wall,
        gpu_seconds=gpu_wall,
        cpu_over_gpu_ratio=cpu_wall / gpu_wall,
        cpu_pipeline_history=cpu_pipeline,
        gpu_pipeline_history=gpu_pipeline,
        cpu_history=[lp_record(stage, environments, hybrid=False)
                     for _ in range(steps) for stage in ("maxmin", "aggregate", "exchange")],
        gpu_history=[lp_record(stage, environments, hybrid=True)
                     for _ in range(steps) for stage in ("maxmin", "aggregate", "exchange")],
        errors=[dict(pha_relative=0.001, biomass_g_l=0.002, phv_fraction=0.003)
                for _ in range(environments)],
    )
    run["online_cpu_lp_calls"] = sum(row["cpu_lp_calls"] for row in run["gpu_history"])
    run["cpu_solver_runs"] = sum(
        row["cpu_lp_calls"] for row in run["cpu_history"])
    run["online_cpu_solver_runs"] = run["online_cpu_lp_calls"]
    return dict(
        status="completed",
        configuration=dict(
            pipeline_cpu_stages=True, hybrid=True, cpu_backend="dictionary",
            tie_policy="original3", candidate_oracle=False, gpu_stages=["maxmin"],
            steps=steps, environments=environments, repeats=1),
        runs=[run],
    )


def test_summary_validates_and_groups_twenty_step_blocks():
    report = complete_report()
    summary = summarize_pipeline_report(report)

    assert summary["status"] == "validated_completed_pipeline"
    run = summary["runs"][0]
    assert [(row["start_step"], row["end_step"], row["steps"])
            for row in run["step_blocks"]["cpu"]] == [(1, 20, 20), (21, 21, 1)]
    assert run["step_blocks"]["cpu"][0]["total_seconds"] == pytest.approx(10.0)
    assert run["step_blocks"]["hybrid"][0]["cpu_wait_seconds"] == pytest.approx(1.4)
    assert summary["overall"]["cpu_over_hybrid_ratio"] == pytest.approx(
        report["runs"][0]["cpu_seconds"] / report["runs"][0]["gpu_seconds"])
    assert summary["overall"]["gpu_maxmin_bank_accepts"] == 21
    assert summary["overall"]["gpu_maxmin_cpu_lp_calls"] == 21
    assert summary["overall"]["gpu_total_cpu_lp_calls"] == 105
    cpu_counts = summary["overall"]["cpu_reference_history_counts"]
    assert sum(row["cpu_lp_calls"] for row in cpu_counts.values()) == 126
    assert sum(row["cpu_solver_runs"] for row in cpu_counts.values()) == 126
    assert summary["overall"]["cpu_reference_total_lp_calls"] == 126
    assert summary["overall"]["gpu_total_cpu_solver_runs"] == 105
    assert summary["overall"]["gpu_total_numerical_retries"] == 0
    assert "no confidence interval" in summary["overall"]["cpu_over_hybrid_ratio_scope"]
    assert summary["overall"]["original_lp_certificate_maxima"]["hybrid"][
        "exchange"]["relative_kkt_gap"] == pytest.approx(1e-9)


@pytest.mark.parametrize("mutation, match", [
    (lambda report: report.update(status="benchmark"), "status must be completed"),
    (lambda report: report["configuration"].update(cpu_backend="persistent"),
     "cpu_backend"),
    (lambda report: report["runs"][0]["gpu_completed_steps"].__setitem__(0, 20),
     "did not complete every step"),
    (lambda report: report["runs"][0]["errors"][0].update(pha_relative=0.01001),
     "exceeds 0.01"),
    (lambda report: report["runs"][0]["cpu_pipeline_history"][2].update(cycle=7),
     "cycle IDs are not sequential"),
    (lambda report: report["runs"][0]["gpu_pipeline_history"][0].update(
        cpu_wait_seconds=99.0), "do not sum"),
    (lambda report: report["runs"][0]["gpu_history"][0].pop("dual_violation"),
     "dual_violation"),
    (lambda report: report["runs"][0]["gpu_history"][0]["primal_residual"].__setitem__(
        0, float("nan")), "finite and nonnegative"),
    (lambda report: report["runs"][0]["gpu_rows"][0].update(pha=float("nan")),
     "finite and nonnegative"),
    (lambda report: report["runs"][0]["errors"][0].update(pha_relative=0.002),
     "disagrees with endpoints"),
])
def test_summary_fails_closed_on_incomplete_or_invalid_diagnostics(mutation, match):
    report = complete_report()
    mutation(report)
    with pytest.raises(PipelineSummaryError, match=match):
        summarize_pipeline_report(report)


def test_summary_rejects_pipeline_cycle_totals_beyond_wall_tolerance():
    report = complete_report(steps=1)
    report["runs"][0]["cpu_seconds"] = 0.1
    report["runs"][0]["cpu_over_gpu_ratio"] = (
        report["runs"][0]["cpu_seconds"] / report["runs"][0]["gpu_seconds"])
    with pytest.raises(PipelineSummaryError, match="exceed wall time"):
        summarize_pipeline_report(report)


def test_lp_requests_solver_runs_and_retries_remain_separate():
    report = complete_report(steps=1)
    first = report["runs"][0]["cpu_history"][0]["rows"][0]
    first["cpu_solver_runs"] = 2
    first["numerical_retry_count"] = 1
    report["runs"][0]["cpu_solver_runs"] += 1

    overall = summarize_pipeline_report(report)["overall"]
    assert overall["cpu_reference_total_lp_calls"] == 6
    assert overall["cpu_reference_total_solver_runs"] == 7
    assert overall["cpu_reference_total_numerical_retries"] == 1


def test_cli_prints_json_without_rewriting_input(tmp_path, capsys):
    path = tmp_path / "report.json"
    original = json.dumps(complete_report(steps=2), sort_keys=True)
    path.write_text(original, encoding="utf-8")

    assert main([str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["configuration"]["steps"] == 2
    assert path.read_text(encoding="utf-8") == original


def test_summary_does_not_mutate_parsed_report():
    report = complete_report(steps=2)
    before = copy.deepcopy(report)
    summarize_pipeline_report(report)
    assert report == before


def speculative_report():
    report = complete_report(steps=1, environments=4)
    report["configuration"]["speculative_cpu"] = True
    run = report["runs"][0]
    row = run["gpu_history"][0]
    row.update(candidate_engine="speculative_legacy", bank_accepts=2, cpu_lp_calls=3,
               cpu_results_used=2, cpu_speculative_unused=1, cpu_speculative_cancelled=1,
               gpu_nonbank_accepts=0)
    for index, cpu in enumerate(row["groups"][0]["rows"]):
        cpu["cpu_result_used"] = index != 0
    row["speculative_cpu"] = dict(cpu_jobs_actual_started=3, cpu_jobs_cancelled=1,
        cpu_results_used=2, unused_cpu_results_gpu_accepted=1,
        joined_all=True, outstanding_jobs=0)
    run["online_cpu_lp_calls"] = sum(item["cpu_lp_calls"] for item in run["gpu_history"])
    run["online_cpu_solver_runs"] = run["online_cpu_lp_calls"]
    return report


def test_speculative_actual_work_and_final_adoption_are_counted_separately():
    summary = summarize_pipeline_report(speculative_report())
    count = summary["overall"]["gpu_history_counts"]["maxmin"]
    assert count["bank_accepts"] == 2
    assert count["cpu_lp_calls"] == 3
    assert count["cpu_results_used"] == 2
    assert count["cpu_speculative_unused"] == 1
    assert count["cpu_speculative_cancelled"] == 1
    assert count["bank_accepts"] + count["cpu_lp_calls"] - count["cpu_speculative_unused"] == 4
    assert summary["overall"]["gpu_total_cpu_results_used"] == 10
    assert summary["overall"]["gpu_total_cpu_lp_calls"] == 11
    assert summary["overall"]["warnings"] == []


def test_unused_cpu_failure_is_preserved_as_warning_not_final_certificate_failure():
    report = speculative_report()
    cpu = report["runs"][0]["gpu_history"][0]["groups"][0]["rows"][0]
    cpu.update(success=False, message="unused CPU numerical failure", relative_kkt_gap=float("nan"))
    summary = summarize_pipeline_report(report)
    assert summary["status"] == "validated_completed_pipeline"
    assert summary["overall"]["gpu_total_cpu_speculative_unused_failures"] == 1
    warning = summary["overall"]["warnings"][0]
    assert warning["diagnostic"]["message"] == "unused CPU numerical failure"
    assert warning["diagnostic"]["relative_kkt_gap"] == "nan"
    assert warning["diagnostic"]["success"] is False
    assert summary["runs"][0]["warnings"] == [warning]
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(cpu_results_used=3), "speculative route counts"),
    (lambda row: row.update(cpu_speculative_cancelled=2), "speculative route counts"),
    (lambda row: row.update(cpu_speculative_unused=0), "speculative route counts"),
    (lambda row: row.update(gpu_nonbank_accepts=1), "does not support GPU repair"),
    (lambda row: row["groups"][0]["rows"][0].pop("cpu_result_used"), "cpu_result_used boolean"),
    (lambda row: row["groups"][0]["rows"][0].update(cpu_result_used=True), "adoption counts"),
    (lambda row: row["groups"][0]["rows"][1].update(success=False), "failed exact-CPU row used"),
    (lambda row: row["speculative_cpu"].update(cpu_jobs_actual_started=4), "telemetry disagrees"),
    (lambda row: row["speculative_cpu"].update(joined_all=False), "not all joined"),
])
def test_speculation_summary_rejects_missing_or_inconsistent_accounting(mutation, match):
    report = speculative_report()
    mutation(report["runs"][0]["gpu_history"][0])
    with pytest.raises(PipelineSummaryError, match=match):
        summarize_pipeline_report(report)
