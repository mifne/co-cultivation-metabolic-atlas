import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.summarize_gpu_first_batch import (
    GPU_FIRST_PHASE_FIELDS,
    REQUIRED_SOURCE_FILES,
    GpuFirstBatchSummaryError,
    main,
    summarize_gpu_first_batch_report,
)


MODELS = {"species_a": "a" * 64, "species_b": "b" * 64}


def _source_snapshot(directory, *, coverage=False):
    names = set(REQUIRED_SOURCE_FILES)
    if coverage:
        names.update({"src/coverage_router.py", "src/coverage_router_binding.py"})
    hashes = {}
    for index, name in enumerate(sorted(names)):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"snapshot {index}: {name}\n", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _cpu_row(stage, environment_id, *, solver_runs=1, retries=0, used=None,
             dictionary_reason="nearest_centroid", dictionary_candidate=0):
    row = dict(
        stage=stage,
        environment_id=environment_id,
        success=True,
        certificate_passed=True,
        cpu_lp_calls=1,
        cpu_solver_runs=solver_runs,
        numerical_retry_count=retries,
        dictionary_reason=dictionary_reason,
        dictionary_candidate=dictionary_candidate,
        primal_residual=1e-8,
        dual_violation=1e-10,
        relative_kkt_gap=1e-9,
    )
    if used is not None:
        row["cpu_result_used"] = used
    return row


def _cpu_record(stage, cycle, environments):
    rows = [_cpu_row(stage, environment_id)
            for environment_id in range(environments)]
    return dict(
        stage=stage,
        pipeline_cycle=cycle,
        batch=environments,
        bank_accepts=0,
        candidate_evaluations=0,
        cpu_lp_calls=environments,
        cpu_solver_runs=environments,
        numerical_retry_count=0,
        accepted=[True] * environments,
        rows=rows,
        primal_residual=[row["primal_residual"] for row in rows],
        dual_violation=[row["dual_violation"] for row in rows],
        relative_kkt_gap=[row["relative_kkt_gap"] for row in rows],
    )


def _gpu_record(stage, environments, candidate_limit, *, fallback="exact"):
    if fallback == "exact":
        cpu_positions = list(range(1, environments))
        routes = ["gpu_dictionary_batch"] + ["cpu_after_gpu_certificate"] * (environments - 1)
    else:
        cpu_positions = []
        routes = ["gpu_dictionary_batch"] * environments
    cpu_rows = [_cpu_row(stage, position, used=True) for position in cpu_positions]
    phases = dict(
        parse_normalize_seconds=0.01,
        stack_upload_seconds=0.02,
        device_rank_seconds=0.01,
        evaluation_submission_seconds=0.02,
        completion_download_validation_seconds=0.02,
        rejected_cpu_seconds=0.02,
    )
    assert tuple(phases) == GPU_FIRST_PHASE_FIELDS
    return dict(
        stage=stage,
        batch=environments,
        environment_ids=list(range(environments)),
        gpu_first_batch=True,
        gpu_cpu_fallback=fallback,
        candidate_router="nearest_centroid",
        candidate_engine="gpu_first_heterogeneous",
        temporal_candidate_policy="within-budget",
        certificate_only=True,
        device_routing=True,
        packed_result_transfer=True,
        candidate_evaluations=environments * candidate_limit,
        bank_accepts=environments - len(cpu_positions),
        gpu_nonbank_accepts=0,
        cpu_lp_calls=len(cpu_positions),
        cpu_results_used=len(cpu_positions),
        cpu_speculative_unused=0,
        cpu_speculative_cancelled=0,
        cpu_solver_runs=len(cpu_positions),
        numerical_retry_count=0,
        preparation_seconds=0.03,
        bank_seconds=0.05,
        seconds=sum(phases.values()),
        routes=routes,
        accepted=[True] * environments,
        candidate_count=[None] * environments,
        observable_dispersion=[None] * environments,
        repair_policy="gpu_first_no_legacy_repair",
        repair_eligible=[False] * environments,
        cpu_initial_basis_proposals=bool(cpu_positions),
        groups=[dict(
            ids=cpu_positions,
            environment_ids=cpu_positions,
            route="cpu_fallback",
            rows=cpu_rows,
        )],
        gpu_first_phases=phases,
        primal_residual=[1e-8] * environments,
        dual_violation=[1e-10] * environments,
        relative_kkt_gap=[1e-9] * environments,
    )


def _pipeline_cycles(steps, environments):
    rows = []
    for cycle in range(steps):
        parts = dict(
            maxmin_service_seconds=0.10,
            main_thread_resume_seconds=0.20,
            cpu_preparation_seconds=0.05,
            cpu_wait_seconds=0.14,
            scheduler_overhead_seconds=0.01,
        )
        rows.append(dict(
            cycle=cycle,
            batch=environments,
            **parts,
            total_seconds=sum(parts.values()),
            async_span_seconds={"aggregate": 0.2, "exchange": 0.3},
        ))
    return rows


def _endpoints(environments):
    cpu = [dict(
        pha=2.0,
        phv_fraction=0.5,
        biomass={"species_a": 1.0, "species_b": 2.0},
        metabolites={"substrate": 3.0},
    ) for _ in range(environments)]
    gpu = [dict(
        pha=1.998,
        phv_fraction=0.497,
        biomass={"species_a": 0.998, "species_b": 2.0},
        metabolites={"substrate": 3.001},
    ) for _ in range(environments)]
    errors = [dict(pha_relative=0.001, biomass_g_l=0.002, phv_fraction=0.003)
              for _ in range(environments)]
    return cpu, gpu, errors


def complete_report(source_directory, *, steps=2, environments=2,
                    repeats=1, fallback="exact"):
    candidate_limit = 2
    configuration = dict(
        gpu_first_batch=True,
        gpu_cpu_fallback=fallback,
        pipeline_cpu_stages=True,
        hybrid=True,
        cpu_backend="dictionary",
        tie_policy="original3",
        candidate_oracle=False,
        candidate_diagnostics=False,
        speculative_cpu=False,
        heterogeneous_candidates=True,
        hybrid_rounds=0,
        cpu_basis_handoff=False,
        gpu_stages=list(("maxmin", "aggregate", "exchange")),
        steps=steps,
        environments=environments,
        repeats=repeats,
        seed=100,
        seed_stride=environments,
        execution_order="alternate" if repeats > 1 else "cpu-first",
        candidate_limit=candidate_limit,
        bank=str(source_directory.parent / "bank"),
        coverage_router=None,
        coverage_router_sha256=None,
    )
    report = dict(
        status="completed",
        configuration=configuration,
        model_fingerprints=copy.deepcopy(MODELS),
        offline_bank_manifest=dict(
            status="completed",
            model_fingerprints=copy.deepcopy(MODELS),
            train_seeds=[1, 2, 3],
            stages=[dict(
                stage=stage,
                key=[stage, 2, 3, 1],
                entries=[{
                    "filename": f"{stage}_{index}.npz",
                    "sha256": hashlib.sha256(
                        f"{stage}:{index}".encode("ascii")).hexdigest(),
                } for index in range(3)],
            ) for stage in ("maxmin", "aggregate", "exchange")],
        ),
        gpu_first_batch_policy=dict(
            cpu_schedule="asynchronous aggregate/exchange CPU pipeline",
            gpu_schedule="full-cohort maxmin/aggregate/exchange stage barriers",
            cpu_fallback=fallback,
            comparison_kind="strong CPU pipeline versus end-to-end GPU stage batching",
            isolated_hardware_comparison=False,
        ),
        source_hashes=_source_snapshot(source_directory),
        runs=[],
    )
    bank = source_directory.parent / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    (bank / "manifest.json").write_text(
        json.dumps(report["offline_bank_manifest"], indent=2), encoding="utf-8")
    if fallback == "reject":
        report["performance_comparison_valid"] = False
    for repeat in range(repeats):
        cpu_pipeline = _pipeline_cycles(steps, environments)
        cpu_history = [_cpu_record(stage, cycle, environments)
                       for cycle in range(steps)
                       for stage in ("maxmin", "aggregate", "exchange")]
        gpu_history = [_gpu_record(stage, environments, candidate_limit,
                                  fallback=fallback)
                       for _ in range(steps)
                       for stage in ("maxmin", "aggregate", "exchange")]
        cpu_wall = sum(row["total_seconds"] for row in cpu_pipeline) + 0.1
        gpu_wall = sum(row["seconds"] for row in gpu_history) + 0.1
        cpu_endpoints, gpu_endpoints, errors = _endpoints(environments)
        first = configuration["seed"] + repeat * environments
        run = dict(
            repeat=repeat,
            seeds=list(range(first, first + environments)),
            execution_order=("cpu-first" if repeat % 2 == 0 else "gpu-first")
                            if repeats > 1 else "cpu-first",
            cold_first_use_included=repeat == 0,
            cpu_driver_kind="pipeline",
            gpu_driver_kind="barrier",
            failure=None,
            all_endpoint_gates_passed=True,
            cpu_completed_steps=[steps] * environments,
            gpu_completed_steps=[steps] * environments,
            cpu_rows=cpu_endpoints,
            gpu_rows=gpu_endpoints,
            errors=errors,
            cpu_seconds=cpu_wall,
            gpu_seconds=gpu_wall,
            cpu_pipeline_history=cpu_pipeline,
            cpu_history=cpu_history,
            gpu_history=gpu_history,
            cpu_solver_runs=sum(row["cpu_solver_runs"] for row in cpu_history),
            online_cpu_lp_calls=sum(row["cpu_lp_calls"] for row in gpu_history),
            online_cpu_solver_runs=sum(row["cpu_solver_runs"] for row in gpu_history),
            model_bridge_cpu_lp_stage_calls=0,
        )
        if fallback == "exact":
            run["cpu_over_gpu_ratio"] = cpu_wall / gpu_wall
        else:
            run["performance_comparison_valid"] = False
        report["runs"].append(run)
    return report


def _router_schema(stage_contract):
    _, rows, columns, _ = stage_contract["key"]
    shapes = {
        "rhs": [rows],
        "lower": [columns],
        "upper": [columns],
        "c": [columns],
        "delta": [1, columns],
        "col_scale": [columns],
        "row_scale": [rows],
    }
    width = sum(int(np.prod(shape)) for shape in shapes.values())
    schema = {
        "version": 1,
        "field_order": [
            "rhs", "lower", "upper", "c", "delta", "col_scale", "row_scale"],
        "field_shapes": shapes,
        "original_feature_width": width,
        "selected_feature_indices": [0, 1],
        "selected_feature_width": 2,
        "encoding": "nan_to_num(0,+1e13,-1e13);sign(x)*log1p(abs(x));float32",
        "flattening": "C-order within fields, then field_order concatenation",
    }
    digest = hashlib.sha256(json.dumps(
        schema, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return schema, digest


def _write_router_artifact(path, *, stage, stage_contract, bank_sha,
                           coverage_sha, models=MODELS):
    schema, schema_sha = _router_schema(stage_contract)
    count = stage_contract["candidate_count"]
    architecture = {
        "input_dim": 2,
        "candidate_count": count,
        "hidden_dim": 0,
        "selected_feature_count": 2,
        "activation": "linear",
    }
    metadata = {
        "format": "multilabel_original_lp_coverage_router",
        "version": 1,
        "architecture": architecture,
        "provenance": {
            "bank_sha256": bank_sha,
            "coverage_sha256": coverage_sha,
            "feature_schema_sha256": schema_sha,
            "model_fingerprints": copy.deepcopy(models),
            "candidate_ids": copy.deepcopy(stage_contract["candidate_ids"]),
            "stage": stage,
            "stage_key": copy.deepcopy(stage_contract["key"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
        indices=np.array([0, 1], dtype=np.int64),
        mean=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        w1=np.zeros((2, count), dtype=np.float32),
        b1=np.zeros(count, dtype=np.float32),
    )
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema": schema,
        "schema_sha256": schema_sha,
        "architecture": architecture,
        "metadata": metadata,
    }


def _stage_contract(report, stage):
    raw = next(row for row in report["offline_bank_manifest"]["stages"]
               if row["stage"] == stage)
    return {
        "key": raw["key"],
        "candidate_count": len(raw["entries"]),
        "candidate_ids": [entry["filename"] + ":" + entry["sha256"]
                          for entry in raw["entries"]],
    }


def _record_rows(record):
    if isinstance(record.get("rows"), list):
        return record["rows"]
    return [row for group in record.get("groups", [])
            for row in group.get("rows", [])]


def _add_router_bindings(report, sources, tmp_path, *, stages,
                         legacy_singular=False):
    stages = tuple(stages)
    bank_manifest = (tmp_path / "bank" / "manifest.json")
    bank_sha = hashlib.sha256(bank_manifest.read_bytes()).hexdigest()
    bindings = {}
    specs = {}
    for offset, stage in enumerate(stages):
        contract = _stage_contract(report, stage)
        coverage_sha = hashlib.sha256(
            f"coverage:{stage}".encode("ascii")).hexdigest()
        path = tmp_path / "routers" / f"{stage}.npz"
        artifact = _write_router_artifact(
            path, stage=stage, stage_contract=contract, bank_sha=bank_sha,
            coverage_sha=coverage_sha)
        binding = {
            "path": str(path.resolve()),
            "sha256": artifact["sha256"],
            "stage": stage,
            "stage_key": copy.deepcopy(contract["key"]),
            "bank_manifest_sha256": bank_sha,
            "coverage_sha256": coverage_sha,
            "feature_schema_sha256": artifact["schema_sha256"],
            "feature_schema": artifact["schema"],
            "candidate_count": contract["candidate_count"],
            "candidate_ids": contract["candidate_ids"],
            "architecture": artifact["architecture"],
            "scope": "fixture",
            "cpu_binding_and_numpy_setup_seconds": 0.001 + offset * 0.0001,
            "gpu_binding_and_upload_setup_seconds": 0.002 + offset * 0.0001,
        }
        bindings[stage] = binding
        specs[stage] = (str(path.resolve()), artifact["sha256"])

    config = report["configuration"]
    if "maxmin" in specs:
        config["coverage_router"], config["coverage_router_sha256"] = specs["maxmin"]
    config["stage_coverage_router"] = [
        [stage, specs[stage][0], specs[stage][1]]
        for stage in stages if stage != "maxmin"]
    if legacy_singular:
        assert stages == ("maxmin",)
        report["coverage_router_binding"] = bindings["maxmin"]
    else:
        report["coverage_router_bindings"] = bindings
        if "maxmin" in bindings:
            report["coverage_router_binding"] = bindings["maxmin"]
    report["coverage_router_comparator"] = {
        "same_final_candidate_bank": True,
        "same_learned_proposer": True,
        "cpu_persistent_warm_basis": True,
    }
    report["source_hashes"] = _source_snapshot(sources, coverage=True)

    cpu_setup = sum(row["cpu_binding_and_numpy_setup_seconds"]
                    for row in bindings.values())
    gpu_setup = sum(row["gpu_binding_and_upload_setup_seconds"]
                    for row in bindings.values())
    for run in report["runs"]:
        for record in run["cpu_history"]:
            if record["stage"] in bindings:
                for row in _record_rows(record):
                    if row["dictionary_reason"] != "existing_cpu_basis":
                        row["dictionary_reason"] = "learned_coverage"
        for record in run["gpu_history"]:
            if record["stage"] in bindings:
                record["candidate_router"] = "learned_coverage"
                for row in _record_rows(record):
                    if row["dictionary_reason"] != "existing_cpu_basis":
                        row["dictionary_reason"] = "learned_coverage"
        cold = run["cold_first_use_included"]
        run_cpu_setup = cpu_setup if cold else 0.0
        run_gpu_setup = gpu_setup if cold else 0.0
        run["cpu_coverage_router_setup_seconds"] = run_cpu_setup
        run["gpu_coverage_router_setup_seconds"] = run_gpu_setup
        cpu_including = run["cpu_seconds"] + run_cpu_setup
        gpu_including = run["gpu_seconds"] + run_gpu_setup
        run["cpu_seconds_including_coverage_router_setup"] = cpu_including
        run["gpu_seconds_including_coverage_router_setup"] = gpu_including
        if config["gpu_cpu_fallback"] == "exact":
            run["cpu_over_gpu_ratio_including_coverage_router_setup"] = (
                cpu_including / gpu_including)
    return bindings


def test_valid_exact_report_separates_cpu_pipeline_and_gpu_stage_accounting(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources, steps=3, environments=2, repeats=2)

    summary = summarize_gpu_first_batch_report(
        report, source_directory=sources, block_size=2)

    assert summary["status"] == "validated_completed_gpu_first_batch"
    assert summary["source_snapshot"]["files_verified"] == len(REQUIRED_SOURCE_FILES)
    assert summary["overall"]["performance_comparison_valid"] is True
    assert summary["overall"]["cpu_reference_total_lp_calls"] == 36
    assert summary["overall"]["gpu_bank_accepts"] == 18
    assert summary["overall"]["gpu_exact_cpu_lp_calls"] == 18
    assert summary["overall"]["gpu_exact_cpu_solver_runs"] == 18
    assert summary["overall"]["gpu_candidate_evaluations"] == 72
    assert summary["overall"]["cpu_over_gpu_ratio"] == pytest.approx(
        sum(run["cpu_seconds"] for run in report["runs"])
        / sum(run["gpu_seconds"] for run in report["runs"]))
    run = summary["runs"][0]
    assert [(block["start_step"], block["end_step"], block["steps"])
            for block in run["cpu_pipeline_blocks"]] == [(1, 2, 2), (3, 3, 1)]
    assert [(block["start_step"], block["end_step"], block["steps"])
            for block in run["gpu_first_blocks"]] == [(1, 2, 2), (3, 3, 1)]
    assert run["gpu_first_stage_seconds"] == pytest.approx(
        sum(row["seconds"] for row in report["runs"][0]["gpu_history"]))


def test_completed_reject_mode_remains_diagnostic_and_has_no_ratio(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources, fallback="reject")

    summary = summarize_gpu_first_batch_report(report, source_directory=sources)

    assert summary["status"] == "validated_completed_gpu_only_diagnostic"
    assert summary["overall"]["performance_comparison_valid"] is False
    assert "cpu_over_gpu_ratio" not in summary["overall"]
    assert summary["overall"]["gpu_exact_cpu_lp_calls"] == 0
    assert summary["overall"]["gpu_bank_accepts"] == 12


@pytest.mark.parametrize("mutation,match", [
    (lambda report: report.update(status="incomplete_or_accuracy_failed"),
     "status must be completed"),
    (lambda report: report["runs"][0].update(failure={"error": "uncertified"}),
     "failure diagnostic"),
    (lambda report: report["runs"][0].update(gpu_driver_kind="pipeline"),
     "pipeline CPU and barrier GPU"),
    (lambda report: report["runs"][0]["gpu_completed_steps"].__setitem__(0, 1),
     "did not complete"),
    (lambda report: report["runs"][0]["gpu_history"].pop(),
     "gpu_history must contain"),
    (lambda report: report["runs"][0]["gpu_history"][0]["gpu_first_phases"].update(
        device_rank_seconds=0.5), "phase sum"),
    (lambda report: report["runs"][0]["gpu_history"][0].update(bank_accepts=2),
     "bank accept count"),
    (lambda report: report["runs"][0]["gpu_history"][0].update(
        cpu_speculative_unused=1), "repair/speculation"),
    (lambda report: report["runs"][0]["gpu_history"][0].update(
        candidate_evaluations=3), "B x K"),
    (lambda report: report["runs"][0].update(online_cpu_lp_calls=0),
     "online_cpu_lp_calls"),
    (lambda report: report["runs"][0].update(cpu_solver_runs=0),
     "cpu_solver_runs"),
    (lambda report: report["runs"][0].update(model_bridge_cpu_lp_stage_calls=1),
     "model-bridge CPU LP"),
    (lambda report: report["runs"][0]["gpu_history"][0]["dual_violation"].__setitem__(
        0, 2e-7), "original-LP limit"),
    (lambda report: report["runs"][0]["errors"][0].update(pha_relative=0.002),
     "reported pha_relative"),
    (lambda report: report["offline_bank_manifest"].update(
        model_fingerprints={"changed": "c" * 64}), "model fingerprints"),
])
def test_malformed_or_incomplete_evidence_fails_closed(tmp_path, mutation, match):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    mutation(report)
    with pytest.raises(GpuFirstBatchSummaryError, match=match):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_false_source_hash_is_detected_from_archived_bytes(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    report["source_hashes"]["src/gpu_first_batch_lp.py"] = "f" * 64

    with pytest.raises(GpuFirstBatchSummaryError, match="SHA256 mismatch"):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_missing_required_source_is_not_hidden_by_remaining_valid_hashes(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    report["source_hashes"].pop("src/gpu_candidate_transfer.py")

    with pytest.raises(GpuFirstBatchSummaryError, match="missing required files"):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_gpu_only_reject_failure_cannot_be_summarized_as_performance_success(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources, fallback="reject")
    report["status"] = "incomplete_or_accuracy_failed"
    run = report["runs"][0]
    run["failure"] = {"error": "An LP rejected"}
    run["all_endpoint_gates_passed"] = False
    run["gpu_completed_steps"][0] = 0

    with pytest.raises(GpuFirstBatchSummaryError, match="not performance successes"):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_evaluation_seeds_must_be_distinct_and_outside_training(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources, repeats=2)
    report["configuration"]["seed_stride"] = 0
    report["runs"][1]["seeds"] = report["runs"][0]["seeds"].copy()
    with pytest.raises(GpuFirstBatchSummaryError, match="duplicated across runs"):
        summarize_gpu_first_batch_report(report, source_directory=sources)

    report = complete_report(sources, repeats=1)
    report["offline_bank_manifest"]["train_seeds"].append(100)
    with pytest.raises(GpuFirstBatchSummaryError, match="overlap offline training"):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_report_is_not_mutated_and_cli_is_read_only(tmp_path, capsys):
    path = tmp_path / "benchmark.json"
    sources = path.with_suffix(".sources")
    report = complete_report(sources)
    before = copy.deepcopy(report)
    summary = summarize_gpu_first_batch_report(report, source_directory=sources)
    assert report == before
    assert summary["configuration"]["gpu_driver_kind"] == "barrier"

    original = json.dumps(report, sort_keys=True)
    path.write_text(original, encoding="utf-8")
    assert main([str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "validated_completed_gpu_first_batch"
    assert path.read_text(encoding="utf-8") == original


def test_all_stage_router_bindings_are_verified_against_artifacts(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources, repeats=2)
    bindings = _add_router_bindings(
        report, sources, tmp_path, stages=("maxmin", "aggregate", "exchange"))

    summary = summarize_gpu_first_batch_report(report, source_directory=sources)

    assert set(summary["coverage_router_bindings"]) == {
        "maxmin", "aggregate", "exchange"}
    for stage, binding in bindings.items():
        checked = summary["coverage_router_bindings"][stage]
        assert checked["sha256"] == binding["sha256"]
        assert checked["candidate_count"] == 3
        assert checked["stage_key"][0] == stage
    assert summary["runs"][0]["gpu_history_counts"]["aggregate"][
        "candidate_evaluations"] == 8


def test_legacy_single_maxmin_router_binding_remains_supported(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    _add_router_bindings(
        report, sources, tmp_path, stages=("maxmin",), legacy_singular=True)

    summary = summarize_gpu_first_batch_report(report, source_directory=sources)

    assert set(summary["coverage_router_bindings"]) == {"maxmin"}
    assert summary["configuration"]["coverage_router_specs"]["maxmin"][
        "sha256"] == report["configuration"]["coverage_router_sha256"]


@pytest.mark.parametrize("mutation,match", [
    (lambda report: report["coverage_router_bindings"].pop("aggregate"),
     "binding stages disagree"),
    (lambda report: report["configuration"]["stage_coverage_router"][0].__setitem__(
        2, "f" * 64), "stage/artifact SHA"),
    (lambda report: report["coverage_router_bindings"]["aggregate"].update(
        candidate_count=2), "candidate count disagrees"),
    (lambda report: report["runs"][0]["gpu_history"][1].update(
        candidate_router="nearest_centroid"), "candidate router disagrees"),
    (lambda report: report["runs"][0]["cpu_history"][1]["rows"][0].update(
        dictionary_reason="nearest_centroid"), "dictionary proposer disagrees"),
])
def test_all_stage_router_contract_mutations_fail_closed(
        tmp_path, mutation, match):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    _add_router_bindings(
        report, sources, tmp_path, stages=("maxmin", "aggregate", "exchange"))
    mutation(report)

    with pytest.raises(GpuFirstBatchSummaryError, match=match):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_router_artifact_bytes_are_checked_against_pinned_sha(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    bindings = _add_router_bindings(
        report, sources, tmp_path, stages=("maxmin", "aggregate", "exchange"))
    path = Path(bindings["exchange"]["path"])
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(GpuFirstBatchSummaryError, match="artifact SHA256 mismatch"):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_router_artifact_model_provenance_cannot_be_rebound_by_updating_sha(
        tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    bindings = _add_router_bindings(
        report, sources, tmp_path, stages=("maxmin", "aggregate", "exchange"))
    binding = bindings["aggregate"]
    path = Path(binding["path"])
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files
                  if name != "metadata"}
        metadata = json.loads(str(stored["metadata"]))
    metadata["provenance"]["model_fingerprints"] = {"forged": "f" * 64}
    np.savez_compressed(
        path, metadata=np.array(json.dumps(metadata, sort_keys=True)), **arrays)
    new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    binding["sha256"] = new_sha
    report["configuration"]["stage_coverage_router"][0][2] = new_sha

    with pytest.raises(GpuFirstBatchSummaryError, match="model_fingerprints"):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_router_bank_manifest_pin_is_verified_from_configured_file(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    _add_router_bindings(report, sources, tmp_path, stages=("maxmin",))
    manifest = tmp_path / "bank" / "manifest.json"
    # Semantically identical JSON is still a different pinned bank artifact.
    manifest.write_text(json.dumps(report["offline_bank_manifest"]), encoding="utf-8")

    with pytest.raises(GpuFirstBatchSummaryError, match="not pinned"):
        summarize_gpu_first_batch_report(report, source_directory=sources)


def test_additional_stage_router_requires_router_sources_in_snapshot(tmp_path):
    sources = tmp_path / "run.sources"
    report = complete_report(sources)
    _add_router_bindings(report, sources, tmp_path, stages=("aggregate",))
    report["source_hashes"].pop("src/coverage_router.py")

    with pytest.raises(GpuFirstBatchSummaryError, match="missing required files"):
        summarize_gpu_first_batch_report(report, source_directory=sources)
