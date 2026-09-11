"""Host-only policy tests for the all-stage GPU batch benchmark driver."""
import sys

import pytest

from scripts.benchmark_compact_gpu import (
    benchmark_driver_kind,
    diagnostic_only_performance,
    drive_benchmark_side,
    forbid_highs_in_gpu_run,
)


@pytest.mark.parametrize(
    "pipeline,gpu_first,cpu_kind,gpu_kind",
    [
        (False, False, "barrier", "barrier"),
        (True, False, "pipeline", "pipeline"),
        (True, True, "pipeline", "barrier"),
        (False, True, "barrier", "barrier"),
    ],
)
def test_driver_policy_is_side_specific(pipeline, gpu_first, cpu_kind, gpu_kind):
    assert benchmark_driver_kind(
        "cpu", pipeline_cpu_stages=pipeline, gpu_first_batch=gpu_first
    ) == cpu_kind
    assert benchmark_driver_kind(
        "gpu", pipeline_cpu_stages=pipeline, gpu_first_batch=gpu_first
    ) == gpu_kind


def test_driver_policy_rejects_unknown_side():
    with pytest.raises(ValueError, match="side"):
        benchmark_driver_kind("accelerator", pipeline_cpu_stages=True,
                              gpu_first_batch=True)


def test_gpu_first_uses_strong_cpu_pipeline_and_gpu_barrier():
    calls = []

    def pipeline(*args, **kwargs):
        calls.append(("pipeline", args, kwargs))
        return "cpu-result"

    def barrier(*args, **kwargs):
        calls.append(("barrier", args, kwargs))
        return "gpu-result"

    common = (["env"], ["action"], "backend", "snapshot", "progress")
    cpu = drive_benchmark_side(
        *common, side="cpu", pipeline_cpu_stages=True, gpu_first_batch=True,
        workers=7, pipeline_driver=pipeline, barrier_driver=barrier
    )
    gpu = drive_benchmark_side(
        *common, side="gpu", pipeline_cpu_stages=True, gpu_first_batch=True,
        workers=7, pipeline_driver=pipeline, barrier_driver=barrier
    )

    assert (cpu, gpu) == ("cpu-result", "gpu-result")
    assert [row[0] for row in calls] == ["pipeline", "barrier"]
    assert calls[0][2] == {"workers": 7}
    assert calls[1][2] == {}
    assert calls[0][1] == calls[1][1] == common


@pytest.mark.parametrize(
    "hybrid,gpu_first,fallback,expected",
    [
        (False, False, "exact", True),
        (True, False, "exact", False),
        (True, True, "exact", False),
        (True, True, "reject", True),
    ],
)
def test_reject_diagnostic_forbids_online_highs(hybrid, gpu_first, fallback, expected):
    assert forbid_highs_in_gpu_run(
        hybrid=hybrid, gpu_first_batch=gpu_first,
        gpu_cpu_fallback=fallback
    ) is expected


def test_reject_and_oracle_modes_cannot_report_performance():
    assert diagnostic_only_performance(candidate_oracle=True)
    assert diagnostic_only_performance(
        gpu_first_batch=True, gpu_cpu_fallback="reject"
    )
    assert not diagnostic_only_performance(
        gpu_first_batch=True, gpu_cpu_fallback="exact"
    )


def _gpu_first_argv(tmp_path, *extra):
    output = tmp_path / "existing.json"
    output.touch()
    return [
        "benchmark", "--bank", str(tmp_path / "not-loaded"),
        "--output", str(output), "--hybrid", "--hybrid-rounds", "0",
        "--tie-policy", "original3", "--cpu-backend", "dictionary",
        "--heterogeneous-candidates", "--pipeline-cpu-stages",
        "--gpu-first-batch", "--gpu-stages", "maxmin", "aggregate", "exchange",
        *extra,
    ]


@pytest.mark.parametrize("fallback", ["exact", "reject"])
def test_valid_gpu_first_policy_reaches_output_guard_before_gpu_import(
        tmp_path, monkeypatch, fallback):
    from scripts.benchmark_compact_gpu import main

    monkeypatch.setattr(sys, "argv", _gpu_first_argv(
        tmp_path, "--gpu-cpu-fallback", fallback
    ))
    with pytest.raises(FileExistsError):
        main()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda argv: [arg for arg in argv if arg != "--hybrid"],
        lambda argv: argv + ["--speculative-cpu"],
        lambda argv: [arg for arg in argv if arg != "--heterogeneous-candidates"],
        lambda argv: [arg for arg in argv if arg != "--pipeline-cpu-stages"],
        lambda argv: ["persistent" if arg == "dictionary" else arg for arg in argv],
        lambda argv: ["secondary" if arg == "original3" else arg for arg in argv],
        lambda argv: ["1" if arg == "0" else arg for arg in argv]
                     + ["--repair-operators", "unused"],
        lambda argv: argv[:argv.index("--gpu-stages") + 1] + ["maxmin"],
        lambda argv: argv + ["--candidate-diagnostics"],
    ],
)
def test_gpu_first_invalid_policy_fails_before_artifact_or_gpu_setup(
        tmp_path, monkeypatch, mutation):
    from scripts.benchmark_compact_gpu import main

    monkeypatch.setattr(sys, "argv", mutation(_gpu_first_argv(tmp_path)))
    with pytest.raises(ValueError, match="GPU-first|Pipelining|Speculative"):
        main()


def test_reject_policy_without_gpu_first_fails_before_output_guard(tmp_path, monkeypatch):
    from scripts.benchmark_compact_gpu import main

    output = tmp_path / "existing.json"
    output.touch()
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--bank", str(tmp_path / "not-loaded"),
        "--output", str(output), "--hybrid", "--hybrid-rounds", "0",
        "--tie-policy", "original3", "--gpu-cpu-fallback", "reject",
    ])
    with pytest.raises(ValueError, match="reject policy"):
        main()
