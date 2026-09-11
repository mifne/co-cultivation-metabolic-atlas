from copy import deepcopy

import pytest

from tests.test_gpu_exchange_tie_audit import pair, step_history
from scripts.audit_gpu_device_replay import audit, audit_history


def device_history(history):
    history = deepcopy(history)
    for stage in history:
        stage["gpu"]["method"] = "gpu_device_bounded_simplex"
        if "tie_break" in stage:
            stage["tie_break"]["gpu"]["method"] = "gpu_device_bounded_simplex"
    return history


def test_device_method_identified_without_accepting_reference_as_device():
    assert audit_history(device_history(step_history()), 1)["passed"]
    assert not audit_history(step_history(), 1)["passed"]


@pytest.mark.parametrize("bad", ["cpu", "residual", "tie", "objective", "attempt"])
def test_device_failure_cannot_pass(bad):
    history = device_history(step_history())
    tie = history[2]["tie_break"]
    if bad == "cpu":
        tie["gpu"]["cpu_lp_calls"] = 1
    elif bad == "residual":
        tie["max_original_residual"] = 1e-4
    elif bad == "tie":
        history[2].pop("tie_break")
    elif bad == "objective":
        history[2]["selected_primary_value"] = 1.1
    else:
        tie["gpu"]["gpu_lp_attempts"] = float("nan")
    assert not audit_history(history, 1)["passed"]


def test_device_full_replay_still_requires_original_accuracy_and_source_gates():
    replay, reference = pair()
    replay["metadata"]["backend"] = "device-exchange-tie"
    replay["metadata"]["diagnostic_source_sha256"].update({p: "synthetic" for p in
        ("src/gpu_device_bounded_simplex.py", "src/gpu_simplex_pivot_batch.py", "scripts/run_gpu_device_regression.py")})
    replay["backend_history"] = device_history(replay["backend_history"])
    assert audit(replay, reference, check_sources=False)["passed"]
    replay["rows"][-1]["after"]["pha"] = 1.02
    assert "endpoint:pha_relative" in audit(replay, reference, check_sources=False)["failures"]
    replay["metadata"]["diagnostic_source_sha256"].pop("src/gpu_device_bounded_simplex.py")
    assert "missing_device_source:src/gpu_device_bounded_simplex.py" in audit(replay, reference, check_sources=False)["failures"]


def test_separate_cache_declaration_requires_four_correct_namespaces():
    replay, reference = pair()
    replay["metadata"]["backend"] = "device-exchange-tie"
    replay["metadata"]["basis_cache_mode"] = "separate_original_objectives"
    replay["metadata"]["diagnostic_source_sha256"].update({p: "synthetic" for p in
        ("src/gpu_device_bounded_simplex.py", "src/gpu_simplex_pivot_batch.py", "scripts/run_gpu_device_regression.py",
         "src/gpu_stage_cache_backend.py")})
    replay["backend_history"] = device_history(replay["backend_history"])
    for i, stage in enumerate(replay["backend_history"]):
        stage["gpu"]["warm_cache_namespace"] = ("maxmin", "aggregate", "exchange_primary")[i % 3]
        if i % 3 == 2:
            stage["tie_break"]["gpu"]["warm_cache_namespace"] = "exchange_tie"
    assert audit(replay, reference, check_sources=False)["passed"]
    replay["backend_history"][2]["tie_break"]["gpu"]["warm_cache_namespace"] = "exchange_primary"
    assert "incorrect_tie_cache_namespace:2" in audit(replay, reference, check_sources=False)["failures"]
