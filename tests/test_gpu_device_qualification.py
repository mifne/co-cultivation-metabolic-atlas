import numpy as np
import pytest

from scripts import qualify_gpu_device as qualification
from scripts.audit_gpu_device_replay import audit as real_audit
from tests.test_gpu_exchange_qualification import fixture_pair
from tests.test_gpu_device_audit import device_history


HASHES = {p: "synthetic" for p in ("src/gpu_exchange_tie_break.py", "src/gpu_device_bounded_simplex.py",
    "src/gpu_simplex_pivot_batch.py", "scripts/run_gpu_device_regression.py")}


def test_new_seeds_are_distinct_from_previously_inspected_conditions():
    assert qualification.SEEDS == list(range(20286201, 20286206))
    assert not set(qualification.SEEDS) & {20286001, *range(20286101, 20286106)}


def test_cpu_path_delegates_to_the_unchanged_reference_loop(monkeypatch):
    actions = np.zeros((120, 5), dtype=np.float32)
    checkpoint = object()
    marker = object()
    def original(a, seed, gpu, callback):
        assert a is actions and seed == 20286201 and gpu is False and callback is checkpoint
        return marker
    monkeypatch.setattr(qualification, "original_cpu_run_side", original)
    assert qualification.run_side(actions, 20286201, False, checkpoint) is marker


def test_audit_accepts_device_method_and_counts_four_stages(monkeypatch):
    monkeypatch.setattr(qualification, "audit", lambda r, c: real_audit(r, c, check_sources=False))
    exact, gpu = fixture_pair()
    gpu["backend_history"] = device_history(gpu["backend_history"])
    result = qualification.audit_pair(20286201, exact, gpu, HASHES)
    assert result["passed"]
    assert result["lp_audit"]["actual_gpu_lp_calls"] == 480


@pytest.mark.parametrize("issue", ["old_gpu", "cpu_fallback", "partial", "inaccurate", "missing_tie", "source_missing"])
def test_failure_never_becomes_a_qualification_pass(monkeypatch, issue):
    monkeypatch.setattr(qualification, "audit", lambda r, c: real_audit(r, c, check_sources=False))
    exact, gpu = fixture_pair()
    gpu["backend_history"] = device_history(gpu["backend_history"])
    hashes = dict(HASHES)
    if issue == "old_gpu":
        gpu["backend_history"][0]["gpu"]["method"] = "gpu_bounded_simplex"
    elif issue == "cpu_fallback":
        gpu["cpu_lp_stage_calls"] = 1
    elif issue == "partial":
        gpu["steps"] = 119
        gpu["trajectory"].pop()
    elif issue == "inaccurate":
        gpu["trajectory"][-1]["pha"] = 1.02
    elif issue == "missing_tie":
        gpu["backend_history"][2].pop("tie_break")
    else:
        hashes.pop("src/gpu_device_bounded_simplex.py")
    assert not qualification.audit_pair(20286201, exact, gpu, hashes)["passed"]
