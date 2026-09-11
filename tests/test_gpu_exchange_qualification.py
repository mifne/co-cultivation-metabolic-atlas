from copy import deepcopy

from scripts import qualify_gpu_exchange_tie as qualification
from scripts.audit_gpu_exchange_tie_replay import POLICY, audit as actual_audit


def fixture_pair():
    trace = [dict(step=i, accepted=True, status="optimal; stage=parsimonious_exchange",
        pha=1., phv_fraction=.1, biomass={"a": .2}, metabolites={"nh4_e": .1}) for i in range(120)]
    stage = dict(success=True, cpu_optimization_allowed=False, max_original_residual=1e-9,
        cpu_iteration_counts=dict(simplex_iteration_count=-1, ipm_iteration_count=-1, crossover_iteration_count=-1),
        gpu=dict(success=True, status="optimal", method="gpu_bounded_simplex", cpu_lp_calls=0,
            max_original_residual=1e-9, gpu_lp_attempts=1), actual_gpu_lp_calls=1)
    history = []
    for _ in range(120):
        history.extend([deepcopy(stage), deepcopy(stage), dict(deepcopy(stage),
            actual_gpu_lp_calls=2, tie_break=deepcopy(stage), primary_optimum=1.,
            selected_primary_value=1., primary_face_allowance=1e-10, selection_policy=POLICY)])
    exact = dict(steps=120, trajectory=trace, model_fingerprints={"a": "synthetic"},
        cpu_lp_stage_calls=360, gpu_lp_stage_calls=0)
    gpu = dict(deepcopy(exact), cpu_lp_stage_calls=0, gpu_lp_stage_calls=360, backend_history=history)
    return exact, gpu


def test_prespecified_seeds_and_actual_call_audit(monkeypatch):
    assert qualification.SEEDS == list(range(20286101, 20286106))
    monkeypatch.setattr(qualification, "audit", lambda r, c: actual_audit(r, c, check_sources=False))
    exact, gpu = fixture_pair()
    checked = qualification.audit_pair(20286101, exact, gpu, {"src/gpu_exchange_tie_break.py": "synthetic"})
    assert checked["passed"]
    assert checked["lp_audit"]["actual_gpu_lp_calls"] == 480


def test_rejects_cpu_replacement_and_incomplete_reference(monkeypatch):
    monkeypatch.setattr(qualification, "audit", lambda r, c: actual_audit(r, c, check_sources=False))
    exact, gpu = fixture_pair()
    gpu["cpu_lp_stage_calls"] = 1
    exact["cpu_lp_stage_calls"] = 240
    checked = qualification.audit_pair(20286101, exact, gpu, {"src/gpu_exchange_tie_break.py": "synthetic"})
    assert not checked["passed"]
    assert "cpu_lp_stage_calls" in checked["failures"]
    assert "cpu_reference_stage_counts" in checked["failures"]
