from copy import deepcopy

import pytest

from scripts.audit_gpu_exchange_tie_replay import POLICY, audit, audit_history


def stage():
    return dict(success=True, cpu_optimization_allowed=False,
        cpu_iteration_counts=dict(simplex_iteration_count=-1, ipm_iteration_count=-1, crossover_iteration_count=-1),
        max_original_residual=1e-9, actual_gpu_lp_calls=1,
        gpu=dict(method="gpu_bounded_simplex", status="optimal", success=True,
            cpu_lp_calls=0, gpu_lp_attempts=1, max_original_residual=1e-9))


def step_history():
    records = [stage(), stage(), stage()]
    records[2].update(tie_break=stage(), actual_gpu_lp_calls=2, selection_policy=POLICY,
        primary_optimum=1., selected_primary_value=1., primary_face_allowance=1e-10)
    return records


def test_counts_actual_four_lp_stages():
    result = audit_history(step_history(), 1)
    assert result["passed"]
    assert result["actual_gpu_lp_calls"] == 4


@pytest.mark.parametrize("mutation", [
    lambda h: h[2]["tie_break"].update(success=False),
    lambda h: h[2]["tie_break"]["gpu"].update(cpu_lp_calls=1),
    lambda h: h[2]["tie_break"]["cpu_iteration_counts"].update(simplex_iteration_count=1),
    lambda h: h[2]["tie_break"].update(max_original_residual=1e-3),
    lambda h: h[2].update(selected_primary_value=1.1),
    lambda h: h[2].update(primary_face_allowance=1.),
    lambda h: h[2].pop("tie_break"),
    lambda h: h[2].update(actual_gpu_lp_calls=1),
    lambda h: h[2].update(selected_primary_value=float("nan")),
    lambda h: h[2]["tie_break"]["cpu_iteration_counts"].update(ipm_iteration_count=float("nan")),
    lambda h: h[2]["tie_break"]["gpu"].update(gpu_lp_attempts=1.5),
])
def test_rejects_failed_extra_stage_or_changed_objective(mutation):
    history = step_history()
    mutation(history)
    assert not audit_history(history, 1)["passed"]


def pair():
    state = dict(pha=1., phv_fraction=.1, biomass={"a": .2}, metabolites={"val__L_e": 0.})
    status = "optimal; stage=parsimonious_exchange"
    hashes = {"dummy_source": "synthetic"}
    reference = dict(implementation_sha256=hashes, runs=[dict(seed=1, exact=dict(
        model_fingerprints={"a": "synthetic"}, trajectory=[dict(step=i, accepted=True,
            status=status, **{k: deepcopy(v) for k, v in state.items() if k != "metabolites"}) for i in range(120)]))])
    replay = dict(status="completed", metadata=dict(seed=1, backend="tableau-exchange-tie", stop_after=120,
        implementation_sha256=hashes, diagnostic_source_sha256={"src/gpu_exchange_tie_break.py": "synthetic"},
        model_fingerprints={"a": "synthetic"}),
        rows=[dict(step=i+1, status=status, after=deepcopy(state)) for i in range(120)],
        backend_history=[r for _ in range(120) for r in step_history()])
    return replay, reference


def test_recomputes_full_horizon_endpoint_not_cached_error():
    replay, reference = pair()
    assert audit(replay, reference, check_sources=False)["passed"]
    replay["max_pha_replay_difference"] = 0.
    replay["rows"][-1]["after"]["pha"] = 1.02
    assert "endpoint:pha_relative" in audit(replay, reference, check_sources=False)["failures"]


def test_partial_trajectory_cannot_pass():
    replay, reference = pair()
    replay["rows"].pop()
    assert "trajectory_length" in audit(replay, reference, check_sources=False)["failures"]


def test_missing_stage_or_nonfinite_pool_cannot_pass():
    replay, reference = pair()
    replay["rows"][40]["status"] = "optimal; stage=aggregate"
    replay["rows"][30]["after"]["metabolites"]["val__L_e"] = float("nan")
    result = audit(replay, reference, check_sources=False)
    assert not result["passed"]
    assert "incomplete_stage:40" in result["failures"]
    assert "nonfinite:30" in result["failures"]
