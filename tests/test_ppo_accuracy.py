import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.ppo_accuracy import (TRAJECTORY_UNITS, screen_lp_metric, screen_ipm_record,
                              generalized_advantage, compare_trajectories, compare_rankings)


def metric(**changes):
    values = dict(primal_residual=1e-9, dual_violation=0., relative_kkt_gap=1e-8,
                  objective=-.005)
    values.update(changes)
    return values


def trajectory():
    return dict(rewards=[[1.], [2.]], terminated=[[False], [True]],
        truncated=[[False], [False]], units=dict(TRAJECTORY_UNITS), species_ids=["A", "B"],
        pha_repeat=[[0.], [.1]], pha_mass=[[0.], [.008609]], phv_fraction=[[0.], [0.]],
        biomass=[[[.1, .1]], [[.11, .11]]], ph=[[6.5], [6.5]])


def test_small_objective_gap_is_not_percentage():
    row = screen_lp_metric(metric(relative_kkt_gap=1e-4), stage="maxmin", objective_rtol=.01)
    assert row["absolute_complementarity"] == 1e-4
    assert row["objective_budget"] == pytest.approx(5.0001e-5)
    assert not row["passed"]


@pytest.mark.parametrize("changes", [{"primal_residual": 1e-4}, {"dual_violation": 1e-6},
    {"objective": np.nan}, {"relative_kkt_gap": -1.}, {"primal_residual": True},
    {"objective": 1e308, "relative_kkt_gap": 1e308}])
def test_lp_fail_closed(changes):
    assert not screen_lp_metric(metric(**changes), stage="exchange", objective_rtol=.01)["passed"]


def test_zero_objective_has_absolute_budget_only():
    row = screen_lp_metric(metric(objective=0., relative_kkt_gap=2e-9),
                           stage="aggregate", objective_rtol=1., objective_atol=1e-9)
    assert not row["passed"]
    assert row["objective_budget"] == 1e-9


def test_corrupt_initial_excluded_and_unique_env_count():
    good = metric(relative_kkt_gap=0.)
    bad = metric(primal_residual=.1)
    record = dict(stage="maxmin", step=1, batch=1, ipm=dict(iterations=2, status="failed",
        metrics=[bad], checkpoints=[dict(iteration=0, metrics=[good]),
                                   dict(iteration=1, metrics=[bad]), dict(iteration=2, metrics=[bad])]))
    result = screen_ipm_record(record)
    assert result["initial_checkpoints_excluded"] == 1
    assert all(g["environments_ever_passed"] == 0 for g in result["grids"])
    record["ipm"]["checkpoints"][1]["metrics"] = [good]
    assert all(g["environments_ever_passed"] == 1 for g in screen_ipm_record(record)["grids"])


def test_gae_termination_vs_truncation_bootstrap():
    rewards = [[1.], [100.]]
    values, next_values = [[2.], [2.]], [[10.], [10.]]
    terminated = [[False], [True]]
    truncated = [[True], [False]]
    result = generalized_advantage(rewards, values, next_values, terminated, truncated,
                                   gamma=.9, gae_lambda=1.)
    np.testing.assert_allclose(result[:, 0], [8., 98.])


def test_comparison_explicit_units_returns_and_budget():
    ref, cand = trajectory(), trajectory()
    cand["rewards"][1][0] += .5
    cand["pha_repeat"][1][0] += .001
    result = compare_trajectories(ref, cand, gamma=.9, absolute_budgets={"raw_return": .25})
    assert result["max_absolute_errors"]["raw_return"] == pytest.approx(.5)
    assert result["max_absolute_errors"]["discounted_return"] == pytest.approx(.45)
    assert result["max_absolute_errors"]["pha_repeat"] == pytest.approx(.001)
    assert not result["all_explicit_budgets_passed"]
    assert compare_trajectories(ref, ref)["all_explicit_budgets_passed"] is None


@pytest.mark.parametrize("change", ["shape", "nan", "units", "bool", "species", "unknown_budget"])
def test_trajectory_rejects_incompatible_data(change):
    ref, cand = trajectory(), trajectory()
    budgets = None
    if change == "shape": cand["pha_mass"] = [.1, .2]
    elif change == "nan": cand["ph"][0][0] = np.nan
    elif change == "units": cand["units"]["pha_repeat"] = "g/L"
    elif change == "bool": cand["terminated"] = [[0], [1]]
    elif change == "species": cand["species_ids"].reverse()
    elif change == "unknown_budget": budgets = {"universal_percentage": .01}
    with pytest.raises(ValueError): compare_trajectories(ref, cand, absolute_budgets=budgets)


def test_value_identity_and_zero_variance():
    ref = trajectory()
    ref.update(rewards=[[0.], [0.]], values=[[0.], [0.]], next_values=[[0.], [0.]],
               value_function_id="fixed_critic_and_normalizer_sha")
    cand = copy.deepcopy(ref)
    result = compare_trajectories(ref, cand)
    assert result["gae"]["normalized_rmse"] is None
    assert result["gae"]["zero_reference_variance"]
    cand["value_function_id"] = "another_network"
    with pytest.raises(ValueError): compare_trajectories(ref, cand)


def test_event_and_advantage_mismatches():
    ref = trajectory()
    ref.update(values=[[0.], [0.]], next_values=[[0.], [0.]], value_function_id="frozen")
    cand = copy.deepcopy(ref)
    cand["rewards"] = [[-1.], [-2.]]
    cand["terminated"][1][0] = False
    result = compare_trajectories(ref, cand)
    assert result["max_absolute_errors"]["gae_sign_mismatches"] == 2
    assert result["max_absolute_errors"]["terminated_mismatches"] == 1


def test_rankings_match_identifiers_and_respect_ties():
    ref = [dict(seed=1, action_id="a", **{"return": 1.}), dict(seed=1, action_id="b", **{"return": 2.})]
    cand = [dict(seed=1, action_id="b", **{"return": 1.}), dict(seed=1, action_id="a", **{"return": 2.})]
    assert compare_rankings(ref, cand)["per_seed"][0]["reversed_pairs"] == 1
    assert compare_rankings(ref, cand, tie_margin=2.)["per_seed"][0]["agreement_fraction"] is None
    with pytest.raises(ValueError): compare_rankings(ref, cand[:1])
    with pytest.raises(ValueError): compare_rankings(ref+ref[:1], cand)


def test_explicit_metabolic_events_and_domains():
    ref, cand = trajectory(), trajectory()
    ref["events"] = {"nitrogen_limited": [[True], [True]]}
    cand["events"] = {"nitrogen_limited": [[False], [True]]}
    report = compare_trajectories(ref, cand, absolute_budgets={"event:nitrogen_limited": 0})
    assert report["max_absolute_errors"]["event:nitrogen_limited"] == 1
    assert not report["all_explicit_budgets_passed"]
    cand["pha_mass"][0][0] = -.1
    with pytest.raises(ValueError): compare_trajectories(ref, cand)


@pytest.mark.parametrize("kwargs", [{"objective_rtols": []}, {"objective_rtols": [np.nan]},
                                   {"objective_atol": -1.}])
def test_screen_requires_finite_nonempty_tolerance_grid(kwargs):
    record = dict(stage="maxmin", step=1, batch=1, ipm=dict(iterations=0, metrics=[metric()]))
    with pytest.raises(ValueError): screen_ipm_record(record, **kwargs)


def test_rankings_overflow_rejected():
    rows = [dict(seed=1, action_id="a", **{"return": 1e308}),
            dict(seed=1, action_id="b", **{"return": -1e308})]
    with pytest.raises(ValueError): compare_rankings(rows, rows)


def test_cli_saved_lp_audit_does_not_overwrite(tmp_path):
    record = dict(stage="maxmin", step=1, batch=1, ipm=dict(iterations=0, metrics=[metric()]))
    source = tmp_path / "lp.json"
    source.write_text(json.dumps(record))
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts/audit_ppo_accuracy.py"
    command = [sys.executable, str(script), "--inputs", str(source), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["cpu_lp_calls"] == report["gpu_calls"] == 0
    assert not report["reference_vectors_loaded"]
    assert len(report["lp_records"]) == 1
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "never overwrite" in again.stderr
    assert output.read_bytes() == original


def test_cli_trajectory_only(tmp_path):
    source = tmp_path / "trajectory.json"
    source.write_text(json.dumps(trajectory()))
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts/audit_ppo_accuracy.py"
    subprocess.run([sys.executable, str(script), "--inputs", "--reference-trajectory", str(source),
                    "--candidate-trajectory", str(source), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["lp_records"] == []
    assert report["trajectory"]["max_absolute_errors"]["raw_return"] == 0.
    assert report["trajectory"]["all_explicit_budgets_passed"] is None
