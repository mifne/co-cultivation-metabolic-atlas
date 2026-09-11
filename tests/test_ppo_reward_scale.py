import copy

import numpy as np
import pytest

from scripts.probe_ppo_reward_scale import fixed_actions, summarize_case, pairwise_margins


def toy_result(actions):
    rows = [dict(step=i, accepted=True, status="optimal; stage=parsimonious_exchange",
                 terminated=False, truncated=False, action=a.astype(float).tolist(),
                 reward_raw=10. if i == 0 else .01,
                 pha_repeat_mmol_l=.1, pha=.008609) for i, a in enumerate(actions)]
    return dict(steps=len(actions), trajectory=rows, cpu_lp_stage_calls=3*len(actions),
                gpu_lp_stage_calls=0, model_fingerprints={"OR16": "a", "NS21": "b", "PF": "c"},
                telemetry_units=dict(pha="g/L", pha_repeat_mmol_l="mmol/L"))


def test_predeclared_actions_change_only_common_feed():
    cases = fixed_actions(8)
    assert set(cases) == {"baseline", "lower_common_feed", "higher_common_feed"}
    for action in cases.values():
        assert action.shape == (8, 5) and action.dtype == np.float32
        np.testing.assert_array_equal(action[:, [0, 1, 2, 4]], cases["baseline"][:, [0, 1, 2, 4]])
    np.testing.assert_allclose(cases["higher_common_feed"][:, 3] - cases["baseline"][:, 3], .05,
                               rtol=0., atol=np.finfo(np.float32).eps)


@pytest.mark.parametrize("steps", [0, -1, True, 121, 2.5])
def test_bounded_steps(steps):
    with pytest.raises(ValueError): fixed_actions(steps)


def test_summary_distinguishes_mass_amount_and_reward_scale():
    actions = fixed_actions(2)["baseline"]
    result = summarize_case(toy_result(actions), actions, gamma=.9)
    assert result["raw_return"] == pytest.approx(10.01)
    assert result["discounted_return"] == pytest.approx(10.009)
    assert result["illustrative_1pct_terminal_pha_repeat_reward_units"] == pytest.approx(.5)
    assert result["endpoint_pha_mass_g_l"] == .008609
    assert result["endpoint_pha_repeat_mmol_l"] == .1
    assert result["internal_cpu_optimizer_runs"] is None
    assert result["first_step_fraction_of_absolute_reward_sum"] > .99


@pytest.mark.parametrize("change", ["partial", "failed_stage", "gpu", "counter", "nan", "units", "action", "termination"])
def test_summary_rejects_incompatible_results(change):
    actions = fixed_actions(2)["baseline"]
    result = toy_result(actions)
    if change == "partial": result["steps"] = 1
    elif change == "failed_stage": result["trajectory"][0]["status"] = "optimal; parsimonious_failed=error"
    elif change == "gpu": result["gpu_lp_stage_calls"] = 1
    elif change == "counter": result["cpu_lp_stage_calls"] = 5
    elif change == "nan": result["trajectory"][1]["reward_raw"] = np.nan
    elif change == "units": result["telemetry_units"]["pha_repeat_mmol_l"] = "g/L"
    elif change == "action": result["trajectory"][0]["action"][3] = .9
    elif change == "termination": result["trajectory"][1]["terminated"] = True
    with pytest.raises(ValueError): summarize_case(result, actions)


def test_pairwise_illustrative_scale_not_universal_error_bound():
    actions = fixed_actions(2)["baseline"]
    a = summarize_case(toy_result(actions), actions)
    b = copy.deepcopy(a)
    b["raw_return"] += .01
    b["discounted_return"] += .009
    result = pairwise_margins({"baseline": a, "perturbation": b})[0]
    assert result["absolute_raw_return_margin"] == pytest.approx(.01)
    assert result["illustrative_scale_over_absolute_raw_margin"]["baseline"] == pytest.approx(50.)
    equal = pairwise_margins({"a": a, "b": a})[0]
    assert equal["illustrative_scale_over_absolute_raw_margin"]["a"] is None
    b["model_fingerprints"]["PF"] = "different"
    with pytest.raises(ValueError): pairwise_margins({"a": a, "b": b})


def test_zero_reward_concentration_is_undefined_not_divide_by_zero():
    actions = fixed_actions(2)["baseline"]
    result = toy_result(actions)
    for row in result["trajectory"]: row["reward_raw"] = 0.
    assert summarize_case(result, actions)["first_step_fraction_of_absolute_reward_sum"] is None
