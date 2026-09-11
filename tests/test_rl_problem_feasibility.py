from __future__ import annotations

from scripts.analysis.audit_rl_problem_feasibility import (
    build_verdict,
    maximum_action_medium,
)


def test_maximum_action_medium_exposes_indirect_rescue_but_not_g3ps() -> None:
    medium = maximum_action_medium({})
    assert medium["mnl_e"] == 0.1
    assert medium["ile__L_e"] > 0.0
    assert "g3ps_e" not in medium


def test_verdict_rejects_problem_without_ph_and_nitrogen_observation() -> None:
    static = {
        "baseline": {
            "feasible": False,
            "required_additional_supply": [{"metabolite": "g3ps_e"}],
        },
        "maximum_one_step_action": {"feasible": True},
        "rare_member_screen": {
            "OR16": {"feasible": True},
            "NS21": {"feasible": True},
            "LP": {"feasible": True},
        },
    }
    verdict = build_verdict(static, {"reachable": True}, rollout=None)
    assert not verdict["rl_ready"]
    assert verdict["checks"]["maximum_action_static_growth"]
    assert not verdict["checks"]["direct_ph_control_available"]
    assert not verdict["checks"]["nitrogen_switch_state_observed"]
