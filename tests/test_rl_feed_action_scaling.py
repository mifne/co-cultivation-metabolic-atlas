from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.rl_environment import ConsortiumEnv, SymmetricPPOActionWrapper


def test_rl_feed_scaling_parameters_are_explicit_and_nonnegative() -> None:
    simulator = SimpleNamespace(
        dt=0.2,
        models={"a": object(), "b": object(), "c": object()},
        state=SimpleNamespace(
            species={
                "a": SimpleNamespace(biomass=0.5),
                "b": SimpleNamespace(biomass=0.1),
                "c": SimpleNamespace(biomass=0.1),
            },
            metabolites={},
            rubber_concentration=100.0,
        ),
    )
    env = ConsortiumEnv(
        simulator=simulator,
        max_common_feed_early=0.02,
        max_common_feed_late=0.01,
        max_specific_feed_per_step=0.05,
    )
    assert env.max_common_feed_early == 0.02
    assert env.max_common_feed_late == 0.01
    assert env.max_specific_feed_per_step == 0.05
    assert env.common_feed_action_budget == 0.25
    assert env.excess_common_feed_penalty == 2.0


def test_symmetric_ppo_actions_map_to_physical_controls() -> None:
    mapped = SymmetricPPOActionWrapper.to_physical(
        np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        common_feed_cap=None,
    )
    np.testing.assert_allclose(mapped, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_symmetric_ppo_common_feed_has_operational_safety_cap() -> None:
    mapped = SymmetricPPOActionWrapper.to_physical(
        np.ones(5, dtype=np.float32), common_feed_cap=0.25
    )
    np.testing.assert_allclose(mapped, [1.0, 1.0, 1.0, 0.25, 1.0])
