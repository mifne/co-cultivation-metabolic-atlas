import numpy as np
import pytest

from scripts.analyze_coverage_temporal_routing import _promote, replay_policy


def test_previous_first_can_displace_a_current_learned_hit_at_k1():
    # Row zero certifies candidate 1.  On row one that candidate is stale,
    # while the learned top-1 candidate 0 certifies the current LP.
    coverage = np.array([[False, True, False], [True, False, False]], dtype=bool)
    learned = np.array([[1, 0, 2], [0, 1, 2]], dtype=np.int32)

    previous = replay_policy(coverage, learned, trajectories=1, steps=2,
                             k=1, policy="previous_first")
    learned_first = replay_policy(coverage, learned, trajectories=1, steps=2,
                                  k=1, policy="learned_first")

    assert previous["accepted"].tolist() == [True, False]
    assert previous["selected"].tolist() == [1, -1]
    assert previous["prior"].tolist() == [-1, 1]
    assert not previous["previous_currently_passes"][1]
    assert learned_first["accepted"].tolist() == [True, True]
    assert learned_first["selected"].tolist() == [1, 0]


def test_learned_first_keeps_top1_then_unique_previous_candidate():
    base = np.array([4, 2, 1, 3, 0], dtype=np.int32)
    assert _promote(base, 2, "previous_first").tolist() == [2, 4, 1, 3, 0]
    assert _promote(base, 2, "learned_first").tolist() == [4, 2, 1, 3, 0]
    assert _promote(base, 4, "learned_first").tolist() == base.tolist()
    assert len(np.unique(_promote(base, 1, "learned_first"))) == len(base)


def test_temporal_state_is_isolated_by_trajectory_and_stateless_has_none():
    coverage = np.array([
        [True, False], [False, True],
        [False, True], [True, False],
    ], dtype=bool)
    learned = np.tile(np.array([[0, 1]], dtype=np.int32), (4, 1))

    temporal = replay_policy(coverage, learned, trajectories=2, steps=2,
                             k=1, policy="previous_first")
    stateless = replay_policy(coverage, learned, trajectories=2, steps=2,
                              k=1, policy="stateless_learned")

    # Row two starts a new trajectory and cannot inherit candidate 0 from row 1.
    assert temporal["prior"].tolist() == [-1, 0, -1, -1]
    assert stateless["prior"].tolist() == [-1, -1, -1, -1]
    assert stateless["accepted"].tolist() == [True, False, False, True]


@pytest.mark.parametrize(
    "coverage, learned, trajectories, steps, k, policy",
    [
        (np.ones((2, 2), dtype=np.int8), np.array([[0, 1], [0, 1]]), 1, 2, 1,
         "previous_first"),
        (np.ones((2, 2), dtype=bool), np.array([[0], [0]]), 1, 2, 1,
         "previous_first"),
        (np.ones((2, 2), dtype=bool), np.array([[0, 1], [0, 1]]), 1, 3, 1,
         "previous_first"),
        (np.ones((2, 2), dtype=bool), np.array([[0, 1], [0, 1]]), 1, 2, 3,
         "previous_first"),
        (np.ones((2, 2), dtype=bool), np.array([[0, 1], [0, 1]]), 1, 2, 1,
         "unknown"),
    ],
)
def test_replay_rejects_malformed_inputs(coverage, learned, trajectories, steps, k, policy):
    with pytest.raises(ValueError):
        replay_policy(coverage, learned, trajectories=trajectories,
                      steps=steps, k=k, policy=policy)
