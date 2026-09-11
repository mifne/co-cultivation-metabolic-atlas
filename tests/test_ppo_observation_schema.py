"""Pure observation tests: no GEM loading, LP optimization, GPU, or training."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.rl_environment import ConsortiumEnv


def simulator():
    state = SimpleNamespace(
        species={name: SimpleNamespace(biomass=mass, pha_accumulated=.02)
                 for name, mass in (("OR16", .5), ("NS21", .1), ("PF", .1))},
        metabolites={"h_e": 10**(3-6.5), "o2_e": .1, "glc__D_e": .5,
                     "C30_oligo_e": .02, "odtd_e": .03, "biosurfactant_e": 0.,
                     "arg__L_e": 1., "trp__L_e": 1., "leu__L_e": 1.,
                     "nh4_e": .05, "lac__L_e": .25, "ppa_e": .5},
        rubber_concentration=99.9, time=1.6)
    return SimpleNamespace(state=state, dt=.2, models={name: object() for name in state.species})


def legacy_formula(state, max_time):
    def norm(value, k):
        return np.clip(np.log1p(max(0, value))/np.log1p(k), 0., 1.)
    m = state.metabolites
    values = [np.clip(state.species[name].biomass / 20., 0., 1.) for name in sorted(state.species)]
    values.extend([np.clip(-np.log10(max(1e-12, m['h_e'])/1000.)/14., 0., 1.),
                   np.clip(m['o2_e']/.25, 0., 1.), norm(m['glc__D_e'], 100.),
                   np.clip(state.rubber_concentration/100., 0., 1.), norm(m['C30_oligo_e'], 300.),
                   norm(m['odtd_e'], 500.), norm(sum(s.pha_accumulated for s in state.species.values()), 5000.),
                   norm(m['biosurfactant_e'], 50.), norm(m['arg__L_e'], 100.),
                   norm(m['trp__L_e'], 100.), norm(m['leu__L_e'], 100.),
                   np.clip(state.time/max_time, 0., 1.), float(state.time >= 48.)])
    return np.array(values, dtype=np.float32)


def test_default_is_legacy_bitwise_prefix_compatible():
    sim = simulator()
    legacy = ConsortiumEnv(sim, max_time=24.)
    explicit = ConsortiumEnv(sim, max_time=24., observation_schema="legacy_v1")
    metabolic = ConsortiumEnv(sim, max_time=24., observation_schema="metabolic_v2")
    expected = legacy_formula(sim.state, 24.)
    np.testing.assert_array_equal(legacy._get_observation(sim.state), expected)
    np.testing.assert_array_equal(explicit._get_observation(sim.state), expected)
    np.testing.assert_array_equal(metabolic._get_observation(sim.state)[:16], expected)
    assert legacy.observation_space.shape == (16,)
    assert metabolic.observation_space.shape == (20,)
    assert metabolic.observation_space.contains(metabolic._get_observation(sim.state))
    assert not legacy.observation_schema_metadata["requires_new_policy_and_normalizer"]


def test_feature_metadata_units_order_and_checkpoint_requirement():
    env = ConsortiumEnv(simulator(), observation_schema="metabolic_v2")
    metadata = env.observation_schema_metadata
    assert env.observation_feature_names[-4:] == ("nh4_e", "lac__L_e", "ppa_e", "nh4_below_0p1_mmol_l")
    assert metadata["ordered_feature_names"] == [f["name"] for f in metadata["features"]]
    assert metadata["dimension"] == 20 and metadata["legacy_prefix_dimension"] == 16
    assert [f["input_unit"] for f in metadata["features"][-4:]] == ["mmol/L"] * 4
    assert [f["scale"] for f in metadata["features"][-4:]] == [.1, 1., 1., .1]
    assert metadata["requires_new_policy_and_normalizer"]
    assert metadata["species_order"] == ["NS21", "OR16", "PF"]
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("concentration,flag", [(.099999, 1.), (.1, 0.), (.100001, 0.)])
def test_nh4_threshold_is_strict_and_continuous_feature_is_monotone(concentration, flag):
    sim = simulator()
    sim.state.metabolites["nh4_e"] = concentration
    env = ConsortiumEnv(sim, observation_schema="metabolic_v2")
    obs = env._get_observation(sim.state)
    t = np.log1p(concentration/.1)
    assert obs[16] == pytest.approx(t/(1.+t), abs=1e-7)
    assert obs[19] == flag


def test_added_encoding_zero_huge_finite_and_monotonic():
    sim = simulator()
    env = ConsortiumEnv(sim, observation_schema="metabolic_v2")
    values = []
    for concentration in (0., 1e-20, 1e-9, .01, .1, 1., 1e100, np.finfo(np.float64).max):
        for key in ("nh4_e", "lac__L_e", "ppa_e"):
            sim.state.metabolites[key] = concentration
        obs = env._get_observation(sim.state)
        assert np.isfinite(obs).all() and env.observation_space.contains(obs)
        values.append(obs[16:19].copy())
    np.testing.assert_array_equal(values[0], [0., 0., 0.])
    assert np.all(np.diff(np.asarray(values), axis=0) > 0.)


@pytest.mark.parametrize("key", ["nh4_e", "lac__L_e", "ppa_e"])
@pytest.mark.parametrize("value", [-.01, float("nan"), float("inf"), "0.1", True, [.1], None])
def test_added_features_reject_bad_pool_without_clamping(key, value):
    sim = simulator()
    sim.state.metabolites[key] = value
    env = ConsortiumEnv(sim, observation_schema="metabolic_v2")
    with pytest.raises(ValueError): env._get_observation(sim.state)


@pytest.mark.parametrize("key", ["nh4_e", "lac__L_e", "ppa_e"])
def test_missing_added_feature_fails_closed_but_legacy_is_unchanged(key):
    sim = simulator()
    del sim.state.metabolites[key]
    assert ConsortiumEnv(sim)._get_observation(sim.state).shape == (16,)
    env = ConsortiumEnv(sim, observation_schema="metabolic_v2")
    with pytest.raises(ValueError): env._get_observation(sim.state)


def test_unknown_schema_rejected_without_running_simulator():
    with pytest.raises(ValueError): ConsortiumEnv(simulator(), observation_schema="metabolic_v3")


def test_sensor_subclass_metadata_matches_unchanged_four_dimensional_mask():
    from src.rl_environment_pomdp import RealWorldConsortiumEnv
    sim = simulator()
    env = RealWorldConsortiumEnv(sim, max_time=24.)
    expected = np.asarray([
        sum(s.biomass for s in sim.state.species.values()) / 60.,
        -np.log10(max(1e-12, sim.state.metabolites["h_e"]) / 1000.) / 14.,
        sim.state.metabolites["o2_e"] / .25, sim.state.time / 24.,
    ], dtype=np.float32)
    np.testing.assert_array_equal(env._get_observation(sim.state), expected)
    assert env.observation_space.shape == (4,)
    assert env.observation_space.contains(expected)
    assert env.observation_schema == "sensor_v1"
    metadata = env.observation_schema_metadata
    assert metadata["dimension"] == 4 and metadata["schema"] == "sensor_v1"
    assert metadata["ordered_feature_names"] == ["total_biomass", "ph", "o2_e", "time"]
    assert env.observation_feature_names == tuple(metadata["ordered_feature_names"])
    assert [f["input_unit"] for f in metadata["features"]] == ["g/L", "pH", "mmol/L", "h"]
    assert [f["scale"] for f in metadata["features"]] == [60., 14., .25, 24.]
    assert not metadata["requires_new_policy_and_normalizer"]
    assert not any("nh4" in name for name in metadata["ordered_feature_names"])
    json.dumps(metadata, allow_nan=False)
