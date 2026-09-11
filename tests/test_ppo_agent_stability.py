"""CPU-only PPO plumbing, not evidence of GEM policy convergence."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from src.ppo_agent import ConsortiumPPOAgent


class ToyControl(gym.Env):
    observation_space = gym.spaces.Box(0., 1., (20,), dtype=np.float32)
    action_space = gym.spaces.Box(0., 1., (5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(20, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.full(20, self.t / 8, dtype=np.float32), float(1.-np.mean(action)), False, self.t == 8, {}


@pytest.mark.parametrize('gamma', [.95, .99, .999])
def test_normalization_horizon_matches_ppo_and_cpu_update_is_finite(gamma):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    agent = ConsortiumPPOAgent(ToyControl(), gamma=gamma, n_steps=8, batch_size=8,
                              n_epochs=1, device='cpu', verbose=0, target_kl=.03)
    try:
        assert agent.env.gamma == agent.model.gamma == gamma
        assert agent.model.target_kl == .03
        agent.model.learn(total_timesteps=8)
        assert agent.model.num_timesteps == 8
        assert all(torch.isfinite(parameter).all() for parameter in agent.model.policy.parameters())
        assert np.isfinite(agent.env.ret_rms.var).all()
    finally:
        agent.env.close()
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('kwargs', [dict(gamma=-.1), dict(gamma=np.nan),
    dict(gamma=1.1), dict(target_kl=0.), dict(target_kl=-1.), dict(target_kl=np.inf)])
def test_invalid_stability_parameters_fail_before_constructing_env(kwargs):
    def forbidden():
        raise AssertionError('Must validate before environment creation')
    with pytest.raises(ValueError):
        ConsortiumPPOAgent(forbidden, **kwargs)


class MeasuredToy(ToyControl):
    initial_rubber = 100.

    def step(self, action):
        self.t += 1
        return (np.full(20, self.t / 2, dtype=np.float32), 1., False, self.t == 2,
                dict(rubber_remaining=100. - 10. * self.t,
                     total_rubber_degraded=10. * self.t))


class ContractToy(MeasuredToy):
    def __init__(self, schema='audited_rates_v2'):
        self.schema = schema

    def get_policy_contract(self):
        return dict(control=dict(schema=self.schema, numerics_version='audited_cultivation_v2'))


@pytest.fixture
def agents():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    instances = []
    def create(env):
        agent = ConsortiumPPOAgent(env, n_steps=2, batch_size=2, n_epochs=1, verbose=0)
        instances.append(agent)
        return agent
    yield create
    for agent in instances:
        agent.env.close()
    torch.set_num_threads(old_threads)


def test_evaluation_freezes_statistics_uses_raw_reward_and_true_initial_rubber(agents):
    agent = agents(MeasuredToy())
    before = (agent.env.obs_rms.count, agent.env.ret_rms.count,
              agent.env.obs_rms.mean.copy(), agent.env.obs_rms.var.copy())
    result = agent.evaluate(2)
    assert result['mean_reward'] == 2.
    assert result['mean_degradation'] == pytest.approx(.2)
    assert agent.env.obs_rms.count == before[0]
    assert agent.env.ret_rms.count == before[1]
    np.testing.assert_array_equal(agent.env.obs_rms.mean, before[2])
    np.testing.assert_array_equal(agent.env.obs_rms.var, before[3])
    assert agent.env.training is True and agent.env.norm_reward is True


def test_evaluation_restores_modes_even_if_inference_fails(agents, monkeypatch):
    agent = agents(MeasuredToy())
    def fail(*args, **kwargs):
        raise RuntimeError('inference failed')
    monkeypatch.setattr(agent, 'predict', fail)
    with pytest.raises(RuntimeError, match='inference failed'):
        agent.evaluate(1)
    assert agent.env.training is True and agent.env.norm_reward is True


def test_load_requires_normalizer_and_rejects_different_physical_contract(agents, tmp_path):
    source = agents(ContractToy())
    path = tmp_path / 'controller.zip'
    source.save(str(path))
    target = agents(ContractToy())
    target.load(str(path))
    incompatible = agents(ContractToy('legacy_v1'))
    with pytest.raises(ValueError, match='contract differs'):
        incompatible.load(str(path))
    path.with_suffix('.pkl').unlink()
    with pytest.raises(FileNotFoundError, match='VecNormalize'):
        target.load(str(path))


def test_audited_policy_rejects_legacy_zip_even_with_matching_dimensions(agents, tmp_path):
    legacy = agents(MeasuredToy())
    path = tmp_path / 'legacy.zip'
    legacy.save(str(path))
    audited = agents(ContractToy())
    with pytest.raises(ValueError, match='no cultivation contract'):
        audited.load(str(path))


def test_raw_prediction_applies_saved_normalization_without_updating_it(agents, monkeypatch):
    agent = agents(MeasuredToy())
    agent.env.obs_rms.mean[:] = .2
    agent.env.obs_rms.var[:] = .25
    before = agent.env.obs_rms.count
    seen = []
    def capture(observation, deterministic):
        seen.append(observation.copy())
        return np.zeros(5), None
    monkeypatch.setattr(agent.model, 'predict', capture)
    raw = np.full(20, .7, dtype=np.float32)
    agent.predict_raw(raw)
    np.testing.assert_allclose(seen[0], np.ones(20), atol=1e-6)
    assert agent.env.obs_rms.count == before


class BiologicalContractToy(ContractToy):
    def __init__(self, nitrogen_half_saturation=.1):
        super().__init__()
        self.contract = dict(control=dict(schema=self.schema, numerics_version='audited_cultivation_v2'),
                             biological_parameters=dict(nitrogen_half_saturation=nitrogen_half_saturation))

    def get_policy_contract(self):
        return self.contract


def test_checkpoint_rejects_changed_biology_under_the_same_version_and_dimensions(agents, tmp_path):
    source = agents(BiologicalContractToy(.1))
    path = tmp_path / 'audited.zip'
    source.save(str(path))
    target = agents(BiologicalContractToy(.2))
    with pytest.raises(ValueError, match='contract differs'):
        target.load(str(path))


def test_mutable_environment_metadata_cannot_change_saved_policy_contract(agents, tmp_path):
    plant = BiologicalContractToy(.1)
    agent = agents(plant)
    plant.contract['biological_parameters']['nitrogen_half_saturation'] = .2
    assert agent.policy_contract['biological_parameters']['nitrogen_half_saturation'] == .1
    for operation in [lambda: agent.save(str(tmp_path / 'changed.zip')),
                      lambda: agent.evaluate(1)]:
        with pytest.raises(ValueError, match='configuration changed'):
            operation()
    assert not (tmp_path / 'changed.zip').exists()
