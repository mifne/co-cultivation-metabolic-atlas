"""Evaluation paths must share physical rewards and retain terminal timestamps."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from src.ppo_agent import ConsortiumPPOAgent
from src.callbacks import ConsortiumCallback
from scripts.analysis.compare_rl_vs_baseline import run_simulation


class TimedPlant(gym.Env):
    observation_space = gym.spaces.Box(0., 1., (1,), dtype=np.float32)
    action_space = gym.spaces.Box(0., 1., (5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return (np.array([self.t / 2.], dtype=np.float32), 1., False, self.t == 2,
                dict(time_h=self.t * .2, rubber_remaining=10. - self.t, total_pha=self.t))


def test_rl_comparison_freezes_normalizer_and_matches_baseline_time_and_reward():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    agent = ConsortiumPPOAgent(TimedPlant(), n_steps=2, batch_size=2, n_epochs=1, verbose=0)
    try:
        before = agent.env.obs_rms.count
        learned = run_simulation(None, agent=agent)
        fixed = run_simulation(TimedPlant(), static_action=np.zeros(5))
        assert learned == fixed
        assert learned['time'] == [.2, .4]
        assert learned['reward'] == [1., 1.]
        assert agent.env.obs_rms.count == before
        assert agent.env.training is True and agent.env.norm_reward is True
        # VecEnv's live plant is back at t=0, but its terminal result is .4 h.
        assert agent.env.get_attr('t') == [0]
    finally:
        agent.env.close()
        torch.set_num_threads(old_threads)


def test_callback_records_pf_and_full_species_telemetry():
    records = {}
    callback = ConsortiumCallback()
    callback.model = SimpleNamespace(logger=SimpleNamespace(record=lambda k, v: records.__setitem__(k, v)))
    callback.locals = dict(infos=[dict(biomass_pf=.03,
        biomass_g_l={'Propionibacterium_freudenreichii': .03, 'NS21': .1}, pha_g_l=.2)])
    assert callback._on_step() is True
    assert records['Science/Biomass_Pf'] == .03
    assert records['Science/Biomass/Propionibacterium_freudenreichii'] == .03
    assert records['Science/Biomass/NS21'] == .1
    assert records['Science/PHA_g_l'] == .2
