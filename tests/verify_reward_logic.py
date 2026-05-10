import unittest
import numpy as np
from src.rl_environment import ConsortiumEnv

class MockSimulator:
    def __init__(self):
        self.dt = 0.2
        self.models = {'S1': 1, 'S2': 1, 'S3': 1}
        self.state = type('obj', (object,), {
            'time': 0.0,
            'metabolites': {'h_e': 1000 * 10**(-7.0), 'glc__D_e': 0.0, 'biosurfactant_e': 0.0},
            'species': {
                'S1': type('obj', (object,), {'biomass': 1.0, 'pha_accumulated': 0.0, 'metabolite_uptake': {}, 'metabolite_secretion': {}}),
                'S2': type('obj', (object,), {'biomass': 1.0, 'pha_accumulated': 0.0, 'metabolite_uptake': {}, 'metabolite_secretion': {}}),
                'S3': type('obj', (object,), {'biomass': 1.0, 'pha_accumulated': 0.0, 'metabolite_uptake': {}, 'metabolite_secretion': {}})
            },
            'rubber_concentration': 100.0
        })
    def step(self, *args, **kwargs):
        return self.state
    def get_state_vector(self):
        return np.zeros(14)

class TestRewardLogic(unittest.TestCase):
    def setUp(self):
        self.env = ConsortiumEnv(simulator=MockSimulator())
        self.env.total_timesteps = 1000000
        self.env.current_total_steps = 0
        self.env.last_activity_biomass = 0.0

    def get_reward_at(self, ph, total_biomass):
        h_conc = 1000 * 10**(-ph)
        self.env.simulator.state.metabolites['h_e'] = h_conc
        for name in self.env.simulator.state.species:
            self.env.simulator.state.species[name].biomass = total_biomass / 3.0
        
        # Reset previous states to isolate this step
        self.env.prev_rubber = 100.0
        self.env.last_total_pha = 0.0
        self.env.last_bs = 0.0
        
        _, reward, _, _, _ = self.env.step(np.zeros(5))
        return reward

    def test_ph_penalty_trend(self):
        r7 = self.get_reward_at(7.0, 3.0)
        r8 = self.get_reward_at(8.0, 3.0)
        r10 = self.get_reward_at(10.0, 3.0)
        
        print(f"\n[Reward Test] pH 7.0: {r7:.4f}")
        print(f"[Reward Test] pH 8.0: {r8:.4f}")
        print(f"[Reward Test] pH 10.0: {r10:.4f}")
        
        self.assertGreater(r7, r8)
        self.assertGreater(r8, r10)
        # Check if penalty accelerates (Quadratic)
        # |r10 - r8| = |(0.1*9) - (0.1*1)| = 0.8
        # |r8 - r7| = |(0.1*1) - (0)| = 0.1
        self.assertGreater(abs(r10 - r8), abs(r8 - r7))

    def test_biomass_bonus(self):
        r_low = self.get_reward_at(7.0, 0.3)
        r_high = self.get_reward_at(7.0, 15.0)
        
        print(f"[Reward Test] Biomass 0.3: {r_low:.4f}")
        print(f"[Reward Test] Biomass 15.0: {r_high:.4f}")
        
        self.assertGreater(r_high, r_low)
        # Difference should be (15.0 - 0.3) * 0.02 = 14.7 * 0.02 = 0.294
        self.assertAlmostEqual(r_high - r_low, 0.294, places=3)

if __name__ == '__main__':
    unittest.main()
