import numpy as np
from gymnasium import spaces
from src.rl_environment import ConsortiumEnv
from src.dfba_simulator import ConsortiumState

class RealWorldConsortiumEnv(ConsortiumEnv):
    """
    POMDP (Partially Observable Markov Decision Process) Environment
    Masks the 'God-mode' observation space down to only 4 realistic sensors:
    1. Total Biomass (OD600 equivalent)
    2. pH
    3. Dissolved Oxygen (DO)
    4. Time (normalized)
    
    Reward function is inherited directly from ConsortiumEnv v3.0
    (rubber degradation + PHA accumulation as sole objectives).
    No additional reward overrides — the base reward is already correct.
    """
    def __init__(self, simulator, max_time=672.0):
        super().__init__(simulator, max_time)
        
        # Override observation space: Only 4 dimensions
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        # This subclass masks the base observations; do not advertise the
        # inherited 16-dimensional metabolic/state layout for its 4D sensors.
        self.observation_schema = "sensor_v1"
        features = [
            dict(name="total_biomass", source="sum(species.biomass)", input_unit="g/L",
                 encoding="clip(c/scale,0,1)", scale=60.0),
            dict(name="ph", source="metabolites.h_e", input_unit="pH",
                 encoding="clip(pH/scale,0,1)", scale=14.0),
            dict(name="o2_e", source="metabolites.o2_e", input_unit="mmol/L",
                 encoding="clip(c/scale,0,1)", scale=.25),
            dict(name="time", source="time", input_unit="h",
                 encoding="clip(time/scale,0,1)", scale=float(self.max_time)),
        ]
        self.observation_feature_names = tuple(feature["name"] for feature in features)
        self.observation_schema_metadata = dict(
            schema="sensor_v1", dimension=4,
            ordered_feature_names=list(self.observation_feature_names), features=features,
            encoded_unit="dimensionless", species_order=[],
            requires_new_policy_and_normalizer=False,
            checkpoint_note="Existing sensor-only 4D policy and normalizer layout is unchanged",
            scope="Four-sensor POMDP mask; no species-resolved or NH4 observations",
        )

    def _get_observation(self, state: ConsortiumState) -> np.ndarray:
        obs = []
        
        # 1. Total Biomass (OD600 equivalent): 0-60 g/L -> 0-1.0
        total_biomass = sum(s.biomass for s in state.species.values())
        obs.append(np.clip(total_biomass / 60.0, 0.0, 1.0))
        
        # 2. pH: 0-14 -> 0-1.0
        h_conc = state.metabolites.get('h_e', 0.0001)
        current_ph = -np.log10(max(1e-12, h_conc) / 1000.0)
        obs.append(np.clip(current_ph / 14.0, 0.0, 1.0))
        
        # 3. Dissolved Oxygen (DO): 0-0.25 -> 0-1.0
        obs.append(np.clip(state.metabolites.get('o2_e', 0.25) / 0.25, 0.0, 1.0))
        
        # 4. Time (normalized)
        obs.append(np.clip(state.time / self.max_time, 0.0, 1.0))
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Call parent step (uses the new v3.0 reward function)
        obs_full, reward, terminated, truncated, info = super().step(action)
        
        # Re-mask observation to 4 dimensions (parent returns 16-dim)
        obs = self._get_observation(self.simulator.state)
        
        return obs, reward, terminated, truncated, info
