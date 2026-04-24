"""
Reinforcement Learning Environment for Consortium Control
コンソーシアム制御のための強化学習環境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
from src.dfba_simulator import dFBASimulator, ConsortiumState


class ConsortiumEnv(gym.Env):
    """
    天然ゴム分解コンソーシアム制御のためのRL環境
    
    State: [biomass_1, biomass_2, biomass_3, growth_1, growth_2, growth_3, 
            isoprene, arg, trp, leu, rubber]
    Action: [arg_supplement, trp_supplement, leu_supplement] (連続値 0-1)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        simulator: dFBASimulator,
        max_steps: int = 100,
        target_rubber_degradation: float = 0.9,
        amino_acid_cost: float = 0.1
    ):
        """
        Args:
            simulator: dFBAシミュレーター
            max_steps: 最大ステップ数
            target_rubber_degradation: 目標ゴム分解率
            amino_acid_cost: アミノ酸コスト係数
        """
        super().__init__()
        
        self.simulator = simulator
        self.max_steps = max_steps
        self.target_rubber_degradation = target_rubber_degradation
        self.amino_acid_cost = amino_acid_cost
        
        self.current_step = 0
        self.initial_rubber = simulator.state.rubber_concentration
        
        # 状態空間: 11次元（バイオマス×3 + 増殖速度×3 + 代謝物×4 + ゴム×1）
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )
        
        # 行動空間: 3次元（各アミノ酸の補給量 0-1）
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # ゴム分解速度（簡略化: 固定値）
        self.rubber_degradation_rates = {
            'Gordonia': 0.05,  # g/gDW/h
            'Nocardia': 0.03,
            'Rhodococcus': 0.04
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        環境をリセット
        
        Returns:
            初期状態, 情報辞書
        """
        super().reset(seed=seed)
        
        # シミュレーターを初期状態にリセット
        # （実装簡略化のため、新しいシミュレーターインスタンスを想定）
        self.current_step = 0
        
        obs = self.simulator.get_state_vector()
        info = {'time': self.simulator.state.time}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        1ステップ実行
        
        Args:
            action: [arg_supplement, trp_supplement, leu_supplement] (0-1)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # アクションをアミノ酸補給量に変換（0-1 → 0-10 mM）
        max_supplement = 10.0  # mM
        amino_acid_supplementation = {
            'arginine': action[0] * max_supplement,
            'tryptophan': action[1] * max_supplement,
            'leucine': action[2] * max_supplement
        }
        
        # シミュレーションステップ
        state = self.simulator.step(
            rubber_degradation_rates=self.rubber_degradation_rates,
            amino_acid_supplementation=amino_acid_supplementation
        )
        
        # 報酬計算
        reward = self._calculate_reward(state, action)
        
        # 終了条件
        self.current_step += 1
        terminated = (
            state.rubber_concentration < self.initial_rubber * (1 - self.target_rubber_degradation)
        )
        truncated = self.current_step >= self.max_steps
        
        # 観測
        obs = self.simulator.get_state_vector()
        
        info = {
            'time': state.time,
            'rubber_remaining': state.rubber_concentration,
            'total_biomass': sum(s.biomass for s in state.species.values())
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, state: ConsortiumState, action: np.ndarray) -> float:
        """
        報酬関数
        
        Args:
            state: コンソーシアム状態
            action: 実行したアクション
        
        Returns:
            報酬値
        """
        # ゴム分解報酬
        rubber_degraded = self.initial_rubber - state.rubber_concentration
        degradation_reward = rubber_degraded / self.initial_rubber * 100
        
        # バイオマス多様性報酬（Shannon多様性指数）
        biomasses = np.array([s.biomass for s in state.species.values()])
        total_biomass = biomasses.sum()
        if total_biomass > 0:
            proportions = biomasses / total_biomass
            proportions = proportions[proportions > 0]  # ゼロ除去
            diversity = -np.sum(proportions * np.log(proportions))
            diversity_reward = diversity * 10
        else:
            diversity_reward = 0
        
        # アミノ酸コストペナルティ
        amino_acid_penalty = -self.amino_acid_cost * np.sum(action)
        
        # 総報酬
        reward = degradation_reward + diversity_reward + amino_acid_penalty
        
        return reward
    
    def render(self):
        """環境の可視化（簡易版）"""
        state = self.simulator.state
        print(f"\n=== Step {self.current_step} (t={state.time:.2f}h) ===")
        print(f"Rubber: {state.rubber_concentration:.3f} g/L")
        for species_name, species_state in state.species.items():
            print(f"{species_name}: Biomass={species_state.biomass:.3f} g/L, "
                  f"Growth={species_state.growth_rate:.3f} 1/h")
        print(f"Metabolites: {state.metabolites}")
