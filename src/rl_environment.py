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
    
    State: [biomass_1, ..., biomass_n, growth_1, ..., growth_n, 
            isoprene, arg, trp, leu, rubber]
    Action: [arg_supplement, trp_supplement, leu_supplement] (連続値 0-1)
    
    Note: 種数nは動的に決定される（通常3種）
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        simulator: dFBASimulator,
        max_steps: int = 100,
        target_rubber_degradation: float = 0.8,  # 80%分解を目標（より現実的）
        amino_acid_cost: float = 0.05  # コストを下げて学習を促進
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
        
        # 種数を動的に取得
        self.n_species = len(simulator.state.species)
        self.species_names = list(simulator.state.species.keys())
        
        # 状態空間: n_species * 2 + 4 + 1 次元
        # (バイオマス×n + 増殖速度×n + 代謝物×4 + ゴム×1)
        obs_dim = self.n_species * 2 + 5  # biomass + growth_rate + 4 metabolites + rubber
        
        # 観測空間の上限を設定（NaN/Inf防止）
        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,  # 全ての値を0-100の範囲に制限
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 行動空間: 3次元（各アミノ酸の補給量 0-1）
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # ゴム分解速度（簡略化: 種名に基づいて動的に設定）
        self.rubber_degradation_rates = {}
        for species_name in self.species_names:
            # デフォルト値を設定（種名に応じて調整可能）
            if 'Sphingobium' in species_name or 'Gordonia' in species_name:
                self.rubber_degradation_rates[species_name] = 0.05  # LCP分解菌
            elif 'Pseudomonas' in species_name or 'Cupriavidus' in species_name:
                self.rubber_degradation_rates[species_name] = 0.03  # PHA蓄積菌
            else:
                self.rubber_degradation_rates[species_name] = 0.02  # 安定化菌
        
        print(f"  🔬 環境設定:")
        print(f"    種数: {self.n_species}")
        print(f"    観測空間: {obs_dim}次元")
        print(f"    行動空間: 3次元")
        print(f"    ゴム分解速度: {self.rubber_degradation_rates}")
    
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
        
        # 観測値の妥当性をチェック
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"⚠️  リセット時に無効な観測値を検出: {obs}")
            # 安全な初期値で置き換え
            obs = np.nan_to_num(obs, nan=0.1, posinf=100.0, neginf=0.0)
        
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
        
        # デバッグ出力（最初の数ステップのみ）
        if self.current_step < 3:
            print(f"\n  🎮 ステップ{self.current_step}: アクション={action}")
            print(f"    アミノ酸補給: Arg={amino_acid_supplementation['arginine']:.2f}, "
                  f"Trp={amino_acid_supplementation['tryptophan']:.2f}, "
                  f"Leu={amino_acid_supplementation['leucine']:.2f} mM")
        
        # シミュレーションステップ
        state = self.simulator.step(
            rubber_degradation_rates=self.rubber_degradation_rates,
            amino_acid_supplementation=amino_acid_supplementation
        )
        
        # 報酬計算
        reward = self._calculate_reward(state, action)
        
        # デバッグ出力（最初の数ステップのみ）
        if self.current_step < 3:
            rubber_degraded_ratio = 1 - (state.rubber_concentration / self.initial_rubber)
            total_biomass = sum(s.biomass for s in state.species.values())
            print(f"    ゴム残量: {state.rubber_concentration:.4f} g/L ({rubber_degraded_ratio*100:.2f}% 分解)")
            print(f"    総バイオマス: {total_biomass:.4f} g/L")
            print(f"    報酬: {reward:.2f}")
        
        # 終了条件
        self.current_step += 1
        
        # ゴム分解率による終了
        rubber_degraded_ratio = 1 - (state.rubber_concentration / self.initial_rubber)
        terminated = rubber_degraded_ratio >= self.target_rubber_degradation
        
        # 最大ステップ数による打ち切り
        truncated = self.current_step >= self.max_steps
        
        # バイオマスが全滅した場合も終了（閾値をさらに緩和）
        total_biomass = sum(s.biomass for s in state.species.values())
        if total_biomass < 0.0001:  # 全バイオマスが0.0001 g/L未満（Phase 1: より寛容に）
            terminated = True
            reward = -50.0  # ペナルティ
            print(f"  ⚠️  ステップ{self.current_step}: バイオマス全滅 ({total_biomass:.6f} g/L)")
        
        # 観測
        obs = self.simulator.get_state_vector()
        
        # 観測値の妥当性をチェック
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"⚠️  ステップ{self.current_step}で無効な観測値を検出")
            # 安全な値で置き換え
            obs = np.nan_to_num(obs, nan=0.1, posinf=100.0, neginf=0.0)
            # エピソードを強制終了
            terminated = True
            reward = -100.0  # ペナルティ
        
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
        # 1. ゴム分解報酬（主要報酬）
        rubber_degraded = self.initial_rubber - state.rubber_concentration
        degradation_ratio = rubber_degraded / self.initial_rubber
        degradation_reward = degradation_ratio * 100  # 0-100点
        
        # 2. バイオマス多様性報酬（Shannon多様性指数）
        biomasses = np.array([s.biomass for s in state.species.values()])
        total_biomass = biomasses.sum()
        if total_biomass > 0.01:
            proportions = biomasses / total_biomass
            proportions = proportions[proportions > 1e-6]  # 極小値を除去
            if len(proportions) > 0:
                diversity = -np.sum(proportions * np.log(proportions + 1e-10))
                # 3種の場合、最大多様性はlog(3) ≈ 1.099
                max_diversity = np.log(len(self.species_names))
                diversity_reward = (diversity / max_diversity) * 20  # 0-20点
            else:
                diversity_reward = 0
        else:
            diversity_reward = -10  # バイオマス不足ペナルティ
        
        # 3. 増殖速度報酬（全種が増殖していることを奨励）
        growth_rates = [s.growth_rate for s in state.species.values()]
        avg_growth = np.mean([max(0, g) for g in growth_rates])  # 負の増殖率は0とする
        growth_reward = avg_growth * 5  # 0-5点程度
        
        # 4. アミノ酸コストペナルティ
        amino_acid_penalty = -self.amino_acid_cost * np.sum(action) * 10  # スケール調整
        
        # 5. ボーナス: 目標達成
        if degradation_ratio >= self.target_rubber_degradation:
            bonus = 50.0
        else:
            bonus = 0.0
        
        # 総報酬
        reward = (
            degradation_reward +
            diversity_reward +
            growth_reward +
            amino_acid_penalty +
            bonus
        )
        
        return float(reward)
    
    def render(self):
        """環境の可視化（簡易版）"""
        state = self.simulator.state
        print(f"\n=== Step {self.current_step} (t={state.time:.2f}h) ===")
        print(f"Rubber: {state.rubber_concentration:.3f} g/L")
        for species_name, species_state in state.species.items():
            print(f"{species_name}: Biomass={species_state.biomass:.3f} g/L, "
                  f"Growth={species_state.growth_rate:.3f} 1/h")
        print(f"Metabolites: {state.metabolites}")
