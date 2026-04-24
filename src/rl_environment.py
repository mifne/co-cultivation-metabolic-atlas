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
        max_steps: int = 200,  # 100 -> 200 に増加
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
        
        # 前ステップのゴム濃度を記録（増分報酬用）
        self.last_rubber_concentration = self.initial_rubber
        """
        Args:
            simulator: dFBAシミュレーター
            max_steps: 最大ステップ数
            target_rubber_degradation: 目標ゴム分解率
            amino_acid_cost: アミノ酸コスト係数
        """
        # 種数を動的に取得
        self.n_species = len(simulator.state.species)
        self.species_names = list(simulator.state.species.keys())
        
        # 状態空間: n_species * 2 + 4 + 1 次元
        # (バイオマス×n + 増殖速度×n + 代謝物×4 + ゴム×1)
        obs_dim = self.n_species * 2 + 5  # biomass + growth_rate + 4 metabolites + rubber
        
        # 【修正】観測空間の上限を調整（バイオマス上限を20に）
        self.observation_space = spaces.Box(
            low=0.0,
            high=20.0,  # 50 -> 20（バイオマス上限に合わせる）
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
        
        # 【修正】ゴム分解速度を100倍に引き上げ（学習シグナルを得るため）
        self.rubber_degradation_rates = {}
        for species_name in self.species_names:
            # デフォルト値を設定（種名に応じて調整可能）
            if 'Sphingobium' in species_name or 'Gordonia' in species_name:
                self.rubber_degradation_rates[species_name] = 0.005  # LCP分解菌（0.00005 -> 0.005）
            elif 'Pseudomonas' in species_name or 'Cupriavidus' in species_name:
                self.rubber_degradation_rates[species_name] = 0.003  # PHA蓄積菌（0.00003 -> 0.003）
            else:
                self.rubber_degradation_rates[species_name] = 0.002  # 安定化菌（0.00002 -> 0.002）
        
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
        
        # シミュレーターの状態を完全にリセット
        self.current_step = 0
        
        # 初期ゴム濃度を保存（最初のreset時のみ）
        if not hasattr(self, '_initial_rubber_saved'):
            self._initial_rubber_saved = True
            # 初期ゴム濃度を100倍に設定（エピソード長をさらに延長）
            self.initial_rubber = self.simulator.state.rubber_concentration * 100.0
        
        # 前ステップのゴム濃度をリセット
        self.last_rubber_concentration = self.initial_rubber
        
        # 状態を初期値にリセット
        self.simulator.state.time = 0.0
        self.simulator.state.rubber_concentration = self.initial_rubber
        
        # バイオマスを初期値にリセット
        for species_name in self.simulator.state.species.keys():
            species_state = self.simulator.state.species[species_name]
            
            # 初期バイオマスを復元
            if 'Sphingobium' in species_name:
                species_state.biomass = 0.05
            elif 'Pseudomonas' in species_name:
                species_state.biomass = 0.1
            elif 'Lactobacillus' in species_name:
                species_state.biomass = 0.15
            else:
                species_state.biomass = 0.1
            
            species_state.growth_rate = 0.0
            species_state.metabolite_uptake = {}
            species_state.metabolite_secretion = {}
        
        # 代謝物濃度を初期値にリセット
        initial_metabolites = {
            'glc__D_e': 20.0,
            'arg__L_e': 2.0,
            'trp__L_e': 1.0,
            'leu__L_e': 1.5,
            'ala__L_e': 1.0,
            'asn__L_e': 1.0,
            'asp__L_e': 1.0,
            'cys__L_e': 0.5,
            'gln__L_e': 1.0,
            'glu__L_e': 1.0,
            'gly_e': 1.0,
            'his__L_e': 0.5,
            'ile__L_e': 1.0,
            'lys__L_e': 1.0,
            'met__L_e': 0.5,
            'phe__L_e': 0.8,
            'pro__L_e': 1.0,
            'ser__L_e': 1.0,
            'thr__L_e': 1.0,
            'tyr__L_e': 0.5,
            'val__L_e': 1.0,
            'nh4_e': 20.0,
            'pi_e': 10.0,
            'so4_e': 5.0,
            'o2_e': 21.0,
            'fe2_e': 0.01,
            'fe3_e': 0.01,
            'ca2_e': 0.5,
            'cl_e': 1.0,
            'co2_e': 1.0,
            'cu2_e': 0.001,
            'h_e': 0.0001,
            'h2o_e': 55000.0,
            'k_e': 5.0,
            'mg2_e': 2.0,
            'mn2_e': 0.01,
            'mobd_e': 0.001,
            'na1_e': 10.0,
            'zn2_e': 0.01,
            'thm_e': 0.01,
            'ribflv_e': 0.01,
            'isoprene': 0.0,
            'ac_e': 0.1,
            'lac__D_e': 0.0,
            'lac__L_e': 0.0,
        }
        self.simulator.state.metabolites = initial_metabolites.copy()
        
        # 培地条件を再初期化
        self.simulator._initialize_medium()
        
        obs = self.simulator.get_state_vector()
        
        # 観測値の妥当性をチェック
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"⚠️  リセット時に無効な観測値を検出: {obs}")
            obs = np.nan_to_num(obs, nan=0.1, posinf=100.0, neginf=0.0)
        
        # デバッグ出力（最初の3回のみ）
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        
        if self._reset_count < 3:
            print(f"\n  🔄 環境リセット #{self._reset_count + 1}:")
            print(f"    ゴム濃度: {self.simulator.state.rubber_concentration:.4f} g/L")
            total_biomass = sum(s.biomass for s in self.simulator.state.species.values())
            print(f"    総バイオマス: {total_biomass:.4f} g/L")
            print(f"    観測ベクトル: {obs[:5]}... (最初の5要素)")
        
        self._reset_count += 1
        
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
        
        # デバッグ出力（最初の10ステップ）
        if self.current_step < 10:
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
        
        # デバッグ出力（最初の10ステップ + 10ステップごと）
        should_log = self.current_step < 10 or self.current_step % 10 == 0
        if should_log:
            rubber_degraded_ratio = 1 - (state.rubber_concentration / self.initial_rubber)
            total_biomass = sum(s.biomass for s in state.species.values())
            rubber_degraded_this_step = self.last_rubber_concentration - state.rubber_concentration
            
            print(f"    ゴム残量: {state.rubber_concentration:.4f} g/L ({rubber_degraded_ratio*100:.2f}% 分解)")
            print(f"    今回分解量: {rubber_degraded_this_step:.6f} g/L")
            print(f"    総バイオマス: {total_biomass:.4f} g/L")
            print(f"    報酬: {reward:.2f}")
            
            # 各種のバイオマスと増殖速度を表示
            for species_name, species_state in state.species.items():
                print(f"      {species_name[:20]}: biomass={species_state.biomass:.4f} g/L, μ={species_state.growth_rate:.4f} 1/h")
            
            # 主要代謝物濃度を表示
            key_metabolites = ['glc__D_e', 'arg__L_e', 'trp__L_e', 'leu__L_e', 'nh4_e', 'pi_e', 'o2_e']
            met_status = []
            for met_id in key_metabolites:
                conc = state.metabolites.get(met_id, 0.0)
                met_status.append(f"{met_id.replace('__L_e', '').replace('__D_e', '').replace('_e', '')}={conc:.2f}")
            print(f"      代謝物: {', '.join(met_status)} mM")
        
        # 終了条件
        self.current_step += 1
        
        # ゴム分解率による終了
        rubber_degraded_ratio = 1 - (state.rubber_concentration / self.initial_rubber)
        terminated = rubber_degraded_ratio >= self.target_rubber_degradation
        
        # 最大ステップ数による打ち切り
        truncated = self.current_step >= self.max_steps
        
        # バイオマスが全滅した場合も終了
        total_biomass = sum(s.biomass for s in state.species.values())
        if total_biomass < 0.05:  # 全バイオマスが0.05 g/L未満（0.01 -> 0.05）
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
        報酬関数（増分ベース + 多様性強化 + FBA失敗ペナルティ）
        
        Args:
            state: コンソーシアム状態
            action: 実行したアクション
        
        Returns:
            報酬値
        """
        # 【修正】FBA完全失敗時の早期リターン
        growth_rates = [s.growth_rate for s in state.species.values()]
        if all(g == 0.0 for g in growth_rates):
            # 全種が増殖していない = FBA失敗
            return -50.0
        
        # 1. ゴム分解報酬（増分ベース）- 前ステップからの分解量
        current_rubber = state.rubber_concentration
        rubber_degraded_this_step = self.last_rubber_concentration - current_rubber
        
        # 増分を正規化（初期量に対する割合）
        degradation_increment = rubber_degraded_this_step / self.initial_rubber
        
        # 増分報酬（0-10点程度）
        degradation_reward = degradation_increment * 1000  # スケール調整
        
        # 前ステップの値を更新
        self.last_rubber_concentration = current_rubber
        
        # 2. バイオマス多様性報酬（Shannon多様性指数 + 独占ペナルティ）
        biomasses = np.array([s.biomass for s in state.species.values()])
        total_biomass = biomasses.sum()
        
        # 【修正】バイオマス不足または増殖停止時のペナルティ
        avg_growth = np.mean([max(0, g) for g in growth_rates])
        if total_biomass < 0.01 or avg_growth < 0.001:
            diversity_reward = -10.0  # バイオマス不足または増殖停止
        else:
            # 【修正】独占ペナルティ: 特定の種が上限（20 g/L）に達している場合
            monopoly_penalty = 0.0
            for biomass in biomasses:
                if biomass > 10.0:  # 10 g/L を超えたら独占とみなす（上限20に合わせて調整）
                    monopoly_penalty -= (biomass - 10.0) * 3.0  # ペナルティをさらに強化
            
            proportions = biomasses / total_biomass
            proportions = proportions[proportions > 1e-6]  # 極小値を除去
            if len(proportions) > 0:
                diversity = -np.sum(proportions * np.log(proportions + 1e-10))
                # 3種の場合、最大多様性はlog(3) ≈ 1.099
                max_diversity = np.log(len(self.species_names))
                diversity_reward = (diversity / max_diversity) * 30  # 0-30点（強化）
            else:
                diversity_reward = 0
            
            diversity_reward += monopoly_penalty
        
        # 3. 増殖速度報酬（全種が増殖していることを奨励）
        growth_reward = avg_growth * 5  # 0-5点程度
        
        # 4. アミノ酸コストペナルティ（使用量に応じて）
        amino_acid_penalty = -self.amino_acid_cost * np.sum(action)
        
        # 5. ボーナス: 目標達成（累積分解率で判定）
        total_degradation_ratio = (self.initial_rubber - current_rubber) / self.initial_rubber
        if total_degradation_ratio >= self.target_rubber_degradation:
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
