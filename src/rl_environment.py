import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
import cobra
from .dfba_simulator import dFBASimulator, ConsortiumState

class ConsortiumEnv(gym.Env):
    """
    ゴム分解コンソーシアムの強化学習環境 (v2.2)
    FBAソルバーによる厳密な物質収支に基づき、長期培養（672h）を制御する。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        simulator: dFBASimulator,
        max_time: float = 672.0,
    ):
        super().__init__()
        self.simulator = simulator
        self.max_time = max_time
        self.dt = simulator.dt
        
        # 初期状態の保存
        self.initial_biomass = {name: s.biomass for name, s in simulator.state.species.items()}
        self.initial_metabolites = simulator.state.metabolites.copy()
        self.initial_rubber = simulator.state.rubber_concentration
        
        self.last_total_pha = 0.0
        self.last_bs = 0.0
        self.current_step = 0
        
        # カリキュラム学習用のカウンタ
        self.total_timesteps = 500000 
        self.current_total_steps = 0
        
        self._setup_spaces()
        self.rubber_degradation_rates = {}

    def set_total_timesteps(self, total: int):
        self.total_timesteps = total

    def set_current_total_steps(self, steps: int):
        self.current_total_steps = steps

    def _setup_spaces(self):
        num_species = len(self.simulator.models)
        # Biomass(num_species) + pH(1) + DO(1) + Glc(1) + RubberFrags(1) + PHA(1) + BS(1) + AA(3) + Time(1) + Phase(1)
        # 14 if num_species=3
        total_obs_dim = num_species + 11
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

    def _get_observation(self, state: ConsortiumState) -> np.ndarray:
        obs = []
        # Biomass: 0-20 g/L -> 0-1.0
        for name in sorted(state.species.keys()):
            obs.append(np.clip(state.species[name].biomass / 20.0, 0.0, 1.0))
        
        # pH: 0-14 -> 0-1.0
        h_conc = state.metabolites.get('h_e', 0.0001)
        current_ph = -np.log10(max(1e-12, h_conc) / 1000.0)
        obs.append(np.clip(current_ph / 14.0, 0.0, 1.0))
        
        # Dissolved Oxygen: 0-0.25 -> 0-1.0
        obs.append(np.clip(state.metabolites.get('o2_e', 0.25) / 0.25, 0.0, 1.0))
        
        # Metabolites (Log scale normalization): 0-100 mM -> 0-1.0
        # --- 科学的修正: クリップを明示し、K値を想定最大値に設定 ---
        def norm_met(val, K=100.0): 
            return np.clip(np.log1p(max(0, val)) / np.log1p(K), 0.0, 1.0)
        
        obs.append(norm_met(state.metabolites.get('glc__D_e', 0.0), K=100.0))
        obs.append(norm_met(state.metabolites.get('rubber_fragment_e', 0.0), K=100.0))
        
        # PHA: 0-5000 mmol -> 0-1.0 (想定レンジの拡大)
        total_pha = sum(s.pha_accumulated for s in state.species.values())
        obs.append(norm_met(total_pha, K=5000.0))
        
        # Biosurfactant: 0-50 mM -> 0-1.0
        obs.append(norm_met(state.metabolites.get('biosurfactant_e', 0.0), K=50.0))
        
        obs.append(norm_met(state.metabolites.get('arg__L_e', 0.0), K=100.0))
        obs.append(norm_met(state.metabolites.get('trp__L_e', 0.0), K=100.0))
        obs.append(norm_met(state.metabolites.get('leu__L_e', 0.0), K=100.0))
        
        # Time and Phase
        obs.append(np.clip(state.time / self.max_time, 0.0, 1.0))
        phase = 0.0 if state.time < 48.0 else 1.0
        obs.append(phase)
        return np.array(obs, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.last_total_pha = 0.0
        self.last_bs = 0.0
        self.simulator.state.time = 0.0
        self.simulator.state.rubber_concentration = self.initial_rubber
        
        # --- 科学的種培地 (Starter Medium) & 誘導物質シード ---
        self.simulator.state.metabolites = self.initial_metabolites.copy()
        
        # 初期アミノ酸を調整 (1.0 mM): 急激な代謝スパイクを抑えつつ酵素合成を支援
        self.simulator.state.metabolites['arg__L_e'] = 1.0
        self.simulator.state.metabolites['trp__L_e'] = 1.0
        self.simulator.state.metabolites['leu__L_e'] = 1.0
        # 微量のグルコース (0.5 mM) を追加し、エネルギー基盤を安定させる
        self.simulator.state.metabolites['glc__D_e'] = 0.5
        
        # --- 誘導物質 (Inducer) のシード ---
        # 0.1 mM の分解フラグメントを配置し、分解パスウェイを即座に稼働させる
        self.simulator.state.metabolites['rubber_fragment_e'] = 0.1
        
        # 初期pHを確実に中性付近にするため、プロトン濃度を調整
        self.simulator.state.metabolites['h_e'] = 10**(-7.21) * 1000.0 
        # バイオサーファクタントの「種」として微量配置 (0.005 mM)
        self.simulator.state.metabolites['biosurfactant_e'] = 0.005
        
        for name, s in self.simulator.state.species.items():
            s.biomass = self.initial_biomass[name]
            s.growth_rate = 0.0
            s.pha_accumulated = 0.0
            
        self.simulator._initialize_medium()
        self.prev_rubber = self.initial_rubber
        self.last_activity_biomass = sum(s.biomass for s in self.simulator.state.species.values())

        return self._get_observation(self.simulator.state), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_time = self.simulator.state.time
        
        # フェーズに応じた共通栄養源の供給制限
        if current_time < 48.0:
            max_feed_common = 0.5 # 1.0 -> 0.5 (現実的な初期ブースト)
        else:
            max_feed_common = 0.05 # 0.1 -> 0.05
            
        max_feed_specific = 0.1 # 10.0 -> 0.1 (毎ステップ最大0.1 mM追加。1時間あたり0.5 mM相当)
        nutrient_supplementation = {
            'sn_or16': action[0] * max_feed_specific,
            'sn_ns21': action[1] * max_feed_specific,
            'sn_lp':   action[2] * max_feed_specific,
            'yeast_extract': action[3] * max_feed_common
        }
        dynamic_kla = action[4] * 200.0
        
        # シミュレーター実行 (FBAソルバー呼び出し)
        state = self.simulator.step(
            self.rubber_degradation_rates,
            nutrient_supplementation,
            dynamic_kla=dynamic_kla
        )
        
        obs = self._get_observation(state)
        
        # 共通メトリクスの計算 (info用)
        current_total_pha = sum(s.pha_accumulated for s in state.species.values())
        current_bs = state.metabolites.get('biosurfactant_e', 0.0)
        current_total_biomass = sum(s.biomass for s in state.species.values())
        
        # --- カリキュラム学習報酬ロジック ---
        progress = min(1.0, self.current_total_steps / max(1, self.total_timesteps))
        reward = 0.0
        
        # 1. pH ストライクゾーン (Stage 1 & 2)
        h_conc = state.metabolites.get('h_e', 0.0001)
        current_ph = -np.log10(max(1e-12, h_conc) / 1000.0)
        
        if progress < 0.1:
            target_min, target_max = 4.0, 10.0 # Stage 1: 広め
        else:
            target_min, target_max = 6.0, 8.0  # Stage 2: 厳格
            
        ph_dist = 0.0
        if current_ph < target_min:
            ph_dist = target_min - current_ph
            # 酸性側ペナルティ
            reward -= min(2.0, ph_dist * (0.5 if progress < 0.1 else 1.0))
        elif current_ph > target_max:
            ph_dist = current_ph - target_max
            # --- 科学的非対称性: アルカリ(アンモニア毒性)側ペナルティを2倍に ---
            reward -= min(4.0, ph_dist * (1.0 if progress < 0.1 else 2.0))
            
        if ph_dist == 0:
            reward += 0.05 * (1.0 - progress) # 維持ボーナス

        # 2. 生産報酬 (Stage 3, 4, 5)
        if progress > 0.3:
            # --- 収穫期フェーズ: 生産報酬を滑らかに強化 ---
            # progress 0.3 -> 0.8 で線形補間し、急激な報酬変化によるTD誤差の増大を抑制
            rubber_w = float(np.interp(progress, [0.3, 0.8], [1.0, 500.0]))
            pha_w = float(np.interp(progress, [0.3, 0.8], [0.1, 50.0]))
            bs_w = float(np.interp(progress, [0.3, 0.8], [5.0, 200.0]))

            delta_deg = max(0, self.prev_rubber - state.rubber_concentration)
            reward += delta_deg * rubber_w

            delta_pha = max(0, current_total_pha - self.last_total_pha)
            reward += delta_pha * pha_w

            delta_bs = max(0, current_bs - self.last_bs)
            reward += delta_bs * bs_w

        # 4. リソース投入コストとCCR回避の誘導
        c_specific = 0.05
        c_common = 0.1
        c_kla = 0.005
        cost = (
            c_specific * (action[0] + action[1] + action[2]) +
            c_common * action[3] +
            c_kla * action[4]
        )
        reward -= cost
        
        # CCRの明示的なペナルティ (グルコースが蓄積しているとマイナス)
        glc_conc = state.metabolites.get('glc__D_e', 0.0)
        if glc_conc > 0.1:
            reward -= 0.1 * glc_conc # グルコース蓄積ペナルティ

        # 履歴の更新
        self.last_total_pha = current_total_pha
        self.last_bs = current_bs

        # --- 科学的微量持続供給 (Inducer Influx) ---
        # OR16の分解スイッチを常にONにするための微量供給 (0.001 mM)
        self.simulator.state.metabolites['rubber_fragment_e'] = \
            self.simulator.state.metabolites.get('rubber_fragment_e', 0.0) + 0.001

        # 3. 終了管理と「代謝義務化チェック」

        terminated = any(s.biomass < 0.01 for s in state.species.values())
        
        # ショック療法: 72時間経過してもバイオマスが増えていない場合は強制終了
        if state.time > 72.0:
            if current_total_biomass <= self.last_activity_biomass * 1.01:
                terminated = True
                reward -= 1.0 
            else:
                # 活動が確認されたら基準を更新
                self.last_activity_biomass = current_total_biomass

        if terminated and state.time < self.max_time:
            reward -= 5.0 
            
        truncated = state.time >= self.max_time
        self.current_step += 1
        self.prev_rubber = state.rubber_concentration
        
        info = {
            'rubber_remaining': state.rubber_concentration,
            'ph': current_ph,
            'total_pha': current_total_pha,
            'biosurfactant': current_bs,
            'biomass_or16': state.species.get('Actinoplanes_sp_OR16_lcp', state.species.get('Engine 1', state.species.get(next(iter(state.species)), None))).biomass,
            'biomass_ns21': state.species.get('Rhizobacter_gummiphilus_NS21', state.species.get('Engine 2', None)).biomass if 'Rhizobacter_gummiphilus_NS21' in state.species else 0.0,
            'biomass_lp': state.species.get('Lactobacillus_plantarum', state.species.get('Stabilizer', None)).biomass if 'Lactobacillus_plantarum' in state.species else 0.0
        }
        
        return obs, float(reward), terminated, truncated, info
