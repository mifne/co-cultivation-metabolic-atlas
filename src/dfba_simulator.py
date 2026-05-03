"""
Dynamic Flux Balance Analysis (dFBA) Simulator
天然ゴム分解微生物コンソーシアムのdFBAシミュレーション
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cobra
from cobra.flux_analysis import flux_variability_analysis
import json
from pathlib import Path
import os
import csv

@dataclass
class SpeciesState:
    """各微生物種の状態"""
    biomass: float  # バイオマス濃度 [g/L]
    growth_rate: float  # 増殖速度 [1/h]
    metabolite_uptake: Dict[str, float]  # 代謝物取り込み速度 [mmol/gDW/h]
    metabolite_secretion: Dict[str, float]  # 代謝物分泌速度 [mmol/gDW/h]
    pha_accumulated: float = 0.0  # PHA蓄積量 [mmol]


@dataclass
class ConsortiumState:
    """コンソーシアム全体の状態"""
    time: float  # 時刻 [h]
    species: Dict[str, SpeciesState]  # 各種の状態
    metabolites: Dict[str, float]  # 環境中の代謝物濃度 [mM]
    rubber_concentration: float  # 天然ゴム濃度 [g/L]


class dFBASimulator:
    """dFBAシミュレーター"""

    def __init__(self, 
                 models: Dict[str, cobra.Model],
                 initial_biomass: Dict[str, float],
                 initial_metabolites: Dict[str, float],
                 initial_rubber: float = 100.0,
                 volume: float = 1.0,
                 dt: float = 0.5,
                 data_log_path: Optional[str] = None,
                 max_uptake_rate: float = 20.0,
                 carrying_capacity: float = 20.0,
                 metabolite_inhibition_threshold: float = 20.0,
                 metabolite_inhibition_decay: float = 0.1):
        self.models = models
        self.dt = dt
        self.volume = volume
        self.data_log_path = data_log_path
        self.max_uptake_rate = max_uptake_rate
        self.carrying_capacity = carrying_capacity
        self.metabolite_inhibition_threshold = metabolite_inhibition_threshold
        self.metabolite_inhibition_decay = metabolite_inhibition_decay
        
        # --- 科学的再構築: 培地の緩衝能 (50mM リン酸, pH7.0 -> 6.0) ---
        self.buffering_pool = 16.37 # mmol/L H+ を中和可能
        
        if self.data_log_path:
            os.makedirs(os.path.dirname(self.data_log_path), exist_ok=True)
            
        self.log_fieldnames = [
            'time', 'species', 'growth_rate', 'biomass', 'pha_accumulated', 
            'rubber_concentration', 'cumulative_co2',
            'conc_glc__D_e', 'conc_nh4_e', 'conc_o2_e', 'conc_pi_e', 
            'conc_arg__L_e', 'conc_trp__L_e', 'conc_leu__L_e',
            'conc_rubber_fragment_e', 'conc_odtd_e', 'conc_h2o_e', 'conc_h_e',
            'flux_EX_co2_e', 'flux_EX_pha_c', 'flux_EX_phb_c', 
            'flux_EX_rubber_fragment_e', 'flux_EX_odtd_e'
        ]
        self._log_header_initialized = False
        
        self.state = ConsortiumState(
            time=0.0,
            species={
                name: SpeciesState(
                    biomass=initial_biomass.get(name, 0.0),
                    growth_rate=0.0,
                    metabolite_uptake={},
                    metabolite_secretion={}
                )
                for name in models.keys()
            },
            metabolites=initial_metabolites.copy(),
            rubber_concentration=initial_rubber
        )
        self.cumulative_co2_emission = 0.0
        self.current_step = 0
        
        self.exchange_reactions = self._identify_exchange_reactions()
        
        self.original_bounds = {}
        for species_name, model in self.models.items():
            self.original_bounds[species_name] = {
                rxn.id: (rxn.lower_bound, rxn.upper_bound)
                for rxn in model.exchanges
            }
        
        self._initialize_medium()

    def _identify_exchange_reactions(self) -> Dict[str, Dict[str, str]]:
        exchange_map = {}
        for species_name, model in self.models.items():
            exchange_map[species_name] = {}
            for rxn in model.exchanges:
                if len(rxn.metabolites) == 1:
                    met = list(rxn.metabolites.keys())[0]
                    # M_ プレフィックスを除去した ID をキーにする
                    clean_id = met.id[2:] if met.id.startswith('M_') else met.id
                    exchange_map[species_name][clean_id] = rxn.id
            
            # --- 特殊反応の明示的マッピング (名前が不規則な場合) ---
            # PHA 蓄積 (Sink 反応)
            for pha_id in ['EX_pha_c', 'EX_phb_c', 'R_EX_pha_c']:
                if pha_id in model.reactions:
                    exchange_map[species_name]['pha_c'] = pha_id
            
            # ゴム取り込み (Exchange 反応)
            for rubber_id in ['EX_rubber_e', 'R_EX_rubber_e', 'rubber_high_e']:
                if rubber_id in model.reactions:
                    exchange_map[species_name]['rubber_e'] = rubber_id

            # 栄養供給用エイリアス
            if 'Actinoplanes' in species_name and 'EX_mlttr_e' in exchange_map[species_name].values():
                exchange_map[species_name]['sn_or16'] = 'EX_mlttr_e'
            elif 'Rhizobacter' in species_name and 'EX_ptrc_e' in exchange_map[species_name].values():
                exchange_map[species_name]['sn_ns21'] = 'EX_ptrc_e'
            elif 'Lactobacillus' in species_name and 'EX_mnl_e' in exchange_map[species_name].values():
                exchange_map[species_name]['sn_lp'] = 'EX_mnl_e'
        return exchange_map
    
    def _initialize_medium(self):
        # --- 科学的再構築: リン酸緩衝系の初期化 (50mM, pH 7.0) ---
        self.buffer_total = 50.0  # mM
        self.pKa = 7.21           # Phosphate pKa2
        initial_ph = 7.0
        ratio = 10**(initial_ph - self.pKa)
        self.buffer_base = self.buffer_total * ratio / (1 + ratio)
        self.buffer_acid = self.buffer_total - self.buffer_base

        for species_name, model in self.models.items():
            # 一旦すべての交換反応の取り込みを制限 (培地にないものは 0)
            for rxn in model.exchanges:
                if rxn.lower_bound < 0:
                    rxn.lower_bound = min(0.0, rxn.upper_bound)
            
            # 培地に存在する代謝物の取り込みを許可
            for met_id in self.state.metabolites.keys():
                if met_id in self.exchange_reactions[species_name]:
                    rxn_id = self.exchange_reactions[species_name][met_id]
                    target_rxn = model.reactions.get_by_id(rxn_id)
                    
                    if rxn_id in self.original_bounds[species_name]:
                        lb, _ = self.original_bounds[species_name][rxn_id]
                        target_rxn.lower_bound = min(lb, target_rxn.upper_bound)
                    else:
                        target_rxn.lower_bound = min(-1000.0, target_rxn.upper_bound)
            
            # 水の交換反応を常に許可
            if 'h2o_e' in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name]['h2o_e']
                model.reactions.get_by_id(rxn_id).lower_bound = -1000.0
            
            # --- 効率化: ソルバー設定をここで一度だけ行う ---
            try:
                model.solver.configuration.timeout = 10 
                model.solver.configuration.presolve = True
            except: pass

    def _diagnose_infeasibility(self, species_name: str, model: cobra.Model):
        print(f"    🔍 {species_name} infeasibility diagnosis...")
        with model:
            for rxn in model.exchanges:
                rxn.lower_bound = -1000
                rxn.upper_bound = 1000
            try:
                solution = model.optimize()
                if solution.status == 'optimal':
                    print(f"      ✅ Relaxed optimal: μ={solution.objective_value:.4f} -> Problem: Constraints too tight")
                else:
                    print(f"      ❌ Still infeasible (status: {solution.status}) -> Problem: Model structure")
            except: pass
    
    def set_uptake_constraints(self, species_name: str, metabolite_concentrations: Dict[str, float]):
        model = self.models[species_name]
        max_uptake = self.max_uptake_rate
        
        # 1. すべての交換反応の吸収(lower_bound)を一旦 0 にリセット (培地にないものの吸収を禁止)
        for rxn in model.exchanges:
            # ゴムと特殊な Sink 反応はリセット対象から除外
            if 'rubber_e' in rxn.id or 'pha_c' in rxn.id or 'phb_c' in rxn.id:
                continue
            # lower_bound を 0 にしたいが、upper_bound が負の場合は lb <= ub を維持するためそれに合 わせる
            rxn.lower_bound = min(0.0, rxn.upper_bound)

        # 2. H2O と H+ は常に供給可能とする（水系溶媒のため）
        for h_id in ['h2o_e', 'h_e']:
            if h_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][h_id]
                model.reactions.get_by_id(rxn_id).lower_bound = -1000.0

        # 3. 培地に存在する代謝物の取り込み制約を設定
        for met_id, concentration in metabolite_concentrations.items():
            if met_id in ['h2o_e', 'h_e']: continue # 既に処理済み

            if met_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][met_id]
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                    # モデル本来の最大取り込み能力(original_bounds)を考慮
                    orig_lb, _ = self.original_bounds[species_name].get(rxn_id, (-1000.0, 1000.0))

                    # ミカエリス・メンテン型の速度制限
                    Km = 0.01 if met_id in ['glc__D_e', 'o2_e', 'pi_e', 'nh4_e'] else 0.1
                    uptake_limit = max_uptake * concentration / (Km + concentration)

                    # 培地濃度が極めて低い場合は完全に遮断
                    if concentration < 1e-9: uptake_limit = 0.0

                    # 負の値として設定 (吸収)
                    combined_lb = max(-uptake_limit, orig_lb)
                    rxn.lower_bound = min(0.0, combined_lb)
                except: continue

        # 4. 【科学的整合性】ゴム分解・発現抑制の特殊ロジック
        if 'OR16' in species_name:
            # ゴム取り込みを常に開放 (上限はLCP反応側で制御される)
            if 'rubber_e' in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name]['rubber_e']
                try:
                    model.reactions.get_by_id(rxn_id).lower_bound = -1000.0
                except:
                    pass

            # ゴム分解酵素(LCP)の発現制御
            glc_conc = metabolite_concentrations.get('glc__D_e', 0.0)
            rubber_conc = self.state.rubber_concentration

            if 'R_LCP' in model.reactions:
                lcp_rxn = model.reactions.get_by_id('R_LCP')

                # --- トランスクリプトーム加重誘導ロジック ---
                # 誘導倍率 (Rubber/Glucose): lcp1=22.2, lcp2=17.1, lcp3=335.0
                # 比活性重み (Lcp1基準): Lcp1=1.0, Lcp2=0.14, Lcp3=0.055

                # 基礎発現レベル (V_base)
                v_base = 0.1 # mmol/gDW/h

                # ゴムによる誘導係数 (マイケルソン・メンテン型で飽和を表現)
                K_rubber = 1.0 # g/L
                induction_factor = rubber_conc / (K_rubber + rubber_conc) if rubber_conc > 1e-6 else 0.0

                # 加重誘導倍率の計算
                # Lcp1 contribution: 22.2 * 1.0 = 22.2
                # Lcp2 contribution: 17.1 * 0.14 = 2.39
                # Lcp3 contribution: 335.0 * 0.055 = 18.42
                # Total max induction = 22.2 + 2.39 + 18.42 = 43.01
                max_induction = 43.01

                effective_induction = 1.0 + (max_induction * induction_factor)

                # グルコースによる抑制 (CCR)
                K_inhibition = 0.05 # mM
                repression_factor = K_inhibition / (K_inhibition + glc_conc)

                # 最終的なフラックス上限
                lcp_limit = v_base * effective_induction * repression_factor
                lcp_rxn.upper_bound = lcp_limit

            # ゴム(ポリマー)の取り込み制約 (旧ロジックとの互換性)
            target_rubber_ids = ['EX_rubber_e', 'rubber_high_e', 'R_EX_rubber_e']

            for rid in target_rubber_ids:
                actual_rid = self.exchange_reactions[species_name].get(rid) or (rid if rid in model.reactions else None)
                if actual_rid and actual_rid in model.reactions:
                    rxn = model.reactions.get_by_id(actual_rid)
                    concentration = self.state.rubber_concentration
                    uptake_limit = max_uptake * concentration / (1.0 + concentration)
                    rxn.lower_bound = -uptake_limit
                    break

    
    def solve_fba(self, species_name: str) -> Optional[cobra.Solution]:
        model = self.models[species_name]
        if not hasattr(self, '_fba_stats'):
            self._fba_stats = {name: {'success': 0, 'failure': 0} for name in self.models.keys()}
            
        try:
            # 数値的不安定性によるクラッシュを避けるため、最適化を実行
            solution = model.optimize()
            if solution.status == 'optimal':
                # 極端な値（Inf/-Inf/NaN）をチェック
                if not np.isfinite(solution.objective_value):
                    return None
                self._fba_stats[species_name]['success'] += 1
                return solution
            else:
                self._fba_stats[species_name]['failure'] += 1
                return None
        except Exception as e:
            self._fba_stats[species_name]['failure'] += 1
            print(f"⚠️ {species_name}: Solver Numerical Instability / Crash suppressed: {e}")
            return None

    def update_biomass(self, species_name: str, growth_rate: float) -> float:
        species_state = self.state.species[species_name]
        inhibition_factor = 1.0 # 初期値
        
        # 負の増殖速度（数値誤差または意図的なDecay）の処理
        if growth_rate < 0:
            effective_growth_rate = growth_rate # Decayをそのまま適用
        else:
            h_e_conc = max(1e-12, self.state.metabolites.get('h_e', 0.0001))
            current_ph = -np.log10(h_e_conc / 1000.0)
            
            if 'Lactobacillus' in species_name:
                opt_ph, lower_tol, upper_tol, decay = 5.5, 1.5, 1.0, 2.0
                if current_ph < opt_ph: ph_diff = max(0, opt_ph - current_ph - lower_tol)
                else: ph_diff = max(0, current_ph - opt_ph - upper_tol)
            else:
                opt_ph, tolerance, decay = 7.0, 1.0, 2.0
                ph_diff = max(0, abs(current_ph - opt_ph) - tolerance)
            
            if ph_diff > 0: inhibition_factor *= np.exp(-decay * ph_diff)

            # --- 科学的修正: 避難所の破壊（極限pHでの死滅） ---
            death_rate = 0.0
            if current_ph > 10.5:
                # 強アルカリ下での細胞溶解: pH 11.9で μ = -0.2/h
                death_rate = 0.2 * (current_ph - 10.5) / (11.9 - 10.5)
            elif current_ph < 4.0:
                death_rate = 0.1 * (4.0 - current_ph) / (4.0 - 2.0)

            if current_ph > 12.0 or current_ph < 3.0: inhibition_factor = 0.0

            effective_growth_rate = (growth_rate * inhibition_factor) - death_rate

        effective_growth_rate = np.clip(effective_growth_rate, -0.5, 2.0)

        total_biomass = sum(s.biomass for s in self.state.species.values())
        capacity_factor = max(0.0, 1.0 - total_biomass / self.carrying_capacity)

        dX = effective_growth_rate * species_state.biomass * (capacity_factor if effective_growth_rate > 0 else 1.0) * self.dt
        # 下限を 0.0 にして絶滅を許可する
        species_state.biomass = np.clip(species_state.biomass + dX, 0.0, self.carrying_capacity)
        species_state.growth_rate = effective_growth_rate
        species_state.last_inhibition_factor = inhibition_factor
        return effective_growth_rate
    
    def _update_environment(self, species_solutions: Dict[str, Dict[str, float]]):
        """全菌種のフラックスを統合して共有環境の状態を更新"""
        total_delta_metabolites = {}
        net_h_flux_mmol = 0.0
        
        # ステップ開始時に統計データをクリア
        for s in self.state.species.values():
            s.metabolite_uptake.clear()
            s.metabolite_secretion.clear()

        for species_name, fluxes in species_solutions.items():
            biomass = self.state.species[species_name].biomass
            for met_id, rxn_id in self.exchange_reactions[species_name].items():
                # rxn_id がモデルにない場合のフォールバック（例：特殊反応）
                actual_rid = rxn_id
                if actual_rid not in fluxes and met_id == 'rubber_e':
                    # モデル固有のゴム反応IDを探す
                    for fallback in ['EX_rubber_e', 'R_EX_rubber_e', 'rubber_high_e']:
                        if fallback in fluxes:
                            actual_rid = fallback
                            break
                            
                flux = fluxes.get(actual_rid, 0.0)
                delta = flux * biomass * self.dt / self.volume
                
                if met_id == 'h_e':
                    net_h_flux_mmol += delta
                elif met_id == 'nh4_e' and flux > 0:
                    # アンモニア放出による中和効果 (NH3 + H+ -> NH4+)
                    net_h_flux_mmol -= delta
                elif met_id == '2mba_e' and flux > 0:
                    # 有機酸（バイオサーファクタント代替）の放出による酸性化
                    net_h_flux_mmol += delta
                if met_id != 'h_e':
                    total_delta_metabolites[met_id] = total_delta_metabolites.get(met_id, 0.0) + delta
                
                # 個別の取り込み/分泌統計を更新
                if flux < -1e-9: self.state.species[species_name].metabolite_uptake[met_id] = -flux
                elif flux > 1e-9: self.state.species[species_name].metabolite_secretion[met_id] = flux
            
            # --- 科学的修正: PHA蓄積量の更新 ---
            # 内部反応 'EX_pha_c' または 'EX_phb_c' のフラックスを累積
            for pha_id in ['EX_pha_c', 'EX_phb_c']:
                pha_flux = fluxes.get(pha_id, 0.0)
                if pha_flux > 0: # 蓄積
                    self.state.species[species_name].pha_accumulated += pha_flux * biomass * self.dt
        for met_id, delta in total_delta_metabolites.items():
            if met_id in self.state.metabolites:
                self.state.metabolites[met_id] = max(0.0, self.state.metabolites[met_id] + delta)
            elif delta > 0:
                self.state.metabolites[met_id] = delta

        # --- 科学的 pH 更新 (Henderson-Hasselbalch) ---
        self.buffer_base -= net_h_flux_mmol
        self.buffer_acid += net_h_flux_mmol
        self.buffer_base = np.clip(self.buffer_base, 0.001, self.buffer_total - 0.001)
        self.buffer_acid = self.buffer_total - self.buffer_base
        
        current_ph = self.pKa + np.log10(self.buffer_base / self.buffer_acid)
        self.state.metabolites['h_e'] = 10**(3.0 - current_ph)

    def degrade_rubber(self, degradation_rates: Dict[str, float]):
        total_degradation_g = 0.0
        
        # --- 科学的シナジー: 10 kDa 糖脂質タンパク質複合体効果 ---
        bs_conc_mM = self.state.metabolites.get('biosurfactant_e', 0.0)
        # 飽和定数 K_bs = 0.01 mM, 最大1.5倍 (1.0 + 0.5) の加速に設定
        bs_boost = 1.0 + (0.5 * bs_conc_mM / (0.01 + bs_conc_mM))
        
        for species_name, species_state in self.state.species.items():
            # FBAモデルによるアクティブなゴム分解フラックスを取得 ('rubber_e' キーを使用)
            fba_rubber_flux = species_state.metabolite_uptake.get('rubber_e', 0.0)
            
            # ゴム(C5H8)の分子量 68.12. BS効果を反映
            # fba_rubber_flux は取り込み量（正の値）として記録されている
            rate = (abs(fba_rubber_flux) * 68.12 / 1000.0) * bs_boost
            total_degradation_g += rate * species_state.biomass * self.dt
        
        self.state.rubber_concentration = max(0.0, self.state.rubber_concentration - total_degradation_g)
        # 注意: 断片 (rubber_fragment_e) の生成は、FBAモデル内の R_LCP 反応と
        # その後の EX_rubber_fragment_e 分泌によって自動的に環境へ反映されるため、
        # ここでの手動加算は二重計上を避けるため廃止する。
            
    def _log_telemetry(self):
        if not self.data_log_path:
            return
            
        file_exists = os.path.exists(self.data_log_path)
        with open(self.data_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fieldnames)
            if not self._log_header_initialized and not file_exists:
                writer.writeheader()
                self._log_header_initialized = True
            elif not self._log_header_initialized:
                self._log_header_initialized = True

            for species_name, species_state in self.state.species.items():
                # ログ用データの収集
                row = {
                    'time': self.state.time,
                    'species': species_name,
                    'growth_rate': species_state.growth_rate,
                    'biomass': species_state.biomass,
                    'pha_accumulated': species_state.pha_accumulated,
                    'rubber_concentration': self.state.rubber_concentration,
                    'cumulative_co2': self.cumulative_co2_emission,
                    'conc_glc__D_e': self.state.metabolites.get('glc__D_e', 0.0),
                    'conc_nh4_e': self.state.metabolites.get('nh4_e', 0.0),
                    'conc_o2_e': self.state.metabolites.get('o2_e', 0.0),
                    'conc_pi_e': self.state.metabolites.get('pi_e', 0.0),
                    'conc_arg__L_e': self.state.metabolites.get('arg__L_e', 0.0),
                    'conc_trp__L_e': self.state.metabolites.get('trp__L_e', 0.0),
                    'conc_leu__L_e': self.state.metabolites.get('leu__L_e', 0.0),
                    'conc_rubber_fragment_e': self.state.metabolites.get('rubber_fragment_e', 0.0),
                    'conc_odtd_e': self.state.metabolites.get('odtd_e', 0.0),
                    'conc_h2o_e': self.state.metabolites.get('h2o_e', 0.0),
                    'conc_h_e': self.state.metabolites.get('h_e', 0.0001),
                    'flux_EX_co2_e': species_state.metabolite_secretion.get('co2_e', 0.0),
                    'flux_EX_pha_c': species_state.metabolite_secretion.get('pha_c', 0.0),
                    'flux_EX_phb_c': species_state.metabolite_secretion.get('phb_c', 0.0),
                    'flux_EX_rubber_fragment_e': species_state.metabolite_secretion.get('rubber_fragment_e', 0.0),
                    'flux_EX_odtd_e': species_state.metabolite_secretion.get('odtd_e', 0.0)
                }
                writer.writerow(row)

    def step(self, rubber_degradation_rates: Dict[str, float], nutrient_supplementation: Dict[str, float], dynamic_kla: float = 50.0) -> ConsortiumState:
        # 溶存酸素の更新 (解析解による無条件安定化)
        o2_sat = 0.25
        current_o2 = self.state.metabolites.get('o2_e', o2_sat)
        self.state.metabolites['o2_e'] = o2_sat - (o2_sat - current_o2) * np.exp(-dynamic_kla * self.dt)

        # 栄養添加 (種特異的栄養素のみ)
        supplementation_map = {
            'sn_or16': 'mlttr_e', # OR16専用 (マルトトリオース)
            'sn_ns21': 'ptrc_e',  # NS21専用 (プトレシン)
            'sn_lp':   'mnl_e'    # LP専用 (マンニトール)
        }
        for nut_name, met_id in supplementation_map.items():
            supplement = np.clip(nutrient_supplementation.get(nut_name, 0.0), 0.0, 10.0)
            self.state.metabolites[met_id] = self.state.metabolites.get(met_id, 0.0) + supplement
        
        # 【科学的整合性】酵母エキス(YE)相当の複合栄養源の添加
        # ここに含まれるグルコース等の共通栄養源を全菌種が奪い合う
        if 'yeast_extract' in nutrient_supplementation:
            ye_amount = np.clip(nutrient_supplementation['yeast_extract'], 0.0, 10.0)
            ye_composition = {
                'glc__D_e': 1.0,  # 共通炭素源 (グルコース)
                'nh4_e': 0.5,     # 共通窒素源 (アンモニウム)
                # アミノ酸
                'arg__L_e': 0.1, 'trp__L_e': 0.05, 'leu__L_e': 0.1, 'ile__L_e': 0.1, 'val__L_e': 0.1,
                'lys__L_e': 0.1, 'met__L_e': 0.05, 'phe__L_e': 0.05, 'his__L_e': 0.05, 'tyr__L_e': 0.05,
                'thr__L_e': 0.1, 'cys__L_e': 0.05, 'ala__L_e': 0.1, 'asp__L_e': 0.1, 'glu__L_e': 0.1,
                'gly_e': 0.1, 'pro__L_e': 0.1, 'ser__L_e': 0.1, 'asn__L_e': 0.1, 'gln__L_e': 0.1,
                # ビタミン・核酸
                'nac_e': 0.01, 'ribflv_e': 0.01, 'pnto__R_e': 0.01, 'thm_e': 0.01, 
                'btn_e': 0.001, '4abz_e': 0.01, 'fol_e': 0.001, 'nicnt_e': 0.01,
                'ade_e': 0.01, 'gua_e': 0.01, 'ura_e': 0.01, 'cytd_e': 0.01
            }
            for met_id, coeff in ye_composition.items():
                self.state.metabolites[met_id] = self.state.metabolites.get(met_id, 0.0) + ye_amount * coeff

        # 各種の代謝計算
        total_co2_flux = 0.0
        species_solutions = {}
        
        for species_name in self.models.keys():
            self.set_uptake_constraints(species_name, self.state.metabolites)
            
            # --- 科学的修正: PHA Spillover Metabolism (窒素枯渇時のOverflow) ---
            model = self.models[species_name]
            if 'NS21' in species_name:
                env_nh4 = self.state.metabolites.get('nh4_e', 0.0)
                if env_nh4 < 0.1:  # 窒素枯渇
                    # 目的関数をPHA生成に変更
                    if 'EX_pha_c' in model.reactions:
                        model.objective = 'EX_pha_c'
                    elif 'EX_phb_c' in model.reactions:
                        model.objective = 'EX_phb_c'
                else:
                    # 成長に戻す
                    if 'Growth' in model.reactions:
                        model.objective = 'Growth'

            solution = self.solve_fba(species_name)
            
            if solution is not None:
                # 目的関数が切り替わっていても、実際のバイオマス増殖フラックスを正しく取得する
                if 'Growth' in solution.fluxes:
                    raw_mu = solution.fluxes['Growth']
                elif 'BIOMASS_LLA' in solution.fluxes:
                    raw_mu = solution.fluxes['BIOMASS_LLA']
                else:
                    raw_mu = solution.objective_value
                    
                eff_mu = self.update_biomass(species_name, raw_mu)

                # Flux scaling must NEVER be negative (which would reverse reactions).
                # Living cells' metabolism scales with inhibition. Death rate decreases biomass but doesn't invert metabolism.
                factor = self.state.species[species_name].last_inhibition_factor

                # スケーリング済みフラックスを保存
                species_solutions[species_name] = {k: v * factor for k, v in solution.fluxes.items()}                
                # CO2排出の集計
                co2_rxn = self.exchange_reactions[species_name].get('co2_e')
                if co2_rxn:
                    total_co2_flux += species_solutions[species_name][co2_rxn] * self.state.species[species_name].biomass
            else:
                self.update_biomass(species_name, -0.01)

        # 全菌種のフラックスを統合して環境を更新
        self._update_environment(species_solutions)

        # ゴム分解の実行
        self.degrade_rubber(rubber_degradation_rates)
        
        # CO2蓄積
        self.cumulative_co2_emission += total_co2_flux * self.dt
        
        # ログ出力
        if self.current_step % 1 == 0: # 毎ステップ記録
            self._log_telemetry()

        self.state.time += self.dt
        self.current_step += 1
        return self.state
    
    def get_state_vector(self) -> np.ndarray:
        state_vec = []
        for name in sorted(self.models.keys()): state_vec.append(np.clip(self.state.species[name].biomass, 0.0, self.carrying_capacity))
        for name in sorted(self.models.keys()): state_vec.append(np.clip(self.state.species[name].growth_rate, -0.5, 2.0))
        for met in ['glc__D_e', 'nh4_e', 'rubber_fragment_e', 'odtd_e', 'arg__L_e', 'trp__L_e', 'leu__L_e', 'h_e']:
            state_vec.append(np.clip(self.state.metabolites.get(met, 0.0), 0.0, 1000.0))
        for name in sorted(self.models.keys()): state_vec.append(np.clip(self.state.species[name].pha_accumulated, 0.0, 1000.0))
        state_vec.append(np.clip(self.state.rubber_concentration, 0.0, 100000.0))
        return np.nan_to_num(np.array(state_vec, dtype=np.float32), nan=0.0)
