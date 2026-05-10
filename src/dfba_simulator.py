"""
Dynamic Flux Balance Analysis (dFBA) Simulator
天然ゴム分解微生物コンソーシアムのdFBAシミュレーション
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
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

    # --- 数値的安定性のための定数 ---
    MARGIN = 0.0  # マージンを撤廃し、物理的に正確な境界条件を使用


    # --- 実用化修正: 酵母エキス(YE)および微量必須成分の定義 ---
    # TDDに基づく実証済みの必須成分リスト (OR16, NS21, LP 全体の最小公倍数)
    YE_COMPONENTS = [
        'ca2_e', 'cl_e', 'cobalt2_e', 'cu_e', 'cu2_e', 'k_e', 'mg2_e', 'mn2_e', 
        'pnto__R_e', 'so4_e', 'thm_e', 'zn2_e', 
        'salchs4fe_e', 'tsul_e', 'ump_e', 'xtsn_e', # OR16生存に必須
        'istfrnB_e', 'tyrp_e',                     # NS21生存に必須
        'glu__L_e', 'leu__L_e', 'nmn_e', 'ppi_e', 'ura_e', # LPおよび全体補助
        'fe2_e', 'fe3_e', 'h2o_e', 'h_e', 'pi_e', 'nh4_e'
    ]

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
                    # --- 科学制約: 多重プレフィックス (M_M_...) の再帰的除去 ---
                    met_id = met.id
                    while met_id.startswith('M_'):
                        met_id = met_id[2:]
                    exchange_map[species_name][met_id] = rxn.id
            
            # --- 特殊反応の明示的マッピング (名前が不規則な場合) ---
            # PHA 蓄積 (Sink 反応)
            for pha_id in ['EX_pha_c', 'EX_phb_c', 'R_EX_pha_c', 'EX_phb_lp']:
                if pha_id in model.reactions:
                    exchange_map[species_name]['pha_c'] = pha_id
            
            # ゴム分解（ポリマー切断）の明示的マッピング
            # OR16/NS21共に 'EX_rubber_bulk_e' (C5単位) をゴム減少のトリガーとする
            for rid in ['EX_rubber_bulk_e', 'R_EX_rubber_bulk_e']:
                if rid in model.reactions:
                    exchange_map[species_name]['rubber_e'] = rid
                    break

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
            # ソルバー設定の初期化
            try:
                model.solver.configuration.timeout = 10 
                model.solver.configuration.presolve = False 
                # 数値的精度と速度のバランスを最適化
                model.solver.configuration.tolerances.feasibility = 1e-7
                model.solver.configuration.tolerances.optimality = 1e-7
                model.solver.configuration.tolerances.integrality = 1e-7
            except: pass

            # 一旦すべての交換反応の取り込みを制限 (培地にないものは 0)
            for rxn in model.exchanges:
                if rxn.lower_bound < 0:
                    # 数値的マージン確保
                    rxn.lower_bound = min(0.0, rxn.upper_bound - self.MARGIN)
            
            # 培地に存在する代謝物の取り込みを許可
            for met_id in self.state.metabolites.keys():
                if met_id in self.exchange_reactions[species_name]:
                    rxn_id = self.exchange_reactions[species_name][met_id]
                    target_rxn = model.reactions.get_by_id(rxn_id)
                    
                    if rxn_id in self.original_bounds[species_name]:
                        lb, _ = self.original_bounds[species_name][rxn_id]
                        target_rxn.lower_bound = min(lb, target_rxn.upper_bound - self.MARGIN)
                    else:
                        target_rxn.lower_bound = min(-1000.0, target_rxn.upper_bound - self.MARGIN)
            
            # 水の交換反応を常に許可
            if 'h2o_e' in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name]['h2o_e']
                target_rxn = model.reactions.get_by_id(rxn_id)
                target_rxn.lower_bound = min(-1000.0, target_rxn.upper_bound - self.MARGIN)

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

    def set_uptake_constraints(self, species_name: str, metabolite_concentrations: Dict[str, float], dynamic_kla: float = 50.0):
        model = self.models[species_name]
        max_uptake = self.max_uptake_rate
        
        # 1. すべての交換反応の吸収(lower_bound)を一旦 0 にリセット (培地にないものの吸収を禁止)
        for rxn in model.exchanges:
            if 'rubber_e' in rxn.id or 'pha_c' in rxn.id or 'phb_c' in rxn.id:
                continue
            rxn.lower_bound = min(0.0, rxn.upper_bound - self.MARGIN)

        # --- 科学的修正: 酵母エキス(YE)パッケージの概念を導入 ---
        ye_conc = metabolite_concentrations.get('yeast_extract_e', 0.0)
        
        # 1. 培地中の必須微量要素を常に供給 (Background Nutrients)
        # 科学的根拠: 実験培地には微量の金属イオンや必須補因子が含まれており、
        # これがゼロだと炭素源があっても増殖(FBA解)が不可能になる。
        for met_id in self.YE_COMPONENTS:
            if met_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][met_id]
                target_rxn = model.reactions.get_by_id(rxn_id)
                # 微量(0.1)を常に許可
                target_rxn.lower_bound = min(-0.1, target_rxn.upper_bound - self.MARGIN)
        
        # 2. 酵母エキスが供給されている場合は、取り込み上限をさらに開放
        if ye_conc > 0.001:
            for met_id in self.YE_COMPONENTS:
                if met_id in self.exchange_reactions[species_name]:
                    rxn_id = self.exchange_reactions[species_name][met_id]
                    target_rxn = model.reactions.get_by_id(rxn_id)
                    target_rxn.lower_bound = min(-1.0, target_rxn.upper_bound - self.MARGIN)

        # 2. H2O, O2, H+ は生存に必須のため常に供給可能とする
        # 科学的修正: H+ の取り込み上限を -5.0 に制限し、数学的な「プロトン食い」による
        # 非現実的なpH急上昇を防止する。放出側(pH低下)は制限しない。
        for h_id in ['h2o_e', 'h_e', 'o2_e']:
            if h_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][h_id]
                target_rxn = model.reactions.get_by_id(rxn_id)
                bound = -1000.0 if h_id != 'h_e' else -5.0
                target_rxn.lower_bound = min(bound, target_rxn.upper_bound - self.MARGIN)

        # 3. 培地に存在する代謝物の取り込み制約を設定
        for met_id, concentration in metabolite_concentrations.items():
            if met_id in ['h2o_e', 'h_e']: continue

            if met_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][met_id]
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                    orig_lb, _ = self.original_bounds[species_name].get(rxn_id, (-1000.0, 1000.0))
                    Km = 0.01 if met_id in ['glc__D_e', 'o2_e', 'pi_e', 'nh4_e'] else 0.1
                    v_max_kinetics = max_uptake * concentration / (Km + concentration)

                    current_biomass = max(1e-6, self.state.species[species_name].biomass)
                    
                    if met_id == 'o2_e':
                        o2_sat = 0.25
                        o2_supply_capacity = max(1.0, dynamic_kla * (o2_sat - min(o2_sat, concentration)))
                        max_physical_flux = o2_supply_capacity / current_biomass
                    elif met_id in ['rubber_fragment_e', 'odtd_e', 'C30_oligo_e', 'rubber_bulk_e']:
                        max_physical_flux = 1000.0
                    else:
                        max_physical_flux = concentration / (current_biomass * self.dt)

                    # 数値的安定化: 非現実的な巨大フラックスをキャップ
                    effective_max_physical = min(max_physical_flux, max_uptake * 10.0)
                    uptake_limit = min(v_max_kinetics, effective_max_physical)

                    combined_lb = max(-uptake_limit, orig_lb)
                    rxn.lower_bound = min(0.0, combined_lb, rxn.upper_bound - self.MARGIN)
                except: continue

        # --- 科学的修正: 生存維持のための微量アミノ酸供給 (Cryptic Growth) ---
        # 培地から完全に栄養が尽きても、ゴム分解酵素を維持できるよう極微量の供給を全菌種に許可
        for aa_met in ['arg__L_e', 'trp__L_e', 'leu__L_e']:
            if aa_met in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][aa_met]
                rxn = model.reactions.get_by_id(rxn_id)
                # 維持レベル (0.01 mmol/gDW/h)
                rxn.lower_bound = min(-0.01, rxn.lower_bound)
    
    def solve_fba(self, species_name: str) -> Optional[cobra.Solution]:
        model = self.models[species_name]
        try:
            solution = model.optimize()
            if solution.status == 'optimal':
                if not np.isfinite(solution.objective_value):
                    return None
                return solution
            return None
        except Exception:
            return None

    def update_biomass(self, species_name: str, growth_rate: float) -> float:
        species_state = self.state.species[species_name]
        inhibition_factor = 1.0
        if growth_rate < 0:
            effective_growth_rate = growth_rate
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
            death_rate = 0.0
            if current_ph > 10.5: death_rate = 0.2 * (current_ph - 10.5) / (11.9 - 10.5)
            elif current_ph < 4.0: death_rate = 0.1 * (4.0 - current_ph) / (4.0 - 2.0)
            if current_ph > 12.0 or current_ph < 3.0: inhibition_factor = 0.0
            effective_growth_rate = (growth_rate * inhibition_factor) - death_rate

        effective_growth_rate = np.clip(effective_growth_rate, -0.5, 2.0)
        total_biomass = sum(s.biomass for s in self.state.species.values())
        capacity_factor = max(0.0, 1.0 - total_biomass / self.carrying_capacity)
        dX = effective_growth_rate * species_state.biomass * (capacity_factor if effective_growth_rate > 0 else 1.0) * self.dt
        species_state.biomass = np.clip(species_state.biomass + dX, 0.0, self.carrying_capacity)
        species_state.growth_rate = effective_growth_rate
        species_state.last_inhibition_factor = inhibition_factor
        return effective_growth_rate
    
    def _update_environment(self, species_solutions: Dict[str, Dict[str, float]]):
        total_delta_metabolites = {}
        net_h_flux_mmol = 0.0
        total_ye_uptake = 0.0  # 酵母エキスの総消費量を追跡
        
        for s in self.state.species.values():
            s.metabolite_uptake.clear()
            s.metabolite_secretion.clear()

        for species_name, fluxes in species_solutions.items():
            biomass = self.state.species[species_name].biomass
            for met_id, rxn_id in self.exchange_reactions[species_name].items():
                actual_rid = rxn_id
                if actual_rid not in fluxes and met_id == 'rubber_e':
                    for fallback in ['EX_rubber_e', 'R_EX_rubber_e', 'rubber_high_e']:
                        if fallback in fluxes: actual_rid = fallback; break
                            
                flux = fluxes.get(actual_rid, 0.0)
                delta = flux * biomass * self.dt / self.volume
                
                # 酵母エキス成分の消費を追跡
                if flux < -1e-9 and met_id in self.YE_COMPONENTS:
                    total_ye_uptake += abs(delta)

                if met_id == 'h_e': net_h_flux_mmol += delta
                if met_id != 'h_e':
                    total_delta_metabolites[met_id] = total_delta_metabolites.get(met_id, 0.0) + delta
                
                if flux < -1e-9: self.state.species[species_name].metabolite_uptake[met_id] = -flux
                elif flux > 1e-9: self.state.species[species_name].metabolite_secretion[met_id] = flux
            
            pha_rxn_id = self.exchange_reactions[species_name].get('pha_c')
            if pha_rxn_id:
                pha_flux = fluxes.get(pha_rxn_id, 0.0)
                if pha_flux > 0: self.state.species[species_name].pha_accumulated += pha_flux * biomass * self.dt
        
        # 酵母エキスの物理的な減少を適用
        if 'yeast_extract_e' in self.state.metabolites:
            self.state.metabolites['yeast_extract_e'] = max(0.0, self.state.metabolites['yeast_extract_e'] - total_ye_uptake)

        for met_id, delta in total_delta_metabolites.items():
            if met_id == 'yeast_extract_e': continue # 既に処理済み
            if met_id in self.state.metabolites: self.state.metabolites[met_id] = max(0.0, self.state.metabolites[met_id] + delta)
            elif delta > 0: self.state.metabolites[met_id] = delta

        self.buffer_base -= net_h_flux_mmol
        self.buffer_acid += net_h_flux_mmol
        self.buffer_base = np.clip(self.buffer_base, 0.001, self.buffer_total - 0.001)
        self.buffer_acid = self.buffer_total - self.buffer_base
        current_ph = self.pKa + np.log10(self.buffer_base / self.buffer_acid)
        self.state.metabolites['h_e'] = 10**(3.0 - current_ph)

    def degrade_rubber(self, degradation_rates: Dict[str, float]):
        """
        細胞外酵素によるゴムポリマーの物理的切断とオリゴマー生成
        FBAの利己的最適化に依存せず、バイオマスと環境条件に基づき計算する
        """
        total_degradation_g = 0.0
        
        # 1. バイオサーファクタントによる分解ブースト (溶解性向上)
        bs_conc_mM = self.state.metabolites.get('biosurfactant_e', 0.0)
        bs_boost = 1.0 + (0.5 * bs_conc_mM / (0.01 + bs_conc_mM))
        
        # 2. カタボライト抑制 (グルコースが多すぎると酵素分泌が止まる)
        glc_conc = self.state.metabolites.get('glc__D_e', 0.0)
        K_inhibition = 0.5
        repression_factor = K_inhibition / (K_inhibition + glc_conc)

        rubber_conc = self.state.rubber_concentration
        if rubber_conc < 1e-6: return
            
        # Vmax (mmol of C5 units / gDW / h) 
        # 信号強化のため、より高活性な酵素活性を想定 (5倍に引き上げ)
        v_max_lcp = 2.5 # OR16 (Latex Clearing Protein)
        v_max_rox = 1.5 # NS21 (Rubber Oxygenase)

        # 3. 各菌のバイオマスに基づき、細胞外での物理的切断を計算
        # 部分一致で菌種を特定し、それぞれの活性を合計
        for species_name, species_state in self.state.species.items():
            # OR16 (LCP): Endo-cleavage -> C30オリゴマー
            if 'OR16' in species_name:
                or16_biomass = species_state.biomass
                or16_flux = v_max_lcp * or16_biomass * (rubber_conc / (1.0 + rubber_conc)) * repression_factor * bs_boost
                
                deg_g_or16 = (or16_flux * 68.12 / 1000.0) * self.dt
                if deg_g_or16 > 0:
                    # 6つのC5ユニットから1つのC30オリゴマーが生成される
                    c30_produced = (or16_flux / 6.0) * self.dt / self.volume
                    self.state.metabolites['C30_oligo_e'] = self.state.metabolites.get('C30_oligo_e', 0.0) + c30_produced
                    total_degradation_g += deg_g_or16

            # NS21 (Rox): RoxA/B -> ODTD (C15)
            elif 'NS21' in species_name:
                ns21_biomass = species_state.biomass
                ns21_flux = v_max_rox * ns21_biomass * (rubber_conc / (1.0 + rubber_conc)) * repression_factor * bs_boost
                
                deg_g_ns21 = (ns21_flux * 68.12 / 1000.0) * self.dt
                if deg_g_ns21 > 0:
                    # 3つのC5ユニットから1つのC15 (ODTD) が生成される
                    odtd_produced = (ns21_flux / 3.0) * self.dt / self.volume
                    self.state.metabolites['odtd_e'] = self.state.metabolites.get('odtd_e', 0.0) + odtd_produced
                    total_degradation_g += deg_g_ns21
                
        self.state.rubber_concentration = max(0.0, self.state.rubber_concentration - total_degradation_g)
            
    def _log_telemetry(self):
        if not self.data_log_path: return
        file_exists = os.path.exists(self.data_log_path)
        with open(self.data_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fieldnames)
            if not self._log_header_initialized and not file_exists:
                writer.writeheader()
                self._log_header_initialized = True
            elif not self._log_header_initialized: self._log_header_initialized = True
            for species_name, species_state in self.state.species.items():
                row = {
                    'time': self.state.time, 'species': species_name, 'growth_rate': species_state.growth_rate,
                    'biomass': species_state.biomass, 'pha_accumulated': species_state.pha_accumulated,
                    'rubber_concentration': self.state.rubber_concentration, 'cumulative_co2': self.cumulative_co2_emission,
                    'conc_glc__D_e': self.state.metabolites.get('glc__D_e', 0.0), 'conc_nh4_e': self.state.metabolites.get('nh4_e', 0.0),
                    'conc_o2_e': self.state.metabolites.get('o2_e', 0.0), 'conc_pi_e': self.state.metabolites.get('pi_e', 0.0),
                    'conc_arg__L_e': self.state.metabolites.get('arg__L_e', 0.0), 'conc_trp__L_e': self.state.metabolites.get('trp__L_e', 0.0),
                    'conc_leu__L_e': self.state.metabolites.get('leu__L_e', 0.0), 'conc_rubber_fragment_e': self.state.metabolites.get('rubber_fragment_e', 0.0),
                    'conc_odtd_e': self.state.metabolites.get('odtd_e', 0.0), 'conc_h2o_e': self.state.metabolites.get('h2o_e', 0.0),
                    'conc_h_e': self.state.metabolites.get('h_e', 0.0001), 'flux_EX_co2_e': species_state.metabolite_secretion.get('co2_e', 0.0),
                    'flux_EX_pha_c': species_state.metabolite_secretion.get('pha_c', 0.0), 'flux_EX_phb_c': species_state.metabolite_secretion.get('phb_c', 0.0),
                    'flux_EX_rubber_fragment_e': species_state.metabolite_secretion.get('rubber_fragment_e', 0.0), 'flux_EX_odtd_e': species_state.metabolite_secretion.get('odtd_e', 0.0)
                }
                writer.writerow(row)

    def step(self, rubber_degradation_rates: Dict[str, float], nutrient_supplementation: Dict[str, float], dynamic_kla: float = 50.0) -> ConsortiumState:
        o2_sat = 0.25
        current_o2 = self.state.metabolites.get('o2_e', o2_sat)
        self.state.metabolites['o2_e'] = o2_sat - (o2_sat - current_o2) * np.exp(-dynamic_kla * self.dt)
        supplementation_map = {'sn_or16': 'mlttr_e', 'sn_ns21': 'ptrc_e', 'sn_lp': 'mnl_e'}
        for nut_name, met_id in supplementation_map.items():
            supplement = np.clip(nutrient_supplementation.get(nut_name, 0.0), 0.0, 10.0)
            self.state.metabolites[met_id] = self.state.metabolites.get(met_id, 0.0) + supplement
        
        if 'yeast_extract' in nutrient_supplementation:
            ye_amount = np.clip(nutrient_supplementation['yeast_extract'], 0.0, 10.0)
            ye_composition = {
                'glc__D_e': 1.0, 'nh4_e': 0.5, 'arg__L_e': 0.1, 'trp__L_e': 0.05, 'leu__L_e': 0.1, 'ile__L_e': 0.1, 'val__L_e': 0.1,
                'lys__L_e': 0.1, 'met__L_e': 0.05, 'phe__L_e': 0.05, 'his__L_e': 0.05, 'tyr__L_e': 0.05,
                'thr__L_e': 0.1, 'cys__L_e': 0.05, 'ala__L_e': 0.1, 'asp__L_e': 0.1, 'glu__L_e': 0.1,
                'gly_e': 0.1, 'pro__L_e': 0.1, 'ser__L_e': 0.1, 'asn__L_e': 0.1, 'gln__L_e': 0.1,
                'nac_e': 0.01, 'ribflv_e': 0.01, 'pnto__R_e': 0.01, 'thm_e': 0.01, 'btn_e': 0.001, '4abz_e': 0.01, 'fol_e': 0.001, 'nicnt_e': 0.01,
                'ade_e': 0.01, 'gua_e': 0.01, 'ura_e': 0.01, 'cytd_e': 0.01
            }
            for met_id, coeff in ye_composition.items():
                self.state.metabolites[met_id] = self.state.metabolites.get(met_id, 0.0) + ye_amount * coeff

        total_co2_flux = 0.0
        species_solutions = {}
        for species_name in self.models.keys():
            self.set_uptake_constraints(species_name, self.state.metabolites, dynamic_kla)
            model = self.models[species_name]
            pha_rxn_id = self.exchange_reactions[species_name].get('pha_c')
            if 'NS21' in species_name:
                env_nh4 = self.state.metabolites.get('nh4_e', 0.0)
                if env_nh4 < 0.1 and pha_rxn_id: model.objective = pha_rxn_id
                else:
                    if 'R_Growth' in model.reactions: model.objective = 'R_Growth'
                    elif 'Growth' in model.reactions: model.objective = 'Growth'
            elif 'OR16' in species_name:
                if 'R_Growth' in model.reactions: model.objective = 'R_Growth'
                elif 'Growth' in model.reactions: model.objective = 'Growth'
            solution = self.solve_fba(species_name)
            if solution is not None:
                if 'R_Growth' in solution.fluxes: raw_mu = solution.fluxes['R_Growth']
                elif 'Growth' in solution.fluxes: raw_mu = solution.fluxes['Growth']
                elif 'R_BIOMASS_LLA' in solution.fluxes: raw_mu = solution.fluxes['R_BIOMASS_LLA']
                elif 'BIOMASS_LLA' in solution.fluxes: raw_mu = solution.fluxes['BIOMASS_LLA']
                else: raw_mu = solution.objective_value
                eff_mu = self.update_biomass(species_name, raw_mu)
                factor = self.state.species[species_name].last_inhibition_factor
                species_solutions[species_name] = {k: v * factor for k, v in solution.fluxes.items()}                
                co2_rxn = self.exchange_reactions[species_name].get('co2_e')
                if co2_rxn: total_co2_flux += species_solutions[species_name][co2_rxn] * self.state.species[species_name].biomass
            else: self.update_biomass(species_name, -0.01)

        self._update_environment(species_solutions)
        self.degrade_rubber(rubber_degradation_rates)
        self.cumulative_co2_emission += total_co2_flux * self.dt
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
