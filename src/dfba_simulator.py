"""
Dynamic Flux Balance Analysis (dFBA) Simulator
天然ゴム分解微生物コンソーシアムのdFBAシミュレーション
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cobra
from cobra.flux_analysis import flux_variability_analysis


@dataclass
class SpeciesState:
    """各微生物種の状態"""
    biomass: float  # バイオマス濃度 [g/L]
    growth_rate: float  # 増殖速度 [1/h]
    metabolite_uptake: Dict[str, float]  # 代謝物取り込み速度 [mmol/gDW/h]
    metabolite_secretion: Dict[str, float]  # 代謝物分泌速度 [mmol/gDW/h]


@dataclass
class ConsortiumState:
    """コンソーシアム全体の状態"""
    time: float  # 時刻 [h]
    species: Dict[str, SpeciesState]  # 各種の状態
    metabolites: Dict[str, float]  # 環境中の代謝物濃度 [mM]
    rubber_concentration: float  # 天然ゴム濃度 [g/L]


class dFBASimulator:
    """dFBAシミュレーター"""
    
    def __init__(
        self,
        models: Dict[str, cobra.Model],
        initial_biomass: Dict[str, float],
        initial_metabolites: Dict[str, float],
        initial_rubber: float,
        volume: float = 1.0,
        dt: float = 0.1
    ):
        """
        Args:
            models: 各種のCOBRAモデル {species_name: model}
            initial_biomass: 初期バイオマス濃度 {species_name: concentration [g/L]}
            initial_metabolites: 初期代謝物濃度 {metabolite_id: concentration [mM]}
            initial_rubber: 初期天然ゴム濃度 [g/L]
            volume: 培養液量 [L]
            dt: 時間ステップ [h]
        """
        self.models = models
        self.volume = volume
        self.dt = dt
        
        # 初期状態の設定
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
        
        # 代謝物の交換反応IDマッピング
        self.exchange_reactions = self._identify_exchange_reactions()
    
    def _identify_exchange_reactions(self) -> Dict[str, Dict[str, str]]:
        """
        各モデルの交換反応を特定
        
        Returns:
            {species_name: {metabolite_id: exchange_reaction_id}}
        """
        exchange_map = {}
        for species_name, model in self.models.items():
            exchange_map[species_name] = {}
            for rxn in model.exchanges:
                # 交換反応から代謝物IDを抽出
                if len(rxn.metabolites) == 1:
                    met = list(rxn.metabolites.keys())[0]
                    exchange_map[species_name][met.id] = rxn.id
        return exchange_map
    
    def set_uptake_constraints(
        self,
        species_name: str,
        metabolite_concentrations: Dict[str, float],
        max_uptake_rate: float = 10.0
    ):
        """
        代謝物濃度に基づいて取り込み制約を設定
        
        Args:
            species_name: 種名
            metabolite_concentrations: 代謝物濃度 {metabolite_id: concentration [mM]}
            max_uptake_rate: 最大取り込み速度 [mmol/gDW/h]
        """
        model = self.models[species_name]
        
        for met_id, concentration in metabolite_concentrations.items():
            if met_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][met_id]
                rxn = model.reactions.get_by_id(rxn_id)
                
                # Monod式による取り込み速度の制限
                Km = 0.1  # 半飽和定数 [mM]
                uptake_limit = max_uptake_rate * concentration / (Km + concentration)
                
                # 取り込み反応の下限を設定（負の値 = 取り込み）
                rxn.lower_bound = -uptake_limit
    
    def solve_fba(self, species_name: str) -> Optional[cobra.Solution]:
        """
        単一種のFBAを解く
        
        Args:
            species_name: 種名
        
        Returns:
            FBA解（失敗時はNone）
        """
        model = self.models[species_name]
        
        try:
            solution = model.optimize()
            if solution.status == 'optimal':
                return solution
            else:
                return None
        except Exception as e:
            print(f"FBA failed for {species_name}: {e}")
            return None
    
    def update_biomass(self, species_name: str, growth_rate: float):
        """
        バイオマスを更新
        
        Args:
            species_name: 種名
            growth_rate: 増殖速度 [1/h]
        """
        species_state = self.state.species[species_name]
        species_state.biomass *= np.exp(growth_rate * self.dt)
        species_state.growth_rate = growth_rate
    
    def update_metabolites(
        self,
        species_name: str,
        solution: cobra.Solution
    ):
        """
        代謝物濃度を更新
        
        Args:
            species_name: 種名
            solution: FBA解
        """
        species_state = self.state.species[species_name]
        biomass = species_state.biomass
        
        # 交換反応のフラックスを取得
        for met_id, rxn_id in self.exchange_reactions[species_name].items():
            flux = solution.fluxes[rxn_id]  # mmol/gDW/h
            
            # 環境中の代謝物濃度を更新
            # flux > 0: 分泌, flux < 0: 取り込み
            delta_concentration = flux * biomass * self.dt / self.volume
            
            if met_id in self.state.metabolites:
                self.state.metabolites[met_id] += delta_concentration
                self.state.metabolites[met_id] = max(0, self.state.metabolites[met_id])
            
            # 取り込み・分泌速度を記録
            if flux < 0:
                species_state.metabolite_uptake[met_id] = -flux
            elif flux > 0:
                species_state.metabolite_secretion[met_id] = flux
    
    def degrade_rubber(self, degradation_rates: Dict[str, float]):
        """
        天然ゴムの分解
        
        Args:
            degradation_rates: 各種の分解速度 {species_name: rate [g/gDW/h]}
        """
        total_degradation = 0.0
        
        for species_name, rate in degradation_rates.items():
            biomass = self.state.species[species_name].biomass
            degradation = rate * biomass * self.dt
            total_degradation += degradation
        
        self.state.rubber_concentration -= total_degradation
        self.state.rubber_concentration = max(0, self.state.rubber_concentration)
        
        # 分解産物（イソプレノイド）を環境に追加
        # 簡略化: ゴム1gから0.5 mmolのイソプレノイドが生成されると仮定
        isoprene_yield = 0.5  # mmol/g
        isoprene_produced = total_degradation * isoprene_yield / self.volume
        
        if 'isoprene' in self.state.metabolites:
            self.state.metabolites['isoprene'] += isoprene_produced
    
    def step(
        self,
        rubber_degradation_rates: Dict[str, float],
        amino_acid_supplementation: Dict[str, float]
    ) -> ConsortiumState:
        """
        1タイムステップのシミュレーション
        
        Args:
            rubber_degradation_rates: ゴム分解速度 {species_name: rate}
            amino_acid_supplementation: アミノ酸補給 {amino_acid_id: concentration [mM]}
        
        Returns:
            更新後のコンソーシアム状態
        """
        # アミノ酸補給
        for aa_id, conc in amino_acid_supplementation.items():
            if aa_id in self.state.metabolites:
                self.state.metabolites[aa_id] += conc
            else:
                self.state.metabolites[aa_id] = conc
        
        # 各種のFBAを解く
        for species_name in self.models.keys():
            # 取り込み制約を設定
            self.set_uptake_constraints(species_name, self.state.metabolites)
            
            # FBAを解く
            solution = self.solve_fba(species_name)
            
            if solution is not None:
                # バイオマス更新
                growth_rate = solution.objective_value
                self.update_biomass(species_name, growth_rate)
                
                # 代謝物濃度更新
                self.update_metabolites(species_name, solution)
        
        # ゴム分解
        self.degrade_rubber(rubber_degradation_rates)
        
        # 時刻を進める
        self.state.time += self.dt
        
        return self.state
    
    def get_state_vector(self) -> np.ndarray:
        """
        状態ベクトルを取得（RL環境用）
        
        Returns:
            状態ベクトル
        """
        state_vec = []
        
        # バイオマス濃度
        for species_name in sorted(self.models.keys()):
            state_vec.append(self.state.species[species_name].biomass)
        
        # 増殖速度
        for species_name in sorted(self.models.keys()):
            state_vec.append(self.state.species[species_name].growth_rate)
        
        # 主要代謝物濃度
        key_metabolites = ['isoprene', 'arginine', 'tryptophan', 'leucine']
        for met_id in key_metabolites:
            state_vec.append(self.state.metabolites.get(met_id, 0.0))
        
        # ゴム濃度
        state_vec.append(self.state.rubber_concentration)
        
        return np.array(state_vec, dtype=np.float32)
