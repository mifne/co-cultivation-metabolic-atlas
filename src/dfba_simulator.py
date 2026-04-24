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
        
        # モデルの初期培地条件を設定
        self._initialize_medium()
    
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
            
            print(f"  📊 {species_name}: {len(exchange_map[species_name])}個の交換反応を検出")
            
        return exchange_map
    
    def _initialize_medium(self):
        """
        各モデルの初期培地条件を設定（豊富培地）
        """
        print("\n  🧪 初期培地条件を設定中...")
        
        for species_name, model in self.models.items():
            # 豊富培地: 全ての交換反応を開放
            medium = {}
            
            for rxn in model.exchanges:
                # 取り込み反応（負の下限）のみを開放
                if rxn.lower_bound < 0:
                    # 元の下限を保持（制約を緩めない）
                    medium[rxn.id] = abs(rxn.lower_bound)
            
            # mediumを設定
            try:
                model.medium = medium
                print(f"    ✅ {species_name}: {len(medium)}個の交換反応を開放")
            except Exception as e:
                print(f"    ⚠️  {species_name}: 培地設定エラー - {e}")
            
            # 初期FBAテスト
            try:
                solution = model.optimize()
                if solution.status == 'optimal':
                    print(f"    ✅ {species_name}: 初期FBA成功 (growth={solution.objective_value:.4f})")
                else:
                    print(f"    ⚠️  {species_name}: 初期FBA失敗 (status={solution.status})")
                    # 診断情報を出力
                    self._diagnose_infeasibility(species_name, model)
            except Exception as e:
                print(f"    ❌ {species_name}: 初期FBAエラー - {e}")
    
    def _diagnose_infeasibility(self, species_name: str, model: cobra.Model):
        """
        FBA失敗の原因を診断
        
        Args:
            species_name: 種名
            model: COBRAモデル
        """
        print(f"    🔍 {species_name} の診断中...")
        
        # 1. バイオマス反応の確認
        if model.objective:
            obj_rxn = list(model.objective.variables.keys())[0]
            print(f"      目的関数: {obj_rxn.id}")
            print(f"      境界: [{obj_rxn.lower_bound}, {obj_rxn.upper_bound}]")
        
        # 2. 閉じている必須交換反応を探す
        essential_metabolites = ['glc__D_e', 'o2_e', 'nh4_e', 'pi_e', 'h2o_e']
        for met_id in essential_metabolites:
            if met_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][met_id]
                rxn = model.reactions.get_by_id(rxn_id)
                if rxn.lower_bound == 0 and rxn.upper_bound == 0:
                    print(f"      ⚠️  {met_id} の交換反応が閉じています: {rxn_id}")
        
        # 3. 制約の緩和を試行
        print(f"      🔧 制約緩和テスト中...")
        with model:
            # 全ての交換反応を完全に開放
            for rxn in model.exchanges:
                rxn.lower_bound = -1000
                rxn.upper_bound = 1000
            
            try:
                solution = model.optimize()
                if solution.status == 'optimal':
                    print(f"      ✅ 制約緩和後は最適化成功 → 培地条件が原因")
                else:
                    print(f"      ❌ 制約緩和後も失敗 → モデル構造の問題")
            except Exception as e:
                print(f"      ❌ 制約緩和テストエラー: {e}")
    
    def set_uptake_constraints(
        self,
        species_name: str,
        metabolite_concentrations: Dict[str, float],
        max_uptake_rate: float = 20.0
    ):
        """
        代謝物濃度に基づいて取り込み制約を設定
        
        Args:
            species_name: 種名
            metabolite_concentrations: 代謝物濃度 {metabolite_id: concentration [mM]}
            max_uptake_rate: 最大取り込み速度 [mmol/gDW/h]
        """
        model = self.models[species_name]
        
        # 交換反応を探索（複数のID形式に対応）
        for met_id, concentration in metabolite_concentrations.items():
            # 濃度が極めて低い場合はスキップ
            if concentration < 0.001:
                continue
            
            # 代替IDも試行（例: glc_e と glc__D_e）
            possible_ids = [met_id]
            if '_e' in met_id and '__' not in met_id:
                # glc_e -> glc__D_e のような変換を試行
                base = met_id.replace('_e', '')
                possible_ids.append(f"{base}__D_e")
                possible_ids.append(f"{base}__L_e")
            
            for possible_id in possible_ids:
                if possible_id in self.exchange_reactions[species_name]:
                    rxn_id = self.exchange_reactions[species_name][possible_id]
                    try:
                        rxn = model.reactions.get_by_id(rxn_id)
                        
                        # 元の下限を保存
                        original_lower = rxn.lower_bound
                        
                        # Monod式による取り込み速度の制限
                        Km = 0.5  # 半飽和定数 [mM]
                        uptake_limit = max_uptake_rate * concentration / (Km + concentration)
                        
                        # 取り込み反応の下限を設定（負の値 = 取り込み）
                        new_lower_bound = -uptake_limit
                        
                        # 元の下限より緩い制約のみ適用（制約を強めない）
                        if new_lower_bound > original_lower:
                            new_lower_bound = original_lower
                        
                        # 境界値の妥当性をチェック
                        if new_lower_bound > rxn.upper_bound:
                            new_lower_bound = rxn.upper_bound
                        
                        # 安全な範囲に制限
                        new_lower_bound = max(new_lower_bound, -1000.0)
                        
                        rxn.lower_bound = new_lower_bound
                        
                        break  # 成功したらループを抜ける
                    except KeyError:
                        continue
                    except ValueError as e:
                        # 境界値エラーをキャッチ
                        print(f"  ⚠️  {species_name}/{rxn_id}: 境界値エラー - {e}")
                        continue
    
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
                # 解の妥当性をチェック
                if solution.objective_value < 0:
                    print(f"⚠️  {species_name}: 負の目的関数値 ({solution.objective_value:.4f})")
                    return None
                
                # 異常に大きい値をチェック
                if solution.objective_value > 1000:
                    print(f"⚠️  {species_name}: 異常に大きい目的関数値 ({solution.objective_value:.4f})")
                    return None
                
                return solution
            else:
                print(f"⚠️  {species_name}: FBA最適化失敗 (status: {solution.status})")
                
                # 初回失敗時のみ詳細診断（ログの氾濫を防ぐ）
                if not hasattr(self, '_diagnosed'):
                    self._diagnosed = set()
                
                if species_name not in self._diagnosed:
                    self._diagnose_infeasibility(species_name, model)
                    self._diagnosed.add(species_name)
                
                return None
                
        except Exception as e:
            print(f"❌ {species_name}: FBA解法エラー: {e}")
            return None
    
    def update_biomass(self, species_name: str, growth_rate: float):
        """
        バイオマスを更新
        
        Args:
            species_name: 種名
            growth_rate: 増殖速度 [1/h]
        """
        species_state = self.state.species[species_name]
        
        # 増殖速度を妥当な範囲にクリップ（-1.0 - 5.0 1/h）
        growth_rate = np.clip(growth_rate, -1.0, 5.0)
        
        # dX/dt = μ * X
        dX = growth_rate * species_state.biomass * self.dt
        new_biomass = species_state.biomass + dX
        
        # バイオマスを妥当な範囲にクリップ（0.001 - 100 g/L）
        new_biomass = np.clip(new_biomass, 0.001, 100.0)
        
        species_state.biomass = new_biomass
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
        # アミノ酸補給（0-10 mMの範囲にクリップ）
        # BiGG Models標準IDを使用
        amino_acid_map = {
            'arginine': 'arg__L_e',
            'tryptophan': 'trp__L_e',
            'leucine': 'leu__L_e'
        }
        
        for aa_name, met_id in amino_acid_map.items():
            supplement = amino_acid_supplementation.get(aa_name, 0.0)
            supplement = np.clip(supplement, 0.0, 10.0)
            if met_id in self.state.metabolites:
                self.state.metabolites[met_id] += supplement
                # 代謝物濃度も上限を設定（0-100 mM）
                self.state.metabolites[met_id] = np.clip(
                    self.state.metabolites[met_id], 0.0, 100.0
                )
            else:
                self.state.metabolites[met_id] = supplement
        
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
            else:
                # FBA失敗時は増殖速度を0に設定
                self.state.species[species_name].growth_rate = 0.0
        
        # ゴム分解
        self.degrade_rubber(rubber_degradation_rates)
        
        # 時刻を進める
        self.state.time += self.dt
        
        return self.state
    
    def get_state_vector(self) -> np.ndarray:
        """
        状態ベクトルを取得（RL環境用）
        
        Returns:
            状態ベクトル（正規化・クリップ済み）
        """
        state_vec = []
        
        # バイオマス濃度（0-100 g/Lの範囲にクリップ）
        for species_name in sorted(self.models.keys()):
            biomass = self.state.species[species_name].biomass
            biomass = np.clip(biomass, 0.0, 100.0)
            state_vec.append(biomass)
        
        # 増殖速度（-1.0 - 10.0 1/hの範囲にクリップ）
        for species_name in sorted(self.models.keys()):
            growth_rate = self.state.species[species_name].growth_rate
            growth_rate = np.clip(growth_rate, -1.0, 10.0)
            state_vec.append(growth_rate)
        
        # 主要代謝物濃度（0-100 mMの範囲にクリップ）
        key_metabolites = ['isoprene', 'arg__L_e', 'trp__L_e', 'leu__L_e']
        for met_id in key_metabolites:
            conc = self.state.metabolites.get(met_id, 0.0)
            conc = np.clip(conc, 0.0, 100.0)
            state_vec.append(conc)
        
        # ゴム濃度（0-100 g/Lの範囲にクリップ）
        rubber = np.clip(self.state.rubber_concentration, 0.0, 100.0)
        state_vec.append(rubber)
        
        # NaNやInfをチェックして置き換え
        state_array = np.array(state_vec, dtype=np.float64)
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=100.0, neginf=0.0)
        
        return state_array.astype(np.float32)
