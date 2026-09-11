"""
Dynamic Flux Balance Analysis (dFBA) Simulator
天然ゴム分解微生物コンソーシアムのdFBAシミュレーション
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cobra
import pandas as pd
from cobra.core.solution import Solution
from cobra.util.array import create_stoichiometric_matrix
from cobra.flux_analysis import flux_variability_analysis
import json
from pathlib import Path
import os
import csv
import warnings
import time

from .utils import COEXISTENCE_DEFINED_FEED_MMOL_L_H

from .cuopt_solver import CuOptConfig, CuOptFbaSolver, CuOptUnavailable
from .community_solver import CooperativeCommunityFbaSolver, JointCommunityFbaSolver
from .frozen_community_inputs import (
    FrozenCommunityInputBuilder,
    FrozenCommunityStepInputs,
)
from .metabolite_ids import canonical_metabolite_id
from .fba_surrogate import (
    FbaSurrogateBackend,
    SurrogatePrediction,
    SurrogateUnavailable,
    extract_lp_features,
)

@dataclass
class SpeciesState:
    """各微生物種の状態"""
    biomass: float  # バイオマス濃度 [g/L]
    growth_rate: float  # 増殖速度 [1/h]
    metabolite_uptake: Dict[str, float]  # 代謝物取り込み速度 [mmol/gDW/h]
    metabolite_secretion: Dict[str, float]  # 代謝物分泌速度 [mmol/gDW/h]
    pha_accumulated: float = 0.0  # total intracellular PHA repeat units [mmol/L]
    phb_accumulated: float = 0.0  # intracellular 3HB repeat units [mmol/L]
    phv_accumulated: float = 0.0  # intracellular 3HV repeat units [mmol/L]


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
    BOUND_EPS = 1e-6  # genuinely inverted boundsを修復する際の数値ガード

    # Extracellular rubber-cleavage defaults. Units are substrate-equivalent
    # mmol gDW^-1 h^-1 (C5 for bulk routes, C30 for the oligo RoxA route).
    # These are explicit calibration parameters, not genome-derived kcat data.
    DEFAULT_POLYMER_RATES = {
        "lcp_c5": 2.5,
        "roxb_c5": 0.75,
        "roxa_direct_c5": 0.75,
        "roxa_oligo_c30": 0.25,
    }
    RUBBER_C5_MOLAR_MASS_G_PER_MOL = 68.12
    PHB_REPEAT_G_PER_MMOL = 0.08609
    PHV_REPEAT_G_PER_MMOL = 0.10012


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

    # 炭素を多く含み、炭素源として誤用されやすい成分
    CARBON_RICH_YE = [
        'glu__L_e', 'leu__L_e', 'ump_e', 'xtsn_e', 'ura_e', 'tyrp_e'
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
                 metabolite_inhibition_decay: float = 0.1,
                 solver_backend: Optional[str] = None,
                 cuopt_method: str = "pdlp",
                 fba_mode: str = "separate",
                 surrogate_dir: Optional[str] = None,
                 surrogate_device: str = "cuda",
                 surrogate_service: Optional[Any] = None,
                 surrogate_audit_interval: int = 128,
                 surrogate_objective_rtol: float = 0.02,
                 surrogate_objective_atol: float = 1e-5,
                 surrogate_ood_threshold: float = 8.0,
                 polymer_oxygen_fraction: float = 0.25,
                 ph_control_target: Optional[float] = 6.5,
                 cooperative_parsimony: bool = True,
                 cooperative_optimize_live_objectives: bool = False,
                 cooperative_highs_presolve: bool = True,
                 cooperative_highs_method: str = "highs",
                 cooperative_capture_training_snapshot: bool = False,
                 cooperative_surrogate_artifact: Optional[str] = None,
                 cooperative_surrogate_device: str = "cuda",
                 cooperative_surrogate_top_k: int = 16,
                 cooperative_surrogate_interpolation_k: int = 1,
                 cooperative_surrogate_interpolation_power: float = 2.0,
                 cooperative_surrogate_exact_interval: int = 128,
                 cooperative_surrogate_distance_threshold: Optional[float] = None,
                 cooperative_surrogate_require_qualified: bool = True,
                 cooperative_surrogate_validation_manifest: Optional[str] = None,
                 cooperative_gpu_qp_projection: bool = False,
                 cooperative_gpu_qp_only: bool = False,
                 cooperative_gpu_qp_candidates: int = 64,
                 cooperative_gpu_qp_rerank_pool: int | None = None,
                 cooperative_gpu_qp_decision_strength: float | None = None,
                 cooperative_gpu_qp_blend_candidates: int = 1,
                 cooperative_gpu_qp_blend_distance_power: float = 2.0,
                 cooperative_gpu_qp_max_iterations: int = 400,
                 cooperative_gpu_qp_normalized_tolerance: float = 2e-5,
                 cooperative_gpu_qp_match_pha: bool = False,
                 cooperative_gpu_qp_multioutput_strength: float = 0.0,
                 cooperative_gpu_qp_independent_species: bool = False,
                 cooperative_gpu_qp_service: Optional[Any] = None,
                 max_pha_fraction_g_gdcw: float = 0.80,
                 phv_requires_rubber_intermediate: bool = True,
                 cooperative_linear_program_backend: Optional[Any] = None,
                 cooperative_frozen_inputs: bool = False):
        self.models = models
        self.dt = dt
        self.volume = volume
        self.data_log_path = data_log_path
        self.max_uptake_rate = max_uptake_rate
        self.carrying_capacity = carrying_capacity
        self.metabolite_inhibition_threshold = metabolite_inhibition_threshold
        self.metabolite_inhibition_decay = metabolite_inhibition_decay
        # ``glpk`` remains the reproducible default. ``auto`` tries cuOpt and
        # falls back to GLPK when the optional NVIDIA wheel/GPU is unavailable.
        self.solver_backend = (solver_backend or os.environ.get("DFBA_SOLVER", "glpk")).lower()
        if self.solver_backend not in {
            "glpk", "highs", "cuopt", "auto", "surrogate"
        }:
            raise ValueError(
                "solver_backend must be one of: glpk, highs, cuopt, auto, surrogate"
            )
        self.cuopt_method = cuopt_method
        self.cuopt_time_limit = float(
            os.environ.get("DFBA_CUOPT_TIME_LIMIT", "5.0")
        )
        self.fba_mode = (fba_mode or "separate").lower()
        if self.fba_mode not in {"separate", "joint", "cooperative"}:
            raise ValueError("fba_mode must be 'separate', 'joint', or 'cooperative'")
        self.cooperative_frozen_inputs = bool(cooperative_frozen_inputs)
        if self.cooperative_frozen_inputs and self.fba_mode != "cooperative":
            raise ValueError(
                "cooperative_frozen_inputs requires fba_mode='cooperative'"
            )
        if self.solver_backend == "surrogate" and self.fba_mode != "separate":
            raise ValueError("the surrogate backend requires fba_mode='separate'")
        self.assigned_gpu = os.environ.get("DFBA_ASSIGNED_GPU")
        self.assigned_gpu_slot = os.environ.get("DFBA_GPU_SLOT")
        self.surrogate_dir = surrogate_dir or os.environ.get("DFBA_SURROGATE_DIR")
        self.surrogate_device = surrogate_device
        self.surrogate_service = surrogate_service
        self.surrogate_audit_interval = max(0, int(surrogate_audit_interval))
        self.surrogate_objective_rtol = float(surrogate_objective_rtol)
        self.surrogate_objective_atol = float(surrogate_objective_atol)
        self.surrogate_ood_threshold = float(surrogate_ood_threshold)
        self.polymer_oxygen_fraction = float(
            np.clip(polymer_oxygen_fraction, 0.0, 1.0)
        )
        if not 0.0 <= float(max_pha_fraction_g_gdcw) < 1.0:
            raise ValueError("max_pha_fraction_g_gdcw must be in [0, 1)")
        # Provisional safety ceiling, not an NS21 measurement.  Biomass is
        # treated as residual (non-PHA) cell mass, so f = P/(X + P).
        self.max_pha_fraction_g_gdcw = float(max_pha_fraction_g_gdcw)
        # NS21 produced PHBV from natural rubber but PHB from glucose in the
        # 2024 GC/NMR study.  Keep this phenotype guard configurable for future
        # defined 3HV-precursor feeding experiments.
        self.phv_requires_rubber_intermediate = bool(
            phv_requires_rubber_intermediate
        )
        self.ph_control_target = (
            None if ph_control_target is None
            else float(np.clip(ph_control_target, 0.0, 14.0))
        )
        self.cooperative_parsimony = bool(cooperative_parsimony)
        self.cooperative_optimize_live_objectives = bool(
            cooperative_optimize_live_objectives
        )
        self.cooperative_highs_presolve = bool(cooperative_highs_presolve)
        self.cooperative_highs_method = str(cooperative_highs_method)
        self.cooperative_linear_program_backend = cooperative_linear_program_backend
        self.cooperative_capture_training_snapshot = bool(
            cooperative_capture_training_snapshot
        )
        self.cooperative_surrogate_artifact = cooperative_surrogate_artifact
        self.cooperative_surrogate_device = cooperative_surrogate_device
        self.cooperative_surrogate_top_k = max(1, int(cooperative_surrogate_top_k))
        self.cooperative_surrogate_interpolation_k = max(
            1, int(cooperative_surrogate_interpolation_k)
        )
        self.cooperative_surrogate_interpolation_power = max(
            0.25, float(cooperative_surrogate_interpolation_power)
        )
        self.cooperative_surrogate_exact_interval = max(
            0, int(cooperative_surrogate_exact_interval)
        )
        self.cooperative_surrogate_distance_threshold = (
            None
            if cooperative_surrogate_distance_threshold is None
            else float(cooperative_surrogate_distance_threshold)
        )
        self.cooperative_surrogate_require_qualified = bool(
            cooperative_surrogate_require_qualified
        )
        self.cooperative_surrogate_validation_manifest = (
            cooperative_surrogate_validation_manifest
        )
        self.cooperative_gpu_qp_projection = bool(cooperative_gpu_qp_projection)
        self.cooperative_gpu_qp_only = bool(cooperative_gpu_qp_only)
        self.cooperative_gpu_qp_candidates = max(
            2, int(cooperative_gpu_qp_candidates)
        )
        self.cooperative_gpu_qp_rerank_pool = (
            None
            if cooperative_gpu_qp_rerank_pool is None
            else max(
                self.cooperative_gpu_qp_candidates,
                int(cooperative_gpu_qp_rerank_pool),
            )
        )
        self.cooperative_gpu_qp_decision_strength = (
            None
            if cooperative_gpu_qp_decision_strength is None
            else max(0.0, float(cooperative_gpu_qp_decision_strength))
        )
        self.cooperative_gpu_qp_blend_candidates = max(
            1, int(cooperative_gpu_qp_blend_candidates)
        )
        self.cooperative_gpu_qp_blend_distance_power = max(
            0.0, float(cooperative_gpu_qp_blend_distance_power)
        )
        self.cooperative_gpu_qp_max_iterations = max(
            1, int(cooperative_gpu_qp_max_iterations)
        )
        self.cooperative_gpu_qp_normalized_tolerance = float(
            cooperative_gpu_qp_normalized_tolerance
        )
        self.cooperative_gpu_qp_match_pha = bool(cooperative_gpu_qp_match_pha)
        self.cooperative_gpu_qp_multioutput_strength = float(cooperative_gpu_qp_multioutput_strength)
        self.cooperative_gpu_qp_independent_species = bool(cooperative_gpu_qp_independent_species)
        self.cooperative_gpu_qp_service = cooperative_gpu_qp_service
        self.cumulative_base_added_mmol_l = 0.0
        self.cumulative_acid_added_mmol_l = 0.0
        self.cumulative_defined_feed_g_l = 0.0
        self.last_defined_feed_g_l = 0.0
        self._surrogate_backend: Optional[FbaSurrogateBackend] = None
        self._surrogate_load_attempted = False
        self._surrogate_objective_cache: Dict[
            str, Tuple[int, np.ndarray, float]
        ] = {}
        self.surrogate_attempts = 0
        self.surrogate_accepts = 0
        self.surrogate_rejections = 0
        self.surrogate_audits = 0
        self.surrogate_audit_failures = 0
        self.surrogate_rejection_reasons: Dict[str, int] = {}
        self.surrogate_inference_seconds = 0.0
        self._cuopt_solvers: Dict[str, CuOptFbaSolver] = {}
        self._joint_solver: Optional[JointCommunityFbaSolver] = None
        self._cooperative_solver: Optional[CooperativeCommunityFbaSolver] = None
        self._frozen_community_input_builder: Optional[
            FrozenCommunityInputBuilder
        ] = None
        self._joint_auto_fallback = False
        self._cuopt_fallback_warned = False
        self._cuopt_disabled_species: set[str] = set()
        self._cuopt_audited_species: set[str] = set()
        self.gpu_solve_attempts = 0
        self.gpu_solve_successes = 0
        self.gpu_audit_failures = 0
        self.cpu_fallback_solves = 0
        self.last_step_timing: Dict[str, float] = {}
        self.step_timing_totals: Dict[str, float] = {
            "pre_solve_seconds": 0.0,
            "solve_seconds": 0.0,
            "post_solve_seconds": 0.0,
            "total_seconds": 0.0,
        }
        self.solve_attempts = 0
        self.solve_successes = 0
        self.last_fba_solutions: Dict[str, Optional[Solution]] = {}
        self.last_polymer_fluxes: Dict[str, float] = {}
        
        # --- 科学的再構築: 培地の緩衝能 (50mM リン酸, pH7.0 -> 6.0) ---
        self.buffering_pool = 16.37 # mmol/L H+ を中和可能
        
        if self.data_log_path:
            os.makedirs(os.path.dirname(self.data_log_path), exist_ok=True)

        self.log_fieldnames = [
            'time', 'species', 'growth_rate', 'biomass', 'pha_accumulated',
            'phb_accumulated', 'phv_accumulated', 'phv_mol_fraction',
            'rubber_concentration', 'cumulative_co2',
            'conc_glc__D_e', 'conc_nh4_e', 'conc_o2_e', 'conc_pi_e', 
            'conc_arg__L_e', 'conc_trp__L_e', 'conc_leu__L_e',
            'conc_rubber_fragment_e', 'conc_odtd_e', 'conc_h2o_e', 'conc_h_e',
            'flux_EX_co2_e', 'flux_EX_pha_c', 'flux_EX_phv_c', 'flux_EX_phb_c',
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

        # Base GEMs must permit a zero exchange flux. Scenario-specific
        # measured ranges are imposed by the environment, not treated as
        # compulsory uptake/secretion in every simulation.
        for model in self.models.values():
            for reaction in model.exchanges:
                reaction.bounds = (
                    min(float(reaction.lower_bound), 0.0),
                    max(float(reaction.upper_bound), 0.0),
                )

        # Insoluble rubber cleavage is handled once, in degrade_rubber().
        # These bounds prevent the intracellular LP from independently
        # consuming the same bulk-polymer carbon.
        self._enforce_polymer_boundary_separation()
        
        self.original_bounds = {}
        for species_name, model in self.models.items():
            # 飢餓時の計算破綻を防ぐため、強制的な内部フラックス（バイオサーファクタント生成など）を解除
            for rxn in model.reactions:
                if rxn.id == 'R_GLYCOLIPOPROTEIN_SYN_SEC':
                    rxn.lower_bound = 0.0

            for rxn in model.exchanges:
                if rxn.upper_bound < 0:
                    rxn.upper_bound = 1000.0
            self.original_bounds[species_name] = {
                rxn.id: (rxn.lower_bound, rxn.upper_bound)
                for rxn in model.exchanges
            }
        
        self._initialize_medium()

        if self.cooperative_frozen_inputs:
            # Opt-in rollout contract: model structure, exchange
            # classification and static bounds/objectives are inspected once.
            # Subsequent dFBA steps update NumPy arrays, not COBRA/optlang.
            self.enable_frozen_community_inputs()

        if self.solver_backend == "surrogate":
            for species_name, model in self.models.items():
                features = extract_lp_features(model)
                reaction_count = len(model.reactions)
                self._surrogate_objective_cache[species_name] = (
                    id(model.objective),
                    features[2 * reaction_count : 3 * reaction_count].copy(),
                    float(features[-1]),
                )

    def _enforce_polymer_boundary_separation(self) -> None:
        """Keep extracellular polymer chemistry out of the steady-state LP."""

        kinetic_reactions = {
            "R_LCP",
            "R_ROXB",
            "R_ROXA",
            "R_ROXA_BULK",
        }
        for model in self.models.values():
            if "EX_rubber_bulk_e" in model.reactions:
                reaction = model.reactions.get_by_id("EX_rubber_bulk_e")
                if reaction.bounds != (0.0, 0.0):
                    reaction.bounds = (0.0, 0.0)
            for reaction_id in kinetic_reactions:
                if reaction_id in model.reactions:
                    reaction = model.reactions.get_by_id(reaction_id)
                    if reaction.bounds != (0.0, 0.0):
                        reaction.bounds = (0.0, 0.0)

    def enable_frozen_community_inputs(self):
        """Enable and prevalidate the fixed-GEM cooperative array path.

        Call this outside a timed rollout after all intended model curation and
        scenario setup. The live-editable legacy path remains the default.  A
        cooperative solver is also constructed and aligned here so neither its
        structural setup nor the first full contract fingerprint is charged to
        the first online step.
        """

        if self.fba_mode != "cooperative":
            raise ValueError(
                "cooperative_frozen_inputs requires fba_mode='cooperative'"
            )
        builder = self._frozen_community_input_builder
        if builder is None:
            builder = FrozenCommunityInputBuilder(
                self.models,
                self.exchange_reactions,
                self.original_bounds,
            )
        else:
            builder.validate_static_contract(
                self.models,
                self.exchange_reactions,
                self.original_bounds,
            )
        solver = self._get_or_create_cooperative_solver()
        solver.validate_frozen_contract(self.models, builder.contract)
        self._frozen_community_input_builder = builder
        self.cooperative_frozen_inputs = True
        return builder.contract

    def _identify_exchange_reactions(self) -> Dict[str, Dict[str, str]]:
        exchange_map = {}
        for species_name, model in self.models.items():
            exchange_map[species_name] = {}
            for rxn in model.exchanges:
                if len(rxn.metabolites) == 1:
                    met = list(rxn.metabolites.keys())[0]
                    # --- 科学制約: 多重プレフィックス (M_M_...) の再帰的除去 ---
                    met_id = canonical_metabolite_id(met.id)
                    exchange_map[species_name][met_id] = rxn.id

            # --- 特殊反応の明示的マッピング (名前が不規則な場合) ---
            # PHA 蓄積 (intracellular sink reactions are not in
            # ``model.exchanges`` even though COBRA marks them as boundary).
            for pha_id in ['EX_pha_c', 'EX_phb_c', 'R_EX_pha_c', 'EX_phb_lp']:
                if pha_id in model.reactions:
                    exchange_map[species_name]['pha_c'] = pha_id
            if 'EX_phv_c' in model.reactions:
                exchange_map[species_name]['phv_c'] = 'EX_phv_c'
            
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
        # --- 科学的再構築: リン酸緩衝系の初期化 ---
        self.buffer_total = 100.0  # mM (LP由来の乳酸に対する緩衝力を最大化)
        self.pKa = 7.21           # Phosphate pKa2
        initial_ph = self.ph_control_target if self.ph_control_target is not None else 7.0
        ratio = 10**(initial_ph - self.pKa)
        self.buffer_base = self.buffer_total * ratio / (1 + ratio)
        self.buffer_acid = self.buffer_total - self.buffer_base
        self.state.metabolites["h_e"] = 10.0 ** (3.0 - initial_ph)

        for species_name, model in self.models.items():
            # ソルバー設定の初期化
            self._configure_cpu_solver(model)

            # 一旦すべての交換反応の取り込みを制限 (培地にないものは 0)
            for rxn in model.exchanges:
                if rxn.lower_bound < 0:
                    rxn.lower_bound = self._safe_lb(0.0, rxn.upper_bound)
            
            # 培地に存在する代謝物の取り込みを許可
            for met_id in self.state.metabolites.keys():
                if met_id in self.exchange_reactions[species_name]:
                    rxn_id = self.exchange_reactions[species_name][met_id]
                    target_rxn = model.reactions.get_by_id(rxn_id)
                    
                    if rxn_id in self.original_bounds[species_name]:
                        lb, _ = self.original_bounds[species_name][rxn_id]
                        target_rxn.lower_bound = self._safe_lb(lb, target_rxn.upper_bound)
                    else:
                        target_rxn.lower_bound = self._safe_lb(-1000.0, target_rxn.upper_bound)
            
            # 水の交換反応を常に許可
            if 'h2o_e' in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name]['h2o_e']
                target_rxn = model.reactions.get_by_id(rxn_id)
                target_rxn.lower_bound = self._safe_lb(-1000.0, target_rxn.upper_bound)

        frozen_builder = getattr(self, "_frozen_community_input_builder", None)
        if frozen_builder is not None:
            # Reset is the rollout boundary at which the expensive full
            # fingerprint is intentionally rechecked. Per-step checks remain
            # array-only.
            frozen_builder.validate_static_contract(
                self.models,
                self.exchange_reactions,
                self.original_bounds,
            )

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

    def _safe_lb(self, desired_lb: float, upper_bound: float) -> float:
        """Clamp an uptake lower bound without altering fixed reactions.

        Fixed variables (``lb == ub``) are valid LP constructs.  The previous
        implementation changed every fixed zero reaction into ``[-1e-6, 0]``,
        which changed the feasible polytope and made the GPU problem much more
        degenerate.  The GLPK-only compatibility gap is applied immediately
        before a CPU solve in :meth:`_sanitize_bounds` instead.
        """
        lb = min(desired_lb, upper_bound)
        return round(lb, 6)

    def _sanitize_bounds(self, model: cobra.Model, require_strict_gap: bool = False):
        """Validate finite ordered bounds without changing fixed variables."""
        for rxn in model.reactions:
            # 数値的極限値のガード
            if not np.isfinite(rxn.lower_bound): rxn.lower_bound = -1000.0
            if not np.isfinite(rxn.upper_bound): rxn.upper_bound = 1000.0

            if rxn.lower_bound > rxn.upper_bound:
                rxn.lower_bound = rxn.upper_bound

            # GLPKが嫌がる極端な値を制限
            rxn.lower_bound = float(np.clip(rxn.lower_bound, -1000.0, 1000.0))
            rxn.upper_bound = float(np.clip(rxn.upper_bound, -1000.0, 1000.0))


    def set_uptake_constraints(self, species_name: str, metabolite_concentrations: Dict[str, float], dynamic_kla: float = 50.0):
        model = self.models[species_name]
        max_uptake = self.max_uptake_rate

        # No LP is solved between the reset, basal-supply and medium passes.
        # Keep their original assignment order (including last-alias-wins),
        # but synchronize only the final lower bound to COBRA/optlang. Do not
        # cache model.exchanges: annotations, compartments and reactions may
        # be edited between calls, changing COBRA's boundary classification.
        pending_lower_bounds = {}

        def queue_lower_bound(reaction, lower):
            # Validate at the original assignment site. In particular, the
            # six-decimal rounding can make a negative upper bound invalid;
            # a failing medium assignment must leave its preceding value.
            reaction._check_bounds(lower, reaction.upper_bound)
            pending_lower_bounds[reaction] = lower

        def commit_lower_bounds():
            for reaction, lower in pending_lower_bounds.items():
                if reaction.lower_bound != lower:
                    # Public setters retain solver synchronization and COBRA
                    # context-manager restoration; internal fields are not
                    # modified directly.
                    reaction.lower_bound = lower

        try:
            # 1. 培地にない交換反応の取り込みを禁止する初期値を設定
            for rxn in model.exchanges:
                if rxn.id in {'EX_rubber_bulk_e', 'R_EX_rubber_bulk_e'}:
                    if rxn.bounds != (0.0, 0.0):
                        rxn.bounds = (0.0, 0.0)
                    continue
                if 'pha_c' in rxn.id or 'phb_c' in rxn.id:
                    continue
                queue_lower_bound(rxn, self._safe_lb(0.0, rxn.upper_bound))

            # 2. H2O, O2, H+ は生存に必須のため常に供給可能とする
            for h_id in ['h2o_e', 'h_e', 'o2_e']:
                if h_id in self.exchange_reactions[species_name]:
                    rxn_id = self.exchange_reactions[species_name][h_id]
                    target_rxn = model.reactions.get_by_id(rxn_id)
                    bound = -1000.0 if h_id != 'h_e' else -0.1
                    queue_lower_bound(target_rxn, self._safe_lb(bound, target_rxn.upper_bound))
        except Exception:
            # Preserve completed earlier assignments if a malformed basal
            # mapping or bound raises, as in the original sequential path.
            commit_lower_bounds()
            raise

        # 3. 培地に存在する代謝物の取り込み制約を設定
        total_biomass = None
        for met_id, concentration in metabolite_concentrations.items():
            if met_id in ['h2o_e', 'h_e']: continue

            # Bulk natural rubber is an insoluble phase rather than a freely
            # diffusible extracellular metabolite.  It is consumed only by
            # degrade_rubber(), which creates soluble C30/ODTD pools.
            if met_id in {'rubber_e', 'rubber_bulk_e', 'M_rubber_bulk_e'}:
                continue

            if met_id in self.exchange_reactions[species_name]:
                rxn_id = self.exchange_reactions[species_name][met_id]
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                    orig_lb, _ = self.original_bounds[species_name].get(rxn_id, (-1000.0, 1000.0))
                    Km = 0.01 if met_id in ['glc__D_e', 'o2_e', 'pi_e', 'nh4_e'] else 0.1
                    v_max_kinetics = max_uptake * concentration / (Km + concentration)

                    if total_biomass is None:
                        # Biomass cannot change during this bound-only pass.
                        # Keep the original species lookup/error semantics and
                        # summation order, but do not repeat it per metabolite.
                        max(1e-6, self.state.species[species_name].biomass)
                        total_biomass = sum(max(1e-6, s.biomass) for s in self.state.species.values())
                    
                    # KLa transfer has already been integrated at the start
                    # of step().  Every dissolved solute, including oxygen,
                    # must therefore share the actual remaining inventory.
                    # Using the same total-biomass denominator ensures that
                    # separate species LPs cannot each consume the full pool.
                    max_physical_flux = concentration / (total_biomass * self.dt)

                    effective_max_physical = min(max_physical_flux, max_uptake * 10.0)
                    uptake_limit = min(v_max_kinetics, effective_max_physical)

                    combined_lb = max(-uptake_limit, orig_lb)
                    queue_lower_bound(rxn, self._safe_lb(min(0.0, combined_lb), rxn.upper_bound))
                except: continue

        commit_lower_bounds()

        # Extracellular Lcp/Rox reactions remain disabled in this LP. Their
        # induction and substrate limitation are evaluated once in the
        # extracellular kinetic layer, after intracellular flux integration.
        self._enforce_polymer_boundary_separation()
    
    def solve_fba(self, species_name: str) -> Optional[cobra.Solution]:
        model = self.models[species_name]
        if self.solver_backend == "surrogate":
            # The surrogate feature packer checks all bounds vectorially and
            # every returned flux is guarded against them. Avoid the legacy
            # Python reaction-by-reaction sanitize pass on every prediction;
            # it dominated otherwise sub-millisecond GPU inference.
            return self._solve_surrogate_with_exact_guard(species_name, model)
        # optimize()前に全反応のlb < ubを保証 (GLPKのCレベルアサーション防止)
        self._sanitize_bounds(
            model, require_strict_gap=False
        )
        if self.solver_backend == "highs":
            return self._solve_exact_cpu_lp(model)
        if (
            self.solver_backend in {"cuopt", "auto"}
            and species_name not in self._cuopt_disabled_species
        ):
            try:
                solver = self._cuopt_solvers.get(species_name)
                if solver is None:
                    solver = CuOptFbaSolver(
                        model,
                        CuOptConfig(
                            method=self.cuopt_method,
                            time_limit=self.cuopt_time_limit,
                        ),
                    )
                    self._cuopt_solvers[species_name] = solver
                self.gpu_solve_attempts += 1
                gpu_solution = solver.solve(model)
                if (
                    gpu_solution is None
                    and self.cuopt_method.strip().lower() == "barrier"
                    and solver.stats.method.strip().lower() == "barrier"
                ):
                    # GEM stoichiometric matrices are rank deficient by
                    # construction. cuDSS barrier can reject their KKT matrix
                    # before iterating, while PDLP succeeds after the adapter's
                    # dependent-row reduction. Retry on the GPU before opening
                    # the CPU circuit breaker.
                    warnings.warn(
                        f"cuOpt Barrier failed for {species_name} "
                        f"(status={solver.stats.status}); retrying GPU PDLP",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    solver = CuOptFbaSolver(
                        model,
                        CuOptConfig(
                            method="pdlp",
                            time_limit=self.cuopt_time_limit,
                            crossover=True,
                        ),
                    )
                    self._cuopt_solvers[species_name] = solver
                    self.gpu_solve_attempts += 1
                    gpu_solution = solver.solve(model)
                if gpu_solution is None:
                    transient_timeout = solver.stats.status == "timelimit"
                    if not transient_timeout:
                        self._cuopt_disabled_species.add(species_name)
                    self.cpu_fallback_solves += 1
                    warnings.warn(
                        f"cuOpt returned no validated solution for {species_name} "
                        f"(status={solver.stats.status}); using exact CPU LP "
                        + (
                            "for this step and retrying GPU next step"
                            if transient_timeout
                            else "for this species for the remainder of the run"
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    self.gpu_solve_successes += 1
                    if species_name not in self._cuopt_audited_species:
                        # A small residual proves feasibility, but not that the
                        # first-order GPU method attained the same optimum. Do
                        # one CPU audit before allowing the GPU result to drive
                        # the learning trajectory.
                        cpu_audit = self._solve_exact_cpu_lp(model)
                        self._cuopt_audited_species.add(species_name)
                        if (
                            cpu_audit is None
                            or cpu_audit.status != "optimal"
                            or not np.isfinite(cpu_audit.objective_value)
                            or not np.isclose(
                                gpu_solution.objective_value,
                                cpu_audit.objective_value,
                                rtol=1e-4,
                                atol=1e-6,
                            )
                        ):
                            self.gpu_audit_failures += 1
                            self.cpu_fallback_solves += 1
                            self._cuopt_disabled_species.add(species_name)
                            warnings.warn(
                                f"cuOpt objective audit failed for {species_name}: "
                                f"GPU={gpu_solution.objective_value:.8g}, "
                                f"CPU={getattr(cpu_audit, 'objective_value', float('nan')):.8g}; "
                                "using GLPK for this species",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            return (
                                cpu_audit
                                if cpu_audit is not None
                                and cpu_audit.status == "optimal"
                                and np.isfinite(cpu_audit.objective_value)
                                else None
                            )
                        return gpu_solution
                    else:
                        return gpu_solution
            except CuOptUnavailable:
                if self.solver_backend == "cuopt":
                    raise
                self._fallback_to_glpk("cuOpt is not installed; using GLPK")
            except Exception as exc:
                if self.solver_backend == "cuopt":
                    raise
                self._fallback_to_glpk(f"cuOpt failed ({type(exc).__name__}); using GLPK")
        elif self.solver_backend in {"cuopt", "auto"}:
            self.cpu_fallback_solves += 1
        try:
            if self.solver_backend in {"cuopt", "auto"}:
                return self._solve_exact_cpu_lp(model)
            self._sanitize_bounds(model, require_strict_gap=False)
            solution = model.optimize()
            if solution.status == 'optimal' and np.isfinite(solution.objective_value):
                return solution
            # infeasibleは正常な結果 → ソルバー再構築不要（高速化）
            # time_limitの場合のみ再構築（内部状態が破損している可能性）
            if solution.status == 'time_limit':
                self._reset_solver(model)
            return None
        except Exception:
            self._reset_solver(model)
            return None

    def _get_local_surrogate_backend(self) -> Optional[FbaSurrogateBackend]:
        if self._surrogate_load_attempted:
            return self._surrogate_backend
        self._surrogate_load_attempted = True
        if not self.surrogate_dir:
            return None
        try:
            self._surrogate_backend = FbaSurrogateBackend(
                self.models,
                self.surrogate_dir,
                device=self.surrogate_device,
                ood_threshold=self.surrogate_ood_threshold,
            )
        except (OSError, ValueError, SurrogateUnavailable) as exc:
            warnings.warn(
                f"FBA surrogate could not be loaded ({exc}); using exact CPU LP",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._surrogate_backend

    @staticmethod
    def _prediction_from_service(payload: Dict[str, Any]) -> SurrogatePrediction:
        solution_payload = payload.get("solution")
        solution = None
        if solution_payload is not None:
            solution = Solution(
                objective_value=float(solution_payload["objective_value"]),
                status=str(solution_payload["status"]),
                fluxes=pd.Series(
                    np.asarray(solution_payload["fluxes"], dtype=np.float64),
                    index=solution_payload["reaction_ids"],
                    dtype=float,
                ),
            )
        return SurrogatePrediction(
            solution=solution,
            accepted=bool(payload["accepted"]),
            reason=str(payload["reason"]),
            ood_score=float(payload["ood_score"]),
            max_bound_violation=float(payload["max_bound_violation"]),
            max_mass_balance_residual=float(payload["max_mass_balance_residual"]),
            max_relative_mass_balance_residual=float(
                payload["max_relative_mass_balance_residual"]
            ),
            inference_seconds=float(payload["inference_seconds"]),
        )

    def _surrogate_lp_features(
        self, species_name: str, model: cobra.Model
    ) -> np.ndarray:
        """Fast feature packing with a cached sparse objective vector."""

        cached = self._surrogate_objective_cache.get(species_name)
        if cached is None or cached[0] != id(model.objective):
            full = extract_lp_features(model)
            n = len(model.reactions)
            cached = (
                id(model.objective),
                full[2 * n : 3 * n].copy(),
                float(full[-1]),
            )
            self._surrogate_objective_cache[species_name] = cached
        reactions = list(model.reactions)
        lower = np.fromiter((r.lower_bound for r in reactions), dtype=np.float32)
        upper = np.fromiter((r.upper_bound for r in reactions), dtype=np.float32)
        if np.any(lower > upper):
            self._sanitize_bounds(model, require_strict_gap=False)
            lower = np.fromiter((r.lower_bound for r in reactions), dtype=np.float32)
            upper = np.fromiter((r.upper_bound for r in reactions), dtype=np.float32)
        return np.concatenate(
            (lower, upper, cached[1], np.asarray([cached[2]], dtype=np.float32))
        )

    def _set_surrogate_objective(
        self, species_name: str, model: cobra.Model, reaction_id: str
    ) -> None:
        self._set_surrogate_objective_weights(
            species_name, model, {reaction_id: 1.0}
        )

    def _set_surrogate_objective_weights(
        self,
        species_name: str,
        model: cobra.Model,
        reaction_weights: Dict[str, float],
    ) -> None:
        """Set a mass-weighted objective and keep GPU feature cache coherent."""

        objective_reactions = {
            model.reactions.get_by_id(reaction_id): float(weight)
            for reaction_id, weight in reaction_weights.items()
        }
        model.objective = objective_reactions
        model.objective_direction = "max"
        if self.solver_backend != "surrogate":
            return
        objective = np.zeros(len(model.reactions), dtype=np.float32)
        for reaction_id, weight in reaction_weights.items():
            objective[model.reactions.index(reaction_id)] = float(weight)
        self._surrogate_objective_cache[species_name] = (
            id(model.objective),
            objective,
            1.0,
        )

    def _solve_surrogate_with_exact_guard(
        self, species_name: str, model: cobra.Model
    ) -> Optional[Solution]:
        """Use GPU inference only when all scientific/runtime guards pass."""

        self.surrogate_attempts += 1
        prediction: Optional[SurrogatePrediction] = None
        try:
            if self.surrogate_service is not None:
                payload = self.surrogate_service.predict(
                    species_name, self._surrogate_lp_features(species_name, model)
                )
                prediction = self._prediction_from_service(payload)
            else:
                backend = self._get_local_surrogate_backend()
                if backend is not None:
                    prediction = backend.predict_feature_batch(
                        species_name,
                        self._surrogate_lp_features(species_name, model),
                    )[0]
        except Exception as exc:
            prediction = SurrogatePrediction(
                solution=None,
                accepted=False,
                reason=f"service_error:{type(exc).__name__}",
                ood_score=float("inf"),
                max_bound_violation=float("inf"),
                max_mass_balance_residual=float("inf"),
                max_relative_mass_balance_residual=float("inf"),
                inference_seconds=0.0,
            )

        if prediction is None:
            prediction = SurrogatePrediction(
                solution=None,
                accepted=False,
                reason="artifact_unavailable",
                ood_score=float("inf"),
                max_bound_violation=float("inf"),
                max_mass_balance_residual=float("inf"),
                max_relative_mass_balance_residual=float("inf"),
                inference_seconds=0.0,
            )
        return self._finalize_surrogate_prediction(species_name, model, prediction)

    def _finalize_surrogate_prediction(
        self,
        species_name: str,
        model: cobra.Model,
        prediction: SurrogatePrediction,
    ) -> Optional[Solution]:
        """Apply guards, audits and fallback to an already computed result."""

        self.surrogate_inference_seconds += prediction.inference_seconds
        if not prediction.accepted or prediction.solution is None:
            self.surrogate_rejections += 1
            self.surrogate_rejection_reasons[prediction.reason] = (
                self.surrogate_rejection_reasons.get(prediction.reason, 0) + 1
            )
            self.cpu_fallback_solves += 1
            return self._solve_exact_cpu_lp(model)

        should_audit = (
            self.surrogate_audit_interval > 0
            and self.surrogate_attempts % self.surrogate_audit_interval == 0
        )
        if should_audit:
            self.surrogate_audits += 1
            exact = self._solve_exact_cpu_lp(model)
            approximate = prediction.solution.objective_value
            audit_ok = (
                exact is not None
                and exact.status == "optimal"
                and np.isclose(
                    approximate,
                    exact.objective_value,
                    rtol=self.surrogate_objective_rtol,
                    atol=self.surrogate_objective_atol,
                )
            )
            if not audit_ok:
                self.surrogate_audit_failures += 1
                self.cpu_fallback_solves += 1
                self.surrogate_rejection_reasons["objective_audit"] = (
                    self.surrogate_rejection_reasons.get("objective_audit", 0) + 1
                )
                return exact

        self.surrogate_accepts += 1
        return prediction.solution

    def _solve_surrogate_species_batch(self) -> Dict[str, Optional[Solution]]:
        """Submit all three GEM requests in one cross-process RPC."""

        if self.surrogate_service is None:
            return {
                species_name: self._solve_surrogate_with_exact_guard(
                    species_name, model
                )
                for species_name, model in self.models.items()
            }
        species_names = list(self.models)
        self.surrogate_attempts += len(species_names)
        try:
            payloads = self.surrogate_service.predict_many(
                {
                    species_name: self._surrogate_lp_features(species_name, model)
                    for species_name, model in self.models.items()
                }
            )
        except Exception as exc:
            reason = f"service_error:{type(exc).__name__}"
            predictions = {
                species_name: SurrogatePrediction(
                    solution=None,
                    accepted=False,
                    reason=reason,
                    ood_score=float("inf"),
                    max_bound_violation=float("inf"),
                    max_mass_balance_residual=float("inf"),
                    max_relative_mass_balance_residual=float("inf"),
                    inference_seconds=0.0,
                )
                for species_name in species_names
            }
        else:
            predictions = {
                species_name: self._prediction_from_service(payloads[species_name])
                for species_name in species_names
            }
        return {
            species_name: self._finalize_surrogate_prediction(
                species_name, self.models[species_name], predictions[species_name]
            )
            for species_name in species_names
        }

    @staticmethod
    def _solve_exact_cpu_lp(model: cobra.Model) -> Optional[Solution]:
        """Solve the exact signed-flux LP used by the cuOpt adapter on CPU.

        This is used for GPU objective audits and circuit-breaker fallbacks.
        It deliberately avoids COBRA/GLPK's split forward/reverse variables and
        the historical fixed-bound widening, so CPU and GPU receive identical
        ``S @ v = 0`` problems.
        """

        from scipy.optimize import linprog

        matrix = create_stoichiometric_matrix(
            model, array_type="lil", dtype=np.float64
        ).tocsr()
        objective = np.asarray(
            [float(reaction.objective_coefficient) for reaction in model.reactions],
            dtype=np.float64,
        )
        direction = -1.0 if model.objective.direction == "max" else 1.0
        result = linprog(
            direction * objective,
            A_eq=matrix,
            b_eq=np.zeros(matrix.shape[0], dtype=np.float64),
            bounds=[
                (float(reaction.lower_bound), float(reaction.upper_bound))
                for reaction in model.reactions
            ],
            method="highs",
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return None
        objective_value = float(np.dot(objective, result.x))
        return Solution(
            objective_value=objective_value,
            status="optimal",
            fluxes=pd.Series(
                np.asarray(result.x, dtype=np.float64),
                index=[reaction.id for reaction in model.reactions],
                dtype=float,
            ),
        )

    def solve_community_fba(self) -> Dict[str, Optional[cobra.Solution]]:
        """Solve all species as one block-diagonal community LP.

        ``separate`` mode remains the default. In joint mode, ``glpk`` and
        ``auto`` use SciPy/HiGHS for the combined CPU LP; ``cuopt`` uses the
        single GPU LP. ``auto`` falls back to HiGHS when cuOpt is unavailable.
        """
        backend = self.solver_backend
        if backend == "auto" and self._joint_auto_fallback:
            backend = "scipy"
        elif backend == "auto":
            backend = "cuopt"
        if backend in {"glpk", "highs"}:
            backend = "scipy"
        joint_method = self.cuopt_method
        if joint_method.lower() == "concurrent":
            # cuOpt 26.2 concurrent mode is unstable for this disconnected
            # community matrix on RTX 4060 (native process abort). PDLP is the
            # safe GPU method for the joint path.
            warnings.warn(
                "joint cuOpt concurrent mode is unsupported; using PDLP",
                RuntimeWarning,
                stacklevel=2,
            )
            joint_method = "pdlp"
        try:
            if self._joint_solver is None or self._joint_solver.backend != backend:
                self._joint_solver = JointCommunityFbaSolver(
                    self.models,
                    backend=backend,
                    config=CuOptConfig(
                        method=joint_method,
                        time_limit=self.cuopt_time_limit,
                    ),
                )
            solutions = self._joint_solver.solve(self.models)
            if (
                backend == "cuopt"
                and self._joint_solver.stats.status
                not in {"optimal", "primalfeasible", "primal_feasible"}
            ):
                # Barrier+crossover is the preferred basic-solution path, but
                # cuDSS 0.8 can reject this large disconnected matrix on some
                # RTX 40xx/cuOpt combinations. Retry once with PDLP without
                # crossover; this keeps the explicit GPU request on the GPU
                # instead of silently returning an empty species solution.
                if self.cuopt_method.lower() == "barrier":
                    warnings.warn(
                        "joint cuOpt barrier failed; retrying with PDLP "
                        "without crossover",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._joint_solver = JointCommunityFbaSolver(
                        self.models,
                        backend="cuopt",
                        config=CuOptConfig(
                            method="pdlp",
                            time_limit=self.cuopt_time_limit,
                            crossover=False,
                        ),
                    )
                    solutions = self._joint_solver.solve(self.models)
                    if self._joint_solver.stats.status in {
                        "optimal",
                        "primalfeasible",
                        "primal_feasible",
                    }:
                        return solutions
                raise RuntimeError(
                    "joint cuOpt solve failed with status "
                    f"{self._joint_solver.stats.status}"
                )
            return solutions
        except (CuOptUnavailable, RuntimeError) as exc:
            if self.solver_backend != "auto":
                raise
            warnings.warn(
                "cuOpt joint FBA failed "
                f"({type(exc).__name__}); falling back to SciPy/HiGHS",
                RuntimeWarning,
                stacklevel=2,
            )
            self._joint_solver = JointCommunityFbaSolver(
                self.models,
                backend="scipy",
            )
            self._joint_auto_fallback = True
            return self._joint_solver.solve(self.models)

    def _get_or_create_cooperative_solver(self) -> CooperativeCommunityFbaSolver:
        """Construct the cooperative solver once and return it without solving."""

        if self._cooperative_solver is None:
            self._cooperative_solver = CooperativeCommunityFbaSolver(
                self.models,
                original_exchange_bounds=self.original_bounds,
                coexistence_fraction=0.99,
                objective_fraction=0.99,
                maximum_coexistence_growth=0.005,
                optimize_live_objectives=self.cooperative_optimize_live_objectives,
                parsimonious_exchange=self.cooperative_parsimony,
                highs_presolve=self.cooperative_highs_presolve,
                highs_method=self.cooperative_highs_method,
                linear_program_backend=self.cooperative_linear_program_backend,
                capture_training_snapshot=self.cooperative_capture_training_snapshot,
                surrogate_artifact=self.cooperative_surrogate_artifact,
                surrogate_device=self.cooperative_surrogate_device,
                surrogate_top_k=self.cooperative_surrogate_top_k,
                surrogate_interpolation_k=self.cooperative_surrogate_interpolation_k,
                surrogate_interpolation_power=self.cooperative_surrogate_interpolation_power,
                surrogate_exact_interval=self.cooperative_surrogate_exact_interval,
                surrogate_distance_threshold=self.cooperative_surrogate_distance_threshold,
                surrogate_require_qualified=self.cooperative_surrogate_require_qualified,
                surrogate_validation_manifest=(
                    self.cooperative_surrogate_validation_manifest
                ),
                gpu_qp_projection=self.cooperative_gpu_qp_projection,
                gpu_qp_only=self.cooperative_gpu_qp_only,
                gpu_qp_candidates=self.cooperative_gpu_qp_candidates,
                gpu_qp_rerank_pool=self.cooperative_gpu_qp_rerank_pool,
                gpu_qp_decision_strength=(
                    self.cooperative_gpu_qp_decision_strength
                ),
                gpu_qp_blend_candidates=(
                    self.cooperative_gpu_qp_blend_candidates
                ),
                gpu_qp_blend_distance_power=(
                    self.cooperative_gpu_qp_blend_distance_power
                ),
                gpu_qp_max_iterations=self.cooperative_gpu_qp_max_iterations,
                gpu_qp_normalized_tolerance=(
                    self.cooperative_gpu_qp_normalized_tolerance
                ),
                gpu_qp_match_pha=self.cooperative_gpu_qp_match_pha,
                gpu_qp_multioutput_strength=self.cooperative_gpu_qp_multioutput_strength,
                gpu_qp_independent_species=self.cooperative_gpu_qp_independent_species,
                gpu_qp_service=self.cooperative_gpu_qp_service,
            )
        return self._cooperative_solver

    def solve_cooperative_fba(
        self,
        frozen_inputs: Optional[FrozenCommunityStepInputs] = None,
    ) -> Dict[str, Optional[cobra.Solution]]:
        """Solve a shared-medium max-min LP with simultaneous cross-feeding."""

        solver = self._get_or_create_cooperative_solver()
        biomass = {
            species: max(1e-12, float(state.biomass))
            for species, state in self.state.species.items()
        }
        return solver.solve(
            self.models,
            biomass_g_l=biomass,
            medium_mmol_l=self.state.metabolites,
            dt=self.dt,
            max_crossfeed_uptake=self.max_uptake_rate,
            frozen_inputs=frozen_inputs,
        )

    def _reset_solver(self, model: cobra.Model):
        """ソルバーの内部状態を再構築する"""
        if self.solver_backend in {"cuopt", "auto"}:
            # Rebuild the persistent representation after a numerical or
            # structural failure; the next call will recreate it lazily.
            species_name = next((name for name, item in self.models.items() if item is model), None)
            if species_name is not None:
                self._cuopt_solvers.pop(species_name, None)
            if self.solver_backend == "cuopt":
                return
        self._configure_cpu_solver(model)

    @staticmethod
    def _configure_cpu_solver(model: cobra.Model):
        try:
            model.solver = 'glpk'
            model.solver.configuration.timeout = 3
            model.solver.configuration.presolve = False
            model.solver.configuration.tolerances.feasibility = 1e-6
            model.solver.configuration.tolerances.optimality = 1e-6
        except: pass

    def _fallback_to_glpk(self, reason: str):
        self.solver_backend = "glpk"
        self._cuopt_solvers.clear()
        if not self._cuopt_fallback_warned:
            warnings.warn(reason, RuntimeWarning, stacklevel=2)
            self._cuopt_fallback_warned = True

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
    
    def _update_environment(
        self,
        species_solutions: Dict[str, Dict[str, float]],
        integration_biomass: Optional[Dict[str, float]] = None,
    ):
        total_delta_metabolites = {}
        net_h_flux_mmol = 0.0
        total_ye_uptake = 0.0  # 酵母エキスの総消費量を追跡
        
        for s in self.state.species.values():
            s.metabolite_uptake.clear()
            s.metabolite_secretion.clear()

        for species_name, fluxes in species_solutions.items():
            biomass = (
                float(integration_biomass[species_name])
                if integration_biomass is not None
                else self.state.species[species_name].biomass
            )
            for met_id, rxn_id in self.exchange_reactions[species_name].items():
                actual_rid = rxn_id
                if actual_rid not in fluxes and met_id == 'rubber_e':
                    for fallback in ['EX_rubber_bulk_e', 'EX_rubber_e', 'R_EX_rubber_e', 'rubber_high_e']:
                        if fallback in fluxes: actual_rid = fallback; break
                            
                flux = fluxes.get(actual_rid, 0.0)
                delta = flux * biomass * self.dt

                # 酵母エキス成分の消費を追跡
                if flux < -1e-9 and met_id in self.YE_COMPONENTS:
                    total_ye_uptake += abs(delta)

                if met_id == 'h_e': net_h_flux_mmol += delta
                # pha_c/phv_c are intracellular storage-accounting sinks, not
                # extracellular secretions.  Adding them to the broth created
                # a fictitious environmental PHA pool in the former model.
                if met_id not in {'h_e', 'pha_c', 'phv_c'}:
                    total_delta_metabolites[met_id] = total_delta_metabolites.get(met_id, 0.0) + delta
                
                if flux < -1e-9: self.state.species[species_name].metabolite_uptake[met_id] = -flux
                elif flux > 1e-9: self.state.species[species_name].metabolite_secretion[met_id] = flux
            
            storage_state = self.state.species[species_name]
            # Backward compatibility for checkpoints written before PHB and
            # 3HV were tracked separately.
            if (
                storage_state.pha_accumulated > 0.0
                and storage_state.phb_accumulated == 0.0
                and storage_state.phv_accumulated == 0.0
            ):
                storage_state.phb_accumulated = storage_state.pha_accumulated

            storage_fluxes = {}
            for pool_id, field_name, repeat_mass in (
                ('pha_c', 'phb_accumulated', self.PHB_REPEAT_G_PER_MMOL),
                ('phv_c', 'phv_accumulated', self.PHV_REPEAT_G_PER_MMOL),
            ):
                reaction_id = self.exchange_reactions[species_name].get(pool_id)
                flux = fluxes.get(reaction_id, 0.0) if reaction_id else 0.0
                if flux > 1e-9:
                    storage_fluxes[field_name] = (
                        flux * biomass * self.dt,
                        repeat_mass,
                    )

            if storage_fluxes:
                current_mass = (
                    storage_state.phb_accumulated * self.PHB_REPEAT_G_PER_MMOL
                    + storage_state.phv_accumulated * self.PHV_REPEAT_G_PER_MMOL
                )
                fraction = self.max_pha_fraction_g_gdcw
                max_mass = (
                    float('inf') if fraction == 0.0
                    else biomass * fraction / (1.0 - fraction)
                )
                if fraction == 0.0:
                    max_mass = 0.0
                proposed_mass = sum(
                    amount * repeat_mass
                    for amount, repeat_mass in storage_fluxes.values()
                )
                scale = min(
                    1.0,
                    max(0.0, max_mass - current_mass) / proposed_mass,
                ) if proposed_mass > 0.0 else 0.0
                for field_name, (amount, _) in storage_fluxes.items():
                    setattr(
                        storage_state,
                        field_name,
                        getattr(storage_state, field_name) + amount * scale,
                    )
            storage_state.pha_accumulated = (
                storage_state.phb_accumulated + storage_state.phv_accumulated
            )
        
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

    def _apply_ph_stat(self) -> None:
        """Apply the fermenter's independent acid/base pH controller."""

        if self.ph_control_target is None:
            return
        ratio = 10.0 ** (self.ph_control_target - self.pKa)
        target_base = self.buffer_total * ratio / (1.0 + ratio)
        delta = float(target_base - self.buffer_base)
        if delta >= 0.0:
            self.cumulative_base_added_mmol_l += delta
        else:
            self.cumulative_acid_added_mmol_l += -delta
        self.buffer_base = target_base
        self.buffer_acid = self.buffer_total - target_base
        self.state.metabolites["h_e"] = 10.0 ** (3.0 - self.ph_control_target)

    def degrade_rubber(self, degradation_rates: Dict[str, float]):
        """Apply mass-conserving extracellular Lcp/Rox cleavage kinetics.

        Bulk rubber is tracked as g L^-1 and converted to C5-equivalent mmol
        L^-1.  The soluble C30 and ODTD pools are mmol L^-1.  Oxygen is
        consumed with the balanced stoichiometries encoded in the curated
        SBML annotation reactions.  No bulk-rubber flux is also admitted to
        the intracellular FBA problem.

        ``degradation_rates`` may override the documented defaults with the
        explicit keys ``lcp_c5``, ``roxb_c5``, ``roxa_direct_c5`` and
        ``roxa_oligo_c30``.  Legacy ``OR16`` and ``NS21`` keys remain
        supported: OR16 overrides Lcp; NS21 is split equally between RoxB and
        direct RoxA C5 capacity.
        """

        supplied = degradation_rates or {}
        rates = dict(self.DEFAULT_POLYMER_RATES)
        if "OR16" in supplied:
            rates["lcp_c5"] = max(0.0, float(supplied["OR16"]))
        if "NS21" in supplied:
            ns21_total = max(0.0, float(supplied["NS21"]))
            rates["roxb_c5"] = 0.5 * ns21_total
            rates["roxa_direct_c5"] = 0.5 * ns21_total
            rates["roxa_oligo_c30"] = ns21_total / 6.0
        for key in rates:
            if key in supplied:
                rates[key] = max(0.0, float(supplied[key]))

        rubber_g_before = max(0.0, float(self.state.rubber_concentration))
        c30_before = max(0.0, float(self.state.metabolites.get("C30_oligo_e", 0.0)))
        odtd_before = max(0.0, float(self.state.metabolites.get("odtd_e", 0.0)))
        oxygen_before = max(0.0, float(self.state.metabolites.get("o2_e", 0.0)))
        # Extracellular oxygenases and respiratory metabolism operate
        # simultaneously.  Reserve only a configurable share of the current
        # dissolved pool for polymer cleavage so the operator split cannot
        # let the first process consume 100% of DO.  The fraction must be
        # calibrated from OUR/DO measurements.
        polymer_oxygen_budget = oxygen_before * self.polymer_oxygen_fraction
        carbon_before = (
            rubber_g_before * 1000.0 / self.RUBBER_C5_MOLAR_MASS_G_PER_MOL
            + 6.0 * c30_before
            + 3.0 * odtd_before
        )

        biomass_or16 = sum(
            state.biomass for name, state in self.state.species.items() if "OR16" in name
        )
        biomass_ns21 = sum(
            state.biomass for name, state in self.state.species.items() if "NS21" in name
        )
        rubber_saturation = rubber_g_before / (1.0 + rubber_g_before)

        # Carbon catabolite repression is currently evidenced/parameterized
        # only for the OR16 Lcp branch; applying it to RoxA/RoxB would add an
        # unsupported regulatory assumption.
        glucose = max(0.0, float(self.state.metabolites.get("glc__D_e", 0.0)))
        lcp_repression = 0.5 / (0.5 + glucose)

        requested = {
            "lcp_c5": rates["lcp_c5"] * biomass_or16 * rubber_saturation
            * lcp_repression * self.dt,
            "roxb_c5": rates["roxb_c5"] * biomass_ns21 * rubber_saturation * self.dt,
            "roxa_direct_c5": rates["roxa_direct_c5"] * biomass_ns21
            * rubber_saturation * self.dt,
        }
        rubber_c5_available = (
            rubber_g_before * 1000.0 / self.RUBBER_C5_MOLAR_MASS_G_PER_MOL
        )
        requested_c5 = sum(requested.values())
        requested_oxygen = (
            (requested["lcp_c5"] + requested["roxb_c5"]) / 6.0
            + requested["roxa_direct_c5"] / 3.0
        )
        bulk_scale = min(
            1.0,
            rubber_c5_available / requested_c5 if requested_c5 > 0.0 else 1.0,
            polymer_oxygen_budget / requested_oxygen
            if requested_oxygen > 0.0 else 1.0,
        )
        actual = {key: value * bulk_scale for key, value in requested.items()}

        c5_consumed = sum(actual.values())
        oxygen_bulk = (
            (actual["lcp_c5"] + actual["roxb_c5"]) / 6.0
            + actual["roxa_direct_c5"] / 3.0
        )
        c30 = c30_before + (actual["lcp_c5"] + actual["roxb_c5"]) / 6.0
        odtd = odtd_before + actual["roxa_direct_c5"] / 3.0
        oxygen = max(0.0, oxygen_before - oxygen_bulk)
        rubber_g = max(
            0.0,
            rubber_g_before
            - c5_consumed * self.RUBBER_C5_MOLAR_MASS_G_PER_MOL / 1000.0,
        )

        # RoxA may process newly exposed/RoxB-generated oligomer ends, but is
        # not made absolutely dependent on RoxB because the direct bulk route
        # above remains active.
        roxa_oligo_requested = rates["roxa_oligo_c30"] * biomass_ns21 * self.dt
        polymer_oxygen_remaining = max(0.0, polymer_oxygen_budget - oxygen_bulk)
        roxa_oligo = min(roxa_oligo_requested, c30, polymer_oxygen_remaining)
        c30 -= roxa_oligo
        odtd += 2.0 * roxa_oligo
        oxygen -= roxa_oligo

        self.state.rubber_concentration = rubber_g
        self.state.metabolites["C30_oligo_e"] = c30
        self.state.metabolites["odtd_e"] = odtd
        self.state.metabolites["o2_e"] = oxygen

        carbon_after = (
            rubber_g * 1000.0 / self.RUBBER_C5_MOLAR_MASS_G_PER_MOL
            + 6.0 * c30
            + 3.0 * odtd
        )
        self.last_polymer_fluxes = {
            "lcp_c5_mmol_l_step": actual["lcp_c5"],
            "roxb_c5_mmol_l_step": actual["roxb_c5"],
            "roxa_direct_c5_mmol_l_step": actual["roxa_direct_c5"],
            "roxa_oligo_c30_mmol_l_step": roxa_oligo,
            "oxygen_mmol_l_step": oxygen_before - oxygen,
            "rubber_degraded_g_l_step": rubber_g_before - rubber_g,
            "carbon_c5_equivalent_error_mmol_l": carbon_after - carbon_before,
            "bulk_substrate_scale": bulk_scale,
            "polymer_oxygen_fraction": self.polymer_oxygen_fraction,
        }

            
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
                total_repeat_units = (
                    species_state.phb_accumulated + species_state.phv_accumulated
                )
                row = {
                    'time': self.state.time, 'species': species_name, 'growth_rate': species_state.growth_rate,
                    'biomass': species_state.biomass, 'pha_accumulated': species_state.pha_accumulated,
                    'phb_accumulated': species_state.phb_accumulated,
                    'phv_accumulated': species_state.phv_accumulated,
                    'phv_mol_fraction': (
                        species_state.phv_accumulated / total_repeat_units
                        if total_repeat_units > 0.0 else 0.0
                    ),
                    'rubber_concentration': self.state.rubber_concentration, 'cumulative_co2': self.cumulative_co2_emission,
                    'conc_glc__D_e': self.state.metabolites.get('glc__D_e', 0.0), 'conc_nh4_e': self.state.metabolites.get('nh4_e', 0.0),
                    'conc_o2_e': self.state.metabolites.get('o2_e', 0.0), 'conc_pi_e': self.state.metabolites.get('pi_e', 0.0),
                    'conc_arg__L_e': self.state.metabolites.get('arg__L_e', 0.0), 'conc_trp__L_e': self.state.metabolites.get('trp__L_e', 0.0),
                    'conc_leu__L_e': self.state.metabolites.get('leu__L_e', 0.0), 'conc_rubber_fragment_e': self.state.metabolites.get('rubber_fragment_e', 0.0),
                    'conc_odtd_e': self.state.metabolites.get('odtd_e', 0.0), 'conc_h2o_e': self.state.metabolites.get('h2o_e', 0.0),
                    'conc_h_e': self.state.metabolites.get('h_e', 0.0001), 'flux_EX_co2_e': species_state.metabolite_secretion.get('co2_e', 0.0),
                    'flux_EX_pha_c': species_state.metabolite_secretion.get('pha_c', 0.0),
                    'flux_EX_phv_c': species_state.metabolite_secretion.get('phv_c', 0.0),
                    'flux_EX_phb_c': species_state.metabolite_secretion.get('phb_c', 0.0),
                    'flux_EX_rubber_fragment_e': species_state.metabolite_secretion.get('rubber_fragment_e', 0.0), 'flux_EX_odtd_e': species_state.metabolite_secretion.get('odtd_e', 0.0)
                }
                writer.writerow(row)

    def step(self, rubber_degradation_rates: Dict[str, float], nutrient_supplementation: Dict[str, float], dynamic_kla: float = 50.0) -> ConsortiumState:
        step_started = time.perf_counter()
        o2_sat = 0.25
        current_o2 = self.state.metabolites.get('o2_e', o2_sat)
        self.state.metabolites['o2_e'] = o2_sat - (o2_sat - current_o2) * np.exp(-dynamic_kla * self.dt)
        supplementation_map = {
            'sn_or16': 'mlttr_e',
            'sn_ns21': 'ptrc_e',
            'sn_lp': 'mnl_e',
            'helper_lactate': 'lac__L_e',
        }
        for nut_name, met_id in supplementation_map.items():
            supplement = np.clip(nutrient_supplementation.get(nut_name, 0.0), 0.0, 10.0)
            self.state.metabolites[met_id] = self.state.metabolites.get(met_id, 0.0) + supplement

        # One physical common-feed bottle replaces the uncalibrated yeast
        # extract proxy for low-cost coexistence. The control is a dimensionless
        # multiplier of the defined mmol/L/h formulation.
        defined_scale = float(
            np.clip(nutrient_supplementation.get("coexistence_feed_rate", 0.0), 0.0, 10.0)
        )
        defined_mw = {
            "glc__D_e": 180.16,
            "arg__L_e": 174.20,
            "trp__L_e": 204.23,
            "leu__L_e": 131.17,
            "glu__L_e": 147.13,
            "ile__L_e": 131.17,
            "pro__L_e": 115.13,
            "phe__L_e": 165.19,
            "gln__L_e": 146.14,
            "pydam_e": 168.20,
        }
        self.last_defined_feed_g_l = 0.0
        for met_id, rate_mmol_l_h in COEXISTENCE_DEFINED_FEED_MMOL_L_H.items():
            addition = defined_scale * rate_mmol_l_h * self.dt
            self.state.metabolites[met_id] = self.state.metabolites.get(met_id, 0.0) + addition
            self.last_defined_feed_g_l += addition * defined_mw[met_id] / 1000.0
        self.cumulative_defined_feed_g_l += self.last_defined_feed_g_l
        
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

        # Operator split for an insoluble substrate: extracellular Lcp/Rox
        # first use the transferred oxygen and release soluble products; the
        # intracellular GEMs then consume those products and the remaining DO
        # in the same time interval.  Applying cleavage after FBA starved the
        # oxygenases and introduced an artificial one-step product delay.
        self.degrade_rubber(rubber_degradation_rates)

        total_co2_flux = 0.0
        species_solutions = {}
        integration_biomass = {
            species: float(state.biomass)
            for species, state in self.state.species.items()
        }

        # Update all live bounds/objectives first. In joint mode this creates
        # one consistent snapshot for the combined community LP.
        frozen_inputs = None
        if self._frozen_community_input_builder is not None:
            frozen_inputs = self._frozen_community_input_builder.build_step(
                self.state.metabolites,
                integration_biomass,
                dt=self.dt,
                max_uptake_rate=self.max_uptake_rate,
                phv_requires_rubber_intermediate=(
                    self.phv_requires_rubber_intermediate
                ),
                phb_repeat_g_per_mmol=self.PHB_REPEAT_G_PER_MMOL,
                phv_repeat_g_per_mmol=self.PHV_REPEAT_G_PER_MMOL,
            )
        else:
            for species_name in self.models.keys():
                self.set_uptake_constraints(species_name, self.state.metabolites, dynamic_kla)
                model = self.models[species_name]
                pha_rxn_id = self.exchange_reactions[species_name].get('pha_c')
                phv_rxn_id = self.exchange_reactions[species_name].get('phv_c')
                if 'NS21' in species_name:
                    env_nh4 = self.state.metabolites.get('nh4_e', 0.0)
                    rubber_intermediate_available = any(
                        self.state.metabolites.get(metabolite_id, 0.0) > 1e-12
                        for metabolite_id in ('C30_oligo_e', 'odtd_e')
                    )
                    phv_enabled = bool(phv_rxn_id) and (
                        not self.phv_requires_rubber_intermediate
                        or rubber_intermediate_available
                    )

                    if env_nh4 < 0.1 and pha_rxn_id:
                        # PHA accumulation is enabled only in the explicitly
                        # modelled nitrogen-limited phase.  Leaving this sink open
                        # during biomass maximisation lets alternate LP optima
                        # report spurious PHA under nitrogen-replete conditions.
                        pha_rxn = model.reactions.get_by_id(pha_rxn_id)
                        _, original_upper = self.original_bounds[species_name].get(
                            pha_rxn_id, (0.0, 1000.0)
                        )
                        pha_rxn.upper_bound = max(0.0, float(original_upper))
                        storage_objective = {
                            pha_rxn_id: self.PHB_REPEAT_G_PER_MMOL,
                        }
                        if phv_enabled:
                            phv_rxn = model.reactions.get_by_id(phv_rxn_id)
                            _, original_phv_upper = self.original_bounds[species_name].get(
                                phv_rxn_id, (0.0, 1000.0)
                            )
                            phv_rxn.upper_bound = max(0.0, float(original_phv_upper))
                            storage_objective[phv_rxn_id] = self.PHV_REPEAT_G_PER_MMOL
                        elif phv_rxn_id:
                            model.reactions.get_by_id(phv_rxn_id).upper_bound = 0.0
                        self._set_surrogate_objective_weights(
                            species_name, model, storage_objective
                        )
                    else:
                        if pha_rxn_id:
                            model.reactions.get_by_id(pha_rxn_id).upper_bound = 0.0
                        if phv_rxn_id:
                            model.reactions.get_by_id(phv_rxn_id).upper_bound = 0.0
                        if 'R_Growth' in model.reactions:
                            self._set_surrogate_objective(species_name, model, 'R_Growth')
                        elif 'Growth' in model.reactions:
                            self._set_surrogate_objective(species_name, model, 'Growth')
                elif 'OR16' in species_name:
                    if 'R_Growth' in model.reactions:
                        self._set_surrogate_objective(species_name, model, 'R_Growth')
                    elif 'Growth' in model.reactions:
                        self._set_surrogate_objective(species_name, model, 'Growth')

        solve_started = time.perf_counter()
        pre_solve_seconds = solve_started - step_started
        if self.solver_backend == "surrogate" and self.surrogate_service is not None:
            community_solutions = self._solve_surrogate_species_batch()
        elif self.fba_mode == "joint":
            community_solutions = self.solve_community_fba()
        elif self.fba_mode == "cooperative":
            community_solutions = self.solve_cooperative_fba(
                frozen_inputs=frozen_inputs
            )
        else:
            community_solutions = {
                species_name: self.solve_fba(species_name)
                for species_name in self.models.keys()
            }
        self.last_fba_solutions = dict(community_solutions)
        solve_finished = time.perf_counter()
        self.solve_attempts += len(self.models)
        self.solve_successes += sum(
            solution is not None for solution in community_solutions.values()
        )

        for species_name in self.models.keys():
            solution = community_solutions.get(species_name)
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

        # Fluxes were solved at the beginning-of-step biomass. Integrating
        # them with the post-growth biomass creates or destroys shared
        # metabolites, especially for simultaneous cross-feeding.
        self._update_environment(species_solutions, integration_biomass)
        self._apply_ph_stat()
        self.cumulative_co2_emission += total_co2_flux * self.dt
        self._log_telemetry()
        self.state.time += self.dt
        self.current_step += 1
        step_finished = time.perf_counter()
        self.last_step_timing = {
            "pre_solve_seconds": pre_solve_seconds,
            "solve_seconds": solve_finished - solve_started,
            "post_solve_seconds": step_finished - solve_finished,
            "total_seconds": step_finished - step_started,
        }
        for key, value in self.last_step_timing.items():
            self.step_timing_totals[key] += value
        return self.state

    def get_solver_diagnostics(self) -> Dict[str, Any]:
        """Return serializable timing, status, and validity diagnostics."""

        completed_steps = max(self.current_step, 1)
        diagnostics: Dict[str, Any] = {
            "steps": self.current_step,
            "effective_backend": (
                (
                    "surrogate_with_highs_fallback"
                    if self.cpu_fallback_solves
                    else "surrogate"
                )
                if self.solver_backend == "surrogate"
                else
                "glpk_after_cuopt_probe"
                if self.solver_backend in {"cuopt", "auto"}
                and len(self._cuopt_disabled_species) == len(self.models)
                else (
                    "cuopt_with_glpk_fallback"
                    if self.cpu_fallback_solves
                    else self.solver_backend
                )
            ),
            "solve_attempts": self.solve_attempts,
            "solve_successes": self.solve_successes,
            "solve_success_rate": (
                self.solve_successes / self.solve_attempts
                if self.solve_attempts
                else 0.0
            ),
            "gpu_solve_attempts": self.gpu_solve_attempts,
            "gpu_solve_successes": self.gpu_solve_successes,
            "gpu_validated_success_rate": (
                self.gpu_solve_successes / self.gpu_solve_attempts
                if self.gpu_solve_attempts
                else 0.0
            ),
            "gpu_audit_failures": self.gpu_audit_failures,
            "cpu_fallback_solves": self.cpu_fallback_solves,
            "ph_control": {
                "target": self.ph_control_target,
                "base_added_mmol_l": self.cumulative_base_added_mmol_l,
                "acid_added_mmol_l": self.cumulative_acid_added_mmol_l,
            },
            "defined_feed": {
                "cumulative_g_l": self.cumulative_defined_feed_g_l,
                "last_step_g_l": self.last_defined_feed_g_l,
            },
            "cuopt_disabled_species": sorted(self._cuopt_disabled_species),
            "mean_step_timing": {
                key: value / completed_steps
                for key, value in self.step_timing_totals.items()
            },
        }
        if self.solver_backend == "surrogate":
            diagnostics["surrogate"] = {
                "attempts": self.surrogate_attempts,
                "accepted": self.surrogate_accepts,
                "acceptance_rate": (
                    self.surrogate_accepts / self.surrogate_attempts
                    if self.surrogate_attempts
                    else 0.0
                ),
                "rejections": self.surrogate_rejections,
                "rejection_reasons": dict(self.surrogate_rejection_reasons),
                "audits": self.surrogate_audits,
                "audit_failures": self.surrogate_audit_failures,
                "inference_seconds": self.surrogate_inference_seconds,
            }
            if self._surrogate_backend is not None:
                diagnostics["surrogate"]["load_errors"] = dict(
                    self._surrogate_backend.load_errors
                )
        if self._cuopt_solvers:
            diagnostics["cuopt_species"] = {
                name: dict(vars(solver.stats))
                for name, solver in self._cuopt_solvers.items()
            }
        if self._joint_solver is not None:
            diagnostics["joint"] = dict(vars(self._joint_solver.stats))
        if self._cooperative_solver is not None:
            cooperative = dict(vars(self._cooperative_solver.stats))
            cooperative.update(
                {
                    "surrogate_attempts": self._cooperative_solver.surrogate_attempts,
                    "surrogate_accepts": self._cooperative_solver.surrogate_accepts,
                    "surrogate_acceptance_rate": (
                        self._cooperative_solver.surrogate_accepts
                        / self._cooperative_solver.surrogate_attempts
                        if self._cooperative_solver.surrogate_attempts
                        else 0.0
                    ),
                    "surrogate_rejections": self._cooperative_solver.surrogate_rejections,
                    "surrogate_rejection_reasons": dict(
                        self._cooperative_solver.surrogate_rejection_reasons
                    ),
                    "exact_bypasses": self._cooperative_solver.surrogate_exact_bypasses,
                    "cpu_cooperative_solve_calls": self._cooperative_solver.cpu_cooperative_solve_calls,
                    "cpu_lp_stage_calls": self._cooperative_solver.cpu_lp_stage_calls,
                    "gpu_lp_stage_calls": self._cooperative_solver.gpu_lp_stage_calls,
                    "surrogate_disabled_reason": (
                        self._cooperative_solver.surrogate_disabled_reason
                    ),
                    "gpu_qp_projection": (
                        self._cooperative_solver.gpu_qp_projection
                    ),
                    "gpu_qp_only": self._cooperative_solver.gpu_qp_only,
                    "gpu_qp_attempts": self._cooperative_solver.gpu_qp_attempts,
                    "gpu_qp_accepts": self._cooperative_solver.gpu_qp_accepts,
                    "gpu_qp_failures": self._cooperative_solver.gpu_qp_failures,
                    "gpu_qp_target_attempts": self._cooperative_solver.gpu_qp_target_attempts,
                    "gpu_qp_target_successes": self._cooperative_solver.gpu_qp_target_successes,
                    "gpu_multioutput_attempts": self._cooperative_solver.gpu_multioutput_attempts,
                    "gpu_multioutput_improvements": self._cooperative_solver.gpu_multioutput_improvements,
                    "gpu_qp_total_iterations": (
                        self._cooperative_solver.gpu_qp_iterations
                    ),
                    "gpu_qp_seconds": self._cooperative_solver.gpu_qp_seconds,
                    "gpu_qp_last": dict(
                        self._cooperative_solver.gpu_qp_last_diagnostics
                    ),
                    "gpu_qp_failure_diagnostics": list(
                        self._cooperative_solver.gpu_qp_failure_diagnostics
                    ),
                }
            )
            diagnostics["cooperative"] = cooperative
        return diagnostics
    
    def get_state_vector(self) -> np.ndarray:
        state_vec = []
        for name in sorted(self.models.keys()): state_vec.append(np.clip(self.state.species[name].biomass, 0.0, self.carrying_capacity))
        for name in sorted(self.models.keys()): state_vec.append(np.clip(self.state.species[name].growth_rate, -0.5, 2.0))
        for met in ['glc__D_e', 'nh4_e', 'rubber_fragment_e', 'odtd_e', 'arg__L_e', 'trp__L_e', 'leu__L_e', 'h_e']:
            state_vec.append(np.clip(self.state.metabolites.get(met, 0.0), 0.0, 1000.0))
        for name in sorted(self.models.keys()): state_vec.append(np.clip(self.state.species[name].pha_accumulated, 0.0, 1000.0))
        state_vec.append(np.clip(self.state.rubber_concentration, 0.0, 100000.0))
        return np.nan_to_num(np.array(state_vec, dtype=np.float32), nan=0.0)
