import gymnasium as gym
from gymnasium import spaces
import numpy as np
import hashlib
import json
from typing import Dict, Tuple, Optional, List
import cobra
from .dfba_simulator import dFBASimulator, ConsortiumState
from .utils import COEXISTENCE_DEFINED_FEED_MMOL_L_H

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
        max_common_feed_early: float = 2.0,
        max_common_feed_late: float = 1.0,
        max_specific_feed_per_step: float = 0.1,
        common_feed_action_budget: float = 0.25,
        excess_common_feed_penalty: float = 2.0,
        observation_schema: str = "legacy_v1",
        control_schema: str | None = None,
        max_specific_feed_rate_mmol_l_h: float = 0.1,
    ):
        super().__init__()
        if not np.isfinite(max_time) or max_time <= 0:
            raise ValueError("max_time must be positive and finite")
        audited = getattr(simulator, "NUMERICS_VERSION", None) in {
            "audited_cultivation_v2", "physiology_cultivation_v1"}
        self.control_schema = control_schema or ("audited_rates_v2" if audited else "legacy_v1")
        if self.control_schema not in ("legacy_v1", "audited_rates_v2"):
            raise ValueError("unknown control schema")
        if self.control_schema == "audited_rates_v2" and not audited:
            raise ValueError("audited_rates_v2 requires the audited cultivation simulator")
        if audited and self.control_schema != "audited_rates_v2":
            raise ValueError("audited cultivation requires a new audited_rates_v2 policy")
        if not np.isfinite(max_specific_feed_rate_mmol_l_h) or max_specific_feed_rate_mmol_l_h < 0:
            raise ValueError("specific feed rate must be finite and nonnegative")
        self.max_specific_feed_rate_mmol_l_h = float(max_specific_feed_rate_mmol_l_h)
        if observation_schema not in ("legacy_v1", "metabolic_v2"):
            raise ValueError("observation_schema must be 'legacy_v1' or 'metabolic_v2'")
        self.observation_schema = observation_schema
        self.simulator = simulator
        self.max_time = max_time
        self.dt = simulator.dt
        self.max_common_feed_early = max(0.0, float(max_common_feed_early))
        self.max_common_feed_late = max(0.0, float(max_common_feed_late))
        self.max_specific_feed_per_step = max(
            0.0, float(max_specific_feed_per_step)
        )
        self.common_feed_action_budget = float(
            np.clip(common_feed_action_budget, 0.0, 1.0)
        )
        self.excess_common_feed_penalty = max(
            0.0, float(excess_common_feed_penalty)
        )
        
        # 初期状態の保存
        self.initial_biomass = {name: s.biomass for name, s in simulator.state.species.items()}
        self.initial_metabolites = simulator.state.metabolites.copy()
        self.initial_rubber = simulator.state.rubber_concentration
        
        self.last_total_pha = 0.0
        self.last_bs = 0.0
        self.current_step = 0
        
        self._setup_spaces()
        self.control_schema_metadata = dict(
            schema=self.control_schema,
            numerics_version=getattr(simulator, "NUMERICS_VERSION", "legacy_dfba"),
            individual_feed_unit=("mmol/L/h" if audited else "mmol/L/controller_step"),
            individual_feed_max=(self.max_specific_feed_rate_mmol_l_h if audited else self.max_specific_feed_per_step),
            common_feed_unit="multiplier of COEXISTENCE_DEFINED_FEED_MMOL_L_H",
            common_feed_mmol_l_h=dict(COEXISTENCE_DEFINED_FEED_MMOL_L_H),
            common_feed_max_early=self.max_common_feed_early,
            common_feed_max_late=self.max_common_feed_late,
            common_feed_soft_budget=self.common_feed_action_budget,
            kla_unit="1/h", kla_max=200.,
            pha_reward_unit="g/L" if audited else "mmol_repeat/L",
            running_penalty_unit="per_hour" if audited else "per_controller_step",
            feed_cost_note=("Uncalibrated molecular-mass proxy; stock salts, counterions, "
                            "aeration, titrant and monetary costs are not included"),
            requires_new_policy_and_normalizer=audited,
        )
        self.rubber_degradation_rates = {}
        self._initial_gem_identity = self._capture_initial_gem_identity()

    def _capture_initial_gem_identity(self) -> dict:
        """Freeze GEM inputs before live FBA changes bounds and objectives."""
        identities = {}
        for name, model in self.simulator.models.items():
            if not hasattr(model, 'reactions'):
                identities[name] = {'kind': 'non_GEM_test_double'}
                continue
            saved_bounds = getattr(self.simulator, '_initial_bounds', {}).get(name, {})
            original_bounds = getattr(self.simulator, 'original_bounds', {}).get(name, {})
            def number(value):
                value = float(value)
                return value if np.isfinite(value) else str(value)
            payload = dict(
                model_id=model.id,
                metabolites=[(m.id, m.compartment, m.formula, m.charge) for m in model.metabolites],
                reactions=[dict(id=r.id, bounds=[number(v) for v in saved_bounds.get(r.id, r.bounds)],
                                stoichiometry=sorted((m.id, float(c)) for m, c in r.metabolites.items()),
                                gpr=r.gene_reaction_rule, objective_coefficient=float(r.objective_coefficient))
                           for r in model.reactions],
                original_exchange_bounds={rid: [number(v) for v in bounds]
                                          for rid, bounds in original_bounds.items()},
                exchange_mapping=getattr(self.simulator, 'exchange_reactions', {}).get(name, {}),
                objective_direction=model.objective.direction,
            )
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
            identities[name] = dict(model_id=model.id, initial_gem_sha256=digest,
                                    reaction_count=len(model.reactions), metabolite_count=len(model.metabolites))
        return identities

    def _episode_initial_conditions(self) -> dict:
        exact_reset = callable(getattr(self.simulator, 'reset', None))
        initial = getattr(self.simulator, '_initial_state', None) if exact_reset else None
        medium = {k: float(v) for k, v in (initial.metabolites if initial is not None else self.initial_metabolites).items()}
        if not exact_reset:
            medium.update(arg__L_e=1., trp__L_e=1., leu__L_e=1., glc__D_e=.5,
                          C30_oligo_e=0., biosurfactant_e=0.)
            target = getattr(self.simulator, 'ph_control_target', None)
            medium['h_e'] = 10. ** (3. - (7. if target is None else target))
        return dict(
            biomass_g_l=({k: float(s.biomass) for k, s in initial.species.items()} if initial is not None
                         else {k: float(v) for k, v in self.initial_biomass.items()}),
            medium_mmol_l=medium,
            rubber_g_l=float(initial.rubber_concentration if initial is not None else self.initial_rubber),
            reset_profile='simulator_exact' if exact_reset else 'legacy_starter_medium',
        )

    def get_policy_contract(self) -> dict:
        """Physical meaning of a checkpoint, beyond matching array dimensions."""
        parameters = {key: getattr(self.simulator, key, None) for key in (
            'NUMERICAL_REPAIR_REVISION', 'oxygen_scheme', 'max_internal_dt', 'density_policy', 'flux_selection', 'GROWTH_FLOOR_TOLERANCE_H_INV', 'carrying_capacity', 'max_uptake_rate',
            'nitrogen_half_saturation', 'oxygen_saturation', 'polymer_oxygen_half_saturation',
            'polymer_oxygen_fraction', 'max_pha_fraction_g_gdcw', 'phv_requires_rubber_intermediate',
            'configured_buffer_mmol_l', 'pKa', 'ph_control_target', 'volume',
            'metabolite_inhibition_threshold', 'metabolite_inhibition_decay',
            'cooperative_parsimony', 'cooperative_optimize_live_objectives')}
        if hasattr(self.simulator, 'physiology_contract'):
            parameters['physiology'] = self.simulator.physiology_contract()
        return dict(schema="cultivation_policy_contract_v2",
                    control=self.control_schema_metadata,
                    observation=self.observation_schema_metadata,
                    species=sorted(self.simulator.models),
                    fba_mode=getattr(self.simulator, 'fba_mode', None),
                    controller_dt_h=float(self.simulator.dt),
                    biological_parameters=parameters,
                    initial_gem_identity=self._initial_gem_identity,
                    growth_reactions=dict(getattr(self.simulator, 'growth_reactions', {})),
                    storage_repeat_g_per_mmol=dict(
                        phb=getattr(self.simulator, 'PHB_REPEAT_G_PER_MMOL', .08609),
                        phv=getattr(self.simulator, 'PHV_REPEAT_G_PER_MMOL', .10012)),
                    polymer_default_rates=dict(getattr(self.simulator, 'DEFAULT_POLYMER_RATES', {})),
                    polymer_rate_overrides=dict(self.rubber_degradation_rates),
                    initial_conditions=self._episode_initial_conditions(),
                    reward=dict(pha_weight=500., rubber_weight=10., ph_squared_weight=2.,
                                minimum_species_biomass_g_l=.02, low_biomass_weight=10.,
                                feed_mass_proxy_weight=10., excess_common_feed_weight=self.excess_common_feed_penalty,
                                extinction_biomass_g_l=.01, early_extinction_penalty=100.,
                                pha_reward_repeat_g_per_mmol=dict(phb=.08609, phv=.10012),
                                individual_nominal_mw_g_mol=dict(mlttr_e=504.44, ptrc_e=88.15,
                                                                lac__L_e=90.08, mnl_e=182.17)))

    def get_cultivation_configuration(self) -> dict:
        """Report the actual initial culture, including its chosen Pf inoculum."""
        initial_conditions = self._episode_initial_conditions()
        return dict(policy_contract=self.get_policy_contract(),
                    initial_biomass_g_l=initial_conditions['biomass_g_l'],
                    supplied_initial_medium_mmol_l={k: float(v) for k, v in self.initial_metabolites.items()},
                    episode_initial_medium_mmol_l=initial_conditions['medium_mmol_l'],
                    initial_rubber_g_l=initial_conditions['rubber_g_l'],
                    reset_profile=initial_conditions['reset_profile'],
                    ph_control_target=getattr(self.simulator, 'ph_control_target', None),
                    max_internal_dt_h=getattr(self.simulator, 'max_internal_dt', None),
                    nitrogen_half_saturation_mmol_l=getattr(self.simulator, 'nitrogen_half_saturation', None))

    def _setup_spaces(self):
        num_species = len(self.simulator.models)
        # Biomass(num_species) + pH(1) + DO(1) + Glc(1) + Rubber(1) + C30(1) + ODTD(1) + PHA(1) + BS(1) + AA(3) + Time(1) + Phase(1)
        # 16 if num_species=3
        total_obs_dim = num_species + 13 + (4 if self.observation_schema == "metabolic_v2" else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # Versioned metadata describes physical inputs before normalization.
        # The default layout and encoding are unchanged for existing policies.
        features = [dict(name=f"biomass:{name}", source=f"species.{name}.biomass",
                         input_unit="g/L", encoding="clip(c/scale,0,1)", scale=20.0)
                    for name in sorted(self.simulator.state.species)]
        features.extend([
            dict(name="ph", source="metabolites.h_e", input_unit="pH", encoding="clip(pH/scale,0,1)", scale=14.0),
            dict(name="o2_e", source="metabolites.o2_e", input_unit="mmol/L", encoding="clip(c/scale,0,1)", scale=.25),
            dict(name="glc__D_e", source="metabolites.glc__D_e", input_unit="mmol/L", encoding="clip(log1p(c)/log1p(K),0,1)", scale=100.0),
            dict(name="rubber", source="rubber_concentration", input_unit="g/L", encoding="clip(c/scale,0,1)", scale=100.0),
            dict(name="C30_oligo_e", source="metabolites.C30_oligo_e", input_unit="mmol/L", encoding="clip(log1p(c)/log1p(K),0,1)", scale=300.0),
            dict(name="odtd_e", source="metabolites.odtd_e", input_unit="mmol/L", encoding="clip(log1p(c)/log1p(K),0,1)", scale=500.0),
            dict(name="pha_repeat", source="sum(species.pha_accumulated)", input_unit="mmol/L", encoding="clip(log1p(c)/log1p(K),0,1)", scale=5000.0),
            dict(name="biosurfactant_e", source="metabolites.biosurfactant_e", input_unit="mmol/L", encoding="clip(log1p(c)/log1p(K),0,1)", scale=50.0),
            *[dict(name=name, source=f"metabolites.{name}", input_unit="mmol/L",
                   encoding="clip(log1p(c)/log1p(K),0,1)", scale=100.0)
              for name in ("arg__L_e", "trp__L_e", "leu__L_e")],
            dict(name="time", source="time", input_unit="h", encoding="clip(time/scale,0,1)", scale=float(self.max_time)),
            dict(name="phase_48h", source="time", input_unit="h", encoding="time>=threshold", scale=48.0),
        ])
        if self.observation_schema == "metabolic_v2":
            features.extend([
                dict(name=name, source=f"metabolites.{name}", input_unit="mmol/L",
                     encoding="log1p(c/K)/(1+log1p(c/K))", scale=scale)
                for name, scale in (("nh4_e", .1), ("lac__L_e", 1.), ("ppa_e", 1.))
            ])
            features.append(dict(name="nh4_below_0p1_mmol_l", source="metabolites.nh4_e",
                                 input_unit="mmol/L", encoding="c<threshold", scale=.1))
        self.observation_feature_names = tuple(feature["name"] for feature in features)
        self.observation_schema_metadata = dict(
            schema=self.observation_schema, dimension=total_obs_dim,
            ordered_feature_names=list(self.observation_feature_names), features=features,
            encoded_unit="dimensionless", species_order=sorted(self.simulator.state.species),
            legacy_prefix_dimension=num_species + 13,
            requires_new_policy_and_normalizer=self.observation_schema == "metabolic_v2",
            checkpoint_note=("metabolic_v2 requires a new compatible policy/checkpoint and normalizer; "
                             "do not load a legacy 16-dimensional checkpoint as if compatible"),
            threshold_note="NH4 flag describes the observed state, not a branch executed earlier in the transition",
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
        
        # ゴム関連の直接観測 (ポリマー残量, C30, ODTD)
        obs.append(np.clip(state.rubber_concentration / 100.0, 0.0, 1.0))
        obs.append(norm_met(state.metabolites.get('C30_oligo_e', 0.0), K=300.0))
        obs.append(norm_met(state.metabolites.get('odtd_e', 0.0), K=500.0))
        
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
        if self.observation_schema == "metabolic_v2":
            added_concentrations = {}
            for name, scale in (("nh4_e", .1), ("lac__L_e", 1.), ("ppa_e", 1.)):
                # These exact extracellular keys are initialized by utils.py.
                # Missing, negative, or nonfinite pools are not silently zeroed.
                if name not in state.metabolites:
                    raise ValueError(f"metabolic_v2 requires metabolite {name}")
                raw = np.asarray(state.metabolites[name])
                if raw.ndim != 0 or raw.dtype.kind not in "fiu":
                    raise ValueError(f"metabolic_v2 requires numeric scalar {name}")
                concentration = float(raw)
                if not np.isfinite(concentration) or concentration < 0.0:
                    raise ValueError(f"metabolic_v2 requires finite nonnegative {name}")
                added_concentrations[name] = concentration
                # Stable log(1+c/K), preserving tiny positive inputs and
                # avoiding overflow of the ratio for very large finite c.
                if concentration <= np.finfo(np.float64).max * scale:
                    value = float(np.log1p(concentration / scale))
                else:
                    value = float(np.log(concentration) - np.log(scale)
                                  + np.log1p(scale / concentration))
                obs.append(value / (1.0 + value))
            obs.append(float(added_concentrations["nh4_e"] < .1))
        return np.array(obs, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        simulator_reset = getattr(self.simulator, "reset", None)
        if callable(simulator_reset):
            simulator_reset()
            state = self.simulator.state
            self.current_step = 0
            self.last_total_pha = sum(s.pha_accumulated for s in state.species.values())
            self.last_bs = state.metabolites.get('biosurfactant_e', 0.)
            self.prev_rubber = state.rubber_concentration
            self.last_activity_biomass = sum(s.biomass for s in state.species.values())
            self.last_total_biomass = self.last_activity_biomass
            return self._get_observation(state), {"reset_profile": "simulator_exact",
                                                   "control_schema": self.control_schema}
        # Historical starter-medium overrides are retained only for simulators
        # without an explicit reset contract, so old traces remain replayable.
        self.current_step = 0
        self.last_total_pha = 0.0
        self.last_bs = 0.0
        self.simulator.state.time = 0.0
        self.simulator.state.rubber_concentration = self.initial_rubber
        self.simulator.current_step = 0
        self.simulator.cumulative_co2_emission = 0.0
        self.simulator.last_polymer_fluxes = {}
        self.simulator.cumulative_base_added_mmol_l = 0.0
        self.simulator.cumulative_acid_added_mmol_l = 0.0
        self.simulator.cumulative_defined_feed_g_l = 0.0
        self.simulator.last_defined_feed_g_l = 0.0
        
        # --- 科学的種培地 (Starter Medium) & 誘導物質シード ---
        self.simulator.state.metabolites = self.initial_metabolites.copy()
        
        # 初期アミノ酸を調整 (1.0 mM): 急激な代謝スパイクを抑えつつ酵素合成を支援
        self.simulator.state.metabolites['arg__L_e'] = 1.0
        self.simulator.state.metabolites['trp__L_e'] = 1.0
        self.simulator.state.metabolites['leu__L_e'] = 1.0
        # 微量のグルコース (0.5 mM) を追加し、エネルギー基盤を安定させる
        self.simulator.state.metabolites['glc__D_e'] = 0.5
        
        # C30 is not an inducer feed. It must be generated from tracked bulk
        # rubber by the extracellular Lcp/Rox kinetic layer.
        self.simulator.state.metabolites['C30_oligo_e'] = 0.0
        
        # 初期pHを確実に中性付近にするため、プロトン濃度を調整
        self.simulator.state.metabolites['h_e'] = 10**(-7.21) * 1000.0 
        # The L. plantarum biosurfactant route remains an uncalibrated
        # hypothesis and is therefore not seeded as free material.
        self.simulator.state.metabolites['biosurfactant_e'] = 0.0
        
        for name, s in self.simulator.state.species.items():
            s.biomass = self.initial_biomass[name]
            s.growth_rate = 0.0
            s.pha_accumulated = 0.0
            s.phb_accumulated = 0.0
            s.phv_accumulated = 0.0
            
        self.simulator._initialize_medium()
        self.prev_rubber = self.initial_rubber
        self.last_activity_biomass = sum(s.biomass for s in self.simulator.state.species.values())
        self.last_total_biomass = self.last_activity_biomass

        return self._get_observation(self.simulator.state), {"reset_profile": "legacy_starter_medium",
                                                            "control_schema": self.control_schema}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all() or np.any((action < 0.) | (action > 1.)):
            raise ValueError("physical action must contain five finite values in [0, 1]")
        current_time = self.simulator.state.time
        previous_defined_feed = float(self.simulator.cumulative_defined_feed_g_l)
        previous_pha_g_l = self._pha_mass(self.simulator.state)
        
        # フェーズに応じた定義共通培地の供給倍率
        if current_time < 48.0:
            max_feed_common = self.max_common_feed_early
        else:
            max_feed_common = self.max_common_feed_late
            
        max_feed_specific = self.max_specific_feed_per_step
        has_propionibacterium = any(
            "freudenreichii" in name.lower()
            for name in self.simulator.models
        )
        nutrient_supplementation = {
            'sn_or16': action[0] * max_feed_specific,
            'sn_ns21': action[1] * max_feed_specific,
            'coexistence_feed_rate': action[3] * max_feed_common
        }
        if has_propionibacterium:
            # The third physical feed is L-lactate for the helper's
            # lactate -> propionate route; it is not the obsolete WCFS1
            # mannitol-specific feed.
            nutrient_supplementation['helper_lactate'] = (
                action[2] * max_feed_specific
            )
        else:
            nutrient_supplementation['sn_lp'] = action[2] * max_feed_specific
        dynamic_kla = action[4] * 200.0
        
        additional_step_kwargs = {}
        specific_feed_mass_proxy = 0.
        feed_rates = {}
        if self.control_schema == "audited_rates_v2":
            third = 'lac__L_e' if has_propionibacterium else 'mnl_e'
            feed_rates = {name: float(action[i]) * self.max_specific_feed_rate_mmol_l_h
                          for i, name in enumerate(('mlttr_e', 'ptrc_e', third))}
            nutrient_supplementation = {'coexistence_feed_rate': action[3] * max_feed_common}
            additional_step_kwargs['feed_rates_mmol_l_h'] = feed_rates
        # シミュレーター実行 (FBAソルバー呼び出し)
        state = self.simulator.step(
            self.rubber_degradation_rates,
            nutrient_supplementation,
            dynamic_kla=dynamic_kla,
            **additional_step_kwargs,
        )
        elapsed_h = float(state.time - current_time)
        defined_feed_g_l = float(self.simulator.cumulative_defined_feed_g_l) - previous_defined_feed
        if self.control_schema == "audited_rates_v2":
            # Declared molecular forms only; this is not commercial stock cost.
            nominal_mw = {'mlttr_e': 504.44, 'ptrc_e': 88.15,
                          'lac__L_e': 90.08, 'mnl_e': 182.17}
            specific_feed_mass_proxy = sum(rate * elapsed_h * nominal_mw[met] / 1000.
                                           for met, rate in feed_rates.items())
        
        obs = self._get_observation(state)
        
        # 共通メトリクスの計算 (info用)
        current_total_pha = sum(s.pha_accumulated for s in state.species.values())
        current_bs = state.metabolites.get('biosurfactant_e', 0.0)
        current_total_biomass = sum(s.biomass for s in state.species.values())
        
        # ===================================================================
        # 報酬関数 v3.1: 科学的最適化（PHA生産へのインセンティブ強化）
        # ===================================================================
        reward = 0.0
        h_conc = state.metabolites.get('h_e', 0.0001)
        current_ph = -np.log10(max(1e-12, h_conc) / 1000.0)

        # ===== 目的関数（最終成果物 PHA を最優先） =====
        # 1. ゴム分解量: PHA生産の準備段階として評価
        delta_deg = max(0, self.prev_rubber - state.rubber_concentration)
        reward += delta_deg * 10.0  # 50.0 -> 10.0 (重みを下げ、PHAへの移行を促す)

        # 2. PHA蓄積量: 最終目的。窒素枯渇を恐れず最大化させる
        delta_pha = max(0, current_total_pha - self.last_total_pha)
        if self.control_schema == "audited_rates_v2":
            delta_pha = self._pha_mass(state) - previous_pha_g_l
        reward += delta_pha * 500.0 # 50.0 -> 500.0 (10倍に強化)

        # ===== 制約条件（違反した場合のみペナルティ） =====
        # 3. pH安定化: 中性から外れた分だけ二次ペナルティ
        ph_dist = abs(current_ph - 7.0)
        penalty_duration = elapsed_h if self.control_schema == "audited_rates_v2" else 1.
        reward -= 2.0 * (ph_dist ** 2) * penalty_duration

        # 4. 種の多様性維持: PHA生産フェーズでの菌体減少を許容するため閾値を下げる
        #    （飢餓状態での生存限界まで攻めることを許可）
        min_biomass = min(s.biomass for s in state.species.values())
        if min_biomass < 0.02: # 0.05 -> 0.02
            reward -= 10.0 * (0.02 - min_biomass) * penalty_duration

        # ===== 実供給質量に基づく穏やかなコスト制約 =====
        reward -= 10.0 * (defined_feed_g_l + specific_feed_mass_proxy)
        # Fixed-25より多い共通培地を使わずに改善する、事前に定めた
        # 調達・運用制約。閾値以下の操作にはペナルティを課さない。
        excess_common_feed = max(
            0.0, float(action[3]) - self.common_feed_action_budget
        )
        reward -= self.excess_common_feed_penalty * excess_common_feed * penalty_duration
        # バイオマス報酬は「なし」
        # バイオマス報酬撤廃: 菌を増やすこと自体は目的ではない

        # 履歴の更新
        self.last_total_pha = current_total_pha
        self.last_bs = current_bs
        self.last_total_biomass = current_total_biomass

        # ===== 終了条件（緩和: 全種死滅で終了） =====
        # 1種死亡→即終了 を廃止。残存種でパイプラインを維持できる余地を残す
        terminated = all(s.biomass < 0.01 for s in state.species.values())

        if terminated and state.time < self.max_time:
            # 全滅ペナルティ（固定値。残寿命に比例させない）
            reward -= 100.0
            
        truncated = state.time >= self.max_time
        self.current_step += 1
        self.prev_rubber = state.rubber_concentration
        
        info = {
            'rubber_remaining': state.rubber_concentration,
            'total_rubber_degraded': self.initial_rubber - state.rubber_concentration,
            'survival_hours': state.time,
            'ph': current_ph,
            'do': state.metabolites.get('o2_e', 0.25),
            'total_pha': current_total_pha,
            'pha_g_l': self._pha_mass(state),
            'time_h': float(state.time),
            'initial_rubber_g_l': float(self.initial_rubber),
            'control_schema': self.control_schema,
            'numerics_version': getattr(self.simulator, 'NUMERICS_VERSION', 'legacy_dfba'),
            'defined_feed_g_l_step': defined_feed_g_l,
            'specific_feed_mass_proxy_g_l_step': specific_feed_mass_proxy,
            'specific_feed_mmol_l_step': {met: rate * elapsed_h for met, rate in feed_rates.items()},
            'biomass_g_l': {name: float(s.biomass) for name, s in state.species.items()},
            'biomass_pf': sum(s.biomass for name, s in state.species.items() if 'freudenreichii' in name.lower()),
            'biosurfactant': current_bs,
            'biomass_or16': state.species.get('Actinoplanes_sp_OR16_lcp', state.species.get('Engine 1', state.species.get(next(iter(state.species)), None))).biomass,
            'biomass_ns21': state.species.get('Rhizobacter_gummiphilus_NS21', state.species.get('Engine 2')).biomass if ('Rhizobacter_gummiphilus_NS21' in state.species or 'Engine 2' in state.species) else 0.0,
            'biomass_lp': state.species.get('Lactobacillus_plantarum', state.species.get('Stabilizer')).biomass if ('Lactobacillus_plantarum' in state.species or 'Stabilizer' in state.species) else 0.0
        }
        
        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _pha_mass(state) -> float:
        return float(sum(getattr(s, 'phb_accumulated', 0.) * .08609
                         + getattr(s, 'phv_accumulated', 0.) * .10012
                         for s in state.species.values()))

    def get_solver_diagnostics(self) -> dict:
        """Expose simulator diagnostics through VecEnv worker RPC."""

        return self.simulator.get_solver_diagnostics()


class SymmetricPPOActionWrapper(gym.ActionWrapper):
    """Map a symmetric PPO action space to physical pump/kLa fractions.

    PPO represents continuous actions with a Gaussian. A native ``[0, 1]``
    space puts its initial mean on the lower boundary and makes stochastic
    training actions differ sharply from deterministic deployment actions.
    The policy therefore operates in ``[-1, 1]`` while the wrapped culture
    environment continues to receive controls in ``[0, 1]``.
    """

    def __init__(self, env: gym.Env, common_feed_cap: float = 0.25):
        super().__init__(env)
        self.common_feed_cap = float(np.clip(common_feed_cap, 0.0, 1.0))
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    @staticmethod
    def to_physical(
        action: np.ndarray, common_feed_cap: float | None = 0.25
    ) -> np.ndarray:
        policy_action = np.asarray(action, dtype=np.float32)
        physical = np.clip(0.5 * (policy_action + 1.0), 0.0, 1.0)
        if common_feed_cap is not None:
            physical[..., 3] = np.minimum(
                physical[..., 3], float(np.clip(common_feed_cap, 0.0, 1.0))
            )
        return physical.astype(
            np.float32, copy=False
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        return self.to_physical(action, common_feed_cap=self.common_feed_cap)
