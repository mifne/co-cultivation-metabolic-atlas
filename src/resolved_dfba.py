"""Versioned, internally resolved dFBA with coupled oxygen accounting.

The legacy simulator remains unchanged for immutable LP trace replay.
NH4 allocation parameters are hypotheses, not fitted RDE2/NS21 measurements.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .dfba_simulator import dFBASimulator


def oxygen_interval(c0: float, kla: float, dt: float, saturation: float = .25):
    """Constant-rate sink coefficients for dC/dt=kLa*(Cs-C)-OUR.

    Returns free endpoint, sink response time, free mean concentration and
    maximum total withdrawal over the interval that leaves C_end >= 0.
    All concentrations are mmol/L; time is hours.
    """
    if not all(math.isfinite(v) for v in (c0, kla, dt, saturation)):
        raise ValueError('oxygen inputs must be finite')
    if c0 < 0 or kla < 0 or dt <= 0 or saturation <= 0:
        raise ValueError('invalid oxygen interval')
    if kla == 0:
        return c0, dt, c0, c0
    response = -math.expm1(-kla * dt) / kla
    free_end = c0 + (saturation - c0) * kla * response
    free_mean = saturation + (c0 - saturation) * response / dt
    return free_end, response, max(0., free_mean), free_end * dt / response


class ResolvedDFBASimulator(dFBASimulator):
    """CPU reference dynamics, with controller interval ``dt`` unchanged.

    Use ``feed_rates_mmol_l_h`` for continuous feed. Existing named nutrient
    doses retain their once-per-controller-call bolus semantics. External
    mutation of medium is also a bolus and must not be described as continuous.
    """
    NUMERICS_VERSION = 'resolved_oxygen_nh4_v1'

    def __init__(self, *args, max_internal_dt=.025,
                 nitrogen_half_saturation=.1, oxygen_saturation=.25, **kwargs):
        if kwargs.get('fba_mode', 'separate') != 'separate':
            raise ValueError('resolved reference currently requires separate FBA')
        if kwargs.get('cooperative_frozen_inputs', False):
            raise ValueError('legacy frozen LP traces cannot use resolved dynamics')
        if kwargs.get('solver_backend', 'highs') not in ('highs', 'glpk'):
            raise ValueError('resolved reference requires highs or glpk')
        kwargs.setdefault('solver_backend', 'highs')
        super().__init__(*args, **kwargs)
        for key, value in [('max_internal_dt', max_internal_dt),
                           ('nitrogen_half_saturation', nitrogen_half_saturation),
                           ('oxygen_saturation', oxygen_saturation), ('dt', self.dt)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        self.max_internal_dt = float(max_internal_dt)
        self.nitrogen_half_saturation = float(nitrogen_half_saturation)
        self.oxygen_saturation = float(oxygen_saturation)
        self.controller_steps = 0
        self.cumulative_continuous_feed = {}
        self.oxygen_audit = dict(transferred=0., polymer_consumed=0.,
                                 cellular_consumed=0., max_balance_error=0.)
        self._oxygen_context = None
        self.nitrogen_allocation = {}

    def step(self, rubber_degradation_rates, nutrient_supplementation,
             dynamic_kla=50., *, feed_rates_mmol_l_h: Mapping[str, float] | None = None):
        control_dt = float(self.dt)
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError('controller dt must be positive and finite')
        if not math.isfinite(dynamic_kla) or dynamic_kla < 0:
            raise ValueError('kLa must be nonnegative and finite')
        rates = dict(feed_rates_mmol_l_h or {})
        if any(not math.isfinite(v) or v < 0 for v in rates.values()):
            raise ValueError('feed rates must be finite and nonnegative')
        if any(k in {'o2_e', 'h_e', 'h2o_e'} for k in rates):
            raise ValueError('gas/water/pH inputs need their physical interfaces')
        end_time = self.state.time + control_dt
        first = True
        try:
            while self.state.time < end_time - 1e-12:
                self.dt = min(self.max_internal_dt, end_time - self.state.time)
                for met, rate in rates.items():
                    dose = rate * self.dt
                    self.state.metabolites[met] = self.state.metabolites.get(met, 0.) + dose
                    self.cumulative_continuous_feed[met] = self.cumulative_continuous_feed.get(met, 0.) + dose
                supplements = dict(nutrient_supplementation) if first else {
                    key: value for key, value in nutrient_supplementation.items()
                    if key == 'coexistence_feed_rate'}
                c0 = max(0., self.state.metabolites.get('o2_e', self.oxygen_saturation))
                free, response, mean, budget = oxygen_interval(
                    c0, dynamic_kla, self.dt, self.oxygen_saturation)
                self._oxygen_context = dict(c0=c0, free=free, response=response,
                                            mean=mean, budget=budget, polymer=0.)
                # Base step's transfer becomes the identity. This subclass
                # integrates both transfer and consumption in _update_environment.
                super().step(rubber_degradation_rates, supplements, dynamic_kla=0.)
                first = False
            self.state.time = end_time
            self.controller_steps += 1
        finally:
            self.dt = control_dt
            self._oxygen_context = None
        return self.state

    def degrade_rubber(self, rates):
        ctx = self._oxygen_context
        if ctx is None:
            return super().degrade_rubber(rates)
        physical = self.state.metabolites.get('o2_e', 0.)
        self.state.metabolites['o2_e'] = ctx['budget']
        super().degrade_rubber(rates)
        ctx['polymer'] = self.last_polymer_fluxes['oxygen_mmol_l_step']
        # The integration budget is never a dissolved concentration used in
        # uptake kinetics. The free-interval mean is a first-order predictor.
        self.state.metabolites['o2_e'] = physical

    def set_uptake_constraints(self, name, medium, kla=50.):
        super().set_uptake_constraints(name, medium, kla)
        ctx = self._oxygen_context
        rid = self.exchange_reactions[name].get('o2_e')
        if ctx is None or not rid:
            return
        users = [n for n in self.models if 'o2_e' in self.exchange_reactions[n]
                 and self.original_bounds[n].get(self.exchange_reactions[n]['o2_e'], (0., 0.))[0] < 0]
        biomass = sum(max(1e-6, self.state.species[n].biomass) for n in users)
        available = max(0., ctx['budget'] - ctx['polymer'])
        limit = min(self.max_uptake_rate * ctx['mean'] / (.01 + ctx['mean']),
                    available / (max(biomass, 1e-12) * self.dt))
        rxn = self.models[name].reactions.get_by_id(rid)
        original = self.original_bounds[name].get(rid, (0., 0.))[0]
        # Do not round a conservation bound outward to six decimal places.
        rxn.lower_bound = min(rxn.upper_bound, max(-limit, original))

    def solve_fba(self, name):
        if 'NS21' not in name:
            return super().solve_fba(name)
        model = self.models[name]
        growth_id = 'R_Growth' if 'R_Growth' in model.reactions else 'Growth'
        if growth_id not in model.reactions:
            raise ValueError('NS21 growth reaction not found')
        growth = model.reactions.get_by_id(growth_id)
        original_lb = growth.lower_bound
        growth.lower_bound = original_lb
        sinks = {k: self.exchange_reactions[name].get(k) for k in ['pha_c', 'phv_c']}
        for rid in sinks.values():
            if rid: model.reactions.get_by_id(rid).upper_bound = 0.
        self._set_surrogate_objective(name, model, growth_id)
        potential = super().solve_fba(name)
        self.solve_attempts += 1
        self.solve_successes += int(potential is not None)
        if potential is None:
            return None
        nh4 = max(0., self.state.metabolites.get('nh4_e', 0.))
        fraction = nh4 / (self.nitrogen_half_saturation + nh4)
        mu = max(0., float(potential.fluxes[growth_id]))
        growth.lower_bound = max(original_lb, fraction * mu)
        weights = {}
        for key, mass in [('pha_c', self.PHB_REPEAT_G_PER_MMOL),
                          ('phv_c', self.PHV_REPEAT_G_PER_MMOL)]:
            rid = sinks[key]
            allowed = key == 'pha_c' or not self.phv_requires_rubber_intermediate or any(
                self.state.metabolites.get(k, 0.) > 1e-12 for k in ['C30_oligo_e', 'odtd_e'])
            if rid and allowed:
                model.reactions.get_by_id(rid).upper_bound = max(0., self.original_bounds[name].get(rid, (0., 1000.))[1])
                weights[rid] = mass
        self.nitrogen_allocation[name] = dict(nh4=nh4, growth_fraction=fraction,
                                             growth_potential=mu)
        self._set_surrogate_objective_weights(name, model, weights)
        try:
            return super().solve_fba(name)
        finally:
            growth.lower_bound = original_lb

    def _update_environment(self, solutions, integration_biomass=None):
        ctx = self._oxygen_context
        if ctx is None:
            return super()._update_environment(solutions, integration_biomass)
        cellular = sum(-fluxes.get(self.exchange_reactions[name].get('o2_e'), 0.)
                       * integration_biomass[name] * self.dt
                       for name, fluxes in solutions.items())
        withdrawal = ctx['polymer'] + cellular
        final = ctx['free'] - withdrawal / self.dt * ctx['response']
        if final < -1e-7:
            raise RuntimeError(f'oxygen budget exceeded: C_end={final}')
        final = max(0., final)
        transferred = final - ctx['c0'] + withdrawal
        super()._update_environment(solutions, integration_biomass)
        self.state.metabolites['o2_e'] = final
        self.oxygen_audit['transferred'] += transferred
        self.oxygen_audit['polymer_consumed'] += ctx['polymer']
        self.oxygen_audit['cellular_consumed'] += cellular
        self.oxygen_audit['max_balance_error'] = max(self.oxygen_audit['max_balance_error'],
            abs(final - ctx['c0'] - transferred + withdrawal))

    def get_solver_diagnostics(self):
        result = super().get_solver_diagnostics()
        result['resolved_dynamics'] = dict(version=self.NUMERICS_VERSION,
            controller_dt=self.dt, max_internal_dt=self.max_internal_dt,
            controller_steps=self.controller_steps,
            nitrogen_half_saturation=self.nitrogen_half_saturation,
            nitrogen_parameters_calibrated=False, oxygen_saturation=self.oxygen_saturation,
            oxygen=dict(self.oxygen_audit), continuous_feed=dict(self.cumulative_continuous_feed),
            nitrogen_allocation=dict(self.nitrogen_allocation))
        return result
