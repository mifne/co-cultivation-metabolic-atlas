"""Controller-unit and episode-boundary regressions with a deterministic plant."""
import copy
from types import SimpleNamespace

import cobra
import numpy as np
import pytest

from src.rl_environment import ConsortiumEnv


class AuditedPlant:
    NUMERICS_VERSION = 'audited_cultivation_v2'

    def __init__(self, dt=1.):
        self.dt = dt
        self.models = {name: object() for name in ('OR16', 'NS21', 'Propionibacterium_freudenreichii')}
        self.initial = SimpleNamespace(
            time=0., rubber_concentration=10.,
            metabolites={'glc__D_e': .02, 'arg__L_e': .01, 'o2_e': .1,
                         'h_e': .0001, 'nh4_e': .05, 'lac__L_e': 0., 'ppa_e': 0.},
            species={name: SimpleNamespace(biomass=.1, growth_rate=0., pha_accumulated=0.,
                                          phb_accumulated=0., phv_accumulated=0.) for name in self.models})
        self.reset()

    def reset(self):
        self.state = copy.deepcopy(self.initial)
        self.cumulative_defined_feed_g_l = 0.
        self.last_defined_feed_g_l = 0.
        self.delivered = {}
        return self.state

    def step(self, rubber, nutrients, dynamic_kla, *, feed_rates_mmol_l_h=None):
        self.state.time += self.dt
        self.state.rubber_concentration -= .01 * self.dt
        for met, rate in (feed_rates_mmol_l_h or {}).items():
            self.delivered[met] = self.delivered.get(met, 0.) + rate * self.dt
        feed = nutrients.get('coexistence_feed_rate', 0.) * self.dt * .2
        self.cumulative_defined_feed_g_l += feed
        # Mimic a substepped simulator whose last internal interval is tiny.
        self.last_defined_feed_g_l = feed / 40.
        ns = self.state.species['NS21']
        ns.phb_accumulated += .1 * self.dt
        ns.pha_accumulated = ns.phb_accumulated
        return self.state


def test_exact_reset_preserves_requested_medium_and_discards_episode_counters():
    sim = AuditedPlant()
    env = ConsortiumEnv(sim, observation_schema='metabolic_v2')
    env.reset()
    env.step(np.full(5, .5))
    _, info = env.reset()
    assert sim.state.metabolites == sim.initial.metabolites
    assert sim.state.time == 0. and sim.delivered == {}
    assert sim.cumulative_defined_feed_g_l == 0.
    assert env.last_total_pha == 0.
    assert info['reset_profile'] == 'simulator_exact'
    configuration = env.get_cultivation_configuration()
    assert configuration['episode_initial_medium_mmol_l'] == sim.state.metabolites
    assert configuration['initial_biomass_g_l']['Propionibacterium_freudenreichii'] == .1


def test_rate_controls_reward_and_total_doses_are_controller_partition_invariant():
    outcomes = []
    for dt in (1., .2):
        sim = AuditedPlant(dt)
        env = ConsortiumEnv(sim, max_specific_feed_rate_mmol_l_h=.3,
                            common_feed_action_budget=1.)
        env.reset()
        total_reward = 0.
        total_recorded_feed = 0.
        for _ in range(round(1./dt)):
            _, reward, _, _, info = env.step(np.full(5, .5))
            total_reward += reward
            total_recorded_feed += info['defined_feed_g_l_step']
        assert sim.delivered['lac__L_e'] == pytest.approx(.15)
        assert total_recorded_feed == pytest.approx(.2)
        assert info['biomass_pf'] == .1
        assert info['control_schema'] == 'audited_rates_v2'
        assert info['pha_g_l'] == pytest.approx(.1 * .08609)
        outcomes.append(total_reward)
    assert outcomes[0] == pytest.approx(outcomes[1])


def test_legacy_reward_also_charges_all_internal_common_feed():
    sim = AuditedPlant()
    sim.NUMERICS_VERSION = 'resolved_oxygen_nh4_v1'
    env = ConsortiumEnv(sim, common_feed_action_budget=1.)
    env.reset()
    _, reward, _, _, info = env.step(np.array([0., 0., 0., .5, 0.]))
    assert info['defined_feed_g_l_step'] == pytest.approx(.2)
    assert reward == pytest.approx(.01 * 10. + .1 * 500. - .2 * 10.)


@pytest.mark.parametrize('bad', [np.full(5, np.nan), np.full(5, -1.), np.full(5, 1.1), np.ones(4)])
def test_invalid_control_fails_before_mutating_plant(bad):
    sim = AuditedPlant()
    env = ConsortiumEnv(sim)
    env.reset()
    with pytest.raises(ValueError, match='five finite'):
        env.step(bad)
    assert sim.state.time == 0. and sim.delivered == {}


def test_audited_model_rejects_legacy_control_semantics():
    with pytest.raises(ValueError, match='new audited_rates_v2 policy'):
        ConsortiumEnv(AuditedPlant(), control_schema='legacy_v1')


@pytest.mark.parametrize('parameter', ['density_policy', 'flux_selection', 'nitrogen_half_saturation',
    'polymer_oxygen_half_saturation', 'oxygen_saturation', 'max_pha_fraction_g_gdcw', 'max_internal_dt',
    'NUMERICAL_REPAIR_REVISION', 'oxygen_scheme'])
def test_policy_contract_changes_with_biological_and_numerical_parameters(parameter):
    first, second = AuditedPlant(), AuditedPlant()
    values = (('none', 'flux_consistent') if parameter == 'density_policy' else
              ('primary_only', 'parsimonious_exchange') if parameter == 'flux_selection' else (.1, .2))
    setattr(first, parameter, values[0])
    setattr(second, parameter, values[1])
    left = ConsortiumEnv(first).get_policy_contract()
    right = ConsortiumEnv(second).get_policy_contract()
    assert left != right
    assert left['biological_parameters'][parameter] == values[0]
    assert right['biological_parameters'][parameter] == values[1]


def test_policy_contract_preserves_initial_scenario_and_distinguishes_gem_changes():
    def make_model(bound, coefficient):
        model = cobra.Model('same_GEM_name')
        a, b = cobra.Metabolite('a_c', compartment='c'), cobra.Metabolite('b_c', compartment='c')
        reaction = cobra.Reaction('same_reaction_id')
        reaction.bounds = (0., bound)
        reaction.add_metabolites({a: -1., b: coefficient})
        model.add_reactions([reaction])
        model.objective = reaction
        return model
    contracts = []
    for bound, coefficient in [(10., 1.), (20., 1.), (10., 2.)]:
        sim = AuditedPlant()
        sim.models['NS21'] = make_model(bound, coefficient)
        env = ConsortiumEnv(sim)
        contract = env.get_policy_contract()
        assert contract['initial_conditions']['medium_mmol_l']['nh4_e'] == .05
        assert contract['initial_conditions']['biomass_g_l']['Propionibacterium_freudenreichii'] == .1
        contracts.append(contract['initial_gem_identity']['NS21']['initial_gem_sha256'])
        # Live solve bounds/objectives are not the initial GEM identity.
        sim.models['NS21'].reactions[0].bounds = (0., 0.)
        assert env.get_policy_contract()['initial_gem_identity'] == contract['initial_gem_identity']
    assert len(set(contracts)) == 3


def test_policy_contract_distinguishes_initial_medium_with_identical_gems():
    first, second = AuditedPlant(), AuditedPlant()
    second.state.metabolites['nh4_e'] = .2
    assert ConsortiumEnv(first).get_policy_contract() != ConsortiumEnv(second).get_policy_contract()
