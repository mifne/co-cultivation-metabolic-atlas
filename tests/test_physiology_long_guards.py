"""Preflight regressions before investing in long physiology trajectories."""
import numpy as np
import pytest
from tests.test_physiology_dfba import make,model
from src.physiology_dfba import PhysiologyDFBASimulator,is_atp_hydrolysis
from src.rl_environment import ConsortiumEnv


def test_physiology_uses_rate_feed_and_mass_reward_in_real_env_steps():
    runs=[]
    for dt,count in [(.1,1),(.05,2)]:
        sim=make(name='NS21',nh4=0.,dt=dt)
        env=ConsortiumEnv(sim,max_time=1.)
        assert env.control_schema=='audited_rates_v2'
        assert env.control_schema_metadata['pha_reward_unit']=='g/L'
        env.reset();reward=0.
        for _ in range(count):
            _,r,_,_,info=env.step(np.array([.7,0.,0.,0.,.25]))
            reward+=r
        assert sim.cumulative_delivered_mmol_l['mlttr_e']==pytest.approx(.007)
        expected=500*env._pha_mass(sim.state)-10*.007*504.44/1000
        assert reward==pytest.approx(expected,abs=1e-9)
        runs.append((reward,sim.state.species['NS21'].biomass,info['pha_g_l']))
    assert runs[0]==pytest.approx(runs[1],abs=1e-9)


def test_legacy_control_contract_is_rejected_for_physiology():
    with pytest.raises(ValueError,match='new audited_rates_v2'):
        ConsortiumEnv(make(),control_schema='legacy_v1')


def test_reverse_storage_boundary_cannot_supply_unowned_pha():
    m=model();m.reactions.EX_pha_c.lower_bound=-1000
    sim=PhysiologyDFBASimulator({'NS21':m},{'NS21':.1},
        {'glc__D_e':0.,'nh4_e':1.,'o2_e':.25},initial_rubber=0.,dt=.1,ph_control_target=7.,remobilize_pha=False)
    sim.step({},{})
    assert sim.state.species['NS21'].biomass==pytest.approx(.1,abs=1e-10)
    assert sim.state.species['NS21'].phb_accumulated==pytest.approx(0.,abs=1e-10)


@pytest.mark.parametrize('missing',['formula','charge'])
def test_missing_maintenance_chemistry_is_not_a_balance_certificate(missing):
    m=model();setattr(m.metabolites.atp_c,missing,None)
    assert not is_atp_hydrolysis(m.reactions.ATPM)


def test_death_with_live_storage_preserves_total_pha_over_many_steps():
    sim=make(carbon=0.,nh4=0.,dt=1.,basal_death_rates={'A':.2})
    s=sim.state.species['A'];s.phb_accumulated=1.;s.pha_accumulated=1.
    for _ in range(24):sim.step({},{})
    assert s.phb_accumulated+sim.dead_matter['A']['phb_mmol_l']==pytest.approx(1.,abs=1e-9)
    assert s.biomass==pytest.approx(.1*np.exp(-.2*24),abs=1e-9)
    assert sim.element_accounting()['known_medium_closure_residual']==pytest.approx({'C':0.,'N':0.},abs=1e-9)
