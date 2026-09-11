"""Unused uptake capacity must not receive a timestep-dependent oxygen share."""
import math
import pytest
from tests.test_audited_dfba import make_sim


def fixed_population_equilibrium(dt,scheme,inactive=True):
    # Hold populations/nutrients fixed to isolate the fast oxygen equilibrium.
    # B can import oxygen but cannot use it: its sole consuming growth route
    # is shut. All oxygen uptake fluxes are nevertheless ordinary certified LPs.
    sim=make_sim(species=('A','B') if inactive else ('A',),biomass=1.,dt=dt,internal=dt,
        density_policy='none',oxygen_scheme=scheme)
    if inactive:sim.models['B'].reactions.get_by_id('Growth').upper_bound=0.
    sim._integration_biomass={name:1. for name in sim.models}
    sim.state.metabolites['o2_e']=0.
    for _ in range(40):
        solutions=sim._solve_oxygen_coupled(1.,{})
        our=sum(-solution.fluxes['EX_o2_e'] for solution in solutions.values())
        ctx=sim._oxygen_context
        sim.state.metabolites['o2_e']=ctx['free']-our*ctx['response']
    return sim.state.metabolites['o2_e'],sim


def test_unused_capacity_reproduces_the_old_quota_timestep_artifact():
    values=[fixed_population_equilibrium(dt,'quota_mean_v2')[0] for dt in [.025,.00625]]
    assert values[0]>3*values[1]


@pytest.mark.parametrize('dt',[.1,.025,.00625])
def test_shared_endpoint_matches_continuous_equilibrium_with_unused_competitor(dt):
    # 20*D/(.01+D) = 1*(.25-D); stable positive quadratic root.
    b=20.+.01-.25
    expected=2*.25*.01/(b+math.sqrt(b*b+4*.25*.01))
    value,sim=fixed_population_equilibrium(dt,'shared_endpoint_v3')
    alone,_=fixed_population_equilibrium(dt,'shared_endpoint_v3',inactive=False)
    assert value==pytest.approx(expected,abs=1e-8)
    assert value==pytest.approx(alone,abs=1e-8)
    assert sim.accounting_audit['max_oxygen_kinetic_residual']<1e-6
    assert sim._oxygen_last_mean==pytest.approx(value,abs=1e-8)
    sim.reset()
    assert sim._oxygen_last_mean is None and sim._oxygen_last_guess is None


def test_unknown_oxygen_scheme_is_rejected():
    with pytest.raises(ValueError,match='oxygen_scheme'):make_sim(oxygen_scheme='unknown')
