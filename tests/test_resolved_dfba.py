import math
import cobra
import numpy as np
import pytest

from src.resolved_dfba import ResolvedDFBASimulator, oxygen_interval


@pytest.mark.parametrize('kla', [0., 5., 50.])
def test_oxygen_constant_sink_matches_analytical_solution_for_any_partition(kla):
    initial=.2; saturation=.25; sink=.01; duration=2.
    expected=initial-sink*duration if kla==0 else saturation-sink/kla+(initial-saturation+sink/kla)*math.exp(-kla*duration)
    for step in [1., .2, .025]:
        c=initial;transfer=0.
        for _ in range(round(duration/step)):
            free,response,mean,budget=oxygen_interval(c,kla,step,saturation)
            assert budget >= sink*step
            end=free-sink*response
            transfer+=end-c+sink*step;c=end
        assert c==pytest.approx(expected,abs=1e-12)
        assert initial+transfer-sink*duration==pytest.approx(c,abs=1e-12)


def test_oxygen_inventory_bound_and_zero_airflow_limit():
    for step in [1.,.025,.001]:
        free,response,mean,budget=oxygen_interval(.03,50.,step)
        assert free-budget/step*response==pytest.approx(0.,abs=1e-12)
    assert oxygen_interval(.03,0.,1.)[-1]==.03
    with pytest.raises(ValueError):oxygen_interval(.2,-1.,1.)


def toy(dt=1., internal=.025, nh4=.1):
    m=cobra.Model('toy')
    pools={name:cobra.Metabolite(name,compartment='e' if name.endswith('_e') else 'c')
           for name in ['glc__D_e','nh4_e','o2_e','c_c','n_c','o_c','pha_c']}
    for ext,inside in [('glc__D_e','c_c'),('nh4_e','n_c'),('o2_e','o_c')]:
        ex=cobra.Reaction('EX_'+ext);ex.add_metabolites({pools[ext]:-1});ex.bounds=(-20,1000)
        tr=cobra.Reaction('T_'+ext);tr.add_metabolites({pools[ext]:-1,pools[inside]:1})
        m.add_reactions([ex,tr])
    g=cobra.Reaction('Growth');g.add_metabolites({pools['c_c']:-1,pools['n_c']:-1,pools['o_c']:-1})
    p=cobra.Reaction('PHA');p.add_metabolites({pools['c_c']:-1,pools['o_c']:-1,pools['pha_c']:1})
    sink=cobra.Reaction('EX_pha_c');sink.add_metabolites({pools['pha_c']:-1})
    m.add_reactions([g,p,sink]);m.objective=g
    return ResolvedDFBASimulator(models={'NS21':m},initial_biomass={'NS21':.1},
        initial_metabolites={'glc__D_e':2.,'nh4_e':nh4,'o2_e':.25},initial_rubber=0.,
        dt=dt,max_internal_dt=internal,solver_backend='highs',ph_control_target=7.)


def test_continuous_feed_and_endpoint_do_not_depend_on_controller_partition():
    states=[]
    for dt in [1., .2, .1]:
        sim=toy(dt)
        for _ in range(round(1/dt)):
            sim.step({}, {},dynamic_kla=5.,feed_rates_mmol_l_h={'glc__D_e':.1,'nh4_e':.02})
        states.append([sim.state.species['NS21'].biomass,sim.state.species['NS21'].pha_accumulated,
                       sim.state.metabolites['o2_e'],sim.state.metabolites['nh4_e']])
        assert sim.dt==dt
        assert sim.cumulative_continuous_feed['glc__D_e']==pytest.approx(.1)
        assert sim.cumulative_continuous_feed['nh4_e']==pytest.approx(.02)
        assert sim.get_solver_diagnostics()['solve_success_rate']==1.
        audit=sim.oxygen_audit
        assert .25+audit['transferred']-audit['polymer_consumed']-audit['cellular_consumed']==pytest.approx(sim.state.metabolites['o2_e'],abs=1e-10)
    assert np.max(np.abs(np.array(states)-states[0]))<1e-8


def test_growth_and_storage_coexist_continuously_across_old_nh4_threshold():
    outputs=[]
    for n in [.0999,.1001]:
        sim=toy(dt=.025,nh4=n)
        initial=sim.state.species['NS21'].biomass
        sim.step({}, {},dynamic_kla=5.)
        state=sim.state.species['NS21']
        assert state.biomass>initial
        assert state.pha_accumulated>0
        outputs.append(state.pha_accumulated)
    assert abs(outputs[1]/outputs[0]-1)<.01


def test_nondivisible_controller_interval_preserves_duration_and_feed():
    sim=toy(dt=.07,internal=.025)
    for _ in range(10):
        sim.step({}, {},dynamic_kla=5.,feed_rates_mmol_l_h={'glc__D_e':.1})
    assert sim.state.time==pytest.approx(.7,abs=1e-12)
    assert sim.cumulative_continuous_feed['glc__D_e']==pytest.approx(.07,abs=1e-12)
    assert sim.current_step==30
    assert sim.dt==.07


def test_bolus_is_not_repeated_per_internal_step_and_invalid_input_does_not_mutate():
    sim=toy()
    sim.step({}, {'helper_lactate':.4},dynamic_kla=0.)
    assert sim.state.metabolites['lac__L_e']==pytest.approx(.4)
    before=sim.state.time
    with pytest.raises(ValueError):sim.step({}, {},feed_rates_mmol_l_h={'nh4_e':-1})
    assert sim.state.time==before


def test_legacy_teacher_or_incompatible_backend_cannot_silently_use_new_dynamics():
    with pytest.raises(ValueError,match='separate'):
        ResolvedDFBASimulator(fba_mode='cooperative')
    with pytest.raises(ValueError,match='highs or glpk'):
        ResolvedDFBASimulator(solver_backend='surrogate')
