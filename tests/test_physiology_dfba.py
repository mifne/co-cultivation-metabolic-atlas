import math
import copy
import cobra
import pytest
from tests.test_audited_dfba import toy_model
from src.physiology_dfba import PhysiologyDFBASimulator


def model():
    m=toy_model(storage=True)
    for mid,formula in {'glc__D_e':'C','nh4_e':'H4N','o2_e':'O2','co2_e':'CO2',
                        'h_e':'H','c_c':'C','n_c':'N','o_c':'O2','pha_c':'C4H6O2'}.items():
        m.metabolites.get_by_id(mid).formula=formula
    atp=cobra.Metabolite('atp_c',formula='C10H12N5O13P3',compartment='c')
    adp=cobra.Metabolite('adp_c',formula='C10H12N5O10P2',compartment='c')
    pi=cobra.Metabolite('pi_c',formula='HO4P',compartment='c')
    water=cobra.Metabolite('h2o_c',formula='H2O',compartment='c')
    energy=cobra.Reaction('ENERGY');energy.add_metabolites({m.metabolites.c_c:-1,adp:-1,atp:1,m.metabolites.co2_e:1,pi:-1,water:1,m.metabolites.h_e:-1})
    maintenance=cobra.Reaction('ATPM');maintenance.add_metabolites({atp:-1,adp:1,pi:1,water:-1,m.metabolites.h_e:1})
    urea=cobra.Metabolite('urea_e',formula='CH4N2O',compartment='e')
    exchange=cobra.Reaction('EX_urea_e');exchange.add_metabolites({urea:-1});exchange.bounds=(-20,1000)
    nitrogen=cobra.Reaction('UREA');nitrogen.add_metabolites({urea:-1,m.metabolites.n_c:2,m.metabolites.co2_e:1})
    reuse=cobra.Reaction('REUSE');reuse.add_metabolites({m.metabolites.pha_c:-1,m.metabolites.c_c:1})
    m.add_reactions([energy,maintenance,exchange,nitrogen,reuse])
    for met in [atp,adp,pi,water,m.metabolites.h_e]:
        met.charge={'atp_c':-4,'adp_c':-3,'pi_c':-2,'h2o_c':0,'h_e':1}[met.id]
    return m


def make(*,name='A',carbon=20.,nh4=20.,urea=0.,dt=.1,internal=.025,**kwargs):
    return PhysiologyDFBASimulator({name:model()},{name:.1},
        {'glc__D_e':carbon,'nh4_e':nh4,'urea_e':urea,'o2_e':.25,'co2_e':0.,
         'mlttr_e':0.,'ptrc_e':0.,'mnl_e':0.,'lac__L_e':0.},
        initial_rubber=0.,dt=dt,max_internal_dt=internal,ph_control_target=7.,**kwargs)


def test_alternative_nitrogen_prevents_false_nitrogen_starvation():
    new=make(name='NS21',nh4=0.,urea=20.,nitrogen_policy='capacity_ratio')
    old=make(name='NS21',nh4=0.,urea=20.,nitrogen_policy='nh4_legacy')
    new.step({},{});old.step({},{})
    assert new.nitrogen_allocation['NS21']['growth_fraction']==pytest.approx(1.,abs=1e-6)
    assert old.nitrogen_allocation['NS21']['growth_fraction']==0.
    assert new.state.species['NS21'].biomass>old.state.species['NS21'].biomass
    assert new.nitrogen_relief_pools['NS21']==['nh4_e']


@pytest.mark.parametrize('capacity',[.2,10.])
def test_maintenance_is_physical_rate_after_density_scaling(capacity):
    sim=make(carrying_capacity=capacity,maintenance={'A':{'reaction':'ATPM','rate_mmol_g_h':.3}})
    sim.step({},{})
    t=sim.physiology_totals['A']
    assert t['maintenance_used_mmol_l']==pytest.approx(t['maintenance_requested_mmol_l'],abs=1e-9)
    assert sim.models['A'].reactions.ATPM.lower_bound==0.
    assert sim._cell_scale*sim.last_fba_solutions['A'].fluxes['ATPM']==pytest.approx(.3,abs=1e-8)


def test_required_gem_maintenance_is_supported_without_mutating_input():
    m=model();m.reactions.ATPM.lower_bound=.3
    sim=PhysiologyDFBASimulator({'A':m},{'A':.1},{'glc__D_e':20.,'nh4_e':20.,'o2_e':.25},
        maintenance={'A':{'reaction':'ATPM','rate_mmol_g_h':.3}},initial_rubber=0.,dt=.025,ph_control_target=7.)
    sim.step({},{})
    assert m.reactions.ATPM.lower_bound==.3
    assert sim.physiology_trials['A']['used']==pytest.approx(.3,abs=1e-8)


@pytest.mark.parametrize('internal',[.1,.025])
def test_starvation_death_is_exact_and_dead_mass_is_retained(internal):
    sim=make(carbon=0.,dt=1.,internal=internal,
        maintenance={'A':{'reaction':'ATPM','rate_mmol_g_h':.3}},starvation_death_rates={'A':.2})
    sim.step({},{})
    live=sim.state.species['A'].biomass;dead=sim.dead_matter['A']['biomass_g_l']
    assert live==pytest.approx(.1*math.exp(-.2),abs=1e-9)
    assert live+dead==pytest.approx(.1,abs=1e-10)
    assert sim.physiology_trials['A']['deficit_fraction']==pytest.approx(1.)
    sim.reset()
    assert sim.dead_matter['A']['biomass_g_l']==0.
    assert sim.state.species['A'].biomass==.1
    assert not sim.element_exchanges['A']['uptake']


def test_dead_storage_is_not_discarded_or_released_into_medium():
    sim=make(carbon=0.,basal_death_rates={'A':.5})
    s=sim.state.species['A'];s.phb_accumulated=1.;s.pha_accumulated=1.
    sim.step({},{})
    assert s.phb_accumulated+sim.dead_matter['A']['phb_mmol_l']==pytest.approx(1.,abs=1e-10)
    assert sim.state.metabolites.get('pha_c',0.)==0.


def test_existing_storage_degradation_can_pay_maintenance_without_external_carbon():
    sim=make(carbon=0.,nh4=0.,remobilize_pha=True,
        maintenance={'A':{'reaction':'ATPM','rate_mmol_g_h':.3}})
    s=sim.state.species['A'];s.phb_accumulated=.01;s.pha_accumulated=.01
    sim.step({},{})
    assert 0<=s.phb_accumulated<.01
    used=sim.physiology_totals['A']['phb_remobilized_mmol_l']
    assert .01-s.phb_accumulated==pytest.approx(used,abs=1e-9)
    assert sim.physiology_trials['A']['used']==pytest.approx(.3,abs=1e-8)


def test_no_pseudo_element_sufficiency_and_feed_ledger_counts_n_atoms():
    sim=make(nh4=0.,urea=0.)
    sim.step({}, {},feed_rates_mmol_l_h={'urea_e':1.})
    ledger=sim.element_accounting()
    assert ledger['delivered_known']['N']==pytest.approx(.2)
    uptake=ledger['species_known']['A']['uptake']['N']
    secretion=ledger['species_known']['A']['secretion']['N']
    assert ledger['current_known']['N']==pytest.approx(.2-uptake+secretion,abs=1e-9)
    assert not ledger['pha_carbon_origin_identified']


def test_solver_failure_is_not_converted_to_starvation(monkeypatch):
    sim=make(maintenance={'A':{'reaction':'ATPM','rate_mmol_g_h':.3}},starvation_death_rates={'A':1.})
    monkeypatch.setattr(sim,'_solve_lp',lambda *a,**k:None)
    with pytest.raises(RuntimeError,match='LP failed'):sim.step({},{})
    assert sim.dead_matter['A']['biomass_g_l']==0.
    assert not sim.get_solver_diagnostics()['audited_cultivation']['valid']


def test_physiology_parameters_change_policy_contract():
    from src.rl_environment import ConsortiumEnv
    sim=make();env=ConsortiumEnv(sim)
    before=copy.deepcopy(env.get_policy_contract())
    sim.basal_death_rates['A']=.1
    assert env.get_policy_contract()!=before
    with pytest.raises(RuntimeError,match='assumptions changed'):sim.step({},{})


def test_dead_material_dilutes_and_sampled_dead_material_is_accounted():
    sim=make()
    sim.dead_matter['A']['biomass_g_l']=.2
    sim.record_volume_addition(1.,1.1,{'urea_e':.1})
    assert sim.dead_matter['A']['biomass_g_l']==pytest.approx(.2/1.1)
    sim.record_sample(.01)
    event=sim.element_accounting()['physical_events'][-1]
    assert event['dead_matter_removed']['A']['biomass_g']==pytest.approx(.2/1.1*.01)


def test_cli_factory_accepts_physiology_and_rejects_wrong_config_mode(tmp_path):
    from main import cultivation_options,make_env
    from types import SimpleNamespace
    path=tmp_path/'physiology.json';path.write_text('{"nitrogen_policy":"capacity_ratio"}')
    options=cultivation_options(SimpleNamespace(dynamics='physiology',physiology_config=str(path)))
    assert options['fba_mode']=='separate'
    env=make_env(None,options,preloaded_models={'NS21':model()})()
    assert isinstance(env.simulator,PhysiologyDFBASimulator)
    with pytest.raises(ValueError):cultivation_options(SimpleNamespace(dynamics='audited',physiology_config=str(path)))


@pytest.mark.parametrize('kw',[{'basal_death_rates':{'A':-1}}, {'starvation_death_rates':{'A':1}},
                             {'maintenance':{'missing':{}}}, {'nitrogen_policy':'total_N'},
                             {'remobilize_pha':'false'}])
def test_invalid_assumptions_rejected(kw):
    with pytest.raises(ValueError):make(**kw)


def test_arbitrary_atp_consuming_reaction_cannot_be_declared_maintenance():
    with pytest.raises(ValueError,match='ATP hydrolysis'):
        make(maintenance={'A':{'reaction':'ENERGY','rate_mmol_g_h':.3}})


def test_b12_annotation_repair_removes_only_pseudogene_support_and_preserves_flux_bounds():
    from src.b12_evidence import inspect_and_curate
    name='Rhizobacter_gummiphilus_NS21'
    m=model();r=cobra.Reaction('METS');r.name='Methionine synthase'
    r.add_metabolites({m.metabolites.c_c:-1});r.bounds=(0.,10.)
    r.gene_reaction_rule='A4W93_27845 or A4W93_24875';m.add_reactions([r])
    curated,report=inspect_and_curate({name:m})
    assert curated[name].reactions.METS.gene_reaction_rule=='A4W93_24875'
    assert m.reactions.METS.gene_reaction_rule=='A4W93_27845 or A4W93_24875'
    assert curated[name].reactions.METS.bounds==(0.,10.)
    assert not report[name]['external_b12_dependency_established']
    assert report[name]['changes'][0]['removed_pseudogene_support']==['A4W93_27845']


def test_new_medium_accounting_closes_with_feed_and_cell_exchange():
    sim=make(nh4=0.,urea=1.)
    sim.step({}, {},feed_rates_mmol_l_h={'urea_e':.2})
    assert sim.element_accounting()['known_medium_closure_residual']==pytest.approx({'C':0.,'N':0.},abs=1e-9)


def test_jar_updates_dead_material_and_medium_concentration_ledger():
    from src.one_l_jar import OneLJarProfile,OneLJarConfig
    sim=make();jar=OneLJarProfile(sim,config=OneLJarConfig(initial_volume_l=.5))
    sim.dead_matter['A']['biomass_g_l']=.2
    jar._add_well_mixed_volume(.05,{'urea_e':.1})
    assert sim.dead_matter['A']['biomass_g_l']==pytest.approx(.2*.5/.55)
    assert sim.element_accounting()['known_medium_closure_residual']==pytest.approx({'C':0.,'N':0.},abs=1e-9)


def test_incomplete_organic_n_formula_is_not_a_nitrogen_relief_feed():
    m=model();pool=cobra.Metabolite('tnt_e',name='2 4 6 Trinitrotoluene',formula='NO2',compartment='e')
    r=cobra.Reaction('EX_tnt_e');r.add_metabolites({pool:-1});r.bounds=(-20,1000);m.add_reactions([r])
    sim=PhysiologyDFBASimulator({'NS21':m},{'NS21':.1},{'nh4_e':1.,'tnt_e':2.},initial_rubber=0.)
    assert sim.nitrogen_relief_pools['NS21']==['nh4_e']
    assert 'tnt_e' not in sim.pool_elements
    assert sim.element_accounting()['unclassified_current']['tnt_e']==2.
