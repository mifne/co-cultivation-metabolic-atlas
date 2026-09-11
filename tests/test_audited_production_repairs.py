"""Exercise the integrated fixes without research adapters or global solver mutation."""
import copy
import numpy as np
import pandas as pd
import pytest
from cobra.core.solution import Solution
import src.audited_dfba as module
from tests.test_audited_dfba import make_sim,_snapshot_lp_call,_assert_identical_lp
from src.dfba_simulator import dFBASimulator


def test_primary_recomputation_is_instance_local_and_certified(monkeypatch):
    sim=make_sim(flux_selection='parsimonious_exchange')
    original=module.linprog;calls=[]
    count=len(sim.models['A'].reactions)
    growth=[r.id for r in sim.models['A'].reactions].index('Growth')
    def solver(c,**kwargs):
        calls.append(_snapshot_lp_call(c,kwargs));result=original(c,**kwargs)
        if len(calls)==1:
            result=copy.deepcopy(result);result.x[growth]+=2e-8
        return result
    monkeypatch.setattr(module,'linprog',solver)
    solution=sim._solve_lp('A',{'Growth':1.})
    assert solution is not None
    assert solution.fluxes['Growth']==pytest.approx(.5,abs=1.1e-9)
    primary=[call for call in calls if len(call['c'])==count]
    assert len(primary)==2;_assert_identical_lp(*primary)
    acc=sim.accounting_audit
    assert acc['lp_requests']==acc['lp_certified_requests']+acc['superseded_uncertified_requests']
    assert acc['superseded_uncertified_requests']==1
    assert acc['primary_recomputations'][0]['history'][0]['recovered']
    assert module.linprog is solver
    assert sim._lp_algorithm_override is None
    sim.reset()
    assert 'primary_recomputations' not in sim.accounting_audit
    assert 'superseded_uncertified_requests' not in sim.accounting_audit


def test_c30_roundoff_is_corrected_before_uptake_and_ledgered_on_commit(monkeypatch):
    sim=make_sim();sim.state.rubber_concentration=1.
    sim._integration_biomass={'A':.1}
    def potential(self,rates):
        self.last_polymer_fluxes=dict(lcp_c5_mmol_l_step=.7,roxb_c5_mmol_l_step=.2,
            roxa_direct_c5_mmol_l_step=0.,roxa_oligo_c30_mmol_l_step=.15000000000000005,
            bulk_substrate_scale=1.)
    monkeypatch.setattr(dFBASimulator,'degrade_rubber',potential)
    sim._prepare_oxygen(50.,{},.25)
    correction=sim._oxygen_context['c30_roundoff_added']
    assert 0<correction<1e-12
    assert sim.state.metabolites['C30_oligo_e']==0.
    assert sim.accounting_audit['roundoff_added_mmol_l']=={}
    zero=Solution(objective_value=0.,status='optimal',fluxes=pd.Series(np.zeros(len(sim.models['A'].reactions)),
        index=[r.id for r in sim.models['A'].reactions]))
    sim._integrate({'A':zero})
    assert sim.accounting_audit['roundoff_added_mmol_l']['C30_oligo_e']==correction


def test_large_c30_defect_is_not_hidden_by_production_roundoff(monkeypatch):
    sim=make_sim();sim.state.rubber_concentration=1.;sim._integration_biomass={'A':.1}
    def potential(self,rates):
        self.last_polymer_fluxes=dict(lcp_c5_mmol_l_step=.7,roxb_c5_mmol_l_step=.2,
            roxa_direct_c5_mmol_l_step=0.,roxa_oligo_c30_mmol_l_step=.1501,bulk_substrate_scale=1.)
    monkeypatch.setattr(dFBASimulator,'degrade_rubber',potential)
    sim._prepare_oxygen(50.,{},.25)
    assert sim.state.metabolites['C30_oligo_e'] < -1e-12
    assert 'c30_roundoff_added' not in sim._oxygen_context
