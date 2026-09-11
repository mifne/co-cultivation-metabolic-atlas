"""A numerically overestimated primary must not make selection falsely infeasible."""
import copy
import pytest
import src.audited_dfba as module
from tests.test_audited_dfba import make_sim, _snapshot_lp_call, _assert_identical_lp
from scripts.analysis.audited_precision_adapter_20260908 import install


def test_recompute_primary_without_relaxing_secondary_or_mass_balance(monkeypatch):
    sim=make_sim(flux_selection='parsimonious_exchange')
    original=module.linprog
    original_method=module.AuditedDFBASimulator._solve_lp
    calls=[]
    count=len(sim.models['A'].reactions)
    growth=list(r.id for r in sim.models['A'].reactions).index('Growth')
    def solver(c,**kwargs):
        calls.append(_snapshot_lp_call(c,kwargs))
        result=original(c,**kwargs)
        if len(calls)==1:
            result=copy.deepcopy(result);result.x[growth]+=2e-8
        return result
    monkeypatch.setattr(module,'linprog',solver)
    monkeypatch.setattr(module.AuditedDFBASimulator,'_solve_lp',original_method)
    install()
    solution=sim._solve_lp('A',{'Growth':1.})
    assert solution is not None
    assert solution.fluxes['Growth']==pytest.approx(.5,abs=1.1e-9)
    primaries=[call for call in calls if len(call['c'])==count]
    assert len(primaries)==2
    _assert_identical_lp(*primaries)
    assert sim.accounting_audit['primary_recomputations'][0]['history'][0]['recovered']
    acc=sim.accounting_audit
    assert acc['superseded_uncertified_requests']==1
    assert acc['lp_requests']==acc['lp_certified_requests']+1
    assert acc['max_lp_residual']<=1e-7
    assert module.linprog is solver


def test_successful_primary_never_recomputed(monkeypatch):
    sim=make_sim(flux_selection='parsimonious_exchange')
    monkeypatch.setattr(module.AuditedDFBASimulator,'_solve_lp',module.AuditedDFBASimulator._solve_lp)
    install()
    assert sim._solve_lp('A',{'Growth':1.}) is not None
    assert 'primary_recomputations' not in sim.accounting_audit
    assert sim.accounting_audit['lp_requests']==2
