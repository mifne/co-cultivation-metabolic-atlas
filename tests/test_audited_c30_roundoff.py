"""The boundary adapter changes only rejected negative roundoff trials."""
from types import SimpleNamespace
import copy
import pytest
from src.audited_dfba import AuditedDFBASimulator
from scripts.analysis.audited_roundoff_adapter_20260908 import install

@pytest.fixture
def adapted(monkeypatch):
    def prepare(self,value):
        self.state.metabolites={'C30_oligo_e':value,'odtd_e':2.}
        self.last_polymer_fluxes={'carbon_c5_equivalent_error_mmol_l':0.}
        self._oxygen_context={}
    def integrate(self):self.integrations+=1
    monkeypatch.setattr(AuditedDFBASimulator,'_prepare_oxygen',prepare)
    monkeypatch.setattr(AuditedDFBASimulator,'_integrate',integrate)
    install()
    return SimpleNamespace(state=SimpleNamespace(metabolites={}),last_polymer_fluxes={},
        _oxygen_context={},accounting_audit={'roundoff_added_mmol_l':{}},integrations=0)

def test_only_committed_roundoff_trial_is_added_to_ledger(adapted):
    sim=adapted
    AuditedDFBASimulator._prepare_oxygen(sim,-5e-20)
    assert sim.state.metabolites['C30_oligo_e']==0.
    assert sim.last_polymer_fluxes['carbon_c5_equivalent_error_mmol_l']==pytest.approx(3e-19,abs=1e-30)
    assert sim.accounting_audit['roundoff_added_mmol_l']=={}
    # Rejecting an exploratory oxygen trial must not add its correction.
    AuditedDFBASimulator._prepare_oxygen(sim,.1)
    AuditedDFBASimulator._integrate(sim)
    assert sim.accounting_audit['roundoff_added_mmol_l']=={}
    AuditedDFBASimulator._prepare_oxygen(sim,-7e-20)
    AuditedDFBASimulator._integrate(sim)
    assert sim.accounting_audit['roundoff_added_mmol_l']['C30_oligo_e']==pytest.approx(7e-20,abs=1e-30)

@pytest.mark.parametrize('value',[0.,.1,1e-20,-1e-10])
def test_nontriggering_values_are_unchanged_and_large_negatives_not_hidden(adapted,value):
    AuditedDFBASimulator._prepare_oxygen(adapted,value)
    assert adapted.state.metabolites=={'C30_oligo_e':value,'odtd_e':2.}
    assert adapted.last_polymer_fluxes['carbon_c5_equivalent_error_mmol_l']==0.
    assert adapted._oxygen_context=={}
