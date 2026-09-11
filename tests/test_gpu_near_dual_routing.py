"""Eligibility is based on one candidate's certificates, never variance."""
import numpy as np
import pytest

from src.gpu_hybrid_lp import near_dual_maxmin_rows


def sample():
    return dict(input_family_valid=np.ones(5,dtype=bool),primal_residual=np.full(5,.01),
                dual_violation=np.zeros(5),relative_kkt_gap=np.zeros(5),
                basis_dual_violation=np.zeros(5),primal_violation_count=np.array([1,2,3,0,1]))


def test_near_dual_repair_limits_stage_violation_count_and_existing_acceptance():
    candidate=sample()
    np.testing.assert_array_equal(near_dual_maxmin_rows('maxmin',[False]*4+[True],candidate),[True,True,False,False,False])
    assert not near_dual_maxmin_rows('aggregate',[False]*5,candidate).any()
    assert not near_dual_maxmin_rows('exchange',[False]*5,candidate).any()


@pytest.mark.parametrize('field,value',[
    ('input_family_valid',False),('primal_residual',np.inf),('primal_residual',1e-6),
    ('dual_violation',1e-6),('relative_kkt_gap',1.),('basis_dual_violation',1e-7),
    ('primal_violation_count',np.nan),('primal_violation_count',1.5)])
def test_every_candidate_gate_must_hold_on_the_same_row(field,value):
    candidate=sample()
    candidate[field]=candidate[field].astype(bool if field=='input_family_valid' else float)
    candidate[field][0]=value
    assert not near_dual_maxmin_rows('maxmin',[False]*5,candidate)[0]
    assert near_dual_maxmin_rows('maxmin',[False]*5,candidate)[1]


@pytest.mark.parametrize('missing',[None,'wrong_shape','wrong_type'])
def test_unavailable_or_malformed_diagnostics_fail_closed(missing):
    candidate=sample()
    candidate['basis_dual_violation']=None if missing is None else np.array([0.]) if missing=='wrong_shape' else np.array(['0']*5)
    assert not near_dual_maxmin_rows('maxmin',[False]*5,candidate).any()
