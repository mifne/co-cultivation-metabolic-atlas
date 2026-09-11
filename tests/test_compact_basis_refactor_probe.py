import copy
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_certified_basis import NormalizedLP
from scripts.probe_compact_basis_refactor import refactor_same_basis


def fixture():
    original=(csr_matrix([[1.,1.]]),np.array([2.]),np.zeros(2),np.full(2,np.inf),np.array([1.,2.]),1)
    a,rhs,lo,hi,c,neq=original
    cs=np.array([3.,5.]);rs=np.array([7.])
    scaled=NormalizedLP(csr_matrix(a.toarray()*rs[:,None]/cs),rhs*rs,lo*cs,hi*cs,c/cs,neq,cs,rs)
    basis=dict(basic=np.array([0]),active=np.array([0]),kind=np.array([2,-1]))
    return original,scaled,basis


def test_invertible_coordinates_and_full_original_certificate_without_lp_optimizer():
    original,scaled,basis=fixture();before=copy.deepcopy(basis)
    result=refactor_same_basis(original,scaled,basis)
    assert result['success'] and result['status']=='certified'
    assert result['lp_optimizer_calls']==result['basis_changes']==0
    assert not result['reference_vectors_read']
    assert result['certificates'][0]['relative_kkt_gap']<1e-12
    assert result['maximum_absolute_original_x']==pytest.approx(2.)
    for key in basis:np.testing.assert_array_equal(before[key],basis[key])


def test_feasible_nonoptimal_basis_is_not_rescued_by_refinement():
    original,scaled,basis=fixture()
    basis.update(basic=np.array([1]),kind=np.array([-1,2]))
    result=refactor_same_basis(original,scaled,basis)
    assert not result['success'] and len(result['certificates'])==3
    assert all(r['primal_residual']<1e-12 and
               (r['dual_violation']>1e-7 or r['relative_kkt_gap']>1e-7)
               for r in result['certificates'])


@pytest.mark.parametrize('key,value',[
    ('basic',np.array([0,0])),('basic',np.array([2])),('basic',np.array([0.])),
    ('active',np.array([-1])),('kind',np.array([2,2])),('kind',np.array([2,9])),
])
def test_malformed_basis_fails_before_factorization(key,value):
    original,scaled,basis=fixture();basis[key]=value
    with pytest.raises(ValueError):refactor_same_basis(original,scaled,basis)


def test_singular_current_basis_is_reported_not_accepted():
    original,scaled,basis=fixture();scaled.a=csr_matrix([[0.,1.]])
    result=refactor_same_basis(original,scaled,basis)
    assert not result['success'] and result['status']=='singular_or_invalid_basis'
