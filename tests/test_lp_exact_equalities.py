"""CPU algebra only: numerical candidates never authorize unproved deletion."""
from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import src.lp_exact_equalities as module
from src.lp_exact_equalities import ExactEqualityReduction,equality_fingerprint
from src.lp_trace import problem_hash


def problem(a,rhs=None,neq=None):
    a=csr_matrix(a,dtype=np.float64)
    rhs=np.zeros(a.shape[0]) if rhs is None else np.array(rhs,dtype=np.float64)
    return a,rhs,np.full(a.shape[1],-2.),np.full(a.shape[1],3.),np.arange(a.shape[1],dtype=float),a.shape[0] if neq is None else neq


def assert_exact_stored_map(plan):
    original=plan.original[0].toarray()
    reduced=plan.reduced[0].toarray()
    compression=plan.dual_compression.toarray()
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            reconstructed=sum((Fraction.from_float(float(reduced[k,j]))*
                Fraction.from_float(float(compression[k,i])) for k in range(len(reduced))),Fraction(0))
            assert reconstructed==Fraction.from_float(float(original[i,j]))
        reconstructed_rhs=sum((Fraction.from_float(float(plan.reduced[1][k]))*
            Fraction.from_float(float(compression[k,i])) for k in range(len(reduced))),Fraction(0))
        assert reconstructed_rhs==Fraction.from_float(float(plan.original[1][i]))


def test_exact_integer_and_dyadic_dependencies_and_dual_maps():
    p=problem([[1.,1.,1.,0.],[0.,1.,-1.,1.],[1.,3.,-1.,2.]], [3.,1.,5.])
    plan=ExactEqualityReduction(p)
    assert len(plan.removed_rows)==1
    assert plan.reduced[-1]==2
    assert plan.summary['rows_removed_by_numerical_tolerance_alone']==0
    assert_exact_stored_map(plan)
    original_y=np.array([1.2,-3.,.7])
    reduced_y=plan.compress_dual(original_y)
    np.testing.assert_allclose(p[0].T@original_y,plan.reduced[0].T@reduced_y,atol=2e-14)
    lifted=plan.lift_dual(reduced_y)
    np.testing.assert_array_equal(lifted[plan.removed_rows],0.)
    np.testing.assert_allclose(p[0].T@lifted,p[0].T@original_y,atol=2e-14)
    for proof in plan.proofs:
        assert set(proof.basis_rows).isdisjoint(plan.removed_rows)
        assert all(isinstance(c,Fraction) for c in proof.coefficients)
    dyadic=ExactEqualityReduction(problem([[1.,0.],[0.,1.],[.5,.25]]))
    assert len(dyadic.removed_rows)==1
    assert_exact_stored_map(dyadic)


@pytest.mark.parametrize('tiny',[1e-15,1e-200,1e-300])
def test_near_dependency_with_any_nonzero_coefficient_is_retained(tiny):
    plan=ExactEqualityReduction(problem([[1.,1.,0.],[1.,1.,tiny]]))
    assert len(plan.removed_rows)==0
    assert plan.summary['numerical_candidate_count']==1
    assert plan.reduced[0][1,2]==tiny
    assert plan.summary['matrix_identity_rejections']>0


def test_inconsistent_rhs_is_retained_even_with_exact_matrix_dependency():
    plan=ExactEqualityReduction(problem([[1.,0.],[2.,0.]], [1.,2.+1e-14]))
    assert len(plan.removed_rows)==0
    assert plan.summary['rhs_identity_rejections']>0
    np.testing.assert_array_equal(plan.reduced[1],[1.,2.+1e-14])


def test_zero_equalities_only_removed_when_rhs_exactly_zero_and_inequalities_kept():
    p=problem([[0.,0.],[0.,0.],[1.,0.],[0.,0.]], [0.,1e-300,2.,3.],neq=3)
    plan=ExactEqualityReduction(p)
    np.testing.assert_array_equal(plan.removed_rows,[0])
    np.testing.assert_array_equal(plan.rows,[1,2,3])
    assert plan.proofs[0].kind=='exact_zero_equality'
    assert plan.reduced[-1]==2
    assert_exact_stored_map(plan)


def test_inequality_order_bounds_and_objective_are_unchanged():
    p=problem([[1.,1.],[2.,2.],[1.,0.],[2.,2.],[0.,0.]], [1.,2.,2.,5.,7.],neq=2)
    plan=ExactEqualityReduction(p)
    assert len(plan.removed_rows)==1
    assert np.all(np.diff(plan.rows)>0)
    np.testing.assert_array_equal(plan.rows[-3:],[2,3,4])
    np.testing.assert_array_equal(plan.reduced[0][plan.reduced[-1]:].toarray(),p[0][2:].toarray())
    np.testing.assert_array_equal(plan.reduced[1][plan.reduced[-1]:],p[1][2:])
    for i in (2,3,4):np.testing.assert_array_equal(plan.reduced[i],p[i])
    assert_exact_stored_map(plan)


def test_non_dyadic_dependency_is_conservatively_retained_for_exact_sparse_map():
    # Equal normalized rows make original row 0 the stable first QR pivot.
    plan=ExactEqualityReduction(problem([[3.],[1.]]))
    assert len(plan.removed_rows)==0
    assert plan.summary['nonrepresentable_weight_rejections']>0


def test_no_equalities_and_all_zero_equalities_do_not_need_qr(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('QR should not be called')
    monkeypatch.setattr(module,'qr',forbidden)
    for p,removed in ((problem([[1.,2.]],neq=0),0),
                      (problem([[0.,0.],[0.,0.]]),2)):
        plan=ExactEqualityReduction(p)
        assert len(plan.removed_rows)==removed
        assert plan.summary['qr_seconds']==0.
        assert_exact_stored_map(plan)


def test_source_is_not_mutated_and_owned_snapshots_and_maps_are_readonly():
    p=problem([[1.,0.],[2.,0.]], [1.,2.])
    before=problem_hash(p)
    plan=ExactEqualityReduction(p)
    assert problem_hash(p)==before
    for array in (*plan.original[1:5],*plan.reduced[1:5],plan.rows,plan.removed_rows,
                  plan.original[0].data,plan.reduced[0].data,plan.dual_compression.data):
        assert not array.flags.writeable
        if array.size:
            with pytest.raises(ValueError):array.flat[0]=99.
    with pytest.raises(AttributeError):plan.rows=np.array([1])
    reported=plan.summary
    reported['removed_rows']=999
    assert plan.summary['removed_rows']==1
    p[0].data[:]=8.
    p[1][:]=8.
    assert plan.original_hash==problem_hash(plan.original)==before
    plan.validate_integrity()


@pytest.mark.parametrize('target',['original','reduced','compression','rows'])
def test_deliberate_readonly_override_is_detected(target):
    plan=ExactEqualityReduction(problem([[1.,0.],[2.,0.]], [1.,2.]))
    array=(plan.original[1] if target=='original' else plan.reduced[1] if target=='reduced'
           else plan.dual_compression.data if target=='compression' else plan.rows)
    array.flags.writeable=True
    array[0]+=1
    with pytest.raises(ValueError,match='modified'):plan.validate_integrity()


def test_reuse_skips_qr_and_copies_current_non_equality_inputs(monkeypatch):
    p=problem([[1.,0.],[2.,0.],[0.,1.]], [1.,2.,3.],neq=2)
    first=ExactEqualityReduction(p)
    changed=problem([[1.,0.],[2.,0.],[1.,1.],[0.,1.]], [1.,2.,9.,8.],neq=2)
    changed[2][:]=-5.
    changed[3][:]=20.
    changed[4][:]=4.
    assert equality_fingerprint(changed)==first.equality_fingerprint
    def forbidden(*args,**kwargs):raise AssertionError('Matching proof reuse must skip QR')
    monkeypatch.setattr(module,'qr',forbidden)
    second=ExactEqualityReduction(changed,reuse_from=first)
    assert second.summary['reused_proofs'] is True
    assert second.summary['qr_seconds']==0.
    assert second.summary['proof_seconds']==0.
    assert second.proofs is first.proofs or second.proofs==first.proofs
    assert second.proof_fingerprint==first.proof_fingerprint
    assert second.dual_compression.shape==(3,4)
    for i in (2,3,4):np.testing.assert_array_equal(second.reduced[i],changed[i])
    np.testing.assert_array_equal(second.reduced[1][-2:],[9.,8.])
    assert_exact_stored_map(second)


@pytest.mark.parametrize('change',['coefficient','rhs','columns','neq'])
def test_reuse_rejects_mismatched_equalities(change):
    p=problem([[1.,0.],[2.,0.]], [1.,2.])
    first=ExactEqualityReduction(p)
    if change=='coefficient':p[0].data[0]+=.1
    elif change=='rhs':p[1][0]+=1e-15
    elif change=='columns':p=problem([[1.,0.,0.],[2.,0.,0.]], [1.,2.])
    else:p=(*p[:-1],1)
    with pytest.raises(ValueError,match='fingerprint'):
        ExactEqualityReduction(p,reuse_from=first)


@pytest.mark.parametrize('kwargs',[dict(max_dimension=1),dict(max_memory_mb=1e-9),
    dict(max_candidates=0),dict(max_support=0),dict(max_denominator=True),
    dict(max_dimension=6001),dict(qr_threshold=0.)])
def test_configuration_dimension_and_memory_guards_fail_before_qr(kwargs,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Guard should run before dense QR')
    monkeypatch.setattr(module,'qr',forbidden)
    with pytest.raises(ValueError):
        ExactEqualityReduction(problem([[1.,0.],[2.,0.]]),**kwargs)


def test_normalization_cannot_silently_erase_extreme_nonzero_coefficient():
    with pytest.raises(ValueError,match='loses nonzero'):
        ExactEqualityReduction(problem([[1e300,1e-300]]))


@pytest.mark.parametrize('bad',[np.array([1.]),np.array([np.inf,0.]),np.ones((2,1))])
def test_dual_vector_validation(bad):
    plan=ExactEqualityReduction(problem([[1.,0.],[2.,0.]]))
    with pytest.raises(ValueError,match='Finite vector'):plan.compress_dual(bad)
