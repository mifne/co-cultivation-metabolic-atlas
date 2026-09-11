"""Analytic CPU LP examples and optional CUDA parity; no reference optimizer."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.lp_direct_dual_audit import audit_direct_dual,prepare_direct_dual_audit


def box_problem(lo=(-2.,1.),hi=(3.,4.),c=(2.,-3.)):
    n=len(c)
    return (csr_matrix((0,n)),np.empty(0),np.array(lo),np.array(hi),np.array(c),0)


def mixed_problem():
    return (csr_matrix([[1.,1.],[1.,0.]]),np.array([3.,1.]),np.zeros(2),
            np.full(2,10.),np.ones(2),1)


def test_box_dual_objective_includes_lower_positive_and_upper_negative_costs():
    prepared=prepare_direct_dual_audit(box_problem())
    x=np.array([[-2.,4.],[0.,2.]])
    result=prepared.evaluate(x,np.empty((2,0)))
    np.testing.assert_allclose(result['dual_objective'],[-16.,-16.])
    np.testing.assert_allclose(result['primal_objective'],[-16.,-6.])
    np.testing.assert_allclose(result['signed_gap'],[0.,10.])
    np.testing.assert_allclose(result['relative_signed_gap'],[0.,10./16.])
    assert result['dual_feasible'].all() and result['finite_arithmetic'].all()
    np.testing.assert_array_equal(result['dual_stationarity_residual'],0.)
    assert 'passed' not in result and 'certificate_passed' not in result


def test_equality_and_inequality_dual_sign_convention_and_gap_decomposition():
    problem=mixed_problem()
    x=np.array([[1.,2.],[1.,2.],[1.,3.]])
    y=np.array([[1.,0.],[0.,-1.],[1.,0.]])
    result=audit_direct_dual(problem,x,y)
    np.testing.assert_allclose(result['dual_objective'],[3.,-1.,3.])
    np.testing.assert_allclose(result['signed_gap'],[0.,4.,1.])
    np.testing.assert_allclose(result['decomposition_gap'],result['signed_gap'])
    np.testing.assert_allclose(result['equality_dual_residual_contribution'],[0.,0.,1.])
    np.testing.assert_allclose(result['equality_primal_residual'],[0.,0.,1.])
    assert result['dual_feasible'].all()


def test_invalid_inequality_dual_is_not_mislabeled_as_a_lower_bound():
    result=audit_direct_dual(mixed_problem(),np.array([1.,2.]),np.array([1.,.5]))
    assert np.isfinite(result['dual_objective'][0])
    assert not result['dual_feasible'][0]
    assert result['dual_lower_bound'][0]==-np.inf
    assert result['dual_sign_violation'][0]==.5


def test_unbounded_variable_domains_handle_zero_times_infinity_without_nan():
    problem=box_problem(lo=(-np.inf,0.,-np.inf,-3.),hi=(5.,np.inf,np.inf,np.inf),
                        c=(-2.,3.,0.,0.))
    result=audit_direct_dual(problem,np.array([5.,0.,17.,6.]),np.empty(0))
    assert result['dual_objective'][0]==-10.
    assert result['signed_gap'][0]==0.
    assert result['dual_feasible'][0] and result['finite_arithmetic'][0]
    assert not any(np.isnan(v).any() for v in result.values())


@pytest.mark.parametrize('lo,hi,c',[
    ((-np.inf,),(4.,),(2.,)),
    ((0.,),(np.inf,),(-2.,)),
    ((-np.inf,),(np.inf,),(2.,)),
])
def test_missing_needed_bound_gives_negative_infinite_dual_and_invalid_domain(lo,hi,c):
    result=audit_direct_dual(box_problem(lo,hi,c),np.zeros(1),np.empty(0))
    assert result['dual_objective'][0]==-np.inf
    assert result['dual_lower_bound'][0]==-np.inf
    assert result['signed_gap'][0]==np.inf
    assert result['relative_signed_gap'][0]==np.inf
    assert result['dual_bound_domain_violation'][0]==2.
    assert result['dual_stationarity_residual'][0]==2.
    assert not result['dual_feasible'][0] and not result['finite_arithmetic'][0]


def test_fixed_bound_supports_either_reduced_cost_sign():
    result=audit_direct_dual(box_problem((2.,-1.),(2.,-1.),(3.,-4.)),
                             np.array([2.,-1.]),np.empty(0))
    assert result['dual_objective'][0]==10.
    assert result['dual_feasible'][0] and result['signed_gap'][0]==0.


def test_large_equality_dual_exposes_gap_omitted_by_complementarity_only_sum():
    # All bound/inequality complementarity terms are zero. Yet y_eq times
    # a 1e-6 primal equality residual contributes 100 to the direct gap.
    problem=(csr_matrix([[1.]]),np.array([0.]),np.array([-1.]),np.array([1.]),
             np.array([1e8]),1)
    result=audit_direct_dual(problem,np.array([1e-6]),np.array([1e8]))
    assert result['equality_primal_residual'][0]==1e-6
    assert result['bound_gap_contribution'][0]==0.
    assert result['inequality_dual_residual_contribution'][0]==0.
    assert result['equality_dual_residual_contribution'][0]==100.
    assert result['signed_gap'][0]==100.
    assert result['relative_signed_gap'][0]==1.


def test_negative_direct_gap_preserves_sign_and_reports_weak_duality_violation():
    problem=(csr_matrix([[1.]]),np.array([1.]),np.array([0.]),np.array([2.]),np.array([1.]),1)
    result=audit_direct_dual(problem,np.array([0.]),np.array([1.]))
    assert result['signed_gap'][0]==-1.
    assert result['relative_signed_gap'][0]==-1.
    assert result['relative_absolute_gap'][0]==1.
    assert result['weak_duality_violation'][0]==1.
    assert result['equality_dual_residual_contribution'][0]==-1.


def test_invalid_pair_lanes_fail_closed_without_changing_valid_lanes():
    x=np.array([[-2.,4.],[np.nan,4.],[-2.,4.]])
    y=np.empty((3,0))
    result=audit_direct_dual(box_problem(),x,y)
    assert result['finite_inputs'].tolist()==[True,False,True]
    assert result['finite_arithmetic'].tolist()==[True,False,True]
    assert result['signed_gap'][1]==np.inf
    assert result['primal_residual'][1]==np.inf
    assert result['signed_gap'][0]==result['signed_gap'][2]==0.
    result=audit_direct_dual(mixed_problem(),np.array([[1.,2.],[1.,2.]]),
                             np.array([[1.,np.inf],[1.,0.]]))
    assert not result['dual_feasible'][0] and result['dual_feasible'][1]
    assert result['relative_signed_gap'][0]==np.inf


def test_objective_overflow_is_reported_not_accepted_as_finite_gap():
    problem=box_problem((0.,0.),(2.,2.),(1e308,1e308))
    result=audit_direct_dual(problem,np.array([2.,2.]),np.empty(0))
    assert not result['finite_arithmetic'][0]
    assert result['relative_signed_gap'][0]==np.inf


def test_snapshot_and_candidate_inputs_are_not_mutated():
    problem=mixed_problem()
    prepared=prepare_direct_dual_audit(problem)
    x,y=np.array([1.,2.]),np.array([1.,0.])
    saved_x,saved_y=x.copy(),y.copy()
    problem[0].data[:]=9.
    problem[1][:]=9.
    result=prepared.evaluate(x,y)
    assert result['signed_gap'][0]==0.
    np.testing.assert_array_equal(x,saved_x)
    np.testing.assert_array_equal(y,saved_y)


@pytest.mark.parametrize('replacement,index',[
    (np.array([np.nan,1.]),4),
    (np.array([np.inf,1.]),1),
    (np.array([np.inf,0.]),2),
    (np.array([-np.inf,2.]),3),
    (np.array([11.,0.]),2),
    (3,5),(True,5),(np.ones(3),4),
])
def test_invalid_problem_is_rejected(replacement,index):
    problem=list(mixed_problem())
    problem[index]=replacement
    with pytest.raises(ValueError):prepare_direct_dual_audit(problem)


def test_invalid_candidate_shapes_and_dtypes_are_rejected():
    prepared=prepare_direct_dual_audit(mixed_problem())
    for x,y in ((np.ones(3),np.ones(2)),(np.ones(2),np.ones(3)),
                (np.ones((2,2)),np.ones(2)),(np.ones(2,dtype=np.float32),np.ones(2))):
        with pytest.raises(ValueError):prepared.evaluate(x,y)


def test_cuda_direct_dual_audit_matches_numpy_and_returns_device_arrays():
    import cupy as cp
    problem=mixed_problem()
    x=np.array([[1.,2.],[1.,3.],[0.,2.]])
    y=np.array([[1.,0.],[1.,0.],[0.,-1.]])
    expected=audit_direct_dual(problem,x,y)
    prepared=prepare_direct_dual_audit(problem,xp=cp)
    result=prepared.evaluate(cp.asarray(x),cp.asarray(y))
    for key,value in result.items():
        assert isinstance(value,cp.ndarray)
        np.testing.assert_allclose(value.get(),expected[key],atol=1e-12,rtol=1e-12)
    with cp.cuda.Stream():
        with pytest.raises(ValueError,match='device and stream'):
            prepared.evaluate(cp.asarray(x),cp.asarray(y))


def test_cuda_direct_dual_unbounded_domains_and_invalid_lanes_are_fail_closed():
    import cupy as cp
    problem=box_problem((-np.inf,),(np.inf,),(1.,))
    result=audit_direct_dual(problem,cp.array([[0.],[cp.nan]]),cp.empty((2,0)),xp=cp)
    assert result['dual_objective'].get()[0]==-np.inf
    assert np.isinf(result['relative_signed_gap'].get()).all()
    assert not result['dual_feasible'].get().any()
