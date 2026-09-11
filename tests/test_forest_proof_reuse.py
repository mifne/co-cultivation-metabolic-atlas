import numpy as np
import pytest
from scipy.sparse import csr_matrix
from unittest.mock import patch
from src.lp_equality_reduction import HomogeneousEqualityReduction,prepare_independent_forest_plans


def problem(scale=1.,rhs=0.,bound=2.):
    return (csr_matrix([[scale,-scale,0.],[0.,1.,1.]]),np.array([rhs,3.]),
        np.zeros(3),np.full(3,bound),np.array([0.,0.,-1.]),1)


def test_identical_proofs_reused_but_lane_bounds_and_objects_independent():
    inputs=[problem(),problem(bound=1.)]
    original=HomogeneousEqualityReduction.from_problem
    with patch.object(HomogeneousEqualityReduction,'from_problem',wraps=original) as make:
        plans=prepare_independent_forest_plans(inputs)
        assert make.call_count==1
    assert plans[0] is not plans[1]
    assert not np.shares_memory(plans[0].T.data,plans[1].T.data)
    assert not np.shares_memory(plans[0].weights,plans[1].weights)
    for plan,p in zip(plans,inputs):
        actual=plan.reduce(p);fresh=original(p).reduce(p)
        assert (actual.problem[0]!=fresh.problem[0]).nnz==0
        for got,expected in zip(actual.problem[1:5],fresh.problem[1:5]):np.testing.assert_array_equal(got,expected)
        np.testing.assert_array_equal(actual.lower_witness,fresh.lower_witness)
        np.testing.assert_array_equal(actual.upper_witness,fresh.upper_witness)


def test_changed_equality_or_homogeneity_never_reuses_proof():
    original=HomogeneousEqualityReduction.from_problem
    with patch.object(HomogeneousEqualityReduction,'from_problem',wraps=original) as make:
        plans=prepare_independent_forest_plans([problem(),problem(scale=2.),problem(rhs=1.)])
        assert make.call_count==3
    assert len(plans[2].eliminated_rows)==0


def test_hash_collision_still_requires_exact_equality_comparison():
    original=HomogeneousEqualityReduction.from_problem
    with patch('src.lp_equality_reduction._equality_fingerprint',return_value='collision'):
        with patch.object(HomogeneousEqualityReduction,'from_problem',wraps=original) as make:
            prepare_independent_forest_plans([problem(),problem(scale=2.)])
            assert make.call_count==2


def test_cuda_within_batch_face_rebind_has_current_owned_numeric_data():
    from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
    from src.lp_zero_face import ZeroFaceReduction
    p=(csr_matrix([[1.,1.,0.],[0.,0.,1.]]),np.array([0.,3.]),
       np.zeros(3),np.full(3,2.),np.array([0.,0.,-1.]),1)
    hi=p[3].copy();hi[2]=1.
    second=(p[0],p[1],p[2],hi,p[4],p[5])
    with ZeroFaceGpuBatchedIPM([p,second],reuse_equality_proofs=True) as s:
        assert s.face_plan_reuse_count==1
        for actual,original in zip(s.face_plans,[p,second]):
            fresh=ZeroFaceReduction(original)
            assert actual.original_hash==fresh.original_hash
            assert actual.reduced_hash==fresh.reduced_hash
            actual.validate_integrity()
        assert not np.shares_memory(s.face_plans[0].original[3],s.face_plans[1].original[3])
    hi[0]=0.
    with ZeroFaceGpuBatchedIPM([p,second],reuse_equality_proofs=True) as s:
        assert s.face_plan_reuse_count==0  # explicit fixed topology changed: freshly proved


def test_cuda_cached_forest_and_uncached_outputs_are_identical():
    from src.gpu_forest_ipm import prepare_forest_batch
    inputs=[problem(),problem(bound=1.)]
    fresh=prepare_forest_batch(inputs)
    cached=prepare_forest_batch(inputs,reuse_equality_proofs=True)
    for one,two in zip(fresh[2],cached[2]):
        assert (one.problem[0]!=two.problem[0]).nnz==0
        for x,y in zip(one.problem[1:5],two.problem[1:5]):np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('corrupt',['map','witness','objective'])
def test_cached_independent_reproof_still_rejects_corrupt_lane(corrupt):
    from src.gpu_forest_map import _prepare_host_maps
    inputs=[problem(),problem(bound=1.)]
    plans=prepare_independent_forest_plans(inputs)
    reductions=[plan.reduce(p) for plan,p in zip(plans,inputs)]
    _prepare_host_maps(plans,reductions,reuse_equality_proofs=True)
    if corrupt=='map':plans[1].T.data[0]+=1.
    elif corrupt=='witness':reductions[1].upper_witness[0]=-99
    else:reductions[1].problem[4][0]+=1.
    with pytest.raises(ValueError):_prepare_host_maps(plans,reductions,reuse_equality_proofs=True)
