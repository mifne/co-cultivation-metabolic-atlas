import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.lp_reserve_proposal import reserve_proposal
from src.lp_trace import problem_hash


def problem():
    return (csr_matrix([[1.,-1.,0.,0.],[0.,1.,1.,0.],[0.,0.,0.,1.]]),
        np.array([0.,12.,2.]),np.array([-10.,0.,3.,0.]),
        np.array([10.,20.,4.,1.]),np.array([0.,0.,0.,-1.]),1)


@pytest.mark.parametrize('fraction',[.01,.5,1.])
def test_reserve_is_strict_subset_and_keeps_objective_equalities(fraction):
    p=problem();digest=problem_hash(p)
    q=reserve_proposal(p,fraction)
    assert problem_hash(p)==digest
    assert np.all(q[2]>=p[2]) and np.all(q[3]<=p[3])
    assert np.all(q[1][p[-1]:]<=p[1][p[-1]:])
    np.testing.assert_array_equal(q[1][:p[-1]],p[1][:p[-1]])
    np.testing.assert_array_equal(q[4],p[4])
    assert q[2][-1]==p[2][-1] and q[3][-1]==p[3][-1]
    # The interval [3,4] may not be narrowed into an impossible interval.
    assert np.all(q[2]<q[3])
    q[0].data[:]=999.;q[4][:]=999.
    assert problem_hash(p)==digest


@pytest.mark.parametrize('fraction',[0.,-1.,1.1,np.nan,np.inf,True])
def test_invalid_reserve_rejected(fraction):
    with pytest.raises(ValueError):reserve_proposal(problem(),fraction)


@pytest.mark.parametrize('x,rhs,passed',[(1.,2.,True),(.5,2.,False),(1.,.9,False),(np.nan,2.,False)])
def test_cuda_current_original_gate_is_not_a_cache_acceptance(x,rhs,passed):
    import cupy as cp
    from scripts.probe_gpu_reserve_reoptimization import current_certificate
    p=(csr_matrix([[1.]]),np.array([rhs]),np.array([0.]),np.array([1.]),np.array([-1.]),0)
    verification,_=current_certificate([p],cp.asarray([[x]]),cp)
    assert verification['qualified'] is passed
