"""An auxiliary success is never an original-LP optimality certificate."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scripts.probe_ipm_box_face_sequence import qualify_original
from src.gpu_block_lp import box_face_feasibility


@pytest.mark.parametrize('x,rhs,accepted,qualified',[
    (1.,2.,True,True),(.5,2.,True,False),(1.,.8,True,False),
    (1.,2.,False,False),(float('nan'),2.,True,False)])
def test_cuda_original_gate_checks_feasibility_optimality_and_auxiliary_acceptance(x,rhs,accepted,qualified):
    import cupy as cp
    p=(csr_matrix([[1.]]),np.array([rhs]),np.array([0.]),
       np.array([1.]),np.array([-1.]),0)
    result=dict(x=cp.asarray([[x]]),y=cp.asarray([[-999.]]),accepted=cp.asarray([accepted]))
    before=result['y'].copy()
    outcome=qualify_original([p],result,cp)
    assert outcome['qualified'] is qualified
    assert bool(cp.array_equal(result['y'],before))


def test_auxiliary_does_not_mutate_original_objective_or_bounds():
    p=(csr_matrix([[1.]]),np.array([.8]),np.array([0.]),np.array([1.]),np.array([-1.]),0)
    auxiliary=box_face_feasibility(p)
    assert auxiliary[2][0]==auxiliary[3][0]==1. and auxiliary[4][0]==0.
    assert p[2][0]==0. and p[3][0]==1. and p[4][0]==-1.
