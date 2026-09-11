from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_block_lp import assemble_blocks,certify_blocks_device
from src.gpu_ray_reoptimization import ray_proposal,ComponentRescaling


def workspace(upper=2.):
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as device_csr
    problem=(csr_matrix([[1.,-1.,0.],[-1.,0.,1.]]),np.zeros(2),np.zeros(3),
        np.array([upper,upper,1.]),np.array([0.,0.,-1.]),1)
    packed=assemble_blocks([problem])
    s=SimpleNamespace(cp=cp,batch=1,full_n=3,full_m=2,full_neq=1,
        factor=SimpleNamespace(_context=lambda:None,device=cp.cuda.Device().id),
        _full_assembled=(device_csr(packed[0]),*(cp.asarray(v) for v in packed[1:])))
    return s,problem


def test_gpu_ray_current_bounds_and_original_certificate():
    import cupy as cp
    s,p=workspace();previous=cp.array([[4.,4.,1.]])
    x,info=ray_proposal(s,previous)
    assert info['valid_interval']==[True]
    assert .25<info['alpha'][0]<.5
    metrics=certify_blocks_device([p],s._full_assembled,x.ravel(),cp.zeros(2),cp=cp,
        allow_box_dual=False,require_direct_dual=True)
    assert metrics[0]['certificate_passed']
    cp.testing.assert_array_equal(previous,cp.array([[4.,4.,1.]]))
    assert p[3][0]==2.


def test_empty_ray_does_not_claim_lp_infeasible():
    import cupy as cp
    s,p=workspace(.5)
    _,info=ray_proposal(s,cp.array([[4.,4.,1.]]))
    assert info['valid_interval']==[False]
    assert info['empty_interval_is_not_LP_infeasibility']


def test_bad_prior_and_unsupported_objective_reject():
    import cupy as cp
    s,_=workspace()
    with pytest.raises(ValueError):ray_proposal(s,cp.zeros((1,3),dtype=cp.float32))
    s._full_assembled[-1][0]=1.
    with pytest.raises(ValueError):ray_proposal(s,cp.zeros((1,3)))


def test_component_scaling_is_causal_owned_and_conflicts_are_not_assumed():
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as device_csr
    s,p=workspace()
    a=csr_matrix([[1.,-1.,0.],[-1.,0.,1.],[1.,1.,0.]])
    p=(a,np.array([0.,0.,5.]),p[2],p[3],p[4],1)
    packed=assemble_blocks([p]);s.full_m=3;s.full_problems=(p,)
    s._full_assembled=(device_csr(packed[0]),*(cp.asarray(v) for v in packed[1:]))
    helper=ComponentRescaling(s);prior=cp.array([[4.,4.,1.]])
    with pytest.raises(ValueError):helper.propose(prior)
    helper.snapshot();saved=helper.previous.copy()
    s._full_assembled[0].data[-2:]*=2
    proposal,info=helper.propose(prior)
    cp.testing.assert_allclose(proposal,cp.array([[2.,2.,1.]]))
    assert info['scaled_components']==1
    cp.testing.assert_array_equal(saved,helper.previous)
    cp.testing.assert_array_equal(prior,cp.array([[4.,4.,1.]]))
    s._full_assembled[0].data[-1]*=2
    proposal,info=helper.propose(prior)
    assert info['conflicting_components']==1
    cp.testing.assert_array_equal(proposal,prior)
