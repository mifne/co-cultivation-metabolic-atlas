import numpy as np
import pytest
from src.gpu_centrality_corrector import centrality_rhs,composite_weight


def test_centrality_changes_only_outlier_targets_and_preserves_inputs():
    s=np.ones((1,2));z=s.copy();ds=np.array([[-2.,0.]]);dz=np.zeros((1,2));rc=s.copy()
    proposed,valid,_=centrality_rhs(s,z,ds,dz,np.array([.2]),np.array([.49]),rc,
        np.array([True]),xp=np)
    assert valid.all()
    np.testing.assert_allclose(proposed,[[-.02,1.]])
    np.testing.assert_array_equal(rc,[[1.,1.]])


def test_inactive_nonfinite_correction_is_masked_before_products():
    s=np.array([[1.,1.],[np.nan,np.inf]])
    with np.errstate(all='raise'):
        proposed,valid,_=centrality_rhs(s,s,s,s,np.array([1.,np.nan]),np.array([.5,np.nan]),s,
            np.array([True,False]),xp=np)
    assert not valid[1] and np.isfinite(proposed).all() and (proposed[1]==0.).all()


def test_composite_weight_is_lane_independent_and_ignores_invalid_extra():
    s=np.ones((2,2));ds=np.array([[-2.,0.],[-2.,0.]]);dz=np.zeros((2,2))
    new=np.array([[-1.,0.],[np.nan,np.nan]])
    weights=composite_weight(s,s,ds,dz,new,dz,np.array([True,False]),xp=np)
    np.testing.assert_array_equal(weights,[1.,0.])


@pytest.mark.parametrize('budget',[1,2,4])
def test_cuda_multiple_corrections_keep_original_certificate(budget):
    from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
    from tests.test_gpu_dual_schur import inputs,options
    opts=options();opts['newton_backend']='augmented';opts['centrality_corrections']=budget
    with GpuCondensedBatchedIPM(inputs(),**opts) as s:
        r=s.solve(iterations=100)
        assert r['accepted'].all() and r['centrality_corrections']==budget
        v=np.asarray(r['centrality_diagnostics'])
        if v.size:
            chosen=v[:,:,6].astype(bool)
            assert np.all(v[:,:,7][chosen]<=s.forcing_eta)
            assert np.all(v[:,:,5][chosen]>=v[:,:,4][chosen]+.01)
