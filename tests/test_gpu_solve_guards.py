import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_sparse_factor import UniformCudssFactor
from src.gpu_solve_guards import FusedSolveGuards


@pytest.mark.parametrize('nrhs',[1,3])
@pytest.mark.parametrize('layout',['contiguous','strided','negative','broadcast'])
def test_cuda_fused_guards_match_legacy_and_keep_owned_results(nrhs,layout):
    import cupy as cp
    a=csr_matrix([[3.,1.],[1.,2.]])
    with UniformCudssFactor(a,batch_size=3,nrhs=nrhs,refinement_steps=0) as factor:
        factor.factor(cp.tile(cp.asarray(a.data),(3,1)))
        base=cp.arange(3*4*nrhs,dtype=cp.float64).reshape(3,4,nrhs)+1.
        rhs=base[:,:2,:].copy()
        if layout=='strided':rhs=base[:,::2,:]
        elif layout=='negative':rhs=rhs[:,::-1,::-1]
        elif layout=='broadcast':rhs=cp.broadcast_to(rhs[:1],rhs.shape)
        old=factor._solve_device_checked(rhs)
        new=factor._solve_device_checked_fused(rhs)
        np.testing.assert_allclose(new.get(),old.get(),rtol=1e-14,atol=1e-14)
        snapshot=new.copy()
        factor._solve_device_checked_fused(rhs*2.)
        np.testing.assert_array_equal(new.get(),snapshot.get())


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_cuda_invalid_rhs_is_zero_before_native_and_nan_after(bad,monkeypatch):
    import cupy as cp
    a=csr_matrix(np.eye(3))
    with UniformCudssFactor(a,batch_size=2,nrhs=2,refinement_steps=0) as factor:
        factor.factor(cp.ones((2,3)))
        original=factor._execute; observed=[]
        def execute(phase):
            observed.append(factor.rhs.copy())
            return original(phase)
        monkeypatch.setattr(factor,'_execute',execute)
        rhs=cp.ones((2,3,2));rhs[0,1,0]=bad
        result=factor._solve_device_checked_fused(rhs)
        assert bool(cp.all(observed[0][0]==0.))
        assert bool(cp.all(cp.isnan(result[0])))
        np.testing.assert_array_equal(result[1].get(),np.ones((3,2)))
        np.testing.assert_array_equal(factor._internal_invalid_rhs_count.get(),[1,0])
        np.testing.assert_array_equal(factor._internal_nonfinite_output_count.get(),[0,0])
        good=factor._solve_device_checked_fused(cp.ones_like(rhs))
        assert bool(cp.all(cp.isfinite(good))) and not factor.failed


def test_cuda_nonfinite_native_output_poisoned_without_hiding_other_lanes(monkeypatch):
    import cupy as cp
    a=csr_matrix(np.eye(3))
    with UniformCudssFactor(a,batch_size=2,refinement_steps=0) as factor:
        factor.factor(cp.ones((2,3)))
        original=factor._execute
        def execute(phase):
            original(phase); factor.solution[1,0,0]=cp.inf
        monkeypatch.setattr(factor,'_execute',execute)
        result=factor._solve_device_checked_fused(cp.ones((2,3,1)))
        assert bool(cp.all(cp.isnan(result[1]))) and bool(cp.all(cp.isfinite(result[0])))
        np.testing.assert_array_equal(factor._internal_nonfinite_output_count.get(),[0,1])


def test_cuda_guard_refuses_replaced_native_buffer_before_kernel():
    import cupy as cp
    with UniformCudssFactor(csr_matrix(np.eye(3))) as factor:
        guards=FusedSolveGuards(factor)
        factor.rhs=factor.rhs.copy()
        with pytest.raises(ValueError,match='buffer changed'):guards.pack(cp.ones((1,3,1)))


def test_cuda_guard_protocol_prevents_stale_input_validity():
    import cupy as cp
    with UniformCudssFactor(csr_matrix(np.eye(3))) as factor:
        guards=FusedSolveGuards(factor)
        with pytest.raises(RuntimeError):guards.finish()
        guards.pack(cp.ones((1,3,1)))
        with pytest.raises(RuntimeError):guards.pack(cp.ones((1,3,1)))
