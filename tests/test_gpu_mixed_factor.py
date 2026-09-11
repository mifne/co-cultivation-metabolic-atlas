"""FP32 proposals retain FP64 residuals and reject native corruption."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_sparse_factor import UniformCudssFactor,CudssError
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM


@pytest.mark.parametrize('layout',['uniform','block_diagonal'])
def test_mixed_factor_outer_double_refinement(layout):
    import cupy as cp
    a=np.array([[4.13,1.,0.],[1.,-2.17,1.],[0.,1.,3.29]])
    pattern=csr_matrix(a);rhs=cp.asarray(np.array([[[1.],[2.],[3.]],[[3.],[1.],[2.]]]))
    with UniformCudssFactor(pattern,batch_size=2,precision='float32',execution_layout=layout) as f:
        for scale in (1.,1.3):
            f.factor(cp.asarray(np.stack([pattern.data*scale,pattern.data*(scale+1)])))
            matrix=cp.asarray(np.stack([a*scale,a*(scale+1)]))
            x=f._solve_device_checked(rhs)
            for _ in range(3):x+=f._solve_device_checked(rhs-matrix@x)
            assert x.dtype==cp.float64 and f._native_values.dtype==cp.float32
            assert float(cp.max(cp.abs(matrix@x-rhs)))<1e-10
        assert f.analysis_count==1 and f.factor_count==2
        bad=rhs.copy();bad[0,0,0]=cp.nan
        ans=f._solve_device_checked(bad)
        assert bool(cp.all(cp.isnan(ans[0]))) and bool(cp.all(cp.isfinite(ans[1])))
        f._native_rhs=f._native_rhs.copy()
        with pytest.raises(CudssError,match='buffer identity'):f.solve(rhs)


def test_fp32_matrix_overflow_fails_closed():
    import cupy as cp
    with UniformCudssFactor(csr_matrix([[1.]]),precision='float32') as f:
        with pytest.raises(CudssError,match='overflow'):f.factor(cp.array([[1e300]]))
        assert f.failed and not f.factored


def test_fp32_ipm_requires_original_globalization():
    with pytest.raises(ValueError,match='globalized original-Newton'):
        GpuCondensedBatchedIPM([],factor_precision='float32')


def test_cuda_mixed_ipm_certifies_original_toy():
    p=(csr_matrix([[1.,-1.,0.],[0.,1.,1.]]),np.array([0.,2.]),
       np.zeros(3),np.array([2.,2.,1.]),np.array([0.,0.,-1.]),1)
    with GpuCondensedBatchedIPM([p],factor_precision='float32',globalized=True,
            regularization=1e-6,newton_krylov_iterations=16,predictor_corrector=True,
            predictor_affine_fraction=.995,ipm_initialization='balanced',device_checked_solves=True) as s:
        result=s.solve(iterations=100)
        assert result['accepted'].all()
        assert result['factor_native_precision']=='float32'
