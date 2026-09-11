"""Alternate symbolic/numeric schedules must solve the unchanged equations."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_sparse_factor import UniformCudssFactor, CudssError


@pytest.mark.parametrize('ordering',['default','amd'])
@pytest.mark.parametrize('algorithm',['default','alternate'])
@pytest.mark.parametrize('layout',['uniform','block_diagonal'])
def test_factor_schedules_refactor_and_preserve_independent_lanes(ordering,algorithm,layout):
    import cupy as cp
    a=np.array([[4.,1.,0.],[1.,-2.,1.],[0.,1.,3.]])
    pattern=csr_matrix(a)
    with UniformCudssFactor(pattern,batch_size=2,ordering=ordering,algorithm=algorithm,
                            execution_layout=layout) as f:
        for scale in (1.,1.25):
            f.factor(cp.asarray(np.stack([pattern.data*scale,pattern.data*(scale+1)])))
            rhs=np.array([[[1.],[2.],[3.]],[[3.],[1.],[2.]]])
            result=f.solve(cp.asarray(rhs)).get()
            for lane in range(2):
                np.testing.assert_allclose((a*(scale+lane))@result[lane],rhs[lane],rtol=1e-10,atol=1e-10)
        assert f.analysis_count==1 and f.factor_count==2
        f.ordering='changed'
        with pytest.raises(CudssError,match='algorithm changed'):f.solve(cp.asarray(rhs))


@pytest.mark.parametrize('kwargs',[{'ordering':'unknown'},{'algorithm':'unknown'}])
def test_unknown_factor_schedule_rejected(kwargs):
    with pytest.raises(ValueError,match='ordering or factor algorithm'):
        UniformCudssFactor(csr_matrix([[1.]]),**kwargs)
