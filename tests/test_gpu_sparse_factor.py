import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_sparse_factor import UniformCudssFactor,CudssError,canonical_square_pattern


def test_uniform_batch_numeric_updates_and_multiple_rhs_without_new_analysis():
    import cupy as cp
    a=csr_matrix([[4.,1.],[1.,-3.]])
    with UniformCudssFactor(a,batch_size=3,nrhs=2) as factor:
        rhs=cp.asarray(np.arange(12,dtype=float).reshape(3,2,2)+1)
        factors=np.array([1.,2.,4.])
        values=cp.asarray(factors[:,None]*a.data)
        factor.factor(values);answer=factor.solve(rhs)
        expected=np.stack([np.linalg.solve(a.toarray()*s,r) for s,r in zip(factors,rhs.get())])
        np.testing.assert_allclose(answer.get(),expected,rtol=1e-11,atol=1e-11)
        saved=answer.copy()
        factor.factor(values*2);changed=factor.solve(rhs)
        np.testing.assert_allclose(changed.get(),expected/2,rtol=1e-11,atol=1e-11)
        np.testing.assert_array_equal(saved.get(),answer.get())
        assert factor.analysis_count==1 and factor.factor_count==factor.solve_count==2
        assert factor.info()==0
    with pytest.raises(CudssError):factor.factor(values)
    factor.close()


def test_wrong_shape_nonfinite_and_stream_are_rejected():
    import cupy as cp
    with UniformCudssFactor(csr_matrix(np.eye(2))) as factor:
        with pytest.raises(CudssError):factor.solve(cp.ones((1,2,1)))
        with pytest.raises(ValueError):factor.factor(cp.ones((2,2)))
        with pytest.raises(ValueError):factor.factor(cp.full((1,2),cp.nan))
        with cp.cuda.Stream():
            with pytest.raises(CudssError):factor.factor(cp.ones((1,2)))
        factor.factor(cp.ones((1,2)));assert factor.info()==0


def test_pattern_is_owned_and_preserves_explicit_zeros():
    a=csr_matrix(([1.,0.,1.],([0,0,1],[0,1,1])),shape=(2,2))
    copied=canonical_square_pattern(a);a.data[:]=9
    assert copied.nnz==3 and copied.data[1]==0


@pytest.mark.parametrize('matrix',[np.zeros((2,3)),np.array([[np.nan]])])
def test_invalid_host_patterns_rejected(matrix):
    with pytest.raises(ValueError):canonical_square_pattern(matrix)


def test_asymmetric_numeric_update_is_not_silently_treated_as_symmetric():
    import cupy as cp
    with UniformCudssFactor(csr_matrix([[2.,1.],[1.,-2.]])) as factor:
        with pytest.raises(ValueError):factor.factor(cp.asarray([[2.,9.,1.,-2.]]))


def test_general_matrix_and_multiple_batch_values():
    import cupy as cp
    a=csr_matrix([[3.,2.],[1.,4.]])
    with UniformCudssFactor(a,batch_size=2,matrix_type='general') as factor:
        factor.factor(cp.asarray(np.stack([a.data,2*a.data])))
        rhs=cp.ones((2,2,1))
        result=factor.solve(rhs).get()
        np.testing.assert_allclose(result[0,:,0],np.linalg.solve(a.toarray(),np.ones(2)),atol=1e-12)
        np.testing.assert_allclose(result[1,:,0],result[0,:,0]/2,atol=1e-12)
