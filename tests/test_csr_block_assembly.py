import numpy as np
import pytest
from scipy.sparse import csr_matrix, block_diag as reference
from src.csr_block_assembly import block_diag


@pytest.mark.parametrize('seed',range(20))
def test_canonical_blocks_match_scipy_exactly_and_own_output(seed):
    rng=np.random.default_rng(seed)
    blocks=[]
    for i in range(6):
        a=rng.normal(size=(i, 7-i))
        a[rng.random(a.shape)<.7]=0.
        blocks.append(csr_matrix(a))
    expected=reference(blocks,format='csr')
    actual=block_diag(blocks)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    for name in ('data','indices','indptr'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
        assert all(not np.may_share_memory(getattr(actual,name),getattr(a,name)) for a in blocks)


def test_duplicates_unsorted_and_stale_flags_preserve_general_scipy_behavior():
    a=csr_matrix(([2.,-2.,1.], [1,1,0], [0,3]),shape=(1,2))
    a.has_sorted_indices=True
    a.has_canonical_format=True
    expected=reference([a,a],format='csr')
    actual=block_diag([a,a])
    for name in ('data','indices','indptr'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))


@pytest.mark.parametrize('dtype',[None,np.float32,np.float64,np.int32,np.complex128])
def test_explicit_zero_empty_rows_and_dtype(dtype):
    blocks=[csr_matrix((0,4)),csr_matrix(([0.,2.],[0,1],[0,0,2]),shape=(2,3)),csr_matrix((3,0))]
    actual=block_diag(blocks,dtype=dtype)
    expected=reference(blocks,format='csr',dtype=dtype)
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    for name in ('data','indices','indptr'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))


@pytest.mark.parametrize('part',['indices','indptr'])
def test_malformed_csr_rejected(part):
    a=csr_matrix(np.eye(3))
    if part=='indices': a.indices[0]=3
    else: a.indptr[-1]=2
    with pytest.raises(ValueError): block_diag([a])


@pytest.mark.parametrize('pointers',[
    np.array([0,2**63-1,1,3],dtype=np.int64),
    np.array([0,-2**63,1,3],dtype=np.int64),
    np.array([0,2,1,3],dtype=np.uint64)])
def test_invalid_pointer_arithmetic_rejected_before_repeat(pointers):
    a=csr_matrix(np.eye(3))
    a.indptr=pointers
    with pytest.raises(ValueError): block_diag([a])


def test_later_malformed_block_checked_before_noncanonical_fallback():
    first=csr_matrix(([1.,2.],[0,0],[0,2]),shape=(1,1))
    later=csr_matrix(np.eye(2))
    later.indptr=np.array([0,2**63-1,2],dtype=np.int64)
    with pytest.raises(ValueError): block_diag([first,later])
