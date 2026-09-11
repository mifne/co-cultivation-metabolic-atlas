"""Exact CSR block diagonal assembly without a temporary COO conversion.

The fast path accepts already canonical CSR inputs. Noncanonical or other
formats retain SciPy's general construction, including duplicate summation.
This is input-only host algebra, not a GPU operation or an LP optimizer.
"""
import numpy as np
from scipy.sparse import block_diag as scipy_block_diag, csr_matrix


def block_diag(blocks, format='csr', dtype=None):
    blocks = tuple(blocks)
    if format != 'csr' or not blocks or any(type(a) is not csr_matrix for a in blocks):
        return scipy_block_diag(blocks, format=format, dtype=dtype)
    # Do not rely on cached has_canonical_format flags of legacy mutable CSR.
    needs_fallback = False
    for a in blocks:
        if (a.indptr.ndim != 1 or a.indices.ndim != 1 or a.data.ndim != 1
                or len(a.indptr) != a.shape[0]+1 or len(a.indices) != len(a.data)
                or a.indptr[0] != 0 or a.indptr[-1] != len(a.data)
                or a.indices.dtype not in (np.dtype('int32'),np.dtype('int64'))
                or a.indptr.dtype not in (np.dtype('int32'),np.dtype('int64'))
                or np.any(a.indptr < 0) or np.any(a.indptr > len(a.data))
                or np.any(a.indptr[1:] < a.indptr[:-1])
                or np.any(a.indices < 0) or np.any(a.indices >= a.shape[1])):
            raise ValueError('Malformed CSR cannot be assembled')
        rows = np.repeat(np.arange(a.shape[0], dtype=np.int64), np.diff(a.indptr))
        if np.any((rows[1:] == rows[:-1]) & (a.indices[1:] <= a.indices[:-1])):
            needs_fallback = True
    if needs_fallback:
        return scipy_block_diag(blocks, format=format, dtype=dtype)
    row_sizes = [a.shape[0] for a in blocks]
    col_sizes = [a.shape[1] for a in blocks]
    counts = [len(a.data) for a in blocks]
    shape = (sum(row_sizes), sum(col_sizes))
    index_dtype = np.int64 if max(*shape, sum(counts)) >= np.iinfo(np.int32).max else np.int32
    col_offsets = np.cumsum([0, *col_sizes[:-1]], dtype=np.int64)
    nnz_offsets = np.cumsum([0, *counts[:-1]], dtype=np.int64)
    # Match SciPy: concatenate in the common input dtype, then apply the
    # explicitly requested constructor cast (including lossy numeric casts).
    data = np.concatenate([a.data for a in blocks])
    if dtype is not None: data = data.astype(dtype, copy=False)
    indices = np.concatenate([a.indices.astype(index_dtype, copy=False)+offset
                              for a, offset in zip(blocks, col_offsets)]).astype(index_dtype, copy=False)
    indptr = np.concatenate([*(a.indptr[:-1].astype(index_dtype, copy=False)+offset
                              for a, offset in zip(blocks, nnz_offsets)),
                            np.asarray([sum(counts)], dtype=index_dtype)]).astype(index_dtype, copy=False)
    return csr_matrix((data, indices, indptr), shape=shape)
