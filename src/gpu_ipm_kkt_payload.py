"""Exact host numeric KKT scatter into an already verified symbolic pattern.

No numeric solve, matrix approximation or persistent cache occurs here. A
fresh union-coverage proof is required on every call; malformed/noncanonical
CSR storage is rejected rather than silently canonicalized.
"""

import math

import numpy as np


def _canonical_csr_parts(matrix, *, shape=None, label):
    """Validate raw CSR storage; never trust cached scipy canonical flags."""
    if getattr(matrix, 'format', None) != 'csr':
        raise ValueError(label+' must be CSR')
    if (len(matrix.shape) != 2 or any(type(v) is not int or v < 0 for v in matrix.shape)
            or (shape is not None and matrix.shape != shape)):
        raise ValueError(label+' has incompatible dimensions')
    data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
    if (not all(isinstance(value, np.ndarray) and value.ndim == 1
                for value in (data, indices, indptr))
            or data.dtype != np.float64
            or indices.dtype not in (np.int32, np.int64)
            or indptr.dtype not in (np.int32, np.int64)):
        raise ValueError(label+' requires FP64 values and int32/int64 CSR coordinates')
    rows, columns = matrix.shape
    if (len(indptr) != rows+1 or len(indices) != len(data)
            or int(indptr[0]) != 0 or int(indptr[-1]) != len(data)
            or np.any(indptr < 0) or np.any(indptr[1:] < indptr[:-1])
            or np.any(indices < 0) or np.any(indices >= columns)
            or max(rows, columns, len(data)) >= 2**31
            or not np.isfinite(data).all()):
        raise ValueError(label+' has malformed, nonfinite or unsupported CSR storage')
    row = np.repeat(np.arange(rows, dtype=np.int64), np.diff(indptr))
    # Strictly increasing columns independently validate both sorting and the
    # absence of duplicates even if .has_canonical_format was cached earlier.
    if np.any((row[1:] == row[:-1]) & (indices[1:] <= indices[:-1])):
        raise ValueError(label+' must have sorted, duplicate-free CSR rows')
    return data, indices.astype(np.int64, copy=False), row


def build_condensed_kkt_payload(forms, host_pattern, *, n, ne, q, regularization):
    """Return exact ``(values[B,nnz], diagonal[size])`` without bmat/union.

    Each constraint form supplies E at index 0 and G at index 2. E is [ne,n]
    and H comprises the first q rows of G. The returned matrix is precisely
    [[delta*I,E.T,H.T],[E,-delta*I,0],[H,0,-I]], including explicit stored
    zeros and zero entries absent from individual lanes but present in the
    other lanes' union. q=0 and ne=0 are supported; n and batch are positive.

    The symbolic CSR must contain exactly the union of all lane coordinates
    and every diagonal. Missing or extra entries both reject the update. Both
    triangles are mapped explicitly; no asymmetric symbolics are accepted.
    Inputs and the native symbolic pattern remain untouched and unaliased
    with the returned independent numeric arrays. Every call validates CSR
    storage and reconstructs coverage; no unchecked layout cache is reused.
    """
    if (any(type(value) is not int for value in (n, ne, q))
            or n < 1 or min(ne, q) < 0 or n+ne+q >= 2**31):
        raise ValueError('Explicit supported n>0, ne>=0, q>=0 dimensions required')
    if (type(regularization) not in (float, int) or not math.isfinite(regularization)
            or regularization <= 0.):
        raise ValueError('Finite positive scalar regularization required')
    forms = tuple(forms)
    if not forms:
        raise ValueError('At least one independent constraint form is required')
    size = n+ne+q
    _, pattern_columns, pattern_rows = _canonical_csr_parts(
        host_pattern, shape=(size, size), label='Symbolic KKT pattern')
    keys = pattern_rows*size+pattern_columns
    coverage = np.zeros(len(keys), dtype=bool)

    def positions(wanted, label):
        result = np.searchsorted(keys, wanted)
        if (np.any(result >= len(keys))
                or not np.array_equal(keys[result], wanted)):
            raise ValueError('Symbolic KKT pattern is missing '+label+' coordinates')
        coverage[result] = True
        return result

    diagonal = positions(np.arange(size, dtype=np.int64)*(size+1), 'diagonal')
    values = np.zeros((len(forms), len(keys)), dtype=np.float64)
    values[:, diagonal[:n]] = regularization
    values[:, diagonal[n:n+ne]] = -regularization
    values[:, diagonal[n+ne:]] = -1.
    for lane, form in enumerate(forms):
        if not isinstance(form, (tuple, list)) or len(form) < 3:
            raise ValueError('Each form must supply E and G in positions 0 and 2')
        e, g = form[0], form[2]
        e_data, e_columns, e_rows = _canonical_csr_parts(
            e, shape=(ne, n), label=f'Lane {lane} E')
        if (getattr(g, 'ndim', None) != 2 or g.shape[1] != n or g.shape[0] < q):
            raise ValueError(f'Lane {lane} G has incompatible dimensions')
        g_data, g_columns, g_rows = _canonical_csr_parts(g, label=f'Lane {lane} G')
        h_end = int(g.indptr[q])
        h_data, h_columns, h_rows = g_data[:h_end], g_columns[:h_end], g_rows[:h_end]
        for name, data, columns, row in (('E', e_data, e_columns, n+e_rows),
                                          ('H', h_data, h_columns, n+ne+h_rows)):
            lower = positions(row*size+columns, f'lane {lane} {name}')
            upper = positions(columns*size+row, f'lane {lane} {name}.T')
            values[lane, lower] = data
            values[lane, upper] = data
    if not np.all(coverage):
        raise ValueError('Symbolic KKT pattern contains extra coordinates outside all lane matrices')
    return values, diagonal
