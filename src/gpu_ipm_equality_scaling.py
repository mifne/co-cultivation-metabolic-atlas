"""Lossless power-of-two normalization of internal IPM equality rows only.

This is host setup for the equivalent equality system R E x = R b. It does
not alter the original LP, x coordinates, objective, inequalities or bounds.
The caller owns the inverse warm-dual map y_scaled=y_original/R and the
physical row-dual map y_original=R*y_scaled, including its sign convention.
Final certificates must always use the unchanged original LP and physical
dual variables. Better-conditioned working equations are not a certificate.
"""

import numpy as np
from scipy.sparse import isspmatrix_csr


def scale_equality_form(form):
    """Return (scaled_form, row_scale) for (E,b,G,h,fixed,il,iu).

    For each nonzero equality row, including b in the maximum, choose a
    binary exponent so max(abs(E_row),abs(b_row)) lies in [0.5,1). All-zero
    rows use scale 1 and remain all-zero; inconsistent 0*x=b rows are kept.
    An exponent outside [-40,40] is explicitly rejected rather than clipped.
    Every stored E coefficient and b value must round-trip exactly through
    ldexp; a subnormal lost or rounded during down-scaling rejects the form.

    E must be a valid FP64 CSR matrix and b a matching FP64 vector. E/b are
    copied. The five remaining form members retain their original references
    and are never mutated. No matrix densification, QR, LP solve or GPU work
    occurs. The returned positive FP64 row_scale is independent storage.
    """
    if not isinstance(form, (tuple, list)) or len(form) != 7:
        raise ValueError('An equality constraint form (E,b,G,h,fixed,il,iu) is required')
    e, b, g, h, fixed, il, iu = form
    if (not isspmatrix_csr(e) or e.dtype != np.float64 or e.ndim != 2
            or e.shape[1] < 1 or not isinstance(b, np.ndarray)
            or b.dtype != np.float64 or b.shape != (e.shape[0],)):
        raise ValueError('FP64 CSR equalities and matching FP64 RHS required')
    # Validate sparse indices/offsets before using them to construct row IDs.
    # No canonicalization/summing of duplicate coefficients is performed.
    checked_e = e.copy()
    checked_e.check_format(full_check=True)
    if (not np.isfinite(e.data).all() or not np.isfinite(b).all()):
        raise ValueError('Finite equality coefficients and RHS required')
    if (not hasattr(g, 'shape') or len(g.shape) != 2 or g.shape[1] != e.shape[1]
            or np.shape(h) != (g.shape[0],) or np.shape(fixed) != (e.shape[1],)):
        raise ValueError('Remaining form dimensions must match the equality variables')
    for indices in (il, iu):
        if (not isinstance(indices, np.ndarray) or indices.ndim != 1
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= e.shape[1])):
            raise ValueError('Bound-column index arrays must be valid integer vectors')

    rows = np.repeat(np.arange(e.shape[0], dtype=np.int64), np.diff(e.indptr))
    maximum = np.abs(b).copy()
    np.maximum.at(maximum, rows, np.abs(e.data))
    _, magnitude_exponent = np.frexp(maximum)
    exponent = np.where(maximum > 0., -magnitude_exponent, 0).astype(np.int32)
    if np.any(exponent < -40) or np.any(exponent > 40):
        raise ValueError('Equality normalization exponent exceeds the bounded range [-40,40]')
    row_scale = np.ldexp(np.ones(e.shape[0], dtype=np.float64), exponent)
    scaled_e = checked_e
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        scaled_e.data = np.ldexp(e.data, exponent[rows])
        scaled_b = np.ldexp(b, exponent)
        restored_data = np.ldexp(scaled_e.data, -exponent[rows])
        restored_b = np.ldexp(scaled_b, -exponent)
    if (not np.isfinite(scaled_e.data).all() or not np.isfinite(scaled_b).all()
            or not np.array_equal(restored_data, e.data)
            or not np.array_equal(restored_b, b)):
        raise ValueError('Equality scaling failed finite lossless ldexp roundtrip')
    # Verify the requested row envelope without relying on logs or rounded
    # multiplication by arbitrary decimal scales.
    scaled_maximum = np.abs(scaled_b).copy()
    np.maximum.at(scaled_maximum, rows, np.abs(scaled_e.data))
    nonzero = maximum > 0.
    if np.any((scaled_maximum[nonzero] < .5) | (scaled_maximum[nonzero] >= 1.)):
        raise ValueError('Equality scaling did not attain the normalization envelope')
    return (scaled_e, scaled_b, g, h, fixed, il, iu), row_scale
