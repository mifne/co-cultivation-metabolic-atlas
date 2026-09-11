"""Host conversion of variable bounds with a dense numeric fast path.

The two callers deliberately have different public contracts.  The persistent
CPU comparator validates SciPy-style bounds, including broadcasting one pair;
the compiled GPU adapter historically only unpacked the supplied iterable.
``checked`` preserves that distinction while sharing the common numeric path.
"""

import numpy as np


def split_variable_bounds(bounds, variables, *, checked):
    """Return independent lower/upper arrays without changing caller semantics.

    Real dense numeric arrays avoid conversion to ``object`` and Python element
    iteration.  Object-valued and unusual inputs retain the callers' original
    conversion paths so ``None`` handling and failure behaviour stay unchanged.
    """
    if bounds is None:
        if checked:
            return np.zeros(variables), np.full(variables, np.inf)
        # Match ``np.array([0] * variables)`` exactly, including the empty-case
        # dtype, while avoiding construction and traversal of a Python list.
        lower = (np.empty(0) if variables == 0
                 else np.zeros(variables, dtype=np.int_))
        return lower, np.full(variables, np.inf)

    if checked:
        values = np.asarray(bounds)
        if values.shape == (2,):
            values = np.broadcast_to(values, (variables, 2))
        if values.shape != (variables, 2):
            raise ValueError('Bounds must be a pair or one pair per variable')
        if values.dtype.kind in 'biuf':
            return (np.array(values[:, 0], dtype=float, copy=True),
                    np.array(values[:, 1], dtype=float, copy=True))

        # Preserve identity-based ``None`` replacement and Python float
        # conversion for object, string, complex, and other unusual dtypes.
        values = np.asarray(bounds, dtype=object)
        if values.shape == (2,):
            values = np.broadcast_to(values, (variables, 2))
        lower = np.array(
            [-np.inf if value is None else value for value in values[:, 0]],
            dtype=float,
        )
        upper = np.array(
            [np.inf if value is None else value for value in values[:, 1]],
            dtype=float,
        )
        return lower, upper

    # The unchecked adapter accepts any number of bound pairs and does not
    # broadcast a single pair.  Restrict acceleration to ordinary dense real
    # containers for which column copies exactly match two list comprehensions.
    values = bounds if type(bounds) is np.ndarray else None
    if (values is not None and values.ndim == 2
            and values.shape[1] == 2 and values.dtype.kind in 'biuf'):
        if values.shape[0] == 0:
            return np.empty(0), np.empty(0)
        # Iterating a non-native-endian ndarray yields native NumPy scalars, so
        # the legacy list comprehensions also returned native-endian arrays.
        dtype = values.dtype.type
        return (np.array(values[:, 0], dtype=dtype, copy=True),
                np.array(values[:, 1], dtype=dtype, copy=True))

    # This is intentionally two passes: a one-shot iterator therefore retains
    # the legacy behaviour where the lower pass exhausts it before the upper.
    lower = np.array(
        [-np.inf if lo is None else lo for lo, hi in bounds]
    )
    upper = np.array(
        [np.inf if hi is None else hi for lo, hi in bounds]
    )
    return lower, upper
