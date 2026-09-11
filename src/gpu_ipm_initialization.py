"""Positive inequality-dual initial points for the experimental GPU IPM.

This helper does not solve or certify an LP. ``balanced`` is an opt-in
infeasible-start candidate, not a convergence guarantee: its floor makes
``s*z = 1`` only where the supplied dual/reduced-cost proposal does not exceed
that floor. Original-LP acceptance and certified warm-start handling remain
the caller's responsibility. No vector is copied to the CPU or mutated.
"""


def initialize_inequality_dual(s, original_inequality_dual,
                              reduced_cost_lower, reduced_cost_upper, *,
                              mode='legacy', xp):
    """Return positive FP64 ``z`` in [original inequalities, lower, upper] order.

    All inputs are two-dimensional arrays on the supplied NumPy/CuPy backend
    with the same nonempty batch axis. The three proposal widths must sum to
    the nonempty slack width; individual proposal blocks may be empty.
    ``original_inequality_dual`` uses HiGHS' nonpositive row-dual convention,
    while reduced costs use ``c - A.T @ original_row_dual``. Therefore the
    corresponding positive-dual proposals are ``-row_dual``, ``reduced_cost``
    and ``-reduced_cost``.

    ``legacy`` preserves the original elementwise floor of 1. ``balanced``
    uses the fixed internal complementarity target 1, i.e. floor ``1/s``.
    Slack values must be finite and strictly positive; an unrepresentable
    reciprocal fails closed, without clipping or changing the LP. Large
    legitimate warm duals are retained and need not have unit complementarity.

    CUDA arrays must belong to the same current device. Stream/lifetime
    synchronization belongs to the caller, as arrays have no origin-stream
    identity. Scalar validation may synchronize; vector arithmetic stays on
    the supplied backend. This function never initializes a CUDA context when
    used with NumPy.
    """
    if type(mode) is not str or mode not in ('legacy', 'balanced'):
        raise ValueError('Select the explicit legacy or balanced initializer')
    arrays = (s, original_inequality_dual, reduced_cost_lower, reduced_cost_upper)
    if any(not isinstance(value, xp.ndarray) or value.ndim != 2
           or value.dtype != xp.float64 for value in arrays):
        raise ValueError('FP64 backend arrays of shape [batch, width] required')
    batch, width = s.shape
    if (batch < 1 or width < 1
            or any(value.shape[0] != batch for value in arrays[1:])
            or sum(value.shape[1] for value in arrays[1:]) != width):
        raise ValueError('Matching nonempty batch and exactly partitioned slack width required')
    if hasattr(xp, 'cuda'):
        device = xp.cuda.runtime.getDevice()
        if any(value.device.id != device for value in arrays):
            raise ValueError('All arrays must belong to the same current CUDA device')
    if not all(bool(xp.all(xp.isfinite(value))) for value in arrays):
        raise ValueError('Finite slacks and dual/reduced-cost proposals required')
    if not bool(xp.all(s > 0.)):
        raise ValueError('Strictly positive slacks required')

    if mode == 'balanced':
        if bool(xp.any(s < 1. / xp.finfo(xp.float64).max)):
            raise ValueError('Slack reciprocal is not representable in FP64')
        floor = 1. / s
    else:
        floor = 1.
    proposal = xp.concatenate((-original_inequality_dual, reduced_cost_lower,
                               -reduced_cost_upper), axis=1)
    z = xp.maximum(floor, proposal)
    if not bool(xp.all(xp.isfinite(z) & (z > 0.))):
        raise ValueError('Initializer did not produce finite strictly positive duals')
    return z
