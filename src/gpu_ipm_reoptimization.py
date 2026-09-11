"""Experimental current-LP restart that discards stale interior duals.

This constructs an infeasible-start IPM proposal, never an LP solution or a
certificate. It is not a full implementation of Gondzio--Grothey warm starts.
Only the primal x is retained. All equality multipliers are reset and every
inequality multiplier/slack is reconstructed from current c, G and h. No
model coefficient, bound, acceptance tolerance or source array is modified.
"""

import math


def centered_bound_restart(solver, x, *, mu):
    """Return ``((xcopy, yzeros, z, s), metadata)`` for the CURRENT LP.

    The inequality order is [q original rows, finite lower, finite upper].
    With y=0, the bound-dual proposal is [0, max(c[il],0), max(-c[iu],0)].
    Let p=max(proposal,sqrt(mu)); then s=max(h-Gx,mu/p) and
    z=max(proposal,mu/s). A bound-cost proposal above the barrier floor is
    retained, so s*z is not necessarily equal to mu on every row. Fixed
    variables remain represented by the solver's existing equality rows;
    neither x nor their bounds are projected or relaxed by this function.

    ``mu`` must be a built-in real scalar in [1e-8,1e-2], not a bool or array.
    NumPy is a tiny reference path; CuPy arithmetic stays on the solver's
    bound device/stream. Only scalar validation synchronizes. Previous y/z/s
    are never read. Finite output is not evidence of feasibility, stationarity
    or optimality: all original-LP acceptance checks remain mandatory.

    The prepared solver and x must not be mutated concurrently. The result
    owns independent arrays and has no automatic acceptance/warm-state cache.
    """
    if (type(mu) not in (int, float) or not math.isfinite(mu)
            or not 1e-8 <= mu <= 1e-2):
        raise ValueError('Explicit finite scalar mu in [1e-8, 1e-2] required')
    xp = solver.cp
    if getattr(xp, '__name__', None) not in ('numpy', 'cupy'):
        raise ValueError('Only NumPy reference or CuPy GPU arithmetic is supported')
    solver.factor._context()
    dimensions = (solver.batch, solver.n, solver.ne, solver.ng, solver.q)
    if (any(type(value) is not int for value in dimensions)
            or min(solver.batch, solver.n, solver.ng) < 1
            or solver.ne < 0 or not 0 <= solver.q <= solver.ng):
        raise ValueError('Consistent positive batch/variable/inequality dimensions required')
    arrays = ((x, (solver.batch, solver.n)),
              (solver.c, (solver.batch, solver.n)),
              (solver.h, (solver.batch, solver.ng)))
    for value, shape in arrays:
        if (not isinstance(value, xp.ndarray) or value.shape != shape
                or value.dtype != xp.float64):
            raise ValueError('Exact-shape FP64 backend x/c/h arrays required')
    for index in (solver.il, solver.iu):
        if (not isinstance(index, xp.ndarray) or index.ndim != 1
                or index.dtype not in (xp.int32, xp.int64)):
            raise ValueError('One-dimensional integer lower/upper coordinate arrays required')
    if solver.q + len(solver.il) + len(solver.iu) != solver.ng:
        raise ValueError('Inequality order must exactly partition q/lower/upper rows')
    if solver.g.shape != (solver.batch*solver.ng, solver.batch*solver.n):
        raise ValueError('Current block-diagonal inequality operator has wrong shape')
    if getattr(solver.g, 'dtype', None) != xp.float64:
        raise ValueError('Current inequality operator must use FP64')
    if xp.__name__ == 'cupy':
        device = solver.factor.device
        if (xp.cuda.runtime.getDevice() != device
                or xp.cuda.get_current_stream().ptr != solver.factor.stream.ptr
                or any(value.device.id != device for value, _shape in arrays)
                or any(index.device.id != device for index in (solver.il, solver.iu))
                or solver.g.data.device.id != device):
            raise ValueError('All inputs must share the bound current CUDA device/stream')
    # Aggregate host decisions only; no vector download or CPU optimizer.
    if not bool(xp.all(xp.stack([xp.all(xp.isfinite(value)) for value, _shape in arrays]))):
        raise ValueError('Finite current x/c/h arrays required')
    for index in (solver.il, solver.iu):
        if bool(xp.any((index < 0) | (index >= solver.n))):
            raise ValueError('Bound coordinates lie outside current variables')
        ordered = xp.sort(index)
        if bool(xp.any(ordered[1:] == ordered[:-1])):
            raise ValueError('Each bound coordinate group must contain unique indices')

    xcopy = x.copy()
    y = xp.zeros((solver.batch, solver.ne), dtype=xp.float64)
    activity = solver._mv(solver.g, xcopy, solver.ng)
    if (not isinstance(activity, xp.ndarray)
            or activity.shape != (solver.batch, solver.ng)
            or activity.dtype != xp.float64
            or (xp.__name__ == 'cupy' and activity.device.id != solver.factor.device)):
        raise ValueError('Current GPU inequality product returned incompatible coordinates')
    raw_slack = solver.h - activity
    if not bool(xp.all(xp.isfinite(activity) & xp.isfinite(raw_slack))):
        raise ValueError('Current inequality activity/slack is not representable in FP64')
    proposal = xp.concatenate((xp.zeros((solver.batch, solver.q), dtype=xp.float64),
        xp.maximum(solver.c[:, solver.il], 0.),
        xp.maximum(-solver.c[:, solver.iu], 0.)), axis=1)
    p = xp.maximum(proposal, math.sqrt(mu))
    slack_floor = mu/p
    s = xp.maximum(raw_slack, slack_floor)
    z_floor = mu/s
    z = xp.maximum(proposal, z_floor)
    # The next Newton iteration needs both representable complementarity and
    # a strictly positive finite s/z; do not silently clip overflow/underflow.
    complementarity, ratio, inverse_ratio = s*z, s/z, z/s
    finite_positive = (xp.isfinite(slack_floor) & (slack_floor > 0.)
        & xp.isfinite(s) & (s > 0.) & xp.isfinite(z_floor) & (z_floor > 0.)
        & xp.isfinite(z) & (z > 0.) & xp.isfinite(complementarity)
        & (complementarity > 0.) & xp.isfinite(ratio) & (ratio > 0.)
        & xp.isfinite(inverse_ratio) & (inverse_ratio > 0.))
    if not bool(xp.all(finite_positive)):
        raise ValueError('Restart slack/dual/complementarity/ratio is not representable in FP64')
    solver.factor._context()
    metadata = dict(method='experimental_current_cost_centered_bound_restart',
        mu=float(mu), old_duals_discarded=True, old_slacks_discarded=True,
        primal_x_unchanged=True, current_LP_requires_new_certificate=True,
        current_bound_dual_cost_source='current_c_with_zero_equality_and_inequality_row_duals',
        original_model_or_bounds_modified=False, cpu_lp_calls=0,
        vector_downloads=0, backend=xp.__name__,
        scope='Infeasible-start proposal only; not a solution or convergence guarantee; not a full Gondzio-Grothey method')
    return (xcopy, y, z, s), metadata
