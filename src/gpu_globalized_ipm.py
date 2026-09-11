"""Opt-in, bounded globalized inexact primal-dual Newton experiment.

Only the *inner* direction accuracy is inexact. Final success still comes from
the solver's unchanged original-LP certificate. A centered direction, fixed
weighted nonlinear KKT merit, common primal/dual step, and Armijo backtracking
replace the heuristic predictor/corrector in this optional path. An additional
explicit option uses a safeguarded predictor/corrector with the same full
direction and actual-merit guards, and one centered fallback. This is not a
polynomial-convergence claim for rank-deficient/degenerate GEMs.

References: Eisenstat & Walker (1996), doi:10.1137/0917003;
Zanetti & Gondzio (2023), doi:10.1137/22M1490041. The particular fixed forcing
parameter and globalization below are an experimental design, not a literal
implementation of either paper's complete algorithm.
"""

import math
import time
from contextlib import contextmanager

import numpy as np

from .gpu_newton_krylov import _norm2, batched_gmres


def weighted_blocks(blocks, weights, *, xp):
    """Concatenate independently weighted blocks; weights stay fixed per LP."""
    return xp.concatenate([value * weight[:, None]
                           for value, weight in zip(blocks, weights)
                           if value.shape[1]], axis=1)


def relative_forcing(defect, target, *, xp):
    """True relative 2-norm: no max(1, RHS) floor near convergence."""
    numerator, denominator = _norm2(defect, xp), _norm2(target, xp)
    ratio = numerator / xp.where(denominator > 0., denominator, 1.)
    ratio = xp.where(denominator > 0., ratio,
                     xp.where(numerator == 0., 0., xp.inf))
    return xp.where(xp.isfinite(numerator) & xp.isfinite(denominator), ratio, xp.inf)


def full_defect_from_eliminated(residual, z, n, ne, *, xp):
    """Map rhs-K0*d to full stationarity/equality/complementarity defects.

    ds=-rg-Gdx makes the inequality linearization exact algebraically. The
    eliminated third-row residual must be MULTIPLIED by z to recover the
    physical complementarity defect, even when z is very large or small.
    Actual direction acceptance also recomputes all four full blocks.
    """
    return xp.concatenate((-residual[:, :n], -residual[:, n:n+ne],
                           z * residual[:, n+ne:]), axis=1)


def _positive_step(value, delta, *, xp):
    ratio = -value / xp.where(delta < 0., delta, -1.)
    return xp.minimum(1., xp.min(xp.where(delta < 0., ratio, xp.inf), axis=1))


def guarded_corrector_rhs(s, z, comp, mu, ds_affine, dz_affine, eligible, *, xp,
                          affine_fraction=1.0, return_diagnostics=False):
    """Mehrotra RHS only from already qualified, finite affine directions.

    Mask before multiplication: an unqualified lane's huge/NaN directions
    must not enter either the predictor product or ds_affine*dz_affine.
    Overflow-prone qualified products also fail closed to the centered path;
    they are not clipped into a seemingly valid corrector.
    The optional affine fraction changes only the point used to estimate
    sigma. It does not scale the qualified affine cross product or admit a
    state update. Negative predicted slacks are diagnosed, never clipped.
    """
    if (isinstance(affine_fraction, bool) or not math.isfinite(affine_fraction)
            or not 0. < affine_fraction <= 1. or type(return_diagnostics) is not bool):
        raise ValueError('Affine fraction in (0,1] and boolean diagnostic flag required')
    finite_direction = (xp.isfinite(mu)
        & xp.all(xp.isfinite(ds_affine) & xp.isfinite(dz_affine), axis=1))
    valid = eligible & finite_direction & (mu > 0.)
    reason = xp.where(~eligible, 1, xp.where(~finite_direction, 2,
                      xp.where(mu <= 0., 3, 0))).astype(xp.int32)
    ds = xp.where(valid[:, None], ds_affine, 0.)
    dz = xp.where(valid[:, None], dz_affine, 0.)
    largest = xp.finfo(xp.float64).max
    product_safe = xp.all(xp.abs(ds) <= largest/xp.maximum(1., xp.abs(dz)), axis=1)
    reason = xp.where(valid & ~product_safe, 4, reason)
    valid &= product_safe
    ds, dz = xp.where(valid[:, None], ds, 0.), xp.where(valid[:, None], dz, 0.)
    ap = affine_fraction*_positive_step(s, ds, xp=xp)
    ad = affine_fraction*_positive_step(z, dz, xp=xp)
    # Preserve the actual summands before later safety masks zero ds/dz.
    # Diagnostic roundoff scales must describe this prediction arithmetic.
    step_s, step_z = ap[:, None]*ds, ad[:, None]*dz
    ps, pz = s+step_s, z+step_z
    predicted_finite = xp.all(xp.isfinite(ps) & xp.isfinite(pz), axis=1)
    predicted_nonnegative = xp.all((ps >= 0.) & (pz >= 0.), axis=1)
    predicted_product_safe = xp.all(xp.abs(ps) <= largest/xp.maximum(1., xp.abs(pz)), axis=1)
    reason = xp.where(valid & ~predicted_finite, 5, reason)
    reason = xp.where(valid & predicted_finite & ~predicted_nonnegative, 6, reason)
    reason = xp.where(valid & predicted_finite & predicted_nonnegative & ~predicted_product_safe, 7, reason)
    # Retain signed raw predicted mu for diagnosis, even when a tiny negative
    # boundary coordinate invalidates this target. Never clip it to authorize
    # a corrector. Products are still masked BEFORE unsafe multiplication.
    raw_safe = valid & predicted_finite & predicted_product_safe
    raw_ps, raw_pz = xp.where(raw_safe[:, None], ps, 0.), xp.where(raw_safe[:, None], pz, 0.)
    raw_mu = xp.sum((raw_ps*raw_pz)/s.shape[1], axis=1)
    denominator = xp.where(raw_safe, mu, 1.)
    ratio_safe = raw_safe & (xp.abs(raw_mu) <= largest*xp.minimum(denominator, 1.))
    raw_ratio = xp.where(ratio_safe, raw_mu, 0.)/denominator
    cube_safe = ratio_safe & xp.isfinite(raw_ratio) & (xp.abs(raw_ratio) < largest**(1./3.))
    raw_sigma = xp.where(cube_safe, raw_ratio, 0.)**3
    valid &= predicted_finite & predicted_nonnegative & predicted_product_safe
    sigma = xp.clip(xp.where(valid, raw_ratio, 0.), 0., 1.)**3
    reason = xp.where(valid & ~ratio_safe, 9, reason)
    valid &= ratio_safe
    ds, dz = xp.where(valid[:, None], ds, 0.), xp.where(valid[:, None], dz, 0.)
    correction = ds*dz
    rc = comp+correction-sigma[:, None]*mu[:, None]
    finite_rc = xp.all(xp.isfinite(rc), axis=1) & xp.isfinite(sigma)
    reason = xp.where(valid & ~finite_rc, 8, reason)
    valid &= finite_rc
    result = (xp.where(valid[:, None], rc, 0.), xp.where(valid, sigma, 0.), valid)
    if not return_diagnostics:
        return result
    # Dimensionless roundoff units are DIAGNOSTICS, not an acceptance tolerance.
    # Scale each summand before adding to avoid overflow in |old|+|alpha*d|.
    eps, tiny = xp.finfo(xp.float64).eps, xp.finfo(xp.float64).tiny
    scale_s = eps*xp.abs(s)+eps*xp.abs(step_s)
    scale_z = eps*xp.abs(z)+eps*xp.abs(step_z)
    negative_s = xp.max(xp.maximum(0., -ps)/xp.maximum(tiny, scale_s), axis=1)
    negative_z = xp.max(xp.maximum(0., -pz)/xp.maximum(tiny, scale_z), axis=1)
    diagnostic = dict(eligible=eligible, fraction=xp.full_like(mu, affine_fraction),
        alpha_p=ap, alpha_d=ad, min_predicted_s=xp.min(ps, axis=1),
        min_predicted_z=xp.min(pz, axis=1), max_negative_s_roundoff_units=negative_s,
        max_negative_z_roundoff_units=negative_z, mu=mu,
        raw_mu_affine=xp.where(raw_safe, raw_mu, xp.nan),
        raw_sigma=xp.where(cube_safe, raw_sigma, xp.nan), safe=valid, failure_reason=reason)
    return (*result, diagnostic)


def backtrack_centered_step(state, direction, blocks, derivative, weights,
                            eligible, *, xp, max_backtracks=12, armijo=1e-4):
    """Independent common-step Armijo trials, using exact LP residual formulas.

    state/direction order is (x,y,z,s). All four residual blocks are retained:
    (stationarity, equality, inequality, complementarity). No matrix product
    is needed per backtrack because the first three blocks are affine and
    the final block is the exact product of the trial slack and multiplier.
    Returned state is never modified in-place; failed lanes remain unchanged.
    """
    if (type(max_backtracks) is not int or not 0 <= max_backtracks <= 32
            or not math.isfinite(armijo) or not 0. < armijo < 1.):
        raise ValueError('Bounded backtracking and an Armijo constant in (0,1) required')
    x, y, z, s = state
    dx, dy, dz, ds = direction
    f = weighted_blocks(blocks, weights, xp=xp)
    jd = weighted_blocks(derivative, weights, xp=xp)
    norm = _norm2(f, xp)
    before = .5 * norm * norm
    slope = xp.sum(f * jd, axis=1)
    finite_direction = xp.all(xp.isfinite(xp.concatenate(direction, axis=1)), axis=1)
    descent = (eligible & finite_direction & xp.isfinite(before)
               & xp.isfinite(slope) & (slope < 0.))
    alpha = .995 * xp.minimum(_positive_step(s, ds, xp=xp), _positive_step(z, dz, xp=xp))
    alpha = xp.where(descent, alpha, 0.)
    taken = xp.zeros_like(eligible)
    backtracks = xp.zeros(eligible.shape, dtype=xp.int32)
    after = before.copy()
    selected_alpha = xp.zeros_like(alpha)
    result = tuple(value.copy() for value in state)
    for attempt in range(max_backtracks + 1):
        pending = descent & ~taken
        if not bool(xp.any(pending)):
            break
        trial = tuple(value + alpha[:, None] * delta
                      for value, delta in zip(state, direction))
        tx, ty, tz, ts = trial
        trial_blocks = tuple(blocks[k] + alpha[:, None] * derivative[k] for k in range(3))
        trial_blocks += (ts * tz,)
        trial_norm = _norm2(weighted_blocks(trial_blocks, weights, xp=xp), xp)
        trial_merit = .5 * trial_norm * trial_norm
        finite = xp.all(xp.isfinite(xp.concatenate(trial, axis=1)), axis=1)
        valid = (pending & finite & xp.all(ts > 0., axis=1) & xp.all(tz > 0., axis=1)
                 & xp.isfinite(trial_merit) & (alpha > 0.)
                 & (trial_merit <= before + armijo * alpha * slope))
        result = tuple(xp.where(valid[:, None], value, old)
                       for value, old in zip(trial, result))
        selected_alpha = xp.where(valid, alpha, selected_alpha)
        after = xp.where(valid, trial_merit, after)
        backtracks = xp.where(valid, attempt, backtracks)
        taken |= valid
        alpha = xp.where(pending & ~valid, .5 * alpha, alpha)
    backtracks = xp.where(descent & ~taken, max_backtracks, backtracks)
    return result, dict(taken=taken, descent=descent, alpha=selected_alpha,
                        before=before, after=after, slope=slope, backtracks=backtracks)


def backtrack_primal_dual_step(state,direction,blocks,derivative,weights,eligible,*,xp):
    """Independent primal/dual boundary limits, with the SAME actual merit.

    Newton forcing is checked on the original unscaled direction by caller.
    This only changes the line-search path, as separate primal/dual steps in
    standard IPMs. A common backtracking parameter still enforces Armijo;
    actual original residuals must be recomputed before commit by caller.
    """
    x,y,z,s=state;dx,dy,dz,ds=direction
    finite=eligible&xp.all(xp.isfinite(xp.concatenate(direction,axis=1)),axis=1)
    safe=tuple(xp.where(finite[:,None],v,0.) for v in direction)
    dx,dy,dz,ds=safe
    ap=_positive_step(s,ds,xp=xp);ad=_positive_step(z,dz,xp=xp)
    scaled=(ap[:,None]*dx,ad[:,None]*dy,ad[:,None]*dz,ap[:,None]*ds)
    scaled_derivative=(ad[:,None]*derivative[0],ap[:,None]*derivative[1],
        ap[:,None]*derivative[2],z*scaled[3]+s*scaled[2])
    result,line=backtrack_centered_step(state,scaled,blocks,scaled_derivative,weights,finite,xp=xp)
    line['alpha_primal']=line['alpha']*ap
    line['alpha_dual']=line['alpha']*ad
    return result,line


def _history(values, cp):
    """Only compact diagnostics leave the GPU; nonfinite diagnostics are null."""
    if not values:
        return []
    array = cp.stack(values).get()
    return np.where(np.isfinite(array), array, None).tolist()


def retry_regularizations(base, budget, *, floor=1e-8):
    """A bounded, decreasing retry list; never raise delta to reach the floor."""
    if (type(budget) is not int or not 0 <= budget <= 2
            or not math.isfinite(base) or base <= 0.
            or not math.isfinite(floor) or floor<=0.):
        raise ValueError('Finite positive base regularization and retry budget 0..2 required')
    values = []
    previous = base
    for attempt in range(budget):
        value = max(floor, base * .1**(attempt+1))
        if value >= previous:
            break
        values.append(value)
        previous = value
    return values


def barrier_regularization(base, mu, active, *, xp):
    """Experimental scalar preconditioner schedule, not an LP perturbation.

    Keep the configured delta early; reduce it when it exceeds one percent
    of the largest active complementarity mu. A shared batch factor therefore
    does not force a less advanced environment to use the smallest lane's
    delta. The fixed 1e-12 floor bounds this numerical experiment. Final full
    Newton forcing and the original LP certificate remain unchanged.
    """
    if not math.isfinite(base) or not 0.<base<=1e-2:
        raise ValueError('Positive bounded base regularization required')
    if (not isinstance(mu,xp.ndarray) or mu.ndim!=1 or mu.size<1 or mu.dtype!=xp.float64
            or not isinstance(active,xp.ndarray) or active.shape!=mu.shape or active.dtype!=xp.bool_):
        raise ValueError('FP64 complementarity and boolean active vectors required')
    if not bool(xp.any(active)):
        return float(base)
    if not bool(xp.all(~active | (xp.isfinite(mu)&(mu>0.)))):
        raise ValueError('Finite positive active complementarity required')
    largest=float(xp.max(xp.where(active,mu,0.)))
    return min(float(base),max(1e-12,.01*largest))


@contextmanager
def temporary_regularization(solver, value):
    """Restore configuration even if factorization or a solve raises."""
    base = solver.regularization
    solver.regularization = value
    try:
        yield
    finally:
        solver.regularization = base


def select_better_direction(previous, candidate, previous_error, candidate_error,
                            requested, *, xp):
    """Only finite, strictly better requested lanes replace cached directions."""
    selected = (requested & xp.all(xp.isfinite(candidate), axis=1)
                & xp.isfinite(candidate_error) & (candidate_error < previous_error))
    return (xp.where(selected[:, None], candidate, previous),
            xp.where(selected, candidate_error, previous_error), selected)


def solve_globalized_ipm(solver, *, initial_x=None, initial_y=None, iterations=60,
                         check_interval=1, capture_failure=False, regularization_retries=0,
                         predictor_corrector=False, predictor_affine_fraction=1.0,
                         ipm_initialization='legacy',regularization_schedule='fixed',
                         internal_warm_start=None,factor_reuse_interval=1):
    """Bounded centered or safeguarded predictor/corrector GPU Newton steps.

    A constant forcing eta is relative to the current FULL nonlinear Newton
    target. Thus the absolute inner error shrinks with convergence without
    imposing a tiny fixed tolerance at every iteration. Uncertified failed
    lanes freeze independently; there is no CPU LP call. Optional predictor/
    corrector failure permits one same-state centered direction, not an
    uncertified step. Regularization retries are shared across all direction
    targets in an outer iteration, rather than multiplied by the PC stages.
    """
    from .gpu_sparse_factor import CudssError
    if solver.closed:
        raise RuntimeError('Closed IPM workspace')
    solver.factor._context()
    eta = getattr(solver, 'forcing_eta', .05)
    if (not math.isfinite(eta) or not 0. < eta <= .25
            or type(iterations) is not int or iterations < 0
            or type(check_interval) is not int or check_interval < 1
            or type(capture_failure) is not bool or type(predictor_corrector) is not bool):
        raise ValueError('Valid finite forcing eta in (0,.25] and iteration/check budgets required')
    if (isinstance(predictor_affine_fraction, bool) or not math.isfinite(predictor_affine_fraction)
            or not 0. < predictor_affine_fraction <= 1.
            or (not predictor_corrector and predictor_affine_fraction != 1.)):
        raise ValueError('Predictor affine fraction must be in (0,1] and requires PC when not 1')
    cp = solver.cp
    if ipm_initialization not in ('legacy','balanced'):
        raise ValueError('Select legacy or balanced IPM initialization')
    if regularization_schedule not in ('fixed','barrier'):
        raise ValueError('Select fixed or barrier preconditioner regularization')
    if (type(factor_reuse_interval) is not int or not 1 <= factor_reuse_interval <= 8
            or (factor_reuse_interval > 1 and
                (not getattr(solver,'_condense_bounds',False)
                 or solver.krylov_coordinates != 'condensed'
                 or not solver.newton_krylov_iterations
                 or regularization_schedule != 'fixed'))):
        raise ValueError('Factor reuse interval 1..8 requires condensed Krylov and fixed regularization')
    retry_values = retry_regularizations(solver.regularization, regularization_retries)
    before = time.perf_counter()
    solver.failure_snapshot = None
    solver._last_internal_state = None
    if internal_warm_start is not None:
        from .gpu_ipm_warm_state import BoundGpuWarmState
        if not isinstance(internal_warm_start, BoundGpuWarmState):
            raise ValueError('A causally bound GPU interior proposal is required')
        x, y, z, s = internal_warm_start.initialize(solver)
        original_y = solver._row_dual(y, z)
    else:
        # A bound internal proposal already owns all four variables; do not
        # construct, validate and then discard a cold point in that path.
        x = solver._initial(initial_x, (solver.batch, solver.n))
        original_y = solver._initial(initial_y, (solver.batch, solver.m))
        y = cp.zeros((solver.batch, solver.ne), dtype=cp.float64)
        y[:, :solver.neq] = -original_y[:, :solver.neq]
        reduced = solver.c - solver._mv(solver.assembled[0].T, original_y, solver.n)
        y[:, solver.neq:] = -reduced[:, solver.fixed]
        if getattr(solver,'equality_row_scaling',False):
            y /= solver.equality_scale
        s = cp.maximum(1., solver.h - solver._mv(solver.g, x, solver.ng))
        from .gpu_ipm_initialization import initialize_inequality_dual
        z = initialize_inequality_dual(s, original_y[:, solver.neq:],
            reduced[:, solver.il], reduced[:, solver.iu], mode=ipm_initialization, xp=cp)
    # Balanced is an alternative positive internal point, not a modification
    # to a bound, objective, supplied x/y, or the original certificate.
    metrics = solver.certificate(x, original_y)
    accepted = np.asarray([r['certificate_passed'] for r in metrics], dtype=bool)
    done = cp.asarray(accepted)
    failed = cp.zeros(solver.batch, dtype=cp.int32)
    accepted_iteration = np.where(accepted, 0, -1)
    result_x, result_y = x.copy(), original_y.copy()
    checkpoints = [dict(iteration=0, accepted=int(accepted.sum()), metrics=list(metrics))]
    factor_seconds = solve_seconds = certificate_seconds = 0.
    effective_factor=getattr(solver,'numerical_factor',solver.factor)
    start_factor, start_solve = effective_factor.factor_count, effective_factor.solve_count
    completed = attempted = 0
    dirty = False
    history, newton_history, krylov_history, krylov_codes = [], [], [], []
    retry_history = []
    target_history, pc_history = [], []
    affine_history = []
    regularization_schedule_history=[]
    affine_columns = ['eligible', 'fraction', 'alpha_p', 'alpha_d', 'min_predicted_s',
        'min_predicted_z', 'max_negative_s_roundoff_units', 'max_negative_z_roundoff_units',
        'mu', 'raw_mu_affine', 'raw_sigma', 'safe', 'failure_reason']
    roles=5 if getattr(solver,'centrality_corrections',0) else 4
    direction_attempt_counts = {role: 0 for role in range(roles)}
    direction_solve_counts = {role: 0 for role in range(roles)}
    centrality_history=[]
    primal_dual_step_history=[]
    extrapolation_history=[]
    retry_factor_seconds = 0.
    retry_factor_attempts = 0
    numeric_error = None
    cached_factor_scaling = cached_factor_delta = None
    last_factor_iteration = 0
    factor_reuse_count = factor_refresh_count = 0
    factor_refresh_seconds = 0.
    initial_mu = cp.sum((s*z)/solver.ng, axis=1)
    initialization_diagnostics=cp.stack((initial_mu,cp.min(s,axis=1),cp.max(s,axis=1),
        cp.min(z,axis=1),cp.max(z,axis=1),cp.min(z/s,axis=1),cp.max(z/s,axis=1)),axis=1)

    def fixed_weight(data, width):
        magnitude = (cp.max(cp.abs(data), axis=1) if width else cp.ones(solver.batch))
        return 1. / (cp.maximum(1., magnitude) * math.sqrt(max(1, width)))

    weights = (fixed_weight(solver.c, solver.n), fixed_weight(solver.b, solver.ne),
               fixed_weight(solver.h, solver.ng),
               1. / (cp.maximum(1., initial_mu) * math.sqrt(solver.ng)))

    def residuals(state=None):
        cx, cy, cz, cs = (x, y, z, s) if state is None else state
        return (solver.c + solver._mv(solver.et, cy, solver.n) + solver._mv(solver.gt, cz, solver.n),
                solver._mv(solver.e, cx, solver.ne)-solver.b,
                solver._mv(solver.g, cx, solver.ng)+cs-solver.h, cs*cz)

    def record_certificate(iteration):
        nonlocal metrics, accepted, done, result_x, result_y, certificate_seconds, dirty
        stamp = time.perf_counter()
        candidate_y = solver._row_dual(y, z)
        latest = solver.certificate(x, candidate_y)
        newly = ~accepted & np.asarray([r['certificate_passed'] for r in latest], dtype=bool)
        select = cp.asarray(~accepted)[:, None]
        result_x = cp.where(select, x, result_x)
        result_y = cp.where(select, candidate_y, result_y)
        metrics = [old if accepted[i] else latest[i] for i, old in enumerate(metrics)]
        accepted |= newly
        accepted_iteration[newly] = iteration
        done = cp.asarray(accepted)
        checkpoints.append(dict(iteration=iteration, accepted=int(accepted.sum()), metrics=list(metrics)))
        certificate_seconds += time.perf_counter()-stamp
        dirty = False

    for iteration in range(1, iterations+1):
        active = ~done & (failed == 0)
        if not bool(cp.any(active)):
            break
        attempted = iteration
        blocks = residuals()
        rd, rp, rg, comp = blocks
        mu = cp.sum(comp/solver.ng, axis=1)
        ratio = s/z
        finite = (cp.all(cp.isfinite(cp.concatenate(blocks, axis=1)), axis=1)
                  & cp.all(cp.isfinite(ratio) & (ratio > 0.), axis=1)
                  & cp.isfinite(mu) & (mu>0.))
        failed = cp.where(active & ~finite, 1, failed)
        active &= finite
        if not bool(cp.any(active)):
            break
        safe_ratio = cp.where(active[:, None], ratio, 1.)
        solver._current_factor_active=active
        iteration_delta=(barrier_regularization(solver.regularization,mu,active,xp=cp)
                         if regularization_schedule=='barrier' else solver.regularization)
        iteration_retry_values=(retry_regularizations(iteration_delta,regularization_retries,floor=1e-12)
                                if regularization_schedule=='barrier' else retry_values)
        regularization_schedule_history.append(dict(iteration=iteration,delta=iteration_delta))
        lagged_factor = (cached_factor_scaling is not None
            and iteration-last_factor_iteration < factor_reuse_interval
            and cached_factor_delta == iteration_delta)
        stamp = time.perf_counter()
        try:
            if lagged_factor:
                scaling = cached_factor_scaling
                factor_reuse_count += 1
            else:
                with temporary_regularization(solver,iteration_delta):
                    scaling = solver._factor_newton(safe_ratio)
                last_factor_iteration = iteration
        except CudssError as error:
            failed = cp.where(active, 5, failed)
            numeric_error = str(error)
            factor_seconds += time.perf_counter()-stamp
            break
        cp.cuda.get_current_stream().synchronize()
        factor_seconds += time.perf_counter()-stamp
        stamp = time.perf_counter()

        def products(answer, rc, target):
            dx = answer[:, :solver.n]
            dy = answer[:, solver.n:solver.n+solver.ne]
            dz = answer[:, solver.n+solver.ne:]
            gdx = solver._mv(solver.g, dx, solver.ng)
            ds = -rg-gdx
            derivative = (solver._mv(solver.et, dy, solver.n)+solver._mv(solver.gt, dz, solver.n),
                          solver._mv(solver.e, dx, solver.ne), gdx+ds, z*ds+s*dz)
            defect = weighted_blocks((rd+derivative[0], rp+derivative[1], rg+derivative[2],
                                      rc+derivative[3]), weights, xp=cp)
            return (dx, dy, dz, ds), derivative, relative_forcing(defect, target, xp=cp)

        gmres_iterations = cp.zeros(solver.batch, dtype=cp.int32)
        gmres_termination = cp.zeros(solver.batch, dtype=cp.int32)
        iteration_retry_count = cp.zeros(solver.batch, dtype=cp.int32)
        iteration_retry_factor_seconds = 0.
        last_factor_scaling = scaling
        last_factor_delta = iteration_delta
        retry_cursor = 0
        old_state = (x, y, z, s)

        def compute_direction(current_scaling, requested, rc, target, rhs):
            """Same nonlinear state/RHS for initial and safeguarded retries."""
            answer = solver._solve_newton(cp.where(requested[:, None], rhs, 0.),
                                           safe_ratio, current_scaling)
            direction, derivative, forcing = products(answer, rc, target)
            need = requested & (~cp.isfinite(forcing) | (forcing > eta))
            invalid_weight = cp.zeros(solver.batch, dtype=cp.bool_)
            inner_iterations = cp.zeros(solver.batch, dtype=cp.int32)
            inner_termination = cp.zeros(solver.batch, dtype=cp.int32)
            if solver.newton_krylov_iterations and bool(cp.any(need)):
                if solver.krylov_coordinates == 'condensed':
                    from .gpu_globalized_condensed import globalized_condensed_gmres
                    finite_answer = cp.all(cp.isfinite(answer), axis=1)
                    initial = cp.where((need & finite_answer)[:, None], answer, 0.)
                    improved, diagnostic, invalid_weight = globalized_condensed_gmres(
                        solver, rhs, initial, safe_ratio, current_scaling, need,
                        weights, target, eta, z=z)
                    # An algebraically small condensed residual cannot hide
                    # amplification when bound directions are reconstructed.
                    # Acceptance still uses every actual full nonlinear block.
                    trial_direction, trial_derivative, trial_forcing = products(improved, rc, target)
                    answer, forcing, _ = select_better_direction(answer, improved, forcing,
                        trial_forcing, need & ~invalid_weight, xp=cp)
                    return (answer, forcing, diagnostic['iterations'],
                            diagnostic['termination'], invalid_weight)
                # Left weighting converts the eliminated residual into the
                # full stationarity/equality/complementarity defect norm.
                # The inequality block is zero algebraically and is checked
                # explicitly again by products() after the Krylov solve.
                linear_weight = cp.concatenate((cp.broadcast_to(weights[0][:, None], (solver.batch, solver.n)),
                    cp.broadcast_to(weights[1][:, None], (solver.batch, solver.ne)),
                    weights[3][:, None]*z), axis=1).copy()
                valid_weight = cp.all(cp.isfinite(linear_weight) & (linear_weight > 0.), axis=1)
                invalid_weight = need & ~valid_weight
                need &= valid_weight
                linear_weight = cp.where(need[:, None], linear_weight, 1.)
                finite_answer = cp.all(cp.isfinite(answer), axis=1)
                initial = cp.where((need & finite_answer)[:, None], answer, 0.)
                masked_rhs = cp.where(need[:, None], linear_weight*rhs, 0.)
                masked_target = cp.where(need[:, None], target, 0.)
                improved, diagnostic = batched_gmres(masked_rhs, initial,
                    lambda value: linear_weight*solver._kkt_mv(value, safe_ratio, regularized=False),
                    lambda value: solver._solve_newton(value/linear_weight, safe_ratio, current_scaling),
                    lambda ignored, residual: relative_forcing(residual, masked_target, xp=cp),
                    xp=cp, max_iterations=solver.newton_krylov_iterations, tolerance=eta,
                    microkernels=getattr(solver,'krylov_microkernels','none'),
                    defer_lane_checks=getattr(solver,'krylov_defer_lane_checks',False),
                    workspace=solver._gmres_workspace(solver.size) if getattr(solver,'reuse_gmres_workspace',False) else None)
                trial_direction, trial_derivative, trial_forcing = products(improved, rc, target)
                answer, forcing, _ = select_better_direction(answer, improved, forcing,
                                                              trial_forcing, need, xp=cp)
                inner_iterations = diagnostic['iterations']
                inner_termination = diagnostic['termination']
            return answer, forcing, inner_iterations, inner_termination, invalid_weight

        def solve_target(rc, requested, role):
            """One target with its own RHS norm, and only same-state retries."""
            nonlocal last_factor_scaling, last_factor_delta, retry_cursor
            nonlocal factor_seconds, retry_factor_seconds, iteration_retry_factor_seconds
            nonlocal retry_factor_attempts, gmres_iterations, gmres_termination
            nonlocal iteration_retry_count
            nonlocal lagged_factor,last_factor_iteration,factor_refresh_count,factor_refresh_seconds
            requested = requested.copy()
            target = weighted_blocks((rd, rp, rg, rc), weights, xp=cp)
            rhs = cp.concatenate((-rd, -rp, rc/z-rg), axis=1)
            finite = cp.all(cp.isfinite(rhs), axis=1) & cp.isfinite(_norm2(target, cp))
            valid_request = requested & finite
            rhs = cp.where(valid_request[:, None], rhs, 0.)
            answer = cp.zeros_like(rhs)
            forcing = cp.full_like(mu, cp.inf)
            direction_delta = cp.full_like(mu, last_factor_delta)
            inner_total = cp.zeros(solver.batch, dtype=cp.int32)
            inner_codes = cp.zeros(solver.batch, dtype=cp.int32)
            target_retries = cp.zeros(solver.batch, dtype=cp.int32)
            invalid_weight = cp.zeros(solver.batch, dtype=cp.bool_)

            def run_direction(current_scaling, selected):
                direction_attempt_counts[role] += 1
                solve_start = effective_factor.solve_count
                try:
                    return compute_direction(current_scaling, selected, rc, target, rhs)
                finally:
                    direction_solve_counts[role] += effective_factor.solve_count-solve_start

            if bool(cp.any(valid_request)):
                # A preceding target may have consumed a retry and left a
                # smaller-delta factor in the workspace. Reuse that factor
                # with its matching scaling/configuration, then restore the
                # caller's base delta even on exceptions.
                with temporary_regularization(solver, last_factor_delta):
                    answer, forcing, inner_total, inner_codes, invalid_weight = run_direction(
                        last_factor_scaling, valid_request)
            valid_request &= ~invalid_weight
            # A stale factor is ONLY a preconditioner for the current K0.
            # Retry the same nonlinear state with a current GPU factor before
            # consuming any regularization retry or committing an update.
            refresh_request = valid_request & (~cp.isfinite(forcing) | (forcing > eta))
            if lagged_factor and bool(cp.any(refresh_request)):
                refresh_stamp=time.perf_counter()
                try:
                    with temporary_regularization(solver,last_factor_delta):
                        last_factor_scaling=solver._factor_newton(safe_ratio)
                    cp.cuda.get_current_stream().synchronize()
                finally:
                    elapsed=time.perf_counter()-refresh_stamp
                    factor_seconds+=elapsed
                    iteration_retry_factor_seconds+=elapsed
                    factor_refresh_seconds+=elapsed
                    factor_refresh_count+=1
                lagged_factor=False
                last_factor_iteration=iteration
                with temporary_regularization(solver,last_factor_delta):
                    candidate,candidate_forcing,extra_iterations,extra_codes,extra_invalid=run_direction(
                        last_factor_scaling,refresh_request)
                # Treat a nonfinite old defect as +inf for finite-only selection.
                old_forcing=cp.where(cp.isfinite(forcing),forcing,cp.inf)
                answer,forcing,_selected=select_better_direction(answer,candidate,old_forcing,
                    candidate_forcing,refresh_request & ~extra_invalid,xp=cp)
                inner_total+=extra_iterations
                inner_codes=cp.where(refresh_request,extra_codes,inner_codes)
            while retry_cursor < len(iteration_retry_values):
                retry_request = (valid_request & cp.isfinite(forcing) & (forcing > eta)
                                 & cp.all(cp.isfinite(answer), axis=1))
                if not bool(cp.any(retry_request)):
                    break
                trial_delta = iteration_retry_values[retry_cursor]
                retry_cursor += 1
                target_retries += retry_request.astype(cp.int32)
                iteration_retry_count += retry_request.astype(cp.int32)
                with temporary_regularization(solver, trial_delta):
                    retry_factor_attempts += 1
                    retry_stamp = time.perf_counter()
                    try:
                        trial_scaling = solver._factor_newton(safe_ratio)
                        cp.cuda.get_current_stream().synchronize()
                    finally:
                        elapsed = time.perf_counter()-retry_stamp
                        factor_seconds += elapsed
                        retry_factor_seconds += elapsed
                        iteration_retry_factor_seconds += elapsed
                    last_factor_scaling = trial_scaling
                    last_factor_delta = trial_delta
                    last_factor_iteration = iteration
                    lagged_factor = False
                    candidate, candidate_forcing, extra_iterations, extra_codes, extra_invalid = run_direction(
                        trial_scaling, retry_request)
                answer, forcing, selected = select_better_direction(
                    answer, candidate, forcing, candidate_forcing, retry_request & ~extra_invalid, xp=cp)
                direction_delta = cp.where(selected, trial_delta, direction_delta)
                inner_total += extra_iterations
                inner_codes = cp.where(retry_request, extra_codes, inner_codes)
                retry_history.append(cp.stack((cp.full_like(mu, iteration), cp.full_like(mu, retry_cursor),
                    cp.full_like(mu, trial_delta), retry_request, candidate_forcing, forcing, selected,
                    cp.full_like(mu, role)), axis=1))
            direction, derivative, forcing = products(answer, rc, target)
            reliable = valid_request & cp.isfinite(forcing) & (forcing <= eta)
            code = cp.where(requested & ~valid_request, 1,
                            cp.where(requested & ~reliable, 2, 0)).astype(cp.int32)
            gmres_iterations += inner_total
            gmres_termination = cp.where(requested, inner_codes, gmres_termination)
            target_history.append(cp.stack((cp.full_like(mu, iteration), cp.full_like(mu, role),
                requested, valid_request, forcing, reliable, inner_total, inner_codes,
                target_retries, direction_delta), axis=1))
            return dict(answer=answer, rhs=rhs, forcing=forcing, direction=direction,
                        derivative=derivative, reliable=reliable, code=code,
                        delta=direction_delta,rc=rc)

        def checked_globalization(candidate,line_search):
            """All candidates use the untouched old iterate and actual merit."""
            reliable = candidate['reliable']
            trial_state, line = line_search(old_state, candidate['direction'],
                blocks, candidate['derivative'], weights, reliable, xp=cp)
            code = cp.where(reliable & ~line['descent'], 3, candidate['code'])
            code = cp.where(reliable & line['descent'] & ~line['taken'], 4, code)
            # Prediction is algebraically exact, not necessarily numerically
            # identical to the actual sparse matrix products. Verify before
            # any state is committed, including the centered fallback.
            if bool(cp.any(line['taken'])):
                actual_norm = _norm2(weighted_blocks(residuals(trial_state), weights, xp=cp), cp)
                actual_merit = .5*actual_norm*actual_norm
                verified = (line['taken'] & cp.isfinite(actual_merit)
                    & (actual_merit <= line['before'] + 1e-4*line['alpha']*line['slope']))
                code = cp.where(line['taken'] & ~verified, 6, code)
                trial_state = tuple(cp.where(verified[:, None], value, old)
                                    for value, old in zip(trial_state, old_state))
                line['after'] = cp.where(verified, actual_merit, line['before'])
                line['alpha'] = cp.where(verified, line['alpha'], 0.)
                line['taken'] = verified
                for name in ('alpha_primal','alpha_dual'):
                    if name in line:line[name]=cp.where(verified,line[name],0.)
            return dict(candidate, state=trial_state, line=line, code=code)

        def globalize(candidate):
            common=checked_globalization(candidate,backtrack_centered_step)
            if getattr(solver,'step_policy','common')=='common':return common
            separate=checked_globalization(candidate,backtrack_primal_dual_step)
            choose=(separate['line']['taken']&(~common['line']['taken']|
                (separate['line']['after']<common['line']['after'])))
            common['line']['alpha_primal']=common['line']['alpha']
            common['line']['alpha_dual']=common['line']['alpha']
            separate['state']=tuple(cp.where(choose[:,None],a,b)
                for a,b in zip(separate['state'],common['state']))
            separate['code']=cp.where(choose,separate['code'],common['code'])
            separate['line']={k:cp.where(choose,v,common['line'][k]) for k,v in separate['line'].items()}
            separate['line']['used_separate']=choose
            return separate

        try:
            if predictor_corrector:
                affine = solve_target(comp, active, 1)
                # Full forcing MUST qualify the affine direction before any
                # nonlinear predictor/cross product is constructed.
                corrector_rc, sigma, safe_affine, affine_info = guarded_corrector_rhs(
                    s, z, comp, mu, affine['direction'][3], affine['direction'][2],
                    affine['reliable'], xp=cp, affine_fraction=predictor_affine_fraction,
                    return_diagnostics=True)
                affine_history.append(cp.stack((cp.full_like(mu, iteration),
                    *(affine_info[key] for key in affine_columns)), axis=1))
                chosen = globalize(solve_target(corrector_rc, safe_affine, 2))
                pc_taken = chosen['line']['taken'].copy()
                fallback_request = active & ~pc_taken
                fallback_reason = cp.where(~affine['reliable'], 1,
                    cp.where(~safe_affine, 2, cp.where(~chosen['reliable'], 3,
                    cp.where(chosen['code'] == 3, 4,
                    cp.where(chosen['code'] == 4, 5, cp.where(chosen['code'] == 6, 6, 0))))))
                fallback_reason = cp.where(fallback_request, fallback_reason, 0)
                fallback_taken = cp.zeros_like(active)
                if bool(cp.any(fallback_request)):
                    fallback = globalize(solve_target(comp-.1*mu[:, None], fallback_request, 3))
                    fallback_taken = fallback['line']['taken']
                    # Preserve already accepted PC lanes. The other lanes get
                    # exactly ONE centered attempt from the same old state.
                    for key in ('answer', 'rhs','rc'):
                        chosen[key] = cp.where(fallback_request[:, None], fallback[key], chosen[key])
                    for key in ('forcing', 'reliable', 'code', 'delta'):
                        chosen[key] = cp.where(fallback_request, fallback[key], chosen[key])
                    chosen['state'] = tuple(cp.where(fallback_request[:, None], fv, cv)
                        for fv, cv in zip(fallback['state'], chosen['state']))
                    if getattr(solver,'centrality_corrections',0):
                        for key in ('direction','derivative'):
                            chosen[key]=tuple(cp.where(fallback_request[:,None],fv,cv)
                                for fv,cv in zip(fallback[key],chosen[key]))
                    chosen['line'] = {key: cp.where(fallback_request, fallback['line'][key], value)
                                      for key, value in chosen['line'].items()}
                pc_history.append(cp.stack((cp.full_like(mu, iteration), active,
                    affine['reliable'], safe_affine, sigma, pc_taken, fallback_request,
                    fallback_reason, fallback_taken), axis=1))
                # Multiple centrality proposals reuse the CURRENT factor at
                # the unchanged old iterate. Only full-Newton-qualified,
                # strictly positive, independently merit-verified improvements
                # replace the already usable PC/centered direction.
                mcc_active=active&chosen['line']['taken']&(chosen['line']['alpha']<.95)
                for correction_index in range(getattr(solver,'centrality_corrections',0)):
                    if not bool(cp.any(mcc_active)):break
                    from .gpu_centrality_corrector import centrality_rhs,composite_weight
                    proposed_rc,requested,aspiration=centrality_rhs(s,z,chosen['direction'][3],
                        chosen['direction'][2],sigma*mu,chosen['line']['alpha'],chosen['rc'],mcc_active,xp=cp)
                    extra=solve_target(proposed_rc,requested,4)
                    eligible=requested&extra['reliable']
                    omega=composite_weight(s,z,chosen['direction'][3],chosen['direction'][2],
                        extra['direction'][3],extra['direction'][2],eligible,xp=cp)
                    safe_answer=cp.where(eligible[:,None],extra['answer'],chosen['answer'])
                    safe_rc=cp.where(eligible[:,None],proposed_rc,chosen['rc'])
                    safe_rhs=cp.where(eligible[:,None],extra['rhs'],chosen['rhs'])
                    mixed_answer=chosen['answer']+omega[:,None]*(safe_answer-chosen['answer'])
                    mixed_rc=chosen['rc']+omega[:,None]*(safe_rc-chosen['rc'])
                    target=weighted_blocks((rd,rp,rg,mixed_rc),weights,xp=cp)
                    direction,derivative,forcing=products(mixed_answer,mixed_rc,target)
                    reliable=eligible&cp.isfinite(forcing)&(forcing<=eta)
                    candidate=globalize(dict(answer=mixed_answer,rc=mixed_rc,
                        rhs=chosen['rhs']+omega[:,None]*(safe_rhs-chosen['rhs']),
                        forcing=forcing,direction=direction,derivative=derivative,reliable=reliable,
                        code=cp.where(reliable,0,2),delta=extra['delta']))
                    improved=(candidate['line']['taken']&
                        (candidate['line']['alpha']>=chosen['line']['alpha']+.01)&
                        (candidate['line']['after']<=chosen['line']['after']))
                    centrality_history.append(cp.stack((cp.full_like(mu,iteration),
                        cp.full_like(mu,correction_index+1),requested,omega,
                        chosen['line']['alpha'],candidate['line']['alpha'],improved,forcing),axis=1))
                    for key in ('answer','rhs','rc'):
                        chosen[key]=cp.where(improved[:,None],candidate[key],chosen[key])
                    for key in ('forcing','reliable','code','delta'):
                        chosen[key]=cp.where(improved,candidate[key],chosen[key])
                    for key in ('state','direction','derivative'):
                        chosen[key]=tuple(cp.where(improved[:,None],new,old)
                            for new,old in zip(candidate[key],chosen[key]))
                    chosen['line']={key:cp.where(improved,candidate['line'][key],old)
                        for key,old in chosen['line'].items()}
                    mcc_active=improved&(chosen['line']['alpha']<.95)
            else:
                chosen = globalize(solve_target(comp-.1*mu[:, None], active, 0))
        except CudssError as error:
            failed = cp.where(active, 5, failed)
            numeric_error = str(error)
            cp.cuda.get_current_stream().synchronize()
            solve_seconds += time.perf_counter()-stamp-iteration_retry_factor_seconds
            break
        answer, rhs, forcing = chosen['answer'], chosen['rhs'], chosen['forcing']
        cached_factor_scaling=last_factor_scaling
        cached_factor_delta=last_factor_delta
        direction_delta, line = chosen['delta'], chosen['line']
        failed = cp.where(active, chosen['code'], failed)
        x, y, z, s = chosen['state']
        if getattr(solver,'certified_primal_extrapolation',False):
            from .gpu_primal_extrapolation import certified_extrapolation
            probe_started=time.perf_counter()
            # The affine direction is only an auxiliary terminal candidate;
            # unlike an IPM update, it is usable only if already certified.
            source=affine if predictor_corrector else chosen
            candidate,passed,probe_metrics=certified_extrapolation(solver,old_state[0],
                source['direction'][0],active&source['reliable'])
            passed&=~accepted
            extrapolation_history.append(dict(iteration=iteration,accepted=int(passed.sum()),
                seconds=time.perf_counter()-probe_started))
            if passed.any():
                selected=cp.asarray(passed)[:,None]
                x=cp.where(selected,candidate,x)
                # A certified terminal primal need not be strictly interior.
                # Export a positive slack PROPOSAL for the next LP, not a
                # claim that its internal y/z/s are certified multipliers.
                repair=cp.maximum(1e-12,solver.h-solver._mv(solver.g,x,solver.ng))
                s=cp.where(selected,repair,s)
                result_x=cp.where(selected,candidate,result_x)
                result_y=cp.where(selected,0.,result_y)
                metrics=[probe_metrics[i] if passed[i] else old for i,old in enumerate(metrics)]
                accepted|=passed
                accepted_iteration[passed]=iteration
                done=cp.asarray(accepted)
                checkpoints.append(dict(iteration=iteration,accepted=int(accepted.sum()),
                    source='certified_full_step_primal',metrics=list(metrics)))
        if 'alpha_primal' in line:
            primal_dual_step_history.append(cp.stack((line['alpha_primal'],line['alpha_dual'],line['used_separate']),axis=1))
        history.append(cp.stack((mu, cp.full_like(mu, eta), forcing, line['before'], line['after'],
                                  line['alpha'], line['backtracks'], failed, gmres_iterations,
                                  gmres_termination, iteration_retry_count, direction_delta), axis=1))
        newton_history.append(cp.stack((mu, line.get('alpha_primal',line['alpha']),
            line.get('alpha_dual',line['alpha']), forcing, forcing), axis=1))
        krylov_history.append(gmres_iterations)
        krylov_codes.append(gmres_termination)
        if capture_failure and solver.failure_snapshot is None and bool(cp.any(active & (failed != 0))):
            solver.failure_snapshot = {key: value.copy() for key, value in dict(rhs=rhs, answer=answer,
                iteration=cp.full_like(mu,iteration),forcing=forcing,active=active,
                residual=rhs-solver._kkt_mv(answer, safe_ratio, regularized=False),
                x=old_state[0], y=old_state[1], z=old_state[2], s=old_state[3],
                rd=rd, rp=rp, rg=rg, scaling=last_factor_scaling, kkt_values=solver.values, failed=failed,
                last_factor_regularization=cp.full_like(mu, last_factor_delta),
                selected_direction_regularization=direction_delta).items()}
        if bool(cp.any(line['taken'])):
            completed = iteration
            dirty = True
        cp.cuda.get_current_stream().synchronize()
        solve_seconds += time.perf_counter()-stamp-iteration_retry_factor_seconds
        if dirty and (iteration % check_interval == 0 or iteration == iterations
                      or not bool(cp.any(~done & (failed == 0)))):
            record_certificate(iteration)
    if dirty:
        record_certificate(completed)
    failed_host = failed.get()
    if accepted.all():
        status = 'certified'
    elif numeric_error:
        status = 'gpu_numeric_failure'
    elif np.any(failed_host):
        status = ('partial_failure' if np.any((~accepted) & (failed_host == 0))
                  else 'uncertified_failed_lanes')
    else:
        status = 'iteration_limit'
    # Include compact diagnostic downloads in reported wall time. They are not
    # device numerical work, but they are part of this measured solve API.
    newton_summary = _history(newton_history, cp)
    krylov_summary = _history(krylov_history, cp)
    krylov_code_summary = _history(krylov_codes, cp)
    globalization_summary = _history(history, cp)
    retry_summary = _history(retry_history, cp)
    target_summary = _history(target_history, cp)
    pc_summary = _history(pc_history, cp)
    initialization_summary = _history([initialization_diagnostics], cp)[0]
    device_solve_diagnostics = None
    if getattr(solver, 'device_checked_solves', False):
        if solver.factor.failed:
            device_solve_diagnostics = {'unavailable': 'Factor workspace failed; see numerical failure status'}
        else:
            device_solve_diagnostics = {
                key: value.get().tolist() if isinstance(value, cp.ndarray) else value
                for key, value in effective_factor.internal_diagnostics().items()}
    affine_summary = _history(affine_history, cp)
    if getattr(solver, 'retain_internal_state', False) and accepted.all():
        # Only retain GPU states whose original primal has passed. Export
        # rechecks the full original pair and never returns scratch aliases.
        solver._last_internal_state = tuple(v.copy() for v in (x, y, z, s))
    cp.cuda.get_current_stream().synchronize()
    return dict(x=result_x, y=result_y, metrics=metrics, accepted=accepted,
        accepted_iteration=accepted_iteration, status=status, iterations=attempted,
        completed_update_iteration=completed,
        checkpoints=checkpoints, cpu_lp_calls=0, analysis_count=solver.factor.analysis_count+
            (effective_factor.analysis_count if effective_factor is not solver.factor else 0),
        factor_count=effective_factor.factor_count-start_factor, solve_count=effective_factor.solve_count-start_solve,
        newton_backend=getattr(solver,'newton_backend','augmented'),
        shared_factor_group_size=getattr(solver,'shared_factor_group_size',1),
        shared_factor_active=getattr(solver,'_shared_factor_enabled',False),
        native_rhs_count=effective_factor.nrhs,
        centrality_corrections=getattr(solver,'centrality_corrections',0),
        step_policy=getattr(solver,'step_policy','common'),
        certified_primal_extrapolation=getattr(solver,'certified_primal_extrapolation',False),
        primal_extrapolation_diagnostics=extrapolation_history,
        primal_dual_step_diagnostics=_history(primal_dual_step_history,cp),
        primal_dual_step_diagnostic_columns=['alpha_primal','alpha_dual','separate_selected'],
        centrality_diagnostics=_history(centrality_history,cp),
        centrality_diagnostic_columns=['iteration','correction','requested','weight','old_step','new_step','taken','full_forcing'],
        schur_setup_seconds=getattr(getattr(solver,'_dual_schur',None),'setup_seconds',0.),
        factor_reuse_interval=factor_reuse_interval,factor_reuse_count=factor_reuse_count,
        factor_refresh_count=factor_refresh_count,factor_refresh_seconds=factor_refresh_seconds,
        factor_execution_layout=getattr(effective_factor,'execution_layout','uniform'),
        factor_ordering=getattr(solver.factor,'ordering','default'),
        factor_algorithm=getattr(solver.factor,'algorithm','default'),
        factor_native_precision=getattr(solver.factor,'precision','float64'),
        within_batch_face_proof_reuse_count=getattr(solver,'face_plan_reuse_count',0),
        native_factor_dimension=getattr(effective_factor,'_native_n',effective_factor.n),
        native_factor_batch=getattr(effective_factor,'_native_batch',solver.batch),
        fused_solve_guards=getattr(solver,'fused_solve_guards',False),
        factor_seconds=factor_seconds, triangular_solve_and_update_seconds=solve_seconds,
        certificate_seconds=certificate_seconds, total_seconds=time.perf_counter()-before,
        setup_seconds=solver.setup_seconds, kkt_dimension=solver.size, kkt_nnz=solver.factor.nnz,
        factored_dimension=effective_factor.n, factored_nnz=effective_factor.nnz,matrix_type=solver.matrix_type,
        regularization=solver.regularization, equilibration_rounds=solver.equilibration_rounds,
        newton_refinements=0, newton_relative_tolerance=solver.newton_relative_tolerance,
        original_newton_target=True, newton_krylov_iterations=solver.newton_krylov_iterations,
        newton_diagnostics=newton_summary,
        newton_diagnostic_columns=['mu', 'primal_step', 'dual_step', 'centered_full_forcing_ratio',
                                  'centered_full_forcing_ratio'],
        newton_error_definition='weighted full original nonlinear Newton defect 2-norm / target 2-norm',
        krylov_iterations_by_direction=krylov_summary,
        krylov_termination_by_direction=krylov_code_summary,
        krylov_coordinates=solver.krylov_coordinates,
        krylov_microkernels=getattr(solver,'krylov_microkernels','none'),
        krylov_defer_lane_checks=getattr(solver,'krylov_defer_lane_checks',False),
        reuse_gmres_workspace=getattr(solver,'reuse_gmres_workspace',False),
        gmres_workspace_statistics=[dict(width=w.shape[1],nbytes=w.nbytes,use_count=w.use_count)
            for w in getattr(solver,'_gmres_workspaces',{}).values()],
        device_checked_solves=getattr(solver,'device_checked_solves',False),
        factor_refinements=getattr(solver,'factor_refinements',2),
        equality_row_scaling=getattr(solver,'equality_row_scaling',False),
        internal_factor_device_diagnostics=device_solve_diagnostics,
        krylov_width=(solver.condensed_size if solver.krylov_coordinates=='condensed' else solver.size),
        krylov_termination_codes={0:'converged_or_not_needed', 1:'iteration_limit', 2:'breakdown', 3:'nonfinite'},
        globalized=True, forcing_eta=float(eta), centering_sigma=.1, backtrack_limit=12,
        predictor_corrector=predictor_corrector,
        ipm_initialization=ipm_initialization,
        internal_warm_start=(None if internal_warm_start is None else dict(internal_warm_start.metadata)),
        internal_state_retained=solver._last_internal_state is not None,
        initialization_diagnostics=initialization_summary,
        initialization_diagnostic_columns=['mu','s_min','s_max','z_min','z_max','z_over_s_min','z_over_s_max'],
        predictor_affine_fraction=float(predictor_affine_fraction),
        predictor_affine_diagnostics=affine_summary,
        predictor_affine_diagnostic_columns=['iteration', *affine_columns],
        predictor_affine_failure_codes={0:'safe', 1:'affine_not_eligible',
            2:'nonfinite_direction_or_mu', 3:'nonpositive_mu', 4:'affine_product_overflow_risk',
            5:'predicted_nonfinite', 6:'predicted_negative', 7:'predicted_product_overflow_risk',
            8:'corrector_nonfinite', 9:'sigma_ratio_unrepresentable'},
        predictor_affine_roundoff_scale='eps*abs(old)+eps*abs(alpha*direction); diagnostic only, no acceptance tolerance',
        direction_target_diagnostics=target_summary,
        direction_target_diagnostic_columns=['iteration', 'role', 'requested', 'valid_request',
            'full_forcing_ratio', 'reliable', 'krylov_iterations', 'krylov_code',
            'regularization_retries', 'selected_direction_delta'],
        direction_role_codes={0:'centered', 1:'affine', 2:'corrector', 3:'centered_fallback',4:'centrality'},
        direction_attempt_counts_by_role=direction_attempt_counts,
        triangular_solve_counts_by_role=direction_solve_counts,
        predictor_corrector_diagnostics=pc_summary,
        predictor_corrector_diagnostic_columns=['iteration', 'active', 'affine_reliable',
            'affine_product_safe', 'sigma', 'pc_step_taken', 'fallback_requested',
            'fallback_reason', 'fallback_step_taken'],
        predictor_corrector_fallback_codes={0:'not_needed', 1:'affine_forcing_or_nonfinite',
            2:'unsafe_affine_product', 3:'corrector_forcing_or_nonfinite', 4:'non_descent',
            5:'backtracking_exhausted', 6:'recomputed_merit_rejected'},
        globalization_diagnostics=globalization_summary,
        globalization_diagnostic_columns=['mu', 'eta', 'full_forcing_ratio', 'merit_before', 'merit_after',
                                          'step', 'backtracks', 'failure_code', 'krylov_iterations', 'krylov_code',
                                          'regularization_retries', 'selected_direction_delta'],
        regularization_retries=regularization_retries,
        regularization_schedule=regularization_schedule,
        regularization_schedule_history=regularization_schedule_history,
        regularization_retry_factor_attempts=retry_factor_attempts,
        regularization_retry_factor_seconds=retry_factor_seconds,
        regularization_retry_diagnostics=retry_summary,
        regularization_retry_diagnostic_columns=['iteration', 'retry', 'delta', 'requested',
                                                 'candidate_forcing', 'best_forcing', 'selected', 'role'],
        failed_environments=failed_host.tolist(), numeric_error=numeric_error,
        failure_codes={0:'not_failed', 1:'nonfinite_or_unrepresentable', 2:'inexact_direction_rejected',
                       3:'non_descent_direction', 4:'backtracking_exhausted', 5:'gpu_factor_or_solve_failed',
                       6:'recomputed_merit_rejected'},
        scope='Experimental globalized GPU Newton with unchanged original LP certificate; host setup/control remain')
