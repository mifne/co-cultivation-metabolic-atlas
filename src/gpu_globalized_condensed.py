"""Condensed FGMRES proposal for the globalized full-forcing IPM path.

Bound duals are algebraically eliminated during Arnoldi. The numerator uses
the equivalent condensed stationarity/equality/complementarity defect; its
denominator remains the CURRENT FULL FOUR-BLOCK nonlinear target norm.
This helper never accepts a Newton update: the caller must expand and check
the actual full forcing, including roundoff-amplified bound complementarity.
"""
from .gpu_newton_krylov import batched_gmres,_norm2,GMRES_NONFINITE


def _relative_to_full_target_norm(residual,target_norm,*,xp):
    numerator=_norm2(residual,xp)
    ratio=numerator/xp.where(target_norm>0.,target_norm,1.)
    ratio=xp.where(target_norm>0.,ratio,xp.where(numerator==0.,0.,xp.inf))
    return xp.where(xp.isfinite(numerator)&xp.isfinite(target_norm),ratio,xp.inf)


def globalized_condensed_gmres(solver,rhs,answer,ratio,scaling,requested,
                               weights,target,eta,*,z):
    """Return ``(expanded_proposal, diagnostics, invalid_weight_mask)``.

    ``rhs/answer`` retain the full eliminated-Newton width n+ne+ng;
    ``target`` is weighted (rd,rp,rg,rc), width n+ne+2*ng. ``weights`` are the
    same four fixed per-LP block weights used by the caller's actual forcing
    test. z, not merely s/z, is required for complementarity defect units.

    For condensed u=(dx,dy,dz_q), use Wc=(wd,we,wcomp*z_q), Kc0 as target,
    and Kc_delta as a right preconditioner. Removed bound equations have zero
    residual only in exact arithmetic. Returned ``diagnostics['converged']``
    describes the condensed proxy, NOT actual full Newton or LP acceptance.
    The caller MUST recompute products() and require full_forcing <= eta.

    No optimizer, CPU vector transfer, LP change, factorization, or bound-dual
    reconstruction occurs inside Arnoldi. The existing factor and its matching
    scaling are reused. All inactive/invalid lanes preserve their input answer.
    """
    import math
    cp=solver.cp
    if not getattr(solver,'_condense_bounds',False):
        raise ValueError('Globalized condensed Krylov requires a bound-condensed solver')
    if not getattr(solver,'original_newton_target',False):
        raise ValueError('Globalized condensed Krylov must target original K0')
    if not math.isfinite(eta) or not 0.<eta<=.25:
        raise ValueError('Finite globalized forcing eta in (0,.25] required')
    solver.factor._context()
    batch=solver.batch
    full_target_width=solver.n+solver.ne+2*solver.ng

    def array(value,shape,*,dtype=cp.float64):
        if not isinstance(value,cp.ndarray) or value.shape!=shape or value.dtype!=dtype:
            raise ValueError('Exact-shape array on the solver arithmetic device required')
        if hasattr(cp,'cuda') and value.device.id!=solver.factor.device:
            raise ValueError('Wrong CUDA device for condensed globalized input')
        return value

    for value,shape in ((rhs,(batch,solver.size)),(answer,(batch,solver.size)),
                        (ratio,(batch,solver.ng)),(scaling,(batch,solver.condensed_size))):
        array(value,shape)
    array(requested,(batch,),dtype=cp.bool_)
    array(target,(batch,full_target_width))
    array(z,(batch,solver.ng))
    if not isinstance(weights,(tuple,list)) or len(weights)!=4:
        raise ValueError('Four original nonlinear block weights required')
    for weight in weights:array(weight,(batch,))
    finite_linear=(cp.all(cp.isfinite(rhs))&cp.all(cp.isfinite(answer))
        &cp.all(cp.isfinite(ratio)&(ratio>0.))
        &cp.all(cp.isfinite(scaling)&(scaling>0.)))
    if not bool(finite_linear):
        raise ValueError('Finite Newton inputs and positive ratio/scaling required')

    weight_valid=cp.ones(batch,dtype=cp.bool_)
    for weight in weights:weight_valid&=cp.isfinite(weight)&(weight>0.)
    weight_valid&=cp.all(cp.isfinite(z)&(z>0.),axis=1)
    # Compute the four-block norm once, not a different norm of packed b_c.
    full_target_norm=_norm2(cp.where(requested[:,None],target,0.),cp)
    weight_valid&=cp.isfinite(full_target_norm)
    linear_weight=cp.concatenate((
        cp.broadcast_to(weights[0][:,None],(batch,solver.n)),
        cp.broadcast_to(weights[1][:,None],(batch,solver.ne)),
        weights[3][:,None]*z[:,:solver.q]),axis=1).copy()
    weight_valid&=cp.all(cp.isfinite(linear_weight)&(linear_weight>0.),axis=1)
    invalid_weight=requested&~weight_valid
    active=requested&weight_valid
    linear_weight=cp.where(active[:,None],linear_weight,1.)
    full_target_norm=cp.where(active,full_target_norm,0.)
    packed=solver._pack_rhs(rhs,ratio)
    initial=cp.where(active[:,None],answer[:,:solver.condensed_size],0.)
    weighted_rhs=cp.where(active[:,None],linear_weight*packed,0.)

    improved,diagnostics=batched_gmres(weighted_rhs,initial,
        lambda value:linear_weight*solver._condensed_mv(value,ratio,regularized=False),
        lambda value:scaling*solver._factor_solve(
            (scaling*(value/linear_weight))[:,:,None])[:,:,0],
        lambda _rhs,residual:_relative_to_full_target_norm(residual,full_target_norm,xp=cp),
        xp=cp,max_iterations=solver.newton_krylov_iterations,tolerance=eta,
        microkernels=getattr(solver,'krylov_microkernels','none'),
        defer_lane_checks=getattr(solver,'krylov_defer_lane_checks',False),
        workspace=solver._gmres_workspace(solver.condensed_size) if getattr(solver,'reuse_gmres_workspace',False) else None)
    # Exactly one full reconstruction, after all Arnoldi iterations finish.
    expanded=solver._expand_direction(improved,rhs,ratio)
    expanded=cp.where(active[:,None],expanded,answer)
    diagnostics['termination']=cp.where(invalid_weight,GMRES_NONFINITE,diagnostics['termination'])
    diagnostics['nonfinite']=diagnostics['nonfinite']|invalid_weight
    diagnostics['converged']=diagnostics['converged']&~invalid_weight
    diagnostics.update(coordinates='condensed',krylov_width=solver.condensed_size,
        full_newton_width=solver.size,full_target_width=full_target_width,
        full_target_norm=full_target_norm,
        error_scope='condensed forcing proxy; actual full nonlinear forcing and LP certificate remain caller checks')
    return expanded,diagnostics,invalid_weight
