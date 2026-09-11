"""Full-step primal proposals, accepted ONLY by the unchanged original LP gate.

Not an interior Newton update. Invalid full steps leave all solver state alone.
The cheap reduced residual filter is rejection-only, never a certificate.
"""
import numpy as np


def certified_extrapolation(solver,old_x,direction,eligible):
    cp=solver.cp
    solver.factor._context()
    for v in (old_x,direction):
        if (not isinstance(v,cp.ndarray) or v.dtype!=cp.float64
            or v.shape!=(solver.batch,solver.n) or v.device.id!=solver.factor.device):
            raise ValueError('Current-device FP64 primal coordinates required')
    if (not isinstance(eligible,cp.ndarray) or eligible.dtype!=cp.bool_
        or eligible.shape!=(solver.batch,) or eligible.device.id!=solver.factor.device):
        raise ValueError('Current-device boolean eligibility required')
    # Intersect the Newton ray with ALL current reduced inequalities and both
    # sides of the equalities. This differs from the positivity step for the
    # artificial IPM s/z: primal feasibility, not interior centrality, matters
    # to a terminal original-LP certificate. The band is only a proposal aid.
    # No feasibility interval is interpreted as original infeasibility.
    g0=solver._mv(solver.g,old_x,solver.ng)-solver.h
    gd=solver._mv(solver.g,direction,solver.ng)
    e0=solver._mv(solver.e,old_x,solver.ne)-solver.b
    ed=solver._mv(solver.e,direction,solver.ne)
    residual=cp.concatenate((g0,e0,-e0),axis=1)
    slope=cp.concatenate((gd,ed,-ed),axis=1)
    room=1e-7-residual
    quotient=room/cp.where(slope!=0.,slope,1.)
    lower=cp.maximum(0.,cp.max(cp.where(slope<0.,quotient,-cp.inf),axis=1))
    upper=cp.minimum(1.,cp.min(cp.where(slope>0.,quotient,cp.inf),axis=1))
    interval=(lower<=upper)&cp.all((slope!=0.)|(room>=0.),axis=1)
    interval&=cp.all(cp.isfinite(residual)&cp.isfinite(slope),axis=1)
    alpha=cp.where(interval,.5*(lower+upper),1.)
    candidate=old_x+alpha[:,None]*direction
    mask=eligible&interval&cp.all(cp.isfinite(candidate),axis=1)
    eq=solver._mv(solver.e,candidate,solver.ne)-solver.b
    iq=solver._mv(solver.g,candidate,solver.ng)-solver.h
    if solver.ne:mask&=cp.max(cp.abs(eq),axis=1)<=1e-6
    if solver.ng:mask&=cp.max(iq,axis=1)<=1e-6
    eligible_host=mask.get()
    if not eligible_host.any():
        return candidate,np.zeros(solver.batch,dtype=bool),None
    # In zero-face/forest workspaces this certificate includes all lifts and
    # the full, CURRENT original LP. y=0 is only a candidate dual: for a
    # non-box-optimal primal the gate must reject it.
    metrics=solver.certificate(candidate,cp.zeros((solver.batch,solver.m),dtype=cp.float64))
    passed=eligible_host&np.asarray([m['certificate_passed'] for m in metrics],dtype=bool)
    return candidate,passed,metrics
