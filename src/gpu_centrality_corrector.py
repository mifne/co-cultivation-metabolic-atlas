"""Guarded multiple-centrality candidate RHS; never a convergence certificate.

Outlier correction and aspiration follow Colombo/Gondzio (2008), with the
existing solver's common primal/dual step and original merit gates retained.
"""


def centrality_rhs(s,z,ds,dz,mu,alpha,rc,eligible,*,xp):
    if s.ndim!=2 or any(v.shape!=s.shape for v in (z,ds,dz,rc)):
        raise ValueError('Matching complementarity block shapes required')
    if any(v.shape!=(s.shape[0],) for v in (mu,alpha,eligible)):
        raise ValueError('One target/step/eligibility per lane required')
    valid=(eligible&xp.isfinite(mu)&(mu>0.)&xp.isfinite(alpha)&(alpha>=0.)&(alpha<=1.)
        &xp.all(xp.isfinite(s)&xp.isfinite(z)&xp.isfinite(ds)&xp.isfinite(dz)&xp.isfinite(rc),axis=1))
    safe_s=xp.where(valid[:,None],s,1.);safe_z=xp.where(valid[:,None],z,1.)
    safe_ds=xp.where(valid[:,None],ds,0.);safe_dz=xp.where(valid[:,None],dz,0.)
    aspiration=xp.minimum(1.,1.5*xp.where(valid,alpha,0.)+.3)
    trial_s=safe_s+aspiration[:,None]*safe_ds
    trial_z=safe_z+aspiration[:,None]*safe_dz
    largest=xp.finfo(xp.float64).max
    valid&=xp.all(xp.isfinite(trial_s)&xp.isfinite(trial_z)&(
        xp.abs(trial_s)<=largest/xp.maximum(1.,xp.abs(trial_z))),axis=1)
    valid&=mu<=largest/10.
    trial=xp.where(valid[:,None],trial_s,0.)*xp.where(valid[:,None],trial_z,0.)
    target=xp.where(valid,mu,0.)[:,None]
    correction=xp.clip(trial,.1*target,10.*target)-trial
    proposed=xp.where(valid[:,None],rc,0.)-correction
    valid&=xp.all(xp.isfinite(proposed),axis=1)
    return xp.where(valid[:,None],proposed,0.),valid,aspiration


def composite_weight(s,z,base_ds,base_dz,new_ds,new_dz,eligible,*,xp):
    """Small device grid, chosen only by positive-step extent (not acceptance)."""
    weights=xp.asarray([1.,.5,.25],dtype=xp.float64)
    eligible=eligible&xp.all(xp.isfinite(new_ds)&xp.isfinite(new_dz),axis=1)
    new_ds=xp.where(eligible[:,None],new_ds,base_ds)
    new_dz=xp.where(eligible[:,None],new_dz,base_dz)
    ds=base_ds[:,None,:]+weights[None,:,None]*(new_ds-base_ds)[:,None,:]
    dz=base_dz[:,None,:]+weights[None,:,None]*(new_dz-base_dz)[:,None,:]
    limit=xp.ones((s.shape[0],3),dtype=xp.float64)
    for value,delta in ((s,ds),(z,dz)):
        ratio=-value[:,None,:]/xp.where(delta<0.,delta,-1.)
        limit=xp.minimum(limit,xp.min(xp.where(delta<0.,ratio,xp.inf),axis=2))
    finite=xp.all(xp.isfinite(ds)&xp.isfinite(dz),axis=2)
    limit=xp.where(eligible[:,None]&finite,limit,-xp.inf)
    best=xp.argmax(limit,axis=1)
    return xp.where(eligible,weights[best],0.)
