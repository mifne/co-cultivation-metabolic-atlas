"""Experimental damped Newton minimization of squared LP violations.

For a zero-cost auxiliary feasibility LP only. No barrier centrality path is
followed. Each direction solves the current generalized Hessian in FP64 via
the GPU Schur factor; Armijo checks the actual squared violation. A returned
point is a proposal, never a declaration of original optimality/infeasibility.
"""
import time


class GpuFeasibilityNewton:
    def __init__(self,solver):
        if solver._dual_schur is None:raise ValueError('Dual Schur workspace required')
        self.s=solver;self.cp=solver.cp

    def propose(self,x,iterations=30,damping=1e-6):
        s=self.s;cp=self.cp;h=s._dual_schur
        if type(iterations) is not int or iterations<1 or not 1e-12<=damping<=1.:
            raise ValueError('Positive iteration budget and finite damping required')
        if bool(cp.any(s.c!=0)):raise ValueError('Zero-cost auxiliary only; original gate required')
        if not isinstance(x,cp.ndarray) or x.shape!=(s.batch,s.n) or x.dtype!=cp.float64:
            raise ValueError('Exact FP64 current-coordinate primal proposal required')
        x=x.copy();tick=time.perf_counter();history=[]
        start_f=h.factor.factor_count;start_s=h.factor.solve_count
        h.refresh_current_operator()
        def residual(xx):
            e=s._mv(s.e,xx,s.ne)-s.b
            g=s._mv(s.g,xx,s.ng)-s.h
            v=cp.maximum(g,0.)
            return e,g,.5*(cp.sum(e*e,axis=1)+cp.sum(v*v,axis=1))
        for i in range(iterations):
            e,g,merit=residual(x)
            violation=cp.maximum(cp.max(cp.abs(e),axis=1),cp.max(cp.maximum(g,0.),axis=1))
            history.append([i,float(cp.max(violation))])
            if bool(cp.all(violation<=1e-6)):break
            active=(g>0.).astype(cp.float64)
            d=cp.full((s.batch,s.n),damping,dtype=cp.float64)
            d[:,s.il]+=active[:,s.q:s.q+len(s.il)]
            d[:,s.iu]+=active[:,s.q+len(s.il):]
            r=cp.concatenate((cp.ones((s.batch,s.ne)),cp.where(active[:,:s.q]>0,1.,1e12)),axis=1)
            h.factor_diagonal(d,r)
            gradient=s._mv(s.et,e,s.n)+s._mv(s.gt,cp.maximum(g,0.),s.n)
            rhs=cp.concatenate((-gradient,cp.zeros((s.batch,s.ne+s.q))),axis=1)
            dx=h.solve(rhs)[:,:s.n]
            slope=cp.sum(gradient*dx,axis=1)
            valid=cp.all(cp.isfinite(dx),axis=1)&(slope<0.)
            alpha=cp.ones(s.batch);taken=cp.zeros(s.batch,dtype=cp.bool_);next_x=x.copy()
            for _ in range(30):
                trial=x+alpha[:,None]*cp.where(valid[:,None],dx,0.)
                _,_,new_merit=residual(trial)
                good=valid&~taken&cp.isfinite(new_merit)&(new_merit<=merit+1e-4*alpha*slope)
                next_x=cp.where(good[:,None],trial,next_x);taken|=good
                if bool(cp.all(taken|~valid)):break
                alpha=cp.where(taken,alpha,alpha*.5)
            x=next_x
            if not bool(cp.any(taken)):break
        cp.cuda.get_current_stream().synchronize()
        return x,dict(seconds=time.perf_counter()-tick,history=history,
            factor_count=h.factor.factor_count-start_f,solve_count=h.factor.solve_count-start_s,
            cpu_lp_calls=0,original_certificate_required=True)
