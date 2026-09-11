"""Development GPU Douglas-Rachford/Anderson box-optimum feasibility proposals.

An auxiliary box optimum is fixed only in proposal space. An approximate
affine projection (regularized normal equations) is NOT an LP certificate.
Every proposal needs full unchanged-original primal AND dual certification.
Failure does not establish original infeasibility. No CPU optimizer is used.
"""
import time
import numpy as np


class GpuDRFeasibility:
    def __init__(self,solver):
        self.solver=solver;self.cp=solver.cp
        self.helper=solver._dual_schur
        if self.helper is None:raise ValueError('Dual Schur workspace required')
        positions=np.flatnonzero(solver.problems[0][4])
        if len(positions)!=1 or solver.problems[0][4][positions[0]]>=0.:
            raise ValueError('Single negative objective coordinate required')
        self.objective=int(positions[0]);self.cost=solver.c.copy()

    def propose(self,initial=None,*,iterations=500,chunk=25,tolerance=1e-7,depth=8,interval=5):
        s=self.solver;cp=self.cp;helper=self.helper;s.factor._context()
        started=time.perf_counter()
        if (type(iterations) is not int or iterations<1 or type(chunk) is not int or chunk<1
                or type(depth) is not int or not 0<=depth<=20 or type(interval) is not int or interval<1
                or not np.isfinite(tolerance) or tolerance<=0.):
            raise ValueError('Positive DR budget/tolerance and bounded AA depth required')
        if not bool(cp.array_equal(s.c,self.cost)):raise ValueError('Objective changed')
        helper.refresh_current_operator()
        f0=helper.factor.factor_count;c0=helper.factor.solve_count
        d=cp.ones((s.batch,s.n),dtype=cp.float64)
        metric=cp.concatenate((cp.full((s.batch,s.ne),1e-12),cp.ones((s.batch,s.q))),axis=1)
        helper.factor_diagonal(d,metric)
        rhs=cp.concatenate((s.b,s.h[:,:s.q]),axis=1)
        lo=cp.full((s.batch,s.n),-cp.inf);hi=cp.full((s.batch,s.n),cp.inf)
        lo[:,s.il]=-s.h[:,s.q:s.q+len(s.il)]
        hi[:,s.iu]=s.h[:,s.q+len(s.il):]
        if len(s.fixed):lo[:,s.fixed]=hi[:,s.fixed]=s.b[:,s.neq:]
        if not bool(cp.all(cp.isfinite(hi[:,self.objective]))):raise ValueError('Finite optimizing bound required')
        lo[:,self.objective]=hi[:,self.objective]
        low=cp.concatenate((lo,cp.zeros((s.batch,s.q))),axis=1)
        high=cp.concatenate((hi,cp.full((s.batch,s.q),cp.inf)),axis=1)
        if initial is None:x=cp.clip(cp.zeros_like(lo),lo,hi)
        else:
            if (not isinstance(initial,cp.ndarray) or initial.shape!=(s.batch,s.n)
                    or initial.dtype!=cp.float64 or initial.device.id!=s.factor.device
                    or not bool(cp.all(cp.isfinite(initial)))):
                raise ValueError('Finite current-device initial primal required')
            x=cp.clip(initial,lo,hi)
        slack=cp.maximum(0.,rhs[:,s.ne:]-(helper.h@x.ravel()).reshape(s.batch,helper.m)[:,s.ne:])
        value=cp.concatenate((x,slack),axis=1)
        fs=[];gs=[];history=[];aa_count=0
        for iteration in range(1,iterations+1):
            point=cp.clip(value,low,high)
            reflected=2.*point-value
            defect=(helper.h@reflected[:,:s.n].ravel()).reshape(s.batch,helper.m)-rhs
            defect[:,s.ne:]+=reflected[:,s.n:]
            scaled=cp.ascontiguousarray((helper.scaling*defect)[:,:,None])
            dual=helper.scaling*helper.factor._solve_device_checked(scaled)[:,:,0]
            projected_x=reflected[:,:s.n]-(helper.ht@dual.ravel()).reshape(s.batch,s.n)
            projected_w=reflected[:,s.n:]-dual[:,s.ne:]
            result=value+cp.concatenate((projected_x,projected_w),axis=1)-point
            residual=result-value
            if depth and iteration%interval==0:
                fs.append(result);gs.append(residual)
                fs=fs[-depth-1:];gs=gs[-depth-1:]
                if len(fs)>1:
                    df=cp.stack([b-a for a,b in zip(fs[:-1],fs[1:])],axis=1)
                    dg=cp.stack([b-a for a,b in zip(gs[:-1],gs[1:])],axis=1)
                    scale=cp.maximum(1e-100,cp.maximum(cp.max(cp.abs(dg),axis=(1,2)),cp.max(cp.abs(residual),axis=1)))
                    u=dg/scale[:,None,None];v=residual/scale[:,None]
                    gram=u@u.transpose(0,2,1)
                    gram+=1e-8*cp.eye(len(fs)-1)[None]
                    target=(u@v[:,:,None])
                    valid=cp.all(cp.isfinite(gram),axis=(1,2))&cp.all(cp.isfinite(target),axis=(1,2))
                    gram=cp.where(valid[:,None,None],gram,cp.eye(len(fs)-1)[None])
                    target=cp.where(valid[:,None,None],target,0.)
                    gamma=cp.linalg.solve(gram,target)[:,:,0]
                    valid&=cp.all(cp.isfinite(gamma),axis=1)
                    gamma=cp.where(valid[:,None],gamma,0.)
                    correction=cp.sum(df*gamma[:,:,None],axis=1)
                    # Bounded extrapolation, not a claim of the paper's full
                    # safeguarding/convergence theorem for this prototype.
                    weight=cp.minimum(1.,10.*cp.max(cp.abs(residual),axis=1)/
                        cp.maximum(1e-100,cp.max(cp.abs(correction),axis=1)))
                    accelerated=result-weight[:,None]*correction
                    result=cp.where(cp.all(cp.isfinite(accelerated),axis=1)[:,None],accelerated,result)
                    aa_count+=1
            value=result
            if iteration%chunk==0 or iteration==iterations:
                x=cp.clip(value,low,high)[:,:s.n].copy()
                violation=cp.maximum(cp.max(cp.abs((s.e@x.ravel()).reshape(s.batch,s.ne)-s.b),axis=1),
                    cp.maximum(0.,cp.max((s.g@x.ravel()).reshape(s.batch,s.ng)-s.h,axis=1)))
                maximum=float(cp.max(violation));history.append([iteration,maximum])
                if not np.isfinite(maximum) or maximum<=tolerance:break
        cp.cuda.get_current_stream().synchronize()
        return x,dict(method='regularized_affine_box_DR_AA_proposal',iterations=iteration,
            violation_history=history,seconds=time.perf_counter()-started,aa_count=aa_count,
            factor_count=helper.factor.factor_count-f0,solve_count=helper.factor.solve_count-c0,
            cpu_lp_calls=0,original_certificate_required=True,auxiliary_failure_does_not_prove_infeasibility=True)
