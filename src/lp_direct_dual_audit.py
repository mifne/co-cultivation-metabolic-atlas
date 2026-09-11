"""Additional FP64 original-LP dual-objective audit, not a replacement gate.

For min c.T*x, the first ``neq`` rows are equalities and the remaining rows
are A*x <= rhs. Row duals use y_ineq <= 0 and r=c-A.T*y. The Lagrangian lower
bound is rhs.T*y + inf_{lo<=v<=hi} r.T*v. This module neither calls an
optimizer nor reads reference solutions nor changes a solver certificate.

The direct gap retains equality-multiplier-weighted primal residuals. Those
terms can be important when a small equality residual multiplies a large y.
These are ordinary FP64 diagnostics, NOT interval-arithmetic certificates.
Large cancellation is reported explicitly rather than claimed to be proved
accurate. All evaluation results remain NumPy/CuPy arrays of shape [batch].
"""
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr


class DirectDualAudit:
    """Prepare one owned LP snapshot; audit one or many candidate pairs.

    ``problem=(A_csr,rhs,lo,hi,c,neq)`` is host input. Preparation copies it to
    the requested arithmetic device once. evaluate accepts FP64 x/y arrays
    [n]/[m] or matching [B,n]/[B,m]; even single-pair results have shape [1].
    No evaluation-time vector is downloaded or evaluated by a CPU optimizer.
    """
    def __init__(self,problem,*,xp=np):
        if not isinstance(problem,(tuple,list)) or len(problem)!=6:
            raise ValueError('Expected original LP tuple (A_csr,rhs,lo,hi,c,neq)')
        a,rhs,lo,hi,c,neq=problem
        if not isspmatrix_csr(a):
            raise ValueError('Original coefficient matrix must be a SciPy CSR matrix')
        if (a.shape[1]<1 or not a.has_canonical_format or np.iscomplexobj(a.data)
                or not np.isfinite(a.data).all()
                or isinstance(neq,bool) or not isinstance(neq,(int,np.integer))
                or not 0<=neq<=a.shape[0]):
            raise ValueError('Finite canonical CSR and a valid equality count required')
        m,n=a.shape
        arrays=[]
        for value,shape in ((rhs,(m,)),(lo,(n,)),(hi,(n,)),(c,(n,))):
            if np.iscomplexobj(value):
                raise ValueError('Real-valued LP coefficients and bounds required')
            array=np.asarray(value,dtype=np.float64)
            if array.shape!=shape:
                raise ValueError('LP vectors must match the exact matrix dimensions')
            arrays.append(array.copy())
        rhs,lo,hi,c=arrays
        if (not np.isfinite(rhs).all() or not np.isfinite(c).all()
                or np.isnan(lo).any() or np.isnan(hi).any()
                or np.isposinf(lo).any() or np.isneginf(hi).any() or np.any(lo>hi)):
            raise ValueError('Finite RHS/cost and nonempty real-valued bound intervals required')
        a=csr_matrix(a,dtype=np.float64,copy=True)
        self.xp=xp
        self.m,self.n,self.neq=m,n,int(neq)
        self.device=None
        self.stream=None
        if xp is np:
            self.a=a
        else:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix as device_csr
            if xp is not cp:
                raise ValueError('Only NumPy and CuPy arithmetic backends are supported')
            self.device=cp.cuda.runtime.getDevice()
            self.stream=cp.cuda.get_current_stream().ptr
            self.a=device_csr(a)
        self.at=self.a.T.tocsr()
        self.rhs,self.lo,self.hi,self.c=[xp.asarray(v).copy() for v in arrays]
        self.finite_lo=xp.isfinite(self.lo)
        self.finite_hi=xp.isfinite(self.hi)
        # Missing bounds contribute zero only when their corresponding reduced
        # cost sign is zero. The missing-bound domain check remains separate.
        self.safe_lo=xp.where(self.finite_lo,self.lo,0.)
        self.safe_hi=xp.where(self.finite_hi,self.hi,0.)

    def _pair(self,x,y):
        xp=self.xp
        if self.device is not None and (xp.cuda.runtime.getDevice()!=self.device
                or xp.cuda.get_current_stream().ptr!=self.stream):
            raise ValueError('Audit must stay on its prepared CUDA device and stream')
        for value in (x,y):
            if not isinstance(value,xp.ndarray) or value.dtype!=xp.float64 or value.ndim not in (1,2):
                raise ValueError('Matching FP64 candidate arrays are required')
            if self.device is not None and value.device.id!=self.device:
                raise ValueError('Candidate arrays belong to a different CUDA device')
        if x.ndim!=y.ndim:
            raise ValueError('Primal and row-dual candidates must have matching batch dimensions')
        if x.ndim==1:x,y=x[None,:],y[None,:]
        if x.shape[1]!=self.n or y.shape!=(x.shape[0],self.m) or x.shape[0]<1:
            raise ValueError('Candidate arrays must match the exact LP dimensions')
        return x,y

    def evaluate(self,x,y):
        """Return additional numerical diagnostics, with no ``passed`` gate.

        ``dual_objective`` is the direct formula; it is -inf if a missing
        variable bound makes the Lagrangian infimum unbounded. When row dual
        signs are invalid its finite value is NOT a lower bound. Consult
        ``dual_feasible``/``dual_lower_bound`` before interpreting weak duality.
        ``relative_signed_gap`` divides by max(1,|primal|,|dual|); negative
        values are preserved, not hidden by an absolute-value acceptance test.
        """
        xp=self.xp
        x,y=self._pair(x,y)
        batch=x.shape[0]
        zero=xp.zeros(batch,dtype=xp.float64)

        def maximum(value):
            return xp.max(value,axis=1) if value.shape[1] else zero.copy()

        finite_x=xp.all(xp.isfinite(x),axis=1)
        finite_y=xp.all(xp.isfinite(y),axis=1)
        # Invalid input lanes are not allowed to contaminate batched sparse
        # arithmetic. Their final residual/gap diagnostics are fail-closed.
        sx=xp.where(finite_x[:,None],x,0.)
        sy=xp.where(finite_y[:,None],y,0.)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
            activity=(self.a@sx.T).T
            reduced=self.c[None,:]-(self.at@sy.T).T
            row_error=activity-self.rhs[None,:]
            positive=xp.maximum(reduced,0.)
            negative=xp.minimum(reduced,0.)
            lower_dual=xp.where(self.finite_lo[None,:],positive,0.)
            upper_dual=xp.where(self.finite_hi[None,:],-negative,0.)
            stationarity=reduced-lower_dual+upper_dual
            missing_lower=xp.where(self.finite_lo[None,:],0.,positive)
            missing_upper=xp.where(self.finite_hi[None,:],0.,-negative)
            domain_violation=maximum(xp.maximum(missing_lower,missing_upper))
            sign_violation=xp.maximum(0.,maximum(sy[:,self.neq:]))
            bound_terms=self.safe_lo[None,:]*positive+self.safe_hi[None,:]*negative
            row_terms=self.rhs[None,:]*sy
            primal_terms=self.c[None,:]*sx
            raw_primal=xp.sum(primal_terms,axis=1)
            raw_dual=xp.sum(row_terms,axis=1)+xp.sum(bound_terms,axis=1)
            finite_activity=xp.all(xp.isfinite(activity),axis=1)
            finite_reduced=xp.all(xp.isfinite(reduced),axis=1)
            dual_arithmetic=(finite_y&finite_reduced&xp.all(xp.isfinite(bound_terms),axis=1)
                &xp.all(xp.isfinite(row_terms),axis=1)&xp.isfinite(raw_dual))
            primal_arithmetic=finite_x&finite_activity&xp.isfinite(raw_primal)
            domain_valid=finite_reduced&(domain_violation==0.)
            dual_feasible=dual_arithmetic&domain_valid&(sign_violation==0.)
            primal_objective=xp.where(primal_arithmetic,raw_primal,xp.nan)
            dual_objective=xp.where(dual_arithmetic,
                xp.where(domain_valid,raw_dual,-xp.inf),xp.nan)
            dual_lower_bound=xp.where(dual_feasible,dual_objective,-xp.inf)
            objectives_finite=primal_arithmetic&dual_arithmetic&domain_valid
            gap_scale=xp.maximum(1.,xp.maximum(xp.abs(raw_primal),xp.abs(raw_dual)))
            raw_gap=raw_primal-raw_dual
            # Preserve a finite relative diagnostic when only the subtraction
            # of two opposite-sign finite objectives overflows.
            relative_gap=xp.where(xp.isfinite(raw_gap),raw_gap/gap_scale,
                                  raw_primal/gap_scale-raw_dual/gap_scale)
            signed_gap=xp.where(objectives_finite,raw_gap,xp.inf)
            relative_gap=xp.where(objectives_finite,relative_gap,xp.inf)
            equality_primal=maximum(xp.abs(row_error[:,:self.neq]))
            inequality_primal=xp.maximum(0.,maximum(row_error[:,self.neq:]))
            bound_primal=xp.maximum(0.,maximum(xp.maximum(
                xp.where(self.finite_lo[None,:],self.safe_lo[None,:]-sx,0.),
                xp.where(self.finite_hi[None,:],sx-self.safe_hi[None,:],0.))))
            primal_residual=xp.maximum(equality_primal,xp.maximum(inequality_primal,bound_primal))
            weighted_rows=sy*row_error
            equality_contribution=xp.sum(weighted_rows[:,:self.neq],axis=1)
            equality_absolute=xp.sum(xp.abs(weighted_rows[:,:self.neq]),axis=1)
            inequality_contribution=xp.sum(weighted_rows[:,self.neq:],axis=1)
            bound_gap=xp.sum(reduced*sx-bound_terms,axis=1)
            decomposition_gap=bound_gap+inequality_contribution+equality_contribution
            decomposition_difference=raw_gap-decomposition_gap
            primal_abs_terms=xp.sum(xp.abs(primal_terms),axis=1)
            dual_abs_terms=xp.sum(xp.abs(row_terms),axis=1)+xp.sum(xp.abs(bound_terms),axis=1)
            cancellation_ratio=dual_abs_terms/xp.maximum(1.,xp.abs(raw_dual))
            finite_all=(objectives_finite&xp.isfinite(raw_gap)&xp.isfinite(primal_residual)
                &xp.isfinite(equality_absolute)&xp.isfinite(decomposition_gap)
                &xp.isfinite(primal_abs_terms)&xp.isfinite(dual_abs_terms))

        def primal_diagnostic(value):
            return xp.where(primal_arithmetic&xp.isfinite(value),value,xp.inf)

        def pair_diagnostic(value):
            return xp.where(finite_x&finite_y&xp.isfinite(value),value,xp.inf)

        return dict(primal_objective=primal_objective,dual_objective=dual_objective,
            dual_lower_bound=dual_lower_bound,dual_feasible=dual_feasible,
            signed_gap=signed_gap,absolute_gap=xp.abs(signed_gap),
            relative_signed_gap=relative_gap,relative_absolute_gap=xp.abs(relative_gap),
            weak_duality_violation=xp.maximum(0.,-relative_gap),gap_scale=gap_scale,
            primal_residual=primal_diagnostic(primal_residual),
            equality_primal_residual=primal_diagnostic(equality_primal),
            inequality_primal_residual=primal_diagnostic(inequality_primal),
            bound_primal_residual=primal_diagnostic(bound_primal),
            dual_sign_violation=xp.where(dual_arithmetic,sign_violation,xp.inf),
            dual_bound_domain_violation=xp.where(dual_arithmetic,domain_violation,xp.inf),
            dual_stationarity_residual=xp.where(dual_arithmetic,maximum(xp.abs(stationarity)),xp.inf),
            equality_dual_residual_contribution=pair_diagnostic(equality_contribution),
            equality_dual_residual_absolute_sum=pair_diagnostic(equality_absolute),
            equality_dual_norm_inf=xp.where(finite_y,maximum(xp.abs(sy[:,:self.neq])),xp.inf),
            inequality_dual_residual_contribution=pair_diagnostic(inequality_contribution),
            bound_gap_contribution=pair_diagnostic(bound_gap),
            decomposition_gap=pair_diagnostic(decomposition_gap),
            gap_decomposition_difference=pair_diagnostic(decomposition_difference),
            primal_absolute_term_sum=pair_diagnostic(primal_abs_terms),
            dual_absolute_term_sum=pair_diagnostic(dual_abs_terms),
            dual_objective_cancellation_ratio=pair_diagnostic(cancellation_ratio),
            finite_inputs=finite_x&finite_y,finite_arithmetic=finite_all)


def prepare_direct_dual_audit(problem,*,xp=np):
    """Prepare one original LP without any candidate or reference solution."""
    return DirectDualAudit(problem,xp=xp)


def audit_direct_dual(problem,x,y,*,xp=np):
    """Convenience one-shot audit; cache prepare_direct_dual_audit for reuse."""
    return prepare_direct_dual_audit(problem,xp=xp).evaluate(x,y)
