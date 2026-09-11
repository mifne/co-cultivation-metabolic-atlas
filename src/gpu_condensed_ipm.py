"""Algebraic elimination of bound-dual rows from the GPU Newton system.

Every bound remains in the IPM residuals/complementarity and original LP gate.
Only the diagonal bound block is eliminated from each *linear equation*; its
direction is recovered on GPU. No active set guess, CPU LP, or model presolve.
"""
from .gpu_batched_ipm import GpuBatchedIPM


class GpuCondensedBatchedIPM(GpuBatchedIPM):
    _condense_bounds = True

    def _factor_newton(self,ratio):
        cp=self.cp
        bound_ratio=ratio[:,self.q:]
        diagonal=cp.full((self.batch,self.n),self.regularization,dtype=cp.float64)
        diagonal[:,self.il]+=1./bound_ratio[:,:len(self.il)]
        diagonal[:,self.iu]+=1./bound_ratio[:,len(self.il):]
        self.values[:,self.diagonal[:self.n]]=diagonal
        # Refresh delta here as well as the bound-condensed primal diagonal.
        self.values[:,self.diagonal[self.n:self.n+self.ne]]=-self.regularization
        self.values[:,self.diagonal[self.n+self.ne:]]=-ratio[:,:self.q]
        if getattr(self,'_dual_schur',None) is not None:
            return self._dual_schur.factor_newton(ratio)
        values,scaling=self._equilibrate_kkt()
        if getattr(self,'_shared_newton',None) is not None and self._shared_factor_enabled:
            self._shared_newton.factor_newton(values,getattr(self,'_current_factor_active',None))
        else:self.factor.factor(values)
        return scaling

    def _pack_rhs(self,rhs,ratio):
        """Eliminate bound-dual equations without changing the Newton target."""
        cp=self.cp
        bound_rhs=rhs[:,self.condensed_size:]
        divided=bound_rhs/ratio[:,self.q:]
        primal=rhs[:,:self.n].copy()
        primal[:,self.il]-=divided[:,:len(self.il)]
        primal[:,self.iu]+=divided[:,len(self.il):]
        return cp.concatenate((primal,rhs[:,self.n:self.condensed_size]),axis=1)

    def _expand_direction(self,answer,rhs,ratio):
        """Recover every eliminated bound multiplier, in original row order."""
        cp=self.cp
        dx=answer[:,:self.n]
        activity=cp.concatenate((-dx[:,self.il],dx[:,self.iu]),axis=1)
        bound_rhs=rhs[:,self.condensed_size:]
        bound_dual=(activity-bound_rhs)/ratio[:,self.q:]
        return cp.concatenate((answer,bound_dual),axis=1)

    def _solve_newton(self,rhs,ratio,scaling):
        packed=self._pack_rhs(rhs,ratio)
        answer=(self._dual_schur.solve(packed) if getattr(self,'_dual_schur',None) is not None else
                scaling*self._factor_solve((scaling*packed)[:,:,None])[:,:,0])
        return self._expand_direction(answer,rhs,ratio)

    def _condensed_mv(self,vector,ratio,*,regularized=False):
        """Apply the exact bound-condensed K_0 (or K_delta) on the device.

        Bound rows are diagonal contributions to the primal block. Arnoldi
        therefore stores and applies no bound-dual vectors. The original-row
        inequality operator is cached once, without a host numerical solve.
        """
        cp=self.cp
        dx=vector[:,:self.n]
        dy=vector[:,self.n:self.n+self.ne]
        dz=vector[:,self.n+self.ne:]
        reg=self.regularization if regularized else 0.
        primal=reg*dx+self._mv(self.et,dy,self.n)
        primal[:,self.il]+=dx[:,self.il]/ratio[:,self.q:self.q+len(self.il)]
        primal[:,self.iu]+=dx[:,self.iu]/ratio[:,self.q+len(self.il):]
        equality=self._mv(self.e,dx,self.ne)-reg*dy
        if self.q:
            if not hasattr(self,'_condensed_h'):
                rows=(cp.arange(self.batch)[:,None]*self.ng+cp.arange(self.q)).ravel()
                self._condensed_h=self.g[rows,:].tocsr()
                self._condensed_ht=self._condensed_h.T.tocsr()
            primal+=self._mv(self._condensed_ht,dz,self.n)
            inequality=self._mv(self._condensed_h,dx,self.q)-ratio[:,:self.q]*dz
        else:
            inequality=cp.empty((self.batch,0),dtype=cp.float64)
        return cp.concatenate((primal,equality,inequality),axis=1)

    def _krylov_direction(self,rhs,answer,ratio,scaling,need):
        """Optional condensed FGMRES; the reconstructed full K_0 stays final.

        The factor remains K_delta and is only a right preconditioner. Block
        weights use the *original full* RHS norms, not the possibly magnified
        condensed RHS. After Arnoldi, recompute all original equations,
        including eliminated bound rows, and never replace a better full
        direction with a worse condensed candidate. No tolerance is relaxed.
        """
        if getattr(self,'krylov_coordinates','full')!='condensed':
            return super()._krylov_direction(rhs,answer,ratio,scaling,need)
        from .gpu_newton_krylov import (batched_gmres,GMRES_CONVERGED,
                                       GMRES_ITERATION_LIMIT,GMRES_NONFINITE)
        cp=self.cp
        self.factor._context()
        if not self.original_newton_target:
            raise ValueError('Condensed Krylov must target the original Newton equations')
        for value,shape in ((rhs,(self.batch,self.size)),
                            (answer,(self.batch,self.size)),
                            (ratio,(self.batch,self.ng)),
                            (scaling,(self.batch,self.condensed_size))):
            self.factor._array(value,shape)
        if (not isinstance(need,cp.ndarray) or need.shape!=(self.batch,)
                or need.dtype!=cp.bool_):
            raise ValueError('Boolean active mask with exact batch shape required')
        if hasattr(cp,'cuda') and need.device.id!=self.factor.device:
            raise ValueError('Active mask must belong to the factor CUDA device')
        if not bool(cp.all(ratio>0.) & cp.all(scaling>0.)):
            raise ValueError('Strictly positive finite ratio and scaling required')

        packed=self._pack_rhs(rhs,ratio)
        weight=cp.ones_like(packed)
        for first,last,full_last in ((0,self.n,self.n),
                (self.n,self.n+self.ne,self.n+self.ne),
                (self.n+self.ne,self.condensed_size,self.size)):
            if first<last:
                weight[:,first:last]=1./cp.maximum(1.,
                    cp.max(cp.abs(rhs[:,first:full_last]),axis=1))[:,None]
        # Inactive environments are zero systems, preserving row independence.
        improved,diagnostic=batched_gmres(
            cp.where(need[:,None],weight*packed,0.),
            cp.where(need[:,None],answer[:,:self.condensed_size],0.),
            lambda v:weight*self._condensed_mv(v,ratio,regularized=False),
            lambda v:scaling*self._factor_solve((scaling*v/weight)[:,:,None])[:,:,0],
            lambda _rhs,residual:cp.max(cp.abs(residual),axis=1),xp=cp,
            max_iterations=self.newton_krylov_iterations,
            tolerance=self.newton_relative_tolerance,
            microkernels=getattr(self,'krylov_microkernels','none'),
            defer_lane_checks=getattr(self,'krylov_defer_lane_checks',False),
            workspace=self._gmres_workspace(self.condensed_size) if getattr(self,'reuse_gmres_workspace',False) else None)
        candidate=self._expand_direction(improved,rhs,ratio)
        full_error=self._direction_error(rhs,rhs-self._kkt_mv(candidate,ratio,regularized=False))
        previous_error=self._direction_error(rhs,rhs-self._kkt_mv(answer,ratio,regularized=False))
        finite=cp.all(cp.isfinite(candidate),axis=1)&cp.isfinite(full_error)
        better=need&finite&(full_error<previous_error)
        candidate=cp.where(better[:,None],candidate,answer)
        diagnostic['condensed_error']=diagnostic['error']
        diagnostic['condensed_converged']=diagnostic['converged']
        diagnostic['error']=cp.where(better,full_error,previous_error)
        diagnostic['converged']=cp.isfinite(diagnostic['error'])&(
            diagnostic['error']<=self.newton_relative_tolerance)
        diagnostic['termination']=cp.where(need&~finite,GMRES_NONFINITE,
            cp.where(diagnostic['converged'],GMRES_CONVERGED,
                cp.where(diagnostic['termination']==GMRES_CONVERGED,
                         GMRES_ITERATION_LIMIT,diagnostic['termination'])))
        diagnostic['nonfinite']=diagnostic['nonfinite']|(need&~finite)
        return candidate,diagnostic
