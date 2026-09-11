"""Experimental GPU revised bounded simplex with a compiled sparse inverse.

Anchor LP optimization/factorization is offline CPU work. Online bound and
matrix changes, pivot decisions, low-rank basis updates and numerical KKT
checks run on GPU. A fixed small pivot budget fails closed when insufficient.
This is not a production default or a fully resident dFBA environment.
"""
import numpy as np
from contextlib import nullcontext
from collections import OrderedDict
from scipy.sparse import csr_matrix, eye, hstack, vstack


class GpuRevisedBasis:
    @staticmethod
    def prepare_operator(anchor):
        """Reusable host assembly from an offline inverse; not factorization."""
        a=anchor['lp'].a;m,n=a.shape;active=anchor['active']
        inactive=np.setdiff1d(np.arange(m),active)
        inverse=vstack((hstack((anchor['inverse'],csr_matrix((len(active),len(inactive))))),
            hstack((a[inactive][:,anchor['basic']]@anchor['inverse'],-eye(len(inactive),format='csr')))),format='csr')
        inverse=inverse[:,np.argsort(np.r_[active,inactive])].tocsr()
        matrix=hstack((a,-eye(m,format='csr')),format='csr');columns=matrix.tocsc();columns.sum_duplicates()
        return dict(anchor=anchor,inactive=inactive,inverse=inverse,matrix=matrix,columns=columns,
            inverse_transpose=inverse.T.tocsr(),matrix_transpose=matrix.T.tocsr(),
            edge=np.maximum(np.asarray(inverse.multiply(inverse).sum(axis=1)).ravel(),1e-12))

    def __init__(self, anchor, variable_rows, max_pivots=24, primal_tolerance=1e-5,
                 dual_tolerance=1e-7, gap_tolerance=1e-7,capture_safe=False,adapter=None,
                 dual_edge='dantzig',pivot_refinement=0,compact_updates=False,prepared_operator=None,reuse_small_factor=False,skip_unused_dual=False,adaptive_refinement_tolerance=0.,graph_cache_size=4):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        from .gpu_certified_basis import GpuBasisInputAdapter
        if not isinstance(max_pivots,int) or max_pivots<1:
            raise ValueError("Positive pivot budget required")
        self.cp=cp; self.anchor=anchor; self.max_pivots=max_pivots
        if dual_edge not in ('dantzig','devex'):raise ValueError('Unknown dual edge strategy')
        self.dual_edge=dual_edge
        if pivot_refinement not in (0,1,2):raise ValueError('Invalid pivot refinement count')
        self.pivot_refinement=pivot_refinement
        self.compact_updates=bool(compact_updates)
        if reuse_small_factor and not capture_safe:raise ValueError('Factor reuse requires capture-safe math')
        self.reuse_small_factor=reuse_small_factor
        if skip_unused_dual and not capture_safe:raise ValueError('Device phase skipping requires capture-safe math')
        self.skip_unused_dual=skip_unused_dual;self.dual_conditional=None
        if not np.isfinite(adaptive_refinement_tolerance) or adaptive_refinement_tolerance<0:
            raise ValueError('Invalid internal refinement threshold')
        if adaptive_refinement_tolerance and not capture_safe:raise ValueError('Adaptive refinement requires capture-safe math')
        self.adaptive_refinement_tolerance=float(adaptive_refinement_tolerance);self.refine_conditional=None
        if adaptive_refinement_tolerance:
            from .gpu_conditional_capture import ConditionalCapture
            self.refine_conditional=ConditionalCapture(while_mode=False)
        if skip_unused_dual:
            from .gpu_conditional_capture import ConditionalCapture
            self.dual_conditional=ConditionalCapture(while_mode=False)
        if capture_safe:
            from .gpu_capture_math import CaptureMath
            self.math=CaptureMath()
        else:self.math=None
        self.graph_calls=OrderedDict()
        if not isinstance(graph_cache_size,int) or graph_cache_size<1:raise ValueError('Invalid graph cache size')
        self.graph_cache_size=graph_cache_size
        self.graph_compilation_seconds=0.
        if adapter is not None and (adapter.anchor is not anchor or
            not np.array_equal(adapter.variable_rows,np.asarray(sorted(set(variable_rows))))):
            raise ValueError('Shared adapter does not match the compiled basis')
        self.adapter=GpuBasisInputAdapter(anchor,variable_rows) if adapter is None else adapter
        self.primal_tolerance=primal_tolerance; self.dual_tolerance=dual_tolerance; self.gap_tolerance=gap_tolerance
        a=anchor["lp"].a; m,n=a.shape; self.m,self.n=m,n
        prepared=self.prepare_operator(anchor) if prepared_operator is None else prepared_operator
        if prepared['anchor'] is not anchor:raise ValueError('Prepared operator does not match anchor')
        inactive=prepared['inactive'];full_inverse=prepared['inverse'];full_matrix=prepared['matrix']
        self.inverse=gpu_csr(full_inverse)
        self.edge0=cp.asarray(prepared['edge'])
        self.inverse_transpose=gpu_csr(prepared['inverse_transpose'])
        self.matrix=gpu_csr(full_matrix)
        self.matrix_transpose=gpu_csr(prepared['matrix_transpose'])
        # Only entering/leaving columns are required; do not allocate the
        # ~500-700 MiB dense full matrix for every candidate solver.
        columns=prepared['columns']
        self.column_ptr=cp.asarray(columns.indptr,dtype=cp.int32)
        self.column_rows=cp.asarray(columns.indices,dtype=cp.int32)
        self.column_values=cp.asarray(columns.data)
        self.gather_column=cp.RawKernel(r'''
        extern "C" __global__ void gather_column(const int* ptr,const int* rows,
                const double* values,const long long* index,double* output,int m){
            int b=blockIdx.x,j=index[b];
            for(int k=ptr[j]+threadIdx.x;k<ptr[j+1];k+=blockDim.x)
                output[(long long)b*m+rows[k]]=values[k];
        }''','gather_column')
        self.gather_column.compile()
        self.basis0=cp.asarray(np.r_[anchor["basic"],n+inactive])
        self.kind0=cp.asarray(np.r_[anchor["kind"],anchor["row_kind"]])
        self.var=cp.asarray(self.adapter.variable_rows)
        self.initial_u=cp.asarray(full_inverse[:,self.adapter.variable_rows].toarray())
        self.cpu_lp_calls=0

    def prepare_host(self,problems):
        return self.adapter.prepare_host(problems)

    def clear_graph_cache(self):
        """Break solver/graph/warm-owner cycles at explicit cache eviction.

        Referenced warm-state arrays remain alive; a subsequent call may
        safely rebuild graphs using this solver's matrix and math handles.
        """
        calls=list(self.graph_calls.values())
        self.graph_calls.clear()
        for call in calls:call.close()
        restricted=getattr(self,'restricted_solver',None)
        if restricted is not None:
            restricted.clear_graph_cache();self.restricted_solver=None

    def run_device(self,**inputs):
        if self.math is None:return self._trim_updates(self.solve_device(**inputs))
        from .gpu_replay_call import GpuReplayCall,signature
        key=signature(inputs)
        if key not in self.graph_calls:
            while len(self.graph_calls)>=self.graph_cache_size:self.graph_calls.popitem(last=False)[1].close()
            call=GpuReplayCall(self,inputs)
            self.graph_calls[key]=call
            self.graph_compilation_seconds+=call.compilation_seconds
        self.graph_calls.move_to_end(key)
        return self._trim_updates(self.graph_calls[key].run(inputs))

    def _trim_updates(self,result):
        if not self.compact_updates:return result
        # The host reads a size, not a numerical decision or LP solution.
        # This measured synchronization bounds retained warm-state memory.
        warm=result['warm_state'];width=int(warm['used_rank'].max().get())
        trimmed=dict(warm,u=warm['u'][:,:,:width].copy(),update_rows=warm['update_rows'][:,:width].copy())
        return dict(result,warm_state=trimmed)

    def solve_device(self,*,rhs,lower,upper,c,delta,col_scale,row_scale,diagnostics=False,
                     warm_start=None,full_lower=None,full_upper=None,pivot_budget=None,
                     update_warm_matrix=False,conditional=None,_warm_iterations=None):
        cp=self.cp; batch=lower.shape[0]; m,n=self.m,self.n
        mm=(lambda a,b:a@b) if self.math is None else self.math.mm
        einsum=cp.einsum if self.math is None else self.math.einsum
        if conditional and (diagnostics or self.math is None):
            raise ValueError('Conditional capture requires capture-safe math and diagnostics=False')
        total=n+m; k0=len(self.var)
        iterations=self.max_pivots if pivot_budget is None else pivot_budget
        if not isinstance(iterations,int) or not 0<=iterations<=self.max_pivots:
            raise ValueError("Invalid per-call pivot budget")
        if warm_start is not None and (warm_start["owner"] is not self or warm_start["basis"].shape!=(batch,m)):
            raise ValueError("Warm state belongs to a different solver/batch")
        previous_rank=k0 if warm_start is None else warm_start["u"].shape[2]
        prior_rank=previous_rank
        if warm_start is not None and update_warm_matrix and not self.compact_updates:previous_rank+=k0
        rank=previous_rank+iterations
        active_rank=rank
        env=cp.arange(batch)
        bounds_lower=cp.concatenate((lower,cp.where(cp.arange(m)[None]<self.anchor["lp"].neq,rhs,-cp.inf)),axis=1)
        bounds_upper=cp.concatenate((upper,rhs),axis=1)
        if full_lower is not None:bounds_lower=full_lower
        if full_upper is not None:bounds_upper=full_upper
        scale=cp.concatenate((col_scale,row_scale),axis=1)
        cost=cp.concatenate((c,cp.zeros_like(rhs)),axis=1)
        basis=cp.broadcast_to(self.basis0,(batch,m)).copy() if warm_start is None else warm_start["basis"].copy()
        kind=cp.broadcast_to(self.kind0,(batch,total)).copy() if warm_start is None else warm_start["kind"].copy()
        u=cp.zeros((batch,m,rank));v=cp.zeros((batch,k0 if self.compact_updates else rank,m))
        if warm_start is None:
            u[:,:,:k0]=self.initial_u
            v[:,:k0]=delta[:,:,self.basis0.clip(max=n-1)]* (self.basis0<n)[None,None]
        else:
            u[:,:,:prior_rank]=warm_start["u"];v[:,:prior_rank]=warm_start["v"]
            if update_warm_matrix and self.compact_updates:
                # Keep the physical matrix update in the original k0 slots.
                # Unit-slot U columns also contain the OLD matrix's effect
                # on each changed basis column: update that effect exactly.
                # No extra slots accumulate just because time advanced.
                differences=delta-warm_start['delta']
                rows=warm_start['update_rows'][:,k0:prior_rank]
                old_columns=self.basis0[rows.clip(min=0)]
                new_columns=cp.take_along_axis(basis,rows.clip(min=0),axis=1)
                def changed_columns(columns):
                    return cp.take_along_axis(differences,cp.broadcast_to(columns[:,None,:].clip(max=n-1),
                        (batch,k0,columns.shape[1])),axis=2)*(columns<n)[:,None,:]
                correction=mm(self.initial_u,changed_columns(new_columns)-changed_columns(old_columns))
                u[:,:,k0:prior_rank]+=correction*(rows>=0)[:,None,:]
                v[:,:k0]=delta[:,:,self.basis0.clip(max=n-1)]*(self.basis0<n)[None,None]
            elif update_warm_matrix:
                # New row coefficients evaluated on the CURRENT basis. This
                # is an exact low-rank matrix update, not a reused stale flux.
                u[:,:,prior_rank:previous_rank]=self.initial_u
                differences=delta-warm_start['delta']
                v[:,prior_rank:previous_rank]=cp.take_along_axis(differences,
                    cp.broadcast_to(basis[:,None,:].clip(max=n-1),(batch,k0,m)),axis=2)*(basis<n)[:,None,:]
        phase=cp.zeros(batch,dtype=cp.int32)  # 0: primal merit; 1: primal; 2: original-cost dual
        stopped=cp.zeros(batch,dtype=bool)
        valid_input=cp.isfinite(rhs).all(axis=1)&cp.isfinite(c).all(axis=1)&cp.isfinite(delta).all(axis=(1,2))
        valid_input&=cp.isfinite(col_scale).all(axis=1)&(col_scale>0).all(axis=1)
        valid_input&=cp.isfinite(row_scale).all(axis=1)&(row_scale>0).all(axis=1)
        valid_input&=(lower<=upper).all(axis=1)&~cp.isnan(lower).any(axis=1)&~cp.isnan(upper).any(axis=1)
        valid_input&=~cp.isposinf(lower).any(axis=1)&~cp.isneginf(upper).any(axis=1)
        valid_input&=(bounds_lower<=bounds_upper).all(axis=1)&~cp.isnan(bounds_lower).any(axis=1)&~cp.isnan(bounds_upper).any(axis=1)
        valid_input&=~cp.isposinf(bounds_lower).any(axis=1)&~cp.isneginf(bounds_upper).any(axis=1)
        if warm_start is not None and not update_warm_matrix:
            valid_input&=(delta==warm_start["delta"]).all(axis=(1,2))&(rhs==warm_start["rhs"]).all(axis=1)
            valid_input&=(col_scale==warm_start["col_scale"]).all(axis=1)&(row_scale==warm_start["row_scale"]).all(axis=1)
        failed=~valid_input
        pivots=cp.zeros(batch,dtype=cp.int32)
        edge=cp.broadcast_to(self.edge0,(batch,m)).copy() if warm_start is None else warm_start.get(
            'edge_weights',cp.broadcast_to(self.edge0,(batch,m))).copy()
        counter=cp.zeros(1,dtype=cp.int32)
        if self.compact_updates:
            update_rows=cp.full((batch,rank),-2,dtype=cp.int32)
            update_rows[:,:k0]=-1
            update_slots=cp.full((batch,m),-1,dtype=cp.int32)
            used_rank=cp.full(batch,k0,dtype=cp.int32)
            if warm_start is not None:
                update_rows[:,:previous_rank]=warm_start['update_rows']
                update_slots[:]=warm_start['update_slots'];used_rank[:]=warm_start['used_rank']
        def linear_solve(a,b,transpose=False):
            if self.reuse_small_factor:return self.math.solve_factored(a,b,transpose=transpose)
            if transpose:a=a.transpose(0,2,1)
            if self.math is None:return cp.linalg.solve(a,b)
            # Unused update slots have exactly zero U and V, hence an exact
            # identity block in I+VU. GPU counter limits elimination, not
            # mathematical accuracy, without a host pivot-count transfer.
            prefix=cp.max(used_rank,keepdims=True) if self.compact_updates else cp.minimum(previous_rank+counter,rank)
            return self.math.solve(a,b,active_size=prefix)
        identity=cp.eye(rank)[None]
        trace=[];objective_trace=[]
        # Piecewise-linear Phase I minimizes actual bound infeasibility. A
        # fixed arbitrary bound-distance objective can spend hundreds of
        # degenerate dual pivots without approaching feasibility.
        feasibility_cost=cp.zeros_like(cost)

        def update_product():
            if not self.compact_updates:return mm(v[:,:active_rank],u[:,:,:active_rank])
            # The first k0 rows are physical matrix updates. Every later V
            # row is a unit vector at a changed basis position, or exactly 0.
            # Gather instead of multiplying a mostly-zero (rank x m) matrix.
            out=cp.zeros((batch,active_rank,active_rank))
            out[:,:k0]=mm(v[:,:k0],u[:,:,:active_rank])
            rows=update_rows[:,k0:active_rank]
            out[:,k0:]=cp.take_along_axis(u[:,:,:active_rank],rows.clip(min=0)[:,:,None],axis=1)*(rows>=0)[:,:,None]
            return out

        def apply_v(value,transpose=False):
            if not self.compact_updates:
                return einsum('brm,br->bm' if transpose else 'brm,bm->br',v[:,:active_rank],value)
            rows=update_rows[:,k0:active_rank];ids=rows.clip(min=0)
            if transpose:
                out=einsum('brm,br->bm',v[:,:k0],value[:,:k0])
                cp.add.at(out,(env[:,None],ids),cp.where(rows>=0,value[:,k0:active_rank],0.))
            else:
                out=cp.zeros((batch,active_rank))
                out[:,:k0]=einsum('brm,bm->br',v[:,:k0],value)
                out[:,k0:]=cp.take_along_axis(value,ids,axis=1)*(rows>=0)
            return out

        def small_system():
            small=identity[:,:active_rank,:active_rank]+update_product()
            if not self.reuse_small_factor:return small
            prefix=cp.max(used_rank,keepdims=True) if self.compact_updates else cp.minimum(previous_rank+counter,rank)
            return self.math.factor(small,active_size=prefix)

        def matvec(x):
            value=mm(self.matrix,x.T).T
            value[:,self.var]+=einsum("bkn,bn->bk",delta,x[:,:n])
            return value

        def transvec(y):
            value=mm(self.matrix_transpose,y.T).T
            value[:,:n]+=einsum("bkn,bk->bn",delta,y[:,self.var])
            return value

        def inverse_apply(value,small,transpose=False):
            active_u=u[:,:,:active_rank];active_v=v[:,:active_rank]
            if transpose:
                initial=mm(self.inverse_transpose,value.T).T
                if active_rank==0:return initial
                q=linear_solve(small,einsum("bmr,bm->br",active_u,value)[:,:,None],transpose=True)[:,:,0]
                return initial-mm(self.inverse_transpose,apply_v(q,True).T).T
            initial=mm(self.inverse,value.T).T
            if active_rank==0:return initial
            q=linear_solve(small,apply_v(initial)[:,:,None])[:,:,0]
            return initial-einsum("bmr,br->bm",active_u,q)

        def refinement_scope(error):
            if not conditional or not self.adaptive_refinement_tolerance:return nullcontext()
            # This is an internal arithmetic trigger, not an acceptance
            # tolerance. The complete original-LP certificate is unchanged.
            needed=cp.max(cp.abs(error),axis=1)>self.adaptive_refinement_tolerance
            return self.refine_conditional.iteration(~needed,cp.zeros(batch,dtype=bool))

        def primal_state(small):
            x=cp.where(kind==-1,bounds_lower,cp.where(kind==1,bounds_upper,0.))
            x=cp.where(cp.isfinite(x),x,0.)
            x[env[:,None],basis]=inverse_apply(-matvec(x),small)
            for _ in range(self.pivot_refinement):
                error=-matvec(x)
                with refinement_scope(error):x[env[:,None],basis]+=inverse_apply(error,small)
            return x

        def state(small,objective):
            x=primal_state(small)
            y=inverse_apply(objective[env[:,None],basis],small,True)
            reduced=objective-transvec(y)
            return x,y,reduced

        def column(index):
            result=cp.zeros((batch,m))
            self.gather_column((batch,),(128,),(self.column_ptr,self.column_rows,self.column_values,
                cp.ascontiguousarray(index,dtype=cp.int64),result,np.int32(m)))
            addition=delta[env[:,None],cp.arange(k0)[None],cp.minimum(index,n-1)[:,None]]
            result[:,self.var]+=cp.where((index<n)[:,None],addition,0.)
            return result

        if iterations:
            active_rank=rank if self.math is not None or self.compact_updates else previous_rank
            small=small_system()
            initial_y=inverse_apply(cost[env[:,None],basis],small,True)
            initial_rc=cost-transvec(initial_y)
            sign_bad=cp.where(kind==-1,-initial_rc,cp.where(kind==1,initial_rc,cp.abs(initial_rc)))
            sign_bad=cp.where((kind!=2)&(bounds_upper-bounds_lower>1e-12),sign_bad,0.)
            dual_feasible=cp.max(sign_bad*scale,axis=1)<=1e-8
            phase[:]=cp.where(dual_feasible,2,0)
        loop_count=(min(iterations,1) if conditional and conditional.while_mode else iterations)
        if _warm_iterations is not None:loop_count=min(loop_count,_warm_iterations)
        for iteration in range(loop_count):
            active_rank=rank if self.math is not None or self.compact_updates else previous_rank+iteration
            with (conditional.iteration(stopped,failed,counter,iterations) if conditional else nullcontext()):
                small=small_system()
                # Phase selection below needs only primal feasibility. The
                # old dual solve was immediately overwritten after selecting
                # the phase objective; remove that redundant inverse/SpMM.
                x=primal_state(small)
                xb=x[env[:,None],basis]; lb=bounds_lower[env[:,None],basis]; ub=bounds_upper[env[:,None],basis]
                bs=scale[env[:,None],basis]
                violations=cp.maximum(lb-xb,xb-ub)/bs
                row=cp.argmax(violations,axis=1)
                if self.dual_edge=='devex':
                    # Devex-style approximate edge pricing, not exact DSE.
                    # Pricing may change the path, never the final gates.
                    weighted=cp.maximum(lb-xb,xb-ub)/cp.sqrt(cp.maximum(edge,1e-12))
                    row=cp.where(phase==2,cp.argmax(weighted,axis=1),row)
                primal_bad=cp.max(violations,axis=1)>1e-8
                switch=(phase!=1)&~primal_bad&~failed
                phase[:]=cp.where(switch,1,phase)
                # Refresh original objective immediately when feasibility is found.
                feasibility_cost=cp.where(x<bounds_lower-1e-9*scale,-1./scale,
                    cp.where(x>bounds_upper+1e-9*scale,1./scale,0.))
                objective=cp.where(phase[:,None]!=0,cost,feasibility_cost)
                y=inverse_apply(objective[env[:,None],basis],small,True)
                for _ in range(self.pivot_refinement):
                    error=(objective-transvec(y))[env[:,None],basis]
                    with refinement_scope(error):y+=inverse_apply(error,small,True)
                reduced=objective-transvec(y)
                direction=cp.where(kind==-1,1.,cp.where(kind==1,-1.,cp.where(reduced<0,1.,-1.)))
                movable=(kind!=2)&(bounds_upper-bounds_lower>1e-12)
                score=cp.where(movable,-reduced*direction,-cp.inf)
                enter_primal=cp.argmax(score,axis=1)
                optimal=(phase==1)&~primal_bad&(cp.max(score,axis=1)<=1e-8)
                stopped|=optimal
                use_dual=(phase==2)&primal_bad
                merit_phase=phase==0
    
                # Dual simplex: repair a violating basic bound. With objective zero
                # all reduced costs are dual-feasible; then primal simplex optimizes.
                sign=cp.where(xb[env,row]<lb[env,row],1.,-1.)
                e=cp.zeros((batch,m)); e[env,row]=1.
                # These buffers must exist outside the conditional body.
                # No active dual-phase row => no dual ratio test is needed.
                # GPU IF is nested in the GPU WHILE, with no host readback.
                inverse_row=cp.zeros((batch,m));dual_row=cp.zeros_like(cost)
                with (self.dual_conditional.iteration(~use_dual,stopped|failed)
                      if conditional and self.skip_unused_dual else nullcontext()):
                    inverse_row[:]=inverse_apply(e,small,True)
                    for _ in range(self.pivot_refinement):
                        error=e-transvec(inverse_row)[env[:,None],basis]
                        with refinement_scope(error):inverse_row+=inverse_apply(error,small,True)
                    dual_row[:]=transvec(inverse_row)
                dual_direction=cp.where(kind==-1,1.,cp.where(kind==1,-1.,cp.where(-sign[:,None]*dual_row>0,1.,-1.)))
                alpha=-sign[:,None]*dual_row*dual_direction
                eligible=movable&(alpha>1e-10)
                ratio=cp.where(eligible,cp.maximum(reduced*dual_direction,0.)/cp.maximum(alpha,1e-300),cp.inf)
                minimum=cp.min(ratio,axis=1)
                eligible&=ratio<=minimum[:,None]+1e-10
                enter_dual=cp.argmax(cp.where(eligible,alpha,-1.),axis=1)
                no_dual=~cp.isfinite(minimum)
                enter=cp.where(use_dual,enter_dual,enter_primal)
                direction_enter=cp.where(use_dual,dual_direction[env,enter],direction[env,enter])
                col=column(enter)
                transformed=inverse_apply(col,small)
                for _ in range(self.pivot_refinement):
                    trial=cp.zeros_like(cost);trial[env[:,None],basis]=transformed
                    error=col-matvec(trial)
                    with refinement_scope(error):transformed+=inverse_apply(error,small)
                movement=transformed*direction_enter[:,None]
                distance=cp.where(movement>0,cp.maximum(xb-lb,0.),cp.maximum(ub-xb,0.))
                # Phase I may start outside bounds. Its next event is entering
                # the interval, not a spurious zero-length leaving-bound pivot.
                below=xb<lb-1e-9*bs;above=xb>ub+1e-9*bs
                merit_distance=cp.where(below,cp.where(movement<0,lb-xb,cp.inf),
                    cp.where(above,cp.where(movement>0,xb-ub,cp.inf),distance))
                distance=cp.where(merit_phase[:,None],merit_distance,distance)
                ratio_p=cp.where(cp.abs(movement)>1e-10,distance/cp.maximum(cp.abs(movement),1e-300),cp.inf)
                relaxed=cp.where(cp.abs(movement)>1e-10,(distance+1e-9)/cp.maximum(cp.abs(movement),1e-300),cp.inf)
                limiting=ratio_p<=cp.min(relaxed,axis=1)[:,None]
                row_primal=cp.argmax(cp.where(limiting,cp.abs(transformed),-1.),axis=1)
                basic_step=ratio_p[env,row_primal]
                own_step=cp.where(direction_enter>0,bounds_upper[env,enter]-x[env,enter],x[env,enter]-bounds_lower[env,enter])
                flip=~use_dual&(own_step<basic_step)
                row=cp.where(use_dual,row,row_primal)
                leaving=basis[env,row]
                pivot=transformed[env,row]
                active=~stopped&~failed
                bad=active&((use_dual&no_dual)|(~use_dual&~cp.isfinite(cp.minimum(basic_step,own_step)))|
                            (merit_phase&(cp.max(score,axis=1)<=1e-10))|
                            (~flip&((cp.abs(pivot)<1e-12)|~cp.isfinite(pivot))))
                failed|=bad; active&=~bad
                do_pivot=active&~flip
                if self.dual_edge=='devex':
                    exact_row_weight=cp.sum(inverse_row*inverse_row,axis=1)
                    safe_pivot=cp.where(cp.abs(pivot)>1e-12,pivot,1.)
                    updated=cp.maximum(edge,exact_row_weight[:,None]*(transformed/safe_pivot[:,None])**2)
                    updated[env,row]=exact_row_weight/(safe_pivot**2)
                    edge[:]=cp.where((do_pivot&use_dual)[:,None],updated,edge)
                event_kind=cp.where(merit_phase&below[env,row],-1,
                    cp.where(merit_phase&above[env,row],1,cp.where(movement[env,row]>0,-1,1)))
                leaving_kind=cp.where(use_dual,cp.where(sign>0,-1,1),event_kind)
                kind[env,leaving]=cp.where(do_pivot,leaving_kind,kind[env,leaving])
                kind[env,enter]=cp.where(do_pivot,2,cp.where(active&flip,-kind[env,enter],kind[env,enter]))
                delta_col=col-column(leaving)
                update=mm(self.inverse,delta_col.T).T
                if self.compact_updates:
                    old_slot=update_slots[env,row]
                    allocate=do_pivot&(old_slot<0)
                    slot=cp.where(old_slot>=0,old_slot,used_rank).clip(max=rank-1)
                    # Repeated pivots at the same basis position telescope:
                    # sum B0^-1(a_new-a_old) = B0^-1(a_current-a_initial).
                    u[env,:,slot]+=cp.where(do_pivot[:,None],update,0.)
                    update_rows[env,slot]=cp.where(do_pivot,row,update_rows[env,slot])
                    update_slots[env,row]=cp.where(do_pivot,slot,old_slot)
                    used_rank+=allocate.astype(cp.int32)
                elif self.math is not None:
                    position=previous_rank+counter
                    u[:,:,position]=cp.where(do_pivot[:,None,None],update[:,:,None],0.)
                    v[:,position,:]=0.;v[env,position,row]=do_pivot.astype(cp.float64)
                else:
                    u[:,:,previous_rank+iteration]=cp.where(do_pivot[:,None],update,0.)
                    v[:,previous_rank+iteration,:]=0.; v[env,previous_rank+iteration,row]=do_pivot.astype(cp.float64)
                basis[env,row]=cp.where(do_pivot,enter,leaving)
                pivots+=active.astype(cp.int32)
                counter+=1
                if diagnostics:
                    objective_trace.append(cp.sum(cost*x,axis=1))
                    trace.append(cp.stack((cp.max(violations,axis=1),cp.sum(cp.maximum(violations,0.),axis=1),
                        row,enter,leaving,pivot,phase,stopped,failed),axis=1))
    
        active_rank=rank
        small=small_system()
        x,y,reduced=state(small,cost)
        # Refine using the original full matrix, not just the stored inverse.
        for _ in range(2):
            x[env[:,None],basis]+=inverse_apply(-matvec(x),small)
            y+=inverse_apply((cost-transvec(y))[env[:,None],basis],small,True)
        reduced=cost-transvec(y)
        equation=cp.max(cp.abs(matvec(x))/row_scale,axis=1)
        bounds=cp.max(cp.maximum(bounds_lower-x,x-bounds_upper)/scale,axis=1)
        primal=cp.maximum(equation,cp.maximum(bounds,0.))
        rc=reduced*scale
        dual=cp.maximum(cp.max(cp.where(~cp.isfinite(bounds_lower),rc,0.),axis=1),
                        cp.max(cp.where(~cp.isfinite(bounds_upper),-rc,0.),axis=1))
        selected=cp.where(reduced>=0,bounds_lower,bounds_upper)
        gap=cp.sum(cp.abs(reduced*(x-cp.where(cp.isfinite(selected),selected,x))),axis=1)
        value=cp.sum(cost*x,axis=1); gap/=cp.maximum(1.,cp.abs(value))
        accepted=~failed&cp.isfinite(x).all(axis=1)&cp.isfinite(y).all(axis=1)&cp.isfinite(gap)
        accepted&=(primal<=self.primal_tolerance)&(dual<=self.dual_tolerance)&(gap<=self.gap_tolerance)
        accepted&=(lower<=upper).all(axis=1)&(col_scale>0).all(axis=1)&(row_scale>0).all(axis=1)
        result=dict(accepted=accepted,values=cp.where(accepted[:,None],x[:,:n]/col_scale,cp.nan),
            objective=cp.where(accepted,value,cp.nan),primal_residual=primal,dual_violation=dual,
            relative_kkt_gap=gap,pivots=pivots,cpu_lp_calls=0,
            scope="GPU revised basis with numerical KKT checks; finite pivot budget")
        result["warm_state"]=dict(owner=self,basis=basis,kind=kind,u=u,v=v,delta=delta,rhs=rhs,
            edge_weights=edge,
            col_scale=col_scale,row_scale=row_scale,lower=bounds_lower,upper=bounds_upper,
            x=x,y=y,reduced=reduced,scale=scale,cost=cost)
        if self.compact_updates:
            result['warm_state'].update(update_rows=update_rows,update_slots=update_slots,used_rank=used_rank)
            result['unique_update_columns']=used_rank-k0
        if diagnostics:
            result.update(trace=cp.stack(trace) if trace else cp.empty((0,batch,9)),
                objective_trace=cp.stack(objective_trace) if objective_trace else cp.empty((0,batch)),
                basis=basis,raw_values=x,raw_kind=kind,raw_y=y)
        return result
