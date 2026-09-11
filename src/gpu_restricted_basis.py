"""GPU restricted-column LPs with exact lifting and original-LP certification.

Choose K nonbasic directions, eliminate the M basic variables algebraically,
and pivot an M x K tableau. The expensive full sparse inverse is used for
projection/lifting, NOT at every restricted pivot. Restricting columns may
miss feasibility/optimality: only the full original-LP certificate accepts.
"""
from collections import OrderedDict
from contextlib import nullcontext
import numpy as np


def restricted_basis_capacity(previous_rank, physical_rank, rows, columns,
                              direction_count, max_pivots, pivot_aware=False):
    """Power-of-two storage bound, including immutable physical-update slots.

    A restricted pivot exchanges at most one basic variable; a bound flip
    exchanges none. After P pivots, the root basis therefore misses at most
    its previous missing count plus P variables. The restricted LP can also
    introduce only its K initially selected nonbasics: later pivots exchange
    variables within that same M+K set. Thus the increase is <= min(K, P),
    including continuation with an entirely different selected-column set.
    ``previous_rank`` must bound every row's retained used_rank (not truncate
    it). The basis has M slots and its complement has N variables, providing
    the independent min(M, N) bound on root-basis replacements.
    """
    # Keep one inert slot for a zero-pivot budget: downstream kernels and
    # the explicit capacity guard require a nonempty replacement dimension.
    extra=max(1,min(direction_count,max_pivots)) if pivot_aware else direction_count
    return min(physical_rank+rows,physical_rank+columns,
        1<<(max(1,previous_rank+extra)-1).bit_length())


class GpuRestrictedBasis:
    def __init__(self,base,max_pivots=256,rank_bucket=False,pivot_aware_capacity=False):
        if not base.compact_updates or base.math is None:
            raise ValueError('Restricted solver needs compact capture-safe base')
        self.base=base;self.cp=base.cp;self.math=base.math;self.max_pivots=max_pivots
        self.rank_bucket=rank_bucket
        self.pivot_aware_capacity=bool(pivot_aware_capacity)
        from .gpu_conditional_capture import ConditionalCapture
        self.pricing_conditional=ConditionalCapture(while_mode=False) if rank_bucket else None
        cp=self.cp;m,n=base.m,base.n
        host_kind=np.r_[base.anchor['kind'],base.anchor['row_kind']]
        self.nonbasic0=cp.asarray(np.flatnonzero(host_kind!=2))
        pool_lookup=np.full(m+n,-1,dtype=np.int32);pool_lookup[np.flatnonzero(host_kind!=2)]=np.arange(len(self.nonbasic0))
        self.pool_lookup=cp.asarray(pool_lookup)
        lookup=np.full(m+n,-1,dtype=np.int32)
        inactive=np.setdiff1d(np.arange(m),base.anchor['active'])
        lookup[np.r_[base.anchor['basic'],n+inactive]]=np.arange(m)
        self.lookup=cp.asarray(lookup)
        self.sparse_columns=cp.RawKernel(r'''
        extern "C" __global__ void inverse_columns(const int* ap,const int* ai,const double* av,
            const int* ip,const int* ii,const double* iv,const long long* columns,double* out,int m){
            int work=blockIdx.x;long long column=columns[work];double* target=out+(long long)work*m;
            // One block owns one output column. The barrier orders sparse
            // summands deterministically; no floating atomic accumulation.
            for(int edge=ap[column];edge<ap[column+1];edge++){
                int row=ai[edge];double coefficient=av[edge];
                for(int pos=ip[row]+threadIdx.x;pos<ip[row+1];pos+=blockDim.x)
                    target[ii[pos]]+=coefficient*iv[pos];
                __syncthreads();
            }
        }''','inverse_columns',options=('--fmad=false',))
        self.sparse_columns.compile()
        self.scatter_basis=cp.RawKernel(r'''
        extern "C" __global__ void scatter_basis(const long long* rows,const long long* columns,
            long long* basis,int* slots,int batch,int m,int width,int offset){
            int i=blockIdx.x*blockDim.x+threadIdx.x;
            if(i>=batch*width)return;
            long long row=rows[i];int b=i/width,j=i%width;
            if(row<m){basis[(long long)b*m+row]=columns[i];slots[(long long)b*m+row]=offset+j;}
        }''','scatter_basis')
        self.scatter_basis.compile()
        self.graph_calls=OrderedDict();self.graph_compilation_seconds=0.

    def clear_graph_cache(self):
        while self.graph_calls:self.graph_calls.popitem()[1].close()

    def run_device(self,**inputs):
        from .gpu_replay_call import GpuReplayCall,signature
        if self.rank_bucket:
            cp=self.cp;warm=inputs.get('warm_start');k0=len(self.base.var)
            previous=k0 if warm is None else int(warm['used_rank'].max().get())
            width=inputs['selected_columns'].shape[1] if inputs.get('selected_columns') is not None else inputs.get('columns',32)
            capacity=restricted_basis_capacity(previous,k0,self.base.m,self.base.n,
                width,self.max_pivots,self.pivot_aware_capacity)
            inputs['basis_capacity']=capacity
            if warm is not None:
                old=warm['u'].shape[2];count=min(old,capacity);batch=warm['u'].shape[0]
                u=cp.zeros((batch,self.base.m,capacity));u[:,:,:count]=warm['u'][:,:,:count]
                rows=cp.full((batch,capacity),-2,dtype=cp.int32);rows[:,:count]=warm['update_rows'][:,:count]
                inputs['warm_start']=dict(warm,u=u,update_rows=rows)
        key=signature(inputs)
        if key not in self.graph_calls:
            self.clear_graph_cache()
            call=GpuReplayCall(self,inputs);self.graph_calls[key]=call
            self.graph_compilation_seconds+=call.compilation_seconds
        out=self.graph_calls[key].run(inputs)
        return out if self.rank_bucket else self.base._trim_updates(out)

    def solve_device(self,*,rhs,lower,upper,c,delta,col_scale,row_scale,columns=32,selected_columns=None,
                     warm_start=None,reuse_lifted_state=False,basis_capacity=None,compute_expansion=True,
                     conditional=None,_warm_iterations=None):
        cp=self.cp;base=self.base;math=self.math;mm=math.mm
        batch,n=lower.shape;m=base.m;k0=len(base.var);total=m+n
        if not isinstance(columns,int) or columns<1:raise ValueError('Positive restricted dimension required')
        k=min(columns,len(self.nonbasic0)) if selected_columns is None else selected_columns.shape[1]
        if selected_columns is not None and (selected_columns.ndim!=2 or selected_columns.shape[0]!=batch or selected_columns.dtype!=cp.int64 or k<1):
            raise ValueError('Selected directions must be batch x K int64')
        env=cp.arange(batch)
        inputs=dict(rhs=rhs,lower=lower,upper=upper,c=c,delta=delta,col_scale=col_scale,row_scale=row_scale)
        if reuse_lifted_state:
            if warm_start is None or warm_start['owner'] is not base or warm_start['basis'].shape!=(batch,m):
                raise ValueError('Lifted state belongs to a different solver/batch')
            raw=warm_start
        else:raw=base.solve_device(**inputs,warm_start=warm_start,pivot_budget=0)['warm_state']
        lo,hi,scale,cost=(raw[x] for x in ('lower','upper','scale','cost'))
        basis=raw['basis'].copy()
        x=raw['x'];xb=x[env[:,None],basis].copy()
        physical_v=delta[:,:,base.basis0.clip(max=n-1)]*(base.basis0<n)[None,None]
        initial_u=raw['u'];prior_rank=initial_u.shape[2]
        prior_active=cp.max(raw['used_rank'],keepdims=True)
        prior_rows=raw['update_rows'][:,k0:];prior_valid=prior_rows>=0
        def apply_v_matrix(value):
            out=cp.zeros((batch,prior_rank,value.shape[2]))
            out[:,:k0]=mm(physical_v,value)
            out[:,k0:]=cp.take_along_axis(value,prior_rows.clip(min=0)[:,:,None],axis=1)*prior_valid[:,:,None]
            return out
        small=cp.eye(prior_rank)[None]+apply_v_matrix(initial_u)

        def inverse_apply(value,transpose=False):
            if transpose:
                initial_value=mm(base.inverse_transpose,value.T).T
                q=math.solve(small.transpose(0,2,1),mm(initial_u.transpose(0,2,1),value[:,:,None]),active_size=prior_active)[:,:,0]
                correction=mm(physical_v.transpose(0,2,1),q[:,:k0,None])[:,:,0]
                cp.add.at(correction,(env[:,None],prior_rows.clip(min=0)),cp.where(prior_valid,q[:,k0:],0.))
                return initial_value-mm(base.inverse_transpose,correction.T).T
            projected=mm(base.inverse,value.transpose(1,0,2).reshape(m,-1)).reshape(m,batch,-1).transpose(1,0,2)
            q=math.solve(small,apply_v_matrix(projected),active_size=prior_active)
            return projected-mm(initial_u,q)

        def matrix_columns(ids):
            count=ids.shape[1];out=cp.zeros((batch*count,m))
            base.gather_column((batch*count,),(128,),(base.column_ptr,base.column_rows,base.column_values,
                cp.ascontiguousarray(ids.reshape(-1),dtype=cp.int64),out,np.int32(m)))
            out=out.reshape(batch,count,m).transpose(0,2,1).copy()
            addition=cp.take_along_axis(delta,cp.broadcast_to(ids[:,None,:].clip(max=n-1),(batch,k0,count)),axis=2)
            out[:,base.var,:]+=addition*(ids<n)[:,None,:]
            return out

        def inverse_columns(ids):
            count=ids.shape[1];out=cp.zeros((batch*count,m));inverse_t=base.inverse_transpose
            self.sparse_columns((batch*count,),(256,),(base.column_ptr,base.column_rows,base.column_values,
                inverse_t.indptr,inverse_t.indices,inverse_t.data,cp.ascontiguousarray(ids.reshape(-1),dtype=cp.int64),out,np.int32(m)))
            out=out.reshape(batch,count,m).transpose(0,2,1)
            addition=cp.take_along_axis(delta,cp.broadcast_to(ids[:,None,:].clip(max=n-1),(batch,k0,count)),axis=2)
            return out+mm(base.initial_u,addition*(ids<n)[:,None,:])

        # Full-space Phase-I pricing gives candidate directions; it is not an
        # acceptance test. Fixed nonbasic columns are allowed only as inert
        # padding when a tiny test LP has fewer than K movable columns.
        merit=cp.where(x<lo-1e-9*scale,-1./scale,cp.where(x>hi+1e-9*scale,1./scale,0.))
        primal_bad=cp.max(cp.maximum(lo-x,x-hi)/scale,axis=1)>1e-8
        rc=cp.zeros_like(cost)
        with (self.pricing_conditional.iteration(~primal_bad,cp.zeros(batch,dtype=bool))
              if conditional and self.pricing_conditional else nullcontext()):
            y=inverse_apply(merit[env[:,None],basis],True)
            rc[:]=merit-mm(base.matrix_transpose,y.T).T
            rc[:,:n]-=mm(delta.transpose(0,2,1),y[:,base.var,None])[:,:,0]
        pricing=cp.where(primal_bad[:,None],rc,raw['reduced'])
        ids=self.nonbasic0 if warm_start is None else cp.arange(total,dtype=cp.int64)
        kind0=raw['kind'][:,ids]
        direction=cp.where(kind0==-1,1.,cp.where(kind0==1,-1.,cp.where(pricing[:,ids]<0,1.,-1.)))
        score=cp.where((hi[:,ids]-lo[:,ids]>1e-12)&(kind0!=2),-pricing[:,ids]*direction,-cp.inf)
        # Prefer every nonbasic variable over basic padding, including fixed
        # nonbasics. Exactly n nonbasic columns always exist.
        score=cp.where(kind0==2,-cp.inf,cp.where(cp.isneginf(score),-1e300,score))
        selected=(cp.broadcast_to(ids,(batch,len(ids)))[env[:,None],cp.argsort(-score,axis=1)[:,:k]]
            if selected_columns is None else selected_columns.copy())
        valid_columns=((selected>=0)&(selected<total)).all(axis=1)
        selected=selected.clip(0,total-1)
        valid_columns&=(raw['kind'][env[:,None],selected]!=2).all(axis=1)
        valid_columns&=(cp.diff(cp.sort(selected,axis=1),axis=1)!=0).all(axis=1)
        if warm_start is not None:
            valid_columns&=(delta==warm_start['delta']).all(axis=(1,2))&(rhs==warm_start['rhs']).all(axis=1)
            valid_columns&=(col_scale==warm_start['col_scale']).all(axis=1)&(row_scale==warm_start['row_scale']).all(axis=1)
        if reuse_lifted_state:
            valid_columns&=(lower==lo[:,:n]).all(axis=1)&(upper==hi[:,:n]).all(axis=1)&(c==cost[:,:n]).all(axis=1)
        nonbasic=selected.copy();xn=x[env[:,None],nonbasic].copy()
        kind=raw['kind'][env[:,None],nonbasic].copy()
        projected=inverse_columns(nonbasic)
        tableau=projected-mm(initial_u,math.solve(small,apply_v_matrix(projected),active_size=prior_active))
        restricted_rc=cost[env[:,None],nonbasic]-mm(cost[env[:,None],basis][:,None,:],tableau)[:,0,:]
        sign_bad=cp.where(kind==-1,-restricted_rc,cp.where(kind==1,restricted_rc,cp.abs(restricted_rc)))
        sign_bad=cp.where(hi[env[:,None],nonbasic]-lo[env[:,None],nonbasic]>1e-12,sign_bad,0.)
        phase=cp.where(cp.max(sign_bad*scale[env[:,None],nonbasic],axis=1)<=1e-8,2,0).astype(cp.int32)
        stopped=cp.zeros(batch,dtype=bool);failed=cp.zeros(batch,dtype=bool);pivots=cp.zeros(batch,dtype=cp.int32)
        counter=cp.zeros(1,dtype=cp.int32)
        count=1 if conditional and conditional.while_mode else self.max_pivots
        if _warm_iterations is not None:count=min(count,_warm_iterations)
        for _ in range(count):
            with (conditional.iteration(stopped,failed,counter,self.max_pivots) if conditional else nullcontext()):
                lb=lo[env[:,None],basis];ub=hi[env[:,None],basis];bs=scale[env[:,None],basis]
                ln=lo[env[:,None],nonbasic];un=hi[env[:,None],nonbasic];ns=scale[env[:,None],nonbasic]
                violations=cp.maximum(lb-xb,xb-ub)/bs
                row=cp.argmax(violations,axis=1);primal_bad=cp.max(violations,axis=1)>1e-8
                phase[:]=cp.where(~primal_bad,1,phase)
                cb=cp.where(phase[:,None]==0,cp.where(xb<lb-1e-9*bs,-1./bs,cp.where(xb>ub+1e-9*bs,1./bs,0.)),cost[env[:,None],basis])
                cn=cp.where(phase[:,None]==0,0.,cost[env[:,None],nonbasic])
                reduced=cn-mm(cb[:,None,:],tableau)[:,0,:]
                direction=cp.where(kind==-1,1.,cp.where(kind==1,-1.,cp.where(reduced<0,1.,-1.)))
                movable=un-ln>1e-12
                score=cp.where(movable,-reduced*direction,-cp.inf)
                enter_primal=cp.argmax(score,axis=1)
                stopped|=(phase==1)&~primal_bad&(cp.max(score,axis=1)<=1e-8)
                use_dual=(phase==2)&primal_bad
                sign=cp.where(xb[env,row]<lb[env,row],1.,-1.)
                dual_row=tableau[env,row,:]
                dual_direction=cp.where(kind==-1,1.,cp.where(kind==1,-1.,cp.where(-sign[:,None]*dual_row>0,1.,-1.)))
                alpha=-sign[:,None]*dual_row*dual_direction
                eligible=movable&(alpha>1e-10)
                ratio=cp.where(eligible,cp.maximum(reduced*dual_direction,0.)/cp.maximum(alpha,1e-300),cp.inf)
                minimum=cp.min(ratio,axis=1);eligible&=ratio<=minimum[:,None]+1e-10
                enter_dual=cp.argmax(cp.where(eligible,alpha,-1.),axis=1)
                enter=cp.where(use_dual,enter_dual,enter_primal)
                direct=cp.where(use_dual,dual_direction[env,enter],direction[env,enter])
                col=tableau[env,:,enter].copy();movement=col*direct[:,None]
                distance=cp.where(movement>0,cp.maximum(xb-lb,0.),cp.maximum(ub-xb,0.))
                below=xb<lb-1e-9*bs;above=xb>ub+1e-9*bs
                merit_distance=cp.where(below,cp.where(movement<0,lb-xb,cp.inf),cp.where(above,cp.where(movement>0,xb-ub,cp.inf),distance))
                distance=cp.where((phase==0)[:,None],merit_distance,distance)
                ratios=cp.where(cp.abs(movement)>1e-10,distance/cp.maximum(cp.abs(movement),1e-300),cp.inf)
                relaxed=cp.where(cp.abs(movement)>1e-10,(distance+1e-9)/cp.maximum(cp.abs(movement),1e-300),cp.inf)
                limiting=ratios<=cp.min(relaxed,axis=1)[:,None]
                row_primal=cp.argmax(cp.where(limiting,cp.abs(col),-1.),axis=1)
                basic_step=ratios[env,row_primal]
                own_step=cp.where(direct>0,un[env,enter]-xn[env,enter],xn[env,enter]-ln[env,enter])
                flip=~use_dual&(own_step<basic_step);row=cp.where(use_dual,row,row_primal)
                pivot=col[env,row]
                active=~stopped&~failed
                bad=active&((use_dual&~cp.isfinite(minimum))|(~use_dual&~cp.isfinite(cp.minimum(basic_step,own_step)))|
                    ((phase==0)&(cp.max(score,axis=1)<=1e-10))|(~flip&((cp.abs(pivot)<1e-12)|~cp.isfinite(pivot))))
                failed|=bad;active&=~bad;do_pivot=active&~flip
                target=cp.where(sign>0,lb[env,row],ub[env,row])
                dual_step=(xb[env,row]-target)/cp.where(cp.abs(movement[env,row])>1e-300,movement[env,row],1.)
                step=cp.where(active,cp.where(use_dual,dual_step,cp.minimum(basic_step,own_step)),0.)
                entering_value=xn[env,enter]+direct*step
                xb-=movement*step[:,None]
                leaving_kind=cp.where(use_dual,cp.where(sign>0,-1,1),cp.where((phase==0)&below[env,row],-1,
                    cp.where((phase==0)&above[env,row],1,cp.where(movement[env,row]>0,-1,1))))
                leaving_value=cp.where(leaving_kind==-1,lb[env,row],ub[env,row])
                xn[env,enter]=cp.where(do_pivot,leaving_value,entering_value)
                xb[env,row]=cp.where(do_pivot,entering_value,xb[env,row])
                safe=cp.where(do_pivot,pivot,1.)
                row_values=tableau[env,row,:].copy()/safe[:,None]
                tableau-=cp.where(do_pivot[:,None,None],col[:,:,None]*row_values[:,None,:],0.)
                tableau[env,row,:]=cp.where(do_pivot[:,None],row_values,tableau[env,row,:])
                tableau[env,:,enter]=cp.where(do_pivot[:,None],-col/safe[:,None],tableau[env,:,enter])
                tableau[env,row,enter]=cp.where(do_pivot,1./safe,tableau[env,row,enter])
                old=basis[env,row].copy();basis[env,row]=cp.where(do_pivot,nonbasic[env,enter],old)
                nonbasic[env,enter]=cp.where(do_pivot,old,nonbasic[env,enter])
                kind[env,enter]=cp.where(do_pivot,leaving_kind,cp.where(active&flip,-kind[env,enter],kind[env,enter]))
                pivots+=active.astype(cp.int32);counter+=1

        # Retain earlier basis improvements across column-generation rounds.
        # The tableau width stays K even if the compact basis needs more
        # replacements. Canonicalize relative to the immutable root basis.
        if basis_capacity is not None and (not isinstance(basis_capacity,int) or basis_capacity<=k0):
            raise ValueError('Invalid padded basis capacity')
        # A direct/non-bucket call can use the allocated prior rank as a
        # conservative upper bound without a host read inside CUDA capture.
        # run_device's bucket path instead uses max(used_rank) above.
        extra=max(1,min(k,self.max_pivots)) if self.pivot_aware_capacity else k
        capacity=min(m,n,prior_rank-k0+extra if basis_capacity is None else basis_capacity-k0)
        kinds=raw['kind'].copy();kinds[env[:,None],nonbasic]=kind;kinds[env[:,None],basis]=2
        valid_columns&=(kinds[:,base.basis0]!=2).sum(axis=1)<=capacity
        missing=cp.sort(cp.where(kinds[:,base.basis0]!=2,cp.arange(m)[None],m),axis=1)[:,:capacity]
        valid=missing<m;rows=missing.clip(max=m-1);oldcols=base.basis0[rows]
        entering=self.lookup[basis]<0
        eorder=cp.argsort(cp.where(entering,cp.arange(m)[None],m+cp.arange(m)[None]),axis=1)[:,:capacity]
        newcols=cp.take_along_axis(basis,eorder,axis=1)
        static=inverse_columns(newcols)-inverse_columns(oldcols)
        u=cp.concatenate((cp.broadcast_to(base.initial_u,(batch,m,k0)),static*valid[:,None,:]),axis=2)
        canonical=cp.broadcast_to(base.basis0,(batch,m)).copy()
        slots=cp.full((batch,m),-1,dtype=cp.int32)
        self.scatter_basis(((batch*capacity+127)//128,),(128,),(cp.ascontiguousarray(missing,dtype=cp.int64),
            cp.ascontiguousarray(newcols,dtype=cp.int64),canonical,slots,np.int32(batch),np.int32(m),np.int32(capacity),np.int32(k0)))
        kinds[env[:,None],canonical]=2
        warm=dict(owner=base,basis=canonical,kind=kinds,u=u,v=physical_v,delta=delta,rhs=rhs,
            col_scale=col_scale,row_scale=row_scale,update_rows=cp.concatenate((cp.full((batch,k0),-1,dtype=cp.int32),cp.where(valid,missing,-2).astype(cp.int32)),axis=1),
            update_slots=slots,used_rank=(k0+valid.sum(axis=1)).astype(cp.int32))
        out=base.solve_device(**inputs,warm_start=warm,pivot_budget=0)
        out['accepted']&=valid_columns
        out['values']=cp.where(out['accepted'][:,None],out['values'],cp.nan)
        # Price excluded directions at the lifted solution. A feasible RMP
        # uses the original objective; an infeasible one uses bound-distance
        # Phase I. This enriches the space from actual residuals, rather than
        # taking a larger prefix of an obsolete initial ranking.
        actual=out['warm_state'];full_x=actual['x']
        actual_kind=actual['kind'][env[:,None],nonbasic]
        actual_rc=actual['reduced'][env[:,None],nonbasic]
        restricted_bad=cp.where(actual_kind==-1,-actual_rc,cp.where(actual_kind==1,actual_rc,cp.abs(actual_rc)))
        restricted_bad=cp.where(hi[env[:,None],nonbasic]-lo[env[:,None],nonbasic]>1e-12,
            restricted_bad*scale[env[:,None],nonbasic],0.)
        out['restricted_lifted_dual_violation']=cp.maximum(cp.max(restricted_bad,axis=1),0.)
        # Some callers immediately fall back to CPU or independently reprice
        # the next round. For them this full-space expansion is unused work.
        # Keep the full certificate and restricted-dual diagnostic above;
        # None explicitly distinguishes "not computed" from a zero score.
        expansion=None
        if compute_expansion:
            phase_rc=cp.zeros_like(cost)
            with (self.pricing_conditional.iteration(out['primal_residual']<=1e-8,cp.zeros(batch,dtype=bool))
                  if conditional and self.pricing_conditional else nullcontext()):
                merit=cp.where(full_x<lo-1e-9*scale,-1./scale,cp.where(full_x>hi+1e-9*scale,1./scale,0.))
                dual_rhs=merit[env[:,None],canonical]
                vu=cp.zeros((batch,k0+capacity,k0+capacity));vu[:,:k0]=mm(physical_v,u)
                vu[:,k0:]=cp.take_along_axis(u,rows[:,:,None],axis=1)*valid[:,:,None]
                active=cp.max((k0+valid.sum(axis=1)).astype(cp.int32),keepdims=True)
                q=math.solve((cp.eye(k0+capacity)[None]+vu).transpose(0,2,1),mm(u.transpose(0,2,1),dual_rhs[:,:,None]),active_size=active)[:,:,0]
                correction=mm(physical_v.transpose(0,2,1),q[:,:k0,None])[:,:,0]
                cp.add.at(correction,(env[:,None],rows),cp.where(valid,q[:,k0:],0.))
                phase_y=mm(base.inverse_transpose,(dual_rhs-correction).T).T
                phase_rc[:]=merit-mm(base.matrix_transpose,phase_y.T).T
                phase_rc[:,:n]-=mm(delta.transpose(0,2,1),phase_y[:,base.var,None])[:,:,0]
            prices=cp.where((out['primal_residual']>1e-8)[:,None],phase_rc,actual['reduced'])
            ids=self.nonbasic0
            signs=cp.where(base.kind0[ids]==-1,1.,cp.where(base.kind0[ids]==1,-1.,cp.where(prices[:,ids]<0,1.,-1.)))
            expansion=cp.where(hi[:,ids]-lo[:,ids]>1e-12,-prices[:,ids]*signs,-cp.inf)
            expansion[env[:,None],self.pool_lookup[selected].clip(min=0)]=-cp.inf
        out.update(restricted_pivots=pivots,restricted_columns=k,restricted_failed=failed,
            selected_columns=selected,expansion_scores=expansion,
            scope='Restricted-column GPU LP plus complete original-LP certification')
        return out
