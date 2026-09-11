"""Compact parametric basis maps with full original-LP numerical certificates.

Offline sparse-LU projects only RHS/bound/cost directions occurring in the
declared input family. Online equality guards reject an unsupported family;
the complete unmodified LP primal/dual/gap checks are still authoritative.
No full inverse on the GPU, no online factorization or CPU optimization.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


def equality_groups(values):
    _,representatives,mapping=np.unique(values.T,axis=0,return_index=True,return_inverse=True)
    return representatives,mapping


def project_basis(anchor,data,offset,variable_rows,extra_costs=()):
    p=anchor['lp'];basic=anchor['basic'];active=anchor['active'];kind=anchor['kind']
    m,n=p.a.shape;nb=len(basic)
    lower=np.vstack([data['lower'],p.lower]);upper=np.vstack([data['upper'],p.upper])
    xn=np.where(kind==-1,lower,np.where(kind==1,upper,0.))
    if not np.isfinite(xn).all():raise ValueError('Nonfinite nonbasic training bound')
    representatives,mapping=equality_groups(xn)
    gx=csr_matrix((np.ones(n),(np.arange(n),mapping)),shape=(n,len(representatives)))
    rhs=np.vstack([data['rhs'],p.rhs])[:,active]
    rhs_indices=np.flatnonzero(np.any(rhs!=0.,axis=0))
    costs=np.vstack([data['c'],p.c,*extra_costs])[:,basic]
    cost_representatives,cost_mapping=equality_groups(costs)
    gc=csr_matrix((np.ones(nb),(np.arange(nb),cost_mapping)),shape=(nb,len(cost_representatives)))
    lookup={int(row):i for i,row in enumerate(active)}
    update_positions=np.array([j for j,row in enumerate(variable_rows) if row in lookup],dtype=int)
    update_active=np.array([lookup[variable_rows[j]] for j in update_positions],dtype=int)
    changed=(data['delta']-offset)[:,update_positions][:,:,basic]
    delta_support=np.flatnonzero(np.any(changed!=0.,axis=(0,1)))
    factor=splu(p.a[active][:,basic].tocsc())
    def eye_columns(indices):
        matrix=np.zeros((nb,len(indices)));matrix[indices,np.arange(len(indices))]=1.
        return matrix
    def solve_columns(matrix,trans='N'):
        # SuperLU's native solve must not receive a zero-column RHS.
        return factor.solve(matrix,trans=trans) if matrix.shape[1] else np.empty((nb,0))
    return dict(basic=basic,active=active,kind=kind,offset=offset,
        representatives=representatives,mapping=mapping,rhs_indices=rhs_indices,
        cost_representatives=cost_representatives,cost_mapping=cost_mapping,
        update_positions=update_positions,update_active=update_active,delta_support=delta_support,
        p_rhs=solve_columns(eye_columns(rhs_indices)),p_bound=solve_columns((p.a[active]@gx).toarray()),
        u=solve_columns(eye_columns(update_active)),p_cost=solve_columns(gc.toarray(),trans='T'),
        p_delta=solve_columns(eye_columns(delta_support),trans='T'))


def certify(cp,a,at,var,neq,inputs,x,y,math=None):
    mm=(lambda a,b:a@b) if math is None else math.mm
    einsum=cp.einsum if math is None else math.einsum
    rhs,lo,hi,c,delta,cs,rs=(inputs[k] for k in ('rhs','lower','upper','c','delta','col_scale','row_scale'))
    activity=mm(a,x.T).T;activity[:,var]+=einsum('bkn,bn->bk',delta,x)
    reduced=c-mm(at,y.T).T-einsum('bkn,bk->bn',delta,y[:,var])
    def maximum(v):return cp.maximum(cp.max(v,axis=1),0.) if v.shape[1] else cp.zeros(len(x))
    error=(activity-rhs)/rs
    primal=cp.maximum(maximum(cp.abs(error[:,:neq])),cp.maximum(maximum(error[:,neq:]),maximum(cp.maximum(lo-x,x-hi)/cs)))
    rc=reduced*cs
    dual=cp.maximum(maximum(cp.maximum(cp.where(~cp.isfinite(lo),rc,0.),cp.where(~cp.isfinite(hi),-rc,0.))),maximum((y*rs)[:,neq:]))
    target=cp.where(reduced>=0,lo,hi)
    gap=cp.sum(cp.abs(reduced*(x-cp.where(cp.isfinite(target),target,x))),axis=1)
    gap+=cp.sum(cp.abs(y[:,neq:]*(rhs-activity)[:,neq:]),axis=1)
    objective=cp.sum(c*x,axis=1);gap/=cp.maximum(1.,cp.abs(objective))
    accepted=cp.isfinite(x).all(axis=1)&cp.isfinite(y).all(axis=1)&cp.isfinite(gap)
    accepted&=(primal<=1e-5)&(dual<=1e-7)&(gap<=1e-7)
    return dict(accepted=accepted,primal_residual=primal,dual_violation=dual,relative_kkt_gap=gap,
        objective=objective,raw_values=x,raw_y=y,raw_reduced=reduced,activity=activity)


class CompactEvaluator:
    def __init__(self,bank,arrays):
        self.bank=bank;self.cp=bank.cp
        self.d={k:self.cp.asarray(v) for k,v in arrays.items()}
        self.repair_path=None

    def repair_anchor(self):
        """Load an offline inverse; never factor/optimize in the online path."""
        import hashlib
        from .gpu_certified_basis import GpuBasisEvaluator,NormalizedLP
        if self.repair_path is None:raise ValueError('Missing offline repair operator')
        path,digest=self.repair_path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Repair artifact changed')
        with np.load(path,allow_pickle=False) as d:
            inverse=csr_matrix((d['inverse_data'],d['inverse_indices'],d['inverse_indptr']),shape=tuple(d['inverse_shape']))
        d={k:self.d[k].get() for k in ('basic','active','kind','offset')};bank=self.bank
        a=bank.host_a.tolil(copy=True);a[bank.variable_rows]=a[bank.variable_rows]+csr_matrix(d['offset'][0]);a=a.tocsr()
        m,n=a.shape;row_kind=np.full(m,2,dtype=np.int8)
        row_kind[d['active']]=np.where(d['active']<bank.neq,-1,1)
        # No fictitious optimized solution: this object supplies only the
        # basis matrix/operator; all RHS, bounds and objectives are runtime inputs.
        lp=NormalizedLP(a,np.zeros(m),np.zeros(n),np.zeros(n),np.zeros(n),bank.neq,np.ones(n),np.ones(m))
        anchor=dict(lp=lp,basic=d['basic'],active=d['active'],kind=d['kind'],row_kind=row_kind,inverse=inverse)
        return anchor

    def repair_adapter(self):
        from .gpu_certified_basis import GpuBasisInputAdapter
        return GpuBasisInputAdapter(self.repair_anchor(),self.bank.variable_rows)

    def evaluate(self,inputs):
        if self.bank.math is not None:
            from .gpu_replay_call import GpuReplayCall,signature
            key=(id(self),signature(inputs));cache=self.bank.graph_cache
            if key not in cache:
                while len(cache)>=self.bank.graph_cache_size:cache.popitem(last=False)[1].close()
                cache[key]=GpuReplayCall(self,inputs)
                self.bank.graph_compilation_seconds+=cache[key].compilation_seconds
            cache.move_to_end(key)
            return cache[key].run(inputs)
        return self.solve_device(**inputs)

    def solve_device(self,conditional=None,_warm_iterations=None,**inputs):
        cp=self.cp;d=self.d;b=self.bank
        mm=(lambda a,b:a@b) if b.math is None else b.math.mm
        einsum=cp.einsum if b.math is None else b.math.einsum
        solve=cp.linalg.solve if b.math is None else b.math.solve
        lo,hi,c,rhs=inputs['lower'],inputs['upper'],inputs['c'],inputs['rhs']
        xn=cp.where(d['kind']==-1,lo,cp.where(d['kind']==1,hi,0.))
        theta=xn[:,d['representatives']]
        cb=c[:,d['basic']];theta_c=cb[:,d['cost_representatives']]
        changed=(inputs['delta']-d['offset'])[:,d['update_positions']][:,:,d['basic']]
        active_rhs=rhs[:,d['active']]
        valid=cp.isfinite(xn).all(axis=1)&cp.isfinite(rhs).all(axis=1)&cp.isfinite(c).all(axis=1)
        valid&=cp.isfinite(inputs['delta']).all(axis=(1,2))&(lo<=hi).all(axis=1)
        valid&=~cp.isnan(lo).any(axis=1)&~cp.isnan(hi).any(axis=1)
        for name in ('col_scale','row_scale'):
            valid&=cp.isfinite(inputs[name]).all(axis=1)&(inputs[name]>0).all(axis=1)
        valid&=cp.max(cp.abs(xn-theta[:,d['mapping']]),axis=1)<=1e-12
        valid&=cp.max(cp.abs(cb-theta_c[:,d['cost_mapping']]),axis=1)<=1e-12
        residual_rhs=active_rhs.copy();residual_rhs[:,d['rhs_indices']]=0.
        valid&=cp.max(cp.abs(residual_rhs),axis=1)<=1e-12
        extra=changed.copy();extra[:,:,d['delta_support']]=0.
        if extra.shape[1]:valid&=cp.max(cp.abs(extra),axis=(1,2))<=1e-12
        theta=cp.nan_to_num(theta);theta_c=cp.nan_to_num(theta_c)
        xb=mm(active_rhs[:,d['rhs_indices']],d['p_rhs'].T)-mm(theta,d['p_bound'].T)
        y0=mm(theta_c,d['p_cost'].T)
        if len(d['update_positions']):
            delta=inputs['delta']-d['offset']
            xb-=mm(einsum('bkn,bn->bk',delta[:,d['update_positions']],xn),d['u'].T)
            small=cp.eye(len(d['update_positions']))[None]+mm(changed,d['u'])
            correction=solve(small,mm(changed,xb[:,:,None]))[:,:,0]
            xb-=mm(correction,d['u'].T)
            q=solve(small.transpose(0,2,1),y0[:,d['update_active'],None])[:,:,0]
            y0-=mm(einsum('bkn,bk->bn',changed[:,:,d['delta_support']],q),d['p_delta'].T)
        x=xn.copy();x[:,d['basic']]=xb
        y=cp.zeros_like(rhs);y[:,d['active']]=y0
        out=certify(cp,b.a,b.at,b.var,b.neq,inputs,x,y,b.math)
        bound_error=cp.maximum(lo-x,x-hi)/inputs['col_scale']
        row_error=(out['activity']-rhs)/inputs['row_scale']
        row_error[:,:b.neq]=cp.abs(row_error[:,:b.neq])
        out['primal_violation_count']=cp.sum(bound_error>1e-5,axis=1)+cp.sum(row_error>1e-5,axis=1)
        out['primal_violation_l1']=cp.maximum(bound_error,0.).sum(axis=1)+cp.maximum(row_error,0.).sum(axis=1)
        rc=out['raw_reduced']*inputs['col_scale']
        sign=cp.where(d['kind']==-1,-rc,cp.where(d['kind']==1,rc,cp.abs(rc)))
        sign=cp.where((d['kind']!=2)&(hi>lo),sign,0.)
        out['basis_dual_violation']=cp.maximum(cp.max(sign,axis=1),0.)
        if y.shape[1]>b.neq:
            out['basis_dual_violation']=cp.maximum(out['basis_dual_violation'],cp.max((y*inputs['row_scale'])[:,b.neq:],axis=1))
        out['accepted']&=valid;out['input_family_valid']=valid
        out['values']=cp.where(out['accepted'][:,None],x/inputs['col_scale'],cp.nan)
        return out


class _CompactCohortReplay:
    """Capture the fixed Python candidate loop once, without nested graphs."""
    def __init__(self,bank,candidates):
        self.bank=bank;self.cp=bank.cp;self.candidates=candidates

    def solve_device(self,conditional=None,_warm_iterations=None,**inputs):
        bank=self.bank
        result=bank.evaluate_cohort(inputs,self.candidates,_eager=True)
        # A replay does not run Python assignment. Return the metadata arrays
        # with the graph so the caller can restore the matching bank references
        # after switching between cached batch shapes/candidate sets.
        return dict(result=result,scores=bank.last_candidate_scores,
            dual=bank.last_candidate_dual,candidates=bank.last_candidates)


class CompactBank:
    def __init__(self,root,variable_rows,entries,centers,feature_indices,feature_scale,*,capture=False,full_batch_candidates=False,graph_cache_size=48,candidate_ranking='residual',selected_features=True):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        self.cp=cp;self.host_a=root['a'];self.neq=root['neq'];self.variable_rows=np.array(variable_rows)
        self.var=cp.asarray(self.variable_rows);self.a=gpu_csr(self.host_a);self.at=gpu_csr(self.host_a.T.tocsr())
        self.centers=cp.asarray(centers);self.feature_indices=cp.asarray(feature_indices);self.feature_scale=cp.asarray(feature_scale)
        self.feature_selector=None
        if selected_features:
            from .selected_lp_features import SelectedLPFeatures
            m,n=self.host_a.shape
            host_indices=feature_indices.get() if hasattr(feature_indices,'__cuda_array_interface__') else feature_indices
            self.feature_selector=SelectedLPFeatures(host_indices,dict(rhs=(m,),lower=(n,),upper=(n,),c=(n,),
                delta=(len(self.variable_rows),n),col_scale=(n,),row_scale=(m,)))
        from collections import OrderedDict
        self.math=None;self.full_batch_candidates=full_batch_candidates;self.graph_cache=OrderedDict()
        self.cohort_graph_cache=OrderedDict();self.cohort_graph_cache_size=2
        self.cohort_graph_compilation_seconds=0.
        self.graph_cache_size=graph_cache_size;self.graph_compilation_seconds=0.
        if candidate_ranking not in ('residual','count','count_only'):raise ValueError('Unknown candidate ranking')
        self.candidate_ranking=candidate_ranking
        if capture:
            from .gpu_capture_math import CaptureMath
            self.math=CaptureMath()
        self.evaluators=[CompactEvaluator(self,d) for d in entries]
        self.observable_matrix=None;self.observable_scales=None

    def proposal_features(self,inputs):
        """Routing only: select before encoding, without reducing the LP."""
        from .gpu_neural_basis_proposal import features
        selector=getattr(self,'feature_selector',None)
        return selector.gpu_features(inputs) if selector is not None else features(inputs)[:,self.feature_indices]

    def configure_observables(self,matrix,scales):
        """Enable diagnostic candidate spread in physical observable units.

        ``matrix`` maps original (unscaled) LP columns to observables. Scales
        must be positive finite tolerances in the corresponding units. This
        spread is a routing feature, never a feasibility/optimality certificate.
        """
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        matrix=csr_matrix(matrix,dtype=np.float64,copy=True)
        scales=np.asarray(scales,dtype=np.float64)
        if matrix.shape[1]!=self.host_a.shape[1] or not matrix.shape[0]:
            raise ValueError('Observable matrix must have nonzero rows and match LP columns')
        if not np.isfinite(matrix.data).all():raise ValueError('Nonfinite observable matrix')
        if scales.ndim!=1 or scales.shape[0]!=matrix.shape[0]:
            raise ValueError('Observable scales must be one-dimensional and match rows')
        if not np.isfinite(scales).all() or not (scales>0).all():
            raise ValueError('Observable scales must be positive and finite')
        matrix.sum_duplicates();matrix.sort_indices()
        self.clear_cohort_graph_cache()
        self.observable_matrix=gpu_csr(matrix)
        self.observable_scales=self.cp.asarray(scales)
        return self

    def _update_observable_moments(self,state,index,ids,candidate,col_scale):
        """GPU-only Welford update; count each environment/candidate at most once."""
        cp=self.cp;count,mean,m2,seen=state
        physical=candidate['raw_values']/col_scale
        mm=(lambda a,b:a@b) if self.math is None else self.math.mm
        observable=mm(self.observable_matrix,physical.T).T
        valid=candidate['input_family_valid']&cp.isfinite(physical).all(axis=1)
        valid&=cp.isfinite(observable).all(axis=1)&~seen[ids,index]
        seen[ids,index]=True
        # Sanitize rejected candidates before arithmetic, so their NaN/Inf
        # values cannot contaminate valid accumulators through 0 * NaN.
        observable=cp.where(valid[:,None],observable,mean[ids])
        next_count=count[ids]+valid.astype(cp.int64)
        delta=observable-mean[ids]
        next_mean=mean[ids]+delta/cp.maximum(next_count,1)[:,None]
        m2[ids]+=delta*(observable-next_mean)
        mean[ids]=next_mean;count[ids]=next_count

    def clear_graph_cache(self):
        while self.graph_cache:self.graph_cache.popitem()[1].close()
        self.clear_cohort_graph_cache()

    def clear_cohort_graph_cache(self):
        while self.cohort_graph_cache:self.cohort_graph_cache.popitem()[1].close()

    def rank(self,inputs=None):
        """Proposal ordering from the latest screening; never an acceptance decision."""
        cp=self.cp
        order=cp.argsort(self.last_candidate_scores,axis=1)
        if self.candidate_ranking=='count_only':return order
        dual=cp.take_along_axis(self.last_candidate_dual,order,axis=1)
        # Exact lexicographic partition: a huge floating penalty would erase
        # the score differences between non-dual-feasible candidates.
        ordinal=cp.arange(order.shape[1])[None]+(~dual)*order.shape[1]
        return cp.take_along_axis(order,cp.argsort(ordinal,axis=1),axis=1)

    def prepare_host(self,problems):
        cp=self.cp;deltas=[]
        root=self.host_a;variables=self.variable_rows
        if not np.isfinite(root.data).all():raise ValueError('Nonfinite compact reference matrix')
        pattern=getattr(self,'_prepare_host_pattern',None)
        eligible=(isinstance(root,csr_matrix) and root.has_canonical_format
            and variables.dtype.kind in 'iu' and np.all(variables>=0)
            and np.all(variables<root.shape[0]) and len(np.unique(variables))==len(variables))
        if eligible:
            if (pattern is None or pattern['shape']!=root.shape
                    or not np.array_equal(pattern['variables'],variables)
                    or not np.array_equal(pattern['indptr'],root.indptr)
                    or not np.array_equal(pattern['indices'],root.indices)):
                data_rows=np.repeat(np.arange(root.shape[0]),np.diff(root.indptr))
                row_map=np.full(root.shape[0],-1,dtype=int);row_map[variables]=np.arange(len(variables))
                local=row_map[data_rows];positions=np.flatnonzero(local>=0)
                pattern=dict(shape=root.shape,variables=variables.copy(),indptr=root.indptr.copy(),
                    indices=root.indices.copy(),fixed=np.flatnonzero(local<0),positions=positions,
                    local_rows=local[positions],columns=root.indices[positions].copy())
                self._prepare_host_pattern=pattern
        else:pattern=None
        for p in problems:
            if p.a.shape!=self.host_a.shape or p.neq!=self.neq:raise ValueError('Changed compact LP shape')
            if not np.isfinite(p.a.data).all():raise ValueError('Nonfinite compact LP matrix')
            if (pattern is not None and isinstance(p.a,csr_matrix) and p.a.has_canonical_format
                    and np.array_equal(p.a.indptr,pattern['indptr'])
                    and np.array_equal(p.a.indices,pattern['indices'])):
                # Full coefficient checks remain mandatory. Matching CSR
                # layouts only avoids constructing/subsetting a difference CSR;
                # it never assumes that fixed-row numerical values stayed put.
                change=p.a.data-root.data
                if not np.isfinite(change).all():raise ValueError('Nonfinite compact matrix change')
                if np.max(np.abs(change[pattern['fixed']]),initial=0)>2e-12:
                    raise ValueError('Changed unsupported matrix row')
                dense=np.zeros((len(variables),root.shape[1]),dtype=change.dtype)
                dense[pattern['local_rows'],pattern['columns']]=change[pattern['positions']]
                deltas.append(dense)
                continue
            delta=(p.a-self.host_a).tocsr();fixed=np.ones(delta.shape[0],dtype=bool);fixed[self.variable_rows]=False
            if not np.isfinite(delta.data).all():raise ValueError('Nonfinite compact matrix change')
            if np.max(np.abs(delta[fixed].data),initial=0)>2e-12:raise ValueError('Changed unsupported matrix row')
            deltas.append(delta[self.variable_rows].toarray())
        return dict(delta=cp.asarray(np.stack(deltas)),**{k:cp.asarray(np.stack([getattr(p,k) for p in problems]))
            for k in ('rhs','lower','upper','c','col_scale','row_scale')})

    def evaluate_device(self,inputs=None,order=None,**kwargs):
        from .gpu_neural_basis_proposal import features
        if inputs is None:inputs=kwargs
        cp=self.cp;batch,n=inputs['lower'].shape
        if order is None:
            x=self.proposal_features(inputs)
            distance=cp.sum(((x[:,None]-self.centers[None])/self.feature_scale)**2,axis=2)
            order=cp.argsort(distance,axis=1).get()
        accepted=cp.zeros(batch,dtype=bool);indices=cp.full(batch,-1,dtype=cp.int32)
        result={k:cp.zeros((batch,n)) for k in ('values','raw_values','raw_reduced')}
        result['raw_y']=cp.zeros_like(inputs['rhs']);result['activity']=cp.zeros_like(inputs['rhs'])
        for k in ('objective','primal_residual','dual_violation','relative_kkt_gap'):result[k]=cp.full(batch,cp.inf)
        result['input_family_valid']=cp.zeros(batch,dtype=bool);evaluations=0
        best_score=cp.full(batch,cp.inf);best_index=cp.zeros(batch,dtype=cp.int32);best_dual=cp.zeros(batch,dtype=bool)
        self.last_candidates=[]
        self.last_candidate_scores=cp.full((batch,len(self.evaluators)),cp.inf)
        self.last_candidate_dual=cp.zeros((batch,len(self.evaluators)),dtype=bool)
        moments=None
        if self.observable_matrix is not None:
            shape=(batch,self.observable_matrix.shape[0])
            moments=(cp.zeros(batch,dtype=cp.int64),cp.zeros(shape),cp.zeros(shape),
                cp.zeros((batch,len(self.evaluators)),dtype=bool))
        calculated={}
        for rank in range(order.shape[1]):
            pending=np.flatnonzero(~accepted.get())
            if not len(pending):break
            for index in np.unique(order[pending,rank]):
                ids=pending[order[pending,rank]==index];ix=cp.asarray(ids)
                if self.full_batch_candidates:
                    if int(index) not in calculated:
                        calculated[int(index)]=self.evaluators[index].evaluate(inputs)
                        if moments is not None:
                            self._update_observable_moments(moments,index,cp.arange(batch),
                                calculated[int(index)],inputs['col_scale'])
                    candidate={k:v[ix] for k,v in calculated[int(index)].items()}
                else:
                    candidate=self.evaluators[index].evaluate({k:v[ix] for k,v in inputs.items()})
                    if moments is not None:
                        self._update_observable_moments(moments,index,ix,candidate,inputs['col_scale'][ix])
                score=cp.maximum(candidate['primal_residual']/1e-5,
                    cp.maximum(candidate['dual_violation']/1e-7,candidate['relative_kkt_gap']/1e-7))
                if self.candidate_ranking in ('count','count_only'):
                    l1=candidate['primal_violation_l1']
                    score=candidate['primal_violation_count']+l1/(1.+l1)
                dual_ok=candidate['basis_dual_violation']<=1e-8
                self.last_candidate_scores[ix,index]=cp.where(candidate['input_family_valid'],score,cp.inf)
                self.last_candidate_dual[ix,index]=dual_ok&candidate['input_family_valid']
                preference=(score<best_score[ix]) if self.candidate_ranking=='count_only' else ((dual_ok&~best_dual[ix])|((dual_ok==best_dual[ix])&(score<best_score[ix])))
                improve=candidate['input_family_valid']&cp.isfinite(score)&preference
                best_score[ix]=cp.where(improve,score,best_score[ix]);best_index[ix]=cp.where(improve,index,best_index[ix])
                best_dual[ix]=cp.where(improve,dual_ok,best_dual[ix])
                self.last_candidates.append((int(index),ids.copy(),{k:candidate[k] for k in
                    ('accepted','input_family_valid','primal_residual','dual_violation','relative_kkt_gap')}))
                for k in result:result[k][ix]=candidate[k]
                accepted[ix]=candidate['accepted'];indices[ix]=cp.where(candidate['accepted'],index,-1)
                evaluations+=len(ids)
        result.update(accepted=accepted,candidate_index=indices,candidate_evaluations=evaluations,cpu_lp_calls=0,
            best_candidate_index=best_index,best_candidate_dual_feasible=best_dual)
        result['values']=cp.where(accepted[:,None],result['values'],cp.nan)
        if moments is not None:
            count,mean,m2,_=moments
            standard_deviation=cp.sqrt(cp.maximum(m2,0.)/cp.maximum(count-1,1)[:,None])
            dispersion=cp.max(standard_deviation/self.observable_scales,axis=1)
            result['candidate_count']=count
            result['observable_dispersion']=cp.where(count>=2,dispersion,cp.inf)
        return result

    def evaluate_cohort(self,inputs,candidate_indices,*,_eager=False):
        """Screen a fixed shared candidate set against every environment.

        Every candidate receives the original full-LP certificate on the full
        batch, even for environments where it was not a nearest-neighbor
        proposal. Selection stays on the GPU: the first certified solution is
        retained, and uncertified environments retain the best repair proposal.
        There is no per-candidate host acceptance check or batch subdivision.
        """
        candidates=np.asarray(candidate_indices)
        if candidates.ndim!=1 or (candidates.size and candidates.dtype.kind not in 'iu'):
            raise ValueError('Cohort candidate indices must be a one-dimensional host integer array')
        if (len(np.unique(candidates))!=len(candidates)
                or np.any(candidates<0) or np.any(candidates>=len(self.evaluators))):
            raise ValueError('Cohort candidate indices must be unique and in range')
        cp=self.cp;batch,n=inputs['lower'].shape
        accepted=cp.zeros(batch,dtype=bool);indices=cp.full(batch,-1,dtype=cp.int32)
        result={k:cp.zeros((batch,n)) for k in ('values','raw_values','raw_reduced')}
        result['raw_y']=cp.zeros_like(inputs['rhs']);result['activity']=cp.zeros_like(inputs['rhs'])
        for k in ('objective','primal_residual','dual_violation','relative_kkt_gap'):
            result[k]=cp.full(batch,cp.inf)
        result['input_family_valid']=cp.zeros(batch,dtype=bool)
        best_score=cp.full(batch,cp.inf);best_index=cp.zeros(batch,dtype=cp.int32)
        best_dual=cp.zeros(batch,dtype=bool)
        self.last_candidates=[]
        self.last_candidate_scores=cp.full((batch,len(self.evaluators)),cp.inf)
        self.last_candidate_dual=cp.zeros((batch,len(self.evaluators)),dtype=bool)
        host_ids=np.arange(batch);device_ids=cp.arange(batch)
        moments=None
        if self.observable_matrix is not None:
            shape=(batch,self.observable_matrix.shape[0])
            moments=(cp.zeros(batch,dtype=cp.int64),cp.zeros(shape),cp.zeros(shape),
                cp.zeros((batch,len(self.evaluators)),dtype=bool))
        for index in candidates:
            index=int(index)
            candidate=(self.evaluators[index].solve_device(**inputs) if _eager
                else self.evaluators[index].evaluate(inputs))
            if moments is not None:
                self._update_observable_moments(moments,index,device_ids,candidate,inputs['col_scale'])
            score=cp.maximum(candidate['primal_residual']/1e-5,
                cp.maximum(candidate['dual_violation']/1e-7,candidate['relative_kkt_gap']/1e-7))
            if self.candidate_ranking in ('count','count_only'):
                l1=candidate['primal_violation_l1']
                score=candidate['primal_violation_count']+l1/(1.+l1)
            dual_ok=candidate['basis_dual_violation']<=1e-8
            self.last_candidate_scores[:,index]=cp.where(candidate['input_family_valid'],score,cp.inf)
            self.last_candidate_dual[:,index]=dual_ok&candidate['input_family_valid']
            preference=(score<best_score) if self.candidate_ranking=='count_only' else (
                (dual_ok&~best_dual)|((dual_ok==best_dual)&(score<best_score)))
            improve=candidate['input_family_valid']&cp.isfinite(score)&preference
            first_accept=~accepted&candidate['accepted']
            select=first_accept|(~accepted&improve)
            best_score=cp.where(select,score,best_score)
            best_index=cp.where(select,index,best_index)
            best_dual=cp.where(select,dual_ok,best_dual)
            for key,value in result.items():
                mask=select.reshape((batch,)+(1,)*(value.ndim-1))
                result[key]=cp.where(mask,candidate[key],value)
            indices=cp.where(first_accept,index,indices)
            accepted|=candidate['accepted']
            self.last_candidates.append((index,host_ids.copy(),{k:candidate[k] for k in
                ('accepted','input_family_valid','primal_residual','dual_violation','relative_kkt_gap')}))
        result.update(accepted=accepted,candidate_index=indices,
            candidate_evaluations=batch*len(candidates),cpu_lp_calls=0,
            best_candidate_index=best_index,best_candidate_dual_feasible=best_dual)
        result['values']=cp.where(accepted[:,None],result['values'],cp.nan)
        if moments is not None:
            count,mean,m2,_=moments
            standard_deviation=cp.sqrt(cp.maximum(m2,0.)/cp.maximum(count-1,1)[:,None])
            dispersion=cp.max(standard_deviation/self.observable_scales,axis=1)
            result['candidate_count']=count
            result['observable_dispersion']=cp.where(count>=2,dispersion,cp.inf)
        return result

    def evaluate_cohort_replay(self,inputs,candidate_indices):
        """Replay candidate screening, spread, and first-accept merge as one graph.

        Cold compilation is included in ``graph_compilation_seconds``. Cache
        entries retain private pools and use at most two fixed cohort/shape
        combinations per bank; ``clear_graph_cache`` releases both cache types.
        """
        from .gpu_replay_call import GpuReplayCall,signature
        candidates=np.asarray(candidate_indices)
        if candidates.ndim!=1 or (candidates.size and candidates.dtype.kind not in 'iu'):
            raise ValueError('Cohort candidate indices must be a one-dimensional host integer array')
        if (len(np.unique(candidates))!=len(candidates)
                or np.any(candidates<0) or np.any(candidates>=len(self.evaluators))):
            raise ValueError('Cohort candidate indices must be unique and in range')
        candidates=tuple(int(index) for index in candidates)
        if self.math is None:
            from .gpu_capture_math import CaptureMath
            self.math=CaptureMath()
        # Ranking is a captured static branch. Observable changes invalidate
        # this cache in configure_observables, rather than silently replaying
        # the old projection or scale values.
        key=(candidates,signature(inputs),self.candidate_ranking)
        cache=self.cohort_graph_cache
        if key not in cache:
            while len(cache)>=self.cohort_graph_cache_size:cache.popitem(last=False)[1].close()
            cache[key]=GpuReplayCall(_CompactCohortReplay(self,candidates),inputs)
            self.graph_compilation_seconds+=cache[key].compilation_seconds
            self.cohort_graph_compilation_seconds+=cache[key].compilation_seconds
        cache.move_to_end(key)
        output=cache[key].run(inputs)
        self.last_candidate_scores=output['scores'];self.last_candidate_dual=output['dual']
        self.last_candidates=output['candidates']
        return output['result']
