"""One-dimensional current-LP feasible ray proposal on the GPU.

All non-objective fluxes are scaled together. The objective coordinate is
held at its CURRENT finite box optimum. Every current equality, inequality
and bound restricts the scalar interval; the final original certificate is
mandatory. An empty interval is NOT a proof that the LP is infeasible.
"""
import numpy as np


class ComponentRescaling:
    """Causal coefficient-ratio proposal, preserving homogeneous components.

    Host setup identifies equality-connected reaction components. Current
    shared-row/previous shared-row coefficient ratios supply scale proposals,
    not biological assumptions or certificates. Conflicting ratios use 1.
    """
    def __init__(self,solver):
        s=solver;cp=s.cp;self.s=s
        tags=[];masks=[];offset=0
        for a,b,lo,hi,c,neq in s.full_problems:
            parent=np.arange(a.shape[1])
            def root(j):
                while parent[j]!=j:
                    parent[j]=parent[parent[j]];j=parent[j]
                return j
            for row in range(neq):
                cols=a.indices[a.indptr[row]:a.indptr[row+1]]
                if len(cols):
                    first=root(int(cols[0]))
                    for col in cols[1:]:parent[root(int(col))]=first
            roots=np.array([root(j) for j in range(a.shape[1])])
            _,groups=np.unique(roots,return_inverse=True)
            groups=groups+offset;offset=int(groups.max())+1
            tags.append(groups)
            rows=np.repeat(np.arange(a.shape[0]),np.diff(a.indptr))
            objective_rows=np.zeros(a.shape[0],dtype=bool)
            objective_rows[rows[c[a.indices]!=0]]=True
            masks.append((rows>=neq)&~objective_rows[rows])
        self.columns=cp.asarray(np.concatenate(tags),dtype=cp.int32)
        self.entry_groups=self.columns[s._full_assembled[0].indices]
        self.mask=cp.asarray(np.concatenate(masks))
        self.count=offset
        self.previous=None

    def snapshot(self):self.previous=self.s._full_assembled[0].data.copy()

    def propose(self,x):
        s=self.s;cp=s.cp;s.factor._context()
        current=s._full_assembled[0].data
        if self.previous is None or self.previous.shape!=current.shape:
            raise ValueError('Previous certified step coefficients required')
        valid=self.mask&(current!=0)&(self.previous!=0)
        ratio=self.previous/cp.where(valid,current,1.)
        valid&=cp.isfinite(ratio)&(ratio>0)
        low=cp.full(self.count,cp.inf);high=cp.full(self.count,-cp.inf)
        cp.minimum.at(low,self.entry_groups,cp.where(valid,ratio,cp.inf))
        cp.maximum.at(high,self.entry_groups,cp.where(valid,ratio,-cp.inf))
        consistent=cp.isfinite(low)&cp.isfinite(high)&((high-low)<=1e-10*cp.maximum(1.,cp.abs(high)))
        scale=cp.where(consistent,low,1.)
        answer=x*scale[self.columns].reshape(s.batch,s.full_n)
        return answer,dict(component_count=self.count,scaled_components=int(cp.sum(consistent&(scale!=1))),
            conflicting_components=int(cp.sum(cp.isfinite(low)&~consistent)),cpu_lp_calls=0,
            original_certificate_required=True)


def ray_proposal(solver,previous_x):
    s=solver;cp=s.cp;s.factor._context()
    if (not isinstance(previous_x,cp.ndarray) or previous_x.shape!=(s.batch,s.full_n)
            or previous_x.dtype!=cp.float64 or previous_x.device.id!=s.factor.device):
        raise ValueError('Exact original-coordinate FP64 prior GPU primal required')
    a,rowlo,rhs,lo,hi,c=s._full_assembled
    c=c.reshape(s.batch,s.full_n);lo=lo.reshape(c.shape);hi=hi.reshape(c.shape)
    if not bool(cp.all(cp.sum(c!=0,axis=1)==1)&cp.all(cp.isfinite(previous_x))):
        raise ValueError('Single objective and finite previous primal required')
    j=cp.argmax(cp.abs(c),axis=1);lanes=cp.arange(s.batch)
    target=cp.where(c[lanes,j]<0,hi[lanes,j],lo[lanes,j])
    if not bool(cp.all(cp.isfinite(target))):raise ValueError('Finite box optimum required')
    anchor=cp.zeros_like(previous_x);anchor[lanes,j]=target
    direction=previous_x.copy();direction[lanes,j]=0.
    aa=(a@anchor.ravel()).reshape(s.batch,s.full_m)
    ad=(a@direction.ravel()).reshape(s.batch,s.full_m)
    b=rhs.reshape(aa.shape)-aa
    # Proposal equality band is STRICTER than the final 1e-5 original gate.
    q=s.full_neq
    coefficients=cp.concatenate((ad,-ad[:,:q],direction,-direction),axis=1)
    # Do not let harmless roundoff on a zero-fixed coordinate collapse the
    # proposal interval to alpha=0. This band is 100x tighter than the final
    # original primal gate; no model buffer or final threshold is changed.
    limits=cp.concatenate((b+1e-7,-b[:,:q]+1e-7,hi-anchor+1e-7,anchor-lo+1e-7),axis=1)
    quotient=limits/cp.where(coefficients!=0,coefficients,1.)
    lower=cp.maximum(0.,cp.max(cp.where(coefficients<0,quotient,-cp.inf),axis=1))
    upper=cp.minimum(1.,cp.min(cp.where(coefficients>0,quotient,cp.inf),axis=1))
    valid=(lower<=upper)&cp.all(cp.where(coefficients==0,limits>=0,True),axis=1)
    alpha=cp.where(valid,(lower+upper)*.5,1.)
    candidate=anchor+alpha[:,None]*direction
    limiter=cp.argmin(cp.where(coefficients>0,quotient,cp.inf),axis=1)
    return candidate,dict(valid_interval=cp.asnumpy(valid).tolist(),lower=cp.asnumpy(lower).tolist(),
        upper=cp.asnumpy(upper).tolist(),alpha=cp.asnumpy(alpha).tolist(),cpu_lp_calls=0,
        upper_limiter_augmented_index=cp.asnumpy(limiter).tolist(),
        upper_limiter_coefficient=cp.asnumpy(coefficients[lanes,limiter]).tolist(),
        upper_limiter_limit=cp.asnumpy(limits[lanes,limiter]).tolist(),
        empty_interval_is_not_LP_infeasibility=True,original_certificate_required=True)
