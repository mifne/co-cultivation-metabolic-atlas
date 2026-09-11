"""Measured host bridge for compact GPU-only LP maps and tie certificates."""
import time
from types import SimpleNamespace
import numpy as np
from .gpu_compiled_community_backend import lp_arrays,stage_key


def certify_same_basis_tie(bank,primary,primary_inputs,secondary_inputs,secondary,primary_bound):
    cp=bank.cp;rs=secondary_inputs['row_scale'];cs=secondary_inputs['col_scale'];neq=bank.neq
    x=primary['raw_values'];rp=primary['raw_reduced'];r2=secondary['raw_reduced']
    yp=primary['raw_y'];y2=secondary['raw_y'];lo=secondary_inputs['lower'];hi=secondary_inputs['upper']
    nonzero=cp.abs(rp*cs)>1e-8
    ratio=cp.where(nonzero&((rp*r2)<0),-r2/cp.where(nonzero,rp,1.),0.)
    multiplier=cp.maximum(cp.max(ratio,axis=1),0.)
    row_nonzero=(yp[:,neq:]*rs[:,neq:]) < -1e-8
    row_ratio=cp.where(row_nonzero&(y2[:,neq:]>0),-y2[:,neq:]/cp.where(row_nonzero,yp[:,neq:],1.),0.)
    if row_ratio.shape[1]:multiplier=cp.maximum(multiplier,cp.max(row_ratio,axis=1))
    multiplier=multiplier*(1+1e-10)+1e-12
    y=y2+multiplier[:,None]*yp;reduced=r2+multiplier[:,None]*rp
    primary_value=cp.sum(primary_inputs['c']*x,axis=1)
    primal=cp.maximum(primary['primal_residual'],cp.maximum(primary_value-primary_bound,0.))
    rc=reduced*cs
    dual=cp.maximum(cp.max(cp.maximum(cp.where(~cp.isfinite(lo),rc,0.),cp.where(~cp.isfinite(hi),-rc,0.)),axis=1),0.)
    if y.shape[1]>neq:dual=cp.maximum(dual,cp.max((y*rs)[:,neq:],axis=1))
    target=cp.where(reduced>=0,lo,hi)
    gap=cp.sum(cp.abs(reduced*(x-cp.where(cp.isfinite(target),target,x))),axis=1)
    gap+=cp.sum(cp.abs(y[:,neq:]*(secondary_inputs['rhs']-primary['activity'])[:,neq:]),axis=1)
    gap+=cp.abs(multiplier*(primary_bound-primary_value))
    objective=cp.sum(secondary_inputs['c']*x,axis=1);gap/=cp.maximum(1.,cp.abs(objective))
    valid=primary['accepted']&secondary['input_family_valid']&cp.isfinite(multiplier)&cp.isfinite(gap)
    for name in ('rhs','lower','upper','col_scale','row_scale','delta'):
        axes=tuple(range(1,primary_inputs[name].ndim))
        valid&=(primary_inputs[name]==secondary_inputs[name]).all(axis=axes)
    valid&=(primary_bound>=primary['objective'])&(primal<=1e-5)&(dual<=1e-7)&(gap<=1e-7)
    return dict(accepted=valid,values=cp.where(valid[:,None],x/cs,cp.nan),objective=objective,
        primal_residual=primal,dual_violation=dual,relative_kkt_gap=gap,multiplier=multiplier,cpu_lp_calls=0)


class CompactBackend:
    def __init__(self,coordinates,banks):
        self.coordinates=coordinates;self.banks=banks;self.history=[];self.previous=None

    def solve_batch(self,requests):
        import cupy as cp
        started=time.perf_counter();arrays=[lp_arrays(c,**kw) for c,kw in requests]
        assembled=time.perf_counter()
        keys=[stage_key(a[4],a[0],a[-1],self.coordinates.n_fluxes) for a in arrays]
        if len(set(keys))!=1:raise ValueError('Mixed LP stages')
        key=keys[0];stage=key[0]
        if stage=='exchange_tie':
            if self.previous is None:raise ValueError('Missing compact primary exchange')
            bank,primary,previous_inputs,previous_arrays=self.previous
            for a,p in zip(arrays,previous_arrays):
                if not np.array_equal(a[0][-1].toarray().ravel(),p[4]):raise ValueError('Primary objective changed')
            problems=[self.coordinates.normalize(a[:-1],r[:-1],l,u,c,n) for a,r,l,u,c,n in arrays]
            inputs=bank.prepare_host(problems)
            prepared=time.perf_counter()
            order=primary['candidate_index'].get()[:,None]
            secondary=bank.evaluate_device(inputs,order=order)
            result=certify_same_basis_tie(bank,primary,previous_inputs,inputs,secondary,cp.asarray([a[1][-1] for a in arrays]))
            self.previous=None
        else:
            bank=self.banks[stage];problems=[self.coordinates.normalize(*a) for a in arrays]
            self.last_problems=problems
            inputs=bank.prepare_host(problems);prepared=time.perf_counter();result=bank.evaluate_device(inputs)
            if stage=='exchange':self.previous=(bank,result,inputs,arrays)
        cp.cuda.get_current_stream().synchronize();evaluated=time.perf_counter()
        ok=result['accepted'].get();values=result['values'].get();objectives=result['objective'].get()
        record=dict(stage=stage,batch=len(arrays),accepted=ok.tolist(),seconds=time.perf_counter()-started,
            array_assembly_seconds=assembled-started,normalize_and_upload_seconds=prepared-assembled,
            evaluation_seconds=evaluated-prepared,download_seconds=time.perf_counter()-evaluated,
            candidate_evaluations=result.get('candidate_evaluations',len(arrays)),cpu_lp_calls=0,
            **{k:result[k].get().tolist() for k in ('primal_residual','dual_violation','relative_kkt_gap')})
        if 'input_family_valid' in result:record['input_family_valid']=result['input_family_valid'].get().tolist()
        self.history.append(record)
        return [SimpleNamespace(success=bool(ok[i]),x=values[i] if ok[i] else None,fun=float(objectives[i]) if ok[i] else None,
            message=str(record),diagnostics=dict(success=bool(ok[i]),max_original_residual=record['primal_residual'][i],
                total_seconds=record['seconds'],objective=float(objectives[i]),cpu_lp_calls=0)) for i in range(len(arrays))]
