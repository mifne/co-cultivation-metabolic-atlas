"""Learned initialization + exact mechanistic LP certificate (AMN-inspired).

This is a basis-routing adaptation, not Faure et al.'s AMN-LP/QP reproduction.
Confidence never accepts a flux; a rejected proposal uses the original bank.
Host group dispatch remains explicit; optimization and inference use the GPU.
"""
import hashlib,json
import numpy as np

FIELDS=('rhs','lower','upper','c','delta','col_scale','row_scale')


def bank_identity(bank):
    digest=hashlib.sha256()
    for evaluator in bank.evaluators:
        anchor=evaluator.anchor;p=anchor['lp']
        for value in (p.a.indptr,p.a.indices,p.a.data,p.rhs,p.lower,p.upper,p.c,p.col_scale,p.row_scale,
                      anchor['basic'],anchor['active'],anchor['kind'],anchor['row_kind']):
            array=np.ascontiguousarray(value)
            digest.update(str((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def features(inputs):
    import cupy as cp
    joined=cp.concatenate([inputs[k].reshape(len(inputs[k]),-1) for k in FIELDS],axis=1)
    # Finite encoding is only for proposals; original unmodified inputs must
    # still pass the independent certificate, including nonfinite checks.
    joined=cp.nan_to_num(joined,nan=0.,posinf=1e13,neginf=-1e13)
    return (cp.sign(joined)*cp.log1p(cp.abs(joined))).astype(cp.float32)


class NeuralBasisProposal:
    def __init__(self,path,bank):
        import cupy as cp
        with np.load(path,allow_pickle=False) as data:
            self.metadata=json.loads(str(data['metadata']))
            if self.metadata['bank_identity']!=bank_identity(bank):raise ValueError('Neural bank identity mismatch')
            self.arrays={key:cp.asarray(data[key]) for key in ('indices','mean','scale','w1','b1','w2','b2')}
        a=self.arrays;d=len(a['indices']);h=len(a['b1']);k=len(bank.evaluators)
        if (a['indices'].ndim!=1 or a['indices'].dtype.kind not in 'iu' or
            not bool(((a['indices']>=0)&(a['indices']<self.metadata['feature_width'])).all().get())):
            raise ValueError('Invalid neural feature indices')
        if (a['w1'].shape!=(d,h) or a['w2'].shape!=(h,k) or a['b2'].shape!=(k,) or
            a['mean'].shape!=(d,) or a['scale'].shape!=(d,) or not bool((a['scale']>0).all().get())):
            raise ValueError('Invalid neural artifact dimensions/scaling')
        if any(not bool(cp.isfinite(value).all().get()) for value in a.values()):raise ValueError('Nonfinite neural artifact')

    def logits(self,inputs):
        import cupy as cp
        a=self.arrays;x=features(inputs)
        if x.shape[1]!=self.metadata['feature_width']:raise ValueError('Neural feature layout changed')
        x=(x[:,a['indices']]-a['mean'])/a['scale']
        hidden=cp.tanh(x@a['w1']+a['b1'])
        return hidden@a['w2']+a['b2']

    def predict(self,inputs):
        import cupy as cp
        return cp.argmax(self.logits(inputs),axis=1)

    def rank(self,inputs):
        import cupy as cp
        return cp.argsort(-self.logits(inputs),axis=1)


class NeuralRoutedBasisBank:
    def __init__(self,base,proposal,defer_exhaustive=False):
        self.base=base;self.proposal=proposal
        self.defer_exhaustive=defer_exhaustive
        for name in ('evaluators','root','cp','offsets','offline_cpu_lp_calls'):
            setattr(self,name,getattr(base,name))

    def prepare_host(self,problems):return self.base.prepare_host(problems)

    def evaluate_device(self,**inputs):
        cp=self.cp;batch,n=inputs['lower'].shape
        predicted=self.proposal.predict(inputs).get()
        metrics=('primal_residual','dual_violation','relative_kkt_gap')
        result=dict(accepted=cp.zeros(batch,dtype=bool),values=cp.full((batch,n),cp.nan),
            objective=cp.full(batch,cp.nan),candidate_index=cp.full(batch,-1,dtype=cp.int32),
            best_candidate_index=cp.zeros(batch,dtype=cp.int32),
            best_candidate_dual_feasible=cp.zeros(batch,dtype=bool),
            best_rejected_metrics={k:cp.full(batch,cp.inf) for k in metrics},
            **{k:cp.full(batch,cp.inf) for k in metrics})
        for index in np.unique(predicted):
            ids=cp.asarray(np.flatnonzero(predicted==index))
            candidate=self.evaluators[index]
            subset={k:v[ids] for k,v in inputs.items()}
            proposed=candidate.evaluate_device(**dict(subset,delta=subset['delta']-self.offsets[index]))
            for key in ('accepted','values','objective',*metrics):result[key][ids]=proposed[key]
            result['candidate_index'][ids]=cp.where(proposed['accepted'],index,-1)
            result['best_candidate_index'][ids]=index
            result['best_candidate_dual_feasible'][ids]=proposed['basis_dual_violation']<=1e-8
        pending=np.flatnonzero(~result['accepted'].get())
        proposal_accepts=batch-len(pending)
        if len(pending) and not self.defer_exhaustive:
            ids=cp.asarray(pending)
            fallback=self.base.evaluate_device(**{k:v[ids] for k,v in inputs.items()})
            for key in ('accepted','values','objective','candidate_index','best_candidate_index',
                        'best_candidate_dual_feasible',*metrics):result[key][ids]=fallback[key]
            for key in metrics:result['best_rejected_metrics'][key][ids]=fallback['best_rejected_metrics'][key]
        result.update(cpu_lp_calls=0,neural_proposal_accepts=proposal_accepts,
            exhaustive_fallback_rows=0 if self.defer_exhaustive else len(pending),
            neural_deferred_exhaustive=self.defer_exhaustive,
            scope='Neural proposal + original LP certification; host dispatch')
        return result


class ConstantBasisProposal:
    """Ablation: same repair-first schedule without any learned prediction."""
    def predict(self,inputs):
        import cupy as cp
        return cp.zeros(len(inputs['lower']),dtype=cp.int32)
