"""Paired, two-way cluster bootstrap over training seeds and held-out blocks.

A block is an independently generated trajectory or a fixed GPU trajectory
batch. Never split one batched timing into invented per-trajectory timings.
Timing repeats must be aggregated before this function (e.g. median of three).
"""
import numpy as np
from .graph_training_collection import scaling_decision


def evaluate_curve(document,*,resamples=2000,seed=1741):
    if resamples<200:raise ValueError('At least 200 bootstrap resamples required')
    comparison=document.get('fixed_comparison_id')
    if not isinstance(comparison,str) or not comparison:raise ValueError('Fixed comparison identifier required')
    if document.get('timing_unit') not in ('independent_trajectory','independent_trajectory_batch'):
        raise ValueError('Explicit independent timing unit required')
    rng=np.random.default_rng(seed);points=[];previous=None;seen_sizes=set()
    for item in sorted(document['sizes'],key=lambda r:r['train_count']):
        n=item['train_count']
        if type(n) is not int or n<1 or n in seen_sizes:raise ValueError('Unique positive sizes required')
        seen_sizes.add(n);rows=item['measurements']
        seeds=sorted({r['training_seed'] for r in rows});blocks=sorted({r['block_id'] for r in rows})
        if len(blocks)<3:raise ValueError('At least three independent evaluation blocks required')
        indexed={(r['training_seed'],r['block_id']):r for r in rows}
        if len(indexed)!=len(rows) or len(rows)!=len(seeds)*len(blocks):raise ValueError('Complete seed/block grid without duplicates required')
        current=np.array([[indexed[s,b]['total_seconds'] for b in blocks] for s in seeds],dtype=float)
        baseline=np.array([[indexed[s,b]['baseline_seconds'] for b in blocks] for s in seeds],dtype=float)
        if not (np.isfinite(current).all() and np.isfinite(baseline).all() and (current>0).all() and (baseline>0).all()):
            raise ValueError('Finite positive complete pipeline times required')
        if any(r.get('timing_repeats',0)<3 for r in rows):raise ValueError('At least three timing repeats per cell required')
        si=rng.integers(len(seeds),size=(resamples,len(seeds),1))
        bi=rng.integers(len(blocks),size=(resamples,1,len(blocks)))
        denominator=current[si,bi].sum(axis=(1,2))
        speedup=baseline[si,bi].sum(axis=(1,2))/denominator
        p=dict(train_count=n,training_seeds=len(seeds),evaluation_blocks=len(blocks),
            fixed_comparison_id=comparison,converged=item.get('converged',False),
            all_accuracy_gates_passed=all(r.get('all_accuracy_gates_passed',False) is True for r in rows),
            speedup=float(baseline.sum()/current.sum()),speedup_ci95=np.quantile(speedup,[.025,.975]).tolist())
        if previous is not None:
            ps,pb,pt,base=previous
            if seeds!=ps or blocks!=pb:raise ValueError('Use identical training seeds and evaluation blocks at all sizes')
            if not np.array_equal(baseline,base):raise ValueError('Use the same independently measured baseline across sizes')
            gain=1.-denominator/pt[si,bi].sum(axis=(1,2))
            p['relative_gain_upper_ci95']=float(np.quantile(gain,.975))
            p['relative_gain_ci95']=np.quantile(gain,[.025,.975]).tolist()
        points.append(p);previous=(seeds,blocks,current,baseline)
    return dict(points=points,decision=scaling_decision(points),resamples=resamples,bootstrap_seed=seed,
        method='paired two-way cluster bootstrap: training seeds and independent evaluation blocks',
        caveat='Conditional on declared virtual domain, fixed model and accuracy gates; 2% plateau is a predeclared engineering threshold, not proof of a universal sample maximum')
