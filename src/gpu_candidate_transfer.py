"""One compact result download; ORIGINAL LP validation remains authoritative."""
import numpy as np


METRICS = ('primal_residual','dual_violation','relative_kkt_gap')


def download_candidate(cp, candidate, batch, variables, entries, *, packed=False, deferred=False):
    """Return independent host observations without seven blocking downloads.

    Shape/dtype checks use device metadata only. In packed mode indices outside
    the bank are mapped to -1 before float64 packing (no precision loss for a
    valid small dictionary ID). Invalid flags, values and certificate metrics
    remain invalid; the caller must apply all original host acceptance gates.
    ``deferred=True`` enqueues packing now and returns a single-use-in-pipeline
    download callable. Call it before the next evaluation overwrites graph
    outputs. This lets ALL GPU preparation precede competing CPU workers.
    """
    accepted,selected = candidate['accepted'],candidate['candidate_index']
    values,objective = candidate['values'],candidate['objective']
    metrics = {name:candidate[name] for name in METRICS}
    if accepted.shape != (batch,) or accepted.dtype.kind != 'b':
        raise ValueError('GPU certificate acceptance must be a boolean vector')
    if (selected.shape != (batch,) or selected.dtype.kind not in 'iu'
            or values.shape != (batch,variables) or objective.shape != (batch,)
            or any(v.shape != (batch,) for v in metrics.values())):
        raise ValueError('Malformed GPU candidate dimensions or index dtype')
    if not packed:
        def download():
            return (accepted.get(),selected.get(),values.get(),objective.get(),
                {name:value.get() for name,value in metrics.items()})
        return download if deferred else download()
    if (isinstance(entries,bool) or not isinstance(entries,(int,np.integer))
            or not 0<entries<2**53):
        raise ValueError('Packed transfer requires exactly representable candidate IDs')
    if any(v.dtype != np.dtype('float64') for v in (values,objective,*metrics.values())):
        raise ValueError('Packed original LP values and metrics must be float64')
    safe_selected=cp.where((selected>=0)&(selected<entries),selected,-1)
    table=cp.concatenate((values,objective[:,None],
        *(metrics[name][:,None] for name in METRICS),
        accepted[:,None].astype(cp.float64),safe_selected[:,None].astype(cp.float64)),axis=1)
    def download():
        host=table.get()
        return (host[:,variables+4].astype(bool),host[:,variables+5].astype(np.int64),
            host[:,:variables],host[:,variables],
            {name:host[:,variables+i+1] for i,name in enumerate(METRICS)})
    return download if deferred else download()
