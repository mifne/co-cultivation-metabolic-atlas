"""Temporal proposal ordering only; never certify or reuse an LP answer."""
import numpy as np


POLICIES = ('prepend', 'within-budget', 'off')


def temporal_candidate_order(order, previous, limit=0, policy='prepend'):
    """Preserve the learned top-K set when using ``within-budget``.

    A preceding solution generally expires when the current LP changes. The
    legacy prepend policy can evict a stronger current-query proposal. This
    function reorders IDs only; every proposed solution must be reconstructed
    and certified against the CURRENT full LP.
    """
    if policy not in POLICIES:
        raise ValueError('Unknown temporal candidate policy')
    order = np.asarray(order)
    previous = np.asarray(previous)
    if (order.ndim != 2 or order.dtype.kind not in 'iu'
            or previous.shape != (len(order),) or previous.dtype.kind not in 'iu'):
        raise ValueError('Integer candidate order and one previous ID per row required')
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError('Candidate limit must be a nonnegative integer')
    width = min(int(limit), order.shape[1]) if limit else order.shape[1]
    selected = order[:, :width].copy() if policy != 'prepend' else order.copy()
    if policy != 'off':
        for i, candidate in enumerate(previous):
            # An out-of-range/stale ID must never manufacture an extra slot.
            position = np.flatnonzero(selected[i] == candidate)
            if candidate >= 0 and len(position) == 1:
                j = int(position[0])
                selected[i, 1:j+1] = selected[i, :j].copy()
                selected[i, 0] = candidate
    return selected[:, :width]


def temporal_candidate_order_device(cp, order, previous, limit=0, policy='prepend'):
    """Same shortlist on-device; no rank download or host scalar synchronization."""
    if policy not in POLICIES:
        raise ValueError('Unknown temporal candidate policy')
    if isinstance(limit,bool) or not isinstance(limit,(int,np.integer)) or limit<0:
        raise ValueError('Candidate limit must be a nonnegative integer')
    previous=cp.asarray(previous)
    if (order.ndim!=2 or order.dtype.kind not in 'iu' or previous.shape!=(len(order),)
            or previous.dtype.kind not in 'iu'):
        raise ValueError('Integer candidate order and one previous ID per row required')
    width=min(int(limit),order.shape[1]) if limit else order.shape[1]
    selected=order[:,:width] if policy!='prepend' else order
    if policy!='off':
        keys=cp.where((selected==previous[:,None])&(previous[:,None]>=0),
            -1,cp.arange(selected.shape[1])[None])
        selected=cp.take_along_axis(selected,cp.argsort(keys,axis=1),axis=1)
    return selected[:,:width]
