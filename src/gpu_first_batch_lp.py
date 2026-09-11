"""Strict GPU-first B x K screening for every original community LP stage.

No CPU LP runs concurrently or supplies GPU proposals. All selected candidates
are reconstructed and certified on CUDA, with one packed result download per
stage. Exact CPU work is submitted only for rejected rows, or forbidden by an
explicit reject policy. Host LP assembly/state update remain outside this layer.
"""
from types import SimpleNamespace
import time

import numpy as np

from .gpu_candidate_transfer import download_candidate, METRICS
from .temporal_candidate_order import temporal_candidate_order_device


def device_candidate_order(cp, bank, inputs, router=None):
    """Rank only; neither a network score nor a distance accepts a solution."""
    encoded = bank.proposal_features(inputs)
    if router is not None:
        order, valid = router.rank_with_validity(encoded, k=len(bank.evaluators))
        label = 'learned_coverage'
    else:
        distance = cp.sum(((encoded[:, None]-bank.centers[None])
                           / bank.feature_scale)**2, axis=2)
        valid = cp.isfinite(encoded).all(axis=1) & cp.isfinite(distance).all(axis=1)
        order = cp.argsort(cp.where(cp.isfinite(distance), distance, cp.inf), axis=1)
        label = 'nearest_centroid'
    batch = len(encoded)
    if (order.shape != (batch, len(bank.evaluators)) or order.dtype.kind not in 'iu'
            or valid.shape != (batch,) or valid.dtype.kind != 'b'):
        raise ValueError('Malformed device candidate order or validity mask')
    return order, valid, label


def solve_gpu_first_batch(service, requests, key, arrays, ids, started):
    import cupy as cp
    if key[0] not in ('maxmin', 'aggregate', 'exchange') or service.repair_rounds:
        raise ValueError('GPU-first batch requires an original stage and zero legacy repair')
    if any(kwargs.get('integrality') is not None and np.any(np.asarray(kwargs['integrality']))
           for _, kwargs in requests):
        raise ValueError('GPU-first certificates support continuous LPs only')
    policy = service.gpu_cpu_fallback
    if policy not in ('exact', 'reject'):
        raise ValueError('Unknown GPU-first CPU fallback policy')
    batch = len(requests)
    bank = service.banks[key]
    engine = service.heterogeneous_banks[key]
    if not engine.certificate_only:
        raise ValueError('GPU-first batch requires strict certificate-only evaluation')
    gpu_queued = False
    try:
        problems = [service.coordinates.normalize(*a) for a in arrays]
        service.last_problems = problems
        normalized_at = time.perf_counter()
        gpu_queued = True
        inputs = bank.prepare_host(problems)
        prepared_at = time.perf_counter()
        order, valid, label = device_candidate_order(
            cp, bank, inputs, service.coverage_routers.get(key))
        prior = np.array([service._gpu_first_previous.get((key, env), -1) for env in ids])
        order = temporal_candidate_order_device(cp, order, prior,
            service.candidate_limit, service.temporal_candidate_policy)
        ranked_at = time.perf_counter()
        evaluate = engine.evaluate_replay if service.heterogeneous_replay else engine.evaluate
        candidate = evaluate(inputs, order)
        if candidate['accepted'].shape != (batch,) or candidate['accepted'].dtype.kind != 'b':
            raise ValueError('GPU certificate acceptance must be a boolean vector')
        candidate = dict(candidate, accepted=candidate['accepted'] & valid)
        evaluated_at = time.perf_counter()
        gpu_accepted, selected, values, objective, metrics = download_candidate(
            cp, candidate, batch, len(requests[0][0]), len(bank.evaluators), packed=True)
        finite = (np.isfinite(values).all(axis=1) & np.isfinite(objective)
                  & (selected >= 0) & (selected < len(bank.evaluators)))
        for name, threshold in zip(METRICS, (1e-5, 1e-7, 1e-7)):
            finite &= np.isfinite(metrics[name]) & (metrics[name] >= 0.) & (metrics[name] <= threshold)
        gpu_accepted &= finite
        cp.cuda.get_current_stream().synchronize()
        downloaded_at = time.perf_counter()
    except BaseException as error:
        if gpu_queued:
            try:
                cp.cuda.get_current_stream().synchronize()
            except BaseException as cleanup:
                raise BaseExceptionGroup('GPU-first batch and stream cleanup failed', [error, cleanup])
        raise

    # No speculative CPU work exists. This exact service uses the SAME offline
    # dictionary/router initializer as the CPU comparator, not reference x/y.
    rejected = np.flatnonzero(~gpu_accepted).tolist()
    cpu_rows = []
    accepted = gpu_accepted.copy()
    routes = ['gpu_dictionary_batch' if good else 'gpu_uncertified_no_cpu' for good in accepted]
    if rejected and policy == 'exact':
        cpu_rows = service.direct_cpu.solve_batch([requests[i] for i in rejected],
            environment_ids=[ids[i] for i in rejected])
        if len(cpu_rows) != len(rejected):
            raise RuntimeError('CPU fallback returned an incomplete rejected batch')
        for i, row in zip(rejected, cpu_rows):
            accepted[i] = bool(row.success)
            routes[i] = 'cpu_after_gpu_certificate'
            if row.success:
                values[i], objective[i] = row.x, row.fun
            for field in metrics:
                metrics[field][i] = row.diagnostics[field]
    completed_at = time.perf_counter()
    for i, env in enumerate(ids):
        # Rejected candidates / exact CPU solutions never seed this GPU cache.
        service._gpu_first_previous[(key, env)] = int(selected[i]) if gpu_accepted[i] else -1

    cpu_diagnostics = [dict(row.diagnostics, cpu_result_used=True) for row in cpu_rows]
    cpu_positions = rejected if cpu_rows else []
    record = dict(stage=key[0], batch=batch, environment_ids=list(ids),
        gpu_first_batch=True, gpu_cpu_fallback=policy,
        candidate_router=label, candidate_engine='gpu_first_heterogeneous',
        temporal_candidate_policy=service.temporal_candidate_policy,
        certificate_only=True, device_routing=True, packed_result_transfer=True,
        candidate_evaluations=int(candidate['candidate_evaluations']),
        bank_accepts=int(gpu_accepted.sum()), gpu_nonbank_accepts=0,
        cpu_lp_calls=len(cpu_rows), cpu_results_used=len(cpu_rows),
        cpu_speculative_unused=0, cpu_speculative_cancelled=0,
        cpu_solver_runs=sum(r.get('cpu_solver_runs', 1) for r in cpu_diagnostics),
        numerical_retry_count=sum(r.get('numerical_retry_count', 0) for r in cpu_diagnostics),
        preparation_seconds=prepared_at-started, bank_seconds=downloaded_at-prepared_at,
        seconds=completed_at-started, routes=routes, accepted=accepted.tolist(),
        candidate_count=[None]*batch, observable_dispersion=[None]*batch,
        repair_policy='gpu_first_no_legacy_repair', repair_eligible=[False]*batch,
        cpu_initial_basis_proposals=bool(cpu_rows),
        groups=[dict(ids=cpu_positions, environment_ids=[ids[i] for i in cpu_positions],
                     route='cpu_fallback', rows=cpu_diagnostics)],
        gpu_first_phases=dict(parse_normalize_seconds=normalized_at-started,
            stack_upload_seconds=prepared_at-normalized_at,
            device_rank_seconds=ranked_at-prepared_at,
            evaluation_submission_seconds=evaluated_at-ranked_at,
            completion_download_validation_seconds=downloaded_at-evaluated_at,
            rejected_cpu_seconds=completed_at-downloaded_at),
        timing_scope='Nonoverlapping host-observed intervals; not isolated CUDA kernel time. No CPU LP before GPU validation.',
        **{name:value.tolist() for name, value in metrics.items()})
    service.history.append(record)
    return [SimpleNamespace(success=bool(accepted[i]),
        x=values[i].copy() if accepted[i] else None,
        fun=float(objective[i]) if accepted[i] else None, message=routes[i],
        diagnostics=dict(environment_id=ids[i], stage=key[0], success=bool(accepted[i]),
            route=routes[i], cpu_lp_calls=int(i in cpu_positions),
            total_seconds=record['seconds'],
            **{name:float(value[i]) for name, value in metrics.items()})) for i in range(batch)]
