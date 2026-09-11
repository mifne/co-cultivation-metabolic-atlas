"""Overlap exact CPU LPs with independently certified GPU dictionary probes.

This is an opt-in hybrid scheduling experiment, not a GPU-only solver. CPU
answers never enter the GPU proposal or certificate. Cancellation only applies
to queued CPU tasks whose corresponding GPU solution passed the ORIGINAL LP
certificate; every running task is joined before returning or raising.
"""
from types import SimpleNamespace
import time

import numpy as np


def solve_speculative_maxmin(service, requests, key, arrays, ids, started):
    import cupy as cp
    from .gpu_hybrid_lp import rank_compact_candidates
    from .speculative_cpu_batch import SpeculativeCpuBatch
    from .temporal_candidate_order import temporal_candidate_order,temporal_candidate_order_device
    from .gpu_candidate_transfer import download_candidate

    if key[0]!='maxmin' or service.repair_rounds != 0:
        raise ValueError('Speculation supports only the original maxmin stage without repair')
    batch = len(requests)
    # Use the SAME dictionary initializer and persistent CPU pool as the
    # strongest matched comparator, not the rejected GPU candidate basis.
    phase=getattr(service,'speculative_dispatch_phase','before-prepare')
    if phase not in ('before-prepare','after-submit'):
        raise ValueError('Unknown speculative CPU dispatch phase')
    pending = None
    delayed_cpu_prepare_seconds=0.
    gpu_work_may_be_queued=False
    try:
        if phase=='before-prepare':
            pending=SpeculativeCpuBatch(service.direct_cpu,requests,ids)
        gpu_started = time.perf_counter()
        bank = service.banks[key]
        problems = [service.coordinates.normalize(*a) for a in arrays]
        normalized_at = time.perf_counter()
        service.last_problems = problems
        gpu_work_may_be_queued=True
        inputs = bank.prepare_host(problems)
        prepared = time.perf_counter()
        prior=np.array([service._speculative_previous.get((key,env),-1) for env in ids])
        router_valid=None
        if getattr(service,'device_routing',False):
            router=service.coverage_routers[key]
            order,router_valid=router.rank_with_validity(bank.proposal_features(inputs),k=len(bank.evaluators))
            if (order.shape!=(batch,len(bank.evaluators)) or order.dtype.kind not in 'iu'
                    or router_valid.shape!=(batch,) or router_valid.dtype.kind!='b'):
                raise ValueError('Malformed device router order or validity mask')
            order=temporal_candidate_order_device(cp,order,prior,service.candidate_limit,
                getattr(service,'temporal_candidate_policy','prepend'))
            candidate_router='learned_coverage'
        else:
            order,candidate_router=rank_compact_candidates(
                cp,bank,inputs,getattr(service,'coverage_routers',{}).get(key))
            order = temporal_candidate_order(order,prior,
                service.candidate_limit, getattr(service,'temporal_candidate_policy','prepend'))
        ranked_at = time.perf_counter()
        engine = service.heterogeneous_banks.get(key)
        if engine is not None:
            evaluate = engine.evaluate_replay if service.heterogeneous_replay else engine.evaluate
            candidate = evaluate(inputs,cp.asarray(order))
        elif service.cohort_candidates:
            popularity = np.bincount(order.ravel(),minlength=len(bank.evaluators))
            candidates = np.flatnonzero(popularity)
            candidates = candidates[np.argsort(-popularity[candidates],kind='stable')]
            evaluate = bank.evaluate_cohort_replay if service.cohort_replay else bank.evaluate_cohort
            candidate = evaluate(inputs,candidates)
        else:
            candidate = bank.evaluate_device(inputs,order=order)
        evaluated_at = time.perf_counter()
        # These are full original-LP certificates, NOT a variance cutoff,
        # network confidence or the CPU solver's result on the same input.
        if router_valid is not None:
            if candidate['accepted'].shape!=(batch,) or candidate['accepted'].dtype.kind!='b':
                raise ValueError('GPU certificate acceptance must be a boolean vector')
            # An invalid NN feature/logit row cannot authorize cancellation,
            # even if an incidental selected dictionary candidate certifies.
            candidate=dict(candidate,accepted=candidate['accepted']&router_valid)
        download=download_candidate(
            cp,candidate,batch,len(requests[0][0]),len(bank.evaluators),
            packed=getattr(service,'packed_result_transfer',False),deferred=True)
        submitted_at=time.perf_counter()
        if phase=='after-submit':
            # Packing is queued too: issuing its kernels after CPU dispatch
            # would reintroduce repeated GIL handovers despite a single .get.
            pending=SpeculativeCpuBatch(service.direct_cpu,requests,ids)
            delayed_cpu_prepare_seconds=time.perf_counter()-submitted_at
        gpu_accepted,selected,values,objective,metrics=download()
        # Fail closed if a malformed evaluator contradicts its own flags.
        finite = (np.isfinite(values).all(axis=1) & np.isfinite(objective)
                  & (selected>=0) & (selected<len(bank.evaluators)))
        for field,limit in zip(metrics,(1e-5,1e-7,1e-7)):
            finite &= np.isfinite(metrics[field]) & (metrics[field]>=0.) & (metrics[field]<=limit)
        gpu_accepted &= finite
        cp.cuda.get_current_stream().synchronize()
        gpu_seconds = time.perf_counter()-gpu_started
        cpu_results,telemetry = pending.resolve(gpu_accepted)
    except BaseException as error:
        # No CPU thread may survive into reset, the next physical step or an
        # exception handler. A GPU exception is NOT hidden as a successful run.
        cleanup_errors=[]
        if pending is not None and not (pending.telemetry or {}).get('joined_all',False):
            try:
                pending.drain()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if gpu_work_may_be_queued:
            # A delayed CPU constructor or metadata check can fail after graph
            # work was enqueued but before the download. Drain the GPU stream
            # too before allowing reset/graph disposal or raising to a caller.
            try:
                cp.cuda.get_current_stream().synchronize()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise BaseExceptionGroup('GPU probe and worker/stream cleanup failed',[error,*cleanup_errors])
        raise

    accepted = gpu_accepted.copy()
    routes = ['gpu_dictionary_speculative' if ok else 'cpu_speculative_exact' for ok in gpu_accepted]
    cpu_rows,cpu_ids,cpu_positions = [],[],[]
    for i,result in enumerate(cpu_results):
        if result is not None:
            cpu_rows.append(dict(result.diagnostics,cpu_result_used=not bool(gpu_accepted[i])))
            cpu_ids.append(ids[i])
            cpu_positions.append(i)
        if not gpu_accepted[i]:
            if result is None:
                raise RuntimeError('Uncertified GPU candidate lost its required CPU result')
            accepted[i] = result.success
            if result.success:values[i],objective[i] = result.x,result.fun
            for field in metrics:metrics[field][i] = result.diagnostics[field]
        service._speculative_previous[(key,ids[i])] = int(selected[i]) if gpu_accepted[i] else -1

    unused = sum(bool(gpu_accepted[i]) and result is not None for i,result in enumerate(cpu_results))
    record = dict(stage=key[0],batch=batch,environment_ids=ids,
        candidate_router=candidate_router,
        temporal_candidate_policy=getattr(service,'temporal_candidate_policy','prepend'),
        certificate_only=getattr(service,'certificate_only',False),
        device_routing=getattr(service,'device_routing',False),
        packed_result_transfer=getattr(service,'packed_result_transfer',False),
        speculative_dispatch_phase=phase,
        candidate_engine='speculative_'+('heterogeneous' if engine is not None else 'cohort' if service.cohort_candidates else 'legacy'),
        candidate_evaluations=int(candidate['candidate_evaluations']),
        bank_accepts=int(gpu_accepted.sum()),gpu_nonbank_accepts=0,
        # Executed CPU work includes results superseded by certified GPU
        # values. These counts intentionally can overlap bank_accepts.
        cpu_lp_calls=len(cpu_rows),cpu_results_used=int((~gpu_accepted).sum()),
        cpu_speculative_unused=unused,cpu_speculative_cancelled=batch-len(cpu_rows),
        cpu_solver_runs=sum(r.get('cpu_solver_runs',1) for r in cpu_rows),
        numerical_retry_count=sum(r.get('numerical_retry_count',0) for r in cpu_rows),
        preparation_seconds=prepared-gpu_started,bank_seconds=gpu_seconds-(prepared-gpu_started),
        gpu_probe_span_seconds=gpu_seconds,speculative_cpu=telemetry,
        gpu_host_observed_phases=dict(normalize_seconds=normalized_at-gpu_started,
            stack_upload_seconds=prepared-normalized_at,rank_seconds=ranked_at-prepared,
            evaluate_submission_seconds=evaluated_at-ranked_at,
            pack_submission_seconds=submitted_at-evaluated_at,
            delayed_cpu_prepare_dispatch_seconds=delayed_cpu_prepare_seconds,
            completion_download_validation_seconds=gpu_started+gpu_seconds-submitted_at-delayed_cpu_prepare_seconds,
            scope='Sequential host-observed spans, not isolated CUDA kernel time; overlaps exact CPU workers.'),
        seconds=time.perf_counter()-started,routes=routes,accepted=accepted.tolist(),
        candidate_count=[None]*batch,observable_dispersion=[None]*batch,
        repair_policy='no_repair_speculative_cpu',repair_eligible=[False]*batch,
        cpu_initial_basis_proposals=True,
        groups=[dict(ids=cpu_positions,environment_ids=cpu_ids,route='cpu_fallback',rows=cpu_rows)],
        **{field:value.tolist() for field,value in metrics.items()})
    record['timing_scope']='CPU span overlaps GPU probe; do not sum spans. seconds includes prepare, dispatch, GPU, cancellation and joining ALL running CPU work.'
    service.history.append(record)
    return [SimpleNamespace(success=bool(accepted[i]),x=values[i].copy() if accepted[i] else None,
        fun=float(objective[i]) if accepted[i] else None,message=routes[i],
        diagnostics=dict(environment_id=ids[i],stage=key[0],success=bool(accepted[i]),
            route=routes[i],cpu_lp_calls=int(cpu_results[i] is not None),
            cpu_result_used=not bool(gpu_accepted[i]),total_seconds=record['seconds'],
            **{field:float(value[i]) for field,value in metrics.items()})) for i in range(batch)]
