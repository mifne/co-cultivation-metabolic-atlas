"""CPU-only baseline for exactly the LP inputs used by the GPU IPM probe.

Fresh-model trials and optional identical-input hot repeats are reported
separately. Neither is a changing-state dFBA rollout. Stored reference x/y
are never read, and newly solved CPU vectors are never offered to a GPU.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import select_entries
from src.cpu_repeated_lp import RepeatedCpuLP,_certificate
from src.lp_trace import problem_hash,problem_request


CPU_WORKER_CHOICES=(1,2,4,8,10,16)


def load_inputs(trace,stage,step,batch):
    """Check manifest labels, each NPZ checksum and restored original LP hash."""
    trace=Path(trace)
    if type(batch) is not int or batch not in (1,2,4,8,16,32):
        raise ValueError('Batch must be a power of two from 1 through 32')
    manifest_path=trace/'manifest.json'
    raw=manifest_path.read_bytes()
    manifest=json.loads(raw)
    if batch>len(manifest['seeds']):
        raise ValueError('Distinct recorded environments required; never pad a batch with duplicated LPs')
    entries=[select_entries(manifest,stage,i,[step])[0] for i in range(batch)]
    problems=[_load_problem_without_reference(trace,e) for e in entries]
    provenance=dict(input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        trace_directory=str(trace.resolve()),entries=entries,
        problem_sha256=[problem_hash(p) for p in problems],
        current_reference_vectors_loaded=False,
        cpu_solution_source='current_trial_HiGHS_only',gpu_calls=0)
    return problems,provenance


def history_counts(history):
    """Count real optimizer runs, including retries, not just LP requests."""
    runs=retries=retry_runs=logical=0
    for batch in history:
        logical+=batch['batch']
        derived_runs=derived_retries=0
        if len(batch['rows'])!=batch['batch']:
            raise ValueError('Incomplete CPU history rows')
        for row in batch['rows']:
            attempts=row['solver_attempts']
            if not attempts:
                raise ValueError('Actual solver attempt history required')
            row_runs=sum(int(a['solver_run']) for a in attempts)
            row_retries=len(attempts)-1
            if row_runs!=row['cpu_solver_runs'] or row_retries!=row['numerical_retry_count']:
                raise ValueError('CPU row counters disagree with actual attempts')
            derived_runs+=row_runs
            derived_retries+=row_retries
            retry_runs+=sum(int(a['solver_run']) for a in attempts[1:])
        if (derived_runs!=batch['cpu_solver_runs']
                or derived_retries!=batch['numerical_retry_count']):
            raise ValueError('CPU batch counters disagree with actual attempts')
        runs+=derived_runs
        retries+=derived_retries
    return dict(logical_lp_requests=logical,actual_cpu_optimizer_runs=runs,
        numerical_retry_attempts=retries,numerical_retry_optimizer_runs=retry_runs)


def _timed_batch(service,problems,requests,stage,kind,index,*,require_direct_dual=False):
    start_history=len(service.history)
    before=time.perf_counter()
    results=service.solve_batch(requests,environment_ids=list(range(len(problems))))
    batch_seconds=time.perf_counter()-before
    history=service.history[start_history:]
    if len(history)!=1 or len(results)!=len(problems):
        raise ValueError('Exactly one complete batch history required')
    before=time.perf_counter()
    rows=[]
    for environment,(problem,result) in enumerate(zip(problems,results)):
        a,rhs,_,_,c,neq=problem
        certificate=None
        objective=None
        objective_consistent=False
        direct=None
        direct_passed=not require_direct_dual
        if result.success and result.x is not None:
            # The owner worker copied these current CPU multipliers before
            # returning; the benchmark never touches a native Highs on main.
            solution=result.solution_snapshot
            certificate=_certificate(*problem,SimpleNamespace(col_value=result.x,
                row_dual=solution.row_dual,col_dual=solution.col_dual,
                value_valid=solution.value_valid,dual_valid=solution.dual_valid))
            objective=float(c@np.asarray(result.x))
            objective_consistent=bool(np.isfinite(objective) and np.isclose(
                objective,result.fun,rtol=1e-12,atol=1e-12))
            if require_direct_dual:
                from src.lp_direct_dual_audit import audit_direct_dual
                audit=audit_direct_dual(problem,np.asarray(result.x),np.asarray(solution.row_dual),xp=np)
                direct={k:np.asarray(v).tolist() for k,v in audit.items()}
                ratio=abs(audit['signed_gap'])/max(1.,abs(audit['primal_objective']))
                direct_passed=bool(np.isfinite(audit['dual_objective']) and np.isfinite(ratio) and ratio<=1e-7)
        passed=bool(result.success and result.diagnostics['certificate_passed']
                    and certificate and certificate['certificate_passed'] and objective_consistent and direct_passed)
        rows.append(dict(environment_id=environment,problem_sha256=problem_hash(problem),
            solver_success=bool(result.success),qualified=passed,
            objective=objective,returned_objective=result.fun,
            objective_consistent=objective_consistent,
            independent_original_certificate=certificate,
            independent_direct_dual_audit=direct,
            additional_direct_gate_required=require_direct_dual,
            additional_direct_gate_passed=direct_passed,
            backend_original_certificate={key:result.diagnostics[key] for key in
                ('primal_residual','dual_violation','relative_kkt_gap','certificate_passed')},
            message=result.message))
    verification_seconds=time.perf_counter()-before
    return dict(kind=kind,index=index,solve_batch_wall_seconds=batch_seconds,
        timing_scope='request validation, model creation/update, HiGHS runs and backend original certificates',
        independent_verification_seconds=verification_seconds,
        solve_plus_independent_verification_seconds=batch_seconds+verification_seconds,
        all_original_certificates_passed=all(r['qualified'] for r in rows),
        rows=rows,history=history,**history_counts(history))


def benchmark(problems,stage,*,workers=(1,4),repeats=3,hot_repeats=0,
              service_factory=RepeatedCpuLP):
    """Fresh services per cold repeat; optional hot solves use identical inputs."""
    if (stage not in ('maxmin','aggregate','exchange') or not problems
            or type(repeats) is not int or not 1<=repeats<=20
            or type(hot_repeats) is not int or not 0<=hot_repeats<=20
            or not workers or len(set(workers))!=len(workers)
            or any(type(w) is not int or w not in CPU_WORKER_CHOICES for w in workers)):
        raise ValueError('Invalid bounded CPU benchmark configuration')
    requests=[problem_request(p,stage=stage) for p in problems]
    groups=[]
    for count in workers:
        trials=[]
        for repeat in range(repeats):
            before=time.perf_counter()
            service=service_factory(workers=count,reuse_basis=True,max_numerical_retries=2)
            constructor_seconds=time.perf_counter()-before
            trial=dict(repeat=repeat,constructor_seconds=constructor_seconds,
                fresh_model=True,cold_process_or_os_cache_claimed=False,hot_same_input=[])
            try:
                trial['cold']=_timed_batch(service,problems,requests,stage,'fresh_model',repeat)
                for hot in range(hot_repeats):
                    trial['hot_same_input'].append(_timed_batch(service,problems,requests,
                        stage,'identical_input_hot_repeat_not_dfba',hot))
                trial.update(history_counts(service.history))
            finally:
                before=time.perf_counter()
                service.close()
                trial['close_seconds']=time.perf_counter()-before
            trial['constructor_plus_cold_solve_plus_final_close_seconds']=(constructor_seconds+
                trial['cold']['solve_batch_wall_seconds']+trial['close_seconds'])
            trials.append(trial)
            print(json.dumps(dict(status='cpu_trial_complete',workers=count,repeat=repeat,
                cold_seconds=trial['cold']['solve_batch_wall_seconds'],
                all_certified=trial['cold']['all_original_certificates_passed'])),flush=True)
        cold=[t['cold']['solve_batch_wall_seconds'] for t in trials]
        qualified=all(t['cold']['all_original_certificates_passed'] and
            all(h['all_original_certificates_passed'] for h in t['hot_same_input']) for t in trials)
        groups.append(dict(workers=count,trials=trials,all_trials_qualified=qualified,
            cold_solve_batch_seconds=cold,cold_median_seconds=float(np.median(cold)),
            cold_mean_seconds=float(np.mean(cold)),
            cold_sample_sd_seconds=float(np.std(cold,ddof=1)) if len(cold)>1 else None,
            actual_cpu_optimizer_runs=sum(t['actual_cpu_optimizer_runs'] for t in trials),
            logical_lp_requests=sum(t['logical_lp_requests'] for t in trials),
            numerical_retry_attempts=sum(t['numerical_retry_attempts'] for t in trials),
            numerical_retry_optimizer_runs=sum(t['numerical_retry_optimizer_runs'] for t in trials)))
    return dict(groups=groups,status=('all_cpu_trials_original_certified' if
        all(g['all_trials_qualified'] for g in groups) else 'unqualified_cpu_trial_present'),
        actual_cpu_optimizer_runs=sum(g['actual_cpu_optimizer_runs'] for g in groups),
        logical_lp_requests=sum(g['logical_lp_requests'] for g in groups),
        numerical_retry_attempts=sum(g['numerical_retry_attempts'] for g in groups),
        numerical_retry_optimizer_runs=sum(g['numerical_retry_optimizer_runs'] for g in groups))


def _json_finite(value):
    # Failed original certificates may contain inf. Null is explicitly declared
    # missing/nonfinite, never zero; qualification remains false.
    if isinstance(value,dict):return {k:_json_finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_finite(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_exclusive(output,record):
    output=Path(output)
    serialized=json.dumps(_json_finite(record),indent=2,allow_nan=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        stream.write(serialized)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=ROOT/'results/pf_coverage_holdout4x120_20260905')
    parser.add_argument('--stage',choices=['maxmin','aggregate','exchange'],default='exchange')
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--batch',type=int,choices=[1,2,4,8,16,32],default=4)
    parser.add_argument('--workers',type=int,nargs='+',choices=CPU_WORKER_CHOICES,default=[1,4],
        help='Independent single-thread HiGHS models; sweep worker counts before claiming the best CPU baseline')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--hot-repeats',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('CPU baselines never overwrite prior artifacts')
    problems,provenance=load_inputs(args.trace,args.stage,args.step,args.batch)
    sources=('scripts/benchmark_ipm_cpu_inputs.py','scripts/analyze_coverage_temporal_routing.py',
        'scripts/probe_downstream_gpu_coverage.py','src/cpu_repeated_lp.py','src/lp_trace.py','src/lp_bounds.py')
    record=dict(role='development_same_input_CPU_baseline_not_training_or_rollout',
        stage=args.stage,step=args.step,batch=args.batch,configuration=dict(workers=args.workers,
            repeats=args.repeats,hot_same_input_repeats=args.hot_repeats,reuse_basis=True,
            max_numerical_retries=2,threads_per_HiGHS=1),
        cpu_execution_policy='Stable environment owner thread for native model create/update/solve/snapshot/disposal',
        independent_cpu_dual_source='Owned solution_snapshot copied on the native owner; no main-thread native Highs access',
        original_certificate_limits=dict(primal_residual=1e-5,dual_violation=1e-7,relative_kkt_gap=1e-7),
        environment=dict(python=sys.version,platform=platform.platform(),cpu=platform.processor(),
            logical_cpu_count=os.cpu_count(),highspy=version('highspy'),numpy=np.__version__,
            thread_environment={k:os.environ.get(k) for k in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        source_snapshots={p:(ROOT/p).read_text() for p in sources},
        qualification_scope='CPU original-LP baseline only; no GPU speedup claim',
        hot_repeat_caution='Same state and same LP, not changing-state dFBA; never mix hot and cold timing',
        comparison_caution='GPU failure time is not a speedup; compare only solutions passing the same original certificate',
        nonfinite_numeric_values_encoded_as_null=True,**provenance)
    record.update(benchmark(problems,args.stage,workers=args.workers,repeats=args.repeats,
                            hot_repeats=args.hot_repeats))
    record['completed']=True
    write_exclusive(args.output,record)
    print('Saved '+str(args.output),flush=True)


if __name__=='__main__':main()
