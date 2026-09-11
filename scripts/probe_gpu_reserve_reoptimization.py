"""Causal reserve proposals with current-original GPU optimality certificates.

Saved maxmin inputs only, not a closed-loop dFBA/PPO claim. No CPU optimizer
or reference solution is used. A failed current certificate triggers a fresh
GPU auxiliary solve; an unqualified auxiliary candidate triggers an unchanged
original GPU solve. Every construction/fallback and verification is timed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.benchmark_ipm_sequence import validate_sequence_provenance
from scripts.benchmark_gpu_stream_partitions import _json_safe
from scripts.probe_ipm_restart_mu_matched import _options,_verify,SOURCE_PATHS
from src.lp_reserve_proposal import reserve_proposal
from src.lp_trace import problem_hash
from src.gpu_block_lp import assemble_blocks,certify_blocks_device


def current_certificate(problems,x,cp):
    from cupyx.scipy.sparse import csr_matrix
    started=time.perf_counter()
    packed=assemble_blocks(problems)
    device=(csr_matrix(packed[0]),*(cp.asarray(v) for v in packed[1:]))
    y=cp.zeros((len(problems),problems[0][0].shape[0]),dtype=cp.float64)
    metrics=certify_blocks_device(problems,device,x.ravel(),y.ravel(),cp=cp,
        allow_box_dual=False,require_direct_dual=True)
    accepted=np.asarray([m['certificate_passed'] for m in metrics],dtype=bool)
    result=dict(x=x,y=y,accepted=accepted)
    verified=_verify(problems,result,len(problems))
    cp.cuda.get_current_stream().synchronize()
    return verified,dict(metrics=metrics,seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--batch',type=int,choices=[1,2,4,8,16,32],default=4)
    parser.add_argument('--steps',type=int,nargs='+',default=[2,3,4,5,6,7,8])
    parser.add_argument('--fraction',type=float,default=.5)
    parser.add_argument('--iterations',type=int,default=240)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite earlier experiment')
    if not np.isfinite(args.fraction) or not 0.<args.fraction<=1. or args.iterations<1:
        parser.error('Positive iteration budget and reserve fraction (0,1] required')
    inputs=[load_inputs(args.trace,'maxmin',step,args.batch) for step in args.steps]
    identity=validate_sequence_provenance([p for _,p in inputs],args.steps,'maxmin',args.batch,
        (args.trace/'manifest.json').read_bytes())
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    cp.cuda.Device(0).use()
    record=dict(scope=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        sequence_identity=identity,steps=[],completed=False,cpu_lp_calls=0,current_reference_vectors_loaded=False,
        current_CPU_solutions_passed_to_GPU=False,
        original_certificate_limits=dict(primal_residual=1e-5,dual_violation=1e-7,relative_kkt_gap=1e-7))
    paths=set(SOURCE_PATHS)|{'scripts/probe_gpu_reserve_reoptimization.py','src/lp_reserve_proposal.py',
        'src/gpu_ipm_device_update.py','src/gpu_ipm_device_payload.py','src/gpu_solve_guards.py'}
    record['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)}
    record['source_snapshots']={p:(ROOT/p).read_text() for p in sorted(paths)}
    previous_x=None
    started=time.perf_counter()
    try:
        for step,(problems,provenance) in zip(args.steps,inputs):
            tick=time.perf_counter()
            trial=dict(step=step,problem_sha256=provenance['problem_sha256'],qualified=False,
                reused_previous=False,gpu_solver_calls=0,auxiliary_attempts=[],original_fallback=False)
            record['steps'].append(trial)
            if previous_x is not None:
                verification,diagnostic=current_certificate(problems,previous_x,cp)
                trial['prior_candidate_verification']=verification
                trial['prior_candidate_diagnostic']=diagnostic
                if verification['qualified']:
                    trial.update(verification);trial['reused_previous']=True
            if not trial['qualified']:
                for auxiliary in (True,False):
                    candidate_problems=([reserve_proposal(p,args.fraction) for p in problems] if auxiliary else problems)
                    proposal_stamp=time.perf_counter()
                    attempt=dict(auxiliary=auxiliary,problem_sha256=[problem_hash(p) for p in candidate_problems])
                    trial['auxiliary_attempts'].append(attempt)
                    options=dict(_options(),factor_layout='block_diagonal',reuse_equality_proofs=True)
                    options['retain_internal_state']=False
                    with ForestGpuBatchedIPM(candidate_problems,**options) as solver:
                        trial['gpu_solver_calls']+=1
                        result=solver.solve(iterations=args.iterations)
                        attempt['solve_seconds']=result['total_seconds']
                        attempt['factor_count']=result['factor_count']
                        attempt['auxiliary_solver_accepted']=bool(result['accepted'].all())
                        # On fallback, use the actual original dual; for a
                        # reserve proposal y=0 must prove original optimality.
                        if auxiliary:
                            verification,diagnostic=current_certificate(problems,result['x'],cp)
                            attempt['original_diagnostic']=diagnostic
                        else:
                            verification=_verify(problems,result,args.batch)
                            trial['original_fallback']=True
                        attempt['original_verification']=verification
                        # Current original certification is decisive, even if
                        # the stricter auxiliary LP has not attained its optimum.
                        if verification['qualified']:
                            previous_x=result['x'].copy()
                            trial.update(verification)
                    cp.cuda.get_current_stream().synchronize()
                    attempt['construction_solve_verify_close_seconds']=time.perf_counter()-proposal_stamp
                    if trial['qualified']:break
            trial['full_step_lifecycle_seconds']=time.perf_counter()-tick
            print(json.dumps({k:trial[k] for k in ['step','qualified','reused_previous','gpu_solver_calls',
                'original_fallback','full_step_lifecycle_seconds']}),flush=True)
            if not trial['qualified']:break
        record['completed']=len(record['steps'])==len(args.steps)
    except Exception as error:
        record['error']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        record['sequence_lifecycle_seconds']=time.perf_counter()-started
        record['all_requested_steps_qualified']=record['completed'] and all(s['qualified'] for s in record['steps'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(_json_safe(record),stream,indent=2,allow_nan=False)
        print(f'Saved {args.output}',flush=True)


if __name__=='__main__':main()
