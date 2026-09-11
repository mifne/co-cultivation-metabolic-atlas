"""Experimental objective-bound feasibility IPM; unchanged original LP gate.

Only a single finite-bound objective is supported. Finding a feasible point
at its box optimum proves optimality; failure proves nothing. No CPU solver.
Saved maxmin inputs, not closed-loop PPO. Auxiliary timing includes current
input transformation, device rebind, solve and independent ORIGINAL audit.
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
from scripts.probe_ipm_restart_mu_matched import _options,_verify,SOURCE_PATHS
from scripts.benchmark_gpu_stream_partitions import _json_safe


def auxiliary(problems):
    out=[]
    for a,b,lo,hi,c,neq in problems:
        nz=np.flatnonzero(c)
        if len(nz)!=1 or not np.isfinite(c).all():raise ValueError('Single finite objective required')
        j=int(nz[0]);value=hi[j] if c[j]<0 else lo[j]
        if not np.isfinite(value) or lo[j]>hi[j]:raise ValueError('Finite box optimum required')
        low=lo.copy();high=hi.copy();low[j]=high[j]=value
        out.append((a,b,low,high,np.zeros_like(c),neq))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--steps',type=int,nargs='+',default=[2,3,4])
    p.add_argument('--restart-mu',type=float,default=1e-5)
    p.add_argument('--feasibility-newton',action='store_true')
    p.add_argument('--damping',type=float,default=1e-6)
    p.add_argument('--centered',action='store_true',help='One centered Newton direction per iteration')
    p.add_argument('--primal-extrapolation',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing overwrite')
    trace=ROOT/'results/pf_lp_trace_dev32x41_20260905'
    inputs=[load_inputs(trace,'maxmin',i,args.batch) for i in args.steps]
    identity=validate_sequence_provenance([v for _,v in inputs],args.steps,'maxmin',args.batch,(trace/'manifest.json').read_bytes())
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_device_update import DeviceNumericUpdatePlan
    from src.gpu_block_lp import assemble_blocks,certify_blocks_device
    from cupyx.scipy.sparse import csr_matrix
    record=dict(scope=__doc__,sequence_identity=identity,configuration=vars(args).copy(),steps=[],cpu_lp_calls=0,
        original_tolerances_unchanged=True,failed_auxiliary_is_not_original_infeasibility=True)
    record['configuration']['output']=str(args.output)
    paths=set(SOURCE_PATHS)|{'scripts/probe_gpu_bound_feasibility.py','src/gpu_ipm_device_update.py',
        'src/gpu_feasibility_newton.py','src/gpu_dual_schur.py','src/gpu_shared_newton.py',
        'src/gpu_centrality_corrector.py','src/gpu_primal_extrapolation.py'}
    record['source_sha256']={v:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for v in sorted(paths)}
    solver=None;previous=None;previous_x=None;started=time.perf_counter()
    try:
        for step,(problems,provenance) in zip(args.steps,inputs):
            tick=time.perf_counter();trial=dict(step=step,problem_sha256=provenance['problem_sha256'])
            record['steps'].append(trial)
            aux=auxiliary(problems)
            if solver is None:
                options=_options();options['predictor_corrector']=not args.centered
                options['certified_primal_extrapolation']=args.primal_extrapolation
                if args.centered:options['predictor_affine_fraction']=1.
                solver=ForestGpuBatchedIPM(aux,factor_layout='block_diagonal',reuse_equality_proofs=True,
                    newton_backend='dual_schur' if args.feasibility_newton else 'augmented',**options)
                plan=DeviceNumericUpdatePlan(solver)
            else:trial['update']=plan.rebind(aux)
            bound=None if previous is None else previous.bind(solver,environment_ids=list(range(args.batch)),
                stage='maxmin',step=step,restart_mu=args.restart_mu if args.restart_mu else None)
            if args.feasibility_newton and previous_x is not None:
                from src.gpu_feasibility_newton import GpuFeasibilityNewton
                x,diag=GpuFeasibilityNewton(solver).propose(previous_x,damping=args.damping)
                forest_x,_=solver.lift_device(x,cp.zeros((args.batch,solver.m)))
                full_x=(solver._forest_t@forest_x.ravel()).reshape(args.batch,solver.full_n)
                result=dict(x=full_x,y=cp.zeros((args.batch,solver.full_m)),reduced_x=x,
                    accepted=None,total_seconds=diag['seconds'],factor_count=diag['factor_count'])
                trial['proposal_diagnostic']=diag
            else:result=solver.solve(iterations=240,internal_warm_start=bound)
            trial['solve_seconds']=result['total_seconds'];trial['factor_count']=result['factor_count']
            trial['primal_extrapolation_diagnostics']=result.get('primal_extrapolation_diagnostics')
            # y=0 is the original box-optimum dual, independently checked below.
            packed=assemble_blocks(problems)
            original_assembled=(csr_matrix(packed[0]),*(cp.asarray(v) for v in packed[1:]))
            original_y=cp.zeros_like(result['y'])
            gpu_gate=certify_blocks_device(problems,original_assembled,result['x'].ravel(),original_y.ravel(),
                cp=cp,allow_box_dual=False,require_direct_dual=True)
            original=dict(x=result['x'],y=original_y,
                accepted=np.asarray([v['certificate_passed'] for v in gpu_gate]))
            trial['original_gpu_certificates']=gpu_gate
            trial.update(_verify(problems,original,args.batch))
            if trial['qualified']:
                previous_x=result['reduced_x'].copy()
                if not args.feasibility_newton:
                    previous=solver.export_internal_state(environment_ids=list(range(args.batch)),stage='maxmin',step=step)
            trial['full_step_seconds']=time.perf_counter()-tick
            print(json.dumps({k:trial[k] for k in ('step','qualified','solve_seconds','factor_count','full_step_seconds')}),flush=True)
            if not trial['qualified']:break
        record['all_requested_steps_qualified']=len(record['steps'])==len(args.steps) and all(t['qualified'] for t in record['steps'])
    finally:
        if solver is not None:solver.close()
        record['lifecycle_seconds']=time.perf_counter()-started
        with args.output.open('x') as f:json.dump(_json_safe(record),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
