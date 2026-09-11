"""Development-only GPU feasibility search on an objective box face.

The auxiliary LP is NOT the original LP. Only an independent original-unit
primal/dual/direct-gap certificate with y=0 can qualify its candidate. Failure
does not imply original infeasibility. No CPU reference vectors or optimizer.
Saved-input replay, not closed-loop dFBA/PPO; startup costs are reported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.benchmark_ipm_sequence import _GpuSequenceWorkspace, validate_sequence_provenance
from scripts.benchmark_gpu_stream_partitions import _json_safe
from scripts.probe_ipm_restart_mu_matched import _options, _verify, SOURCE_PATHS
from src.gpu_block_lp import box_face_feasibility
from src.lp_trace import problem_hash


def qualify_original(problems,result,cp):
    # The auxiliary dual is unrelated to the original objective. The proposed
    # original certificate is the analytic bound dual y=0, checked from scratch.
    original=dict(result,y=cp.zeros_like(result['y']))
    return _verify(problems,original,len(problems))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--batch',type=int,choices=[1,2,4,8,16,32],default=4)
    parser.add_argument('--steps',type=int,nargs='+',default=[2,3,4])
    parser.add_argument('--iterations',type=int,default=120)
    parser.add_argument('--restart-mu',type=float,default=1e-5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite an existing experiment')
    if args.iterations<=0:parser.error('Positive iteration budget required')
    if not math.isfinite(args.restart_mu) or args.restart_mu<=0.:
        parser.error('Finite positive restart mu required')
    inputs=[load_inputs(args.trace,'maxmin',step,args.batch) for step in args.steps]
    # Validate the unaltered input provenance, including chronology and lane IDs.
    identity=validate_sequence_provenance([p for _,p in inputs],args.steps,'maxmin',
        args.batch,(args.trace/'manifest.json').read_bytes())
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_device_update import DeviceNumericUpdatePlan
    cp.cuda.Device(0).use()
    synchronize=cp.cuda.get_current_stream().synchronize
    def factory(problems,**options):
        solver=ForestGpuBatchedIPM(problems,**options)
        try:solver._device_numeric_plan=DeviceNumericUpdatePlan(solver)
        except BaseException:
            solver.close();raise
        return solver
    record=dict(role=__doc__,configuration={k:str(v) if isinstance(v,Path) else v
        for k,v in vars(args).items()},sequence_identity=identity,
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cpu_lp_calls=0,current_reference_vectors_loaded=False,
        failure_means_original_infeasible=False,steps=[],completed=False)
    paths=set(SOURCE_PATHS)|{'scripts/probe_ipm_box_face_sequence.py',
        'src/gpu_ipm_device_update.py','src/gpu_ipm_device_payload.py',
        'src/gpu_lp_numeric.py','src/gpu_forest_numeric_bounds.py','src/gpu_segmented_linear.py'}
    record['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)}
    record['source_snapshots']={p:(ROOT/p).read_text() for p in sorted(paths)}
    started=time.perf_counter()
    try:
        with _GpuSequenceWorkspace(factory,_options(),reuse=True,
                rebind=lambda solver,p:solver._device_numeric_plan.rebind(p)) as workspace:
            previous=None
            for step,(problems,provenance) in zip(args.steps,inputs):
                tick=time.perf_counter()
                trial=dict(step=step,qualified=False,original_problem_sha256=provenance['problem_sha256'])
                record['steps'].append(trial)
                auxiliary=[box_face_feasibility(p) for p in problems]
                trial['auxiliary_problem_sha256']=[problem_hash(p) for p in auxiliary]
                solver,metadata=workspace.prepare(auxiliary)
                trial.update(metadata)
                bound=None
                if previous is not None:
                    bound=previous.bind(solver,environment_ids=list(range(args.batch)),
                        stage='maxmin',step=step,interior_floor=0.,repair_slacks=False,
                        restart_mu=args.restart_mu)
                result=solver.solve(iterations=args.iterations,internal_warm_start=bound)
                synchronize()
                trial['solve_api_seconds']=result['total_seconds']
                trial['solver_result']={k:v for k,v in result.items() if not isinstance(v,cp.ndarray)}
                trial.update(qualify_original(problems,result,cp))
                if trial['qualified']:
                    previous=solver.export_internal_state(environment_ids=list(range(args.batch)),
                        stage='maxmin',step=step)
                synchronize()
                trial['full_step_lifecycle_seconds']=time.perf_counter()-tick
                print(json.dumps(dict(step=step,qualified=trial['qualified'],
                    solve_seconds=trial['solve_api_seconds'],
                    full_seconds=trial['full_step_lifecycle_seconds'])),flush=True)
                if not trial['qualified']:
                    record['stop_reason']='Auxiliary candidate failed original LP certification; no fallback attempted'
                    break
        record['completed']=len(record['steps'])==len(args.steps)
    except Exception as error:
        record['error']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        record['sequence_lifecycle_seconds']=time.perf_counter()-started
        record['all_requested_steps_qualified']=record['completed'] and all(t['qualified'] for t in record['steps'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(_json_safe(record),stream,indent=2,allow_nan=False)
        print(f'Saved {args.output}',flush=True)


if __name__=='__main__':main()
