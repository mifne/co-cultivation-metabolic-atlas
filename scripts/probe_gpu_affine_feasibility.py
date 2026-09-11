"""Development-only affine GPU projection, certified on unchanged maxmin LPs."""
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
from src.gpu_block_lp import certify_blocks_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--batch',type=int,choices=[1,2,4,8,16,32],default=4)
    parser.add_argument('--steps',type=int,nargs='+',default=[2,3,4])
    parser.add_argument('--iterations',type=int,default=2000)
    parser.add_argument('--chunk',type=int,default=100)
    parser.add_argument('--cold-ipm',action='store_true',help='First original LP solved on GPU; later proposals use only that causal primal')
    parser.add_argument('--method',choices=['motzkin','douglas_rachford','block'],default='motzkin')
    parser.add_argument('--block-size',type=int,default=32)
    parser.add_argument('--dual-sweeps',type=int,default=40)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite previous experiment')
    if min(args.iterations,args.chunk)<1:parser.error('Positive iteration/chunk budget required')
    inputs=[load_inputs(args.trace,'maxmin',i,args.batch) for i in args.steps]
    identity=validate_sequence_provenance([p for _,p in inputs],args.steps,'maxmin',args.batch,
        (args.trace/'manifest.json').read_bytes())
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_device_update import DeviceNumericUpdatePlan
    from src.gpu_affine_feasibility import GpuAffineFeasibility
    record=dict(scope=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sequence_identity=identity,steps=[],completed=False,cpu_lp_calls=0,current_reference_vectors_loaded=False,
        current_CPU_solutions_passed_to_GPU=False,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    paths=set(SOURCE_PATHS)|{'scripts/probe_gpu_affine_feasibility.py','src/gpu_affine_feasibility.py',
        'src/gpu_ipm_device_update.py','src/gpu_ipm_device_payload.py','src/gpu_solve_guards.py',
        'src/gpu_dr_feasibility.py','src/gpu_dual_schur.py','src/gpu_centrality_corrector.py',
        'src/gpu_block_affine_feasibility.py'}
    record['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)}
    record['source_snapshots']={p:(ROOT/p).read_text() for p in sorted(paths)}
    started=time.perf_counter()
    try:
        backend='dual_schur' if args.method=='douglas_rachford' else 'augmented'
        with ForestGpuBatchedIPM(inputs[0][0],factor_layout='block_diagonal',reuse_equality_proofs=True,
                newton_backend=backend,**_options()) as solver:
            plan=DeviceNumericUpdatePlan(solver)
            if args.method=='douglas_rachford':
                from src.gpu_dr_feasibility import GpuDRFeasibility
                projector=GpuDRFeasibility(solver)
            elif args.method=='block':
                from src.gpu_block_affine_feasibility import GpuBlockAffineFeasibility
                projector=GpuBlockAffineFeasibility(solver,block_size=args.block_size,dual_sweeps=args.dual_sweeps)
            else:projector=GpuAffineFeasibility(solver)
            record['constructor_and_projector_seconds']=time.perf_counter()-started
            previous=None
            for index,(step,(problems,provenance)) in enumerate(zip(args.steps,inputs)):
                tick=started if index==0 else time.perf_counter()
                trial=dict(step=step,problem_sha256=provenance['problem_sha256'],qualified=False)
                record['steps'].append(trial)
                if index:trial['numeric_update']=plan.rebind(problems)
                if tuple(provenance['problem_sha256'])!=tuple(solver.problem_hashes):
                    raise RuntimeError('Current original LP identity mismatch')
                if index==0 and args.cold_ipm:
                    cold=solver.solve(iterations=240)
                    if not _verify(problems,cold,args.batch)['qualified']:
                        raise RuntimeError('Initial GPU original LP did not certify')
                    x=cold['reduced_x'].copy()
                    diagnostic=dict(method='initial_original_GPU_IPM',seconds=cold['total_seconds'],
                        factor_count=cold['factor_count'],cpu_lp_calls=0)
                else:
                    x,diagnostic=projector.propose(previous,iterations=args.iterations,chunk=args.chunk)
                trial['projection']=diagnostic
                forest_x,_=solver.lift_device(x,cp.zeros((args.batch,solver.m)))
                full_x=(solver._forest_t@forest_x.ravel()).reshape(args.batch,solver.full_n)
                y=cp.zeros((args.batch,solver.full_m))
                metrics=certify_blocks_device(problems,solver._full_assembled,full_x.ravel(),y.ravel(),
                    cp=cp,allow_box_dual=False,require_direct_dual=True)
                result=dict(x=full_x,y=y,accepted=np.asarray([r['certificate_passed'] for r in metrics]))
                trial.update(_verify(problems,result,args.batch))
                cp.cuda.get_current_stream().synchronize()
                trial['full_step_lifecycle_seconds']=time.perf_counter()-tick
                print(json.dumps(dict(step=step,qualified=trial['qualified'],
                    full_seconds=trial['full_step_lifecycle_seconds'],projection=diagnostic)),flush=True)
                if not trial['qualified']:break
                previous=x.copy()
        record['completed']=len(record['steps'])==len(args.steps)
    except Exception as error:
        record['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        record['sequence_lifecycle_seconds']=time.perf_counter()-started
        record['all_requested_steps_qualified']=record['completed'] and all(s['qualified'] for s in record['steps'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(_json_safe(record),stream,indent=2,allow_nan=False)
        print(f'Saved {args.output}',flush=True)


if __name__=='__main__':main()
