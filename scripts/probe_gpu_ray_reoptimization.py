"""Causal saved maxmin ray proposal benchmark, no CPU solver or teacher."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.benchmark_ipm_sequence import validate_sequence_provenance
from scripts.probe_ipm_restart_mu_matched import _options,_verify,SOURCE_PATHS
from scripts.benchmark_gpu_stream_partitions import _json_safe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--steps',type=int,nargs='+',default=[2,3,4,5,6,7,8])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--component-scaling',action='store_true')
    p.add_argument('--gpu-fallback',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('No overwrites')
    trace=ROOT/'results/pf_lp_trace_dev32x41_20260905'
    inputs=[load_inputs(trace,'maxmin',i,args.batch) for i in args.steps]
    identity=validate_sequence_provenance([v for _,v in inputs],args.steps,'maxmin',args.batch,(trace/'manifest.json').read_bytes())
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_device_update import DeviceNumericUpdatePlan
    from src.gpu_ray_reoptimization import ray_proposal,ComponentRescaling
    from src.gpu_block_lp import certify_blocks_device
    paths=set(SOURCE_PATHS)|{'scripts/probe_gpu_ray_reoptimization.py','src/gpu_ray_reoptimization.py',
        'src/gpu_ipm_device_update.py','src/gpu_dual_schur.py','src/gpu_shared_newton.py','src/gpu_centrality_corrector.py'}
    record=dict(scope=__doc__,sequence_identity=identity,steps=[],cpu_lp_calls=0,
        source_sha256={v:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for v in sorted(paths)})
    started=time.perf_counter()
    try:
        with ForestGpuBatchedIPM(inputs[0][0],factor_layout='block_diagonal',reuse_equality_proofs=True,**_options()) as s:
            plan=DeviceNumericUpdatePlan(s);previous=None;warm=None
            scaling=ComponentRescaling(s) if args.component_scaling else None
            for step,(problems,provenance) in zip(args.steps,inputs):
                tick=started if previous is None else time.perf_counter()
                trial=dict(step=step,problem_sha256=provenance['problem_sha256']);record['steps'].append(trial)
                if previous is None:
                    result=s.solve(iterations=240);trial['solve_seconds']=result['total_seconds']
                    trial['factor_count']=result['factor_count']
                else:
                    trial['update']=plan.rebind(problems)
                    proposal=previous
                    if scaling is not None:proposal,trial['component_rescaling']=scaling.propose(previous)
                    x,trial['ray']=ray_proposal(s,proposal)
                    y=cp.zeros((s.batch,s.full_m))
                    metrics=certify_blocks_device(problems,s._full_assembled,x.ravel(),y.ravel(),cp=cp,
                        allow_box_dual=False,require_direct_dual=True)
                    result=dict(x=x,y=y,accepted=np.asarray([v['certificate_passed'] for v in metrics]))
                    trial['factor_count']=0;trial['original_gpu_certificates']=metrics
                    trial['ray_accepted']=bool(result['accepted'].all())
                    if not trial['ray_accepted'] and args.gpu_fallback:
                        bound=None if warm is None else warm.bind(s,environment_ids=list(range(args.batch)),
                            stage='maxmin',step=step,restart_mu=1e-5)
                        result=s.solve(iterations=240,internal_warm_start=bound,
                            initial_x=previous if bound is None else None)
                        trial['factor_count']=result['factor_count'];trial['gpu_fallback_seconds']=result['total_seconds']
                trial.update(_verify(problems,result,args.batch))
                if trial['qualified']:
                    previous=result['x'].copy()
                    if scaling is not None:scaling.snapshot()
                    warm=s.export_internal_state(environment_ids=list(range(args.batch)),stage='maxmin',step=step) if trial['factor_count'] else None
                cp.cuda.get_current_stream().synchronize();trial['full_step_seconds']=time.perf_counter()-tick
                print(json.dumps({k:trial[k] for k in ('step','qualified','factor_count','full_step_seconds')}),flush=True)
                if not trial['qualified']:break
        record['all_requested_steps_qualified']=len(record['steps'])==len(args.steps) and all(v['qualified'] for v in record['steps'])
    finally:
        record['lifecycle_seconds']=time.perf_counter()-started
        with args.output.open('x') as f:json.dump(_json_safe(record),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
