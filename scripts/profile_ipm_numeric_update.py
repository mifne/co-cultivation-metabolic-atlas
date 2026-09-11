"""Profile current-input GPU workspace rebind only, without LP optimization.

Timings with cProfile are diagnostic overhead-inclusive, not speed claims.
Inputs come from a saved trajectory, without saved CPU solution vectors.
"""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.benchmark_ipm_sequence import validate_sequence_provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,choices=[4,32],default=32)
    parser.add_argument('--stage',choices=['maxmin','aggregate','exchange'],default='maxmin')
    parser.add_argument('--trace',type=Path,default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-static-forest',action='store_true')
    parser.add_argument('--device-staging',action='store_true')
    parser.add_argument('--direct-kkt-payload',action='store_true')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('No overwrite of profiling artifacts')
    inputs=[load_inputs(args.trace,args.stage,step,args.batch) for step in [2,3]]
    identity=validate_sequence_provenance([p for _,p in inputs],[2,3],args.stage,args.batch,
        (args.trace/'manifest.json').read_bytes())
    paths=['scripts/profile_ipm_numeric_update.py','src/gpu_ipm_numeric_update.py',
        'src/lp_equality_reduction.py','src/lp_zero_face.py','src/lp_exact_equalities.py',
        'src/gpu_block_lp.py','src/gpu_batched_ipm.py','src/gpu_forest_map.py',
        'src/gpu_ipm_staging.py','src/gpu_ipm_kkt_payload.py','src/csr_block_assembly.py']
    record=dict(role='profile_numeric_rebind_only_NOT_solver_or_PPO_performance',
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_snapshots={p:(ROOT/p).read_text() for p in paths},
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        sequence_identity=identity,input_provenance=[p for _,p in inputs],
        cpu_lp_calls=0,gpu_lp_calls=0,current_reference_vectors_loaded=False)
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_numeric_update import rebind_forest_ipm
    started=time.perf_counter()
    with ForestGpuBatchedIPM(inputs[0][0],regularization=1e-6,globalized=True,
            forcing_eta=.1,newton_krylov_iterations=16,exact_equalities=True,
            fix_singleton_equalities=True,second_forest=True,allow_box_dual=True,
            device_checked_solves=True,factor_refinements=0) as solver:
        record['constructor_seconds']=time.perf_counter()-started
        profiler=cProfile.Profile()
        profiler.enable()
        update=rebind_forest_ipm(solver,inputs[1][0],
            reuse_static_forest=args.reuse_static_forest,device_staging=args.device_staging,
            direct_kkt_payload=args.direct_kkt_payload)
        cp.cuda.get_current_stream().synchronize()
        profiler.disable()
        record['numeric_update']=update
        record['native_factor_count']=solver.factor.factor_count
        record['native_solve_count']=solver.factor.solve_count
        output=io.StringIO()
        stats=pstats.Stats(profiler,stream=output).strip_dirs().sort_stats('cumulative')
        stats.print_stats(65)
        record['cumulative_profile']=output.getvalue()
        output=io.StringIO()
        pstats.Stats(profiler,stream=output).strip_dirs().sort_stats('tottime').print_stats(35)
        record['self_time_profile']=output.getvalue()
    record['completed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as output:
        json.dump(record,output,indent=2,allow_nan=False)
    print(record['cumulative_profile'],flush=True)
    print('Saved '+str(args.output),flush=True)


if __name__=='__main__': main()
