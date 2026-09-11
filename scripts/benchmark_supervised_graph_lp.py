"""Held-out input-only replay: previous GPU primal vs learned GNN / GNN+GRU.

No current/future CPU labels are loaded, no CPU optimizer is called. Results
cover maxmin saved-input correction, NOT closed-loop dFBA or PPO speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.probe_ipm_restart_mu_matched import _options,_verify,SOURCE_PATHS
from scripts.benchmark_ipm_sequence import validate_sequence_provenance
from scripts.benchmark_gpu_stream_partitions import _json_safe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--trace',type=Path,default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    p.add_argument('--batch',type=int,choices=[4,8,16,32],default=4)
    p.add_argument('--steps',type=int,nargs='+',default=[2,3,4,5,6,7,8])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--arms',nargs='+',choices=['previous','gnn','gnn_gru'],default=['previous','gnn','gnn_gru'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing overwrite')
    if args.repeats<1:p.error('Positive repeats required')
    import cupy as cp
    import torch
    from src.graph_temporal_lp import LPGraphBatch,GraphTemporalSession
    from src.supervised_graph_lp import load_checkpoint
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_device_update import DeviceNumericUpdatePlan
    from src.graph_primal_restart import bind_graph_restart
    from src.gpu_block_lp import certify_blocks_device
    manifest=json.loads((args.trace/'manifest.json').read_text())
    training=json.loads((args.artifact/'manifest.json').read_text())
    if training['status']!='completed':raise ValueError('Completed training artifact required')
    if manifest['model_fingerprints']!=training['model_fingerprints']:raise ValueError('GEM identity mismatch')
    used_seeds=set(training['train_seeds']+training['development_seeds'])
    if used_seeds.intersection(manifest['seeds'][:args.batch]):raise ValueError('Evaluation seeds leaked into training/selection')
    inputs=[load_inputs(args.trace,'maxmin',s,args.batch) for s in args.steps]
    identity=validate_sequence_provenance([v for _,v in inputs],args.steps,'maxmin',args.batch,(args.trace/'manifest.json').read_bytes())
    used_hash=set(training['training_problem_hashes']+training['development_problem_hashes'])
    if any(used_hash.intersection(v['problem_sha256']) for _,v in inputs):raise ValueError('Evaluation inputs leaked into training/selection')
    model_identity=json.dumps(manifest['model_fingerprints'],sort_keys=True)
    files=set(SOURCE_PATHS)|{'scripts/benchmark_supervised_graph_lp.py','src/graph_primal_restart.py',
        'src/graph_temporal_lp.py','src/supervised_graph_lp.py','src/gpu_ipm_device_update.py'}
    report=dict(scope=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sequence_identity=identity,training_manifest_sha256=hashlib.sha256((args.artifact/'manifest.json').read_bytes()).hexdigest(),
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)},
        source_snapshots={f:(ROOT/f).read_text() for f in sorted(files)},
        cpu_lp_calls=0,current_reference_vectors_loaded=False,evaluation_seed_overlap=0,evaluation_input_overlap=0,
        cuda_device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),trials=[])
    def save():
        args.output.write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False))
    save()
    try:
        for repeat in range(args.repeats):
            order=args.arms[repeat%len(args.arms):]+args.arms[:repeat%len(args.arms)]
            for arm in order:
                trial=dict(repeat=repeat,arm=arm,steps=[],completed=False);report['trials'].append(trial)
                started=time.perf_counter();model=session=None
                if arm!='previous':
                    path=args.artifact/(arm+'.pt');checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
                    expected=next(v['checkpoint_sha256'] for v in training['models'] if v['name']==arm)
                    if checkpoint_hash!=expected:raise ValueError('Checkpoint changed after training')
                    model,meta=load_checkpoint(path)
                    if meta['model_fingerprints']!=manifest['model_fingerprints']:raise ValueError('Checkpoint metadata mismatch')
                    session=GraphTemporalSession(model) if model.temporal else None
                with ForestGpuBatchedIPM(inputs[0][0],factor_layout='block_diagonal',reuse_equality_proofs=True,**_options()) as solver:
                    plan=DeviceNumericUpdatePlan(solver);previous=None
                    trial['initialization_seconds']=time.perf_counter()-started
                    for index,(step,(problems,provenance)) in enumerate(zip(args.steps,inputs)):
                        stamp=time.perf_counter();row=dict(step=step,qualified=False);trial['steps'].append(row)
                        if index:row['numeric_update']=plan.rebind(problems)
                        if tuple(solver.problem_hashes)!=tuple(provenance['problem_sha256']):raise ValueError('Stale prepared LP')
                        bound=None;row['inference_and_graph_seconds']=0.;row['raw_candidate_passed']=0
                        if model is not None:
                            tick=time.perf_counter()
                            graph=LPGraphBatch.from_problems(problems,stage='maxmin',model_identity=model_identity,device='cuda')
                            with torch.inference_mode():
                                if session is not None:proposal,token=session.propose(graph,list(range(args.batch)))
                                else:proposal=model(graph)
                                px=cp.from_dlpack(proposal.x.contiguous());py=cp.from_dlpack(proposal.y.contiguous())
                                cp.cuda.get_current_stream().synchronize()
                            row['inference_and_graph_seconds']=time.perf_counter()-tick
                            tick=time.perf_counter()
                            raw=certify_blocks_device(problems,solver._full_assembled,px.ravel(),py.ravel(),cp=cp,
                                allow_box_dual=True,require_direct_dual=True)
                            row['raw_candidate_passed']=sum(v['certificate_passed'] for v in raw)
                            row['raw_candidate_max_primal']=max(v['primal_residual'] for v in raw)
                            row['raw_candidate_check_seconds']=time.perf_counter()-tick
                            if index:bound=bind_graph_restart(solver,graph,px,checkpoint_sha256=checkpoint_hash)
                        elif previous is not None:
                            bound=previous.bind(solver,environment_ids=list(range(args.batch)),stage='maxmin',step=step,restart_mu=1e-5)
                        # Same cold initial solve for every arm. Later starts
                        # differ only in primal; restart rule / LP gate match.
                        result=solver.solve(iterations=120,internal_warm_start=bound)
                        row.update(solve_seconds=result['total_seconds'],factor_count=result['factor_count'],
                            solve_count=result['solve_count'],accepted_iteration=result['accepted_iteration'].tolist(),
                            solver_status=result['status'])
                        row.update(_verify(problems,result,args.batch))
                        if session is not None:session.commit(token,proposal,result['accepted'])
                        if row['qualified']:previous=solver.export_internal_state(environment_ids=list(range(args.batch)),stage='maxmin',step=step)
                        cp.cuda.get_current_stream().synchronize();row['full_step_seconds']=time.perf_counter()-stamp
                        print(json.dumps(dict(repeat=repeat,arm=arm,**{k:row[k] for k in ('step','qualified','factor_count','solve_seconds','full_step_seconds','raw_candidate_passed')})),flush=True)
                        if not row['qualified']:break
                    trial['completed']=len(trial['steps'])==len(args.steps) and all(v['qualified'] for v in trial['steps'])
                    trial['total_seconds']=time.perf_counter()-started
                    if trial['completed']:
                        trial['hot_seconds']=sum(v['full_step_seconds'] for v in trial['steps'][1:])
                        trial['hot_factor_count']=sum(v['factor_count'] for v in trial['steps'][1:])
                del model,session,solver,previous
                torch.cuda.empty_cache();cp.get_default_memory_pool().free_all_blocks();save()
        report['completed']=all(v['completed'] for v in report['trials'])
    except BaseException as e:
        report['error']=dict(type=type(e).__name__,message=str(e));raise
    finally:save()


if __name__=='__main__':main()
