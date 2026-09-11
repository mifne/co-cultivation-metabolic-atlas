"""Offline independent training: GPU first-step prefix, then CPU four-stage teacher.

CPU fallback is permitted ONLY in this training collector and is counted.
No benchmark seed/input is accepted by this collector. Original chemistry and
control actions remain unchanged. A GPU-produced prefix addresses the state
distribution shift that pure three-stage CPU trajectories do not sample.
"""
import argparse,copy,hashlib,json,sys,time,warnings,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import OptimizeWarning
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from greenlet import getcurrent
from scripts.benchmark_compact_gpu import checked_npz
from scripts.benchmark_basis_bank_rollout import environment,snapshot
from scripts.microbatch_comparison_support import ParallelCpuLP,drive_microbatch
from src.gpu_compact_basis import CompactBank
from src.gpu_certified_basis import CommunityCoordinates
from src.gpu_compiled_community_backend import lp_arrays,stage_key
from src.gpu_batched_compiled_backend import BatchedCompiledBackend,YieldingLPBackend
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend
from src.fba_surrogate import model_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--operators',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20287401)
    p.add_argument('--environments',type=int,default=4);p.add_argument('--steps',type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if not 1<=args.steps<=120 or args.environments<2:raise ValueError('Invalid training size')
    seeds=list(range(args.seed,args.seed+args.environments))
    if set(seeds)&set(range(20287201,20287329)):raise ValueError('Reserved development benchmark seeds')
    warnings.filterwarnings('ignore',category=OptimizeWarning)
    import cupy as cp
    started=time.perf_counter();manifest=json.loads((args.bank/'manifest.json').read_text())
    operators=json.loads((args.operators/'manifest.json').read_text())
    if manifest['status']!='completed' or operators['status']!='completed':raise ValueError('Incomplete source')
    if operators['source_manifest_sha256']!=hashlib.sha256((args.bank/'manifest.json').read_bytes()).hexdigest():raise ValueError('Wrong operators')
    sample,layout=environment(args.seed);fingerprints={k:model_fingerprint(m) for k,m in sample.simulator.models.items()}
    if fingerprints!=manifest['model_fingerprints']:raise ValueError('Stale models')
    banks={}
    for stage in manifest['stages']:
        folder=args.bank/stage['stage'];root=checked_npz(folder,'root.npz',stage['root_sha256']);router=checked_npz(folder,'router.npz',stage['router_sha256'])
        a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
        entries=[checked_npz(folder,e['filename'],e['sha256']) for e in stage['entries']]
        banks[stage['stage']]=CompactBank(dict(a=a,neq=int(root['neq'])),root['variable_rows'],entries,
            router['centers'],router['indices'],router['scale'],full_batch_candidates=True,candidate_ranking='count')
    for entry in operators['entries']:
        banks[entry['stage']].evaluators[entry['index']].repair_path=(args.operators/entry['filename'],entry['sha256'])
    metadata=dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for m in sample.simulator.models.values() for r in m.reactions],
        reaction_species=[k for k,m in sample.simulator.models.items() for r in m.reactions])
    coords=CommunityCoordinates(metadata,banks['maxmin'].host_a,banks['maxmin'].neq)
    gpu=BatchedCompiledBackend(coords,{tuple(s['key']):banks[s['stage']] for s in manifest['stages']},128,repair_cache_size=3,dual_edge='devex')
    cpu=ParallelCpuLP(4);actual=defaultdict(list)
    report=dict(status='training',actual_training_seeds=seeds,actual_training_steps=args.steps,stages=[],
        provenance=[dict(identity=dict(model_fingerprints=fingerprints))],prefix_gpu_history=[],training_history=[],
        cpu_lp_calls=0,source_bank_manifest_sha256=hashlib.sha256((args.bank/'manifest.json').read_bytes()).hexdigest(),
        warning='OFFLINE ONLY: GPU first-step prefix, CPU optimal-face teacher after prefix; counted CPU fallback permitted',
        parent_training_seeds=manifest['train_seeds'])
    report['source_hashes']={}
    for name in ('scripts/collect_gpu_prefix_training.py','src/gpu_compact_basis.py','src/gpu_batched_compiled_backend.py',
                 'src/gpu_revised_basis.py','src/community_solver.py','src/dfba_simulator.py','src/gpu_exchange_tie_break.py'):
        target=args.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        report['source_hashes'][name]=hashlib.sha256(target.read_bytes()).hexdigest()
    def save():(args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    save()
    class Teacher:
        batches=0
        def solve_batch(self,requests):
            tick=time.perf_counter();arrays=[lp_arrays(c,**kw) for c,kw in requests]
            stage=stage_key(arrays[0][4],arrays[0][0],arrays[0][-1],layout.n_fluxes)[0]
            if stage in banks:actual[stage].append([coords.normalize(*a) for a in arrays])
            results=None;gpu_error=None
            if self.batches<4:
                try:results=gpu.solve_batch(requests)
                except Exception as error:gpu_error=str(error)
            bad=list(range(len(requests))) if results is None else [i for i,r in enumerate(results) if not r.success]
            if results is None:results=[None]*len(requests)
            if bad:
                selected=[]
                for i in bad:
                    c,kw=requests[i];kw=dict(kw);kw['options']=dict(kw.get('options',{}),primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
                    selected.append((c,kw))
                solved=cpu.solve_batch(selected);report['cpu_lp_calls']+=len(bad)
                for i,r in zip(bad,solved):
                    residual=float('inf')
                    if r.success:
                        a,rhs,lo,hi,c,neq=arrays[i];error=a@r.x-rhs
                        residual=max(0.,float(np.max(np.abs(error[:neq]),initial=0.)),float(np.max(error[neq:],initial=0.)),
                            float(np.max(lo-r.x,initial=0.)),float(np.max(r.x-hi,initial=0.)))
                    r.diagnostics=dict(success=bool(r.success),objective=float(r.fun) if r.success else None,
                        max_original_residual=residual,total_seconds=(time.perf_counter()-tick)/len(bad),cpu_lp_calls=1,offline_teacher=True)
                    results[i]=r
            report['training_history'].append(dict(stage=stage,step=self.batches//4+1,cpu_rows=bad,gpu_error=gpu_error,
                accepted=[bool(r.success) for r in results],seconds=time.perf_counter()-tick))
            self.batches+=1;save();print(f'offline {self.batches}: {stage}; CPU teacher {len(bad)}/{len(requests)}',flush=True)
            return results
    parent=getcurrent();envs=[]
    for seed in seeds:
        env=copy.deepcopy(sample);env.reset(seed=seed);model=env.simulator._cooperative_solver
        model.linear_program_backend=GpuExchangeTieBreakBackend(model,inner=YieldingLPBackend(parent));envs.append((env,model))
    actions=[np.random.default_rng(s).uniform(.05,.95,(120,5)).astype(np.float32)[:args.steps] for s in seeds]
    try:report['endpoints']=drive_microbatch(envs,actions,Teacher(),snapshot)
    finally:cpu.close()
    report['prefix_gpu_history']=gpu.history
    for stage,steps in actual.items():
        problems=[steps[t][i] for i in range(args.environments) for t in range(args.steps)]
        bank=banks[stage];data=bank.prepare_host(problems)
        folder=args.output/'inputs'/stage;folder.mkdir(parents=True)
        np.savez_compressed(folder/'000000.npz',**{k:v.get() for k,v in data.items()})
        from src.gpu_neural_basis_proposal import features
        x=features(data).get();variable=np.flatnonzero(np.ptp(x,axis=0)>1e-5)
        _,unique=np.unique(x[:,variable].T,axis=0,return_index=True);indices=variable[unique]
        np.savez(args.output/(stage+'.npz'),indices=indices,scale=np.maximum(x[:,indices].std(0),1e-4))
    report.update(status='completed',actual_training_cpu_lp_calls=report['cpu_lp_calls'],total_seconds=time.perf_counter()-started)
    save()


if __name__=='__main__':main()
