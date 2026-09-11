"""Matched dynamic microbatch comparison; includes assembly/transfers/repair.

Environment construction and bank loading are separately reported. CPU is
the unchanged three-LP HiGHS reference, with optional concurrent LP workers.
All requested steps must finish and endpoint gates pass before a
speedup is reported. A one-step result is NOT a 120-step qualification.
"""
import argparse,hashlib,json,sys,time,traceback,shutil,copy,warnings,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from greenlet import greenlet,getcurrent
from scripts.benchmark_basis_bank_rollout import environment,snapshot
from src.fba_surrogate import model_fingerprint
from src.gpu_certified_basis import CommunityCoordinates
from src.compiled_basis_artifact import load_anchor
from src.compiled_basis_compatibility import equivalent_offline_compiler
from src.gpu_basis_bank import GpuBasisBank
from src.gpu_batched_compiled_backend import BatchedCompiledBackend,YieldingLPBackend
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend
from scripts.microbatch_comparison_support import ParallelCpuLP,drive_microbatch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--environments',type=int,default=8);p.add_argument('--steps',type=int,default=1)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--pivots',type=int,default=32)
    p.add_argument('--seed',type=int,default=20286411)
    p.add_argument('--cpu-workers',type=int,default=1)
    p.add_argument('--fresh-environments',action='store_true')
    p.add_argument('--extra-bank',type=Path)
    p.add_argument('--reserve-bank',type=Path)
    p.add_argument('--reserve-steps',type=int,nargs='+',help='Load only declared offline time strata to bound VRAM')
    p.add_argument('--neural-proposals',type=Path)
    p.add_argument('--neural-repair-first',action='store_true')
    p.add_argument('--constant-proposal',action='store_true')
    p.add_argument('--aggregate-portfolio',type=int,default=0)
    p.add_argument('--dual-edge',choices=['dantzig','devex'],default='dantzig')
    p.add_argument('--repair-cache-size',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.environments,args.steps,args.repeats,args.pivots,args.cpu_workers)<1:raise ValueError('Positive sizes required')
    if args.steps>120:raise ValueError('At most 120 defined actions')
    if args.constant_proposal and args.neural_proposals:raise ValueError('Select only one proposal source')
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings('ignore',message='Unrecognized options detected:.*',category=OptimizeWarning)
    import cupy as cp
    report=dict(status='setup',configuration=vars(args)|{'output':str(args.output),
        'extra_bank':str(args.extra_bank) if args.extra_bank else None,
        'reserve_bank':str(args.reserve_bank) if args.reserve_bank else None,
        'neural_proposals':str(args.neural_proposals) if args.neural_proposals else None},
        scope='Coupled host dFBA with GPU microbatched LPs; not fully device-resident environment',
        cpu_baseline='Original three-LP HiGHS dual simplex; threads=1, parallel=False per LP; '+
            f'{args.cpu_workers} concurrent LP workers; host assembly on main thread',runs=[],
        runtime=dict(cupy=cp.__version__,CUPY_ACCELERATORS=os.environ.get('CUPY_ACCELERATORS','default'),
            OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS')))
    sources=['src/gpu_batched_compiled_backend.py','src/gpu_revised_basis.py','src/gpu_capture_math.py',
        'src/gpu_conditional_capture.py','src/gpu_replay_call.py','src/gpu_optimal_face.py','src/gpu_certified_basis.py',
        'src/gpu_basis_bank.py','src/gpu_exchange_tie_break.py','src/community_solver.py','src/dfba_simulator.py',
        'src/compiled_basis_artifact.py','src/compiled_basis_compatibility.py',
        'scripts/microbatch_comparison_support.py','src/gpu_neural_basis_proposal.py',str(Path(__file__).relative_to(ROOT))]
    directory=args.output.with_suffix('.sources');directory.mkdir(parents=True,exist_ok=False)
    report['source_hashes']={}
    for name in sources:
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        report['source_hashes'][name]=hashlib.sha256(target.read_bytes()).hexdigest()
    def save():
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(args.output)
    save();started=time.perf_counter()
    sample,layout=environment(args.seed)
    fingerprints={k:model_fingerprint(v) for k,v in sample.simulator.models.items()}
    meta=dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for model in sample.simulator.models.values() for r in model.reactions],
        reaction_species=[k for k,model in sample.simulator.models.items() for r in model.reactions])
    cache=ROOT/'results/pf_basis_train20286311_4_compiled'
    manifest=json.loads((cache/'manifest.json').read_text())
    if manifest['identity']['train_seed'] in range(args.seed,args.seed+args.environments):
        raise ValueError('Base-bank training/evaluation seed overlap')
    if manifest['identity']['model_fingerprints']!=fingerprints:raise ValueError('Stale models in bank')
    old=(ROOT/'results/pf_gpu_basis_face_short_20260903.sources/src/gpu_certified_basis.py').read_bytes()
    if not equivalent_offline_compiler(old,(ROOT/'src/gpu_certified_basis.py').read_bytes(),manifest['identity']['compiler_source_sha256']):
        raise ValueError('Compiler changed')
    if manifest['identity']['artifact_source_sha256']!=report['source_hashes']['src/compiled_basis_artifact.py']:
        raise ValueError('Artifact loader changed')
    groups=defaultdict(list)
    for entry in manifest['entries']:
        if entry['key'][0]=='exchange_tie':continue
        path=(cache/entry['filename']).resolve()
        if not path.is_relative_to(cache.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Invalid anchor artifact')
        groups[tuple(entry['key'])].append(load_anchor(path))
    first=next(iter(groups.values()))[0]
    if args.extra_bank is not None:
        extension=json.loads((args.extra_bank/'manifest.json').read_text())
        identity=extension['identity']
        if extension['status']!='completed' or identity['model_fingerprints']!=fingerprints:
            raise ValueError('Incomplete or stale extension')
        if set(identity['train_seeds'])&set(range(args.seed,args.seed+args.environments)):
            raise ValueError('Training/evaluation seed overlap')
        if identity['artifact_source_sha256']!=report['source_hashes']['src/compiled_basis_artifact.py']:
            raise ValueError('Extension artifact loader changed')
        previous=(args.extra_bank/'sources/src/gpu_certified_basis.py').read_bytes()
        if not equivalent_offline_compiler(previous,(ROOT/'src/gpu_certified_basis.py').read_bytes(),identity['compiler_source_sha256']):
            raise ValueError('Extension compiler mismatch')
        for entry in extension['entries']:
            path=(args.extra_bank/entry['filename']).resolve()
            if not path.is_relative_to(args.extra_bank.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Invalid extension artifact')
            groups[tuple(entry['key'])].append(load_anchor(path))
        report['additional_offline_bank_provenance']=extension
    coordinates=CommunityCoordinates(meta,first['lp'].a,first['lp'].neq)
    banks={}
    for key,anchors in groups.items():
        rows=list(range(key[-1],key[-1]+3))
        if key[0]=='exchange':rows.append(key[-1]+3+len(meta['exchange_terms']))
        banks[key]=GpuBasisBank(anchors,rows)
        banks[key].evaluate_device(**banks[key].prepare_host([anchors[0]['lp']]*args.environments))
    cp.cuda.get_current_stream().synchronize()
    if args.neural_proposals is not None:
        from src.gpu_neural_basis_proposal import NeuralBasisProposal,NeuralRoutedBasisBank
        neural_manifest=json.loads((args.neural_proposals/'manifest.json').read_text())
        if neural_manifest['status']!='completed':raise ValueError('Incomplete neural training')
        if set(neural_manifest.get('actual_training_seeds',[]))&set(range(args.seed,args.seed+args.environments)):
            raise ValueError('Neural training/evaluation seed overlap')
        for entry in neural_manifest['stages']:
            path=(args.neural_proposals/entry['filename']).resolve()
            if not path.is_relative_to(args.neural_proposals.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Invalid neural artifact')
            key=next(k for k in banks if k[0]==entry['stage'])
            banks[key]=NeuralRoutedBasisBank(banks[key],NeuralBasisProposal(path,banks[key]),
                defer_exhaustive=args.neural_repair_first)
        report['neural_training_provenance']=neural_manifest
    if args.constant_proposal:
        from src.gpu_neural_basis_proposal import ConstantBasisProposal,NeuralRoutedBasisBank
        banks={key:NeuralRoutedBasisBank(bank,ConstantBasisProposal(),defer_exhaustive=True)
               for key,bank in banks.items()}
        report['proposal_ablation']='Always first offline basis; no neural inference; same certified repair-first schedule'
    reserve_banks={}
    if args.reserve_bank:
        reserve_manifest=json.loads((args.reserve_bank/'manifest.json').read_text())
        identity=reserve_manifest['identity']
        if reserve_manifest['status']!='completed' or identity['model_fingerprints']!=fingerprints:
            raise ValueError('Incomplete or stale reserve bank')
        if set(identity['train_seeds'])&set(range(args.seed,args.seed+args.environments)):
            raise ValueError('Reserve training/evaluation seed overlap')
        if identity['artifact_source_sha256']!=report['source_hashes']['src/compiled_basis_artifact.py']:
            raise ValueError('Reserve artifact loader mismatch')
        previous=(args.reserve_bank/'sources/src/gpu_certified_basis.py').read_bytes()
        if not equivalent_offline_compiler(previous,(ROOT/'src/gpu_certified_basis.py').read_bytes(),identity['compiler_source_sha256']):
            raise ValueError('Reserve compiler mismatch')
        reserve_groups=defaultdict(list)
        for entry in reserve_manifest['entries']:
            if args.reserve_steps and entry['step'] not in args.reserve_steps:continue
            path=(args.reserve_bank/entry['filename']).resolve()
            if not path.is_relative_to(args.reserve_bank.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Invalid reserve artifact')
            if entry['key'][0]!='aggregate':raise ValueError('Reserve currently supports aggregate stage only')
            reserve_groups[tuple(entry['key'])].append(load_anchor(path))
        for key,anchors in reserve_groups.items():
            reserve_banks[key]=GpuBasisBank(anchors,list(range(key[-1],key[-1]+3)))
        report['reserve_bank_provenance']=reserve_manifest
        report['loaded_reserve_candidates']=sum(len(anchors) for anchors in reserve_groups.values())
        if not reserve_banks:raise ValueError('No reserve candidates selected')
    service=BatchedCompiledBackend(coordinates,banks,args.pivots,aggregate_portfolio=args.aggregate_portfolio,
        dual_edge=args.dual_edge,reserve_banks=reserve_banks,repair_cache_size=args.repair_cache_size)
    report.update(bank_load_and_initial_environment_seconds=time.perf_counter()-started,
        bank_provenance=manifest,model_fingerprints=fingerprints,status='benchmark')
    save()
    seeds=list(range(args.seed,args.seed+args.environments))
    actions=[np.random.default_rng(seed).uniform(.05,.95,(120,5)).astype(np.float32)[:args.steps] for seed in seeds]
    def forbidden(*args,**kw):raise AssertionError('Online CPU optimization forbidden')
    def make_environment(seed):
        if args.fresh_environments:return environment(seed)
        env=copy.deepcopy(sample);env.reset(seed=seed)
        assert all(env.simulator.models[k] is not sample.simulator.models[k] for k in fingerprints)
        return env,env.simulator._cooperative_solver
    for repeat in range(args.repeats):
        row=dict(repeat=repeat,cold_graph_setup_included=repeat==0)
        report['runs'].append(row)
        started=time.perf_counter()
        cpu_env=[];gpu_env=[]
        for seed in seeds:
            cpu_env.append(make_environment(seed));gpu_env.append(make_environment(seed))
            print(f'repeat {repeat}: prepared paired environment {len(cpu_env)}/{len(seeds)}',flush=True)
        row['environment_construction_seconds']=time.perf_counter()-started
        for env,model in cpu_env+gpu_env:
            if {k:model_fingerprint(v) for k,v in env.simulator.models.items()}!=fingerprints:raise ValueError('Models changed')
        for env,model in cpu_env:model._linprog_options.update(threads=1,parallel=False)
        parent=getcurrent()
        cpu_service=ParallelCpuLP(args.cpu_workers) if args.cpu_workers>1 else None
        started=time.perf_counter();cpu_rows=[]
        if cpu_service is None:
            for i,(env,model) in enumerate(cpu_env):
                for action in actions[i]:
                    env.step(action)
                    if not model.stats.status.startswith('optimal;'):raise RuntimeError(model.stats.status)
                cpu_rows.append(snapshot(env))
        else:
            try:
                with patch('src.community_solver.linprog',lambda c,**kw:parent.switch((c,kw))):
                    cpu_rows=drive_microbatch(cpu_env,actions,cpu_service,snapshot)
            finally:cpu_service.close()
        row['cpu_seconds']=time.perf_counter()-started;row['cpu_rows']=cpu_rows
        row['cpu_lp_calls']=sum(model.cpu_lp_stage_calls for env,model in cpu_env)
        if cpu_service is not None:row['cpu_history']=cpu_service.history
        print(f'repeat {repeat}: CPU {row["cpu_seconds"]:.3f} s',flush=True);save()
        parent=getcurrent()
        for env,model in gpu_env:
            model.linear_program_backend=GpuExchangeTieBreakBackend(model,
                inner=YieldingLPBackend(parent))
        history_start=len(service.history);started=time.perf_counter();failure=None;gpu_rows=[]
        progress=[0]*args.environments
        try:
            with patch('highspy.Highs.run',forbidden),patch('src.community_solver.linprog',forbidden),patch('scipy.optimize.linprog',forbidden):
                gpu_rows=drive_microbatch(gpu_env,actions,service,snapshot,progress)
        except Exception as error:failure=dict(error=str(error),traceback=traceback.format_exc())
        cp.cuda.get_current_stream().synchronize()
        row.update(gpu_seconds=time.perf_counter()-started,gpu_rows=gpu_rows,failure=failure,
            gpu_history=service.history[history_start:],gpu_completed_steps=progress,
            online_cpu_lp_calls=sum(model.cpu_lp_stage_calls for env,model in gpu_env))
        if failure is not None and hasattr(service,'last_problems'):
            problems=service.last_problems
            diagnostic_path=args.output.with_suffix('.failure.npz')
            payload={}
            for i,problem in enumerate(problems):
                for name in ('rhs','lower','upper','c','col_scale','row_scale'):
                    payload[f'{i}_{name}']=getattr(problem,name)
                payload.update({f'{i}_a_'+name:getattr(problem.a,name) for name in ('data','indices','indptr','shape')})
                payload[f'{i}_neq']=problem.neq
            np.savez_compressed(diagnostic_path,**payload)
            row['failure_diagnostic']=dict(path=str(diagnostic_path),
                warning='Evaluation inputs for debugging ONLY; excluded from offline training')
        if failure is None:
            errors=[dict(pha_relative=abs(a['pha']-b['pha'])/max(abs(a['pha']),1e-9),
                biomass_g_l=max(abs(a['biomass'][k]-b['biomass'][k]) for k in a['biomass']),
                phv_fraction=abs(a['phv_fraction']-b['phv_fraction'])) for a,b in zip(cpu_rows,gpu_rows)]
            row['errors']=errors;row['all_endpoint_gates_passed']=all(v<=.01 for e in errors for v in e.values())
            if row['all_endpoint_gates_passed']:row['cpu_over_gpu_ratio']=row['cpu_seconds']/row['gpu_seconds']
        print(f'repeat {repeat}: GPU {row["gpu_seconds"]:.3f} s; failure={failure is not None}; ratio={row.get("cpu_over_gpu_ratio")}',flush=True)
        save()
        if failure is not None:break
    report['status']='completed' if len(report['runs'])==args.repeats and all(r.get('all_endpoint_gates_passed') for r in report['runs']) else 'incomplete_or_accuracy_failed'
    save()


if __name__=='__main__':main()
