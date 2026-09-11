"""Matched end-to-end dFBA comparison with compact certified GPU LP maps."""
import argparse,copy,gc,hashlib,json,os,shutil,sys,time,traceback,warnings
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import OptimizeWarning
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from greenlet import getcurrent
from scripts.benchmark_basis_bank_rollout import environment,snapshot
from scripts.microbatch_comparison_support import ParallelCpuLP,drive_microbatch
from src.gpu_certified_basis import CommunityCoordinates
from src.gpu_compact_basis import CompactBank
from src.gpu_compact_backend import CompactBackend
from src.gpu_batched_compiled_backend import YieldingLPBackend
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend
from src.fba_surrogate import model_fingerprint


def checked_npz(directory,filename,digest):
    path=(directory/filename).resolve()
    if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Invalid compact artifact: '+filename)
    with np.load(path,allow_pickle=False) as data:return dict(data)


def repeat_schedule(seed,environments,repeats,seed_stride=0,execution_order='cpu-first',train_seeds=()):
    """Validate every evaluation seed before starting setup or any trajectory."""
    if min(environments,repeats)<1:raise ValueError('Invalid repeat schedule size')
    if execution_order not in ('cpu-first','gpu-first','alternate'):
        raise ValueError('Invalid execution order')
    training=set(train_seeds);schedule=[]
    for repeat in range(repeats):
        first=seed+repeat*seed_stride
        seeds=list(range(first,first+environments))
        if first<0:raise ValueError('Evaluation seeds must be nonnegative')
        if set(seeds)&training:raise ValueError('Training/test overlap')
        order=('cpu-first' if repeat%2==0 else 'gpu-first') if execution_order=='alternate' else execution_order
        schedule.append(dict(repeat=repeat,seeds=seeds,execution_order=order))
    return schedule


def matched_actions(seeds,steps):
    # Preserve the original 120-row RNG draw followed by slicing. CPU and
    # GPU receive the same actions, regardless of their execution order.
    return [np.random.default_rng(seed).uniform(.05,.95,(120,5)).astype(np.float32)[:steps] for seed in seeds]


def run_ordered_pair(execution_order,cpu_call,gpu_call):
    """Run side-local contexts serially; return results in CPU/GPU order."""
    if execution_order=='cpu-first':return cpu_call(),gpu_call()
    if execution_order=='gpu-first':
        gpu_result=gpu_call();return cpu_call(),gpu_result
    raise ValueError('Execution order must be resolved for this repeat')


def save_failed_cpu_lps(prefix, requests, diagnostics):
    """Diagnostic-only original LPs; never training inputs or valid timings."""
    from src.cpu_repeated_lp import _problem
    artifacts=[]
    for index,row in enumerate(diagnostics):
        if row.get('success',True):continue
        a,rhs,lower,upper,c,neq=_problem(*requests[index])
        path=prefix.with_suffix(f'.cpu_failure_{index}.npz')
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,a_data=a.data,a_indices=a.indices,a_indptr=a.indptr,
            a_shape=a.shape,rhs=rhs,lower=lower,upper=upper,c=c,neq=neq)
        artifacts.append(dict(path=str(path),batch_index=index,
            environment_id=row.get('environment_id',index),stage=row.get('stage'),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return artifacts


def cpu_solver_run_count(history):
    """Count optimizer attempts separately from distinct LP requests."""
    total=0
    for stage in history:
        rows=stage.get('rows')
        if rows is None:
            rows=[row for group in stage.get('groups',[]) if group.get('route')=='cpu_fallback'
                  for row in group.get('rows',[])]
        total+=sum(row.get('cpu_solver_runs',1) for row in rows) if rows else stage.get('cpu_lp_calls',0)
    return total


def json_finite_values(value):
    """Strict JSON: unavailable/nonfinite diagnostics are null, never zero."""
    if isinstance(value,float):return value if np.isfinite(value) else None
    if isinstance(value,dict):return {key:json_finite_values(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_finite_values(item) for item in value]
    return value


def make_hybrid_backend(args, coordinates, keyed_banks, coverage_routers=None,
                        cpu_coverage_routers=None):
    """Zero-repair dictionaries need no full inverse repair artifacts."""
    from src.gpu_hybrid_lp import HybridCertifiedBackend
    if args.hybrid_rounds and not args.repair_operators:
        raise ValueError('Positive GPU repair rounds require verified repair operators')
    return HybridCertifiedBackend(coordinates,keyed_banks,cpu_workers=args.cpu_workers,
        dispersion_threshold=args.dispersion_threshold,candidate_limit=args.candidate_limit,
        repair_rounds=args.hybrid_rounds,repair_columns=(args.restricted_columns or [64])[0],
        repair_pivots=args.restricted_pivots,max_repair_batch=args.max_repair_batch,
        repair_cache_size=args.repair_cache_size,bucket=args.restricted_bucket,
        cohort_candidates=args.cohort_candidates,cohort_replay=args.cohort_replay,
        cpu_basis_proposals=args.cpu_basis_proposals,
        heterogeneous_candidates=args.heterogeneous_candidates,
        heterogeneous_replay=args.heterogeneous_replay,
        candidate_diagnostics=args.candidate_diagnostics,candidate_oracle=args.candidate_oracle,
        repair_policy=args.hybrid_repair_policy,compact_repair_workspace=args.compact_repair_workspace,
        gpu_stages=args.gpu_stages,selected_features=not args.full_feature_encoding,
        cpu_basis_handoff=args.cpu_basis_handoff,
        cpu_exchange_support_updates=args.cpu_exchange_support_updates,
        speculative_cpu=args.speculative_cpu,coverage_routers=coverage_routers,
        cpu_coverage_routers=cpu_coverage_routers,
        temporal_candidate_policy=getattr(args,'temporal_candidate_policy','prepend'),
        certificate_only=getattr(args,'certificate_only',False),
        packed_result_transfer=getattr(args,'packed_result_transfer',False),
        device_routing=getattr(args,'device_routing',False),
        speculative_dispatch_phase=getattr(args,'speculative_dispatch_phase','before-prepare'),
        gpu_first_batch=getattr(args,'gpu_first_batch',False),
        gpu_cpu_fallback=getattr(args,'gpu_cpu_fallback','exact'))


def benchmark_driver_kind(side, *, pipeline_cpu_stages=False, gpu_first_batch=False):
    """Return the explicit scheduling policy for one side of a comparison.

    The GPU-first experiment deliberately retains the stronger asynchronous
    CPU comparator while giving all three GPU stages full-cohort barriers.
    This is an end-to-end policy comparison, not an isolated solver timing.
    """
    if side not in ('cpu','gpu'):
        raise ValueError('Benchmark side must be cpu or gpu')
    if pipeline_cpu_stages and not (side == 'gpu' and gpu_first_batch):
        return 'pipeline'
    return 'barrier'


def drive_benchmark_side(environments, actions, backend, snapshot_fn, progress, *,
                         side, pipeline_cpu_stages=False, gpu_first_batch=False,
                         workers=4, pipeline_driver=None, barrier_driver=None):
    """Drive one side with the recorded scheduling policy.

    Optional driver injection keeps the scheduler selection independently
    testable without importing CuPy or constructing a biological environment.
    """
    kind = benchmark_driver_kind(side, pipeline_cpu_stages=pipeline_cpu_stages,
                                 gpu_first_batch=gpu_first_batch)
    if kind == 'pipeline':
        if pipeline_driver is None:
            from scripts.pipelined_microbatch import drive_pipelined
            pipeline_driver = drive_pipelined
        return pipeline_driver(environments, actions, backend, snapshot_fn,
                               progress, workers=workers)
    if barrier_driver is None:
        barrier_driver = drive_microbatch
    return barrier_driver(environments, actions, backend, snapshot_fn, progress)


def forbid_highs_in_gpu_run(*, hybrid=False, gpu_first_batch=False,
                             gpu_cpu_fallback='exact'):
    """Whether an online HiGHS call is forbidden on the measured GPU side."""
    return (not hybrid) or (gpu_first_batch and gpu_cpu_fallback == 'reject')


def diagnostic_only_performance(*, candidate_oracle=False, gpu_first_batch=False,
                                gpu_cpu_fallback='exact'):
    """Return true when a run is informative but cannot carry a speed ratio."""
    return bool(candidate_oracle or
                (gpu_first_batch and gpu_cpu_fallback == 'reject'))


def coverage_router_specs(primary, primary_sha, additional, *, gpu_first_batch=False):
    """One externally SHA-pinned router per stage; legacy primary is maxmin."""
    if (primary is None) != (primary_sha is None):
        raise ValueError('Coverage router path and SHA256 must be supplied together')
    specs = {} if primary is None else {'maxmin': (Path(primary), primary_sha)}
    for row in additional or []:
        if len(row) != 3:
            raise ValueError('Stage router requires STAGE PATH SHA256')
        stage, path, digest = row
        if not gpu_first_batch or stage not in ('maxmin','aggregate','exchange') or stage in specs:
            raise ValueError('Stage routers require GPU-first batch and unique original stages')
        specs[stage] = (Path(path), digest)
    if any(not isinstance(digest,str) or len(digest)!=64 or
           any(c not in '0123456789abcdef' for c in digest) for _,digest in specs.values()):
        raise ValueError('Coverage router requires a lowercase SHA256')
    return specs


def record_performance_comparison(row, complete, *, diagnostic_only=False):
    """Diagnostic full-bank scans or CPU-prohibited runs carry no speed ratio."""
    if diagnostic_only:
        row.pop('cpu_over_gpu_ratio', None)
        row['performance_comparison_valid'] = False
        row['performance_comparison_note'] = 'Full-dictionary oracle or CPU-prohibited coverage work is diagnostic only; speed ratio intentionally omitted.'
    elif row.get('all_endpoint_gates_passed') and complete:
        row['cpu_over_gpu_ratio'] = row['cpu_seconds']/row['gpu_seconds']
        if ('cpu_seconds_including_coverage_router_setup' in row
                and 'gpu_seconds_including_coverage_router_setup' in row):
            row['cpu_over_gpu_ratio_including_coverage_router_setup'] = (
                row['cpu_seconds_including_coverage_router_setup']
                / row['gpu_seconds_including_coverage_router_setup'])


class Progress(list):
    """Equal, infrequent progress reporting for CPU and GPU timed runs."""
    def __init__(self,count,label):
        super().__init__([0]*count);self.label=label;self.last=0;self.started=time.perf_counter()
    def __setitem__(self,index,value):
        super().__setitem__(index,value);completed=min(self)
        if completed>=self.last+20:
            self.last=completed
            print(f'{self.label}: all {len(self)} environments at step {completed}; {time.perf_counter()-self.started:.1f} s',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--environments',type=int,default=8)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20287201)
    p.add_argument('--seed-stride',type=int,default=0)
    p.add_argument('--execution-order',choices=['cpu-first','gpu-first','alternate'],default='cpu-first')
    p.add_argument('--cpu-backend',choices=['fresh','persistent','dictionary'],default='fresh')
    p.add_argument('--full-feature-encoding',action='store_true',
        help='Ablation: encode every LP routing feature before selecting columns, on both CPU and GPU')
    p.add_argument('--gpu-stages',nargs='+',choices=['maxmin','aggregate','exchange'],
        default=['maxmin','aggregate','exchange'],
        help='Hybrid only: other stages use the exact dictionary-initialized CPU comparator directly')
    p.add_argument('--pipeline-cpu-stages',action='store_true',
        help='Overlap aggregate/exchange CPU solves; GPU-first batch retains this only for the CPU comparator')
    p.add_argument('--gpu-first-batch',action='store_true',
        help='Opt-in: retain the pipelined CPU comparator but batch all three GPU stages before exact fallback')
    p.add_argument('--gpu-cpu-fallback',choices=['exact','reject'],default='exact',
        help='GPU-first batch only: exact solves rejected rows, while reject is diagnostic and has no speed ratio')
    p.add_argument('--stage-coverage-router',action='append',nargs=3,default=[],
        metavar=('STAGE','PATH','SHA256'),
        help='GPU-first batch: additional explicitly bound stage router; repeat for multiple stages')
    p.add_argument('--frozen-inputs',action='store_true',
        help='Both sides: opt-in fixed-GEM array bounds/objectives; initialization counted in environment setup')
    p.add_argument('--cpu-exchange-support-updates',action='store_true',
        help='Both sides: retain exact HiGHS models across validated exchange performance-row support switches')
    p.add_argument('--hybrid',action='store_true')
    p.add_argument('--speculative-cpu',action='store_true',
        help='Opt-in hybrid: overlap maxmin GPU certificate with cancellable exact CPU tasks; not GPU-only')
    p.add_argument('--dispersion-threshold',type=float,default=1.)
    p.add_argument('--candidate-limit',type=int,default=0)
    p.add_argument('--coverage-router',type=Path,
        help='Opt-in SHA-pinned maxmin multi-label coverage router; candidate ordering only')
    p.add_argument('--coverage-router-sha256',
        help='Required lowercase SHA256 of --coverage-router')
    p.add_argument('--temporal-candidate-policy',choices=['prepend','within-budget','off'],default='prepend')
    p.add_argument('--certificate-only',action='store_true',
        help='Speculative heterogeneous mode: omit unused repair diagnostics, never the original LP certificate')
    p.add_argument('--packed-result-transfer',action='store_true')
    p.add_argument('--speculative-dispatch-phase',choices=['before-prepare','after-submit'],default='before-prepare')
    p.add_argument('--device-routing',action='store_true',
        help='Keep learned ranks on GPU and propagate row validity to the final original-LP gate')
    p.add_argument('--candidate-diagnostics',action='store_true',
        help='Save pre-CPU heterogeneous B×K screening metrics; timed diagnostic transfers add overhead')
    p.add_argument('--candidate-oracle',action='store_true',
        help='Diagnostic-only all-dictionary counterfactual; requires candidate diagnostics and suppresses speed ratios')
    p.add_argument('--hybrid-rounds',type=int,default=2)
    p.add_argument('--hybrid-repair-policy',choices=['dispersion','maxmin-dual'],default='dispersion')
    p.add_argument('--compact-repair-workspace',action='store_true',
        help='Bound restricted update capacity by pivot budget and omit unused expansion pricing')
    p.add_argument('--cohort-candidates',action='store_true')
    p.add_argument('--cohort-replay',action='store_true')
    p.add_argument('--heterogeneous-candidates',action='store_true')
    p.add_argument('--heterogeneous-replay',action='store_true')
    p.add_argument('--cpu-basis-proposals',action='store_true')
    p.add_argument('--cpu-basis-handoff',action='store_true',
        help='Opt-in: offer the preceding certified GPU basis when an existing CPU model resumes')
    p.add_argument('--repair-operators',type=Path);p.add_argument('--pivots',type=int,default=48)
    p.add_argument('--repair-cache-size',type=int,default=3)
    p.add_argument('--repair-graph-cache-size',type=int,default=1)
    p.add_argument('--temporal-pivots',type=int,default=0)
    p.add_argument('--restricted-columns',type=int,nargs='+',default=[])
    p.add_argument('--restricted-pivots',type=int,default=256)
    p.add_argument('--restricted-polish-pivots',type=int,default=0)
    p.add_argument('--restricted-rounds',type=int,default=1)
    p.add_argument('--restricted-stage1-warm',action='store_true')
    p.add_argument('--restricted-bucket',action='store_true')
    p.add_argument('--max-repair-batch',type=int,default=8)
    p.add_argument('--compact-capture',action='store_true');p.add_argument('--full-batch-candidates',action='store_true')
    p.add_argument('--candidate-ranking',choices=['residual','count'],default='residual')
    p.add_argument('--exchange-ranking',choices=['residual','count','count_only'])
    p.add_argument('--pivot-refinement',type=int,default=0)
    p.add_argument('--compact-updates',action='store_true')
    p.add_argument('--reuse-small-factor',action='store_true')
    p.add_argument('--skip-unused-dual',action='store_true')
    p.add_argument('--adaptive-refinement-tolerance',type=float,default=0.)
    p.add_argument('--repair-portfolio',type=int,default=0)
    p.add_argument('--portfolio-pivots',type=int,default=32)
    p.add_argument('--portfolio-continuation',choices=['initial','best'],default='best')
    p.add_argument('--portfolio-ranking',choices=['bank','primal_count'],default='bank')
    p.add_argument('--tie-relative-allowance',type=float,default=1e-10)
    p.add_argument('--tie-policy',choices=['original3','secondary'],default='secondary')
    args=p.parse_args()
    router_specs=coverage_router_specs(args.coverage_router,args.coverage_router_sha256,
        args.stage_coverage_router,gpu_first_batch=args.gpu_first_batch)
    if args.speculative_dispatch_phase!='before-prepare' and not args.speculative_cpu:
        raise ValueError('Delayed CPU dispatch requires speculative mode')
    if (args.packed_result_transfer or args.device_routing) and not (
            args.speculative_cpu or args.gpu_first_batch):
        raise ValueError('Device routing/packed transfer require speculative or GPU-first batch mode')
    if args.device_routing and (not args.heterogeneous_candidates or
            (args.coverage_router is None and not args.gpu_first_batch)):
        raise ValueError('Device routing requires heterogeneous candidates and a router unless GPU-first uses nearest ranking')
    if args.certificate_only and not ((args.speculative_cpu or args.gpu_first_batch)
                                      and args.heterogeneous_candidates):
        raise ValueError('Certificate-only requires speculative or GPU-first heterogeneous probes')
    if args.hybrid and args.hybrid_rounds and not args.repair_operators:
        raise ValueError('Positive GPU repair rounds require verified repair operators')
    if args.speculative_cpu and (not args.hybrid or args.cpu_backend!='dictionary'
            or set(args.gpu_stages)!={'maxmin'} or args.hybrid_rounds!=0
            or args.cpu_basis_handoff or args.candidate_diagnostics or args.candidate_oracle
            or args.gpu_first_batch):
        raise ValueError('Speculative comparison requires maxmin-only hybrid, dictionary CPU, zero repair, no handoff/diagnostics')
    if args.gpu_cpu_fallback != 'exact' and not args.gpu_first_batch:
        raise ValueError('CPU reject policy requires GPU-first batch mode')
    if args.gpu_first_batch and (not args.hybrid or args.cpu_backend!='dictionary'
            or set(args.gpu_stages)!={'maxmin','aggregate','exchange'}
            or not args.pipeline_cpu_stages or args.speculative_cpu
            or not args.heterogeneous_candidates or args.hybrid_rounds!=0
            or args.tie_policy!='original3' or args.cpu_basis_handoff
            or args.candidate_diagnostics or args.candidate_oracle):
        raise ValueError('GPU-first batch requires the pipelined dictionary CPU comparator, all three GPU stages, heterogeneous zero-repair, original3, and no speculation/handoff/diagnostics')
    if args.cpu_exchange_support_updates and (not args.hybrid or args.cpu_backend!='dictionary'):
        raise ValueError('Exchange support comparison requires hybrid and dictionary CPU on both sides')
    if args.cpu_basis_handoff and (not args.pipeline_cpu_stages or not args.cpu_basis_proposals
                                   or args.hybrid_rounds != 0):
        raise ValueError('CPU basis handoff requires pipelining, CPU basis proposals, and zero repair rounds')
    pipeline_gpu_stages = ({'maxmin','aggregate','exchange'} if args.gpu_first_batch
                           else {'maxmin'})
    if args.pipeline_cpu_stages and (not args.hybrid or set(args.gpu_stages)!=pipeline_gpu_stages
            or args.cpu_backend!='dictionary' or args.tie_policy!='original3' or args.candidate_oracle):
        raise ValueError('Pipelining requires the configured hybrid stage policy, dictionary CPU, original3, and no oracle')
    if not args.hybrid and set(args.gpu_stages)!={'maxmin','aggregate','exchange'}:
        raise ValueError('Stage selection requires the hybrid backend')
    if args.hybrid and args.tie_policy!='original3':
        raise ValueError('Hybrid comparison requires the original3 objective sequence')
    if args.heterogeneous_candidates and (not args.hybrid or args.cohort_candidates or args.cohort_replay):
        raise ValueError('Heterogeneous candidates require hybrid mode without cohort evaluation')
    if args.heterogeneous_replay and not args.heterogeneous_candidates:
        raise ValueError('Heterogeneous replay requires heterogeneous candidates')
    if args.candidate_diagnostics and not (args.hybrid and args.heterogeneous_candidates):
        raise ValueError('Candidate diagnostics require heterogeneous hybrid candidates')
    if args.candidate_oracle and not (args.candidate_diagnostics and args.hybrid and args.heterogeneous_candidates):
        raise ValueError('Candidate oracle requires heterogeneous candidate diagnostics')
    if (args.coverage_router is None) != (args.coverage_router_sha256 is None):
        raise ValueError('Coverage router path and SHA256 must be supplied together')
    if args.coverage_router is not None and (not args.hybrid or 'maxmin' not in args.gpu_stages
                                             or args.cpu_backend != 'dictionary'):
        raise ValueError('Coverage router requires hybrid GPU maxmin and the dictionary CPU comparator')
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.steps,args.environments,args.repeats,args.cpu_workers)<1 or args.steps>120:raise ValueError('Invalid size')
    if not np.isfinite(args.tie_relative_allowance) or args.tie_relative_allowance<0:raise ValueError('Invalid tie allowance')
    warnings.filterwarnings('ignore',message='Unrecognized options detected:.*',category=OptimizeWarning)
    import cupy as cp
    manifest_path=args.bank/'manifest.json';manifest_bytes=manifest_path.read_bytes()
    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
    manifest=json.loads(manifest_bytes)
    if manifest['status']!='completed':raise ValueError('Incomplete bank')
    schedule=repeat_schedule(args.seed,args.environments,args.repeats,args.seed_stride,
        args.execution_order,manifest['train_seeds'])
    report=dict(status='setup',configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope='Host dFBA state/assembly plus GPU-certified LPs; not entirely GPU resident',
        reference='Original 3-stage CPU HiGHS; '+args.cpu_backend+' model/basis; 1 thread per LP; configured concurrent workers',
        dt_hours=.2,simulated_hours_per_environment=args.steps*.2,
        runtime=dict(cupy=cp.__version__,CUPY_ACCELERATORS=os.environ.get('CUPY_ACCELERATORS')),
        json_nonfinite_encoding='Nonfinite/unavailable diagnostics are null; consult status and certificate flags',
        offline_bank_manifest=manifest,runs=[])
    if args.candidate_diagnostics:
        report['candidate_diagnostic_scope']='Opt-in pre-CPU GPU candidate snapshots. Transfer/logging overhead '
        report['candidate_diagnostic_scope']+='is included in GPU timing; do not treat this diagnostic run as the unchanged performance baseline.'
    if args.candidate_oracle:
        report['performance_comparison_valid']=False
        report['candidate_oracle_scope']='Diagnostic-only all-dictionary counterfactual; normal K-candidate trajectories are unchanged. No speedup ratio is reported.'
    if args.gpu_first_batch:
        report['gpu_first_batch_policy']=dict(
            cpu_schedule='asynchronous aggregate/exchange CPU pipeline',
            gpu_schedule='full-cohort maxmin/aggregate/exchange stage barriers',
            cpu_fallback=args.gpu_cpu_fallback,
            comparison_kind='strong CPU pipeline versus end-to-end GPU stage batching',
            isolated_hardware_comparison=False)
        if args.gpu_cpu_fallback == 'reject':
            report['performance_comparison_valid']=False
            report['gpu_reject_scope']=(
                'Diagnostic GPU-only acceptance test. Uncertified rows stop the trajectory; '
                'no CPU/GPU speed ratio is valid.')
    sources=['src/gpu_restricted_basis.py','src/gpu_compact_basis.py','src/gpu_compact_backend.py','src/gpu_certified_basis.py',
        'src/gpu_revised_basis.py','src/gpu_capture_math.py','src/gpu_replay_call.py',
        'src/gpu_conditional_capture.py','src/gpu_optimal_face.py','src/gpu_compiled_community_backend.py',
        'src/gpu_exchange_tie_break.py','src/gpu_batched_compiled_backend.py','src/community_solver.py',
        'src/dfba_simulator.py','src/rl_environment.py','scripts/microbatch_comparison_support.py',
        'scripts/benchmark_basis_bank_rollout.py','scripts/benchmark_cooperative_surrogate_e2e.py',
        'scripts/benchmark_compact_gpu.py','src/cpu_repeated_lp.py','src/cpu_dictionary_lp.py','src/gpu_hybrid_lp.py',
        'src/selected_lp_features.py','src/gpu_neural_basis_proposal.py','src/frozen_community_inputs.py',
        'src/temporal_candidate_order.py','src/lp_bounds.py']
    if args.heterogeneous_candidates:sources.append('src/gpu_heterogeneous_compact.py')
    if args.speculative_cpu:
        sources.extend(['src/speculative_cpu_batch.py','src/gpu_speculative_lp.py','src/gpu_candidate_transfer.py'])
        report['speculative_count_scope']='online_cpu_lp_calls counts all actually executed CPU tasks, including unused results; GPU accepts and CPU calls overlap. Cancelled pending tasks are separate.'
    if args.gpu_first_batch:
        sources.extend(['src/gpu_first_batch_lp.py','src/gpu_candidate_transfer.py'])
    if args.pipeline_cpu_stages:
        sources.append('scripts/pipelined_microbatch.py')
        report['timing_scope']=(
            'The CPU comparator pipelines aggregate/exchange across environments. '
            + ('The GPU path instead uses full-cohort barriers for every stage; this is an end-to-end scheduling comparison, not an isolated hardware comparison. '
               if args.gpu_first_batch else
               'The GPU path uses the same CPU-tail pipeline. ')
            + 'Asynchronous CPU stage spans overlap host work and must not be added; compare total online wall times.')
    if router_specs:
        sources.extend(['src/coverage_router.py','src/coverage_router_binding.py'])
        report['coverage_router_comparator']=dict(
            gpu_cold_proposal='learned multi-label coverage order',
            cpu_cold_proposal='same learned multi-label coverage order',
            same_final_candidate_bank=True,
            same_learned_proposer=True,
            cpu_persistent_warm_basis=True,
            isolated_hardware_comparison=False,
            comparison_kind='matched-proposer end-to-end execution comparison')
        report['coverage_router_comparator_scope']=(
            'GPU and exact CPU receive the same final candidate banks and the same SHA-pinned FP32 learned stage proposal '
            'weights. CPU invokes the NumPy router only for cold/rebuilt models; an existing valid persistent HiGHS basis '
            'continues to take precedence. Proposal time remains inside each online solve span. This is an end-to-end execution '
            'comparison, not an isolated kernel hardware microbenchmark.')
    directory=args.output.with_suffix('.sources');directory.mkdir(parents=True,exist_ok=False)
    report['source_hashes']={}
    for name in sources:
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        report['source_hashes'][name]=hashlib.sha256(target.read_bytes()).hexdigest()
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(json_finite_values(report),indent=2,allow_nan=False));tmp.replace(args.output)
    save();started=time.perf_counter()
    sample,layout=environment(args.seed)
    fingerprints={k:model_fingerprint(v) for k,v in sample.simulator.models.items()}
    if fingerprints!=manifest['model_fingerprints']:raise ValueError('Stale models')
    banks={}
    for stage in manifest['stages']:
        folder=args.bank/stage['stage']
        root=checked_npz(folder,'root.npz',stage['root_sha256'])
        a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
        router=checked_npz(folder,'router.npz',stage['router_sha256'])
        entries=[checked_npz(folder,e['filename'],e['sha256']) for e in stage['entries']]
        banks[stage['stage']]=CompactBank(dict(a=a,neq=int(root['neq'])),root['variable_rows'],entries,
            router['centers'],router['indices'],router['scale'],capture=args.compact_capture,full_batch_candidates=args.full_batch_candidates,
            selected_features=not args.full_feature_encoding,
            candidate_ranking=(args.exchange_ranking if stage['stage']=='exchange' and args.exchange_ranking else args.candidate_ranking))
        del entries
    report['feature_encoding']=dict(mode='full_then_select' if args.full_feature_encoding else 'select_then_encode',
        scope='Componentwise routing feature extraction only; original LP matrices and certificates unchanged',
        selected_width={stage:int(bank.feature_indices.size) for stage,bank in banks.items()},
        original_width={stage:int(2*bank.host_a.shape[0]+4*bank.host_a.shape[1]+len(bank.variable_rows)*bank.host_a.shape[1])
                        for stage,bank in banks.items()})
    metadata=dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for m in sample.simulator.models.values() for r in m.reactions],
        reaction_species=[k for k,m in sample.simulator.models.items() for r in m.reactions])
    coordinates=CommunityCoordinates(metadata,banks['maxmin'].host_a,banks['maxmin'].neq)
    coverage_routers={};cpu_coverage_routers={}
    report['coverage_router_bindings']={}
    for router_stage,(router_path,router_sha) in router_specs.items():
        from src.coverage_router_binding import load_bound_coverage_router
        stage=next((item for item in manifest['stages'] if item['stage']==router_stage),None)
        if stage is None:raise ValueError('Coverage router requires its matching compact stage bank')
        before=time.perf_counter()
        router,binding=load_bound_coverage_router(router_path,
            router_sha,bank_manifest_sha256=manifest_sha256,
            stage_manifest=stage,bank=banks[router_stage],model_fingerprints=fingerprints,xp=cp)
        binding['gpu_binding_and_upload_setup_seconds']=time.perf_counter()-before
        coverage_routers[tuple(stage['key'])]=router
        before=time.perf_counter()
        cpu_router,cpu_binding=load_bound_coverage_router(router_path,
            router_sha,bank_manifest_sha256=manifest_sha256,
            stage_manifest=stage,bank=banks[router_stage],model_fingerprints=fingerprints,xp=np)
        binding['cpu_binding_and_numpy_setup_seconds']=time.perf_counter()-before
        if {key:value for key,value in cpu_binding.items() if not key.endswith('_seconds')} != {
                key:value for key,value in binding.items() if not key.endswith('_seconds')}:
            raise ValueError('CPU and GPU coverage-router bindings disagree')
        cpu_coverage_routers[tuple(stage['key'])]=cpu_router
        report['coverage_router_bindings'][router_stage]=binding
        if router_stage=='maxmin':report['coverage_router_binding']=binding
    if args.hybrid:
        from src.gpu_hybrid_lp import HybridCertifiedBackend,observable_projection
        for stage,bank in banks.items():
            projection,scales,labels=observable_projection(metadata,bank.host_a.shape[1])
            bank.configure_observables(projection,scales)
        report['scope']='Host dFBA plus certified GPU-first/CPU-fallback LPs; NOT GPU-only'
        if args.gpu_first_batch:
            report['scope']=(
                'Host dFBA plus certified full-cohort GPU batches for all three LP stages; '
                + ('exact CPU fallback for rejected rows; NOT GPU-only'
                   if args.gpu_cpu_fallback == 'exact' else
                   'diagnostic reject-on-uncertified policy; CPU LPs forbidden'))
        if args.candidate_oracle:
            report['scope']+='; DIAGNOSTIC-ONLY all-dictionary oracle, not a performance comparison'
        report['dispersion_definition']='Maximum per-component sample SD of species growth, exchange and PHA-related specific rates; explicit unit scales. Experimental cost-routing feature, NOT an error probability or acceptance gate.'
        report['observable_labels']=labels
    if args.repair_operators:
        from src.gpu_batched_compiled_backend import BatchedCompiledBackend
        repair=json.loads((args.repair_operators/'manifest.json').read_text())
        if repair['status']!='completed' or repair['source_manifest_sha256']!=hashlib.sha256((args.bank/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Incomplete/stale repair operators')
        for entry in repair['entries']:
            stage=next(s for s in manifest['stages'] if s['stage']==entry['stage'])
            if entry['source_sha256']!=stage['entries'][entry['index']]['sha256']:raise ValueError('Repair source mismatch')
            file=(args.repair_operators/entry['filename']).resolve()
            if not file.is_relative_to(args.repair_operators.resolve()):raise ValueError('Repair path outside artifact')
            banks[entry['stage']].evaluators[entry['index']].repair_path=(file,entry['sha256'])
        keyed_banks={tuple(s['key']):banks[s['stage']] for s in manifest['stages']}
        service=BatchedCompiledBackend(coordinates,keyed_banks,
            args.pivots,repair_cache_size=args.repair_cache_size,dual_edge='devex',pivot_refinement=args.pivot_refinement,
            compact_updates=args.compact_updates,repair_portfolio=args.repair_portfolio,
            portfolio_pivots=args.portfolio_pivots,portfolio_continuation=args.portfolio_continuation,require_tie=args.tie_policy=='secondary',
            max_repair_batch=args.max_repair_batch,verbose=True,reuse_small_factor=args.reuse_small_factor,
            skip_unused_dual=args.skip_unused_dual,portfolio_ranking=args.portfolio_ranking,
            adaptive_refinement_tolerance=args.adaptive_refinement_tolerance,repair_graph_cache_size=args.repair_graph_cache_size,
            temporal_pivots=args.temporal_pivots,restricted_columns=args.restricted_columns,
            restricted_pivots=args.restricted_pivots,restricted_polish_pivots=args.restricted_polish_pivots,
            restricted_rounds=args.restricted_rounds,restricted_stage1_warm=args.restricted_stage1_warm,
            restricted_bucket=args.restricted_bucket)
        if args.hybrid:
            service=make_hybrid_backend(args,coordinates,keyed_banks,coverage_routers,
                cpu_coverage_routers)
        report['repair_operator_manifest']=repair
    elif args.hybrid:
        keyed_banks={tuple(s['key']):banks[s['stage']] for s in manifest['stages']}
        service=make_hybrid_backend(args,coordinates,keyed_banks,coverage_routers,
            cpu_coverage_routers)
        report['repair_operator_scope']='Not loaded or generated: zero-repair hybrid uses compact maps and exact CPU fallback only'
    else:service=CompactBackend(coordinates,banks)
    cp.cuda.get_current_stream().synchronize()
    report.update(setup_seconds=time.perf_counter()-started,model_fingerprints=fingerprints,
        initial_gpu_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),status='benchmark')
    save()
    def make(seed):
        env=copy.deepcopy(sample);env.reset(seed=seed);model=env.simulator._cooperative_solver
        model._linprog_options.update(threads=1,parallel=False)
        if args.frozen_inputs:
            env.simulator.enable_frozen_community_inputs()
        return env,model
    def forbidden(*a,**kw):raise AssertionError('Online CPU optimization forbidden')

    def drive(envs,actions,backend,progress,side):
        return drive_benchmark_side(envs,actions,backend,snapshot,progress,
            side=side,pipeline_cpu_stages=args.pipeline_cpu_stages,
            gpu_first_batch=args.gpu_first_batch,workers=args.cpu_workers)

    def run_cpu(cpu_env,actions,row):
        parent=getcurrent()
        started=time.perf_counter()
        if args.cpu_backend=='dictionary':
            from src.cpu_dictionary_lp import CpuDictionaryLP
            cpu_service=CpuDictionaryLP(coordinates,{tuple(s['key']):banks[s['stage']] for s in manifest['stages']},workers=args.cpu_workers,
                selected_features=not args.full_feature_encoding,
                exchange_support_updates=args.cpu_exchange_support_updates,
                coverage_routers=cpu_coverage_routers)
        elif args.cpu_backend=='persistent':
            from src.cpu_repeated_lp import RepeatedCpuLP
            cpu_service=RepeatedCpuLP(args.cpu_workers,n_fluxes=coordinates.n_fluxes)
        else:cpu_service=ParallelCpuLP(args.cpu_workers)
        row['cpu_service_setup_seconds']=time.perf_counter()-started
        last_requests=[];original_solve=cpu_service.solve_batch
        def recorded_solve(requests):
            last_requests[:]=requests
            return original_solve(requests)
        cpu_service.solve_batch=recorded_solve
        progress=Progress(args.environments,'CPU')
        cp.cuda.get_current_stream().synchronize();started=time.perf_counter()
        try:
            with patch('src.community_solver.linprog',lambda c,**kw:parent.switch((c,kw))):
                cpu_rows=drive(cpu_env,actions,cpu_service,progress,'cpu')
            cpu_elapsed=time.perf_counter()-started
        except Exception as error:
            row.update(cpu_seconds=time.perf_counter()-started,cpu_history=cpu_service.history,
                cpu_solver_runs=cpu_solver_run_count(cpu_service.history),
                cpu_completed_steps=list(progress),all_endpoint_gates_passed=False,
                failure=dict(side='cpu_reference',error=str(error),traceback=traceback.format_exc()))
            report['status']='cpu_reference_failed'
            try:
                rows=cpu_service.history[-1].get('rows',[]) if cpu_service.history else []
                failed_requests=getattr(cpu_service,'pipeline_failed_requests',None) or last_requests
                rows=getattr(cpu_service,'pipeline_failed_diagnostics',None) or rows
                row['cpu_failure_artifacts']=save_failed_cpu_lps(args.output,failed_requests,rows)
                row['diagnostic_inputs_warning']='CPU-failed evaluation LPs for diagnosis only; excluded from training'
            except Exception as capture_error:
                row['cpu_failure_artifact_error']=str(capture_error)
            save()
            raise
        finally:cpu_service.close()
        # As on the hybrid side, executor teardown is outside online timing.
        row.update(cpu_seconds=cpu_elapsed,cpu_rows=cpu_rows,cpu_history=cpu_service.history,
            cpu_completed_steps=list(progress),cpu_solver_runs=cpu_solver_run_count(cpu_service.history))
        if args.pipeline_cpu_stages:
            row['cpu_pipeline_history']=getattr(cpu_service,'pipeline_history',[])
        print(f'CPU {row["cpu_seconds"]:.3f} s',flush=True);save()
        return cpu_rows

    def run_gpu(gpu_env,actions,row):
        parent=getcurrent()
        for env,model in gpu_env:
            inner=YieldingLPBackend(parent)
            model.linear_program_backend=GpuExchangeTieBreakBackend(model,inner=inner) if args.tie_policy=='secondary' else inner
            if args.tie_policy=='secondary':model.linear_program_backend.primary_face_relative_allowance=args.tie_relative_allowance
        first=len(service.history);first_pipeline=len(getattr(service,'pipeline_history',[]))
        progress=Progress(args.environments,'GPU');failure=None;gpu_rows=[]
        cp.cuda.get_current_stream().synchronize();started=time.perf_counter()
        try:
            with ExitStack() as stack:
                if forbid_highs_in_gpu_run(hybrid=args.hybrid,
                        gpu_first_batch=args.gpu_first_batch,
                        gpu_cpu_fallback=args.gpu_cpu_fallback):
                    stack.enter_context(patch('highspy.Highs.run',forbidden))
                stack.enter_context(patch('src.community_solver.linprog',forbidden))
                stack.enter_context(patch('scipy.optimize.linprog',forbidden))
                gpu_rows=drive(gpu_env,actions,service,progress,'gpu')
                if (args.gpu_first_batch and args.gpu_cpu_fallback == 'reject'
                        and (any(h.get('cpu_lp_calls',0) for h in service.history[first:])
                             or cpu_solver_run_count(service.history[first:]))):
                    raise AssertionError('GPU reject policy executed a CPU LP')
        except Exception as error:failure=dict(error=str(error),traceback=traceback.format_exc())
        cp.cuda.get_current_stream().synchronize()
        row.update(gpu_seconds=time.perf_counter()-started,gpu_rows=gpu_rows,gpu_completed_steps=progress,
            failure=failure,gpu_history=service.history[first:],
            online_cpu_lp_calls=sum(h.get('cpu_lp_calls',0) for h in service.history[first:]),
            online_cpu_solver_runs=cpu_solver_run_count(service.history[first:]),
            model_bridge_cpu_lp_stage_calls=sum(m.cpu_lp_stage_calls for e,m in gpu_env),
            gpu_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes())
        if benchmark_driver_kind('gpu',pipeline_cpu_stages=args.pipeline_cpu_stages,
                                 gpu_first_batch=args.gpu_first_batch) == 'pipeline':
            row['gpu_pipeline_history']=getattr(service,'pipeline_history',[])[first_pipeline:]
            if failure and getattr(service,'pipeline_failed_requests',None):
                try:
                    row['pipeline_cpu_failure_artifacts']=save_failed_cpu_lps(
                        args.output,service.pipeline_failed_requests,service.pipeline_failed_diagnostics)
                    row['diagnostic_inputs_warning']='Failed evaluation LPs for diagnosis only; excluded from training'
                except Exception as capture_error:
                    row['pipeline_cpu_failure_artifact_error']=str(capture_error)
        save()
        return gpu_rows,progress,failure

    for scheduled in schedule:
        repeat,seeds=scheduled['repeat'],scheduled['seeds']
        actions=matched_actions(seeds,args.steps)
        # Fresh trajectories may reuse immutable operators/compiled kernels,
        # never GPU solutions from an earlier benchmark repetition.
        if hasattr(service,'temporal_cache'):service.temporal_cache.clear()
        if hasattr(service,'reset_trajectory'):service.reset_trajectory()
        row=dict(scheduled,cold_first_use_included=repeat==0,
            cpu_driver_kind=benchmark_driver_kind('cpu',
                pipeline_cpu_stages=args.pipeline_cpu_stages,
                gpu_first_batch=args.gpu_first_batch),
            gpu_driver_kind=benchmark_driver_kind('gpu',
                pipeline_cpu_stages=args.pipeline_cpu_stages,
                gpu_first_batch=args.gpu_first_batch));report['runs'].append(row)
        started=time.perf_counter();cpu_env=[make(s) for s in seeds];gpu_env=[make(s) for s in seeds]
        row['environment_construction_seconds']=time.perf_counter()-started
        print(f'repeat {repeat}: environments ready; {row["execution_order"]}; seeds {seeds[0]}..{seeds[-1]}',flush=True)
        try:
            cpu_rows,gpu_run=run_ordered_pair(row['execution_order'],
                lambda:run_cpu(cpu_env,actions,row),lambda:run_gpu(gpu_env,actions,row))
            gpu_rows,progress,failure=gpu_run
            row['compact_graph_compilation_seconds']={k:b.graph_compilation_seconds for k,b in banks.items()}
            row['compact_graph_pool_bytes']={k:sum(g.pool.total_bytes() for g in b.graph_cache.values()) for k,b in banks.items()}
            row['cohort_graph_pool_bytes']={k:sum(g.pool.total_bytes() for g in getattr(b,'cohort_graph_cache',{}).values()) for k,b in banks.items()}
            row['cohort_graph_compilation_seconds']={k:getattr(b,'cohort_graph_compilation_seconds',0.) for k,b in banks.items()}
            heterogeneous=getattr(service,'heterogeneous_banks',{})
            row['heterogeneous_graph_compilation_seconds']={k[0]:getattr(b,'graph_compilation_seconds',0.) for k,b in heterogeneous.items()}
            row['heterogeneous_graph_pool_bytes']={k[0]:sum(g.pool.total_bytes() for g in getattr(b,'graph_cache',{}).values()) for k,b in heterogeneous.items()}
            row['heterogeneous_setup_weight_bytes']={k[0]:b.setup_weight_bytes for k,b in heterogeneous.items()}
            operator_service=getattr(service,'operators',service)
            row['host_operator_preparation_seconds']=getattr(operator_service,'host_operator_preparation_seconds',0.)
            row['host_operator_cache_hits']=getattr(operator_service,'host_operator_cache_hits',0)
            row['cpu_step_timing_totals']=[dict(e.simulator.step_timing_totals) for e,m in cpu_env]
            row['gpu_step_timing_totals']=[dict(e.simulator.step_timing_totals) for e,m in gpu_env]
            if failure is not None and hasattr(service,'last_problems') and service.history and service.history[-1]['stage'] in banks:
                stage=service.history[-1]['stage'];bank=banks[stage]
                engine=next((b for k,b in heterogeneous.items() if k[0]==stage),None)
                if engine is not None:
                    row['rejected_candidates_format']='environment_by_candidate_slot'
                    row['rejected_candidates']={k:v.get().tolist() for k,v in engine.last_candidates.items()}
                else:
                    row['rejected_candidates']=[dict(index=i,rows=ids.tolist(),**{k:v.get().tolist() for k,v in d.items()})
                        for i,ids,d in bank.last_candidates]
                payload={k:v.get() for k,v in bank.prepare_host(service.last_problems).items()}
                np.savez_compressed(args.output.with_suffix('.failure.npz'),**payload)
                row['diagnostic_inputs_warning']='Evaluation inputs for diagnosis only; excluded from training'
            if failure is None:
                errors=[dict(pha_relative=abs(a['pha']-b['pha'])/max(abs(a['pha']),1e-9),
                    biomass_g_l=max(abs(a['biomass'][k]-b['biomass'][k]) for k in a['biomass']),
                    phv_fraction=abs(a['phv_fraction']-b['phv_fraction'])) for a,b in zip(cpu_rows,gpu_rows)]
                row['errors']=errors;row['all_endpoint_gates_passed']=all(v<=.01 for e in errors for v in e.values())
            if router_specs:
                cold = bool(row['cold_first_use_included'])
                cpu_setup = sum(v['cpu_binding_and_numpy_setup_seconds'] for v in report['coverage_router_bindings'].values()) if cold else 0.
                gpu_setup = sum(v['gpu_binding_and_upload_setup_seconds'] for v in report['coverage_router_bindings'].values()) if cold else 0.
                row.update(cpu_coverage_router_setup_seconds=cpu_setup,
                    gpu_coverage_router_setup_seconds=gpu_setup,
                    cpu_seconds_including_coverage_router_setup=row['cpu_seconds']+cpu_setup,
                    gpu_seconds_including_coverage_router_setup=row['gpu_seconds']+gpu_setup)
            diagnostic_only = diagnostic_only_performance(
                candidate_oracle=args.candidate_oracle,
                gpu_first_batch=args.gpu_first_batch,
                gpu_cpu_fallback=args.gpu_cpu_fallback)
            record_performance_comparison(row, progress==[args.steps]*args.environments,
                diagnostic_only=diagnostic_only)
            if args.gpu_first_batch and args.gpu_cpu_fallback == 'reject':
                row['performance_comparison_note']=(
                    'GPU-first reject-on-uncertified work is diagnostic only; '
                    'CPU/GPU speed ratio intentionally omitted.')
            if diagnostic_only:
                print(f'GPU diagnostic {row["gpu_seconds"]:.3f} s; completed {progress}; failure {failure}; no speed ratio',flush=True)
            else:
                print(f'GPU {row["gpu_seconds"]:.3f} s; completed {progress}; failure {failure}; ratio {row.get("cpu_over_gpu_ratio")}',flush=True)
            save()
        finally:
            # Both trajectories contain full GEM copies. Release references
            # before constructing the next pair; collection is never timed.
            del cpu_env,gpu_env
            gc.collect()
        if failure is not None:break
    report['status']='completed' if len(report['runs'])==args.repeats and all(r.get('all_endpoint_gates_passed') for r in report['runs']) else 'incomplete_or_accuracy_failed'
    save()
    if hasattr(service,'close'):service.close()


if __name__=='__main__':main()
