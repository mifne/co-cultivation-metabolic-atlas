"""Manifest-indexed, bounded-memory access to explicitly split LP teachers."""
import hashlib
import json
from importlib import metadata
from pathlib import Path
import platform
import numpy as np
from .lp_trace import load_trace_lp


TRACE_SOURCE_FILES = (
    'scripts/capture_dfba_lp_trace.py', 'scripts/collect_graph_learning_curve.py',
    'src/graph_training_collection.py', 'src/lp_trace.py', 'src/cpu_repeated_lp.py',
    'src/community_solver.py', 'src/dfba_simulator.py', 'src/rl_environment.py',
    'src/utils.py', 'src/one_l_jar.py', 'src/gene_id_registry.py',
    'src/fba_surrogate.py', 'scripts/benchmark_cooperative_surrogate_e2e.py',
    'scripts/benchmark_basis_bank_rollout.py', 'scripts/benchmark_compact_gpu.py',
    'scripts/microbatch_comparison_support.py', 'src/frozen_community_inputs.py',
    'src/trajectory_design.py', 'src/offline_scipy_teacher.py',
    'src/metabolite_ids.py', 'src/lp_bounds.py', 'src/gpu_compiled_community_backend.py',
    'config/gene_id_registry.json',
)
TEACHER_MODEL_FILES = (
    'models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml',
    'models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml',
    'models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml',
)


def teacher_source_contract(root, *, frozen_inputs, design_profile, teacher_strategy):
    """Immutable input-generator contract; these teachers still use legacy dynamics.

    Seeds, offsets and worker counts may differ between shards. Biology,
    controllers, solver strategy and library versions may not. Source hashes
    include the actual environment factory and reset/action code.
    """
    root = Path(root)
    versions = {}
    for package in ('numpy', 'scipy', 'highspy', 'cobra', 'optlang', 'gymnasium',
                    'greenlet', 'python-libsbml'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    payload = dict(
        schema='teacher_source_environment_v1',
        source_hashes={f: hashlib.sha256((root/f).read_bytes()).hexdigest()
                       for f in TRACE_SOURCE_FILES},
        model_source_hashes={f: hashlib.sha256((root/f).read_bytes()).hexdigest()
                             for f in TEACHER_MODEL_FILES},
        environment=dict(factory='scripts.benchmark_basis_bank_rollout.environment',
            simulator='src.dfba_simulator.dFBASimulator', dynamics='legacy',
            fba_mode='cooperative', consortium_profile='pf-helper3', controller_dt_h=.2,
            initial_nh4_mmol_l=.05, frozen_inputs=bool(frozen_inputs),
            design_profile=design_profile, audited_dynamics_qualified=False),
        teacher_strategy=teacher_strategy,
        runtime=dict(python=platform.python_version(), packages=versions),
    )
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return payload


def validate_teacher_contract(contract):
    if not isinstance(contract, dict) or contract.get('schema') != 'teacher_source_environment_v1':
        raise ValueError('Missing complete teacher source/environment provenance; legacy data remain read-only')
    body = {k: v for k, v in contract.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != contract.get('sha256'):
        raise ValueError('Teacher source/environment contract digest changed')
    if set(contract.get('source_hashes', {})) != set(TRACE_SOURCE_FILES) or set(
            contract.get('model_source_hashes', {})) != set(TEACHER_MODEL_FILES):
        raise ValueError('Incomplete teacher source/model provenance')
    for key in ('environment', 'runtime', 'teacher_strategy'):
        if not contract.get(key):
            raise ValueError(f'Missing teacher contract {key}')


def require_matching_teacher_contract(expected, actual):
    validate_teacher_contract(expected)
    validate_teacher_contract(actual)
    if expected != actual:
        raise ValueError('Teacher source/environment contract differs; use a new collection, do not append or replace old shards')


class GraphTrainingCollection:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.raw=(self.path/'catalog.json').read_bytes()
        self.catalog=json.loads(self.raw)
        if self.catalog['configuration']['schema']!='graph_trajectory_collection_v1':raise ValueError('Unknown collection schema')
        self.sha256=hashlib.sha256(self.raw).hexdigest()
        self.teacher_contract = self.catalog.get('teacher_contract')
        self.provenance_status = ('verified_source_environment_contract' if self.teacher_contract
                                  else 'legacy_incomplete_read_only')
        if self.teacher_contract:
            validate_teacher_contract(self.teacher_contract)
        legacy_source_hashes = None
        self.episodes={'train':[],'selection':[],'test':[]};seen=set();hash_splits={};self.incomplete=[]
        expected_roles=dict(train='training_reference',selection='model_selection_reference',test='development_diagnostic_not_training')
        for shard in sorted(self.catalog['shards'],key=lambda x:(x['split'],x['start'])):
            if shard['status']!='completed':self.incomplete.append(shard);continue
            folder=(self.path/shard['path']).resolve()
            if not folder.is_relative_to(self.path):raise ValueError('Shard path escaped collection')
            raw=(folder/'manifest.json').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=shard['manifest_sha256']:raise ValueError('Shard manifest changed')
            m=json.loads(raw);split=shard['split']
            if self.teacher_contract:
                require_matching_teacher_contract(self.teacher_contract, m.get('teacher_contract'))
                if m.get('source_hashes') != self.teacher_contract['source_hashes']:
                    raise ValueError('Shard source hashes differ from teacher contract')
            elif m.get('teacher_contract'):
                raise ValueError('Cannot mix contracted teachers into a legacy collection')
            elif m.get('source_hashes'):
                if legacy_source_hashes is not None and legacy_source_hashes != m['source_hashes']:
                    raise ValueError('Mixed legacy teacher source hashes')
                legacy_source_hashes = m['source_hashes']
            if m['status']!='completed' or m['role']!=expected_roles[split]:raise ValueError('Invalid shard split/role')
            if m['model_fingerprints']!=self.catalog['model_fingerprints']:raise ValueError('Mixed GEM models')
            if m['completed_steps']!=[self.catalog['configuration']['steps']]*shard['count']:raise ValueError('Incomplete trajectories')
            expected=list(range(self.catalog['configuration']['seed_bases'][split]+shard['start'],
                self.catalog['configuration']['seed_bases'][split]+shard['start']+shard['count']))
            if m['seeds']!=expected or shard['seeds']!=expected:raise ValueError('Shard seed/index mapping differs')
            for lane,seed in enumerate(m['seeds']):
                if seed in seen:raise ValueError('Duplicate seed, including across splits')
                seen.add(seed)
                entries={stage:sorted([e for e in m['entries'] if e['environment_id']==lane and e['stage']==stage],key=lambda e:e['step'])
                    for stage in ('maxmin','aggregate','exchange')}
                for stage,rows in entries.items():
                    if [e['step'] for e in rows]!=list(range(1,m['completed_steps'][lane]+1)):
                        raise ValueError('Missing/duplicated trajectory step')
                    for e in rows:
                        h=e['problem_sha256']
                        if h in hash_splits and hash_splits[h]!=split:raise ValueError('LP input overlap across splits')
                        hash_splits[h]=split
                self.episodes[split].append(dict(seed=seed,folder=folder,entries=entries))

    def select(self,split,count):
        if split not in ('train','selection'):
            raise ValueError('Final test labels cannot be used for fitting/model selection')
        if type(count) is not int or not 1<=count<=len(self.episodes[split]):
            raise ValueError(f'Requested {count} complete {split} trajectories, available {len(self.episodes[split])}')
        if any(s['split']==split and s['start']<count for s in self.incomplete):
            raise ValueError('Incomplete/failed conditions cannot be silently excluded')
        seeds=[e['seed'] for e in self.episodes[split][:count]]
        base=self.catalog['configuration']['seed_bases'][split]
        if seeds!=list(range(base,base+count)):raise ValueError('Nested prefix has missing conditions')
        return self.episodes[split][:count]

    def batches(self,split,count,batch_size,*,stage='maxmin',order=None):
        episodes=self.select(split,count)
        if type(batch_size) is not int or batch_size<1:raise ValueError('Positive integer batch size required')
        if stage not in ('maxmin','aggregate','exchange'):raise ValueError('Unknown LP stage')
        if count%batch_size:raise ValueError('Use whole batches without duplicate padding')
        offsets=list(range(0,count,batch_size)) if order is None else list(order)
        if sorted(offsets)!=list(range(0,count,batch_size)):raise ValueError('Every trajectory must be visited once')
        for first in offsets:
            chosen=episodes[first:first+batch_size]
            # One batch only; caller must release it before loading the next.
            yield [[load_trace_lp(e['folder'],entry) for entry in e['entries'][stage]] for e in chosen]

    def statistics(self,count,*,stage='maxmin',batch_size=4,normalization='rms'):
        if normalization not in ('rms','std'):raise ValueError('Unknown normalization')
        total=0;sums=squares=None
        for trajectories in self.batches('train',count,batch_size,stage=stage):
            for trajectory in trajectories:
                for _,x,y in trajectory:
                    values=[x,y]
                    if sums is None:sums=[np.zeros_like(v) for v in values];squares=[np.zeros_like(v) for v in values]
                    for j,v in enumerate(values):sums[j]+=v;squares[j]+=v*v
                    total+=1
        stats=[]
        for s,q in zip(sums,squares):
            mean=s/total
            scale=np.maximum(np.sqrt(q/total),1.) if normalization=='rms' else np.maximum(np.sqrt(np.maximum(q/total-mean*mean,0.)),1e-3)
            stats.extend((mean,scale))
        return stats


def scaling_decision(points,*,target_speedup=2.,min_repeats=3):
    """Conservative decision on measured learning curves, not dataset-size faith.

    Each point must represent a converged fixed-architecture experiment and
    trajectory-bootstrap confidence interval for speedup vs the SAME baseline.
    The function does not manufacture intervals from correlated LP timesteps.
    """
    if not points:return dict(action='collect_and_measure',next_train_count=32,adequate=False)
    rows=sorted(points,key=lambda p:p['train_count']);last=rows[-1];n=last['train_count']
    if len({r['train_count'] for r in rows})!=len(rows):raise ValueError('Unique learning-curve sizes required')
    for r in rows:
        if type(r['train_count']) is not int or r['train_count']<1:raise ValueError('Positive training size required')
    if last.get('training_seeds',0)<min_repeats or not last.get('converged',False):
        return dict(action='finish_training_repeats_or_convergence',next_train_count=n,adequate=False)
    if not last.get('all_accuracy_gates_passed',False):
        return dict(action='expand_coverage_and_diagnose_errors',next_train_count=2*n,adequate=False)
    ci=last.get('speedup_ci95')
    if not ci or len(ci)!=2 or not np.isfinite(ci).all() or ci[0]>ci[1]:
        return dict(action='measure_trajectory_level_confidence_interval',next_train_count=n,adequate=False)
    if n<=512:
        return dict(action='measure_larger_size_before_cap_claim',next_train_count=2*n,adequate=False,
            reason='512 is not a scientific upper bound; include a larger probe')
    # Even a larger data run cannot justify a cap when targets are missed.
    if ci[0]<target_speedup:
        return dict(action='target_not_met_expand_or_change_method',next_train_count=2*n,adequate=False)
    if len(rows)<3 or any(not r.get('converged',False) for r in rows[-3:]):
        return dict(action='measure_more_sizes',next_train_count=2*n,adequate=False)
    recent=rows[-3:]
    if (any(not r.get('all_accuracy_gates_passed',False) or r.get('training_seeds',0)<min_repeats for r in recent)
        or len({r.get('fixed_comparison_id') for r in recent})!=1 or not recent[0].get('fixed_comparison_id')):
        return dict(action='align_architecture_split_and_repeated_comparisons',next_train_count=n,adequate=False)
    # A measured upper CI bound on gain is required for TWO enlargements.
    gains=[r.get('relative_gain_upper_ci95') for r in rows[-2:]]
    if any(g is None or not np.isfinite(g) or g>.02 for g in gains):
        return dict(action='gain_or_uncertainty_remains',next_train_count=2*n,adequate=False)
    candidate=rows[-2]
    candidate_ci=candidate.get('speedup_ci95',[])
    candidate_count=(candidate['train_count'] if len(candidate_ci)==2 and np.isfinite(candidate_ci).all()
        and candidate_ci[0]>=target_speedup else n)
    return dict(action='candidate_plateau_not_universal_maximum',next_train_count=None,adequate=True,
        candidate_train_count=candidate_count,
        validated_through=n,reason='Targets met and two size increments have <=2% plausible gain; confirm untouched test')
