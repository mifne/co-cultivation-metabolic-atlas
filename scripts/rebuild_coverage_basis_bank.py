"""Rebuild one selected LP stage from training data and strict GPU coverage.

Other stages remain byte-identical. The changed whole-bank manifest invalidates
external SHA-bound learned routers; this script never rebinds those routers.
"""
import argparse,copy,gc,json,shutil,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compact_training_data import (load_compact_training_source,checked_child,checked_npz,
    sha256_file,COMPACT_TRAINING_STAGES,validate_compact_stage)
from src.compact_bank_coverage import (measure_compact_bank_coverage,save_compact_coverage_npz,
    summarize_routing_coverage,CERTIFICATE_THRESHOLDS)
from src.basis_coverage_selection import greedy_set_cover
from src.gpu_compact_basis import CompactBank
from src.compact_basis_rebuild import collect_basis_candidates,project_basis_candidate,BasisRebuildError
from scripts.measure_training_bank_coverage import grouped_summary


def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def validate_training_baseline(source,baseline,raw):
    """Require complete original-LP certificates bound to this stage/source.

    Unstaged legacy provenance is accepted only for maxmin, the only stage the
    old loader supported. All historical hashes/rows and thresholds must match.
    """
    def check_provenance(value):
        if not isinstance(value,dict):raise ValueError('Baseline provenance is missing')
        legacy=source.stage=='maxmin' and 'stage' not in value and 'stage_key' not in value
        expected={k:v for k,v in source.provenance.items()
                  if not (legacy and k in ('stage','stage_key'))}
        if any(value.get(k)!=v for k,v in expected.items()):
            raise ValueError('Training source or selected stage changed since baseline')
    check_provenance(baseline.get('provenance'))
    check_provenance(json.loads(str(raw['metadata'])))
    if (baseline.get('status')!='completed' or raw['complete'].shape!=()
            or raw['complete'].dtype!=np.bool_ or not bool(raw['complete'])
            or raw['schema_version'].shape!=() or raw['schema_version'].dtype.kind not in 'iu'
            or int(raw['schema_version'])!=1 or json.loads(str(raw['failures']))
            or json.loads(str(raw['certificate_thresholds']))!=CERTIFICATE_THRESHOLDS
            or raw['coverage'].dtype!=np.bool_):
        raise ValueError('Baseline must contain complete original-LP certificates')
    old_count=len(source.stage_manifest['entries'])
    if (raw['coverage'].shape!=(source.size,old_count) or raw['candidate_indices'].dtype.kind not in 'iu'
            or not np.array_equal(raw['candidate_indices'],np.arange(old_count))):
        raise ValueError('Baseline candidate order mismatch')


def copy_and_extend_selected_stage(manifest,base_directory,output_directory,*,selected_stage,
                                   pool_directory,pool,selection,centers,source):
    """Copy every inherited stage exactly, extending only the selected one."""
    selected_stage=validate_compact_stage(selected_stage)
    stages=manifest['stages'];names=[row['stage'] for row in stages]
    if (source.stage!=selected_stage or names.count(selected_stage)!=1
            or len(set(names))!=len(names)):
        raise ValueError('One unambiguous selected stage is required')
    # Preserve selected old candidate order; all old entries are mandatory.
    old_count=len(source.stage_manifest['entries'])
    if tuple(selection.selected[:old_count])!=tuple(range(old_count)):
        raise ValueError('Mandatory inherited candidate order changed')
    base_directory=Path(base_directory).resolve();output_directory=Path(output_directory).resolve()
    for row in stages:
        folder=(output_directory/row['stage']).resolve()
        origin=(base_directory/row['stage']).resolve()
        if (not folder.is_relative_to(output_directory) or folder==output_directory
                or not origin.is_relative_to(base_directory) or origin==base_directory):
            raise ValueError('Unsafe inherited stage path')
        folder.mkdir(exist_ok=False)
        files=[('root.npz',row['root_sha256']),('router.npz',row['router_sha256'])]
        files.extend((entry['filename'],entry['sha256']) for entry in row['entries'])
        for name,digest in files:
            target=(folder/name).resolve()
            if not target.is_relative_to(folder) or target.exists():
                raise ValueError('Unsafe or duplicated inherited artifact path')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(checked_child(origin,name,digest),target)
            if sha256_file(target)!=digest:raise ValueError('Inherited artifact copy mismatch')
        if row['stage']!=selected_stage:continue
        for pool_index in selection.selected[old_count:]:
            record=pool[pool_index-old_count];name=f'basis_{len(row["entries"]):04d}.npz'
            if (folder/name).exists():raise FileExistsError(folder/name)
            shutil.copyfile(checked_child(pool_directory,record['pool_filename'],record['sha256']),folder/name)
            if sha256_file(folder/name)!=record['sha256']:raise ValueError('New artifact copy mismatch')
            row['entries'].append(dict(filename=name,sha256=record['sha256'],source='coverage_training_rebuild',
                signature=record['candidate_signature'],source_indices=record['source_indices']))
        # This nearest router changes only in the new, selected-stage folder.
        np.savez(folder/'router.npz',centers=np.stack(centers)[list(selection.selected)],
            indices=source.feature_indices,scale=source.feature_scale)
        row['router_sha256']=sha256_file(folder/'router.npz')
        row['projected_bytes']=selection.selected_bytes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'results/pf_neural_basis_actual4x120_20260904')
    p.add_argument('--bank',type=Path,default=ROOT/'results/pf_compact120_v3_20260904')
    p.add_argument('--stage',choices=COMPACT_TRAINING_STAGES,default='maxmin')
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-total-candidates',type=int,default=96)
    p.add_argument('--max-new-pool',type=int,default=144)
    p.add_argument('--max-projected-mib',type=int,default=300)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--cpu-workers',type=int,default=4)
    args=p.parse_args()
    if min(args.max_total_candidates,args.max_new_pool,args.max_projected_mib,args.batch_size,args.cpu_workers)<1:
        raise ValueError('Positive budgets required')
    baseline=json.loads((args.baseline/'report.json').read_text())
    source=load_compact_training_source(args.data,args.bank,stage=args.stage,
        forbidden_seeds=baseline['provenance']['forbidden_seeds'])
    if source.size!=480 or source.steps!=120 or len(source.seeds)!=4:
        raise ValueError('This rebuild requires the declared 4 x 120 training data')
    from scripts.benchmark_basis_bank_rollout import environment
    from src.fba_surrogate import model_fingerprint
    sample,_=environment(source.seeds[0])
    if {name:model_fingerprint(model) for name,model in sample.simulator.models.items()}!=source.bank_manifest['model_fingerprints']:
        raise ValueError('Current GEMs changed')
    raw=checked_npz(args.baseline,'coverage.npz',baseline['coverage_sha256'])
    validate_training_baseline(source,baseline,raw)
    old_count=len(source.stage_manifest['entries'])
    if args.max_total_candidates<old_count:raise ValueError('Old candidates are mandatory')
    args.output.mkdir(parents=True,exist_ok=False)
    pool_dir=args.output/'pool';pool_dir.mkdir()
    sources=args.output/'sources';sources.mkdir()
    source_hashes={}
    for file in [Path(__file__),ROOT/'src/compact_basis_rebuild.py',ROOT/'src/compact_training_data.py',
        ROOT/'src/compact_bank_coverage.py',ROOT/'src/basis_coverage_selection.py',ROOT/'src/gpu_compact_basis.py',
        ROOT/'src/cpu_repeated_lp.py',ROOT/'src/lp_bounds.py',
        ROOT/'src/selected_lp_features.py',ROOT/'scripts/measure_training_bank_coverage.py']:
        shutil.copyfile(file,sources/file.name);source_hashes[file.name]=sha256_file(file)
    start=time.perf_counter()
    report=dict(status='collecting',stage=source.stage,provenance=source.provenance,sources=source_hashes,
        baseline_sha256=sha256_file(args.baseline/'report.json'),
        budgets=dict(max_total_candidates=args.max_total_candidates,max_new_pool=args.max_new_pool,
                     max_projected_bytes=args.max_projected_mib*1024**2),projected_candidates=[],rejected_candidates=[])
    save_json(args.output/'rebuild_report.json',report)
    try:
        candidates,collector=collect_basis_candidates(source,workers=args.cpu_workers,
            progress=lambda row:print('collect '+json.dumps(row),flush=True) if row['step']==1 or row['step']%10==0 else None)
    except BasisRebuildError as error:
        save_json(args.output/'collector_report.json',error.report)
        report.update(status='failed',failure='CPU collection did not complete')
        save_json(args.output/'rebuild_report.json',report)
        raise
    save_json(args.output/'collector_report.json',collector)
    np.savez_compressed(args.output/'new_basis_statuses.npz',
        col_statuses=np.stack([c['anchor']['col_status'] for c in candidates]),
        row_statuses=np.stack([c['anchor']['row_status'] for c in candidates]),
        signatures=np.array([c['signature'] for c in candidates]),
        first_source_indices=np.array([c['source_index'] for c in candidates]),
        role=np.array('Offline training basis inventory only; never online solution labels'))
    report['new_basis_statuses_sha256']=sha256_file(args.output/'new_basis_statuses.npz')
    report.update(status='projecting',collector_report_sha256=sha256_file(args.output/'collector_report.json'))
    feature=source.selected_features()
    old_router=checked_npz(args.bank/source.stage,'router.npz',source.stage_manifest['router_sha256'])
    centers=list(old_router['centers']);coverage=[raw['coverage']]
    old_arrays=[checked_npz(args.bank/source.stage,e['filename'],e['sha256']) for e in source.stage_manifest['entries']]
    sizes=[sum(v.nbytes for v in d.values()) for d in old_arrays]
    del old_arrays
    union=raw['coverage'].any(axis=1);remaining=list(range(len(candidates)));pool=[]
    while remaining and not union.all() and len(report['projected_candidates'])<args.max_new_pool:
        pick=max(remaining,key=lambda j:(int((~union[np.asarray(candidates[j]['source_indices'],dtype=int)]).sum()),-j))
        remaining.remove(pick);candidate=candidates[pick]
        if not (~union[np.asarray(candidate['source_indices'],dtype=int)]).any():continue
        try:
            arrays,record=project_basis_candidate(candidate,source)
            center=feature[candidate['source_indices'][0]]
            bank=CompactBank(source.root,source.variable_rows,[arrays],center[None],source.feature_indices,
                source.feature_scale,capture=False,selected_features=True)
            measured=measure_compact_bank_coverage(bank,source.batches(args.batch_size),routing_topk=(1,))
            if not measured.complete:raise ValueError('Incomplete candidate coverage: '+str(measured.candidate_failures))
            mask=measured.coverage[:,0]
            self_rows=np.asarray(candidate['source_indices'],dtype=int)
            if not mask[self_rows].all():
                raise ValueError('Candidate fails at least one of its own certified training rows')
            gain=int((mask&~union).sum())
            name=f'candidate_{len(pool):04d}.npz'
            record.update(pool_filename=name,pool_index=old_count+len(pool),candidate_signature=candidate['signature'],
                source_indices=list(map(int,candidate['source_indices'])),covered_rows=int(mask.sum()),
                marginal_at_collection=gain,coverage_timing=measured.timing,
                self_row_gpu_certified=True,all_training_coverage_measured=True)
            record['gpu_projection_certified']=True
            np.savez(pool_dir/name,**arrays);record['sha256']=sha256_file(pool_dir/name)
            pool.append(record);sizes.append(sum(v.nbytes for v in arrays.values()))
            centers.append(center);coverage.append(mask[:,None]);union|=mask
            report['projected_candidates'].append(record)
            print(f'projected {len(pool)}; added {gain}; union {int(union.sum())}/{source.size}',flush=True)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            report['rejected_candidates'].append(dict(candidate_signature=candidate['signature'],
                source_indices=list(map(int,candidate['source_indices'])),error_type=type(error).__name__,error=str(error)))
            print(f'rejected candidate {pick}: {error}',flush=True)
        finally:
            bank=None;arrays=None;gc.collect()
        report['pool_union_covered_rows']=int(union.sum())
        save_json(args.output/'rebuild_report.json',report)
    cov=np.concatenate(coverage,axis=1)
    selection=greedy_set_cover(cov,min(args.max_total_candidates,cov.shape[1]),mandatory=range(old_count),
        candidate_bytes=np.array(sizes,dtype=np.int64),max_bytes=args.max_projected_mib*1024**2)
    np.savez_compressed(args.output/'pool_coverage.npz',coverage=cov,centers=np.stack(centers),
        candidate_bytes=np.array(sizes),selected=np.array(selection.selected),
        candidate_ids=np.array([e['filename']+':'+e['sha256'] for e in source.stage_manifest['entries']]
            +[r['pool_filename']+':'+r['sha256'] for r in pool]),
        metadata=np.array(json.dumps(dict(provenance=source.provenance,old_mandatory_count=old_count))))
    report['pool_coverage_sha256']=sha256_file(args.output/'pool_coverage.npz')
    manifest=copy.deepcopy(source.bank_manifest)
    manifest['inherited_total_seconds']=manifest.pop('total_seconds',None)
    manifest.pop('cpu_lp_calls',None)
    manifest.update(status='building',source_data=str(args.data.resolve()),
        inherited_from_bank_manifest_sha256=source.provenance['base_bank_manifest_sha256'],
        reconstruction_provenance=source.provenance,reconstruction_sources=source_hashes,
        reconstructed_stage=source.stage,
        reconstruction_note=f'Only {source.stage} changed; strict training coverage and deterministic greedy set cover; not independent validation',
        external_router_binding_note='The new whole-bank SHA invalidates previously bound external learned routers. Explicit rebind/retraining and validation are required; no binding has been weakened.',
        inherited_cpu_lp_calls=source.bank_manifest.get('cpu_lp_calls'),
        reconstruction_cpu_lp_calls=collector['offline_cpu_lp_calls'],
        reconstruction_cpu_solver_runs=collector['offline_cpu_solver_runs'])
    copy_and_extend_selected_stage(manifest,args.bank,args.output,selected_stage=source.stage,
        pool_directory=pool_dir,pool=pool,selection=selection,centers=centers,source=source)
    manifest.update(reconstruction_seconds=time.perf_counter()-start)
    save_json(args.output/'manifest.json',manifest)
    final_stage=next(stage for stage in manifest['stages'] if stage['stage']==source.stage)
    final_arrays=[checked_npz(args.output/source.stage,e['filename'],e['sha256']) for e in final_stage['entries']]
    final_bank=CompactBank(source.root,source.variable_rows,final_arrays,np.stack(centers)[list(selection.selected)],
        source.feature_indices,source.feature_scale,capture=False,selected_features=True)
    final=measure_compact_bank_coverage(final_bank,source.batches(args.batch_size),routing_topk=(1,2,4))
    if not final.complete or not np.array_equal(final.coverage,cov[:,selection.selected]):
        raise ValueError('Saved final bank did not reproduce selected strict coverage')
    manifest.update(status='completed',reconstruction_seconds=time.perf_counter()-start,
        self_row_and_all_training_coverage_reverified=True,cpu_lp_calls=collector['offline_cpu_lp_calls'])
    manifest['total_seconds']=manifest['reconstruction_seconds']
    save_json(args.output/'manifest.json',manifest)
    save_compact_coverage_npz(args.output/'coverage.npz',final,dict(**source.provenance,
        bank_sha256=sha256_file(args.output/'manifest.json'),selection=asdict(selection)))
    np.savez_compressed(args.output/'features.npz',features=feature,
        trajectory_ids=np.repeat(source.seeds,source.steps),steps=np.tile(np.arange(1,source.steps+1),len(source.seeds)))
    report.update(status='completed',selection=asdict(selection),final_summary=summarize_routing_coverage(final),
        by_trajectory_period=grouped_summary(final,source),coverage_sha256=sha256_file(args.output/'coverage.npz'),
        features_sha256=sha256_file(args.output/'features.npz'),total_seconds=time.perf_counter()-start)
    save_json(args.output/'rebuild_report.json',report)
    print(json.dumps(report['final_summary'],indent=2),flush=True)


if __name__=='__main__':main()
