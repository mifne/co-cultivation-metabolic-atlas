"""Offline trajectory-level routing comparison and all-training final refit."""
import argparse,json,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compact_training_data import checked_npz,checked_child,sha256_file
from src.coverage_router import fit_coverage_router,coverage_metrics
from src.coverage_router_binding import (compact_router_provenance,compact_candidate_ids,
    compact_feature_schema,ORIGINAL_ROUTER_STAGES)
from src.compact_bank_coverage import CERTIFICATE_THRESHOLDS


def selected_stage_manifest(manifest,stage):
    if stage not in ORIGINAL_ROUTER_STAGES:
        raise ValueError('Router stage must be maxmin, aggregate or exchange')
    matches=[row for row in manifest['stages'] if row.get('stage')==stage]
    if len(matches)!=1:raise ValueError('Exactly one selected router stage is required')
    return matches[0]


def validate_labels(cov,features,manifest,bank_sha,*,stage='maxmin',bank=None):
    """Bind label columns and feature rows before any optimizer sees them."""
    stage_manifest=selected_stage_manifest(manifest,stage)
    count=len(stage_manifest['entries'])
    metadata=json.loads(str(cov['metadata']))
    # Legacy labels predate explicit stage provenance and were maxmin-only.
    # New downstream rebuilds must never masquerade as maxmin labels merely
    # because their candidate counts happen to match.
    if (metadata.get('stage','maxmin')!=stage
            or manifest.get('reconstructed_stage','maxmin')!=stage):
        raise ValueError(f'This router trainer requires {stage} coverage labels')
    explicit_stage='stage' in metadata or 'reconstructed_stage' in manifest
    if (explicit_stage and (metadata.get('stage_key')!=stage_manifest.get('key')
                            or 'key' not in stage_manifest)):
        raise ValueError('Coverage stage key mismatch')
    if 'reconstruction_provenance' in manifest:
        if any(metadata.get(key)!=value for key,value in manifest['reconstruction_provenance'].items()):
            raise ValueError('Coverage reconstruction provenance mismatch')
    if (cov['complete'].shape!=() or cov['complete'].dtype!=np.bool_ or not bool(cov['complete'])
            or cov['schema_version'].shape!=() or cov['schema_version'].dtype.kind not in 'iu'
            or int(cov['schema_version'])!=1 or json.loads(str(cov['failures']))
            or json.loads(str(cov['certificate_thresholds']))!=CERTIFICATE_THRESHOLDS
            or metadata.get('bank_sha256')!=bank_sha
            or metadata.get('role')!='training_only_dictionary_reconstruction'
            or metadata.get('model_fingerprints')!=manifest['model_fingerprints']
            or cov['coverage'].dtype!=np.bool_ or cov['coverage'].ndim!=2
            or count<1 or cov['coverage'].shape[1]!=count or cov['candidate_indices'].dtype.kind not in 'iu'
            or not np.array_equal(cov['candidate_indices'],np.arange(count))):
        raise ValueError('Coverage bank, candidate order, scope or certificate contract mismatch')
    # The manifest SHA plus positional candidate indices fixes the complete
    # ordered candidate IDs even for old coverage NPZs without an ID field.
    if explicit_stage or bank is not None:
        candidate_ids=list(compact_candidate_ids(stage_manifest))
        if 'candidate_ids' in metadata and metadata['candidate_ids']!=candidate_ids:
            raise ValueError('Coverage candidate identities mismatch')
    seeds=metadata['training_seeds'];steps=metadata['steps']
    if (not isinstance(seeds,list) or not seeds or len(set(seeds))!=len(seeds)
            or any(isinstance(value,bool) or not isinstance(value,int) for value in seeds)
            or isinstance(steps,bool) or not isinstance(steps,int) or steps<1):
        raise ValueError('Training trajectory/step contract is invalid')
    if (features['features'].ndim!=2 or features['features'].dtype.kind!='f'
            or not np.isfinite(features['features']).all()
            or features['trajectory_ids'].dtype.kind not in 'iu' or features['steps'].dtype.kind not in 'iu'
            or not np.array_equal(features['trajectory_ids'],np.repeat(seeds,steps))
            or not np.array_equal(features['steps'],np.tile(np.arange(1,steps+1),len(seeds)))
            or features['features'].shape[0]!=len(seeds)*steps
            or cov['coverage'].shape[0]!=len(seeds)*steps):
        raise ValueError('Feature/coverage trajectory row identity mismatch')
    if bank is not None:
        schema,digest=compact_feature_schema(bank)
        key=stage_manifest.get('key')
        if (not isinstance(key,(list,tuple)) or len(key)!=4 or key[0]!=stage
                or list(key[1:3])!=list(bank.host_a.shape)
                or not isinstance(key[3],int) or isinstance(key[3],bool) or not 0<=key[3]<=key[1]
                or getattr(bank,'neq',key[3])!=key[3]):
            raise ValueError('Coverage stage key differs from the actual root layout')
        if (features['features'].shape[1]!=schema['selected_feature_width']
                or 'feature_schema_sha256' in metadata and metadata['feature_schema_sha256']!=digest):
            raise ValueError('Coverage feature schema mismatch')
    if 'routing_order' in cov:
        ranking=cov['routing_order']
        # Coverage labels span the WHOLE bank, but the measurement writer
        # intentionally saves only the declared nearest top-K prefix.
        if (ranking.ndim!=2 or ranking.shape[0]!=cov['coverage'].shape[0]
                or not 1<=ranking.shape[1]<=count or ranking.dtype.kind not in 'iu'
                or np.any(ranking<0) or np.any(ranking>=count)
                or any(len(np.unique(row))!=len(row) for row in ranking)):
            raise ValueError('Coverage routing order must contain unique valid candidates')
        topk=cov.get('routing_topk')
        if topk is None:
            if ranking.shape[1]!=count:
                raise ValueError('A truncated routing order requires explicit routing_topk')
        elif (topk.ndim!=1 or topk.dtype.kind not in 'iu' or not len(topk)
                or len(np.unique(topk))!=len(topk) or np.any(topk<1)
                or int(topk.max())!=ranking.shape[1]):
            raise ValueError('Coverage routing_topk disagrees with the saved ranking prefix')


def load_stage_training_artifacts(directory,*,stage='maxmin'):
    """Hash-check the chosen label/feature files and every bound bank file."""
    directory=Path(directory)
    manifest=json.loads(checked_child(directory,'manifest.json').read_text())
    rebuild=json.loads(checked_child(directory,'rebuild_report.json').read_text())
    if manifest['status']!='completed' or rebuild['status']!='completed':raise ValueError('Incomplete bank')
    if rebuild.get('stage','maxmin')!=stage:raise ValueError('Rebuild report stage mismatch')
    stage_manifest=selected_stage_manifest(manifest,stage)
    cov=checked_npz(directory,'coverage.npz',rebuild['coverage_sha256'])
    data=checked_npz(directory,'features.npz',rebuild['features_sha256'])
    folder=directory/stage
    root=checked_npz(folder,'root.npz',stage_manifest['root_sha256'])
    nearest=checked_npz(folder,'router.npz',stage_manifest['router_sha256'])
    for entry in stage_manifest['entries']:checked_child(folder,entry['filename'],entry['sha256'])
    bank=SimpleNamespace(host_a=csr_matrix(tuple(root['a_shape'])),neq=int(root['neq']),
        variable_rows=root['variable_rows'],feature_indices=nearest['indices'])
    digest=sha256_file(directory/'manifest.json')
    validate_labels(cov,data,manifest,digest,stage=stage,bank=bank)
    coverage_metadata=json.loads(str(cov['metadata']))
    if not isinstance(rebuild.get('provenance'),dict) or any(coverage_metadata.get(key)!=value
                                        for key,value in rebuild['provenance'].items()):
        raise ValueError('Rebuild report provenance disagrees with coverage labels')
    provenance=compact_router_provenance(digest,stage_manifest,bank,
        rebuild['coverage_sha256'],manifest['model_fingerprints'])
    provenance.update(features_sha256=rebuild['features_sha256'],
        rebuild_report_sha256=sha256_file(directory/'rebuild_report.json'))
    return manifest,rebuild,cov,data,provenance


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--stage',choices=ORIGINAL_ROUTER_STAGES,default='maxmin')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=800)
    p.add_argument('--hidden-dims',nargs='+',type=int,default=[0,32]);p.add_argument('--seed',type=int,default=20296000)
    args=p.parse_args();start=time.perf_counter()
    manifest,rebuild,cov,data,provenance=load_stage_training_artifacts(args.bank,stage=args.stage)
    args.output.mkdir(parents=True,exist_ok=False)
    source_dir=args.output/'sources';source_dir.mkdir();sources={}
    for f in [Path(__file__),ROOT/'src/coverage_router.py',ROOT/'src/coverage_router_binding.py',
              ROOT/'src/compact_training_data.py',ROOT/'src/selected_lp_features.py']:
        shutil.copyfile(f,source_dir/f.name);sources[f.name]=sha256_file(f)
    x,y,ids=data['features'],cov['coverage'],data['trajectory_ids']
    seeds=list(dict.fromkeys(ids.tolist()));results=[]
    for hidden in args.hidden_dims:
        folds=[]
        for fold,seed in enumerate(seeds):
            train=np.flatnonzero(ids!=seed);valid=np.flatnonzero(ids==seed)
            router,report=fit_coverage_router(x,y,ids,provenance=provenance,train_indices=train,
                validation_indices=valid,hidden_dim=hidden,epochs=args.epochs,seed=args.seed+fold,device='cpu',top_k=(1,2,4))
            artifact=router.save(args.output/f'internal_h{hidden}_fold{fold}.npz')
            report['artifact']=artifact
            (args.output/f'internal_h{hidden}_fold{fold}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            folds.append(dict(validation_seed=seed,artifact=artifact,
                report_sha256=sha256_file(args.output/f'internal_h{hidden}_fold{fold}.json'),metrics=report['validation_metrics'],
                nearest_metrics=coverage_metrics(y[valid],cov['routing_order'][valid],top_k=(1,2,4))))
        score=sum(f['metrics']['capture_at_k']['4']['captured_rows'] for f in folds)/len(x)
        results.append(dict(hidden_dim=hidden,internal_held_trajectory_k4=score,folds=folds))
        print(f'hidden {hidden}: internal 4-fold K4={score:.4f}',flush=True)
    # Fixed, transparent architecture selection; smaller model breaks a tie.
    selected=max(results,key=lambda r:(r['internal_held_trajectory_k4'],-r['hidden_dim']))
    router,fit=fit_coverage_router(x,y,ids,provenance=provenance,hidden_dim=selected['hidden_dim'],
        epochs=args.epochs,seed=args.seed,device='cpu',top_k=(1,2,4),refit_all=True)
    artifact=router.save(args.output/'router.npz')
    (args.output/'final_fit.json').write_text(json.dumps(fit,indent=2,allow_nan=False))
    report=dict(status='completed',stage=args.stage,provenance=provenance,sources=sources,internal_comparison=results,
        selected_hidden_dim=selected['hidden_dim'],artifact=artifact,
        final_training_metrics=fit['training_metrics'],
        warning='All bank selection and final router fitting use four training trajectories. Internal folds are NOT bank-independent holdout. New-seed closed-loop validation is still required.',
        total_seconds=time.perf_counter()-start)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['final_training_metrics'],indent=2),flush=True)


if __name__=='__main__':main()
