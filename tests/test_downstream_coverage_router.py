"""CPU-only stage-specific label loading, binding and dispatch policy tests.

No fitting, GPU initialization or actual GEM simulation takes place here.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_coverage_router import load_stage_training_artifacts,validate_labels
from src.compact_bank_coverage import CERTIFICATE_THRESHOLDS
from src.compact_training_data import sha256_file
from src.coverage_router_binding import compact_candidate_ids


def make_files(tmp_path,stage,*,legacy=False):
    directory=tmp_path/'bank';folder=directory/stage;folder.mkdir(parents=True)
    np.savez(folder/'root.npz',a_shape=np.array([2,3]),neq=np.array(1),variable_rows=np.array([1]))
    np.savez(folder/'router.npz',indices=np.array([0,3]),scale=np.ones(2),centers=np.zeros((2,2)))
    entries=[]
    for i in range(2):
        file=folder/f'basis_{i:04d}.npz';np.savez(file,unchanged_projection=np.array([i]))
        entries.append(dict(filename=file.name,sha256=sha256_file(file)))
    stage_row=dict(stage=stage,key=[stage,2,3,1],entries=entries,
        root_sha256=sha256_file(folder/'root.npz'),router_sha256=sha256_file(folder/'router.npz'))
    provenance=dict(role='training_only_dictionary_reconstruction',model_fingerprints={'toy':'a'*64},
        training_seeds=[11,12],steps=2,base_bank_manifest_sha256='c'*64,
        training_manifest_sha256='d'*64,input_chunks=[dict(filename=f'inputs/{stage}/000000.npz',sha256='e'*64)])
    if not legacy:provenance.update(stage=stage,stage_key=[stage,2,3,1])
    manifest=dict(status='completed',model_fingerprints={'toy':'a'*64},stages=[stage_row],
        reconstruction_provenance=copy.deepcopy(provenance))
    if not legacy:manifest['reconstructed_stage']=stage
    (directory/'manifest.json').write_text(json.dumps(manifest))
    metadata=dict(provenance,bank_sha256=sha256_file(directory/'manifest.json'))
    cov=dict(complete=np.array(True),schema_version=np.array(1),failures=np.array('[]'),
        certificate_thresholds=np.array(json.dumps(CERTIFICATE_THRESHOLDS)),
        metadata=np.array(json.dumps(metadata)),coverage=np.array([[1,0],[0,1],[1,0],[0,1]],dtype=bool),
        candidate_indices=np.arange(2),routing_order=np.tile([0,1],(4,1)))
    data=dict(trajectory_ids=np.array([11,11,12,12]),steps=np.array([1,2,1,2]),features=np.ones((4,2)))
    np.savez(directory/'coverage.npz',**cov);np.savez(directory/'features.npz',**data)
    report=dict(status='completed',provenance=provenance,coverage_sha256=sha256_file(directory/'coverage.npz'),
        features_sha256=sha256_file(directory/'features.npz'))
    if not legacy:report['stage']=stage
    (directory/'rebuild_report.json').write_text(json.dumps(report))
    return directory,cov,data,manifest


@pytest.mark.parametrize('stage',['maxmin','aggregate','exchange'])
def test_selected_stage_files_load_and_pin_exact_dictionary_and_features(tmp_path,stage):
    directory,_,_,expected=make_files(tmp_path,stage)
    manifest,report,cov,data,provenance=load_stage_training_artifacts(directory,stage=stage)
    assert manifest==expected
    assert provenance['stage']==stage and provenance['stage_key']==[stage,2,3,1]
    assert provenance['candidate_ids']==list(compact_candidate_ids(manifest['stages'][0]))
    assert provenance['bank_sha256']==sha256_file(directory/'manifest.json')
    assert provenance['coverage_sha256']==report['coverage_sha256']
    assert provenance['features_sha256']==report['features_sha256']
    assert provenance['rebuild_report_sha256']==sha256_file(directory/'rebuild_report.json')
    assert cov['coverage'].shape==(4,2) and data['features'].shape==(4,2)


def test_legacy_maxmin_artifacts_still_load_without_explicit_stage(tmp_path):
    directory,_,_,_=make_files(tmp_path,'maxmin',legacy=True)
    assert load_stage_training_artifacts(directory)[-1]['stage']=='maxmin'


@pytest.mark.parametrize('stage',['aggregate','exchange'])
def test_legacy_unstaged_labels_are_never_accepted_for_downstream_stage(tmp_path,stage):
    directory,cov,data,manifest=make_files(tmp_path,stage,legacy=True)
    with pytest.raises(ValueError):load_stage_training_artifacts(directory,stage=stage)
    with pytest.raises(ValueError,match=f'{stage} coverage'):
        validate_labels(cov,data,manifest,sha256_file(directory/'manifest.json'),stage=stage)


@pytest.mark.parametrize('file',['coverage.npz','features.npz','aggregate/root.npz',
    'aggregate/router.npz','aggregate/basis_0000.npz'])
def test_any_changed_stage_label_feature_or_dictionary_file_is_rejected(tmp_path,file):
    directory,_,_,_=make_files(tmp_path,'aggregate')
    with (directory/file).open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='SHA'):
        load_stage_training_artifacts(directory,stage='aggregate')


@pytest.mark.parametrize('mutation',[
    'stage','stage_key','candidate_ids','features','row_order','GEM','training_hash',
    'matrix_shape','feature_schema','nonfinite','routing_order',
])
def test_downstream_label_semantics_fail_closed_before_training(tmp_path,mutation):
    directory,cov,data,manifest=make_files(tmp_path,'exchange')
    meta=json.loads(str(cov['metadata']))
    bank=SimpleNamespace(host_a=SimpleNamespace(shape=(2,3)),variable_rows=np.array([1]),feature_indices=np.array([0,3]))
    if mutation=='stage':meta['stage']='aggregate'
    elif mutation=='stage_key':meta['stage_key']=['exchange',2,9,1]
    elif mutation=='candidate_ids':meta['candidate_ids']=list(reversed(compact_candidate_ids(manifest['stages'][0])))
    elif mutation=='features':data['features']=np.ones((4,3))
    elif mutation=='row_order':data['trajectory_ids']=np.array([11,12,11,12])
    elif mutation=='GEM':meta['model_fingerprints']={'toy':'f'*64}
    elif mutation=='training_hash':meta['training_manifest_sha256']='f'*64
    elif mutation=='matrix_shape':bank.host_a=SimpleNamespace(shape=(4,3))
    elif mutation=='feature_schema':meta['feature_schema_sha256']='f'*64
    elif mutation=='nonfinite':data['features'][0,0]=np.nan
    else:cov['routing_order'][0]=[0,0]
    cov['metadata']=np.array(json.dumps(meta))
    with pytest.raises(ValueError):
        validate_labels(cov,data,manifest,sha256_file(directory/'manifest.json'),stage='exchange',bank=bank)


@pytest.mark.parametrize('stage',['aggregate','exchange'])
def test_hybrid_constructor_allows_matching_downstream_router_without_gpu_use(monkeypatch,stage):
    import src.gpu_hybrid_lp as hybrid
    monkeypatch.setattr(hybrid,'RepeatedCpuLP',lambda *args,**kwargs:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(hybrid,'BatchedCompiledBackend',lambda *args,**kwargs:SimpleNamespace(repairs={}))
    key=(stage,2,3,1)
    bank=SimpleNamespace(feature_indices=np.array([0,3]),evaluators=[object(),object()])
    router=SimpleNamespace(candidate_count=2,input_dim=2,rank=lambda *args:None)
    solver=hybrid.HybridCertifiedBackend(SimpleNamespace(n_fluxes=3),{key:bank},gpu_stages=(stage,),
        coverage_routers={key:router})
    try:assert solver.coverage_routers[key] is router
    finally:solver.close()
    with pytest.raises(ValueError,match='matching original stage'):
        hybrid.HybridCertifiedBackend(SimpleNamespace(n_fluxes=3),{key:bank},gpu_stages=('maxmin',),
            coverage_routers={key:router})
    with pytest.raises(ValueError,match='maxmin-only'):
        hybrid.HybridCertifiedBackend(SimpleNamespace(n_fluxes=3),{key:bank},gpu_stages=(stage,),
            coverage_routers={key:router},speculative_cpu=True,repair_rounds=0)


@pytest.mark.parametrize('stage',['aggregate','exchange'])
def test_cpu_dictionary_already_accepts_matching_downstream_router_without_any_solve(stage):
    from scipy.sparse import csr_matrix
    from src.cpu_dictionary_lp import CpuDictionaryLP
    key=(stage,2,3,1)
    router=SimpleNamespace(xp=np,candidate_count=2,input_dim=2,rank=lambda *args:None)
    evaluator=SimpleNamespace(d=dict(kind=np.array([2,-1,-1]),active=np.array([0])))
    bank=SimpleNamespace(host_a=csr_matrix([[1.,0.,0.],[0.,1.,0.]]),neq=1,
        variable_rows=np.array([1]),centers=np.zeros((2,2)),feature_indices=np.array([0,3]),
        feature_scale=np.ones(2),evaluators=[evaluator,evaluator])
    service=SimpleNamespace()
    backend=CpuDictionaryLP(SimpleNamespace(n_fluxes=3),{key:bank},cpu_service=service,
        coverage_routers={key:router})
    assert backend.banks[key]['coverage_router'] is router
    assert backend.cpu is service
