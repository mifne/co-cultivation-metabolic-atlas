import copy,json
import numpy as np
import pytest
from scripts.train_coverage_router import validate_labels
from src.compact_bank_coverage import CERTIFICATE_THRESHOLDS


def fixture():
    manifest=dict(model_fingerprints={'toy':'a'*64},stages=[dict(stage='maxmin',entries=[{},{}])])
    metadata=dict(bank_sha256='b'*64,role='training_only_dictionary_reconstruction',
        model_fingerprints=manifest['model_fingerprints'],training_seeds=[10,20],steps=2)
    cov=dict(complete=np.array(True),schema_version=np.array(1),failures=np.array('[]'),
        certificate_thresholds=np.array(json.dumps(CERTIFICATE_THRESHOLDS)),metadata=np.array(json.dumps(metadata)),
        coverage=np.ones((4,2),dtype=bool),candidate_indices=np.arange(2))
    data=dict(trajectory_ids=np.array([10,10,20,20]),steps=np.array([1,2,1,2]),features=np.ones((4,3)))
    return cov,data,manifest


def test_valid_contract():
    cov,data,manifest=fixture();validate_labels(cov,data,manifest,'b'*64)


def test_declared_topk_prefix_does_not_require_full_bank_ranking():
    cov,data,manifest=fixture()
    cov.update(routing_order=np.zeros((4,1),dtype=np.int32),routing_topk=np.array([1]))
    validate_labels(cov,data,manifest,'b'*64)


@pytest.mark.parametrize('mutation',['missing','width','duplicate','zero','float','index'])
def test_bad_truncated_ranking_metadata_fails_closed(mutation):
    cov,data,manifest=fixture()
    cov.update(routing_order=np.zeros((4,1),dtype=np.int32),routing_topk=np.array([1]))
    if mutation=='missing':del cov['routing_topk']
    elif mutation=='width':cov['routing_topk']=np.array([2])
    elif mutation=='duplicate':cov['routing_topk']=np.array([1,1])
    elif mutation=='zero':cov['routing_topk']=np.array([0])
    elif mutation=='float':cov['routing_topk']=np.array([1.])
    else:cov['routing_order'][0,0]=2
    with pytest.raises(ValueError):validate_labels(cov,data,manifest,'b'*64)


@pytest.mark.parametrize('field,value',[
    ('candidate_indices',np.array([1,0])),('complete',np.array(1)),('complete',np.array(False)),
    ('schema_version',np.array(2)),('coverage',np.ones((4,2),dtype=float)),
    ('failures',np.array('[{"error":"failed"}]')),
    ('certificate_thresholds',np.array('{"primal_residual": 1.0}'))])
def test_invalid_label_contract(field,value):
    cov,data,manifest=fixture();cov[field]=value
    with pytest.raises(ValueError):validate_labels(cov,data,manifest,'b'*64)


@pytest.mark.parametrize('key,value',[
    ('bank_sha256','c'*64),('role','development_diagnostic_not_training'),('model_fingerprints',{'toy':'d'*64})])
def test_metadata_binding(key,value):
    cov,data,manifest=fixture();meta=json.loads(str(cov['metadata']));meta[key]=value;cov['metadata']=np.array(json.dumps(meta))
    with pytest.raises(ValueError):validate_labels(cov,data,manifest,'b'*64)


def test_feature_row_order():
    cov,data,manifest=fixture();data['trajectory_ids']=np.array([10,20,10,20])
    with pytest.raises(ValueError):validate_labels(cov,data,manifest,'b'*64)


@pytest.mark.parametrize('stage', ['aggregate', 'exchange'])
@pytest.mark.parametrize('location', ['labels', 'manifest'])
def test_downstream_labels_cannot_train_maxmin_router(stage, location):
    cov,data,manifest=fixture()
    if location=='labels':
        metadata=json.loads(str(cov['metadata']));metadata['stage']=stage
        cov['metadata']=np.array(json.dumps(metadata))
    else:
        manifest['reconstructed_stage']=stage
    with pytest.raises(ValueError,match='maxmin coverage'):
        validate_labels(cov,data,manifest,'b'*64)
