from types import SimpleNamespace

import numpy as np
import pytest

from src.coverage_router import CoverageRouter, FORMAT, VERSION
from src.coverage_router_binding import (
    compact_candidate_ids,
    compact_feature_schema,
    compact_router_provenance,
    load_bound_coverage_router,
)


def _stage(stage='maxmin'):
    return dict(stage=stage, key=[stage, 2, 3, 1], entries=[
        dict(filename='basis_0000.npz', sha256='1'*64),
        dict(filename='basis_0001.npz', sha256='2'*64),
    ])


def _bank(indices=(0, 3)):
    return SimpleNamespace(host_a=SimpleNamespace(shape=(2, 3)),
        variable_rows=np.array([1], dtype=np.int64),
        feature_indices=np.array(indices, dtype=np.int64),
        evaluators=[object(), object()])


def _models():
    return {'OR16':'3'*64, 'NS21':'4'*64, 'PF':'5'*64}


def _router(bank=None, stage=None, manifest_sha='6'*64):
    bank = _bank() if bank is None else bank
    stage = _stage() if stage is None else stage
    provenance = compact_router_provenance(
        manifest_sha, stage, bank, '7'*64, _models())
    metadata = dict(format=FORMAT, version=VERSION,
        architecture=dict(input_dim=2, selected_feature_count=2,
            candidate_count=2, hidden_dim=0, activation='linear'),
        provenance=provenance)
    arrays = dict(indices=np.array([0, 1], dtype=np.int64),
        mean=np.zeros(2, dtype=np.float32), scale=np.ones(2, dtype=np.float32),
        w1=np.eye(2, dtype=np.float32), b1=np.zeros(2, dtype=np.float32))
    return CoverageRouter(arrays, metadata)


def test_candidate_ids_and_feature_schema_pin_order_shapes_indices_and_encoding():
    stage, bank = _stage(), _bank()
    assert compact_candidate_ids(stage) == (
        'basis_0000.npz:'+'1'*64, 'basis_0001.npz:'+'2'*64)
    schema, digest = compact_feature_schema(bank)
    assert schema['field_order'] == [
        'rhs', 'lower', 'upper', 'c', 'delta', 'col_scale', 'row_scale']
    assert schema['field_shapes']['delta'] == [1, 3]
    assert schema['selected_feature_indices'] == [0, 3]
    assert len(digest) == 64
    assert compact_feature_schema(_bank((3, 0)))[1] != digest
    wider = _bank(); wider.host_a = SimpleNamespace(shape=(3, 3))
    assert compact_feature_schema(wider)[1] != digest


@pytest.mark.parametrize('stage_name',['maxmin','aggregate','exchange'])
def test_sha_pinned_router_loads_and_binds_every_deployment_identity(tmp_path,stage_name):
    import hashlib
    bank, stage = _bank(), _stage(stage_name)
    router = _router(bank, stage)
    path = tmp_path/'router.npz'
    router.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    device, record = load_bound_coverage_router(path, digest,
        bank_manifest_sha256='6'*64, stage_manifest=stage, bank=bank,
        model_fingerprints=_models(), xp=np)
    assert device.rank(np.array([[2., 1.]], dtype=np.float32)).tolist() == [[0, 1]]
    assert record['candidate_ids'] == list(compact_candidate_ids(stage))
    assert record['bank_manifest_sha256'] == '6'*64
    assert record['stage']==stage_name and record['stage_key']==[stage_name,2,3,1]
    assert 'original certificate' in record['scope']


@pytest.mark.parametrize('mutation', ['manifest', 'candidate_order', 'models', 'features'])
def test_router_binding_rejects_stale_bank_candidate_gem_or_feature_identity(tmp_path, mutation):
    import copy
    import hashlib
    original_bank, original_stage = _bank(), _stage()
    path = tmp_path/'router.npz'
    _router(original_bank, original_stage).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bank, stage, models, manifest = _bank(), copy.deepcopy(_stage()), _models(), '6'*64
    if mutation == 'manifest':
        manifest = '8'*64
    elif mutation == 'candidate_order':
        stage['entries'].reverse()
    elif mutation == 'models':
        models['OR16'] = '9'*64
    else:
        bank.feature_indices = np.array([0, 4], dtype=np.int64)
    with pytest.raises(ValueError, match='mismatch'):
        load_bound_coverage_router(path, digest, bank_manifest_sha256=manifest,
            stage_manifest=stage, bank=bank, model_fingerprints=models, xp=np)


def test_malformed_candidate_manifest_and_wrong_artifact_sha_fail_closed(tmp_path):
    import hashlib
    path = tmp_path/'router.npz'
    _router().save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bad = _stage(); bad['entries'][0]['filename'] = '../basis.npz'
    with pytest.raises(ValueError, match='basename'):
        compact_candidate_ids(bad)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_bound_coverage_router(path, '0'*64, bank_manifest_sha256='6'*64,
            stage_manifest=_stage(), bank=_bank(), model_fingerprints=_models(), xp=np)


@pytest.mark.parametrize('stage_name',['aggregate','exchange'])
def test_same_dimensions_do_not_allow_router_stage_substitution(tmp_path,stage_name):
    bank,stage=_bank(),_stage(stage_name)
    path=tmp_path/'router.npz';saved=_router(bank,stage).save(path)
    with pytest.raises(ValueError,match='mismatch'):
        load_bound_coverage_router(path,saved['sha256'],bank_manifest_sha256='6'*64,
            stage_manifest=_stage('maxmin'),bank=bank,model_fingerprints=_models(),xp=np)


@pytest.mark.parametrize('key',[
    ['aggregate',2,3,1],['maxmin',3,3,1],['maxmin',2,4,1],['maxmin',2,3,3],
    ['maxmin',2,3,-1],['maxmin',2,3,True],['maxmin',2.,3,1],['maxmin',2,3],
])
def test_inconsistent_stage_key_fails_before_binding(key):
    stage=_stage();stage['key']=key
    with pytest.raises(ValueError):
        compact_router_provenance('6'*64,stage,_bank(),'7'*64,_models())


def test_only_original_three_stages_supported():
    with pytest.raises(ValueError,match='original compact stage'):
        compact_router_provenance('6'*64,_stage('exchange_tie'),_bank(),'7'*64,_models())
