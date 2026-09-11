import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.compact_training_data import load_compact_training_source,sha256_file


@pytest.fixture
def fixture(tmp_path):
    data=tmp_path/'data';bank=tmp_path/'bank'
    (data/'inputs/maxmin').mkdir(parents=True);(bank/'maxmin').mkdir(parents=True)
    training=dict(status='completed',actual_training_seeds=[11,12],actual_training_steps=2,
        provenance=[dict(identity=dict(model_fingerprints={'toy':'a'*64}))])
    (data/'manifest.json').write_text(json.dumps(training))
    a=csr_matrix([[1.,2.],[0.,1.]])
    np.savez(bank/'maxmin/root.npz',a_data=a.data,a_indices=a.indices,a_indptr=a.indptr,
        a_shape=a.shape,neq=1,variable_rows=np.array([1]))
    np.savez(bank/'maxmin/router.npz',indices=np.array([0,3]),scale=np.ones(2),centers=np.zeros((1,2)))
    manifest=dict(status='completed',model_fingerprints={'toy':'a'*64},train_seeds=[11,12],source_manifest=training,
        stages=[dict(stage='maxmin',key=['maxmin',2,2,1],entries=[dict(filename='unused.npz')],
        root_sha256=sha256_file(bank/'maxmin/root.npz'),router_sha256=sha256_file(bank/'maxmin/router.npz'))])
    (bank/'manifest.json').write_text(json.dumps(manifest))
    values=dict(rhs=np.arange(8,dtype=float).reshape(4,2),lower=np.zeros((4,2)),upper=np.ones((4,2))*4,
        c=np.ones((4,2)),delta=np.zeros((4,1,2)),col_scale=np.ones((4,2))*2,row_scale=np.ones((4,2))*.5)
    values['delta'][:,0,0]=np.arange(4)
    for start in (0,2):np.savez(data/f'inputs/maxmin/{start:06d}.npz',**{k:v[start:start+2] for k,v in values.items()})
    return data,bank,values


def test_roundtrip_and_row_order(fixture):
    data,bank,values=fixture;s=load_compact_training_source(data,bank)
    assert s.size==4 and s.seeds==(11,12)
    assert s.provenance['row_identity'][2]==dict(index=2,seed=12,step=1)
    assert [len(b['rhs']) for b in s.batches(3)]==[3,1]
    a,rhs,lo,hi,c,neq=s.original_problem(2)
    np.testing.assert_array_equal(a.toarray(),np.array([[1,2],[2,1]])*4)
    np.testing.assert_array_equal(rhs,values['rhs'][2]*2)
    np.testing.assert_array_equal(hi,[2,2]);np.testing.assert_array_equal(c,[2,2])
    assert neq==1 and s.selected_features().shape==(4,2)


@pytest.mark.parametrize('field,change',[
    ('rhs',lambda v:v.astype(np.float32)),('rhs',lambda v:np.array(1.)),
    ('col_scale',lambda v:np.zeros_like(v)),('lower',lambda v:np.full_like(v,np.nan)),
    ('upper',lambda v:np.full_like(v,-1.)),('delta',lambda v:v[:,:0])])
def test_malformed_cache_rejected(fixture,field,change):
    data,bank,values=fixture;chunk={k:v[:2] for k,v in values.items()};chunk[field]=change(chunk[field])
    np.savez(data/'inputs/maxmin/000000.npz',**chunk)
    with pytest.raises(ValueError):load_compact_training_source(data,bank)


def test_seed_model_hash_and_order_guards(fixture):
    data,bank,_=fixture
    with pytest.raises(ValueError):load_compact_training_source(data,bank,forbidden_seeds=[12])
    with pytest.raises(ValueError):load_compact_training_source(data,bank,expected_model_fingerprints={'toy':'b'*64})
    (data/'inputs/maxmin/000002.npz').rename(data/'inputs/maxmin/000003.npz')
    with pytest.raises(ValueError):load_compact_training_source(data,bank)


def test_hash_mismatch(fixture):
    data,bank,_=fixture
    with (bank/'maxmin/router.npz').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError):load_compact_training_source(data,bank)


@pytest.mark.parametrize('stage',['aggregate','exchange'])
def test_stage_specific_inputs_are_loaded_without_maxmin_substitution(fixture,stage):
    data,bank,values=fixture
    (data/'inputs/maxmin').rename(data/'inputs'/stage)
    (bank/'maxmin').rename(bank/stage)
    manifest=json.loads((bank/'manifest.json').read_text())
    manifest['stages'][0].update(stage=stage,key=[stage,2,2,1])
    (bank/'manifest.json').write_text(json.dumps(manifest))
    source=load_compact_training_source(data,bank,stage=stage)
    assert source.stage==stage
    assert source.provenance['stage']==stage
    assert source.provenance['stage_key']==[stage,2,2,1]
    np.testing.assert_array_equal(source.data['c'],values['c'])
    np.testing.assert_array_equal(source.data['delta'],values['delta'])
    assert all(row['filename'].startswith(f'inputs/{stage}/') for row in source.provenance['input_chunks'])
    # The maxmin default remains maxmin; it must not silently pick another stage.
    with pytest.raises(ValueError,match='Exactly one'):
        load_compact_training_source(data,bank)
    with pytest.raises(ValueError,match='Forbidden'):
        load_compact_training_source(data,bank,stage=stage,forbidden_seeds=[12])
    with (bank/stage/'root.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        load_compact_training_source(data,bank,stage=stage)


@pytest.mark.parametrize('stage',['exchange_tie','../maxmin',None,False])
def test_only_declared_supported_stages_are_accepted(fixture,stage):
    with pytest.raises(ValueError,match='Stage must'):
        load_compact_training_source(*fixture[:2],stage=stage)
