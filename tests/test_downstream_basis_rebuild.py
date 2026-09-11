"""Stage selection and artifact contracts without GPU/real training jobs."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.compact_bank_coverage import CERTIFICATE_THRESHOLDS
from src.compact_training_data import sha256_file
from scripts.rebuild_coverage_basis_bank import (copy_and_extend_selected_stage,
    validate_training_baseline)
import scripts.measure_training_bank_coverage as measurement


def baseline_fixture(stage):
    provenance=dict(stage=stage,stage_key=[stage,7,9,4],role='training_only_dictionary_reconstruction',
        base_bank_manifest_sha256='a'*64,training_manifest_sha256='b'*64,
        model_fingerprints={'toy':'c'*64},training_seeds=[11,12],steps=2,
        input_chunks=[dict(filename=f'inputs/{stage}/000000.npz',sha256='d'*64)],
        forbidden_seeds=[30])
    source=SimpleNamespace(stage=stage,provenance=provenance,size=4,
        stage_manifest=dict(entries=[dict(filename='basis_0000.npz')]))
    report=dict(status='completed',provenance=copy.deepcopy(provenance))
    raw=dict(metadata=np.array(json.dumps(provenance)),complete=np.array(True),
        schema_version=np.array(1),failures=np.array('{}'),
        certificate_thresholds=np.array(json.dumps(CERTIFICATE_THRESHOLDS)),
        coverage=np.ones((4,1),dtype=bool),candidate_indices=np.array([0]))
    return source,report,raw


@pytest.mark.parametrize('stage',['maxmin','aggregate','exchange'])
def test_baseline_requires_exact_stage_and_strict_original_certificate(stage):
    source,report,raw=baseline_fixture(stage)
    validate_training_baseline(source,report,raw)
    wrong=copy.deepcopy(report);wrong['provenance']['stage']='exchange' if stage!='exchange' else 'maxmin'
    with pytest.raises(ValueError,match='stage'):
        validate_training_baseline(source,wrong,raw)
    for key,value in [('complete',np.array(False)),('schema_version',np.array(2)),('schema_version',np.array(1.1)),
            ('failures',np.array('{"0":"failed"}')),('coverage',np.ones((4,1))),
            ('candidate_indices',np.array([0.])),
            ('certificate_thresholds',np.array(json.dumps(dict(primal_residual=1.))))]:
        bad=dict(raw);bad[key]=value
        with pytest.raises(ValueError):validate_training_baseline(source,report,bad)


@pytest.mark.parametrize('key',['base_bank_manifest_sha256','training_manifest_sha256',
    'input_chunks','model_fingerprints','training_seeds'])
def test_baseline_npz_metadata_is_checked_independently_of_report(key):
    source,report,raw=baseline_fixture('aggregate')
    changed=copy.deepcopy(source.provenance);changed[key]='changed'
    raw['metadata']=np.array(json.dumps(changed))
    with pytest.raises(ValueError,match='changed'):
        validate_training_baseline(source,report,raw)


@pytest.mark.parametrize('stage',['maxmin','aggregate','exchange'])
def test_unstaged_legacy_baseline_is_only_compatible_with_old_maxmin(stage):
    source,report,raw=baseline_fixture(stage)
    for key in ('stage','stage_key'):report['provenance'].pop(key)
    raw['metadata']=np.array(json.dumps(report['provenance']))
    if stage=='maxmin':validate_training_baseline(source,report,raw)
    else:
        with pytest.raises(ValueError):validate_training_baseline(source,report,raw)


@pytest.mark.parametrize('stage',['maxmin','aggregate','exchange'])
def test_only_selected_stage_changes_all_others_remain_byte_identical(tmp_path,stage):
    base,out,pool=tmp_path/'base',tmp_path/'out',tmp_path/'pool'
    base.mkdir();out.mkdir();pool.mkdir()
    manifest=dict(stages=[])
    for name in ('maxmin','aggregate','exchange'):
        folder=base/name;folder.mkdir()
        (folder/'root.npz').write_bytes(b'root '+name.encode())
        (folder/'router.npz').write_bytes(b'router '+name.encode())
        (folder/'basis_0000.npz').write_bytes(b'basis '+name.encode())
        manifest['stages'].append(dict(stage=name,key=[name,2,3,1],untouched='metadata',
            root_sha256=sha256_file(folder/'root.npz'),router_sha256=sha256_file(folder/'router.npz'),
            entries=[dict(filename='basis_0000.npz',sha256=sha256_file(folder/'basis_0000.npz'))]))
    original=copy.deepcopy(manifest)
    source=SimpleNamespace(stage=stage,stage_manifest=next(row for row in original['stages'] if row['stage']==stage),
        feature_indices=np.array([1]),feature_scale=np.array([2.]))
    (pool/'candidate_0000.npz').write_bytes(b'new certified projection')
    record=dict(pool_filename='candidate_0000.npz',sha256=sha256_file(pool/'candidate_0000.npz'),
        candidate_signature='e'*64,source_indices=[0])
    copy_and_extend_selected_stage(manifest,base,out,selected_stage=stage,pool_directory=pool,pool=[record],
        selection=SimpleNamespace(selected=(0,1),selected_bytes=12),centers=[np.array([0.]),np.array([1.])],source=source)
    for before,after in zip(original['stages'],manifest['stages']):
        name=before['stage']
        assert (base/name/'root.npz').read_bytes()==(out/name/'root.npz').read_bytes()
        assert (base/name/'basis_0000.npz').read_bytes()==(out/name/'basis_0000.npz').read_bytes()
        if name!=stage:
            assert before==after
            assert (base/name/'router.npz').read_bytes()==(out/name/'router.npz').read_bytes()
            assert sorted(path.name for path in (base/name).iterdir())==sorted(path.name for path in (out/name).iterdir())
        else:
            assert len(after['entries'])==2 and after['entries'][0]==before['entries'][0]
            assert after['router_sha256']==sha256_file(out/name/'router.npz')
            assert after['router_sha256']!=before['router_sha256']
            assert (out/name/'basis_0001.npz').read_bytes()==(pool/'candidate_0000.npz').read_bytes()
    # Source artifacts and metadata never change.
    assert source.stage_manifest==next(row for row in original['stages'] if row['stage']==stage)


@pytest.mark.parametrize('stage',['aggregate','exchange'])
def test_coverage_loader_uses_only_selected_stage_paths_and_hashes(monkeypatch,tmp_path,stage):
    calls=[]
    def checked(folder,name,digest):
        calls.append((folder,name,digest))
        if name=='router.npz':return dict(centers=np.zeros((1,1)),indices=np.array([0]),scale=np.ones(1))
        return {'projection':np.ones(1)}
    monkeypatch.setattr(measurement,'checked_npz',checked)
    monkeypatch.setattr(measurement,'CompactBank',lambda *args,**kwargs:(args,kwargs))
    source=SimpleNamespace(stage=stage,stage_manifest=dict(router_sha256='router hash',
        entries=[dict(filename='basis_0000.npz',sha256='basis hash')]),
        root={'actual stage root':stage},variable_rows=np.array([2]),feature_indices=np.array([0]),feature_scale=np.ones(1))
    result=measurement.load_training_bank(source,tmp_path)
    assert calls==[(tmp_path/stage,'basis_0000.npz','basis hash'),(tmp_path/stage,'router.npz','router hash')]
    assert result[0][0] is source.root
    assert result[0][1] is source.variable_rows
