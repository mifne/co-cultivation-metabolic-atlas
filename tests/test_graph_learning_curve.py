import copy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.trajectory_design import trajectory_design,apply_design
from src.graph_training_collection import (scaling_decision, GraphTrainingCollection,
    TRACE_SOURCE_FILES, TEACHER_MODEL_FILES, require_matching_teacher_contract)
from src.learning_curve_evaluation import evaluate_curve


@pytest.mark.parametrize('index',range(4))
def test_design_reproducible_prefix_and_ranges(index):
    d,a=trajectory_design(71000000+index,index,120)
    e,b=trajectory_design(71000000+index,index,8)
    assert d==e
    np.testing.assert_array_equal(a[:8],b)
    assert a.shape==(120,5) and a.dtype==np.float32
    assert a.min()>=.049999 and a.max()<=.950001
    assert all(.75<=v<=1.25 for v in d['biomass_multipliers'])
    assert .5<=d['nh4_multiplier']<=2.
    assert 'hypothetical' in d['calibration']
    if index==1:np.testing.assert_array_equal(a[0],a[9])
    if index==2:assert np.max(abs(np.diff(a,axis=0)))<.4
    if index==3:assert np.all(a[15]>a[0])


@pytest.mark.parametrize('args',[(1,-1,120),(1,0,0),(1,0,121),(1,0,1.5)])
def test_invalid_design(args):
    with pytest.raises(ValueError):trajectory_design(*args)


def test_design_only_modifies_declared_initial_state():
    state=SimpleNamespace(species={str(i):SimpleNamespace(biomass=1.) for i in range(3)},metabolites={'nh4_e':2.,'other':3.})
    env=SimpleNamespace(simulator=SimpleNamespace(state=state))
    d,_=trajectory_design(1,0,120);apply_design(env,d)
    assert [s.biomass for s in state.species.values()]==d['biomass_multipliers']
    assert state.metabolites=={'nh4_e':2*d['nh4_multiplier'],'other':3.}


def point(n):
    return dict(train_count=n,training_seeds=3,converged=True,
        all_accuracy_gates_passed=True,speedup_ci95=[2.1,2.3],
        relative_gain_upper_ci95=.01,fixed_comparison_id='same-architecture-and-validation')


def test_512_is_not_a_cap_even_with_good_metrics():
    r=scaling_decision([point(n) for n in (128,256,512)])
    assert not r['adequate'] and r['next_train_count']==1024


def test_larger_probe_can_support_only_a_conditional_plateau():
    r=scaling_decision([point(n) for n in (256,512,1024)])
    assert r['adequate'] and r['action']=='candidate_plateau_not_universal_maximum'


@pytest.mark.parametrize('field,value',[
    ('training_seeds',1),('converged',False),('all_accuracy_gates_passed',False),
    ('speedup_ci95',[1.5,1.8]),('speedup_ci95',[float('nan'),3.]),
    ('relative_gain_upper_ci95',.1),('fixed_comparison_id','different')])
def test_unsupported_plateau_rejected(field,value):
    rows=[point(n) for n in (256,512,1024)];rows[-1][field]=value
    assert not scaling_decision(rows)['adequate']


def test_duplicate_sizes_rejected():
    with pytest.raises(ValueError):scaling_decision([point(32),point(32)])


def measurements():
    return dict(fixed_comparison_id='fixed',timing_unit='independent_trajectory_batch',sizes=[
        dict(train_count=n,converged=True,measurements=[dict(training_seed=s,block_id=b,
            total_seconds=1.,baseline_seconds=2.2,timing_repeats=3,all_accuracy_gates_passed=True)
            for s in range(3) for b in range(4)]) for n in (256,512,1024)])


def test_paired_bootstrap_constant_example():
    r=evaluate_curve(measurements(),resamples=200)
    assert r['decision']['adequate']
    np.testing.assert_allclose(r['points'][-1]['speedup_ci95'],[2.2,2.2])
    assert r['points'][-1]['relative_gain_upper_ci95']==0.


@pytest.mark.parametrize('error',['duplicate','missing','nan','baseline','unit','repeats','failed_accuracy'])
def test_invalid_or_failed_evaluation(error):
    d=measurements();rows=d['sizes'][-1]['measurements']
    if error=='duplicate':rows.append(copy.deepcopy(rows[0]))
    if error=='missing':rows.pop()
    if error=='nan':rows[0]['total_seconds']=float('nan')
    if error=='baseline':rows[0]['baseline_seconds']=3.
    if error=='unit':d['timing_unit']='correlated_timesteps'
    if error=='repeats':rows[0]['timing_repeats']=1
    if error=='failed_accuracy':
        rows[0]['all_accuracy_gates_passed']=False
        assert not evaluate_curve(d,resamples=200)['decision']['adequate'];return
    with pytest.raises(ValueError):evaluate_curve(d,resamples=200)


def collection_fixture(root):
    catalog=dict(configuration=dict(schema='graph_trajectory_collection_v1',steps=2,
        seed_bases=dict(train=100,selection=200,test=300)),model_fingerprints={'a':'fixed'},shards=[])
    for split,role in [('train','training_reference'),('selection','model_selection_reference'),('test','development_diagnostic_not_training')]:
        folder=root/split;folder.mkdir()
        seeds=list(range(catalog['configuration']['seed_bases'][split],catalog['configuration']['seed_bases'][split]+4))
        m=dict(status='completed',role=role,model_fingerprints={'a':'fixed'},completed_steps=[2]*4,seeds=seeds,
            entries=[dict(environment_id=i,stage=stage,step=t,problem_sha256=f'{split}-{i}-{stage}-{t}')
                for i in range(4) for stage in ('maxmin','aggregate','exchange') for t in (1,2)])
        raw=json.dumps(m).encode();(folder/'manifest.json').write_bytes(raw)
        catalog['shards'].append(dict(path=split,split=split,start=0,count=4,status='completed',seeds=seeds,
            manifest_sha256=hashlib.sha256(raw).hexdigest()))
    (root/'catalog.json').write_text(json.dumps(catalog))
    return catalog


def test_collection_streaming_statistics_use_training_only(tmp_path,monkeypatch):
    collection_fixture(tmp_path);data=GraphTrainingCollection(tmp_path);visited=[]
    def fake_load(folder,entry):
        visited.append(folder.name)
        return None,np.array([2.]),np.array([3.])
    monkeypatch.setattr('src.graph_training_collection.load_trace_lp',fake_load)
    stats=data.statistics(4,batch_size=2)
    assert visited==['train']*8
    np.testing.assert_allclose(np.array(stats).ravel(),[2.,2.,3.,3.])
    with pytest.raises(ValueError):data.select('test',4)
    with pytest.raises(ValueError):list(data.batches('train',4,3))
    with pytest.raises(ValueError):list(data.batches('train',4,2,order=[0,0]))


@pytest.mark.parametrize('error',['hash','seed','role','overlap','missing_step'])
def test_collection_rejects_bad_manifest(tmp_path,error):
    c=collection_fixture(tmp_path);path=tmp_path/'selection'/'manifest.json';m=json.loads(path.read_text())
    if error=='seed':m['seeds'][0]=100
    if error=='role':m['role']='training_reference'
    if error=='overlap':m['entries'][0]['problem_sha256']='train-0-maxmin-1'
    if error=='missing_step':m['entries'].pop()
    if error=='hash':m['extra']='tampered'
    raw=json.dumps(m).encode();path.write_bytes(raw)
    if error!='hash':
        c['shards'][1]['manifest_sha256']=hashlib.sha256(raw).hexdigest()
        (tmp_path/'catalog.json').write_text(json.dumps(c))
    with pytest.raises(ValueError):GraphTrainingCollection(tmp_path)


def teacher_contract_fixture():
    d=dict(schema='teacher_source_environment_v1',
        source_hashes={f:'source' for f in TRACE_SOURCE_FILES},
        model_source_hashes={f:'model' for f in TEACHER_MODEL_FILES},
        environment=dict(dynamics='legacy', fba_mode='cooperative'),
        runtime={'python':'test'}, teacher_strategy='scipy_cold_serial')
    d['sha256']=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    return d


def test_source_contract_covers_environment_and_models():
    assert {'scripts/benchmark_cooperative_surrogate_e2e.py','src/rl_environment.py',
        'src/utils.py','src/dfba_simulator.py'} <= set(TRACE_SOURCE_FILES)
    assert len(TEACHER_MODEL_FILES) == 3
    d=teacher_contract_fixture()
    require_matching_teacher_contract(d,copy.deepcopy(d))
    with pytest.raises(ValueError,match='provenance'):
        require_matching_teacher_contract(None,d)
    e=copy.deepcopy(d);e['source_hashes']['src/rl_environment.py']='changed'
    with pytest.raises(ValueError,match='digest'):
        require_matching_teacher_contract(d,e)
    e.pop('sha256');e['sha256']=hashlib.sha256(json.dumps(e,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError,match='new collection'):
        require_matching_teacher_contract(d,e)


@pytest.mark.parametrize('error',[None,'missing','mixed','source'])
def test_contracted_collection_requires_identical_provenance(tmp_path,error):
    c=collection_fixture(tmp_path);contract=teacher_contract_fixture();c['teacher_contract']=contract
    for shard in c['shards']:
        path=tmp_path/shard['path']/'manifest.json';m=json.loads(path.read_text())
        m['teacher_contract']=copy.deepcopy(contract);m['source_hashes']=dict(contract['source_hashes'])
        if shard['split']=='selection':
            if error=='missing':m.pop('teacher_contract')
            if error=='mixed':m['teacher_contract']['environment']['dynamics']='audited'
            if error=='source':m['source_hashes']['src/utils.py']='changed'
        raw=json.dumps(m).encode();path.write_bytes(raw);shard['manifest_sha256']=hashlib.sha256(raw).hexdigest()
    (tmp_path/'catalog.json').write_text(json.dumps(c))
    if error:
        with pytest.raises(ValueError):GraphTrainingCollection(tmp_path)
    else:
        assert GraphTrainingCollection(tmp_path).provenance_status=='verified_source_environment_contract'


def test_legacy_collection_stays_readable_but_append_rejects_before_catalog_write(tmp_path,monkeypatch):
    import scripts.collect_graph_learning_curve as collector
    c=collection_fixture(tmp_path)
    c['configuration'].update(steps=2,shard_size=4,design_profile='coverage_v1',
        seed_bases=dict(train=71000000,selection=72000000,test=73000000))
    path=tmp_path/'catalog.json';path.write_text(json.dumps(c));before=path.read_bytes()
    monkeypatch.setattr(collector,'teacher_source_contract',lambda *a,**k:teacher_contract_fixture())
    monkeypatch.setattr('sys.argv',['collect_graph_learning_curve.py','--output',str(tmp_path),
        '--count','8','--shard-size','4','--steps','2'])
    with pytest.raises(ValueError,match='legacy data remain read-only'):
        collector.main()
    assert path.read_bytes()==before
    assert not (tmp_path/'train_000004_000008').exists()
