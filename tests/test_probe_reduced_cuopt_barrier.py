"""CPU-only input/provenance/API tests; no CUDA or optimizer execution."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import scripts.probe_reduced_cuopt_barrier as probe
from src.lp_trace import problem_hash


def problem():
    return (csr_matrix([[1.,1.]]),np.array([1.]),np.zeros(2),np.ones(2),
            np.array([1.,2.]),1)


def make_trace(path,count=4):
    path.mkdir()
    p=problem()
    a,rhs,lo,hi,c,neq=p
    entries=[]
    for env in range(count):
        name=f'lp_001_2_{env:03d}.npz'
        target=path/name
        # Loading either reference object would raise with allow_pickle=False.
        # A successful input load therefore checks that these fields stay unread.
        np.savez(target,a_data=a.data,a_indices=a.indices,a_indptr=a.indptr,a_shape=a.shape,
            rhs=rhs,lower=lo,upper=hi,c=c,neq=neq,
            reference_x=np.array([object()],dtype=object),reference_y=np.array([object()],dtype=object))
        entries.append(dict(filename=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            problem_sha256=problem_hash(p),rows=1,columns=2,nonzeros=2,equalities=1,
            stage='exchange',step=1,environment_id=env))
    manifest=dict(status='completed',role='development_diagnostic_not_training',
        seeds=list(range(count)),completed_steps=[1]*count,entries=entries)
    (path/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_verified_inputs_never_load_reference_vectors(tmp_path):
    trace=tmp_path/'trace'
    manifest=make_trace(trace)
    problems,record=probe.load_inputs(trace,'exchange',1,4)
    assert len(problems)==4
    assert record['entries']==manifest['entries']
    assert record['current_reference_vectors_loaded'] is False
    assert record['model_vectors_loaded'] is False
    assert record['input_manifest_sha256']==hashlib.sha256((trace/'manifest.json').read_bytes()).hexdigest()
    assert record['input_problem_sha256']==[problem_hash(problem())]*4


def test_trace_checksum_and_distinct_environment_contracts_fail_closed(tmp_path):
    trace=tmp_path/'trace'
    manifest=make_trace(trace)
    with pytest.raises(ValueError,match='distinct'):
        probe.load_inputs(trace,'exchange',1,32)
    manifest['entries'][0]['sha256']='0'*64
    (trace/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='checksum'):
        probe.load_inputs(trace,'exchange',1,4)


def test_untrusted_near_proposal_loads_only_exact_row_weight_candidate(tmp_path):
    path=tmp_path/'proposal.json'
    content=dict(exact_stored_data_relation=dict(target_zero_face_row=3,
        support=[dict(zero_face_row=1,coefficient=dict(numerator=1,denominator=2))]))
    path.write_text(json.dumps(content))
    candidate,metadata=probe.load_near_proposal(path)
    assert candidate.row==3 and candidate.basis_rows==(1,)
    assert str(candidate.coefficients[0])=='1/2'
    assert metadata['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'NOT exact equivalence' in metadata['scope']
    assert probe.load_near_proposal(None)==(None,None)


class FakeModel:
    def __init__(self):self.received={}
    def __getattr__(self,name):
        if not name.startswith('set_'):raise AttributeError(name)
        def receive(*values):
            self.received[name]=tuple(v.copy() if isinstance(v,np.ndarray) else v for v in values)
        return receive


def fake_backend(method='Barrier',status='Optimal',wrong_shape=False):
    calls=[]
    model=FakeModel()
    solution=SimpleNamespace(
        get_termination_status=lambda:SimpleNamespace(name=status),
        get_solved_by=lambda:SimpleNamespace(name=method),
        get_primal_solution=lambda:np.ones(3 if wrong_shape else 8),
        get_dual_solution=lambda:np.zeros(4),
        get_solve_time=lambda:.25,get_lp_stats=lambda:dict(iterations=8))
    def solve(received,settings):
        calls.append((received,settings))
        return solution
    backend=SimpleNamespace(cp=SimpleNamespace(cuda=SimpleNamespace(runtime=
        SimpleNamespace(deviceSynchronize=lambda:None))),
        lp=SimpleNamespace(DataModel=lambda:model,Solve=solve),settings=object())
    return backend,calls,model


def test_native_barrier_assembles_independent_blocks_and_confirms_solver_method():
    backend,calls,model=fake_backend()
    packed,x,y,record=probe.native_barrier(backend,[problem()]*4)
    assert packed[0].shape==(4,8) and packed[0].nnz==8
    assert len(calls)==1 and calls[0][1] is backend.settings
    assert model.received['set_maximize']==(False,)
    assert record['gpu_backend_confirmed'] and record['cpu_lp_calls']==0
    assert record['status']=='optimal' and record['valid_shape']
    assert record['configuration']['presolve']==0
    assert record['configuration']['crossover'] is False
    assert record['configuration']['time_limit_seconds']==20.


@pytest.mark.parametrize('method',['DualSimplex','Unknown','Concurrent'])
def test_other_or_unknown_solver_never_claims_gpu_only_cpu_zero(method):
    backend,_,_=fake_backend(method=method)
    _,_,_,record=probe.native_barrier(backend,[problem()]*4)
    assert not record['gpu_backend_confirmed'] and record['cpu_lp_calls'] is None


def test_missing_candidate_shape_is_reported_without_fake_vectors():
    backend,_,_=fake_backend(wrong_shape=True,status='TimeLimit')
    _,x,_,record=probe.native_barrier(backend,[problem()]*4)
    assert not record['valid_shape'] and x.shape==(3,)
    assert record['status']=='timelimit'


def test_direct_gate_uses_primal_scale_and_rejects_signed_large_gap():
    p=(csr_matrix([[1.]]),np.array([0.]),np.array([-1.]),np.array([1.]),np.array([1e8]),1)
    rows=probe.audit_pairs([p],np.array([[1e-6]]),np.array([[1e8]]),xp=np)
    assert rows[0]['signed_gap']==100.
    assert rows[0]['relative_direct_gap_primal_scale']==1.
    assert rows[0]['direct_dual_gate_passed'] is False


def test_json_nonfinite_numpy_types_and_exclusive_output(tmp_path):
    path=tmp_path/'result.json'
    probe.write_exclusive(path,dict(values=np.array([np.inf,-np.inf,np.nan]),
                                    accepted=np.bool_(False),count=np.int64(4)))
    content=json.loads(path.read_text())
    assert content==dict(values=['Infinity','-Infinity','NaN'],accepted=False,count=4)
    with pytest.raises(FileExistsError):probe.write_exclusive(path,dict(overwrite=True))
    assert json.loads(path.read_text())==content


def test_serialization_failure_does_not_leave_partial_output(tmp_path):
    path=tmp_path/'result.json'
    with pytest.raises(TypeError):probe.write_exclusive(path,dict(opaque=object()))
    assert not path.exists()


def test_main_existing_output_refused_before_trace_or_backend_access(tmp_path,monkeypatch):
    output=tmp_path/'result.json'
    output.write_text('original')
    def forbidden(*args,**kwargs):raise AssertionError('Must not load inputs')
    monkeypatch.setattr(probe,'load_inputs',forbidden)
    with pytest.raises(FileExistsError):probe.main(['--trace',str(tmp_path/'missing'),'--output',str(output)])
    assert output.read_text()=='original'
