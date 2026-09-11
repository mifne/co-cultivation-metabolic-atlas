"""CPU-only tests of baseline provenance/accounting; no timed LP workload."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.benchmark_ipm_cpu_inputs import (benchmark,history_counts,load_inputs,
                                             write_exclusive,CPU_WORKER_CHOICES,_timed_batch)
from src.lp_trace import problem_request
from src.cpu_repeated_lp import _certificate,_problem
from src.lp_trace import write_trace_lp


def _problem_fixture(rhs=2.):
    return (csr_matrix([[1.]]),np.array([rhs]),np.zeros(1),np.array([3.]),np.array([-1.]),0)


def _trace(tmp_path):
    entries=[]
    for environment in range(2):
        entry=write_trace_lp(tmp_path/f'lp_001_2_{environment:03d}.npz',
            _problem_fixture(2.+environment*.1),reference_x=np.array([999.]),
            reference_y=np.array([999.]))
        entry.update(stage='exchange',step=1,environment_id=environment)
        entries.append(entry)
    manifest=dict(status='completed',role='development_diagnostic_not_training',
        seeds=[10,20],completed_steps=[1,1],entries=entries)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_trace_loader_verifies_inputs_without_indexing_reference_vectors(tmp_path,monkeypatch):
    manifest=_trace(tmp_path)
    original=np.load
    indexed=[]

    class InputsOnly:
        def __init__(self,*args,**kwargs):
            self.data=original(*args,**kwargs)
            self.files=self.data.files

        def __enter__(self):return self
        def __exit__(self,*_):self.data.close()
        def __getitem__(self,key):
            assert key not in ('reference_x','reference_y')
            indexed.append(key)
            return self.data[key]

    monkeypatch.setattr(np,'load',InputsOnly)
    problems,provenance=load_inputs(tmp_path,'exchange',1,2)
    assert len(problems)==2
    assert indexed and not set(indexed)&{'reference_x','reference_y'}
    assert provenance['current_reference_vectors_loaded'] is False
    assert provenance['gpu_calls']==0
    assert provenance['entries']==manifest['entries']
    assert provenance['problem_sha256']==[e['problem_sha256'] for e in manifest['entries']]
    assert provenance['input_manifest_sha256']==hashlib.sha256(
        (tmp_path/'manifest.json').read_bytes()).hexdigest()


def test_trace_loader_never_pads_large_batch_with_repeated_environments(tmp_path):
    _trace(tmp_path)
    with pytest.raises(ValueError,match='Distinct recorded environments'):
        load_inputs(tmp_path,'exchange',1,32)


@pytest.mark.parametrize('tamper',['npz','problem_hash','labels'])
def test_trace_loader_rejects_changed_input_or_identity(tmp_path,tamper):
    manifest=_trace(tmp_path)
    if tamper=='npz':
        path=tmp_path/manifest['entries'][0]['filename']
        path.write_bytes(path.read_bytes()+b'changed')
    elif tamper=='problem_hash':
        manifest['entries'][0]['problem_sha256']='0'*64
    else:
        manifest['entries'][0]['filename']='lp_002_2_000.npz'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):load_inputs(tmp_path,'exchange',1,2)


class FakeCpuService:
    instances=[]
    wrong_dual=False

    def __init__(self,**configuration):
        self.configuration=configuration
        self.history=[]
        self.models={}
        self.closed=False
        self.instances.append(self)

    def solve_batch(self,requests,*,environment_ids):
        results=[]
        diagnostics=[]
        for environment,(c,kwargs) in zip(environment_ids,requests):
            problem=_problem(c,kwargs)
            a,rhs,_,_,_,neq=problem
            x=rhs.copy()
            solution=SimpleNamespace(col_value=x,row_dual=np.array([-1.]),
                col_dual=np.zeros(1),value_valid=True,dual_valid=True)
            certificate=_certificate(*problem,solution)
            assert certificate['certificate_passed']
            row=dict(cpu_solver_runs=2,numerical_retry_count=2,
                solver_attempts=[dict(kind='initial',solver_run=True),
                    dict(kind='strict_basis_refactor',solver_run=False),
                    dict(kind='strict_cold_no_presolve',solver_run=True)],**certificate)
            results.append(SimpleNamespace(success=True,x=x,fun=float(c@x),
                diagnostics=row,solution_snapshot=solution,message='test double: optimal'))
            diagnostics.append(row)
            if self.wrong_dual:solution.row_dual=np.array([1.])
            self.models[(environment,kwargs['_stage'],a.shape[0],len(c),neq)]={
                'solver':SimpleNamespace(getSolution=lambda s=solution:s)}
        self.history.append(dict(batch=len(requests),cpu_solver_runs=2*len(requests),
            numerical_retry_count=2*len(requests),rows=diagnostics))
        return results

    def close(self):self.closed=True


def test_fresh_models_hot_labels_actual_runs_and_original_certificates():
    FakeCpuService.instances=[]
    result=benchmark([_problem_fixture(),_problem_fixture(2.1)],'exchange',
        workers=(1,4),repeats=2,hot_repeats=1,service_factory=FakeCpuService)
    assert len(FakeCpuService.instances)==4
    assert all(s.closed and len(s.history)==2 for s in FakeCpuService.instances)
    assert [s.configuration['workers'] for s in FakeCpuService.instances]==[1,1,4,4]
    assert result['status']=='all_cpu_trials_original_certified'
    # Four models * two batches * two environments * two actual run() calls.
    assert result['actual_cpu_optimizer_runs']==32
    assert result['logical_lp_requests']==16
    assert result['numerical_retry_attempts']==32
    assert result['numerical_retry_optimizer_runs']==16
    for group in result['groups']:
        assert len(group['cold_solve_batch_seconds'])==2
        for trial in group['trials']:
            assert trial['fresh_model']
            assert not trial['cold_process_or_os_cache_claimed']
            assert trial['cold']['kind']=='fresh_model'
            assert trial['hot_same_input'][0]['kind']=='identical_input_hot_repeat_not_dfba'
            assert trial['cold']['rows'][0]['objective']==-2.
            assert trial['cold']['rows'][0]['independent_original_certificate']['certificate_passed']
            assert trial['cold']['solve_batch_wall_seconds']>=0.


def test_success_flag_alone_never_qualifies_cpu_baseline():
    class WrongDualService(FakeCpuService):wrong_dual=True
    result=benchmark([_problem_fixture()],'exchange',workers=(1,),repeats=1,
                     service_factory=WrongDualService)
    assert result['status']=='unqualified_cpu_trial_present'
    row=result['groups'][0]['trials'][0]['cold']['rows'][0]
    assert row['solver_success'] and row['backend_original_certificate']['certificate_passed']
    assert not row['qualified']
    assert not row['independent_original_certificate']['certificate_passed']


def test_benchmark_certification_uses_owned_output_not_main_thread_native_access():
    class NoNativeAccess(FakeCpuService):
        def solve_batch(self,*args,**kwargs):
            results=super().solve_batch(*args,**kwargs)
            def forbidden():raise AssertionError('No native HiGHS getter allowed on benchmark thread')
            for entry in self.models.values():
                entry['solver'].getSolution=forbidden
            return results
    result=benchmark([_problem_fixture()],'maxmin',workers=(1,),repeats=1,
                     service_factory=NoNativeAccess)
    assert result['status']=='all_cpu_trials_original_certified'


def test_additional_direct_gate_uses_returned_snapshot_without_native_getter(monkeypatch):
    service=FakeCpuService()
    original=service.solve_batch
    def solve(*args,**kwargs):
        results=original(*args,**kwargs)
        def forbidden(): raise AssertionError('Extra native getter is not needed')
        for entry in service.models.values(): entry['solver'].getSolution=forbidden
        return results
    service.solve_batch=solve
    p=_problem_fixture()
    trial=_timed_batch(service,[p],[problem_request(p,stage='maxmin')],
                       'maxmin','test',0,require_direct_dual=True)
    assert trial['all_original_certificates_passed']
    row=trial['rows'][0]
    assert row['additional_direct_gate_required'] and row['additional_direct_gate_passed']
    assert row['independent_direct_dual_audit']['signed_gap']==[0.]
    from src import lp_direct_dual_audit
    original_audit=lp_direct_dual_audit.audit_direct_dual
    def wrong_gap(*args,**kwargs):
        result=original_audit(*args,**kwargs)
        result['signed_gap']=1.
        return result
    monkeypatch.setattr(lp_direct_dual_audit,'audit_direct_dual',wrong_gap)
    trial=_timed_batch(service,[p],[problem_request(p,stage='maxmin')],
                       'maxmin','test',1,require_direct_dual=True)
    assert not trial['all_original_certificates_passed']
    assert trial['rows'][0]['independent_original_certificate']['certificate_passed']


def test_cpu_worker_sweep_preserves_service_options_and_certificate_path():
    FakeCpuService.instances=[]
    result=benchmark([_problem_fixture(),_problem_fixture(2.1)],'maxmin',
        workers=CPU_WORKER_CHOICES,repeats=1,service_factory=FakeCpuService)
    assert CPU_WORKER_CHOICES==(1,2,4,8,10,16)
    assert [s.configuration['workers'] for s in FakeCpuService.instances]==list(CPU_WORKER_CHOICES)
    assert all(s.configuration['reuse_basis'] is True and
        s.configuration['max_numerical_retries']==2 and s.closed
        for s in FakeCpuService.instances)
    assert [group['workers'] for group in result['groups']]==list(CPU_WORKER_CHOICES)
    assert result['status']=='all_cpu_trials_original_certified'
    assert result['logical_lp_requests']==2*len(CPU_WORKER_CHOICES)
    assert result['actual_cpu_optimizer_runs']==4*len(CPU_WORKER_CHOICES)


def test_independent_gate_checks_returned_x_not_only_internal_model_solution():
    class WrongReturnedX(FakeCpuService):
        def solve_batch(self,*args,**kwargs):
            results=super().solve_batch(*args,**kwargs)
            for result in results:
                result.x=result.x+1.
                result.fun=-float(result.x[0])
            return results

    result=benchmark([_problem_fixture()],'exchange',workers=(1,),repeats=1,
                     service_factory=WrongReturnedX)
    row=result['groups'][0]['trials'][0]['cold']['rows'][0]
    assert row['objective_consistent']
    assert not row['qualified']
    assert row['independent_original_certificate']['primal_residual']==1.


def test_attempt_counter_disagreement_is_rejected():
    row=dict(cpu_solver_runs=1,numerical_retry_count=0,
             solver_attempts=[dict(solver_run=True)])
    history=[dict(batch=1,cpu_solver_runs=1,numerical_retry_count=0,rows=[row])]
    assert history_counts(history)['actual_cpu_optimizer_runs']==1
    for where in ('row','batch'):
        changed=copy.deepcopy(history)
        (changed[0]['rows'][0] if where=='row' else changed[0])['cpu_solver_runs']=2
        with pytest.raises(ValueError,match='counters'):history_counts(changed)


def test_exclusive_json_never_overwrites_and_marks_nonfinite_as_null(tmp_path):
    path=tmp_path/'benchmark.json'
    record=dict(qualified=False,primal=np.inf,finite=1.,nested=[np.nan,np.int64(2)])
    write_exclusive(path,record)
    before=path.read_bytes()
    assert json.loads(before)==dict(qualified=False,primal=None,finite=1.,nested=[None,2])
    with pytest.raises(FileExistsError):write_exclusive(path,dict(replacement=True))
    assert path.read_bytes()==before


@pytest.mark.parametrize('kwargs',[dict(repeats=0),dict(repeats=True),
    dict(hot_repeats=-1),dict(workers=(1,1)),dict(workers=(3,)),dict(workers=(32,)),
    dict(workers=(True,))])
def test_invalid_bounded_configuration_is_rejected_before_solver_creation(kwargs):
    def forbidden(**_):raise AssertionError('No model should be created')
    with pytest.raises(ValueError):
        benchmark([_problem_fixture()],'exchange',service_factory=forbidden,**kwargs)
