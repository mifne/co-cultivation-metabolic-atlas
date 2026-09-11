"""Coordinator lifecycle tests use tiny fake captures, never claim solver evidence."""
import json
import sys
from types import SimpleNamespace
import pytest
from scripts import collect_graph_learning_curve as collect


def arguments(monkeypatch,root,*extra):
    monkeypatch.setattr(sys,'argv',['collect','--output',str(root),'--count','8','--workers','1',*extra])


def fake_capture(command,**kwargs):
    from pathlib import Path
    dest=Path(command[command.index('--output')+1]);dest.mkdir()
    seed=int(command[command.index('--seed')+1])
    (dest/'manifest.json').write_text(json.dumps(dict(status='completed',model_fingerprints={'toy':'fixed'},
        seeds=list(range(seed,seed+8)),entries=[{}]*24)))
    return SimpleNamespace(returncode=0)


def test_resume_does_not_recompute_completed_shard(tmp_path,monkeypatch):
    arguments(monkeypatch,tmp_path);calls=[]
    def run(command,**kw):calls.append(command);return fake_capture(command,**kw)
    monkeypatch.setattr(collect.subprocess,'run',run)
    collect.main();collect.main()
    assert len(calls)==1
    job=json.loads((tmp_path/'catalog.json').read_text())['requested_job']
    assert job['status']=='completed' and job['training_queued'] is False


def test_failed_retry_preserves_original_and_seeds(tmp_path,monkeypatch):
    arguments(monkeypatch,tmp_path)
    def fail(command,**kw):
        fake_capture(command,**kw)
        return SimpleNamespace(returncode=-6)
    monkeypatch.setattr(collect.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='Failed shard'):collect.main()
    assert json.loads((tmp_path/'catalog.json').read_text())['requested_job']['status']=='failed'
    original=(tmp_path/'train_000000_000008'/'manifest.json').read_bytes()
    with pytest.raises(RuntimeError,match='Incomplete shard'):collect.main()
    arguments(monkeypatch,tmp_path,'--retry-failed')
    monkeypatch.setattr(collect.subprocess,'run',fake_capture);collect.main()
    c=json.loads((tmp_path/'catalog.json').read_text())
    assert c['failed_attempts'][0]['status']=='failed'
    assert c['shards'][0]['path']=='train_000000_000008_retry1'
    assert c['shards'][0]['seeds']==list(range(71000000,71000008))
    assert (tmp_path/'train_000000_000008'/'manifest.json').read_bytes()==original


def test_locked_catalog_is_not_mutated(tmp_path,monkeypatch):
    import fcntl
    arguments(monkeypatch,tmp_path)
    with (tmp_path/'catalog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='Another collector'):collect.main()
    assert not (tmp_path/'catalog.json').exists()


def test_32_trajectory_job_stops_at_requested_count(tmp_path,monkeypatch):
    arguments(monkeypatch,tmp_path,'--count','32');calls=[]
    def run(command,**kw):
        calls.append(command)
        job=json.loads((tmp_path/'catalog.json').read_text())['requested_job']
        assert job['count']==32 and len(job['batches'])==4
        assert job['training_queued'] is False
        return fake_capture(command,**kw)
    monkeypatch.setattr(collect.subprocess,'run',run);collect.main()
    assert len(calls)==4
    job=json.loads((tmp_path/'catalog.json').read_text())['requested_job']
    assert job['status']=='completed' and all(b['status']=='completed' for b in job['batches'])


def test_cold_teacher_rebuilds_and_keeps_certified_payload(tmp_path,monkeypatch):
    import numpy as np
    from scipy.sparse import csr_matrix
    from src.cpu_repeated_lp import RepeatedCpuLP
    from scripts.capture_dfba_lp_trace import CaptureBackend
    cpu=RepeatedCpuLP(1);manifest={'entries':[]};calls=[]
    original=cpu.clear_models
    def clear():calls.append(True);original()
    monkeypatch.setattr(cpu,'clear_models',clear)
    capture=CaptureBackend(cpu,tmp_path,manifest,lambda:None,cold_teacher=True)
    try:
        for stage in ('maxmin','aggregate','exchange'):
            result=capture.solve_batch([(np.array([1.]),dict(A_eq=csr_matrix([[1.]]),b_eq=np.array([1.]),
                bounds=np.array([[0.,2.]]),method='highs-ds',_stage=stage))])
            assert result[0].success
        assert len(calls)==3 and len(manifest['entries'])==3
    finally:cpu.close()


def test_scipy_teacher_payload_roundtrip(tmp_path):
    import numpy as np
    from scipy.sparse import csr_matrix
    from src.offline_scipy_teacher import OfflineScipyTeacher
    from scripts.capture_dfba_lp_trace import CaptureBackend
    cpu=OfflineScipyTeacher();manifest={'entries':[]}
    capture=CaptureBackend(cpu,tmp_path,manifest,lambda:None)
    for stage in ('maxmin','aggregate','exchange'):
        result=capture.solve_batch([(np.array([-1.]),dict(A_ub=csr_matrix([[1.]]),b_ub=np.array([1.]),
            bounds=np.array([[0.,2.]]),method='highs-ds',_stage=stage))])
        assert result[0].success and result[0].x[0]==pytest.approx(1.)
    assert len(manifest['entries'])==3
    cpu.close()
