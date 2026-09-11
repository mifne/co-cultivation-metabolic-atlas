"""Retry acceptance must remain the original LP certificate, not solver status."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
import src.offline_scipy_teacher as teacher


def request():
    return np.array([-1.]),dict(A_ub=csr_matrix([[1.]]),b_ub=np.array([1.]),
        bounds=np.array([[0.,2.]]),method='highs-ds',_stage='maxmin')


def test_no_extra_solve_for_qualified_solution():
    cpu=teacher.OfflineScipyTeacher();r=cpu.solve_batch([request()])[0]
    assert r.diagnostics['cpu_solver_runs']==1
    assert r.diagnostics['relative_kkt_gap']<=1e-7


@pytest.mark.parametrize('failures',[1,2,3])
def test_rejected_gap_triggers_bounded_alternative_solve(monkeypatch,failures):
    original=teacher.linprog;calls=[]
    def solve(*args,**kwargs):
        calls.append(kwargs);r=original(*args,**kwargs)
        if len(calls)<=failures:r.x=r.x-1e-6
        return r
    monkeypatch.setattr(teacher,'linprog',solve)
    r=teacher.OfflineScipyTeacher().solve_batch([request()])[0]
    assert len(calls)==failures+1 and r.diagnostics['numerical_retry_count']==failures
    assert r.diagnostics['relative_kkt_gap']<=1e-7
    assert r.diagnostics['attempts'][0]['certificate']['relative_kkt_gap']>1e-7
    assert calls[1]['options']['dual_feasibility_tolerance']==1e-10
    if failures>=2:assert calls[2]['options']['presolve'] is False
    if failures==3:assert calls[3]['method']=='highs-ipm'


def test_all_rejected_fails_closed_no_model_admitted(monkeypatch):
    original=teacher.linprog;calls=[]
    def solve(*args,**kwargs):
        calls.append(True);r=original(*args,**kwargs);r.x-=1e-6;return r
    monkeypatch.setattr(teacher,'linprog',solve);cpu=teacher.OfflineScipyTeacher()
    with pytest.raises(RuntimeError,match='exhausted 4 attempts'):cpu.solve_batch([request()])
    assert len(calls)==4 and not cpu.models


def test_nonfinite_solution_is_never_accepted(monkeypatch):
    original=teacher.linprog
    def solve(*args,**kwargs):
        r=original(*args,**kwargs);r.x[:]=np.nan;return r
    monkeypatch.setattr(teacher,'linprog',solve)
    with pytest.raises(RuntimeError,match='nonfinite'):teacher.OfflineScipyTeacher().solve_batch([request()])
