"""Synthetic CPU stand-in checks policy/routing, not GPU accuracy evidence."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend


class CpuTestStandIn:
    def __init__(self, fail_second=False):
        self.history=[]
        self.fail_second=fail_second
    def solve(self,c,**kwargs):
        if self.fail_second and len(self.history)==1:
            result=SimpleNamespace(success=False,x=None,fun=None,message="Injected test failure")
        else: result=linprog(c,**kwargs)
        self.history.append(dict(success=result.success,objective=result.fun,
            max_original_residual=0.,total_seconds=.01))
        return result


def problem():
    layout=SimpleNamespace(n_fluxes=2,_exchange_terms={"h2o_e":[("a",0,-1,"water")],
        "val__L_e":[("a",1,-1,"valine")]})
    c=np.array([0.,0.,0.,1.,1.])
    kw=dict(A_eq=csr_matrix([[1.,1.,0.,0.,0.]]),b_eq=[1.],
        A_ub=csr_matrix([[1,0,0,-1,0],[-1,0,0,-1,0],[0,1,0,0,-1],[0,-1,0,0,-1]]),
        b_ub=np.zeros(4),bounds=[(0,1),(0,1),(0,0),(0,None),(0,None)])
    return layout,c,kw


def test_keeps_original_objective_and_counts_extra_lp():
    layout,c,kw=problem()
    original=linprog(c,**kw)
    engine=GpuExchangeTieBreakBackend(layout,inner=CpuTestStandIn())
    selected=engine.solve(c,**kw)
    assert selected.success
    assert abs(selected.fun-original.fun)<1e-8
    assert selected.x[1]<1e-8
    assert selected.x[0]>1-1e-8
    assert engine.actual_gpu_lp_calls==2 and engine.tie_lp_calls==1
    assert engine.history[0]["actual_gpu_lp_calls"]==2


def test_no_earlier_solution_substitution_after_tie_failure():
    layout,c,kw=problem()
    engine=GpuExchangeTieBreakBackend(layout,inner=CpuTestStandIn(fail_second=True))
    selected=engine.solve(c,**kw)
    assert not selected.success and selected.x is None
    assert not engine.history[0]["success"]


def test_other_objectives_not_replaced():
    layout,c,kw=problem()
    engine=GpuExchangeTieBreakBackend(layout,inner=CpuTestStandIn())
    c[0]=-1.
    selected=engine.solve(c,**kw)
    assert selected.success
    assert engine.actual_gpu_lp_calls==1 and engine.tie_lp_calls==0
