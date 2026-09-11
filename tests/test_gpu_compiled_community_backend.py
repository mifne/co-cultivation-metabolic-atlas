import numpy as np
import pytest
from src.gpu_certified_basis import NormalizedLP,compile_basis
from src.gpu_basis_bank import GpuBasisBank
from src.gpu_compiled_community_backend import GpuCompiledCommunityBackend,stage_key
from tests.test_gpu_revised_basis import problem,cpu


class IdentityCoordinates:
    n_fluxes=1
    def normalize(self,a,rhs,lower,upper,c,neq):
        return NormalizedLP(a,rhs,lower,upper,c,neq,np.ones(a.shape[1]),np.ones(a.shape[0]))


def backend(repair_pivots=0):
    p=problem()
    a=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    bank=GpuBasisBank([a],[0])
    key=stage_key(p.c,p.a,p.neq,1)
    return GpuCompiledCommunityBackend(IdentityCoordinates(),{key:bank},repair_pivots=repair_pivots)


def solve(b,p):
    return b.solve(p.c,A_ub=p.a,b_ub=p.rhs,bounds=list(zip(p.lower,p.upper)))


def test_host_adapter_rejects_bank_miss_without_substitution():
    pytest.importorskip("cupy")
    b=backend()
    out=solve(b,problem(rhs=(.1,3.)))
    assert not out.success and out.x is None and out.fun is None
    assert b.history[-1]["status"]=="bank_miss_no_cpu_fallback"
    assert b.history[-1]["best_rejected_metrics"]["primal_residual"]>1e-5


def test_explicit_gpu_repair_is_instrumented_and_does_not_call_cpu(monkeypatch):
    pytest.importorskip("cupy")
    import highspy
    b=backend(repair_pivots=12)
    p=problem(rhs=(.1,3.))
    expected=cpu(p).fun
    def forbidden(*args,**kwargs):raise AssertionError("CPU optimizer called online")
    monkeypatch.setattr(highspy.Highs,"run",forbidden)
    out=solve(b,p)
    assert out.success
    np.testing.assert_allclose(out.fun,expected,atol=1e-8)
    record=b.history[-1]
    assert record["repair_pivots"]>0
    assert record["cpu_lp_calls"]==record["repair_cpu_lp_calls"]==0
    assert record["max_original_residual"]<=1e-5
