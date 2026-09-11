from copy import deepcopy
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from src.gpu_certified_basis import NormalizedLP, compile_basis, GpuBasisEvaluator


def toy(rhs=(5.,3.), coef=1., cost=(-2.,-1.)):
    return NormalizedLP(csr_matrix([[coef,1.],[1.,0.]]),np.array(rhs),np.zeros(2),np.full(2,10.),
        np.array(cost),0,np.ones(2),np.ones(2))


def compile_toy():
    p=toy()
    return compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)


def test_device_basis_updates_rhs_and_matrix_rows_without_cpu_solve(monkeypatch):
    pytest.importorskip("cupy")
    import highspy
    anchor=compile_toy()
    evaluator=GpuBasisEvaluator(anchor,[0])
    problems=[toy(),toy((5.1,3.)),toy((5.,3.),coef=1.05)]
    expected=[linprog(p.c,A_ub=p.a,b_ub=p.rhs,bounds=list(zip(p.lower,p.upper)),method="highs-ds").fun for p in problems]
    def forbidden(*args,**kwargs): raise AssertionError("Online CPU solver forbidden")
    monkeypatch.setattr(highspy.Highs,"run",forbidden)
    data=evaluator.prepare_host(problems)
    result=evaluator.evaluate_device(**data)
    assert result["accepted"].get().all()
    np.testing.assert_allclose(result["objective"].get(),expected,atol=1e-8)
    assert result["cpu_lp_calls"]==0


def test_invalid_basis_never_returns_an_accepted_flux():
    pytest.importorskip("cupy")
    evaluator=GpuBasisEvaluator(compile_toy(),[0])
    problems=[toy((.1,3.)),toy(cost=(2.,1.))]
    result=evaluator.evaluate_device(**evaluator.prepare_host(problems))
    assert not result["accepted"].get().any()
    assert np.isnan(result["values"].get()).all()


def test_unknown_matrix_change_is_rejected():
    pytest.importorskip("cupy")
    evaluator=GpuBasisEvaluator(compile_toy(),[0])
    p=toy(); p.a[1,1]=.1
    with pytest.raises(ValueError,match="outside"):
        evaluator.prepare_host([p])


def test_nonfinite_batch_member_does_not_become_valid():
    cp=pytest.importorskip("cupy")
    evaluator=GpuBasisEvaluator(compile_toy(),[0])
    data=evaluator.prepare_host([toy(),toy()])
    data["c"][1,0]=cp.nan
    result=evaluator.evaluate_device(**data)
    assert result["accepted"].get().tolist()==[True,False]


def test_deferred_factorization_matches_direct_compile():
    from src.gpu_certified_basis import factor_compiled_basis
    p=toy()
    deferred=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq,factorize=False)
    assert "inverse" not in deferred
    factored=factor_compiled_basis(deferred)
    np.testing.assert_allclose(factored["inverse"].toarray(),compile_toy()["inverse"].toarray())
    assert factored["offline_cpu_lp_calls"]==1


def test_same_algorithm_cpu_comparator_matches_gpu():
    pytest.importorskip("cupy")
    from scripts.cpu_basis_comparator import CpuBasisComparator
    anchor=compile_toy()
    gpu=GpuBasisEvaluator(anchor,[0]); cpu=CpuBasisComparator(anchor,[0])
    problems=[toy(),toy((5.1,3.)),toy((.1,3.))]
    g=gpu.evaluate_device(**gpu.prepare_host(problems))
    c=cpu.evaluate_host(**cpu.prepare_host(problems))
    np.testing.assert_array_equal(g["accepted"].get(),c["accepted"])
    np.testing.assert_allclose(g["values"].get(),c["values"],equal_nan=True,atol=1e-8)
