import numpy as np
import pytest
from src.gpu_certified_basis import compile_basis
from src.gpu_basis_bank import GpuBasisBank
from tests.test_gpu_revised_basis import problem,cpu


def test_selects_different_bases_and_rejects_uncovered_region(monkeypatch):
    pytest.importorskip("cupy")
    import highspy
    anchors=[problem(),problem(rhs=(.1,3.))]
    compiled=[compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq) for p in anchors]
    bank=GpuBasisBank(compiled,[0])
    queries=[problem(rhs=(5.2,3.)),problem(rhs=(.15,3.)),problem(cost=(2.,1.))]
    expected=[cpu(p).fun for p in queries[:2]]
    def forbidden(*args,**kwargs): raise AssertionError("Online CPU optimizer forbidden")
    monkeypatch.setattr(highspy.Highs,"run",forbidden)
    result=bank.evaluate_device(**bank.prepare_host(queries))
    assert result["accepted"].get().tolist()==[True,True,False]
    assert result["candidate_index"].get().tolist()==[0,1,-1]
    np.testing.assert_allclose(result["objective"].get()[:2],expected,atol=1e-8)
    assert np.isnan(result["values"].get()[2]).all()
    assert result["cpu_lp_calls"]==0


def test_matrix_offsets_are_relative_to_each_anchor():
    pytest.importorskip("cupy")
    anchors=[problem(),problem(rhs=(.1,3.),coef=1.8)]
    compiled=[compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq) for p in anchors]
    bank=GpuBasisBank(compiled,[0])
    query=problem(rhs=(.2,3.),coef=1.5)
    result=bank.evaluate_device(**bank.prepare_host([query]))
    assert result["accepted"].get()[0]
    np.testing.assert_allclose(result["objective"].get()[0],cpu(query).fun,atol=1e-8)
