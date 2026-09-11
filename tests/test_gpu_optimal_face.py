import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import vstack,csr_matrix
from src.gpu_certified_basis import compile_basis
from src.gpu_revised_basis import GpuRevisedBasis
from src.gpu_optimal_face import solve_on_primary_face
from tests.test_gpu_revised_basis import problem


def test_secondary_face_is_certified_against_original_extra_inequality(monkeypatch):
    cp=pytest.importorskip("cupy")
    import highspy
    p=problem(cost=(-1.,-1.))
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    solver=GpuRevisedBasis(anchor,[0],max_pivots=12)
    data=solver.prepare_host([p,p]);primary=solver.solve_device(**data)
    objectives=np.array([[1.,0.],[0.,1.]])
    expected=[linprog(c,A_ub=vstack((p.a,csr_matrix(p.c[None]))),
        b_ub=np.r_[p.rhs,anchor["cpu_anchor_objective"]+1e-10],bounds=list(zip(p.lower,p.upper)),method="highs-ds").fun for c in objectives]
    def forbidden(*args,**kw):raise AssertionError("CPU LP online")
    monkeypatch.setattr(highspy.Highs,"run",forbidden)
    result=solve_on_primary_face(solver,primary,data,cp.asarray(objectives),cp.array([1e-10,1e-10]))
    assert result["accepted"].get().all()
    np.testing.assert_allclose(result["objective"].get(),expected,atol=1e-7)
    assert result["cpu_lp_calls"]==0


def test_exact_face_candidate_cannot_hide_allowed_loss_objective_error():
    cp=pytest.importorskip("cupy")
    p=problem(cost=(-1.,0.))
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    solver=GpuRevisedBasis(anchor,[0],max_pivots=12)
    data=solver.prepare_host([p]);primary=solver.solve_device(**data)
    result=solve_on_primary_face(solver,primary,data,cp.array([[1.,0.]]),cp.array([.5]))
    # On exact primary face x=3; actual relaxed secondary optimum is x=2.5.
    # Therefore it MUST fail the original-LP gap even though face solve passes.
    assert not result["accepted"].get()[0]
    assert result["relative_kkt_gap"].get()[0]>.1
    assert np.isnan(result["values"].get()).all()


def test_changed_original_bounds_invalidate_primary_face():
    cp=pytest.importorskip("cupy")
    p=problem(cost=(-1.,0.))
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    solver=GpuRevisedBasis(anchor,[0],max_pivots=12)
    data=solver.prepare_host([p]);primary=solver.solve_device(**data)
    changed=dict(data,upper=data["upper"].copy());changed["upper"][0,0]=2.
    result=solve_on_primary_face(solver,primary,changed,cp.array([[1.,0.]]),cp.array([1e-10]))
    assert not result["accepted"].get()[0]
