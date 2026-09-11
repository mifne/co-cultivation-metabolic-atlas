"""Unchanged LP stage routing, no hidden fallback, and optional real GPU tests."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import linprog as reference_linprog
from scipy.sparse import csr_matrix
from src.community_solver import CooperativeCommunityFbaSolver
from src.gpu_lexicographic_lp import CuOptLexicographicBackend, PresolvedCuOptBackend
from tests.test_community_solver import _crossfeeding_models


class RecordingBackend:
    name, method = "test_only_recording", "test_only_reference"
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail
    def solve(self, c, **kw):
        self.calls.append((c.copy(), kw))
        if len(self.calls) == self.fail:
            return SimpleNamespace(success=False, x=None, message="injected failure")
        return reference_linprog(c, **kw)


@pytest.mark.parametrize("fail", [None, 2, 3])
def test_same_three_stages_and_fail_closed(monkeypatch, fail):
    models = _crossfeeding_models()
    args = dict(models=models, biomass_g_l={"producer":1., "consumer":1.},
                medium_mmol_l={"carbon_e":10., "factor_e":0.}, dt=1.)
    cpu = CooperativeCommunityFbaSolver(models, optimize_live_objectives=True)
    exact = cpu.solve(**args)
    backend = RecordingBackend(fail)
    solver = CooperativeCommunityFbaSolver(models, optimize_live_objectives=True, linear_program_backend=backend)
    def forbidden(*a, **kw):
        raise AssertionError("unexpected CPU fallback")
    monkeypatch.setattr("src.community_solver.linprog", forbidden)
    result = solver.solve(**args)
    assert solver.cpu_lp_stage_calls == solver.cpu_cooperative_solve_calls == 0
    assert solver.gpu_lp_stage_calls == (3 if fail is None else fail)
    assert backend.calls[0][0][-1] == -1
    if fail is None:
        for species in models:
            np.testing.assert_allclose(result[species].fluxes, exact[species].fluxes)
    else:
        assert all(value is None for value in result.values())
        assert "failed" in solver.stats.status


@pytest.mark.parametrize("method", ["concurrent", "dual_simplex", "cpu"])
def test_cpu_methods_rejected_before_import(method):
    with pytest.raises(ValueError, match="GPU-only"):
        CuOptLexicographicBackend(method=method)


def test_exact_backend_conflicts_with_surrogate():
    with pytest.raises(ValueError, match="cannot be combined"):
        CooperativeCommunityFbaSolver(_crossfeeding_models(), linear_program_backend=RecordingBackend(),
                                      gpu_qp_projection=True)


@pytest.mark.parametrize("method", ["barrier", "pdlp"])
def test_real_gpu_optimum_and_cached_updates(method):
    pytest.importorskip("cuopt")
    backend = CuOptLexicographicBackend(method=method, time_limit=10, tolerance=1e-9, presolve=0)
    for rhs in (5., 4.):
        result = backend.solve(np.array([-2., -1.]), A_ub=csr_matrix([[1., 1.]]),
                               b_ub=np.array([rhs]), bounds=[(0,3), (0,4)])
        assert result.success, result.message
        np.testing.assert_allclose(result.x, [3, rhs-3], atol=1e-5)
        assert abs(result.fun-(-rhs-3)) < 1e-5


def test_host_presolve_forbids_changed_objective():
    with pytest.raises(ValueError, match="quadratic"):
        PresolvedCuOptBackend(quadratic_regularization=1e-6)


def test_real_gpu_algebraic_postsolve_without_cpu_optimizer(monkeypatch):
    pytest.importorskip("cuopt")
    hp = pytest.importorskip("highspy")
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU optimizer must not run")
    monkeypatch.setattr(hp.Highs, "run", forbidden)
    backend = PresolvedCuOptBackend(method="pdlp", tolerance=1e-8, time_limit=10)
    c = np.array([-2.,-3.,-4.])
    kw = dict(A_ub=csr_matrix([[1.,2.,1.],[2.,1.,3.]]), b_ub=np.array([4.,5.]), bounds=[(0,10)]*3)
    exact = reference_linprog(c, **kw)
    result = backend.solve(c, **kw)
    assert result.success, result.message
    assert abs(result.fun-exact.fun) < 1e-5
    assert all(v <= 0 for v in backend.history[-1]["cpu_iteration_counts"].values())


def test_zero_entries_unbounded_and_redundant_row():
    pytest.importorskip("cuopt")
    backend = CuOptLexicographicBackend(method="pdlp", tolerance=1e-8, time_limit=10, presolve=0)
    matrix = csr_matrix(([1.,0.,1.,1.], [0,1,0,1], [0,2,4]), shape=(2,2))
    result = backend.solve([-1.,1.], A_ub=matrix, b_ub=[3.,5.], bounds=[(0,3),(0,None)])
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [3.,0.], atol=1e-5)
    assert backend.history[-1]["redundant_inequalities"] == 1


@pytest.mark.parametrize("seed", range(4))
def test_experimental_gpu_simplex_against_cpu(seed):
    pytest.importorskip("cupy")
    from src.gpu_bounded_simplex import GpuBoundedSimplex
    rng=np.random.default_rng(seed)
    matrix=rng.normal(size=(8,12)); feasible=rng.uniform(-1,1,12)
    eq=matrix[:3]; ub=matrix[3:]
    kw=dict(A_eq=csr_matrix(eq), b_eq=eq@feasible, A_ub=csr_matrix(ub),
            b_ub=ub@feasible+rng.uniform(0,1,5), bounds=[(-2,2)]*12)
    c=rng.normal(size=12)
    cpu=reference_linprog(c,**kw)
    backend=GpuBoundedSimplex()
    gpu=backend.solve(c,**kw)
    assert gpu.success,gpu.message
    assert abs(gpu.fun-cpu.fun)<1e-5


def test_failed_gpu_problem_has_replayable_snapshot():
    pytest.importorskip("cupy")
    from src.gpu_bounded_simplex import GpuBoundedSimplex
    import json
    backend=GpuBoundedSimplex(max_iterations=0)
    result=backend.solve([-2.,-1.],A_ub=csr_matrix([[1.,1.]]),b_ub=[5.],bounds=[(0,3),(0,4)])
    assert not result.success
    d=backend.failure_snapshot
    metadata=json.loads(str(d["metadata_json"]))
    a=csr_matrix((d["data"],d["indices"],d["indptr"]),shape=tuple(d["shape"]))
    metadata.pop("previous")
    replay=GpuBoundedSimplex().solve(d["c"],A_ub=a,b_ub=d["rhs"],
        bounds=list(zip(d["lower"],d["upper"])),**metadata)
    assert replay.success
    np.testing.assert_allclose(replay.x,[3.,2.],atol=1e-6)


@pytest.mark.parametrize("injected_failures",[1,4])
def test_gpu_primal_repair_does_not_accept_invalid_attempt(monkeypatch,injected_failures):
    pytest.importorskip("cupy")
    from src.gpu_bounded_simplex import GpuBoundedSimplex
    backend=GpuBoundedSimplex()
    real_solve=backend._solve_once
    calls=[]
    matrix=csr_matrix([[1.,1.]])
    def inject(c,**kw):
        np.testing.assert_array_equal(c,[-2.,-1.])
        assert kw["A_ub"] is matrix
        calls.append(1)
        result=real_solve(c,**kw)
        assert result.success
        if len(calls)<=injected_failures:
            result.success=False; result.x=None
            backend.history[-1].update(success=False,max_original_residual=1e-4)
            backend.numerical_repair_basis=backend.basis_cache["default"]
            backend.failure_snapshot={"injected":True}
        return result
    monkeypatch.setattr(backend,"_solve_once",inject)
    result=backend.solve([-2.,-1.],A_ub=matrix,b_ub=[5.],bounds=[(0,3),(0,4)])
    assert result.success==(injected_failures==1)
    assert len(calls)==(2 if injected_failures==1 else 4)
    assert len(backend.history)==1
    assert backend.history[0]["gpu_lp_attempts"]==len(calls)
    assert backend.history[0]["worst_attempt_residual"]==1e-4
    assert (backend.failure_snapshot is None)==result.success
    if result.success: np.testing.assert_allclose(result.x,[3.,2.],atol=1e-6)
