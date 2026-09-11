from types import SimpleNamespace

import numpy as np
import pytest


def test_device_engine_inherits_fail_closed_gpu_repair():
    from src.gpu_device_bounded_simplex import GpuDeviceBoundedSimplex
    from src.gpu_bounded_simplex import GpuBoundedSimplex
    assert GpuDeviceBoundedSimplex.solve is GpuBoundedSimplex.solve
    engine = object.__new__(GpuDeviceBoundedSimplex)
    engine.time_limit = 10.
    engine.history = []
    engine.basis_cache = {}
    engine.max_iterations = 20
    engine.log_to_console = False

    def attempt(c, **kwargs):
        success = bool(engine.history)
        if success:
            assert engine.basis_cache["growth"] == {"basis": ["v0"], "upper": []}
        engine.numerical_repair_basis = {"basis": ["v0"], "upper": []}
        engine.history.append(dict(status="optimal", success=success, min_pivot=1.,
            host_qr_crash_seconds=0., iterations=1, phase_iterations=[1, 0],
            max_original_residual=0. if success else .1))
        return SimpleNamespace(success=success, x=np.array([1.]) if success else None)

    engine._solve_once = attempt
    result = engine.solve([-1.], stage_key="growth")
    assert result.success
    assert engine.history[-1]["gpu_lp_attempts"] == 2
    assert engine.history[-1]["gpu_primal_repairs"] == 1
    assert engine.time_limit == 10.


def test_captured_batch_respects_remaining_iteration_budget():
    cp = pytest.importorskip("cupy")
    from src.gpu_simplex_pivot_batch import PivotBatch
    tableau = cp.asarray([[1., 1., 1., 0.], [1., 0., 0., 1.]])
    x = cp.asarray([0., 0., 5., 3.])
    basis = cp.asarray([2, 3], dtype=cp.int64)
    reduced = cp.asarray([-2., -1., 0., 0.])
    limits = cp.asarray([cp.inf, 4., cp.inf, cp.inf])
    batch = PivotBatch(tableau, x, basis, reduced, limits)
    batch.refresh(tableau, x, basis, reduced, limits, max_pivots=1)
    state, _ = batch.run()
    assert state[1] == 1
    assert state[0] == 3
    np.testing.assert_allclose(tableau.get() @ batch.x.get(), [5., 3.], atol=1e-8)
    with pytest.raises(ValueError):
        batch.refresh(tableau, x, basis, reduced, limits, max_pivots=-1)


@pytest.mark.parametrize("case", ["bounded", "free", "infeasible", "limited"])
def test_full_device_edge_cases(case):
    pytest.importorskip("cupy")
    from scipy.optimize import linprog
    from src.gpu_device_bounded_simplex import GpuDeviceBoundedSimplex
    options = dict(A_eq=[[1., 1.]], b_eq=[1.], bounds=[(-2., 3.), (None, None)])
    c = [1., 2.]
    if case == "bounded":
        options["bounds"] = [(0., 3.), (0., 2.)]
    if case == "infeasible":
        options["bounds"] = [(2., 3.), (2., 3.)]
    cpu = linprog(c, method="highs-ds", **options)
    engine = GpuDeviceBoundedSimplex(max_iterations=0 if case == "limited" else 1000)
    gpu = engine.solve(c, **options)
    if case in {"infeasible", "limited"}:
        assert not gpu.success
        assert gpu.x is None
    else:
        assert cpu.success and gpu.success, gpu.message
        assert abs(gpu.fun-cpu.fun) < 1e-7
        assert engine.history[-1]["max_original_residual"] <= 1e-5
    assert engine.history[-1]["cpu_lp_calls"] == 0
