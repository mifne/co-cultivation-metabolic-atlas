"""Small CUDA checks: fixed work, dynamic updates, graph pointers and safety."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device required", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA device required", allow_module_level=True)

from src.gpu_pdhg_corrector import GpuPdhgCorrector
from src.gpu_pdhg_workspace import GpuPdhgWorkspace


def _problem(scale=1., *, rhs=1., upper=2., cost=-1.):
    return (csr_matrix(np.array([[scale, -1.], [1., 1.]], dtype=np.float64)),
            np.array([0., rhs]), np.array([0., 0.]), np.array([upper, upper]),
            np.array([cost, -.25]), 1)


def _workspace(solver=None, **kwargs):
    solver = solver or GpuPdhgCorrector([_problem(), _problem(rhs=1.4)])
    args = dict(batch_size=solver.batch, n_variables=solver.n,
                n_constraints=solver.m, rhs=solver.rhs, c=solver.c,
                lower=solver.lower, upper=solver.upper, tau=solver.tau,
                sigma=solver.sigma, inequality_mask=solver.inequality_mask,
                theta=solver.theta, env_ids=("a", "b")[:solver.batch], chunk_size=8)
    args.update(kwargs)
    return GpuPdhgWorkspace(solver.a, **args), solver


def _cpu_iterations(solver, iterations, x=None, y=None):
    a = solver.a.get()
    rhs, c, lo, hi, tau, sigma, mask = [cp.asnumpy(v) for v in (
        solver.rhs, solver.c, solver.lower, solver.upper, solver.tau, solver.sigma, solver.inequality_mask)]
    x = np.zeros(c.shape) if x is None else np.asarray(x).reshape(c.shape).copy()
    y = np.zeros(rhs.shape) if y is None else np.asarray(y).reshape(rhs.shape).copy()
    x = np.clip(x, lo, hi)
    dual_upper = np.where(mask, 0., np.inf)
    y = np.minimum(y, dual_upper)
    xbar = x.copy()
    for _ in range(iterations):
        y = np.minimum(y + sigma*(rhs-a@xbar), dual_upper)
        new_x = np.clip(x-tau*(c-a.T@y), lo, hi)
        xbar = new_x+solver.theta*(new_x-x)
        x = new_x
    return x.reshape(solver.batch, solver.n), y.reshape(solver.batch, solver.m)


@pytest.mark.parametrize("iterations", [0, 1, 7, 8, 17, 64])
def test_graph_and_loop_match_cpu_operator_with_chunks_and_tail(iterations):
    ws, solver = _workspace()
    ref_x, ref_y = _cpu_iterations(solver, iterations)
    ws.run(iterations)
    graph_x, graph_y = [cp.asnumpy(v) for v in ws.state()]
    assert ws.run_stats["graph_chunks"] == iterations//8
    assert ws.run_stats["python_tail_iterations"] == iterations % 8
    ws.reset()
    ws.run(iterations, use_graph=False)
    loop_x, loop_y = [cp.asnumpy(v) for v in ws.state()]
    np.testing.assert_array_equal(graph_x, loop_x)
    np.testing.assert_array_equal(graph_y, loop_y)
    np.testing.assert_allclose(graph_x, ref_x, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(graph_y, ref_y, atol=1e-12, rtol=1e-12)
    assert ws.setup_timing["kernels_per_iteration"] == 4


def test_warm_starts_are_projected_and_match_cpu_operator():
    ws, solver = _workspace()
    x = np.array([[-2., 9.], [.5, .7]])
    y = np.array([[3., 4.], [-1., -.3]])
    ws.reset(x, y)
    ws.run(27)
    expected = _cpu_iterations(solver, 27, x, y)
    for actual, reference in zip(ws.state(), expected):
        np.testing.assert_allclose(cp.asnumpy(actual), reference, atol=1e-12, rtol=1e-12)


def test_dynamic_values_vectors_metrics_keep_addresses_and_captured_graph():
    ws, _ = _workspace()
    old_graph = ws.graph
    names = ("rhs", "c", "lower", "upper", "tau", "sigma", "theta", "dual_upper", "x", "y")
    addresses = {name: getattr(ws, name).data.ptr for name in names}
    matrix_addresses = (ws.a.data.data.ptr, ws.at.data.data.ptr)
    updated = GpuPdhgCorrector([_problem(.7, rhs=.8, upper=.9, cost=-.6),
                               _problem(1.3, rhs=1.8, upper=1.4, cost=-.4)], theta=.5, primal_weight=2.)
    ws.update_problem(a=updated.a, rhs=updated.rhs, c=updated.c,
                      lower=updated.lower, upper=updated.upper, tau=updated.tau,
                      sigma=updated.sigma, inequality_mask=updated.inequality_mask,
                      theta=updated.theta, env_ids=iter(["a", "b"]))
    assert ws.graph is old_graph
    assert addresses == {name: getattr(ws, name).data.ptr for name in names}
    assert matrix_addresses == (ws.a.data.data.ptr, ws.at.data.data.ptr)
    np.testing.assert_array_equal(ws.at.get().toarray(), updated.a.get().toarray().T)
    with pytest.raises(RuntimeError, match="reset"):
        ws.run(1)
    ws.reset()
    ws.run(49)
    for actual, expected in zip(ws.state(), _cpu_iterations(updated, 49)):
        np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-12, rtol=1e-12)
    assert ws.update_timing["matrix_values_updated"]


def test_accepted_environments_freeze_and_reset_clears_mask():
    ws, _ = _workspace()
    ws.run(3)
    before = [cp.asnumpy(value) for value in ws.state()]
    ws.set_accepted(cp.array([True, False]))
    ws.run(19)
    after = [cp.asnumpy(value) for value in ws.state()]
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left[0], right[0])
    assert not np.array_equal(before[0][1], after[0][1])
    with pytest.raises(ValueError, match="unfreeze"):
        ws.set_accepted(np.array([False, False]))
    ws.set_accepted(np.array([True, True]))
    all_frozen = [v.copy() for v in ws.state()]
    ws.run(8)
    for expected, actual in zip(all_frozen, ws.state()):
        np.testing.assert_array_equal(cp.asnumpy(expected), cp.asnumpy(actual))
    ws.reset()
    assert cp.asnumpy(ws.accepted).tolist() == [False, False]


@pytest.mark.parametrize("bad", [["b", "a"], ["a", "a"], [True, "b"], ["a"]])
def test_env_id_changes_fail_closed_without_mutating_problem(bad):
    ws, _ = _workspace()
    old = cp.asnumpy(ws.rhs)
    with pytest.raises(ValueError, match="env_ids"):
        ws.update_problem(rhs=np.array([0., 9., 0., 9.]), env_ids=bad)
    np.testing.assert_array_equal(cp.asnumpy(ws.rhs), old)
    ws.run(1)


def test_pattern_changes_and_cross_environment_edges_fail_closed():
    ws, solver = _workspace()
    changed = solver.a.copy()
    changed.indices[1] = 0  # duplicate column is not canonical
    old = cp.asnumpy(ws.a.data)
    with pytest.raises(ValueError, match="sorted, unique"):
        ws.update_problem(a=changed, tau=solver.tau, sigma=solver.sigma)
    np.testing.assert_array_equal(cp.asnumpy(ws.a.data), old)
    changed = solver.a.copy()
    changed.indices[1] = 2
    with pytest.raises(ValueError, match="couple"):
        ws.update_problem(a=changed, tau=solver.tau, sigma=solver.sigma)
    from cupyx.scipy.sparse import csr_matrix as gpu_csr
    changed = gpu_csr(csr_matrix(np.eye(4)))
    with pytest.raises(ValueError, match="pattern"):
        ws.update_problem(a=changed, tau=solver.tau, sigma=solver.sigma)


@pytest.mark.parametrize("field,value", [
    ("rhs", [0., np.nan, 0., 1.]), ("c", [0., np.inf, 0., 1.]),
    ("lower", [0., np.nan, 0., 1.]), ("upper", [0., -np.inf, 0., 1.]),
    ("tau", [0., 1., 1., 1.]), ("sigma", [-1., 1., 1., 1.]),
    ("inequality_mask", [0, 1, 0, 1]), ("rhs", [0., 1.]),
])
def test_invalid_updates_do_not_partially_mutate(field, value):
    ws, _ = _workspace()
    before = cp.asnumpy(ws.c)
    update = dict(c=np.array([-5., -5., -5., -5.]))
    update[field] = value
    with pytest.raises(ValueError):
        ws.update_problem(**update)
    np.testing.assert_array_equal(cp.asnumpy(ws.c), before)
    ws.run(1)


def test_mask_shape_and_warm_start_fail_before_state_mutation():
    ws, _ = _workspace()
    before = [cp.asnumpy(value) for value in ws.state()]
    with pytest.raises(ValueError, match="y"):
        ws.reset(np.ones(4), np.full(4, np.nan))
    for old, new in zip(before, ws.state()):
        np.testing.assert_array_equal(old, cp.asnumpy(new))
    with pytest.raises(ValueError, match="shape"):
        ws.set_accepted(np.array([True]))


def test_matrix_update_requires_new_metric_and_owns_input_buffers():
    ws, solver = _workspace()
    original = cp.asnumpy(ws.rhs)
    solver.rhs.fill(99.)
    np.testing.assert_array_equal(cp.asnumpy(ws.rhs), original)
    with pytest.raises(ValueError, match="tau and sigma"):
        ws.update_problem(a=solver.a)


def test_zero_row_box_problem_graph_runs_without_cpu_optimizer(monkeypatch):
    import highspy
    import scipy.optimize
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU optimizer called")
    monkeypatch.setattr(highspy.Highs, "run", forbidden)
    monkeypatch.setattr(scipy.optimize, "linprog", forbidden)
    problem = (csr_matrix((0, 1)), np.empty(0), np.array([0.]), np.array([2.]), np.array([-1.]), 0)
    solver = GpuPdhgCorrector([problem])
    ws, _ = _workspace(solver)
    ws.run(17)
    x, y = ws.state()
    assert y.shape == (1, 0)
    np.testing.assert_array_equal(cp.asnumpy(x), [[2.]])
    assert solver._certificate(x, y)[0]["certificate_passed"]


def test_workspace_matches_existing_cusparse_iterates_and_original_certificates():
    ws, solver = _workspace()
    # Nonoptimal zero-budget/cold start, no early accept before equal budget.
    reference = solver.solve(iterations=17, check_interval=17)
    assert reference["iterations_run"] == 17
    ws.run(17)
    x, y = ws.state()
    np.testing.assert_allclose(cp.asnumpy(x), cp.asnumpy(reference["x"]), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(y), cp.asnumpy(reference["y"]), atol=1e-12, rtol=1e-12)
    metrics = solver._certificate(x, y)
    for expected, actual in zip(reference["metrics"], metrics):
        for name in ("primal_residual", "dual_violation", "relative_kkt_gap", "objective"):
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=1e-10)
        assert actual["certificate_passed"] == expected["certificate_passed"]


def test_explicit_loop_only_and_budget_validation():
    ws, _ = _workspace(use_graph=False)
    assert ws.graph is None
    ws.run(3)
    assert ws.run_stats["graph_chunks"] == 0
    with pytest.raises(RuntimeError, match="no captured graph"):
        ws.run(8, use_graph=True)
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="iterations"):
            ws.run(value)


@pytest.mark.parametrize("name", ["rhs", "c", "lower", "upper", "tau", "sigma", "inequality_mask", "theta"])
def test_constructor_rejects_missing_initial_numeric_buffers(name):
    with pytest.raises(ValueError, match="required"):
        _workspace(**{name: None})


def test_fused_updates_preserve_nan_instead_of_clipping_it_to_a_valid_endpoint():
    ws, _ = _workspace()
    with ws.stream:
        nan = cp.full(4, cp.nan)
        zeros = cp.zeros(4)
        ones = cp.ones(4)
        done = cp.zeros(4, dtype=cp.bool_)
        y, next_y = cp.empty(4), cp.empty(4)
        x, xbar = cp.empty(4), cp.empty(4)
        ws._row_update(zeros, zeros, nan, ones, zeros, done, y, next_y)
        ws._col_update(zeros, nan, zeros, ones, zeros, ones, ws.theta, done, x, xbar)
    ws.synchronize()
    for value in (y, next_y, x, xbar):
        assert bool(cp.isnan(value).all())
