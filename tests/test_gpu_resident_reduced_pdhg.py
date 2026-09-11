"""Independent tests of dynamic reduced GPU service and original-LP gates."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device required", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA device required", allow_module_level=True)

from src.gpu_reduced_pdhg import GpuReducedPdhgCorrector
from src.gpu_resident_reduced_pdhg import GpuResidentReducedPdhg


def _problem(*, inequality=(1., .5), rhs=.8, lower=(0., -1.),
             upper=(1., 2.), c=(-1., -.3)):
    return (csr_matrix(np.array([[1., -1.], inequality], dtype=np.float64)),
            np.array([0., rhs]), np.array(lower), np.array(upper), np.array(c), 1)


def _service(problems=None, **kwargs):
    problems = problems if problems is not None else [_problem(), _problem(rhs=.9)]
    return GpuResidentReducedPdhg(problems, chunk_size=4, **kwargs)


def _same_result(left, right):
    for name in ("x", "y", "reduced_x", "reduced_y"):
        np.testing.assert_allclose(cp.asnumpy(left[name]), cp.asnumpy(right[name]),
                                   atol=1e-10, rtol=1e-10)
    assert left["accepted"].tolist() == right["accepted"].tolist()
    assert left["iterations_run"] == right["iterations_run"]
    for lrow, rrow in zip(left["metrics"], right["metrics"]):
        for name in ("primal_residual", "dual_violation", "relative_kkt_gap", "objective"):
            np.testing.assert_allclose(lrow[name], rrow[name], atol=1e-10, rtol=1e-10)
        assert lrow["certificate_passed"] == rrow["certificate_passed"]


@pytest.mark.parametrize("iterations", [0, 3, 4, 17])
def test_graph_loop_and_existing_corrector_original_result_equivalence(iterations):
    problems = [_problem(), _problem(rhs=.9)]
    graph, loop = _service(problems), _service(problems, use_graph=False)
    original = GpuReducedPdhgCorrector(problems)
    args = dict(iterations=iterations, check_interval=17)
    result = graph.solve(**args)
    _same_result(result, loop.solve(**args))
    _same_result(result, original.solve(**args))
    assert result["cpu_lp_calls"] == 0
    assert "host input/update/control" in result["scope"]


def test_dynamic_numeric_updates_reuse_graph_and_refresh_original_bound_witnesses():
    service = _service([_problem(rhs=3.), _problem(rhs=3.)])
    graph = service.workspace.graph
    matrix_pointers = [a.data.data.ptr for a in (service.a, service._original_a, service.workspace.a, service.workspace.at)]
    vector_pointers = [a.data.ptr for a in service._original_assembled[2:]]
    assert cp.asnumpy(service._upper_witness).ravel().tolist() == [0, 0]
    assert cp.asnumpy(service._lower_witness).ravel().tolist() == [0, 0]
    changed = [_problem(inequality=(.25, 1.5), rhs=4., lower=(-1., 0.), upper=(2., .7), c=(-.4, -.8)),
               _problem(inequality=(2., .25), rhs=5., lower=(-1., 0.), upper=(2., .9), c=(-.6, -.4))]
    assert service.try_update(changed)
    assert service.workspace.graph is graph
    assert matrix_pointers == [a.data.data.ptr for a in (service.a, service._original_a, service.workspace.a, service.workspace.at)]
    assert vector_pointers == [a.data.ptr for a in service._original_assembled[2:]]
    assert cp.asnumpy(service._upper_witness).ravel().tolist() == [1, 1]
    assert cp.asnumpy(service._lower_witness).ravel().tolist() == [1, 1]
    assert service.last_update_timing["reused"]
    result = service.solve(iterations=12, check_interval=12)
    rebuilt = _service(changed).solve(iterations=12, check_interval=12)
    _same_result(result, rebuilt)
    assert result["all_accepted"]
    np.testing.assert_allclose(cp.asnumpy(result["x"]), [[.7, .7], [.9, .9]], atol=1e-12)


def test_returned_arrays_remain_owned_after_next_problem_and_solve():
    service = _service([_problem(rhs=3.)])
    first = service.solve(iterations=8, check_interval=8)
    snapshots = {name: cp.asnumpy(first[name]) for name in ("x", "y", "reduced_x", "reduced_y")}
    assert service.try_update([_problem(rhs=3., upper=(.3, 2.))])
    second = service.solve(iterations=8, check_interval=8)
    for name, value in snapshots.items():
        np.testing.assert_array_equal(cp.asnumpy(first[name]), value)
        assert first[name].data.ptr != second[name].data.ptr
    assert first["x"].data.ptr != service.workspace.x.data.ptr


@pytest.mark.parametrize("change,reason", [
    ("order", "batch_or_environment_order_changed"),
    ("batch", "batch_or_environment_order_changed"),
    ("original_pattern", "original_csr_pattern_changed"),
    ("reduced_pattern", "reduced_csr_pattern_changed"),
])
def test_try_update_rebuild_requests_do_not_mutate_old_valid_service(change, reason):
    service = _service(environment_ids=[11, 29])
    old_values = cp.asnumpy(service.workspace.a.data)
    old_rhs = cp.asnumpy(service.workspace.rhs)
    problems, ids = [_problem(), _problem()], [11, 29]
    if change == "order":
        ids.reverse()
    elif change == "batch":
        problems, ids = problems[:1], ids[:1]
    elif change == "original_pattern":
        problems[0] = _problem(inequality=(1., 0.))
    else:
        problems[0] = _problem(inequality=(1., -1.))
    assert not service.try_update(problems, environment_ids=ids)
    assert service.last_update_timing["reason"] == reason
    np.testing.assert_array_equal(cp.asnumpy(service.workspace.a.data), old_values)
    np.testing.assert_array_equal(cp.asnumpy(service.workspace.rhs), old_rhs)
    service.solve(iterations=0)


@pytest.mark.parametrize("bad", ["equality", "equality_rhs", "nan", "bounds"])
def test_invalid_problem_rejected_before_host_or_device_state_mutation(bad):
    service = _service()
    old_problem = service.original_problems
    old_values = cp.asnumpy(service._original_a.data)
    old_rhs = cp.asnumpy(service.workspace.rhs)
    problem = list(_problem())
    if bad == "equality":
        problem[0] = csr_matrix([[1., -2.], [1., .5]])
    elif bad == "equality_rhs":
        problem[1] = np.array([1., .8])
    elif bad == "nan":
        problem[4] = np.array([np.nan, -.3])
    else:
        problem[2], problem[3] = np.array([1., 0.]), np.array([2., .5])
    with pytest.raises(ValueError):
        service.try_update([tuple(problem), _problem()])
    assert service.original_problems is old_problem
    np.testing.assert_array_equal(cp.asnumpy(service._original_a.data), old_values)
    np.testing.assert_array_equal(cp.asnumpy(service.workspace.rhs), old_rhs)
    assert service._valid


def test_interrupted_device_update_invalidates_service_until_rebuild(monkeypatch):
    service = _service()
    def fail(**kwargs):
        raise RuntimeError("simulated device update failure")
    monkeypatch.setattr(service.workspace, "update_problem", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        service.try_update([_problem(rhs=2.), _problem(rhs=2.)])
    with pytest.raises(RuntimeError, match="invalidated"):
        service.solve(iterations=1)


def test_final_certificate_regression_must_not_retain_earlier_success(monkeypatch):
    service = _service([_problem(rhs=3., c=(1., .3)), _problem()])
    original = service._certificate
    calls = 0
    def certificate(x, y):
        nonlocal calls
        rows = original(x, y)
        if calls:
            rows[0]["certificate_passed"] = False
            rows[0]["relative_kkt_gap"] = 1.
        rows[1]["certificate_passed"] = False
        calls += 1
        return rows
    monkeypatch.setattr(service, "_certificate", certificate)
    result = service.solve(iterations=4, check_interval=4)
    assert result["checkpoints"][0]["accepted_count"] == 1
    assert not result["all_accepted"]
    assert result["accepted"].tolist() == [False, False]
    assert result["metrics"][0]["accepted_iteration"] == -1
    assert not result["metrics"][0]["success"]
    assert not service.history[-1]["accepted"][0]


def test_certified_frozen_environment_stays_fixed_without_cpu_optimizer(monkeypatch):
    import highspy
    import scipy.optimize
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU optimizer called")
    monkeypatch.setattr(highspy.Highs, "run", forbidden)
    monkeypatch.setattr(scipy.optimize, "linprog", forbidden)
    service = _service([_problem(rhs=3., c=(1., .3)), _problem(rhs=3.)])
    result = service.solve(iterations=8, check_interval=8)
    assert result["all_accepted"]
    assert result["metrics"][0]["accepted_iteration"] == 0
    np.testing.assert_array_equal(cp.asnumpy(result["x"])[0], [0., 0.])
    assert result["cpu_lp_calls"] == 0
