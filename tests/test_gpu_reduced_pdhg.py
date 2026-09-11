"""Tiny CUDA integration checks for exact equality-reduced PDHG."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix


cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("A CUDA device is required", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("A CUDA device is required", allow_module_level=True)

from src.gpu_pdhg_corrector import GpuPdhgCorrector
from src.gpu_reduced_pdhg import GpuReducedPdhgCorrector


def _problem(a, rhs, lower, upper, c, neq):
    return (
        csr_matrix(np.asarray(a, dtype=np.float64)),
        np.asarray(rhs, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
        neq,
    )


def _equal_box_problem(*, upper=1.0):
    # min -x0-x1, x0=x1, 0<=x<=upper.  Eliminating the equality leaves a
    # one-variable box-only LP, so neither a row solver nor CPU crossover is
    # needed to reach the exact upper endpoint.
    return _problem(
        [[1.0, -1.0]],
        [0.0],
        [0.0, 0.0],
        [upper, upper],
        [-1.0, -1.0],
        1,
    )


def test_box_only_reduction_runs_without_any_cpu_optimizer(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU optimizer was called")

    import highspy
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "linprog", forbidden)
    monkeypatch.setattr(highspy.Highs, "run", forbidden)

    corrector = GpuReducedPdhgCorrector(
        [_equal_box_problem()], step_safety=0.9
    )
    assert corrector.reduced_problems[0][0].shape == (0, 1)
    assert corrector.plan.T.shape == (2, 1)

    result = corrector.solve(iterations=3, check_interval=1)

    assert result["all_accepted"]
    assert result["iterations_run"] == 1
    assert result["cpu_lp_calls"] == 0
    assert result["x"].shape == (1, 2)
    assert result["y"].shape == (1, 1)
    assert result["reduced_x"].shape == (1, 1)
    assert result["reduced_y"].shape == (1, 0)
    np.testing.assert_allclose(cp.asnumpy(result["x"]), [[1.0, 1.0]], atol=1e-12)
    assert result["metrics"][0]["certificate_passed"]
    assert "ORIGINAL-LP certificate" in result["scope"]


def test_negative_transform_weight_and_dynamic_bound_witnesses_are_independent():
    # x1=-x0.  Environment 0 is optimal at the reduced lower endpoint.
    # Environment 1 is optimal at z=2, whose limiting ORIGINAL endpoint is
    # x1's lower bound; its transform weight is negative.
    lower_optimum = _problem(
        [[1.0, 1.0]],
        [0.0],
        [0.0, -3.0],
        [2.0, 0.0],
        [1.0, 0.0],
        1,
    )
    negative_weight_upper = _problem(
        [[1.0, 1.0]],
        [0.0],
        [0.0, -2.0],
        [3.0, 0.0],
        [-1.0, 0.0],
        1,
    )
    corrector = GpuReducedPdhgCorrector(
        [lower_optimum, negative_weight_upper], step_safety=0.9
    )

    np.testing.assert_allclose(corrector.plan.weights, [1.0, -1.0])
    assert corrector.reductions[0].lower_witness.tolist() == [0]
    assert corrector.reductions[1].upper_witness.tolist() == [1]

    result = corrector.solve(iterations=4, check_interval=1)

    assert result["all_accepted"]
    assert [row["accepted_iteration"] for row in result["metrics"]] == [0, 2]
    np.testing.assert_allclose(
        cp.asnumpy(result["x"]), [[0.0, 0.0], [2.0, -2.0]], atol=1e-12
    )
    assert all(row["certificate_passed"] for row in result["metrics"])
    # The first block remains frozen while the second one takes two steps.
    np.testing.assert_allclose(cp.asnumpy(result["reduced_x"]).ravel(), [0.0, 2.0])


def test_fixed_plan_accepts_dynamic_vectors_but_rejects_stale_equalities():
    original = _equal_box_problem(upper=1.0)
    first = GpuReducedPdhgCorrector([original])

    dynamic = _equal_box_problem(upper=2.0)
    reused = GpuReducedPdhgCorrector([dynamic], plan=first.plan)
    assert reused.plan is first.plan
    np.testing.assert_allclose(reused.reduced_problems[0][3], [2.0])

    stale_coefficients = list(dynamic)
    stale_coefficients[0] = csr_matrix([[1.0, -2.0]])
    with pytest.raises(ValueError, match="fingerprint"):
        GpuReducedPdhgCorrector([tuple(stale_coefficients)], plan=first.plan)

    stale_rhs = list(dynamic)
    stale_rhs[1] = np.asarray([0.25])
    with pytest.raises(ValueError, match="homogeneous"):
        GpuReducedPdhgCorrector([tuple(stale_rhs)], plan=first.plan)


def test_original_coordinate_warm_start_contract_and_zero_budget_failure():
    # The retained inequality makes x=0 feasible but non-optimal.  A zero
    # budget must not turn the compressed warm start into an accepted answer.
    problem = _problem(
        [[1.0, -1.0], [1.0, 0.0]],
        [0.0, 0.5],
        [0.0, 0.0],
        [2.0, 2.0],
        [-1.0, -1.0],
        1,
    )
    corrector = GpuReducedPdhgCorrector([problem])

    with pytest.raises(ValueError, match="initial_x"):
        corrector.solve(initial_x=np.zeros((1, 1)), iterations=0)
    with pytest.raises(ValueError, match="initial_y"):
        corrector.solve(initial_y=np.zeros((1, 1)), iterations=0)

    result = corrector.solve(
        initial_x=np.zeros((1, 2)),
        initial_y=np.zeros((1, 2)),
        iterations=0,
        check_interval=1,
    )

    assert not result["all_accepted"]
    assert result["accepted"].tolist() == [False]
    assert result["iterations_run"] == 0
    assert result["metrics"][0]["accepted_iteration"] == -1
    assert not result["metrics"][0]["certificate_passed"]
    assert result["cpu_lp_calls"] == 0
    assert result["x"].shape == (1, 2)
    assert result["y"].shape == (1, 2)


def test_original_certificate_rejects_a_corrupted_lift_even_if_reduced_passes(
    monkeypatch,
):
    corrector = GpuReducedPdhgCorrector([_equal_box_problem()])
    z = cp.asarray([[1.0]], dtype=cp.float64)
    reduced_y = cp.empty((1, 0), dtype=cp.float64)
    reduced_metrics = GpuPdhgCorrector._certificate(corrector, z, reduced_y)
    assert reduced_metrics[0]["certificate_passed"]

    real_expand_and_lift = corrector.expand_and_lift

    def corrupt_original_equality(reduced_x, supplied_y):
        x, y = real_expand_and_lift(reduced_x, supplied_y)
        x = x.copy()
        x[0, 1] -= 0.25
        return x, y

    monkeypatch.setattr(corrector, "expand_and_lift", corrupt_original_equality)
    result = corrector.solve(
        initial_x=np.asarray([[1.0, 1.0]]),
        initial_y=np.zeros((1, 1)),
        iterations=0,
        check_interval=1,
    )

    assert not result["all_accepted"]
    assert not result["metrics"][0]["certificate_passed"]
    assert result["metrics"][0]["primal_residual"] >= 0.25
    np.testing.assert_allclose(cp.asnumpy(result["reduced_x"]), [[1.0]])


def test_infeasible_dynamic_bounds_are_rejected_before_gpu_iterations():
    problem = _problem(
        [[1.0, -1.0]],
        [0.0],
        [1.0, 0.0],
        [2.0, 0.5],
        [0.0, 0.0],
        1,
    )
    with pytest.raises(ValueError, match="infeasible"):
        GpuReducedPdhgCorrector([problem])
