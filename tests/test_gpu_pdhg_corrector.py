"""Small-device checks for strict FP64 original-LP PDHG correction."""

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


def _problem(a, rhs, lower, upper, c, neq):
    return (
        csr_matrix(np.asarray(a, dtype=np.float64)),
        np.asarray(rhs, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
        neq,
    )


def test_inequality_dual_sign_and_correction_pass_original_certificate():
    # min -x, x <= 1, 0 <= x <= 2 has x=1 and HiGHS-style row dual y=-1.
    problem = _problem([[1.0]], [1.0], [0.0], [2.0], [-1.0], 0)
    corrector = GpuPdhgCorrector([problem], step_safety=0.9)
    result = corrector.solve(iterations=200, check_interval=10)

    assert result["all_accepted"]
    assert result["accepted"].tolist() == [True]
    np.testing.assert_allclose(cp.asnumpy(result["x"]), [[1.0]], atol=1e-8)
    np.testing.assert_allclose(cp.asnumpy(result["y"]), [[-1.0]], atol=1e-8)
    assert result["metrics"][0]["certificate_passed"]
    assert result["metrics"][0]["dual_violation"] <= 1e-7
    assert result["metrics"][0]["relative_kkt_gap"] <= 1e-7
    assert result["cpu_lp_calls"] == 0


def test_numpy_and_cupy_warm_starts_are_certified_at_iteration_zero():
    problems = [
        _problem([[1.0]], [1.0], [0.0], [3.0], [1.0], 1),
        _problem([[1.0]], [2.0], [0.0], [3.0], [1.0], 1),
    ]
    corrector = GpuPdhgCorrector(problems)
    result = corrector.solve(
        initial_x=np.asarray([[1.0], [2.0]]),
        initial_y=cp.asarray([[1.0], [1.0]], dtype=cp.float64),
        iterations=20,
        check_interval=5,
    )

    assert result["all_accepted"]
    assert result["iterations_run"] == 0
    assert [row["accepted_iteration"] for row in result["metrics"]] == [0, 0]
    assert result["checkpoints"] == [result["checkpoints"][0]]
    assert result["checkpoints"][0]["iteration"] == 0
    assert result["x"].dtype == result["y"].dtype == cp.float64


def test_block_diagonal_problems_remain_independent():
    problems = [
        _problem([[1.0]], [1.0], [0.0], [2.0], [-1.0], 0),
        _problem([[1.0]], [0.5], [0.0], [2.0], [-1.0], 0),
    ]
    corrector = GpuPdhgCorrector(problems, step_safety=0.9)
    result = corrector.solve(iterations=300, check_interval=10)

    assert result["all_accepted"]
    np.testing.assert_allclose(
        cp.asnumpy(result["x"]).ravel(), [1.0, 0.5], atol=1e-8
    )
    np.testing.assert_allclose(
        cp.asnumpy(result["y"]).ravel(), [-1.0, -1.0], atol=3e-7
    )
    assert corrector.a.shape == (2, 2)
    assert corrector.a.nnz == 2


def test_box_only_problem_and_infinite_unused_bound_are_supported():
    # The second variable is fixed by an equality.  The first has no matrix
    # coefficient and must move to its finite objective-optimal upper bound.
    problem = _problem(
        [[0.0, 1.0]], [0.0], [-np.inf, 0.0], [2.0, np.inf], [-1.0, 0.0], 1
    )
    corrector = GpuPdhgCorrector([problem], step_safety=0.9)
    result = corrector.solve(iterations=20, check_interval=1)

    assert result["all_accepted"]
    np.testing.assert_allclose(cp.asnumpy(result["x"]), [[2.0, 0.0]], atol=1e-10)


def test_infeasible_problem_exhausts_budget_and_fails_closed():
    # x <= 0 and x >= 1 cannot both hold.  No iterate may be reported as an
    # accepted optimization result merely because the fixed budget ended.
    problem = _problem(
        [[1.0], [-1.0]], [0.0, -1.0], [-np.inf], [np.inf], [0.0], 0
    )
    corrector = GpuPdhgCorrector([problem])
    result = corrector.solve(iterations=40, check_interval=5)

    assert not result["all_accepted"]
    assert result["accepted"].tolist() == [False]
    assert result["iterations_run"] == 40
    assert result["metrics"][0]["accepted_iteration"] == -1
    assert not result["metrics"][0]["certificate_passed"]
    assert result["cpu_lp_calls"] == 0


def test_diagonal_preconditioner_and_timing_are_explicit():
    problem = _problem([[2.0, -1.0]], [1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0], 1)
    corrector = GpuPdhgCorrector([problem], step_safety=0.9)

    np.testing.assert_allclose(cp.asnumpy(corrector.sigma), [0.3])
    np.testing.assert_allclose(cp.asnumpy(corrector.tau), [0.45, 0.9])
    result = corrector.solve(
        initial_x=[0.5, 0.0], initial_y=[0.0], iterations=0, check_interval=1
    )
    for name in (
        "host_assembly_seconds",
        "gpu_transfer_seconds",
        "gpu_setup_seconds",
        "setup_total_seconds",
        "warm_start_transfer_seconds",
        "correction_seconds",
        "certificate_seconds",
        "host_control_seconds",
        "solve_total_seconds",
    ):
        assert name in result["timing"]
        assert np.isfinite(result["timing"][name])
        assert result["timing"][name] >= 0.0


def test_primal_weight_rescales_reciprocally_without_changing_safe_product():
    problem = _problem(
        [[2.0, -1.0]], [1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0], 1
    )
    baseline = GpuPdhgCorrector([problem], step_safety=0.9)
    weighted = GpuPdhgCorrector(
        [problem], step_safety=0.9, primal_weight=2.0
    )

    np.testing.assert_allclose(
        cp.asnumpy(weighted.sigma), 2.0 * cp.asnumpy(baseline.sigma)
    )
    np.testing.assert_allclose(
        cp.asnumpy(weighted.tau), cp.asnumpy(baseline.tau) / 2.0
    )
    np.testing.assert_allclose(
        cp.asnumpy(weighted.sigma)[:, None] * cp.asnumpy(weighted.tau)[None, :],
        cp.asnumpy(baseline.sigma)[:, None] * cp.asnumpy(baseline.tau)[None, :],
    )

    result = weighted.solve(iterations=0, check_interval=1)
    assert result["primal_weight"] == 2.0
    assert result["timing"]["primal_weight"] == 2.0
    assert result["checkpoints"][0]["primal_weight"] == 2.0
    assert weighted.history[-1]["primal_weight"] == 2.0


@pytest.mark.parametrize("value", [True, 0.0, -1.0, np.nan, np.inf, -np.inf])
def test_primal_weight_rejects_nonpositive_or_nonfinite_values(value):
    problem = _problem([[1.0]], [1.0], [0.0], [2.0], [-1.0], 0)
    with pytest.raises(ValueError, match="primal_weight"):
        GpuPdhgCorrector([problem], primal_weight=value)


@pytest.mark.parametrize(
    "problems,match",
    [
        ([], "At least one"),
        (
            [
                _problem([[1.0]], [1.0], [0.0], [2.0], [1.0], 1),
                _problem([[1.0], [2.0]], [1.0, 2.0], [0.0], [2.0], [1.0], 1),
            ],
            "identical dimensions",
        ),
    ],
)
def test_constructor_rejects_invalid_batches(problems, match):
    with pytest.raises(ValueError, match=match):
        GpuPdhgCorrector(problems)


def test_warm_start_shape_and_budget_validation():
    problem = _problem([[1.0]], [1.0], [0.0], [2.0], [-1.0], 0)
    corrector = GpuPdhgCorrector([problem])
    with pytest.raises(ValueError, match="initial_x"):
        corrector.solve(initial_x=np.zeros((2, 2)), iterations=1)
    with pytest.raises(ValueError, match="iterations"):
        corrector.solve(iterations=True)
    with pytest.raises(ValueError, match="check_interval"):
        corrector.solve(iterations=1, check_interval=0)
