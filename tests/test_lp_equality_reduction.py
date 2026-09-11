"""CPU-only algebra checks for exact homogeneous equality reduction."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.cpu_repeated_lp import _certificate
from src.lp_equality_reduction import HomogeneousEqualityReduction


def _problem(a, rhs, lower, upper, c, neq):
    return (
        csr_matrix(np.asarray(a, dtype=np.float64)),
        np.asarray(rhs, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
        neq,
    )


def _original_certificate(problem, x, y):
    a, _rhs, _lower, _upper, c, _neq = problem
    reduced = c - np.asarray(a.T @ y).ravel()
    solution = SimpleNamespace(
        col_value=np.asarray(x),
        row_dual=np.asarray(y),
        col_dual=reduced,
        value_valid=True,
        dual_valid=True,
    )
    return _certificate(*problem, solution)


def test_equal_variables_reduce_and_lift_original_primal_and_dual_certificate():
    # min -x0-x1, x0=x1, 0<=x<=1 becomes min -2z, 0<=z<=1.
    problem = _problem([[1.0, -1.0]], [0.0], [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], 1)
    plan = HomogeneousEqualityReduction.from_problem(problem)
    reduced = plan.reduce(problem)

    assert plan.transform.shape == (2, 1)
    np.testing.assert_array_equal(plan.eliminated_rows, [0])
    np.testing.assert_allclose(plan.transform.toarray(), [[1.0], [1.0]])
    assert reduced.problem[0].shape == (0, 1)
    np.testing.assert_allclose(reduced.problem[2:5], ([0.0], [1.0], [-2.0]))

    x = reduced.expand_primal([1.0])
    y = reduced.lift_dual(np.empty(0), np.asarray([-2.0]))
    np.testing.assert_allclose(x, [1.0, 1.0])
    np.testing.assert_allclose(y, [1.0])
    assert _original_certificate(problem, x, y)["certificate_passed"]


def test_scaled_relation_uses_large_weight_representative_and_active_bound_witness():
    # -2*x0+x1=0.  Normalizing the largest null-space weight gives
    # x=[0.5,1]*z, avoiding an inverse-small-coefficient primal map.
    problem = _problem([[-2.0, 1.0]], [0.0], [0.0, 0.0], [1.0, 2.0], [-1.0, 0.0], 1)
    plan = HomogeneousEqualityReduction(problem)
    reduced = plan.reduce(problem)

    np.testing.assert_allclose(plan.transform.toarray(), [[0.5], [1.0]])
    np.testing.assert_array_equal(plan.representatives, [1])
    assert reduced.upper_witness.tolist() == [1]
    np.testing.assert_allclose(reduced.problem[3], [2.0])
    np.testing.assert_allclose(reduced.problem[4], [-0.5])

    original_cost = reduced.allocate_bound_reduced_cost([-0.5])
    np.testing.assert_allclose(original_cost, [0.0, -0.5])
    x = reduced.expand_primal([2.0])
    y = reduced.lift_dual([], [-0.5])
    np.testing.assert_allclose(y, [0.5])
    assert _original_certificate(problem, x, y)["certificate_passed"]


def test_negative_weight_maps_reduced_lower_to_original_upper_bound():
    # x0+x1=0, min -x1 becomes min z for x=[1,-1]*z.  z=0 is
    # simultaneously x0's lower and x1's upper endpoint.
    problem = _problem([[1.0, 1.0]], [0.0], [0.0, -1.0], [1.0, 0.0], [0.0, -1.0], 1)
    plan = HomogeneousEqualityReduction(problem)
    reduced = plan.reduce(problem)

    np.testing.assert_allclose(plan.transform.toarray(), [[1.0], [-1.0]])
    assert reduced.lower_witness.tolist() == [0]
    y = reduced.lift_dual([], [1.0])
    np.testing.assert_allclose(y, [-1.0])
    assert _original_certificate(problem, [0.0, 0.0], y)["certificate_passed"]


def test_tree_dual_map_solves_eliminated_equality_multipliers_without_online_factorization():
    problem = _problem(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
        [0.0, 0.0],
        [-np.inf] * 3,
        [np.inf] * 3,
        [0.0] * 3,
        2,
    )
    plan = HomogeneousEqualityReduction(problem)
    difference = np.asarray([1.0, 0.0, -1.0])

    np.testing.assert_allclose(plan.transform.toarray(), [[1.0], [1.0], [1.0]])
    np.testing.assert_allclose(plan.dual_lift_map @ difference, [1.0, 1.0])
    np.testing.assert_allclose(
        problem[0].T @ (plan.dual_lift_map @ difference), difference
    )


def test_cycle_is_retained_while_an_independent_spanning_forest_is_eliminated():
    problem = _problem(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0],
        [0.0] * 3,
        [1.0] * 3,
        [0.0] * 3,
        3,
    )
    plan = HomogeneousEqualityReduction(problem)
    reduced = plan.reduce(problem)

    np.testing.assert_array_equal(plan.eliminated_rows, [0, 1])
    np.testing.assert_array_equal(plan.kept_rows, [2])
    assert reduced.problem[-1] == 1
    assert reduced.problem[0].shape == (1, 1)
    assert reduced.problem[0].nnz == 0


def test_near_zero_coefficient_and_nonhomogeneous_rows_are_retained():
    near_zero = _problem([[1.0, -1e-12]], [0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], 1)
    nonhomogeneous = _problem([[1.0, -1.0]], [1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0], 1)

    assert HomogeneousEqualityReduction(near_zero).eliminated_rows.size == 0
    assert HomogeneousEqualityReduction(nonhomogeneous).eliminated_rows.size == 0


def test_equality_fingerprint_is_fixed_but_retained_rows_and_vectors_are_dynamic():
    initial = _problem(
        [[1.0, -1.0], [2.0, 3.0]],
        [0.0, 8.0],
        [0.0, 0.0],
        [4.0, 4.0],
        [1.0, 2.0],
        1,
    )
    changed = _problem(
        [[1.0, -1.0], [4.0, -1.0]],
        [0.0, 7.0],
        [-1.0, 0.0],
        [3.0, 2.0],
        [-2.0, 5.0],
        1,
    )
    plan = HomogeneousEqualityReduction(initial)
    reduced = plan.reduce(changed).problem

    np.testing.assert_allclose(reduced[0].toarray(), [[3.0]])
    np.testing.assert_allclose(reduced[1], [7.0])
    np.testing.assert_allclose(reduced[2], [0.0])
    np.testing.assert_allclose(reduced[3], [2.0])
    np.testing.assert_allclose(reduced[4], [3.0])

    stale = list(changed)
    stale[0] = csr_matrix([[1.0, -2.0], [4.0, -1.0]])
    with pytest.raises(ValueError, match="fingerprint"):
        plan.reduce(tuple(stale))


def test_bound_intersection_infeasibility_is_rejected_without_relaxation():
    problem = _problem([[1.0, -1.0]], [0.0], [1.0, 0.0], [2.0, 0.5], [0.0, 0.0], 1)
    plan = HomogeneousEqualityReduction(problem)
    with pytest.raises(ValueError, match="infeasible"):
        plan.reduce(problem)


def test_primal_compression_is_exact_on_nullspace_and_projection_only_for_warm_start():
    problem = _problem([[1.0, -1.0]], [0.0], [-5.0, -5.0], [5.0, 5.0], [0.0, 0.0], 1)
    plan = HomogeneousEqualityReduction(problem)

    np.testing.assert_allclose(plan.compress_primal([2.0, 2.0], check=True), [2.0])
    np.testing.assert_allclose(plan.compress_primal([[1.0, 1.0], [2.0, 2.0]]), [[1.0], [2.0]])
    np.testing.assert_allclose(plan.expand_primal([[1.0], [2.0]]), [[1.0, 1.0], [2.0, 2.0]])
    np.testing.assert_allclose(plan.compress_primal([1.0, 2.0]), [1.5])
    with pytest.raises(ValueError, match="does not satisfy"):
        plan.compress_primal([1.0, 2.0], check=True)


def test_missing_reduced_bound_remains_algebraic_but_fails_original_dual_gate():
    problem = _problem(
        [[1.0, -1.0]], [0.0], [-np.inf, -np.inf], [np.inf, np.inf], [1.0, 0.0], 1
    )
    reduced = HomogeneousEqualityReduction(problem).reduce(problem)
    y = reduced.lift_dual([], [1.0])
    certificate = _original_certificate(problem, [0.0, 0.0], y)

    assert not certificate["certificate_passed"]
    assert certificate["dual_violation"] == pytest.approx(1.0)


@pytest.mark.parametrize("value", [True, 0.5, np.nan, np.inf])
def test_scale_ratio_validation(value):
    problem = _problem([[1.0, -1.0]], [0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], 1)
    with pytest.raises(ValueError, match="max_scale_ratio"):
        HomogeneousEqualityReduction(problem, max_scale_ratio=value)
