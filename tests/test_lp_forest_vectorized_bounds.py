"""Small property comparisons against the previous scatter-based algorithm."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_trace import problem_hash


def legacy_scatter_bounds(plan, lower, upper):
    """Frozen reference: production reduce before segmented-bound changes."""
    groups, weights = plan.original_to_reduced, plan.weights
    positive = weights > 0.
    induced_lower, induced_upper = np.empty(len(weights)), np.empty(len(weights))
    induced_lower[positive] = lower[positive] / weights[positive]
    induced_upper[positive] = upper[positive] / weights[positive]
    induced_lower[~positive] = upper[~positive] / weights[~positive]
    induced_upper[~positive] = lower[~positive] / weights[~positive]
    reduced_lower, reduced_upper = np.full(plan.reduced_variables, -np.inf), np.full(plan.reduced_variables, np.inf)
    np.maximum.at(reduced_lower, groups, induced_lower)
    np.minimum.at(reduced_upper, groups, induced_upper)
    if np.any(reduced_lower > reduced_upper):
        raise ValueError('Original bounds are infeasible under eliminated equalities')
    indices = np.arange(plan.original_variables, dtype=np.int64)
    def witnesses(induced, endpoints):
        candidate = np.isfinite(endpoints[groups]) & (induced == endpoints[groups])
        best = np.full(plan.reduced_variables, -np.inf)
        np.maximum.at(best, groups[candidate], np.abs(weights[candidate]))
        preferred = candidate & (np.abs(weights) == best[groups])
        result = np.full(plan.reduced_variables, plan.original_variables, dtype=np.int64)
        np.minimum.at(result, groups[preferred], indices[preferred])
        result[result == plan.original_variables] = -1
        return result
    return reduced_lower, reduced_upper, witnesses(induced_lower, reduced_lower), witnesses(induced_upper, reduced_upper)


def randomized_problem(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    assignments = rng.integers(0, min(n, 8), size=n)
    weights = rng.choice([-1., 1.], n) * np.exp2(rng.integers(-3, 1, size=n))
    rows = []
    for group in np.unique(assignments):
        members = np.flatnonzero(assignments == group)
        for column in members[1:]:
            row = np.zeros(n)
            row[members[0]], row[column] = weights[column], -weights[members[0]]
            rows.append(row)
    a = csr_matrix(np.asarray(rows).reshape((-1, n)))
    p = (a, np.zeros(a.shape[0]), np.full(n, -10.), np.full(n, 10.), rng.integers(-5, 6, n).astype(float), a.shape[0])
    return p, rng


@pytest.mark.parametrize('seed', range(80))
def test_segmented_bounds_and_witnesses_match_original_scatter_over_dynamic_boxes(seed):
    initial, rng = randomized_problem(seed)
    plan = HomogeneousEqualityReduction(initial)
    layout = plan._bound_layout
    groups, weights, count = plan.original_to_reduced, plan.weights, plan.reduced_variables
    for update in range(4):
        # Dyadic maps/boxes create exact ties as well as different active bounds.
        induced_lo = -rng.integers(0, 8, plan.original_variables).astype(float)
        induced_hi = rng.integers(0, 8, plan.original_variables).astype(float)
        lower = np.minimum(weights * induced_lo, weights * induced_hi)
        upper = np.maximum(weights * induced_lo, weights * induced_hi)
        lower[rng.random(len(lower)) < .2] = -np.inf
        upper[rng.random(len(upper)) < .2] = np.inf
        if update == 0:
            lower[groups == 0], upper[groups == 0] = -np.inf, np.inf
        current = (initial[0], initial[1], lower, upper,
                   rng.integers(-5, 6, len(weights)).astype(float), initial[-1])
        before = problem_hash(current)
        expected = legacy_scatter_bounds(plan, lower, upper)
        reduced = plan.reduce(current)
        actual = (*reduced.problem[2:4], reduced.lower_witness, reduced.upper_witness)
        for old, new in zip(expected, actual):
            np.testing.assert_array_equal(new, old)
        assert plan._bound_layout is layout
        assert problem_hash(current) == before
        np.testing.assert_array_equal(reduced.problem[4], np.asarray(plan.transform.T @ current[4]).ravel())
        z = rng.integers(-3, 4, count).astype(float)
        assert current[4] @ plan.expand_primal(z) == pytest.approx(reduced.problem[4] @ z, abs=1e-12)
        # Independently check cost allocation signs and missing-bound fallback.
        costs = rng.choice([-2., 0., 3.], count)
        allocation = np.zeros(len(weights))
        for group, value in enumerate(costs):
            if value == 0.: continue
            witness = expected[2 if value > 0. else 3][group]
            if witness < 0: witness = plan.representatives[group]
            allocation[witness] = value / weights[witness]
        np.testing.assert_array_equal(reduced.allocate_bound_reduced_cost(costs), allocation)


def test_priority_prefers_largest_absolute_weight_then_smallest_original_column():
    # Null map [0.5, -1, 1] has ties at both endpoints; choose column 1,
    # not the smallest group column (0), then not later equal-weight column 2.
    p = (csr_matrix([[2., 1., 0.], [0., 1., 1.]]), np.zeros(2),
         np.array([-1., -2., -2.]), np.array([1., 2., 2.]), np.zeros(3), 2)
    plan = HomogeneousEqualityReduction(p)
    reduced = plan.reduce(p)
    assert reduced.lower_witness.tolist() == reduced.upper_witness.tolist() == [1]
    for old, new in zip(legacy_scatter_bounds(plan, p[2], p[3]),
                       (*reduced.problem[2:4], reduced.lower_witness, reduced.upper_witness)):
        np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize('delta', [np.nextafter(0., 1.), 1e-15, 1.])
def test_infeasible_intersection_has_no_new_tolerance(delta):
    p = (csr_matrix([[1., -1.]]), np.zeros(1), np.array([delta, -1.]),
         np.array([2., 0.]), np.zeros(2), 1)
    plan = HomogeneousEqualityReduction(p)
    with pytest.raises(ValueError, match='infeasible'): legacy_scatter_bounds(plan, p[2], p[3])
    with pytest.raises(ValueError, match='infeasible'): plan.reduce(p)


@pytest.mark.parametrize('field,value', [(1, np.nan), (1, np.inf), (2, np.nan), (3, np.nan),
    (4, np.nan), (4, np.inf), (2, np.inf), (3, -np.inf)])
def test_invalid_vectors_or_impossible_infinite_bounds_fail_closed(field, value):
    p = (csr_matrix([[1., -1.]]), np.zeros(1), np.full(2, -1.), np.ones(2), np.zeros(2), 1)
    plan = HomogeneousEqualityReduction(p)
    bad = list(p)
    bad[field] = p[field].copy()
    bad[field][0] = value
    with pytest.raises(ValueError): plan.reduce(tuple(bad))


@pytest.mark.parametrize('endpoint', [np.inf, -np.inf])
def test_equal_impossible_infinite_fixed_pair_is_rejected(endpoint):
    p = (csr_matrix([[1., -1.]]), np.zeros(1), np.full(2, endpoint), np.full(2, endpoint), np.zeros(2), 1)
    with pytest.raises(ValueError, match='Impossible infinite'): HomogeneousEqualityReduction(p)


def test_finite_bound_overflow_is_rejected_instead_of_silently_dropping_endpoint():
    p = (csr_matrix([[-2., 1.]]), np.zeros(1), np.zeros(2), np.full(2, np.finfo(float).max), np.zeros(2), 1)
    plan = HomogeneousEqualityReduction(p)
    with pytest.raises(ValueError, match='overflowed'): plan.reduce(p)


@pytest.mark.parametrize('field,value', [('weights', np.nan), ('weights', 0.),
    ('weights', np.inf), ('weights', -.25), ('original_to_reduced', 99)])
def test_mutated_dynamic_coordinate_metadata_cannot_use_stale_layout(field, value):
    p, _ = randomized_problem(17)
    plan = HomogeneousEqualityReduction(p)
    getattr(plan, field)[0] = value
    with pytest.raises(ValueError, match='coordinate layout changed'): plan.reduce(p)


def test_cached_layout_is_immutable_and_numeric_results_do_not_alias_it():
    p, _ = randomized_problem(33)
    plan = HomogeneousEqualityReduction(p)
    layout = plan._bound_layout
    for name in ('groups', 'weights', 'order', 'ordered_groups', 'starts', 'positions', 'positive'):
        value = getattr(layout, name)
        assert not value.flags.writeable
        with pytest.raises(ValueError): value.flags.writeable = True
    with pytest.raises(FrozenInstanceError): layout.order = layout.order[::-1]
    first = plan.reduce(p)
    first.lower_witness[:] = -99
    first.problem[2][:] = -99
    second = plan.reduce(p)
    expected = legacy_scatter_bounds(plan, p[2], p[3])
    np.testing.assert_array_equal(second.lower_witness, expected[2])
    np.testing.assert_array_equal(second.problem[2], expected[0])
