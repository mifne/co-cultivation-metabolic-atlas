"""Input-only proofs and dual algebra; never calls a CPU or GPU LP solver."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.lp_trace import problem_hash
from src.lp_zero_face import InfeasibleZeroRow, ZeroFaceReduction


def problem(a, rhs, lo, hi, c, neq):
    return (csr_matrix(a, dtype=float), *(np.asarray(v, dtype=float)
        for v in (rhs, lo, hi, c)), neq)


@pytest.mark.parametrize('coefficient,cost', [(1., 7.), (1., -7.), (-2., 7.), (-2., -7.)])
def test_interior_singleton_requires_signed_zero_reduced_cost(coefficient, cost):
    p = problem([[coefficient, 0]], [0], [-3, 0], [4, 1], [cost, 2], 1)
    before = problem_hash(p)
    r = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert [(w.orientation, w.newly_fixed_columns) for w in r.witnesses] == [('equality', (0,))]
    x, y = r.lift([0], [])
    np.testing.assert_array_equal(x, [0, 0])
    np.testing.assert_allclose(y, [cost/coefficient], atol=0, rtol=0)
    np.testing.assert_array_equal(p[4] - p[0].T@y, [0, 2])
    assert problem_hash(p) == before


def test_feature_is_opt_in_and_does_not_change_legacy_witnesses():
    p = problem([[1, 0]], [0], [-1, 0], [1, 1], [1, 1], 1)
    assert not ZeroFaceReduction(p).witnesses
    p = problem([[1, 0]], [0], [0, 0], [1, 1], [1, 1], 1)
    assert ZeroFaceReduction(p, fix_singleton_equalities=True).witnesses[0].orientation == 'min'


def test_singleton_cascade_repairs_prior_fixed_costs_in_reverse_order():
    p = problem([[2, -3, 0], [1, 0, 0]], [0, 0], [-4, -5, 0], [6, 7, 2], [-11, 13, 1], 2)
    r = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert [w.row for w in r.witnesses] == [1, 0]
    assert all(w.orientation == 'equality' for w in r.witnesses)
    x, y = r.lift([0], [])
    np.testing.assert_allclose(p[0]@x, p[1], atol=0, rtol=0)
    np.testing.assert_allclose(p[4]-p[0].T@y, [0, 0, 1], atol=2e-15, rtol=0)


def test_nonzero_fixed_substitution_is_not_used_as_zero_proof():
    p = problem([[1, 1, -1]], [0], [-2, 1, 1], [3, 1, 1], [1, 0, 0], 1)
    r = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert not r.witnesses  # Even exact cancellation of nonzero fixed terms is out of scope.
    assert r.columns.tolist() == [0]


@pytest.mark.parametrize('lo,hi', [(1., 2.), (-2., -1.)])
def test_zero_outside_singleton_bounds_is_infeasible(lo, hi):
    p = problem([[1]], [0], [lo], [hi], [0], 1)
    with pytest.raises(InfeasibleZeroRow, match='singleton equality'):
        ZeroFaceReduction(p, fix_singleton_equalities=True)


@pytest.mark.parametrize('rhs,neq', [(1e-300, 1), (0., 0)])
def test_nonzero_rhs_or_inequality_never_proves_interior_zero(rhs, neq):
    p = problem([[1]], [rhs], [-1], [1], [0], neq)
    assert not ZeroFaceReduction(p, fix_singleton_equalities=True).witnesses


def test_tiny_relation_is_retained_not_rounded_or_fixed():
    p = problem([[1e-16, 0]], [0], [-1, 0], [1, 1], [1, 1], 1)
    r = ZeroFaceReduction(p, fix_singleton_equalities=True)
    assert not r.witnesses
    assert r.skipped_small_singleton_rows.tolist() == [0]
    assert problem_hash(r.reduced) == problem_hash(p)


def test_threshold_boundary_and_unbounded_singleton():
    p = problem([[1e-12]], [0], [-np.inf], [np.inf], [1e-12], 1)
    r = ZeroFaceReduction(p, fix_singleton_equalities=True)
    x, y = r.lift([], [])
    np.testing.assert_array_equal(x, [0])
    np.testing.assert_array_equal(y, [1])


@pytest.mark.parametrize('kwargs', [dict(fix_singleton_equalities=1),
    dict(singleton_min_coefficient=0), dict(singleton_min_coefficient=-1),
    dict(singleton_min_coefficient=True), dict(singleton_min_coefficient=np.nan),
    dict(singleton_min_coefficient=np.inf)])
def test_singleton_options_validated(kwargs):
    p = problem([[1]], [0], [-1], [1], [0], 1)
    with pytest.raises(ValueError):
        ZeroFaceReduction(p, **kwargs)


def test_signed_lift_overflow_fails_closed():
    p = problem([[1e-12]], [0], [-1], [1], [1e308], 1)
    r = ZeroFaceReduction(p, fix_singleton_equalities=True)
    with pytest.raises(ValueError, match='overflow'):
        r.lift([], [])
