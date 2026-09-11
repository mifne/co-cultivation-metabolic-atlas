"""Small CPU-only exact arithmetic tests; no optimizer, QR or GPU work."""
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.lp_bounded_near_equality import BoundedNearEqualityReduction, NearEqualityCandidate
from src.lp_trace import problem_hash


TINY = 2.**-50


def problem(a=None, rhs=None, lower=None, upper=None, neq=2):
    a = csr_matrix([[1., 0.], [1., TINY], [-1., 0.]] if a is None else a, dtype=np.float64)
    m, n = a.shape
    return (a, np.zeros(m) if rhs is None else np.asarray(rhs, dtype=np.float64),
            np.full(n, -2.) if lower is None else np.asarray(lower, dtype=np.float64),
            np.full(n, 3.) if upper is None else np.asarray(upper, dtype=np.float64),
            np.arange(n, dtype=np.float64), neq)


def candidate():
    return NearEqualityCandidate(1, (0,), (1,))


def test_bounded_defect_rechecked_and_exact_interval_saved():
    p = problem()
    before = problem_hash(p)
    plan = BoundedNearEqualityReduction(p, candidate())
    assert plan.matrix_defect == ((1, Fraction(1, 2**50)),)
    assert plan.rhs_defect == 0
    assert plan.defect_interval == (Fraction(-2, 2**50), Fraction(3, 2**50))
    assert plan.defect_bound == Fraction(3, 2**50)
    assert problem_hash(p) == plan.original_hash == before
    assert plan.rows.tolist() == [0, 2]
    assert plan.removed_rows.tolist() == [1]
    assert plan.reduced[-1] == 1
    assert plan.summary['current_bounds_reverified']
    assert not plan.summary['exact_feasible_set_equivalence_claimed']
    assert plan.summary['requires_original_full_row_certificate']
    assert plan.summary['requires_direct_dual_objective_and_equality_residual_check']
    assert plan.summary['cpu_lp_calls'] == plan.summary['gpu_calls'] == plan.summary['qr_calls'] == 0


def test_static_bound_does_not_imply_full_omitted_row_residual_bound():
    plan = BoundedNearEqualityReduction(problem(), candidate())
    # The missing row inherits the support error even with a 1e-15 defect.
    x = np.array([.01, 1.])
    error = plan.original[0]@x-plan.original[1]
    assert abs(error[1]) > 1e-5
    assert plan.defect_bound < Fraction.from_float(1e-12)
    assert plan.summary['static_defect_bound_is_not_a_solution_certificate']


def test_actual_exact_feasible_set_can_change_despite_tiny_box_defect():
    plan = BoundedNearEqualityReduction(problem(), candidate())
    x = np.array([0., 1.])
    np.testing.assert_array_equal(plan.reduced[0]@x, np.zeros(2))
    assert (plan.original[0]@x)[1] == TINY  # Exact original equality is not satisfied.


def test_negative_defect_and_nonzero_rhs_use_sign_correct_box():
    p = problem(a=[[1., 0.], [1., -TINY]], rhs=[0., TINY], neq=2)
    plan = BoundedNearEqualityReduction(p, candidate())
    assert plan.defect_interval == (Fraction(-4, 2**50), Fraction(1, 2**50))
    assert plan.rhs_defect == Fraction(1, 2**50)


def test_fraction_weights_are_not_rounded_and_multi_support_relation_is_verified():
    p = problem(a=[[3., 0., 0.], [0., 2., 0.], [1., -1., TINY]], neq=3)
    proposal = NearEqualityCandidate(2, (0, 1), (Fraction(1, 3), Fraction(-1, 2)))
    plan = BoundedNearEqualityReduction(p, proposal)
    assert plan.matrix_defect == ((2, Fraction(1, 2**50)),)
    assert plan.candidate.coefficients == (Fraction(1, 3), Fraction(-1, 2))


def test_exact_identity_is_allowed_but_does_not_bypass_checks():
    p = problem(a=[[1., 0.], [2., 0.]], rhs=[2., 4.], neq=2)
    plan = BoundedNearEqualityReduction(p, NearEqualityCandidate(1, (0,), (2,)))
    assert plan.defect_bound == 0
    assert plan.summary['exact_coefficient_rhs_identity']
    assert plan.summary['requires_original_full_row_certificate']


def test_rhs_only_tiny_defect_is_explicit_working_relaxation_not_exact_identity():
    plan = BoundedNearEqualityReduction(problem(a=[[1., 0.], [1., 0.]], rhs=[0., TINY]), candidate())
    assert not plan.matrix_defect
    assert plan.defect_interval == (Fraction(-1, 2**50),)*2
    assert not plan.summary['exact_coefficient_rhs_identity']


def test_current_bounds_coefficient_rhs_changes_are_not_trusted_from_old_candidate():
    prior = BoundedNearEqualityReduction(problem(), candidate())
    for changed in (problem(upper=[3., 1e6]), problem(rhs=[0., 1e-4, 0.]),
                    problem(a=[[1., 0.], [1., 1e-4], [-1., 0.]])):
        with pytest.raises(ValueError, match='exceeds'):
            BoundedNearEqualityReduction(changed, prior.candidate)
    assert prior.defect_bound == Fraction(3, 2**50)


def test_inclusive_threshold_uses_exact_products_not_rounded_underflow():
    p = problem(a=[[1., 0.], [1., 2.**-40]], lower=[0., 0.], upper=[1., 1.])
    accepted = BoundedNearEqualityReduction(p, candidate(), max_defect=2.**-40)
    assert accepted.defect_bound == accepted.max_defect
    with pytest.raises(ValueError, match='exceeds'):
        BoundedNearEqualityReduction(p, candidate(), max_defect=np.nextafter(2.**-40, 0.))
    p = problem(a=[[1., 0.], [1., 1e-300]], lower=[0., 0.], upper=[1., 1e-300])
    plan = BoundedNearEqualityReduction(p, candidate())
    assert plan.defect_bound == Fraction.from_float(1e-300)**2 > 0


@pytest.mark.parametrize('bound', [-np.inf, np.inf])
def test_infinite_bounds_on_defect_support_fail_closed(bound):
    lo, hi = [-2., -2.], [3., 3.]
    (lo if bound < 0 else hi)[1] = bound
    with pytest.raises(ValueError, match='Finite bounds'):
        BoundedNearEqualityReduction(problem(lower=lo, upper=hi), candidate())


def test_infinite_bounds_off_defect_support_do_not_spoil_exact_bound():
    plan = BoundedNearEqualityReduction(problem(lower=[-np.inf, -2.], upper=[np.inf, 3.]), candidate())
    assert plan.defect_bound == Fraction(3, 2**50)


def test_fixed_nonzero_bound_and_rhs_can_cancel_the_defect_exactly():
    p = problem(rhs=[0., 2*TINY, 0.], lower=[-2., 2.], upper=[3., 2.])
    plan = BoundedNearEqualityReduction(p, candidate())
    assert plan.matrix_defect and plan.rhs_defect
    assert plan.defect_bound == 0


def test_inequalities_objective_bounds_order_and_primal_are_unchanged():
    p = problem(a=[[1., 0.], [1., TINY], [0., 1.], [0., 0.]], rhs=[0., 0., 2., 3.])
    plan = BoundedNearEqualityReduction(p, candidate())
    np.testing.assert_array_equal(plan.reduced[0][1:].toarray(), p[0][2:].toarray())
    np.testing.assert_array_equal(plan.reduced[1][1:], p[1][2:])
    for i in (2, 3, 4):
        np.testing.assert_array_equal(plan.reduced[i], p[i])
    x = np.array([1., 2.])
    lifted = plan.lift_primal(x)
    np.testing.assert_array_equal(lifted, x)
    assert not np.shares_memory(lifted, x)


def test_dual_lift_zero_injects_and_compression_does_not_redistribute():
    plan = BoundedNearEqualityReduction(problem(), candidate())
    initial = np.array([2., 100., -3.])
    compressed = plan.compress_dual(initial)
    np.testing.assert_array_equal(compressed, [2., -3.])
    np.testing.assert_array_equal(plan.dual_compression@initial, compressed)
    lifted = plan.lift_dual(compressed)
    np.testing.assert_array_equal(lifted, [2., 0., -3.])
    np.testing.assert_array_equal(plan.original[0].T@lifted, plan.reduced[0].T@compressed)
    assert not np.array_equal(plan.original[0].T@initial, plan.original[0].T@lifted)
    assert plan.original[1]@lifted == plan.reduced[1]@compressed


@pytest.mark.parametrize('args', [
    (True, [0], [1]), (-1, [0], [1]), (1, [1], [1]), (1, [], []),
    (1, [0, 0], [1, 1]), (1, [0], [0]), (1, [0], [1, 2]),
    (1, [0], [1.]), (1, [0], [True]), (1, [False], [1]),
    (1, [-1], [1]), (1, None, [1]), (1, [0], None),
])
def test_invalid_candidates_fail_closed(args):
    with pytest.raises(ValueError):
        NearEqualityCandidate(*args)


@pytest.mark.parametrize('proposal', [NearEqualityCandidate(2, (0,), (1,)), NearEqualityCandidate(1, (2,), (1,)), None, 1])
def test_no_unconditional_known_row_deletion_or_inequality_support(proposal):
    with pytest.raises(ValueError):
        BoundedNearEqualityReduction(problem(), proposal)


@pytest.mark.parametrize('threshold', [0., -1., 1e-11, np.inf, np.nan, True, '1e-12'])
def test_invalid_or_overlarge_threshold_refused(threshold):
    with pytest.raises(ValueError, match='max_defect'):
        BoundedNearEqualityReduction(problem(), candidate(), max_defect=threshold)


@pytest.mark.parametrize('which', ['matrix', 'rhs', 'lower', 'upper', 'cost'])
def test_nonfinite_input_fails_closed(which):
    p = list(problem())
    if which == 'matrix':
        p[0].data[0] = np.nan
    else:
        p[{'rhs': 1, 'lower': 2, 'upper': 3, 'cost': 4}[which]][0] = np.nan
    with pytest.raises(ValueError):
        BoundedNearEqualityReduction(p, candidate())


def test_complex_inputs_not_silently_discarded():
    p = list(problem())
    p[0] = p[0].astype(np.complex128)
    p[0].data[0] += 1j
    with pytest.raises(ValueError, match='Real-valued'):
        BoundedNearEqualityReduction(p, candidate())


def test_ownership_readonly_immutability_and_summary_copy():
    p = problem()
    plan = BoundedNearEqualityReduction(p, candidate())
    for array in (*plan.original[1:5], *plan.reduced[1:5], plan.original[0].data,
                  plan.reduced[0].data, plan.rows, plan.removed_rows, plan.dual_compression.data):
        assert not array.flags.writeable
    with pytest.raises(AttributeError):
        plan.rows = np.array([0])
    with pytest.raises(FrozenInstanceError):
        plan.candidate.row = 0
    changed = plan.summary
    changed['candidate']['basis_rows'][0] = 99
    assert plan.summary['candidate']['basis_rows'] == [0]
    p[0].data[:] = 5.
    p[2][:] = -99.
    plan.validate_integrity()


@pytest.mark.parametrize('which', ['original', 'reduced', 'rows', 'removed', 'selector', 'candidate'])
def test_deliberate_readonly_override_detected(which):
    plan = BoundedNearEqualityReduction(problem(), candidate())
    if which == 'candidate':
        object.__setattr__(plan.candidate, 'row', 0)
    else:
        array = {'original': plan.original[2], 'reduced': plan.reduced[1], 'rows': plan.rows,
                 'removed': plan.removed_rows, 'selector': plan.dual_compression.data}[which]
        array.flags.writeable = True
        array[0] += 1
    with pytest.raises(ValueError, match='modified'):
        plan.validate_integrity()


@pytest.mark.parametrize('bad', [[1.], [1., np.nan], [1., np.inf], [[1.], [2.]], [1j, 0.]])
def test_invalid_primal_and_reduced_dual_vectors_refused(bad):
    plan = BoundedNearEqualityReduction(problem(), candidate())
    for method in (plan.lift_primal, plan.lift_dual):
        with pytest.raises(ValueError, match='Finite real vector'):
            method(bad)


@pytest.mark.parametrize('bad', [[1., 2.], [1., 2., np.inf], [[1.], [2.], [3.]], [1., 2., 1j]])
def test_invalid_original_dual_refused(bad):
    plan = BoundedNearEqualityReduction(problem(), candidate())
    with pytest.raises(ValueError, match='Finite real vector'):
        plan.compress_dual(bad)
