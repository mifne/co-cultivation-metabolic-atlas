from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.cpu_repeated_lp import _certificate
from src.lp_trace import problem_hash
from src.lp_zero_face import InfeasibleZeroRow, ZeroFaceReduction


def problem(a, rhs, lo, hi, c, neq):
    return (csr_matrix(a, dtype=float), np.array(rhs, dtype=float), np.array(lo, dtype=float),
            np.array(hi, dtype=float), np.array(c, dtype=float), neq)


def certify(p, x, y):
    solution = SimpleNamespace(col_value=np.asarray(x), row_dual=np.asarray(y),
        col_dual=p[4]-p[0].T@np.asarray(y), value_valid=True, dual_valid=True)
    return _certificate(*p, solution)


def test_min_face_lifts_cost_sign_and_full_original_gap():
    p = problem([[1, 2, 0], [0, 0, -1]], [0, -1], [0, 0, 0], [4, 5, 2], [-3, -8, 1], 1)
    reduction = ZeroFaceReduction(p)
    assert reduction.forced_zero_columns.tolist() == [0, 1]
    assert reduction.rows.tolist() == [1]
    assert reduction.reduced[-1] == 0
    assert certify(reduction.reduced, [1], [-1])['certificate_passed']
    raw_y = np.array([0., -1.])
    assert not certify(p, [0, 0, 1], raw_y)['certificate_passed']
    x, y = reduction.lift([1], [-1])
    assert y[0] == -4.
    assert certify(p, x, y)['certificate_passed']
    assert p[4]@x == reduction.reduced[4]@np.array([1.])+reduction.objective_offset


@pytest.mark.parametrize('row,lo,hi,c,orientation', [
    ([-1, -2], [0, 0], [3, 4], [-2, -6], 'max'),
    ([1, -2], [-3, 0], [0, 4], [2, -6], 'max'),
    ([-1, 2], [-3, 0], [0, 4], [2, -6], 'min'),
])
def test_negative_coefficients_and_both_orientations(row, lo, hi, c, orientation):
    p = problem([row], [0], lo, hi, c, 1)
    reduction = ZeroFaceReduction(p)
    assert reduction.witnesses[0].orientation == orientation
    assert reduction.reduced[0].shape == (0, 0)
    x, y = reduction.lift([], [])
    assert certify(p, x, y)['certificate_passed']


def test_cascading_witnesses_reverse_order_and_redundant_rows():
    # row0 cannot force anything initially: x0=x1. row1 first fixes x0,
    # and only the next pass fixes x1. Reverse dual lifting must repair x0
    # after the x1 witness changes its reduced cost.
    p = problem([[1, -1, 0], [1, 0, 0], [2, 0, 0], [0, 0, -1]],
        [0, 0, 0, -1], [0, 0, 0], [3, 3, 2], [-1, -9, 1], 3)
    reduction = ZeroFaceReduction(p)
    assert [w.row for w in reduction.witnesses] == [1, 0]
    assert [w.newly_fixed_columns for w in reduction.witnesses] == [(0,), (1,)]
    assert reduction.removed_zero_rows.tolist() == [0, 1, 2]
    x, y = reduction.lift([1], [-1])
    assert certify(p, x, y)['certificate_passed']
    assert y[2] == 0.


def test_explicit_finite_fixed_substitution_and_objective_offset():
    p = problem([[1, 1], [-2, 0], [0, -1]], [5, -6, -1], [3, 0], [3, 3], [7, 1], 1)
    reduction = ZeroFaceReduction(p)
    assert reduction.explicit_fixed_columns.tolist() == [0]
    assert reduction.forced_zero_columns.size == 0
    assert reduction.rows.tolist() == [0, 2]
    assert reduction.reduced[1].tolist() == [2, -1]
    assert reduction.objective_offset == 21.
    x, y = reduction.lift([2], [1, 0])
    assert certify(p, x, y)['certificate_passed']


@pytest.mark.parametrize('a,rhs,lo,hi,neq', [
    ([[0]], [1e-300], [0], [2], 1),
    ([[0]], [-1e-300], [0], [2], 0),
    ([[1]], [2], [1], [1], 1),
    ([[1]], [0], [1], [1], 0),
])
def test_impossible_exact_zero_row_fails_closed(a, rhs, lo, hi, neq):
    with pytest.raises(InfeasibleZeroRow):
        ZeroFaceReduction(problem(a, rhs, lo, hi, [0], neq))


def test_tiny_nonzero_rhs_and_inequality_are_not_used_as_zero_proof():
    p = problem([[1, 1], [1, 0]], [1e-300, 0], [0, 0], [1, 1], [1, 1], 1)
    reduction = ZeroFaceReduction(p)
    assert not reduction.witnesses
    assert reduction.columns.tolist() == [0, 1]
    assert reduction.rows.tolist() == [0, 1]


def test_nonzero_subnormal_coefficients_do_not_underflow_into_zero_proof():
    p = problem([[1e-300, 1e-300]], [0], [1e-300, 0], [1, 1], [0, 0], 1)
    reduction = ZeroFaceReduction(p)
    assert not reduction.witnesses  # Multiplying coefficient*bound would underflow.


def test_free_variable_and_cancelling_bound_extrema_are_not_fixed():
    p = problem([[1, 1], [1, -1]], [0, 0], [-1, -np.inf], [1, np.inf], [0, 0], 2)
    reduction = ZeroFaceReduction(p)
    assert not reduction.witnesses
    assert reduction.columns.tolist() == [0, 1]


def test_source_ownership_readonly_snapshots_and_integrity():
    p = problem([[1, 0]], [0], [0, 0], [1, 1], [-2, 1], 1)
    source_hash = problem_hash(p)
    reduction = ZeroFaceReduction(p)
    assert problem_hash(p) == source_hash == reduction.original_hash
    p[0].data[0] = 17.
    p[2][0] = -3.
    assert problem_hash(reduction.original) == source_hash
    with pytest.raises(ValueError):
        reduction.columns[0] = 0
    with pytest.raises(ValueError):
        reduction.original[4][0] = 100.
    x, y = reduction.lift([0], [])
    assert certify(reduction.original, x, y)['certificate_passed']
    reduction.original[0].data = np.array([22.])
    with pytest.raises(ValueError, match='snapshot'):
        reduction.lift([0], [])


@pytest.mark.parametrize('x,y', [([1, 2], []), ([np.nan], []), ([0], [1]), ([0], [np.inf])])
def test_invalid_lift_inputs_are_rejected(x, y):
    reduction = ZeroFaceReduction(problem([[1, 0]], [0], [0, 0], [1, 1], [0, 1], 1))
    with pytest.raises(ValueError):
        reduction.lift(x, y)


def test_original_explicit_zero_needs_no_forced_dual_sign():
    p = problem([[1, 1]], [0], [0, 0], [0, 1], [-100, -1], 1)
    reduction = ZeroFaceReduction(p)
    assert reduction.explicit_fixed_columns.tolist() == [0]
    assert reduction.witnesses[0].newly_fixed_columns == (1,)
    x, y = reduction.lift([], [])
    assert certify(p, x, y)['certificate_passed']


def test_exact_equality_duplicates_preserve_first_representative_and_duals():
    p = problem([[1, 1], [1, 1], [-1, -1], [-1, 0], [-1, 0]],
        [2, 2, -2, 0, 0], [0, 0], [2, 2], [1, 1], 3)
    reduction = ZeroFaceReduction(p)
    assert reduction.rows.tolist() == [0, 3, 4]  # Inequality duplicates stay.
    assert reduction.removed_duplicate_rows.tolist() == [1, 2]
    assert [(d.row, d.representative_row, d.sign) for d in reduction.duplicate_equalities] == [
        (1, 0, 1), (2, 0, -1)]
    assert reduction.reduced[-1] == 1
    assert certify(reduction.reduced, [1, 1], [1, 0, 0])['certificate_passed']
    x, y = reduction.lift([1, 1], [1, 0, 0])
    np.testing.assert_array_equal(y, [1, 0, 0, 0, 0])
    assert certify(p, x, y)['certificate_passed']


def test_duplicate_sign_works_when_first_representative_is_negative():
    p = problem([[-1, -1], [1, 1], [-1, -1]], [-2, 2, -2],
        [0, 0], [2, 2], [1, 1], 3)
    reduction = ZeroFaceReduction(p)
    assert [(d.row, d.representative_row, d.sign) for d in reduction.duplicate_equalities] == [
        (1, 0, -1), (2, 0, 1)]
    x, y = reduction.lift([1, 1], [-1])
    assert certify(p, x, y)['certificate_passed']


def test_duplicate_removal_can_be_disabled_without_changing_certificate():
    p = problem([[1, 1], [-1, -1]], [2, -2], [0, 0], [2, 2], [1, 1], 2)
    reduction = ZeroFaceReduction(p, remove_duplicate_equalities=False)
    assert reduction.rows.tolist() == [0, 1]
    assert not reduction.duplicate_equalities
    x, y = reduction.lift([1, 1], [1, 0])
    assert certify(p, x, y)['certificate_passed']
    with pytest.raises(ValueError, match='boolean'):
        ZeroFaceReduction(p, remove_duplicate_equalities=1)


def test_no_proportional_near_equal_or_changed_rhs_matching():
    p = problem([[1, 1], [2, 2], [1, 1+1e-15], [1, 1]], [2, 4, 2, 2+1e-15],
        [0, 0], [3, 3], [0, 0], 4)
    reduction = ZeroFaceReduction(p)
    assert reduction.rows.tolist() == [0, 1, 2, 3]
    assert not reduction.duplicate_equalities


def test_duplicates_after_nonzero_fixed_substitution_preserve_objective():
    p = problem([[1, 1, 0], [0, 1, 0], [0, -1, 0]], [2, 1, -1],
        [1, 0, 0], [1, 2, 2], [3, 1, 1], 3)
    reduction = ZeroFaceReduction(p)
    assert reduction.rows.tolist() == [0]
    assert reduction.removed_duplicate_rows.tolist() == [1, 2]
    x, y = reduction.lift([1, 0], [1])
    assert certify(p, x, y)['certificate_passed']
    assert p[4]@x == 4. == reduction.objective_offset+reduction.reduced[4]@np.array([1, 0])


def test_zero_witness_dual_postsolve_with_nonzero_equality_duplicates():
    p = problem([[1, -1, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0],
                 [0, 0, 1, 1], [0, 0, -1, -1]],
        [0, 0, 0, 2, -2], [0, 0, 0, 0], [3, 3, 2, 2], [-1, -9, 1, 1], 5)
    reduction = ZeroFaceReduction(p)
    assert [w.row for w in reduction.witnesses] == [1, 0]
    assert reduction.rows.tolist() == [3]
    assert reduction.removed_zero_rows.tolist() == [0, 1, 2]
    assert reduction.removed_duplicate_rows.tolist() == [4]
    x, y = reduction.lift([1, 1], [1])
    assert y[0] != 0. and y[1] != 0.  # Witness rows restored at ORIGINAL indices.
    assert y[2] == 0. and y[4] == 0.
    assert certify(p, x, y)['certificate_passed']
