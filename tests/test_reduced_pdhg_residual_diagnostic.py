"""CPU-only checks of the diagnostic decomposition (no optimizer)."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scripts.diagnose_reduced_pdhg_residuals import residual_breakdown


def problem(a, rhs, lower, upper, c, neq):
    return (csr_matrix(a,dtype=float),np.asarray(rhs,dtype=float),
            np.asarray(lower,dtype=float),np.asarray(upper,dtype=float),
            np.asarray(c,dtype=float),neq)


def test_bound_gap_is_not_confused_with_box_feasibility():
    p=problem(np.empty((0,1)),[],[0.],[2.],[-1.],0)
    r=residual_breakdown(p,[1.],[])
    assert r['certificate']['primal_residual'] == 0.
    assert r['bound_complementarity_sum'] == 1.
    assert r['row_complementarity_sum'] == 0.
    assert r['certificate']['relative_kkt_gap'] == 1.
    assert r['top_bound_complementarity_columns'][0]['column_index'] == 0


def test_infinite_bound_dual_violation_is_not_hidden_by_zero_gap():
    p=problem(np.empty((0,1)),[],[-np.inf],[np.inf],[-1.],0)
    r=residual_breakdown(p,[0.],[])
    assert r['bound_complementarity_sum'] == 0.
    assert r['dual_components']['no_finite_upper_reduced_cost_violation_max'] == 1.
    assert r['certificate']['dual_violation'] == 1.
    assert not r['certificate']['certificate_passed']


def test_remaining_row_labels_map_back_to_original_coordinates():
    p=problem([[1.],[1.]],[0.,2.],[0.],[3.],[-1.],1)
    r=residual_breakdown(p,[1.],[0.,-1.],kept_rows=[4,8])
    assert r['top_equality_residual_rows'][0]['original_row_index'] == 4
    assert r['top_row_complementarity_rows'][0]['original_row_index'] == 8
    assert r['row_complementarity_sum'] == 1.
    assert r['primal_components']['equality_abs_max'] == 1.


def test_exact_optimum_has_zero_decomposed_residuals():
    p=problem([[1.]],[1.],[0.],[2.],[-1.],0)
    r=residual_breakdown(p,[1.],[-1.])
    assert r['certificate']['certificate_passed']
    assert r['bound_complementarity_sum'] == r['row_complementarity_sum'] == 0.
    assert max(r['primal_components'].values()) == max(r['dual_components'].values()) == 0.


def test_diagnostic_refuses_wrong_coordinates():
    p=problem([[1.]],[1.],[0.],[2.],[-1.],0)
    with pytest.raises(ValueError,match='dimensions'):
        residual_breakdown(p,[0.,1.],[-1.])
