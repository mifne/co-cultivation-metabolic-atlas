import copy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.audit_lp_equalities import (coefficient_statistics, equality_details,
    equality_hash, graph_elimination_bound, low_arity_graph, selected_entries)


def problem(matrix, *, rhs=None, neq=None):
    a = csr_matrix(matrix, dtype=float)
    rows, columns = a.shape
    return (a, np.zeros(rows) if rhs is None else np.array(rhs, dtype=float),
            np.zeros(columns), np.ones(columns), np.zeros(columns), rows if neq is None else neq)


def test_full_hash_tracks_suffix_shape_but_trimmed_hash_ignores_only_zero_suffix():
    first = problem([[1., -5e-5]])
    second = problem([[1., -5e-5, 0., 0.]])
    assert equality_hash(first) != equality_hash(second)
    assert equality_hash(first, trim_trailing_zero_columns=True) == equality_hash(second, trim_trailing_zero_columns=True)
    changed = problem([[1., -5e-5, 0., 1e-15]])
    assert equality_hash(first, trim_trailing_zero_columns=True) != equality_hash(changed, trim_trailing_zero_columns=True)


def test_hash_includes_rhs_coefficient_and_column_identity_exactly():
    base = problem([[1., -5e-5]])
    for changed in (problem([[1., -5e-5]], rhs=[1e-15]),
                    problem([[1., -5e-5+1e-15]]), problem([[-5e-5, 1.]])):
        assert equality_hash(base) != equality_hash(changed)


def test_inequality_changes_do_not_change_equality_hash():
    base = problem([[1., 2.], [3., 4.]], neq=1)
    changed = problem([[1., 2.], [300., 400.]], rhs=[0, 999], neq=1)
    assert equality_hash(base) == equality_hash(changed)


def test_exact_singleton_and_connected_two_entry_zero_propagation_bound():
    lp = problem([[1, 0, 0, 0], [1, -2, 0, 0], [0, 1, -3, 0]])
    graph = low_arity_graph(lp)
    estimate = graph_elimination_bound(graph, [])
    assert estimate["two_entry_forest_independent_relations"] == 2
    assert estimate["zero_columns_implied_by_small_rows_and_supplied_zero_bounds"] == 3
    assert estimate["eliminated_coordinates_lower_bound"] == 3
    assert estimate["remaining_coordinates_upper_bound"] == 1
    all_zero = graph_elimination_bound(graph, [3])
    assert all_zero["eliminated_coordinates_lower_bound"] == 4


def test_cycles_and_higher_arity_rows_not_counted_as_unproven_rank():
    # Inconsistent ratio around a homogeneous cycle forces zero, but the
    # deliberately conservative graph count does not claim that extra rank.
    lp = problem([[1, -1, 0], [0, 1, -1], [1, 0, -2], [1, 1, 1]])
    estimate = graph_elimination_bound(low_arity_graph(lp), [])
    assert estimate["two_entry_cycle_rows_not_counted_as_extra_independent"] == 1
    assert estimate["eliminated_coordinates_lower_bound"] == 2
    assert estimate["remaining_coordinates_upper_bound"] == 1


def test_nonhomogeneous_rows_are_not_used_for_zero_elimination():
    lp = problem([[1, 0], [1, -1]], rhs=[2, 1])
    estimate = graph_elimination_bound(low_arity_graph(lp), [])
    assert estimate["eliminated_coordinates_lower_bound"] == 0


def test_dangerous_coefficient_local_pivot_avoids_inverse_amplification():
    lp = problem([[1, -5e-5], [0, 2]])
    details = equality_details(lp)
    row = details["high_ratio_or_small_coefficient_two_entry_rows"][0]
    assert row["locally_nonamplifying_eliminated_column"] == 0
    assert row["remaining_column"] == 1
    assert row["local_substitution_factor"] == pytest.approx(5e-5)
    assert row["abs_coefficient_ratio"] == pytest.approx(20000)
    assert details["homogeneous_single_entry_rows"] == 1
    assert details["homogeneous_two_entry_rows"] == 1


def test_coefficient_range_is_not_misreported_as_condition_number():
    stats = coefficient_statistics(csr_matrix([[5e-5, 4800.]]))
    assert stats["abs_max_over_min"] == pytest.approx(9.6e7)
    assert stats["numerical_condition_number"] is None
    assert "NOT" in stats["dynamic_range_warning"]


def manifest():
    return dict(status="completed", seeds=[101, 102], completed_steps=[2, 2],
                entries=[dict(environment_id=e, step=t, stage=stage)
                         for e in range(2) for t in (1, 2)
                         for stage in ("maxmin", "aggregate", "exchange")])


def test_selected_trace_requires_all_stages_but_preserves_requested_subset():
    original = manifest()
    selected = selected_entries(original, [2])
    assert len(selected) == 6 and {entry["step"] for entry in selected} == {2}
    assert len(selected_entries(original)) == 12


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bad_steps", "unfinished"])
def test_invalid_trace_selection_rejected(mutation):
    data = copy.deepcopy(manifest())
    steps = None
    if mutation == "missing":
        data["entries"].pop()
    elif mutation == "duplicate":
        data["entries"].append(data["entries"][0])
    elif mutation == "bad_steps":
        steps = [1, 1]
    else:
        data["status"] = "running"
    with pytest.raises(ValueError):
        selected_entries(data, steps)
