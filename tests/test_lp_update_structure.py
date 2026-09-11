"""Small sparse-algebra input audits; no native optimizer or CUDA required."""
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.probe_lp_update_structure import (
    audit, bound_topology, compare_snapshots, map_snapshot,
    matrix_difference, problem_difference, problem_structure,
)


def problem(a, *, rhs=None, lo=None, hi=None, c=None, neq=None):
    a = csr_matrix(a, dtype=np.float64)
    n, m = a.shape[1], a.shape[0]
    return (a, np.zeros(m) if rhs is None else np.asarray(rhs, dtype=float),
            np.zeros(n) if lo is None else np.asarray(lo, dtype=float),
            np.ones(n) if hi is None else np.asarray(hi, dtype=float),
            np.ones(n) if c is None else np.asarray(c, dtype=float),
            m if neq is None else neq)


def test_objective_and_inequality_update_do_not_change_equality_proof_key():
    before = problem([[1, -1], [1, 0]], rhs=[0, 1], neq=1)
    after = problem([[1, -1], [2, 0]], rhs=[0, 2], c=[3, 4], neq=1)
    diff = problem_difference(before, after)
    assert diff['A']['same_csr_support']
    assert diff['A']['coefficient_changes'] == 1
    assert diff['A_equalities']['coefficient_changes'] == 0
    assert diff['same_keys']['exact_equality_rhs_key']
    assert diff['same_keys']['forest_construction_key']
    assert not diff['same_keys']['a_numeric_key']


def test_homogeneous_status_is_in_forest_construction_key():
    before = problem([[1, -1]], rhs=[0])
    after = problem([[1, -1]], rhs=[1])
    first, second = problem_structure(before), problem_structure(after)
    assert first['equality_numeric_key'] == second['equality_numeric_key']
    assert first['forest_construction_key'] != second['forest_construction_key']
    assert first['exact_equality_rhs_key'] != second['exact_equality_rhs_key']


def test_support_added_removed_and_numeric_change_are_distinct():
    old = csr_matrix([[1., 0., 2.], [0., 4., 0.]])
    new = csr_matrix([[2., 3., 0.], [0., 4., 0.]])
    diff = matrix_difference(old, new)
    assert diff['coefficient_changes'] == 3
    assert diff['support_added'] == diff['support_removed'] == 1
    assert diff['changed_rows'] == [0]
    assert diff['changed_columns'] == [0, 1, 2]


def test_numeric_bound_witness_changes_without_forest_coordinate_change():
    before = problem([[1, -1], [1, 0]], rhs=[0, 9], lo=[0, 0], hi=[1, 2], neq=1)
    after = problem([[1, -1], [1, 0]], rhs=[0, 9], lo=[0, 0], hi=[2, 1], neq=1)
    old, new = map_snapshot(before), map_snapshot(after)
    diff = compare_snapshots(old, new)
    assert diff['same_maps']['forest']['coordinate_key']
    assert not diff['same_maps']['forest']['bound_witness_key']
    assert bound_topology(before[2], before[3]) == bound_topology(after[2], after[3])


def test_bound_topology_alone_cannot_authorize_zero_face_map():
    # Three-entry equality is not eliminated by the first two-entry forest.
    before = problem([[1, 1, 1]], lo=[0, 0, 0], hi=[1, 1, 1])
    after = problem([[1, 1, 1]], lo=[-1, -1, -1], hi=[1, 1, 1])
    assert bound_topology(before[2], before[3]) == bound_topology(after[2], after[3])
    diff = compare_snapshots(map_snapshot(before), map_snapshot(after))
    assert not diff['same_maps']['zero_face']['coordinate_key']
    assert not diff['same_working_shape']


def test_sparse_map_audit_never_discovers_exact_qr_or_solves(monkeypatch):
    import scipy.optimize
    import src.lp_exact_equalities as exact
    import src.cpu_repeated_lp as cpu
    def forbidden(*args, **kwargs):
        raise AssertionError('No optimizer or dense QR allowed in input-only audit')
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    monkeypatch.setattr(exact, 'qr', forbidden)
    monkeypatch.setattr(exact.ExactEqualityReduction, '__init__', forbidden)
    monkeypatch.setattr(cpu.RepeatedCpuLP, '__init__', forbidden)
    p = problem([[1, -1, 0], [0, 1, 1], [1, 0, 1]],
                rhs=[0, 2, 3], lo=[-1, -1, -1], hi=[2, 2, 2], neq=2)
    answer = map_snapshot(p, fix_singletons=True, second_forest=True)
    assert not answer['exact_equality_proof_computed']
    json.dumps(answer, allow_nan=False)


@pytest.mark.parametrize('kwargs', [dict(batch=0), dict(batch=33), dict(batch=True),
    dict(steps=[2, 1]), dict(steps=[1, 1]), dict(steps=[True]), dict(stages=['unknown'])])
def test_invalid_bounded_configuration_fails_before_file_read(tmp_path, kwargs):
    with pytest.raises(ValueError):
        audit(tmp_path, **kwargs)
