import copy

import numpy as np
import pytest

from src.basis_coverage_selection import (
    basis_signature,
    greedy_set_cover,
    inventory_basis_statuses,
    validate_training_scope,
)


def _valid_statuses():
    # Two rows total: one basic column and one basic row.
    return np.array([1, 0, 2]), np.array([1, 0])


def test_basis_signature_is_stable_and_validates_complete_basis():
    columns, rows = _valid_statuses()
    assert basis_signature(columns, rows) == basis_signature(columns.copy(), rows.copy())
    assert basis_signature(columns, rows) != basis_signature(np.array([0, 1, 2]), rows)
    with pytest.raises(ValueError, match='exactly one basic'):
        basis_signature(np.array([0, 0, 2]), rows)
    with pytest.raises(ValueError, match='integers'):
        basis_signature(columns.astype(float), rows)


def test_inventory_deduplicates_and_records_only_adjacent_trajectory_changes():
    a_col, a_row = _valid_statuses()
    b_col, b_row = np.array([0, 1, 2]), a_row
    columns = np.stack([a_col, b_col, a_col, a_col])
    rows = np.stack([a_row, b_row, a_row, a_row])
    inventory = inventory_basis_statuses(columns, rows, [11, 11, 22, 22], [1, 2, 1, 2])
    assert inventory['states'] == 4
    assert inventory['unique_basis_count'] == 2
    assert inventory['counts'] == [3, 1]
    assert inventory['state_basis_ids'] == [0, 1, 0, 0]
    assert inventory['adjacent_comparisons'] == 2
    assert inventory['basis_change_count'] == 1
    assert inventory['adjacent_status_hamming'] == [2, 0]


def test_inventory_rejects_duplicate_or_reordered_trajectory_steps():
    columns, rows = _valid_statuses()
    table_c = np.stack([columns, columns])
    table_r = np.stack([rows, rows])
    with pytest.raises(ValueError, match='pairs must be unique'):
        inventory_basis_statuses(table_c, table_r, [1, 1], [1, 1])
    with pytest.raises(ValueError, match='increase'):
        inventory_basis_statuses(table_c, table_r, [1, 1], [2, 1])


def test_inventory_requires_contiguous_trajectory_blocks():
    columns, rows = _valid_statuses()
    table_c = np.stack([columns]*4)
    table_r = np.stack([rows]*4)
    with pytest.raises(ValueError, match='contiguous block'):
        inventory_basis_statuses(table_c, table_r, [1, 2, 1, 2], [1, 1, 2, 2])


def test_training_scope_guards_holdout_and_gem_identity():
    fingerprints = {'a': 'hash-a', 'b': 'hash-b'}
    manifest = dict(status='completed', actual_training_seeds=[101, 102],
        actual_training_steps=120,
        provenance=[dict(identity=dict(model_fingerprints=dict(fingerprints))),
                    dict(identity=dict(model_fingerprints=dict(fingerprints)))])
    scope = validate_training_scope(manifest, fingerprints, forbidden_seeds=[999], required_steps=120)
    assert scope.seeds == (101, 102)
    assert scope.expected_rows == 240
    with pytest.raises(ValueError, match='Forbidden'):
        validate_training_scope(manifest, fingerprints, forbidden_seeds=[102])
    changed = copy.deepcopy(manifest)
    changed['provenance'][1]['identity']['model_fingerprints']['a'] = 'different'
    with pytest.raises(ValueError, match='entry 1'):
        validate_training_scope(changed, fingerprints)


def test_greedy_cover_preserves_mandatory_and_uses_deterministic_gain():
    # Mandatory 0 covers state 0. Candidates 1/2/3 have equal initial gain, so
    # index 1 wins deterministically; index 2 then covers the remaining row.
    coverage = np.array([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    ], dtype=bool)
    result = greedy_set_cover(coverage, 3, mandatory=[0])
    assert result.selected == (0, 1, 2)
    assert result.marginal_weighted_gains == (1.0, 2.0, 1.0)
    assert result.covered_count == 4
    assert result.coverage_rate == 1.0


def test_greedy_cover_respects_byte_budget_and_never_treats_scores_as_coverage():
    coverage = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 0],
    ], dtype=bool)
    result = greedy_set_cover(coverage, 3, mandatory=[0],
        candidate_bytes=np.array([3, 8, 2]), max_bytes=5)
    assert result.selected == (0, 2)
    assert result.selected_bytes == 5
    assert result.covered_count == 2
    with pytest.raises(ValueError, match='boolean'):
        greedy_set_cover(coverage.astype(float), 2)


def test_greedy_cover_rejects_impossible_mandatory_budget():
    coverage = np.eye(3, dtype=bool)
    with pytest.raises(ValueError, match='candidate-count'):
        greedy_set_cover(coverage, 1, mandatory=[0, 1])
    with pytest.raises(ValueError, match='byte budget'):
        greedy_set_cover(coverage, 3, mandatory=[0, 1],
            candidate_bytes=np.array([4, 4, 1]), max_bytes=7)
    for invalid in ([True], [1.9]):
        with pytest.raises(ValueError, match='integers'):
            greedy_set_cover(coverage, 2, mandatory=invalid)


def test_uint64_candidate_cost_cannot_wrap_through_byte_budget():
    coverage = np.array([[1, 0], [0, 1]], dtype=bool)
    huge = np.uint64(np.iinfo(np.int64).max) + np.uint64(11)
    result = greedy_set_cover(coverage, 2, mandatory=[0],
        candidate_bytes=np.array([1, huge], dtype=np.uint64),
        max_bytes=int(huge))
    assert result.selected == (0,)
    assert result.selected_bytes == 1
