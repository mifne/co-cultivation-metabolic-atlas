"""Small CPU-only QR/provenance tests; no real GEM or GPU execution."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.probe_ipm_equality_rank import diagnose_equality_rank, prepare_ipm_equalities
from src.lp_trace import problem_hash


def test_independent_rows_have_full_rank_and_no_deletion():
    matrix = csr_matrix(np.eye(3))
    rhs = np.array([1., 2., 3.])
    before = matrix.copy()
    report = diagnose_equality_rank(matrix, rhs)
    assert report['structural_rank'] == 3
    assert set(report['rank_estimates'].values()) == {3}
    assert report['candidate_dependent_row_indices'] == []
    assert report['rows_deleted'] == 0 and not report['exact_dependency_proven']
    assert report['cpu_lp_calls'] == 0 and report['gpu_calls'] == 0
    np.testing.assert_array_equal(matrix.toarray(), before.toarray())
    np.testing.assert_array_equal(rhs, [1., 2., 3.])


def test_numerically_dependent_rows_reconstruct_with_consistent_rhs():
    matrix = csr_matrix([[1., 0.], [0., 1.], [1., 1.]])
    report = diagnose_equality_rank(matrix, np.array([1., 2., 3.]))
    assert report['structural_rank'] == 2 and report['nominated_rank'] == 2
    assert len(report['candidate_reconstructions']) == 1
    candidate = report['candidate_reconstructions'][0]
    assert candidate['scaled_matrix_reconstruction_inf'] < 1e-14
    assert candidate['matrix_consistent_at_nominated_threshold']
    assert candidate['rhs_consistent_at_nominated_threshold']
    assert not report['exact_dependency_proven']


def test_structural_full_rank_does_not_prove_numeric_full_rank():
    matrix = csr_matrix([[1., 2., 3.], [2., 4., 6.], [0., 1., 1.]])
    report = diagnose_equality_rank(matrix, np.array([1., 2., 3.]))
    assert report['structural_rank'] == 3
    assert report['nominated_rank'] == 2
    assert report['structural_rank_is_only_an_upper_bound']


def test_near_dependence_changes_with_threshold_and_is_not_exact_claim():
    matrix = csr_matrix([[1., 0.], [1., 1e-11]])
    report = diagnose_equality_rank(matrix, np.array([1., 1.]))
    assert report['rank_estimates']['1e-10'] == 1
    assert report['rank_estimates']['1e-12'] == 2
    assert report['rank_estimates']['1e-14'] == 2
    assert not report['exact_dependency_proven'] and report['rows_deleted'] == 0


def test_dependent_matrix_with_inconsistent_rhs_is_reported_not_solved():
    matrix = csr_matrix([[1., 0.], [0., 1.], [1., 1.]])
    report = diagnose_equality_rank(matrix, np.array([1., 2., 9.]))
    candidate = report['candidate_reconstructions'][0]
    assert candidate['matrix_consistent_at_nominated_threshold']
    assert not candidate['rhs_consistent_at_nominated_threshold']
    assert candidate['rhs_backward_error'] > 1e-2
    assert report['cpu_lp_calls'] == 0


def test_zero_rows_and_zero_rank_remain_explicit_in_diagnostic():
    report = diagnose_equality_rank(csr_matrix((2, 3)), np.array([0., 1.]))
    assert report['structural_rank'] == 0 and report['nominated_rank'] == 0
    assert report['row_normalization']['zero_rows'] == [0, 1]
    by_row = {c['row']: c for c in report['candidate_reconstructions']}
    assert by_row[0]['rhs_consistent_at_nominated_threshold']
    assert not by_row[1]['rhs_consistent_at_nominated_threshold']


def test_row_normalization_preserves_rank_across_large_row_scales():
    matrix = np.array([[1., 0.], [0., 1.], [1., 1.]])
    divisor = np.array([1e-9, 1e9, 4.])
    report = diagnose_equality_rank(csr_matrix(divisor[:, None]*matrix), divisor*np.array([1., 2., 3.]))
    assert report['nominated_rank'] == 2
    assert all(c['rhs_consistent_at_nominated_threshold'] for c in report['candidate_reconstructions'])
    np.testing.assert_array_equal(report['row_normalization']['divisors'], divisor)


def test_candidate_reporting_is_bounded_without_hiding_dependency_count():
    matrix = csr_matrix([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    report = diagnose_equality_rank(matrix, np.array([1., 2., 3., 4.]), max_candidates=1)
    assert report['nominated_rank'] == 1
    assert len(report['candidate_dependent_row_indices']) == 3
    assert report['reconstructed_candidate_count'] == 1 and report['omitted_candidate_count'] == 2


def test_dense_memory_and_dimension_guards_run_before_qr(monkeypatch):
    import scripts.probe_ipm_equality_rank as module
    def forbidden(*args, **kwargs):
        raise AssertionError('QR must not run after an allocation guard fails')
    monkeypatch.setattr(module, 'qr', forbidden)
    with pytest.raises(ValueError, match='memory'):
        diagnose_equality_rank(csr_matrix(np.eye(3)), np.zeros(3), max_memory_mb=1e-8)
    with pytest.raises(ValueError, match='dimensions'):
        diagnose_equality_rank(csr_matrix(np.eye(3)), np.zeros(3), max_dimension=2)


@pytest.mark.parametrize('matrix,rhs', [
    (np.array([[np.nan]]), np.array([0.])),
    (np.array([[np.inf]]), np.array([0.])),
    (np.array([[1.]]), np.array([np.inf])),
    (np.eye(2), np.zeros((2, 1))),
    (np.eye(2), np.zeros(1)),
    (np.empty((0, 2)), np.zeros(0)),
    (np.empty((2, 0)), np.zeros(2)),
    (np.array([[1.+1j]]), np.array([1.])),
])
def test_invalid_inputs_fail_explicitly(matrix, rhs):
    with pytest.raises(ValueError):
        diagnose_equality_rank(matrix, rhs)


def test_underflowing_row_normalization_is_not_silent_rank_reduction():
    with pytest.raises(ValueError, match='normalization'):
        diagnose_equality_rank(csr_matrix([[1e308, 1e-308]]), np.array([0.]))


def test_preprocessing_matches_forest_zero_face_and_keeps_source_row_identity():
    matrix = csr_matrix([[1., -1., 0.], [0., 1., -1.], [1., 1., 1.], [1., 0., 0.]])
    problem = (matrix, np.array([0., 0., 3., 10.]), np.zeros(3), np.full(3, 10.),
               np.array([1., 2., 3.]), 3)
    initial_hash = problem_hash(problem)
    equality, rhs, preparation = prepare_ipm_equalities(problem)
    assert equality.shape == (1, 1)
    assert diagnose_equality_rank(equality, rhs)['nominated_rank'] == 1
    assert preparation['standard_equality_row_sources'] == [
        dict(kind='original_lp_row_after_column_transforms', original_row=2)]
    assert preparation['original_lp_sha256'] == initial_hash == problem_hash(problem)
