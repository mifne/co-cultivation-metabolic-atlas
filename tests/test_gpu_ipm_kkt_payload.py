"""Exact CPU reference comparisons for KKT scatter; no CUDA execution."""

import numpy as np
import pytest
from scipy.sparse import bmat, csr_matrix, diags

from src.gpu_batched_ipm import uniform_kkt_pattern
from src.gpu_ipm_kkt_payload import build_condensed_kkt_payload


def _forms(n, ne, q, *, batch=3, seed=0):
    rng = np.random.default_rng(seed)
    forms = []
    for _lane in range(batch):
        e = rng.normal(size=(ne, n))
        g = rng.normal(size=(q+2, n))
        e[rng.random(e.shape) < .55] = 0.
        g[rng.random(g.shape) < .55] = 0.
        e, g = csr_matrix(e), csr_matrix(g)
        # Explicit stored zero is part of the symbolic union, unlike an entry
        # absent from one lane. Include signed zero to check scatter values.
        if e.nnz: e.data[0] = -0.
        if g.nnz: g.data[-1] = 0.
        forms.append((e, np.zeros(ne), g))
    return forms


def _legacy(forms, n, ne, q, delta=1e-6):
    matrices = [bmat([[diags(np.ones(n)), f[0].T, f[2][:q].T],
        [f[0], diags(-np.ones(ne)), None],
        [f[2][:q], None, diags(-np.ones(q))]], format='csr') for f in forms]
    pattern, values, diagonal = uniform_kkt_pattern(matrices)
    values[:, diagonal[:n]] = delta
    values[:, diagonal[n:n+ne]] = -delta
    return pattern, values, diagonal


@pytest.mark.parametrize('n,ne,q', [(1, 0, 0), (3, 0, 2), (4, 2, 0), (5, 3, 4), (2, 4, 3)])
@pytest.mark.parametrize('seed', range(8))
def test_random_heterogeneous_lanes_match_bmat_union_bit_for_bit(n, ne, q, seed):
    forms = _forms(n, ne, q, seed=seed)
    pattern, expected, expected_diagonal = _legacy(forms, n, ne, q)
    actual, diagonal = build_condensed_kkt_payload(forms, pattern,
        n=n, ne=ne, q=q, regularization=1e-6)
    np.testing.assert_array_equal(diagonal, expected_diagonal)
    # Also retain signed-zero bits; no arithmetic reduction/reordering occurs.
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert not np.shares_memory(actual, pattern.data)
    assert not np.shares_memory(diagonal, pattern.indices)


def test_explicit_zero_and_empty_lane_union_coordinates_are_retained():
    first = (csr_matrix(([0.], ([0], [1])), shape=(1, 3)), None, csr_matrix((0, 3)))
    second = (csr_matrix(([2.], ([0], [0])), shape=(1, 3)), None, csr_matrix((0, 3)))
    pattern, expected, expected_diagonal = _legacy([first, second], 3, 1, 0)
    values, diagonal = build_condensed_kkt_payload([first, second], pattern,
        n=3, ne=1, q=0, regularization=1e-6)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(diagonal, expected_diagonal)
    assert pattern.nnz == 8  # 4 diagonal + two structural E entries in both triangles.


@pytest.mark.parametrize('change', ['missing_diagonal', 'missing_lower', 'missing_upper',
                                   'extra_symmetric', 'extra_asymmetric'])
def test_pattern_requires_exact_coordinates_not_a_superset(change):
    forms = [(csr_matrix([[1., 2., 3.]]), None, csr_matrix([[4., 5., 6.]]))]
    pattern, _, _ = _legacy(forms, 3, 1, 1)
    changed = pattern.tolil()
    if change == 'missing_diagonal': changed[0, 0] = 0.
    elif change == 'missing_lower': changed[3, 0] = 0.
    elif change == 'missing_upper': changed[0, 3] = 0.
    else:
        changed[0, 1] = 1.
        if change == 'extra_symmetric': changed[1, 0] = 1.
    with pytest.raises(ValueError, match='missing|extra'):
        build_condensed_kkt_payload(forms, changed.tocsr(), n=3, ne=1, q=1, regularization=1e-6)


def test_pattern_of_one_lane_cannot_hide_coordinates_of_another():
    forms = [(csr_matrix([[1., 0., 0.]]), None, csr_matrix((0, 3))),
             (csr_matrix([[0., 2., 0.]]), None, csr_matrix((0, 3)))]
    incomplete, _, _ = _legacy(forms[:1], 3, 1, 0)
    with pytest.raises(ValueError, match='missing lane 1'):
        build_condensed_kkt_payload(forms, incomplete, n=3, ne=1, q=0, regularization=1e-6)


@pytest.mark.parametrize('target', ['pattern', 'E', 'G'])
@pytest.mark.parametrize('change', ['nan', 'inf', 'fp32', 'unsorted', 'duplicate', 'bad_index',
                                   'bad_pointer', 'bad_pointer_length'])
def test_raw_csr_validation_rejects_corruption_even_with_cached_canonical_flags(target, change):
    forms = [(csr_matrix([[1., 2., 3.]]), None, csr_matrix([[4., 5., 6.]]))]
    pattern, _, _ = _legacy(forms, 3, 1, 1)
    matrix = pattern if target == 'pattern' else forms[0][0 if target == 'E' else 2]
    assert matrix.has_canonical_format
    if change == 'nan': matrix.data[0] = np.nan
    elif change == 'inf': matrix.data[0] = np.inf
    elif change == 'fp32': matrix.data = matrix.data.astype(np.float32)
    elif change == 'unsorted': matrix.indices[:2] = matrix.indices[:2][::-1]
    elif change == 'duplicate': matrix.indices[1] = matrix.indices[0]
    elif change == 'bad_index': matrix.indices[0] = matrix.shape[1]
    elif change == 'bad_pointer': matrix.indptr[0] = 1
    else: matrix.indptr = matrix.indptr[:-1]
    with pytest.raises(ValueError):
        build_condensed_kkt_payload(forms, pattern, n=3, ne=1, q=1, regularization=1e-6)


@pytest.mark.parametrize('change', ['empty_batch', 'bad_form', 'n_zero', 'ne_negative', 'q_bool',
                                   'n_large', 'E_shape', 'G_shape', 'pattern_shape', 'not_csr'])
def test_incompatible_dimensions_and_inputs_are_rejected(change):
    forms = _forms(3, 1, 2)
    pattern, _, _ = _legacy(forms, 3, 1, 2)
    n, ne, q = 3, 1, 2
    if change == 'empty_batch': forms = []
    elif change == 'bad_form': forms[0] = []
    elif change == 'n_zero': n = 0
    elif change == 'ne_negative': ne = -1
    elif change == 'q_bool': q = True
    elif change == 'n_large': n = 2**31
    elif change == 'E_shape': forms[0] = (forms[0][0][:, :2], None, forms[0][2])
    elif change == 'G_shape': forms[0] = (forms[0][0], None, forms[0][2][:1])
    elif change == 'pattern_shape': pattern = pattern[:, :-1]
    else: pattern = pattern.tocsc()
    with pytest.raises(ValueError):
        build_condensed_kkt_payload(forms, pattern, n=n, ne=ne, q=q, regularization=1e-6)


@pytest.mark.parametrize('delta', [0., -1., np.nan, np.inf, True, np.array(1e-6)])
def test_invalid_regularization_rejected(delta):
    forms = _forms(2, 1, 1)
    pattern, _, _ = _legacy(forms, 2, 1, 1)
    with pytest.raises(ValueError, match='regularization'):
        build_condensed_kkt_payload(forms, pattern, n=2, ne=1, q=1, regularization=delta)


def test_owned_readonly_inputs_remain_unchanged_and_output_is_independent():
    forms = _forms(3, 2, 1)
    pattern, _, _ = _legacy(forms, 3, 2, 1)
    arrays = [getattr(matrix, name) for matrix in [pattern, *(f[0] for f in forms), *(f[2] for f in forms)]
              for name in ('data', 'indices', 'indptr')]
    saved = [value.copy() for value in arrays]
    for value in arrays: value.flags.writeable = False
    values, diagonal = build_condensed_kkt_payload(forms, pattern, n=3, ne=2, q=1, regularization=1e-6)
    values[:] = 7.; diagonal[:] = 0
    for value, expected in zip(arrays, saved): np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize('exact', [False, True])
def test_opt_in_numeric_rebind_avoids_bmat_union_and_matches_legacy_payloads(monkeypatch, exact):
    import src.gpu_ipm_numeric_update as update
    from tests.test_gpu_ipm_numeric_update import _mock_solver, _problems
    solver = _mock_solver(exact_equalities=exact)
    states = update.prepare_host_rebind(solver, _problems(1))
    for state in states:
        legacy = update._payloads(solver, state)
        direct = update._payloads(solver, state, _direct_kkt_payload=True)
        for path, expected in legacy[1].items():
            np.testing.assert_array_equal(direct[1][path], expected)
    def forbidden(*_args, **_kwargs): raise AssertionError('Legacy KKT construction called')
    monkeypatch.setattr(update, 'bmat', forbidden)
    monkeypatch.setattr(update, 'uniform_kkt_pattern', forbidden)
    metadata = update.rebind_forest_ipm(solver, _problems(1), direct_kkt_payload=True)
    assert metadata['direct_kkt_payload'] and not solver.factor.factored


def test_direct_rebind_corrupt_pattern_rejection_is_precommit_and_nonmutating():
    import src.gpu_ipm_numeric_update as update
    from tests.test_gpu_ipm_numeric_update import _mock_solver, _problems, _device_snapshot, _assert_snapshot
    solver = _mock_solver()
    saved = _device_snapshot(solver)
    state, hashes = solver._last_internal_state, solver.problem_hashes
    solver.factor.host_pattern.indices[1] = solver.factor.host_pattern.indices[0]
    with pytest.raises(update.NumericRebindRejected):
        update.rebind_forest_ipm(solver, _problems(1), direct_kkt_payload=True)
    _assert_snapshot(solver, saved)
    assert solver._last_internal_state is state and solver.problem_hashes == hashes
    assert solver.factor.factored and not solver.factor.failed
