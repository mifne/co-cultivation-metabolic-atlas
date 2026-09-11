"""CPU algebra tests; CUDA comparison is explicitly selected by the GPU owner."""

import numpy as np
import pytest

from src.gpu_krylov_microkernels import givens_backsolve_inplace, mgs2_inplace


def _data(batch=3, width=37, capacity=8):
    rng = np.random.default_rng(46321)
    basis = np.stack([np.linalg.qr(rng.normal(size=(width, capacity)))[0].T
                      for _ in range(batch)])
    return np.ascontiguousarray(basis), rng.normal(size=(batch, width)), np.zeros(
        (batch, capacity, capacity - 1))


def _reference(basis, work, triangular, column, passes=2):
    work = work.copy()
    triangular = triangular.copy()
    for _ in range(passes):
        for row in range(column + 1):
            coefficient = np.sum(basis[:, row] * work, axis=1)
            triangular[:, row, column] += coefficient
            work -= coefficient[:, None] * basis[:, row]
    return work, triangular


@pytest.mark.parametrize('column', [0, 1, 6])
def test_numpy_matches_existing_two_pass_mgs_exactly(column):
    basis, work, triangular = _data()
    triangular[:] = .125
    before_basis = basis.copy()
    expected_work, expected_h = _reference(basis, work, triangular, column)
    mgs2_inplace(basis, work, triangular, column, xp=np)
    np.testing.assert_array_equal(work, expected_work)
    np.testing.assert_array_equal(triangular, expected_h)
    np.testing.assert_array_equal(basis, before_basis)


def test_second_pass_removes_cancellation_error_for_nearly_dependent_vector():
    basis, work, triangular = _data(batch=1, width=67)
    coefficients = np.arange(1., 8.)
    work[0] = coefficients @ basis[0, :7] + 1e-10 * work[0]
    once, _ = _reference(basis, work, triangular, 6, passes=1)
    once_defect = np.max(np.abs(basis[0, :7] @ once[0]))
    mgs2_inplace(basis, work, triangular, 6, xp=np)
    twice_defect = np.max(np.abs(basis[0, :7] @ work[0]))
    assert once_defect > 1e-17
    assert twice_defect < once_defect * 1e-5
    assert twice_defect < 1e-24


def test_inactive_zero_lane_stays_zero_and_other_lane_is_independent():
    basis, work, triangular = _data(batch=2)
    work[1] = 0.
    expected_work, expected_h = _reference(basis[:1], work[:1], triangular[:1], 5)
    mgs2_inplace(basis, work, triangular, 5, xp=np)
    np.testing.assert_array_equal(work[:1], expected_work)
    np.testing.assert_array_equal(triangular[:1], expected_h)
    np.testing.assert_array_equal(work[1], 0.)
    np.testing.assert_array_equal(triangular[1], 0.)


def test_nonfinite_result_is_not_silently_repaired():
    basis, work, triangular = _data(batch=2)
    work[0, 1] = np.nan
    expected_work, expected_h = _reference(basis[1:], work[1:], triangular[1:], 3)
    mgs2_inplace(basis, work, triangular, 3, xp=np)
    assert not np.isfinite(work[0]).all()
    assert not np.isfinite(triangular[0, :4, 3]).all()
    np.testing.assert_array_equal(work[1:], expected_work)
    np.testing.assert_array_equal(triangular[1:], expected_h)


@pytest.mark.parametrize('case', [
    'column_negative', 'column_large', 'column_bool', 'column_float',
    'basis_dtype', 'work_dtype', 'h_shape', 'work_shape', 'strided', 'overlap',
    'budget_large',
])
def test_rejects_invalid_shapes_layout_or_aliasing(case):
    basis, work, triangular = _data()
    column = 2
    if case == 'column_negative': column = -1
    elif case == 'column_large': column = 7
    elif case == 'column_bool': column = True
    elif case == 'column_float': column = 2.
    elif case == 'basis_dtype': basis = basis.astype(np.float32)
    elif case == 'work_dtype': work = work.astype(np.float32)
    elif case == 'h_shape': triangular = triangular[:, :, :-1].copy()
    elif case == 'work_shape': work = work[:, :-1].copy()
    elif case == 'strided': work = work[:, ::-1]
    elif case == 'overlap':
        basis = basis[:1]
        work = basis[:, 0]
        triangular = triangular[:1]
    elif case == 'budget_large':
        basis = np.zeros((1, 130, 131))
        work = np.zeros((1, 131))
        triangular = np.zeros((1, 130, 129))
    with pytest.raises(ValueError):
        mgs2_inplace(basis, work, triangular, column, xp=np)


@pytest.mark.parametrize('width,column', [(37, 0), (517, 6), (8349, 6)])
def test_cuda_mgs_matches_two_pass_reference(width, column):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA device required')
    basis, work, triangular = _data(width=width)
    # Cover a nearly dependent vector and an inactive lane in the same batch.
    work[0] = np.arange(1., column + 2.) @ basis[0, :column + 1] + 1e-10 * work[0]
    work[2] = 0.
    expected_work, expected_h = _reference(basis, work, triangular, column)
    dbasis, dwork, dh = map(cp.asarray, (basis, work, triangular))
    mgs2_inplace(dbasis, dwork, dh, column, xp=cp)
    actual_work, actual_h = cp.asnumpy(dwork), cp.asnumpy(dh)
    np.testing.assert_allclose(actual_work, expected_work, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(actual_h, expected_h, rtol=2e-12, atol=2e-14)
    np.testing.assert_array_equal(cp.asnumpy(dbasis), basis)
    np.testing.assert_array_equal(actual_work[2], 0.)
    if column:
        assert np.max(np.abs(basis[0, :column + 1] @ actual_work[0])) < 1e-23


def _givens_data(batch=3, budget=6):
    rng = np.random.default_rng(47432)
    h = np.triu(rng.normal(size=(batch, budget + 1, budget)), k=-1)
    for row in range(budget):
        h[:, row, row] += 4.
    cosine = np.zeros((batch, budget))
    sine = np.zeros_like(cosine)
    projected_rhs = np.zeros((batch, budget + 1))
    projected_rhs[:, 0] = np.arange(1., batch + 1.)
    return h, cosine, sine, projected_rhs, np.ones(batch, dtype=np.bool_)


def _givens_reference(h, cosine, sine, projected_rhs, active, column):
    # Same ordinary vector expressions as the unoptimized GMRES body.
    for row in range(column):
        upper = h[:, row, column].copy()
        lower = h[:, row + 1, column].copy()
        h[:, row, column] = cosine[:, row] * upper + sine[:, row] * lower
        h[:, row + 1, column] = -sine[:, row] * upper + cosine[:, row] * lower
    upper, lower = h[:, column, column].copy(), h[:, column + 1, column].copy()
    diagonal = np.hypot(upper, lower)
    active = active & np.isfinite(diagonal) & (diagonal != 0.)
    denominator = np.where(active, diagonal, 1.)
    cosine[:, column] = np.where(active, upper / denominator, 0.)
    sine[:, column] = np.where(active, lower / denominator, 0.)
    h[:, column, column] = diagonal
    h[:, column + 1, column] = 0.
    old_rhs = projected_rhs[:, column].copy()
    projected_rhs[:, column] = cosine[:, column] * old_rhs
    projected_rhs[:, column + 1] = -sine[:, column] * old_rhs
    coefficients = np.zeros((len(active), column + 1))
    for row in range(column, -1, -1):
        tail = np.sum(h[:, row, row + 1:column + 1]
                      * coefficients[:, row + 1:column + 1], axis=1)
        denominator = np.where(active, h[:, row, row], 1.)
        coefficients[:, row] = np.where(active, (projected_rhs[:, row] - tail) / denominator, 0.)
    return coefficients, diagonal


def test_givens_sequence_matches_original_and_direct_least_squares():
    values = _givens_data()
    reference = [value.copy() for value in values]
    unrotated_h = values[0].copy()
    original_rhs = values[3].copy()
    for column in range(values[0].shape[2]):
        expected, expected_diag = _givens_reference(*reference, column)
        actual, actual_diag = givens_backsolve_inplace(*values, column, xp=np)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_diag, expected_diag)
        for before, after in zip(reference, values):
            np.testing.assert_array_equal(after, before)
        for lane in range(len(actual)):
            least_squares = np.linalg.lstsq(unrotated_h[lane, :column + 2, :column + 1],
                                           original_rhs[lane, :column + 2], rcond=None)[0]
            np.testing.assert_allclose(actual[lane], least_squares, atol=2e-14, rtol=2e-14)


def test_givens_stable_hypot_does_not_square_large_entries():
    h, cosine, sine, projected_rhs, active = _givens_data(batch=1, budget=1)
    h[:, :, 0] = 1e200
    projected_rhs[:, 0] = 1e200
    coefficients, diagonal = givens_backsolve_inplace(
        h, cosine, sine, projected_rhs, active, 0, xp=np)
    np.testing.assert_allclose(diagonal, np.sqrt(2.) * 1e200)
    np.testing.assert_allclose(coefficients, .5)
    assert np.isfinite(coefficients).all()


def test_givens_nonfinite_singular_and_inactive_lanes_are_explicit():
    values = _givens_data(batch=5, budget=2)
    h, _, _, _, active = values
    h[0, 0, 0] = np.nan
    h[1, 0, 0] = np.inf
    h[2, :2, 0] = 0.
    active[3] = False
    reference = [value.copy() for value in values]
    with np.errstate(invalid='ignore'):
        expected, expected_diag = _givens_reference(*reference, 0)
        coefficients, diagonal = givens_backsolve_inplace(*values, 0, xp=np)
    np.testing.assert_array_equal(coefficients, expected)
    np.testing.assert_array_equal(diagonal, expected_diag)
    np.testing.assert_array_equal(coefficients[:4], 0.)
    assert np.isnan(diagonal[0])
    assert np.isinf(diagonal[1])
    assert diagonal[2] == 0.
    assert np.isfinite(coefficients[4]).all()
    np.testing.assert_array_equal(active, [True, True, True, False, True])


def test_givens_earlier_singular_diagonal_propagates_to_candidate_guard():
    h, cosine, sine, projected_rhs, active = _givens_data(batch=1, budget=2)
    h[:, 0, 0] = 0.
    cosine[:, 0] = 1.
    projected_rhs[:, 1] = 1.
    with np.errstate(divide='ignore', invalid='ignore'):
        coefficients, diagonal = givens_backsolve_inplace(
            h, cosine, sine, projected_rhs, active, 1, xp=np)
    assert np.isfinite(diagonal).all() and (diagonal > 0.).all()
    assert not np.isfinite(coefficients).all()


@pytest.mark.parametrize('case', [
    'column_negative', 'column_large', 'column_bool', 'h_shape', 'h_dtype',
    'cosine_shape', 'sine_dtype', 'rhs_shape', 'mask_dtype', 'mask_shape',
    'strided', 'overlap', 'budget_large',
])
def test_givens_validation(case):
    h, cosine, sine, projected_rhs, active = _givens_data()
    column = 2
    if case == 'column_negative': column = -1
    elif case == 'column_large': column = 6
    elif case == 'column_bool': column = True
    elif case == 'h_shape': h = h[:, :-1].copy()
    elif case == 'h_dtype': h = h.astype(np.float32)
    elif case == 'cosine_shape': cosine = cosine[:, :-1].copy()
    elif case == 'sine_dtype': sine = sine.astype(np.float32)
    elif case == 'rhs_shape': projected_rhs = projected_rhs[:, :-1].copy()
    elif case == 'mask_dtype': active = active.astype(np.float64)
    elif case == 'mask_shape': active = active[:, None]
    elif case == 'strided': cosine = cosine[:, ::-1]
    elif case == 'overlap': sine = cosine
    elif case == 'budget_large': h, cosine, sine, projected_rhs, active = _givens_data(budget=129)
    with pytest.raises(ValueError):
        givens_backsolve_inplace(h, cosine, sine, projected_rhs, active, column, xp=np)


def test_cuda_givens_sequence_and_invalid_lane_match_reference():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA device required')
    values = _givens_data(batch=5, budget=16)
    values[0][0, 0, 0] = np.nan
    values[0][1, :2, 0] = 0.
    values[4][2] = False
    device_values = [cp.asarray(value) for value in values]
    for column in range(16):
        expected, expected_diag = _givens_reference(*values, column)
        actual, actual_diag = givens_backsolve_inplace(*device_values, column, xp=cp)
        np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(cp.asnumpy(actual_diag), expected_diag, atol=2e-13, rtol=2e-13)
        for host, device in zip(values, device_values):
            np.testing.assert_allclose(cp.asnumpy(device), host, atol=2e-13, rtol=2e-13)
        # Mirror the existing caller's per-lane current-diagonal rejection.
        values[4][...] &= np.isfinite(expected_diag) & (expected_diag != 0.)
        device_values[4] &= cp.isfinite(actual_diag) & (actual_diag != 0.)
