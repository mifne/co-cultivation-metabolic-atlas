"""CPU-only arithmetic tests of the NumPy/CuPy shared Newton GMRES kernel."""

import numpy as np
import pytest

from src.gpu_newton_krylov import (
    GMRES_BREAKDOWN,
    GMRES_CONVERGED,
    GMRES_ITERATION_LIMIT,
    GMRES_NONFINITE,
    batched_gmres,
)


def relative_error(rhs, residual):
    return np.max(np.abs(residual), axis=1) / np.maximum(1., np.max(np.abs(rhs), axis=1))


def solve(matrices, rhs, *, initial=None, regularization=0., **kwargs):
    if initial is None:
        initial = np.zeros_like(rhs)
    regularized = matrices + regularization * np.eye(rhs.shape[1])[None]
    return batched_gmres(
        rhs, initial, lambda x: np.einsum('bij,bj->bi', matrices, x),
        lambda x: np.linalg.solve(regularized, x[..., None])[..., 0],
        relative_error, xp=np, **kwargs)


def test_nonsymmetric_and_indefinite_systems_solve_original_equations():
    matrices = np.array([[[0., 2., -1.], [1., -3., 0.], [2., 1., 4.]],
                         [[2., 1., 0.], [1., -3., 2.], [0., 2., 0.]]])
    expected = np.array([[1., -2., .5], [-3., .25, 2.]])
    rhs = np.einsum('bij,bj->bi', matrices, expected)
    answer, info = solve(matrices, rhs, regularization=.2, tolerance=1e-13)
    np.testing.assert_allclose(answer, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(info['termination'], GMRES_CONVERGED)
    np.testing.assert_allclose(info['error'], relative_error(rhs, rhs - np.einsum(
        'bij,bj->bi', matrices, answer)), atol=0., rtol=0.)
    assert np.all((info['iterations'] > 0) & (info['iterations'] <= 3))
    assert np.max(np.abs(np.einsum('bij,bj->bi', matrices + .2*np.eye(3), answer) - rhs)) > .1


def test_nearly_singular_and_singular_consistent_systems():
    matrices = np.array([np.diag([1., 1e-11, -2.]), np.diag([1., 0., -2.])])
    expected = np.array([[2., 3., -4.], [2., 0., -4.]])
    rhs = np.einsum('bij,bj->bi', matrices, expected)
    # Per-component scaling requires recovering even the tiny second equation.
    def component_error(b, r):
        return np.max(np.abs(r) / np.maximum(np.abs(b), 1e-14), axis=1)

    regularized = matrices + 1e-4*np.eye(3)
    answer, info = batched_gmres(rhs, np.zeros_like(rhs),
        lambda x: np.einsum('bij,bj->bi', matrices, x),
        lambda x: np.linalg.solve(regularized, x[..., None])[..., 0],
        component_error, xp=np, tolerance=1e-10)
    np.testing.assert_allclose(answer, expected, atol=2e-9, rtol=2e-9)
    assert info['converged'].all(), info


def test_already_valid_initial_is_unchanged_and_never_preconditioned():
    rhs = np.array([[1., -2.], [0., 0.]])
    initial = rhs.copy()

    def forbidden(_):
        raise AssertionError('A converged initial vector needs no Arnoldi iteration')

    answer, info = batched_gmres(rhs, initial, lambda x: x.copy(), forbidden,
                                relative_error, xp=np)
    np.testing.assert_array_equal(answer, initial)
    np.testing.assert_array_equal(info['iterations'], 0)
    assert info['converged'].all()
    assert answer is not initial


def test_difficult_environment_does_not_activate_zero_rhs_environment():
    matrix = np.diag([1e-4, 1e-2, .2, 1., 4.]) + np.diag([1., -2., .5, 1.], 1)
    matrices = np.stack([matrix, np.eye(5)])
    rhs = np.array([[1., -1., 2., -2., 3.], [0., 0., 0., 0., 0.]])
    inputs = []

    def precondition(x):
        inputs.append(x.copy())
        return x.copy()

    answer, info = batched_gmres(rhs, np.zeros_like(rhs),
        lambda x: np.einsum('bij,bj->bi', matrices, x), precondition,
        relative_error, xp=np, max_iterations=3, tolerance=1e-14)
    assert info['iterations'].tolist() == [3, 0]
    assert info['termination'].tolist() == [GMRES_ITERATION_LIMIT, GMRES_CONVERGED]
    np.testing.assert_array_equal(answer[1], 0.)
    assert all(np.array_equal(x[1], np.zeros(5)) for x in inputs)
    assert info['error'][0] <= relative_error(rhs, rhs)[0]


def test_no_improvement_preserves_initial_and_reports_breakdown():
    rhs = np.array([[1., 2.], [-1., 3.]])
    initial = np.array([[4., 5.], [-6., 7.]])
    answer, info = batched_gmres(rhs, initial, lambda x: np.zeros_like(x),
        lambda x: x.copy(), relative_error, xp=np)
    np.testing.assert_array_equal(answer, initial)
    np.testing.assert_array_equal(info['error'], relative_error(rhs, rhs))
    np.testing.assert_array_equal(info['iterations'], 1)
    np.testing.assert_array_equal(info['termination'], GMRES_BREAKDOWN)
    assert not info['converged'].any()


def test_varying_preconditioner_uses_actual_stored_directions():
    matrix = np.array([[4., 2., 0.], [-1., 3., 1.], [2., 0., -2.]])
    rhs = np.array([[2., -3., 5.]])
    calls = 0

    def varying(x):
        nonlocal calls
        calls += 1
        return x * np.array([[1. + calls, 1. / calls, -1.]])

    answer, info = batched_gmres(rhs, np.zeros_like(rhs), lambda x: x @ matrix.T,
        varying, relative_error, xp=np, tolerance=1e-13)
    assert calls == 3
    assert info['converged'].all(), info
    np.testing.assert_allclose(answer @ matrix.T, rhs, atol=1e-12, rtol=1e-12)


def test_nonfinite_preconditioner_is_explicit_and_preserves_valid_rows():
    rhs = np.array([[1., -2.], [3., 4.]])
    initial = np.zeros_like(rhs)

    def partially_nonfinite(x):
        z = x.copy()
        z[0] = np.nan
        return z

    answer, info = batched_gmres(rhs, initial, lambda x: x.copy(),
        partially_nonfinite, relative_error, xp=np)
    np.testing.assert_array_equal(answer[0], initial[0])
    np.testing.assert_allclose(answer[1], rhs[1])
    assert info['termination'].tolist() == [GMRES_NONFINITE, GMRES_CONVERGED]
    assert info['iterations'].tolist() == [1, 1]
    assert info['nonfinite'].tolist() == [True, False]
    np.testing.assert_allclose(info['error'], relative_error(rhs, rhs - answer))


def test_true_error_controls_acceptance_and_retains_best_previous_candidate():
    matrix = np.array([[-4., -3., -4.], [0., -2., -2.], [-4., 4., 0.]])
    rhs = np.array([[1., 4., 4.]])
    seen = []

    def measurement(b, residual):
        # A fixed non-L2 weighted norm can increase while GMRES reduces L2.
        value = np.max(np.abs(residual) * np.array([[100., 1., 1.]]), axis=1)
        seen.append(value.copy())
        return value

    answer, info = batched_gmres(rhs, np.zeros_like(rhs), lambda x: x @ matrix.T,
        lambda x: x.copy(), measurement, xp=np, max_iterations=2, tolerance=1e-14)
    assert not info['converged'].any()
    assert seen[1][0] < seen[0][0] < seen[2][0]
    np.testing.assert_allclose(info['error'], np.min(np.stack(seen), axis=0))
    np.testing.assert_allclose(info['error'], measurement(rhs, rhs - answer @ matrix.T))


def test_zero_budget_reports_initial_residual_without_preconditioning():
    rhs = np.array([[1., 2.]])

    def forbidden(_):
        raise AssertionError('Zero budget must not call preconditioner')

    answer, info = batched_gmres(rhs, np.zeros_like(rhs), lambda x: x.copy(),
        forbidden, relative_error, xp=np, max_iterations=0)
    np.testing.assert_array_equal(answer, 0.)
    np.testing.assert_array_equal(info['iterations'], 0)
    np.testing.assert_array_equal(info['termination'], GMRES_ITERATION_LIMIT)


@pytest.mark.parametrize('budget', [-1, 129, 1.5, True])
def test_rejects_unbounded_or_invalid_budgets(budget):
    with pytest.raises(ValueError, match='budget'):
        batched_gmres(np.ones((1, 2)), np.zeros((1, 2)), lambda x: x,
                     lambda x: x, relative_error, xp=np, max_iterations=budget)


def test_rejects_nonfinite_initial_instead_of_substituting_a_new_start():
    with pytest.raises(ValueError, match='Finite'):
        batched_gmres(np.ones((1, 2)), np.array([[np.nan, 0.]]), lambda x: x,
                     lambda x: x, relative_error, xp=np)
