"""CPU math tests plus explicitly named CUDA integration tests.

Run ``-k 'not cuda'`` for the CPU-only checks. CUDA tests are separate so an
independent reviewer does not compete with the main experiment for the GPU.
"""

import numpy as np
import pytest

from src.gpu_globalized_ipm import (
    backtrack_centered_step,
    full_defect_from_eliminated,
    guarded_corrector_rhs,
    relative_forcing,
    retry_regularizations,
    select_better_direction,
    solve_globalized_ipm,
    temporary_regularization,
    weighted_blocks,
)


def test_relative_forcing_has_no_unit_absolute_floor():
    target = np.array([[1e-12, 0.], [0., 0.], [0., 0.]])
    error = np.array([[5e-14, 0.], [0., 0.], [1e-30, 0.]])
    result = relative_forcing(error, target, xp=np)
    assert result[0] == pytest.approx(.05)
    assert result[1] == 0.
    assert np.isinf(result[2])


def test_forcing_is_scale_invariant_and_scaled_norm_avoids_overflow():
    target = np.array([[3., 4.]])
    error = np.array([[.15, .2]])
    for factor in (1e-150, 1., 1e150):
        np.testing.assert_allclose(relative_forcing(error*factor, target*factor, xp=np), [.05])


def test_nonfinite_forcing_is_not_accepted():
    with np.errstate(invalid='ignore'):
        ratio = relative_forcing(np.array([[np.nan], [0.]]),
                                 np.array([[1.], [np.inf]]), xp=np)
    assert np.isinf(ratio).all()


def test_eliminated_residual_requires_multiplier_for_complementarity():
    residual = np.array([[2., 3., 4.], [2e-9, 3e-9, 4e-9]])
    z = np.array([[5.], [1e12]])
    full = full_defect_from_eliminated(residual, z, 1, 1, xp=np)
    np.testing.assert_allclose(full, [[-2., -3., 20.], [-2e-9, -3e-9, 4000.]])


def test_eliminated_mapping_matches_explicit_full_jacobian():
    # Ex=b, Gx+s=h, rd=c+E.T*y+G.T*z, with x/y/z all one-dimensional.
    e, g = 2., -3.
    x, y, z, s = .7, -.2, 4., 2.
    rd, rp, rg, rc = 1.2, -.3, .4, 7.2
    dx, dy, dz = .15, -.7, .8
    ds = -rg-g*dx
    matrix = np.array([[0., e, g], [e, 0., 0.], [g, 0., -s/z]])
    rhs = np.array([-rd, -rp, rc/z-rg])
    residual = (rhs-matrix@np.array([dx, dy, dz]))[None]
    mapped = full_defect_from_eliminated(residual, np.array([[z]]), 1, 1, xp=np)
    explicit = np.array([[rd+e*dy+g*dz, rp+e*dx, rc+z*ds+s*dz]])
    np.testing.assert_allclose(mapped, explicit)
    assert abs(rg+g*dx+ds) < 1e-15


def _line_problem(batch=1):
    one = np.ones((batch, 1))
    empty = np.zeros((batch, 0))
    # A large dual direction gives descent but needs backtracking to avoid
    # overshooting stationarity. Complementarity stays exactly unchanged.
    state = (np.zeros_like(one), np.zeros_like(one), one.copy(), one.copy())
    direction = (np.zeros_like(one), -100.*one, np.zeros_like(one), np.zeros_like(one))
    blocks = (one.copy(), empty, np.zeros_like(one), one.copy())
    derivative = (-100.*one, empty, np.zeros_like(one), np.zeros_like(one))
    weights = tuple(np.ones(batch) for _ in range(4))
    return state, direction, blocks, derivative, weights


def test_backtracking_uses_actual_merit_and_independent_masks():
    state, direction, blocks, derivative, weights = _line_problem(2)
    result, info = backtrack_centered_step(state, direction, blocks, derivative, weights,
                                          np.array([True, False]), xp=np)
    assert info['taken'].tolist() == [True, False]
    assert info['backtracks'][0] > 0
    assert info['after'][0] <= info['before'][0] + 1e-4*info['alpha'][0]*info['slope'][0]
    for value, old in zip(result, state):
        np.testing.assert_array_equal(value[1], old[1])
    assert (result[2] > 0.).all() and (result[3] > 0.).all()


def test_backtracking_exhaustion_or_non_descent_preserves_state():
    state, direction, blocks, derivative, weights = _line_problem()
    result, info = backtrack_centered_step(state, direction, blocks, derivative, weights,
                                          np.array([True]), xp=np, max_backtracks=0)
    assert info['descent'][0] and not info['taken'][0]
    for value, old in zip(result, state):
        np.testing.assert_array_equal(value, old)
    ascent = (100.*np.ones((1, 1)), *derivative[1:])
    result, info = backtrack_centered_step(state, direction, blocks, ascent, weights,
                                          np.array([True]), xp=np)
    assert not info['descent'][0] and not info['taken'][0]
    for value, old in zip(result, state):
        np.testing.assert_array_equal(value, old)


def test_centered_direction_descends_and_preserves_positive_interior():
    one, empty = np.ones((1, 1)), np.zeros((1, 0))
    state = (one.copy(), empty, 2.*one, 2.*one)
    direction = (np.zeros_like(one), empty, -.9*one, -.9*one)
    blocks = (np.zeros_like(one), empty, np.zeros_like(one), 4.*one)
    derivative = (np.zeros_like(one), empty, np.zeros_like(one), -3.6*one)
    weights = tuple(np.ones(1) for _ in range(4))
    result, info = backtrack_centered_step(state, direction, blocks, derivative, weights,
                                          np.array([True]), xp=np)
    assert info['taken'][0] and info['after'][0] < info['before'][0]
    assert np.min(result[2]) > 0. and np.min(result[3]) > 0.
    product = result[2]*result[3]
    assert info['after'][0] == pytest.approx(.5*float(product[0, 0])**2)


def test_weighted_blocks_keep_empty_equalities_and_fixed_scales():
    blocks = (np.array([[2.]]), np.empty((1, 0)), np.array([[4., 6.]]), np.array([[8.]]))
    weights = (np.array([.5]), np.array([1.]), np.array([.25]), np.array([.125]))
    np.testing.assert_allclose(weighted_blocks(blocks, weights, xp=np), [[1., 1., 1.5, 1.]])


def test_retry_regularization_budget_and_floor_never_increase_delta():
    np.testing.assert_allclose(retry_regularizations(1e-6, 2), [1e-7, 1e-8])
    assert retry_regularizations(1e-6, 0) == []
    assert retry_regularizations(1e-8, 2) == []
    assert retry_regularizations(1e-9, 2) == []
    for budget in (-1, 3, True, 1.5):
        with pytest.raises(ValueError):
            retry_regularizations(1e-6, budget)


def test_temporary_regularization_restores_base_even_when_factor_raises():
    from types import SimpleNamespace
    solver = SimpleNamespace(regularization=1e-6)
    with temporary_regularization(solver, 1e-7):
        assert solver.regularization == 1e-7
    assert solver.regularization == 1e-6
    with pytest.raises(RuntimeError, match='factor failed'):
        with temporary_regularization(solver, 1e-8):
            assert solver.regularization == 1e-8
            raise RuntimeError('factor failed')
    assert solver.regularization == 1e-6


def test_direction_retry_retains_better_lanes_and_rejects_nonfinite_candidates():
    previous = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]])
    candidate = 2.*previous
    candidate[3, 0] = np.inf
    previous_error = np.array([.2, .2, .2, .2, .2])
    candidate_error = np.array([.1, .3, .1, .1, np.nan])
    answer, error, selected = select_better_direction(previous, candidate, previous_error,
        candidate_error, np.array([True, True, False, True, True]), xp=np)
    np.testing.assert_array_equal(selected, [True, False, False, False, False])
    np.testing.assert_array_equal(answer[0], candidate[0])
    np.testing.assert_array_equal(answer[1:], previous[1:])
    np.testing.assert_allclose(error, [.1, .2, .2, .2, .2])


def test_corrector_rhs_uses_qualified_affine_product_and_separate_predictor_steps():
    s, z = np.array([[2., 4.]]), np.array([[3., 1.]])
    ds, dz = np.array([[-4., 1.]]), np.array([[-1., .2]])
    comp, mu = s*z, np.mean(s*z, axis=1)
    rc, sigma, valid = guarded_corrector_rhs(s, z, comp, mu, ds, dz,
                                             np.array([True]), xp=np)
    mu_affine = np.mean((s+.5*ds)*(z+dz), axis=1)
    expected_sigma = (mu_affine/mu)**3
    np.testing.assert_allclose(sigma, expected_sigma)
    np.testing.assert_allclose(rc, comp+ds*dz-expected_sigma[:, None]*mu[:, None])
    assert valid.tolist() == [True]


def test_unqualified_affine_never_enters_overflowing_or_nan_products():
    s, z = np.ones((3, 2)), np.ones((3, 2))
    ds = np.array([[-.5, -.5], [1e308, 1e308], [np.nan, np.nan]])
    dz = np.array([[-.5, -.5], [1e308, 1e308], [np.nan, np.nan]])
    with np.errstate(over='raise', invalid='raise'):
        rc, sigma, valid = guarded_corrector_rhs(s, z, s*z, np.ones(3), ds, dz,
            np.array([True, False, False]), xp=np)
    assert valid.tolist() == [True, False, False]
    np.testing.assert_array_equal(rc[1:], 0.)
    np.testing.assert_array_equal(sigma[1:], 0.)
    np.testing.assert_allclose(sigma[0], .25**3)


def test_overflow_risk_affine_product_fails_closed_without_affecting_good_lane():
    s, z = np.ones((2, 1)), np.ones((2, 1))
    ds = np.array([[-.5], [1e308]])
    dz = np.array([[-.5], [1e308]])
    with np.errstate(over='raise', invalid='raise'):
        rc, sigma, valid = guarded_corrector_rhs(s, z, s*z, np.ones(2), ds, dz,
            np.array([True, True]), xp=np)
    assert valid.tolist() == [True, False]
    assert rc[1, 0] == sigma[1] == 0.


def test_affine_boundary_roundoff_is_diagnosed_not_clipped_and_interior_prediction_is_safe():
    s, z = np.array([[3.412655902966238]]), np.ones((1, 1))
    ds, dz = np.array([[-4.63729174277441]]), np.zeros((1, 1))
    comp, mu = s*z, np.mean(s*z, axis=1)
    rc, sigma, valid, diagnostic = guarded_corrector_rhs(s, z, comp, mu, ds, dz,
        np.array([True]), xp=np, return_diagnostics=True)
    assert not valid[0] and diagnostic['failure_reason'][0] == 6
    assert diagnostic['min_predicted_s'][0] < 0.
    assert 0. < diagnostic['max_negative_s_roundoff_units'][0] < 1.
    predicted_step = diagnostic['alpha_p'][0]*ds[0, 0]
    expected_scale = np.finfo(float).eps*(abs(s[0, 0])+abs(predicted_step))
    assert diagnostic['max_negative_s_roundoff_units'][0] == pytest.approx(
        -diagnostic['min_predicted_s'][0]/expected_scale)
    assert diagnostic['raw_mu_affine'][0] < 0.  # Never clipped to justify acceptance.
    assert diagnostic['raw_sigma'][0] < 0.
    assert rc[0, 0] == sigma[0] == 0.
    rc, sigma, valid, diagnostic = guarded_corrector_rhs(s, z, comp, mu, ds, dz,
        np.array([True]), xp=np, affine_fraction=.995, return_diagnostics=True)
    assert valid[0] and diagnostic['failure_reason'][0] == 0
    assert diagnostic['min_predicted_s'][0] > 0.
    assert diagnostic['max_negative_s_roundoff_units'][0] == 0.
    np.testing.assert_allclose(rc, comp+ds*dz-sigma[:, None]*mu[:, None])


def test_affine_fraction_changes_sigma_only_not_qualified_cross_product():
    s, z = np.array([[2., 4.]]), np.array([[3., 1.]])
    ds, dz = np.array([[-4., 1.]]), np.array([[-1., .2]])
    comp, mu = s*z, np.mean(s*z, axis=1)
    estimates = []
    for fraction in (1., .995):
        rc, sigma, valid = guarded_corrector_rhs(s, z, comp, mu, ds, dz,
            np.array([True]), xp=np, affine_fraction=fraction)
        assert valid[0]
        np.testing.assert_allclose(rc-comp+sigma[:, None]*mu[:, None], ds*dz)
        estimates.append(sigma[0])
    assert estimates[0] != estimates[1]


def test_affine_fraction_validation_and_centered_mode_reject_nondefault_fraction():
    from types import SimpleNamespace
    one = np.ones((1, 1))
    for fraction in (0., -1., 1.01, np.nan, np.inf, True):
        with pytest.raises(ValueError, match='Affine fraction'):
            guarded_corrector_rhs(one, one, one, np.ones(1), -one, -one,
                np.array([True]), xp=np, affine_fraction=fraction)
    # Validation precedes device-array initialization, so this check needs no GPU.
    solver = SimpleNamespace(closed=False, forcing_eta=.05,
                             factor=SimpleNamespace(_context=lambda: None))
    with pytest.raises(ValueError, match='requires PC'):
        solve_globalized_ipm(solver, predictor_affine_fraction=.995)


def _example(rhs=1., cost=(1., 2.)):
    from scipy.sparse import csr_matrix
    return (csr_matrix([[1., 1.]]), np.array([rhs]), np.zeros(2), np.full(2, np.inf),
            np.array(cost), 1)


def test_cuda_centered_lp_converges_with_original_certificate_and_initial_history():
    from src.gpu_batched_ipm import GpuBatchedIPM
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    problems = [_example(), _example(2., (3., 1.)), _example(4., (1., 2.))]
    with GpuBatchedIPM(problems) as solver:
        result = solve_globalized_ipm(solver, iterations=80)
        assert result['accepted'].all(), result['metrics']
        np.testing.assert_allclose(result['x'].get(), [[1., 0.], [0., 2.], [4., 0.]], atol=1e-6)
        assert result['cpu_lp_calls'] == 0 and result['analysis_count'] == 1
        assert result['solve_count'] == result['factor_count']
        assert all(row['objective'] == 0. for row in result['checkpoints'][0]['metrics'])
        for problem, x, y in zip(problems, result['x'].get(), result['y'].get()):
            assert paired_certificate(problem, x, y)['certificate_passed']
        history = np.asarray(result['globalization_diagnostics'], dtype=float)
        updated = history[:, :, 5] > 0.
        assert np.all(history[:, :, 4][updated] < history[:, :, 3][updated])


def test_cuda_bad_lane_freezes_without_stopping_other_environment(monkeypatch):
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example(), _example(2.)]) as solver:
        original_solve = solver._solve_newton
        def one_bad_lane(rhs, ratio, scaling):
            answer = original_solve(rhs, ratio, scaling)
            answer[0] = 0.
            return answer
        monkeypatch.setattr(solver, '_solve_newton', one_bad_lane)
        result = solve_globalized_ipm(solver, iterations=80)
        assert result['accepted'].tolist() == [False, True]
        assert result['failed_environments'][0] == 2
        assert result['failed_environments'][1] == 0
        np.testing.assert_array_equal(result['x'].get()[0], [0., 0.])
        np.testing.assert_allclose(result['x'].get()[1], [2., 0.], atol=1e-6)


def test_cuda_certified_initial_pair_remains_exact_and_zero_budget_fails_closed():
    import cupy as cp
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example()]) as solver:
        result = solve_globalized_ipm(solver, initial_x=cp.array([[1., 0.]]),
                                      initial_y=cp.array([[1.]]), iterations=0)
        assert result['accepted'].all() and result['factor_count'] == 0
        np.testing.assert_array_equal(result['x'].get(), [[1., 0.]])
        np.testing.assert_array_equal(result['y'].get(), [[1.]])
        result = solve_globalized_ipm(solver, iterations=0)
        assert not result['accepted'].any() and result['factor_count'] == 0


def test_cuda_original_infeasible_problem_never_reported_as_solved():
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example(-1.)]) as solver:
        result = solve_globalized_ipm(solver, iterations=25)
        assert not result['accepted'].any()
        assert result['cpu_lp_calls'] == 0


def test_cuda_actual_candidate_merit_is_recomputed_before_state_commit(monkeypatch):
    import src.gpu_globalized_ipm as module
    from src.gpu_batched_ipm import GpuBatchedIPM
    original_backtrack = module.backtrack_centered_step
    def corrupted_candidate(*args, **kwargs):
        candidate, info = original_backtrack(*args, **kwargs)
        # Simulate a mismatch between the line-search residual formula and
        # actual candidate arithmetic. The reported predicted merit is left
        # untouched; only a real matrix recomputation can detect this defect.
        candidate[0][:] = 1e6
        return candidate, info
    monkeypatch.setattr(module, 'backtrack_centered_step', corrupted_candidate)
    with GpuBatchedIPM([_example()]) as solver:
        result = solve_globalized_ipm(solver, iterations=3)
        assert not result['accepted'].any()
        assert result['failed_environments'] == [6]
        assert result['iterations'] == 1
        assert result['completed_update_iteration'] == 0
        np.testing.assert_array_equal(result['x'].get(), [[0., 0.]])


def test_cuda_regularization_retry_uses_same_state_and_restores_base(monkeypatch):
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example(), _example(2.)], regularization=1e-6) as solver:
        factor_calls, solve_calls = [], []
        original_factor, original_solve = solver._factor_newton, solver._solve_newton
        def recorded_factor(ratio):
            factor_calls.append((solver.regularization, ratio.copy()))
            return original_factor(ratio)
        def first_preconditioner_bad(rhs, ratio, scaling):
            solve_calls.append((solver.regularization, rhs.copy()))
            answer = original_solve(rhs, ratio, scaling)
            if solver.regularization == 1e-6:
                answer[0] = 0.
            return answer
        monkeypatch.setattr(solver, '_factor_newton', recorded_factor)
        monkeypatch.setattr(solver, '_solve_newton', first_preconditioner_bad)
        result = solve_globalized_ipm(solver, iterations=80, regularization_retries=1)
        assert result['accepted'].all(), result['metrics']
        assert solver.regularization == 1e-6
        assert factor_calls[0][0] == 1e-6 and factor_calls[1][0] == pytest.approx(1e-7)
        np.testing.assert_array_equal(factor_calls[0][1].get(), factor_calls[1][1].get())
        np.testing.assert_array_equal(solve_calls[0][1].get()[0], solve_calls[1][1].get()[0])
        assert result['regularization_retry_factor_attempts'] > 0
        assert result['factor_count'] > result['iterations']
        assert result['cpu_lp_calls'] == 0


def test_cuda_retry_configuration_is_restored_if_retry_factor_raises(monkeypatch):
    import cupy as cp
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example()], regularization=1e-6) as solver:
        original_factor = solver._factor_newton
        def broken_retry(ratio):
            if solver.regularization != 1e-6:
                raise RuntimeError('retry factor failed')
            return original_factor(ratio)
        monkeypatch.setattr(solver, '_factor_newton', broken_retry)
        monkeypatch.setattr(solver, '_solve_newton', lambda rhs, ratio, scaling: cp.zeros_like(rhs))
        with pytest.raises(RuntimeError, match='retry factor failed'):
            solve_globalized_ipm(solver, iterations=3, regularization_retries=1)
        assert solver.regularization == 1e-6


def test_cuda_failed_attempt_after_unchecked_update_has_consistent_iteration_labels(monkeypatch):
    import cupy as cp
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example()]) as solver:
        calls = []
        original_solve = solver._solve_newton
        def second_direction_bad(rhs, ratio, scaling):
            calls.append(None)
            return original_solve(rhs, ratio, scaling) if len(calls) == 1 else cp.zeros_like(rhs)
        monkeypatch.setattr(solver, '_solve_newton', second_direction_bad)
        result = solve_globalized_ipm(solver, iterations=10, check_interval=10)
        assert result['iterations'] == 2
        assert result['completed_update_iteration'] == 1
        assert max(row['iteration'] for row in result['checkpoints']) <= result['iterations']


@pytest.mark.parametrize('affine_fraction', [1., .995])
def test_cuda_predictor_corrector_keeps_original_certificate_and_factor_reuse(affine_fraction):
    from src.gpu_batched_ipm import GpuBatchedIPM
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    problems = [_example(), _example(2., (3., 1.)), _example(4.)]
    with GpuBatchedIPM(problems) as solver:
        result = solve_globalized_ipm(solver, iterations=80, predictor_corrector=True,
                                      predictor_affine_fraction=affine_fraction)
        assert result['accepted'].all(), result['metrics']
        assert result['cpu_lp_calls'] == 0 and result['analysis_count'] == 1
        assert result['factor_count'] == result['iterations']
        assert result['solve_count'] == sum(result['triangular_solve_counts_by_role'].values())
        assert result['solve_count'] <= 3*result['iterations']
        assert result['direction_attempt_counts_by_role'][1] == result['iterations']
        for problem, x, y in zip(problems, result['x'].get(), result['y'].get()):
            assert paired_certificate(problem, x, y)['certificate_passed']
        history = np.asarray(result['globalization_diagnostics'], dtype=float)
        updated = history[:, :, 5] > 0.
        assert np.all(history[:, :, 4][updated] < history[:, :, 3][updated])
        targets = np.asarray(result['direction_target_diagnostics'], dtype=float)
        reliable = targets[:, :, 5] == 1.
        assert np.all(targets[:, :, 4][reliable] <= result['forcing_eta'])
        affine = np.asarray(result['predictor_affine_diagnostics'], dtype=float)
        columns = result['predictor_affine_diagnostic_columns']
        np.testing.assert_array_equal(affine[:, :, columns.index('fraction')], affine_fraction)
        assert result['predictor_affine_fraction'] == affine_fraction


@pytest.mark.parametrize('bad_role', ['affine', 'corrector'])
def test_cuda_pc_rejected_direction_gets_one_same_state_centered_fallback(monkeypatch, bad_role):
    import src.gpu_globalized_ipm as module
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example(), _example(2.)]) as solver:
        calls, qualified = [], []
        original_solve = solver._solve_newton
        original_guard = module.guarded_corrector_rhs
        def bad_first_direction(rhs, ratio, scaling):
            calls.append((rhs.copy(), ratio.copy()))
            answer = original_solve(rhs, ratio, scaling)
            if len(calls) == (1 if bad_role == 'affine' else 2):
                answer[0] = 0.
            return answer
        def recorded_guard(*args, **kwargs):
            qualified.append(args[6].copy())
            return original_guard(*args, **kwargs)
        monkeypatch.setattr(solver, '_solve_newton', bad_first_direction)
        monkeypatch.setattr(module, 'guarded_corrector_rhs', recorded_guard)
        result = solve_globalized_ipm(solver, iterations=1, predictor_corrector=True)
        pc = np.asarray(result['predictor_corrector_diagnostics'])[0]
        assert pc[:, 5].tolist() == [0., 1.]  # Second lane keeps its PC step.
        assert pc[:, 6].tolist() == [1., 0.]
        assert pc[:, 8].tolist() == [1., 0.]
        assert pc[0, 7] == (1 if bad_role == 'affine' else 3)
        assert bool(qualified[0].get()[0]) == (bad_role != 'affine')
        assert result['failed_environments'] == [0, 0]
        assert result['factor_count'] == 1 and len(calls) == 3
        assert result['direction_attempt_counts_by_role'][3] == 1
        assert result['cpu_lp_calls'] == 0
        for _, ratio in calls[1:]:
            np.testing.assert_array_equal(ratio.get(), calls[0][1].get())
        # Initial s=1, z=max(1,c) for these nonnegative-bound LPs. Thus
        # (rc_affine-rc_centered)/z=.1*mu/z, not a uniform .1.
        # Both targets come from the same state, never a failed trial step.
        initial_z = np.maximum(1., _example()[4])
        initial_s = np.ones_like(initial_z)
        expected_rhs_difference = .1*np.mean(initial_s*initial_z)/initial_z
        np.testing.assert_allclose(calls[0][0].get()[0, solver.n+solver.ne:]
                                   - calls[-1][0].get()[0, solver.n+solver.ne:],
                                   expected_rhs_difference)
        assert np.all(result['x'].get()[:, 0] > 0.)


@pytest.mark.parametrize('corrupt_fallback', [False, True])
def test_cuda_pc_actual_merit_failure_falls_back_once_and_never_commits_bad_state(monkeypatch,
                                                                              corrupt_fallback):
    import src.gpu_globalized_ipm as module
    from src.gpu_batched_ipm import GpuBatchedIPM
    original_backtrack = module.backtrack_centered_step
    calls = []
    def corrupted_candidate(*args, **kwargs):
        candidate, info = original_backtrack(*args, **kwargs)
        calls.append(None)
        if len(calls) == 1 or corrupt_fallback:
            candidate[0][:] = 1e6
        return candidate, info
    monkeypatch.setattr(module, 'backtrack_centered_step', corrupted_candidate)
    with GpuBatchedIPM([_example()]) as solver:
        result = solve_globalized_ipm(solver, iterations=1, predictor_corrector=True)
        assert len(calls) == 2
        pc = np.asarray(result['predictor_corrector_diagnostics'])[0, 0]
        assert pc[6] == 1. and pc[7] == 6.
        assert result['direction_attempt_counts_by_role'][3] == 1
        assert result['failed_environments'] == ([6] if corrupt_fallback else [0])
        assert result['completed_update_iteration'] == (0 if corrupt_fallback else 1)
        if corrupt_fallback:
            np.testing.assert_array_equal(result['x'].get(), [[0., 0.]])
            assert not result['accepted'].any()
        else:
            assert np.max(result['x'].get()) < 2.


def test_cuda_pc_retry_budget_is_shared_and_new_target_reuses_matching_factor(monkeypatch):
    from src.gpu_batched_ipm import GpuBatchedIPM
    with GpuBatchedIPM([_example()], regularization=1e-6) as solver:
        calls = []
        original_solve = solver._solve_newton
        def bad_base_and_corrector(rhs, ratio, scaling):
            calls.append((solver.regularization, ratio.copy()))
            answer = original_solve(rhs, ratio, scaling)
            if len(calls) in (1, 3):
                answer[:] = 0.
            return answer
        monkeypatch.setattr(solver, '_solve_newton', bad_base_and_corrector)
        result = solve_globalized_ipm(solver, iterations=1, predictor_corrector=True,
                                      regularization_retries=1)
        assert result['factor_count'] == 2  # Not 2 per predictor/corrector/fallback.
        assert result['regularization_retry_factor_attempts'] == 1
        assert len(calls) == 4  # Affine, affine retry, corrector, centered.
        assert calls[0][0] == 1e-6
        assert all(delta == pytest.approx(1e-7) for delta, _ in calls[1:])
        for _, ratio in calls[1:]:
            np.testing.assert_array_equal(ratio.get(), calls[0][1].get())
        assert solver.regularization == 1e-6
        assert result['failed_environments'] == [0]
        assert result['solve_count'] == sum(result['triangular_solve_counts_by_role'].values())
        assert result['cpu_lp_calls'] == 0
