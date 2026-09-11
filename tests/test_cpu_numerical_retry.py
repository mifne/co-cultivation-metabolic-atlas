"""CPU-only original-certificate recovery; no relaxed acceptance thresholds."""
import numpy as np
import pytest

import src.cpu_repeated_lp as module
from src.cpu_repeated_lp import RepeatedCpuLP, _NUMERICAL_OPTIONS


def _warm_requests():
    arguments = dict(A_ub=[[0., 1.]], b_ub=[1.], bounds=[(0., 1000.), (0., 2.)],
                     options={'presolve': False})
    return ([-1., -1.], arguments), ([2e-9, -1.], arguments)


def _force_rejections(monkeypatch, count):
    original = module._certificate
    calls = []

    def certificate(*args):
        result = original(*args)
        calls.append(result.copy())
        if len(calls) <= count:
            return dict(result, relative_kkt_gap=2e-7, certificate_passed=False)
        return result

    monkeypatch.setattr(module, '_certificate', certificate)
    return calls


def test_real_warm_simplex_optimal_can_fail_summed_gap_and_strict_refactor_recovers():
    first, changed = _warm_requests()
    disabled = RepeatedCpuLP(1, max_numerical_retries=0)
    service = RepeatedCpuLP(1)
    try:
        np.testing.assert_equal(disabled.solve_batch([first])[0].x, [1000., 1.])
        rejected = disabled.solve_batch([changed])[0]
        assert not rejected.success and rejected.x is None
        assert rejected.diagnostics['model_status'] == 'HighsModelStatus.kOptimal'
        assert rejected.diagnostics['relative_kkt_gap'] > 1e-7
        assert rejected.diagnostics['cpu_solver_runs'] == 1

        service.solve_batch([first])
        recovered = service.solve_batch([changed])[0]
        assert recovered.success and recovered.diagnostics['certificate_passed']
        np.testing.assert_equal(recovered.x, [0., 1.])
        attempts = recovered.diagnostics['solver_attempts']
        assert [row['kind'] for row in attempts] == ['initial', 'strict_basis_refactor']
        assert attempts[0]['relative_kkt_gap'] > 1e-7
        assert attempts[1]['relative_kkt_gap'] <= 1e-7
        assert all(attempts[1]['effective_options'][name] == 1e-9 for name in _NUMERICAL_OPTIONS[:-1])
        assert recovered.diagnostics['cpu_lp_calls'] == 1
        assert recovered.diagnostics['cpu_solver_runs'] == 2
        assert recovered.diagnostics['numerical_retry_count'] == 1
        assert service.history[-1]['cpu_lp_calls'] == 1
        assert service.history[-1]['cpu_solver_runs'] == 2
        assert service.history[-1]['numerical_retry_count'] == 1
        assert recovered.diagnostics['effective_options_restored']
        assert recovered.diagnostics['numerical_retry_seconds'] > 0.
        assert recovered.diagnostics['total_seconds'] >= recovered.diagnostics['solve_seconds']
        # The next request must use the original 1e-7 tolerances again.
        next_row = service.solve_batch([first])[0]
        assert next_row.success
        assert all(next_row.diagnostics['solver_attempts'][0]['effective_options'][name] == 1e-7
                   for name in _NUMERICAL_OPTIONS[:-1])
    finally:
        disabled.close(); service.close()


def test_second_retry_rebuilds_original_lp_and_restores_custom_options(monkeypatch):
    service = RepeatedCpuLP(1)
    _force_rejections(monkeypatch, 2)
    request = ([-1., -2.], dict(A_ub=[[1., 1.]], b_ub=[3.], bounds=(0., 2.),
        options={'presolve': True, 'primal_feasibility_tolerance': 1e-10}))
    try:
        result = service.solve_batch([request], environment_ids=[71])[0]
        assert result.success
        np.testing.assert_allclose(result.x, [1., 2.], atol=1e-9)
        attempts = result.diagnostics['solver_attempts']
        assert [row['kind'] for row in attempts] == ['initial', 'strict_basis_refactor', 'strict_cold_no_presolve']
        assert [row['certificate_passed'] for row in attempts] == [False, False, True]
        assert attempts[-1]['effective_options']['presolve'] == 'off'
        # No retry loosens an already tighter user setting.
        assert all(attempts[1]['effective_options'][name] == 1e-10 for name in _NUMERICAL_OPTIONS[:-1])
        assert result.diagnostics['cpu_solver_runs'] == 3
        assert result.diagnostics['effective_options_restored']
        entry = next(iter(service.models.values()))
        assert service._numerical_options(entry['solver']) == attempts[0]['effective_options']
        assert entry['basis'].valid
    finally:
        service.close()


def test_exhausted_certificate_retries_fail_closed_and_discard_uncertified_state(monkeypatch):
    service = RepeatedCpuLP(1)
    _force_rejections(monkeypatch, 100)
    try:
        result = service.solve_batch([([-1.], dict(bounds=(0., 2.)))])[0]
        assert not result.success and result.x is None and result.fun is None
        assert result.diagnostics['cpu_solver_runs'] == 3
        assert result.diagnostics['numerical_retry_count'] == 2
        assert not any(row['certificate_passed'] for row in result.diagnostics['solver_attempts'])
        assert result.diagnostics['effective_options_restored']
        entry = next(iter(service.models.values()))
        assert entry['basis'] is None and not entry['solver'].getBasis().valid
    finally:
        service.close()


@pytest.mark.parametrize('lp_request', [
    ([1.], dict(A_eq=[[1.]], b_eq=[2.], bounds=(0., 1.))),
    ([-1.], dict(bounds=(0., None))),
])
def test_infeasible_and_unbounded_never_enter_numerical_recovery(lp_request):
    service = RepeatedCpuLP(1)
    try:
        result = service.solve_batch([lp_request])[0]
        assert not result.success and result.x is None and result.fun is None
        assert result.diagnostics['cpu_solver_runs'] == 1
        assert result.diagnostics['numerical_retry_count'] == 0
        assert not result.diagnostics['solver_attempts'][0]['optimal']
    finally:
        service.close()


def test_retry_run_exception_is_recorded_and_options_are_restored(monkeypatch):
    service = RepeatedCpuLP(1)
    _force_rejections(monkeypatch, 1)
    original = service.hp.Highs.run
    calls = []

    def run(solver):
        calls.append(solver)
        if len(calls) == 2:
            raise RuntimeError('deliberate retry solver failure')
        return original(solver)

    monkeypatch.setattr(service.hp.Highs, 'run', run)
    try:
        result = service.solve_batch([([-1.], dict(bounds=(0., 2.)))])[0]
        assert not result.success and result.x is None
        assert result.diagnostics['cpu_solver_runs'] == 2
        assert result.diagnostics['numerical_retry_count'] == 1
        assert result.diagnostics['solver_attempts'][-1]['run_status'] == 'exception'
        assert 'deliberate' in result.diagnostics['solver_attempts'][-1]['error']
        assert result.diagnostics['effective_options_restored']
    finally:
        service.close()


def test_exception_before_attempt_append_still_restores_retry_options(monkeypatch):
    service = RepeatedCpuLP(1)
    _force_rejections(monkeypatch, 1)
    original = service._attempt

    def attempt(solver, problem, label, setup_seconds):
        if label != 'initial':
            raise ValueError('deliberate diagnostic failure')
        return original(solver, problem, label, setup_seconds)

    monkeypatch.setattr(service, '_attempt', attempt)
    try:
        with pytest.raises(ValueError, match='diagnostic'):
            service.solve_batch([([-1.], dict(bounds=(0., 2.)))])
        entry = next(iter(service.models.values()))
        assert all(service._numerical_options(entry['solver'])[name] == 1e-7
                   for name in _NUMERICAL_OPTIONS[:-1])
    finally:
        service.close()


def test_error_run_status_cannot_be_overridden_by_an_optimal_stale_solution(monkeypatch):
    service = RepeatedCpuLP(1)
    original = service.hp.Highs.run

    def run(solver):
        original(solver)
        return service.hp.HighsStatus.kError

    monkeypatch.setattr(service.hp.Highs, 'run', run)
    try:
        result = service.solve_batch([([-1.], dict(bounds=(0., 2.)))])[0]
        assert not result.success and result.x is None
        assert result.diagnostics['cpu_solver_runs'] == 1
        assert result.diagnostics['numerical_retry_count'] == 0
        assert result.diagnostics['solver_attempts'][0]['run_status'] == 'HighsStatus.kError'
    finally:
        service.close()
