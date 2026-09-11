"""Opt-in certified-GPU basis handoff tests for the persistent CPU solver."""

import numpy as np
import pytest

from src.cpu_repeated_lp import RepeatedCpuLP


def _box_request(objective, *, matrix=None, rhs=None, override=None, initial=None):
    if matrix is None:
        matrix = np.eye(2)
    if rhs is None:
        rhs = [3.0, 4.0]
    kwargs = dict(
        A_ub=np.asarray(matrix, dtype=float),
        b_ub=np.asarray(rhs, dtype=float),
        bounds=(0.0, 10.0),
        method='highs-ds',
        _stage='maxmin',
    )
    if override is not None:
        kwargs['_basis_override'] = override
    if initial is not None:
        kwargs['_initial_basis'] = initial
    return np.asarray(objective, dtype=float), kwargs


def _row_slack_basis():
    # Two inequality-row slacks are basic; both structural columns are at
    # their lower bounds. HiGHS status integers are lower=0 and basic=1.
    return dict(
        col_status=np.array([0, 0], dtype=np.int32),
        row_status=np.array([1, 1], dtype=np.int32),
    )


def _override(basis=None, *, source='previous_certified_gpu', age=1):
    return dict(
        basis=_row_slack_basis() if basis is None else basis,
        source=source,
        age=age,
    )


def _assert_box_solution(result, expected_x, expected_fun):
    assert result.success
    assert result.diagnostics['certificate_passed']
    np.testing.assert_allclose(result.x, expected_x, atol=1e-9)
    assert result.fun == pytest.approx(expected_fun, abs=1e-9)


class _SetBasisProxy:
    """Inject bounded setter/clear failures while delegating real HiGHS."""

    def __init__(self, solver, highs_status, *, failed_set_calls, fail_clear=False):
        self.solver = solver
        self.highs_status = highs_status
        self.failed_set_calls = failed_set_calls
        self.fail_clear = fail_clear
        self.set_calls = 0
        self.clear_calls = 0

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def setBasis(self, basis):
        self.set_calls += 1
        if self.set_calls <= self.failed_set_calls:
            # Exercise the conservative recovery contract: an error return is
            # not assumed to mean that the solver was left untouched.
            self.solver.setBasis(basis)
            return self.highs_status.kError
        return self.solver.setBasis(basis)

    def clearSolver(self):
        self.clear_calls += 1
        if self.fail_clear:
            return self.highs_status.kError
        return self.solver.clearSolver()


def test_valid_override_replaces_a_hot_basis_but_not_initial_basis_semantics():
    service = RepeatedCpuLP(1)
    try:
        first = service.solve_batch(
            [_box_request([-1.0, -2.0])], environment_ids=[7]
        )[0]
        _assert_box_solution(first, [3.0, 4.0], -11.0)

        # A malformed _initial_basis remains ignored for a hot model, while
        # the separate explicit override is consumed.
        request = _box_request(
            [1.0, 2.0], override=_override(age=1), initial={}
        )
        result = service.solve_batch([request], environment_ids=[7])[0]
        _assert_box_solution(result, [0.0, 0.0], 0.0)
        diagnostics = result.diagnostics
        assert diagnostics['basis_override_requested']
        assert diagnostics['basis_override_used']
        assert not diagnostics['basis_override_rejected']
        assert diagnostics['basis_override_source'] == 'previous_certified_gpu'
        assert diagnostics['basis_override_age'] == 1
        assert diagnostics['basis_override_recovery'] == 'not_needed'
        assert not diagnostics['basis_reused']
        assert not diagnostics['initial_basis_used']
    finally:
        service.close()


@pytest.mark.parametrize(
    'bad_override',
    [
        {},
        dict(basis=_row_slack_basis(), source='uncertified_gpu', age=1),
        _override(age=0),
        _override(age=True),
        _override(basis=dict(col_status=[0], row_status=[1, 1])),
        _override(basis=dict(col_status=[0, 0], row_status=[1, 0])),
        _override(basis=dict(col_status=[0, 7], row_status=[1, 1])),
    ],
)
def test_invalid_override_is_rejected_and_previous_basis_is_reused(bad_override):
    service = RepeatedCpuLP(1)
    try:
        _assert_box_solution(
            service.solve_batch([_box_request([-1.0, -2.0])])[0],
            [3.0, 4.0],
            -11.0,
        )
        result = service.solve_batch(
            [_box_request([1.0, 2.0], override=bad_override)]
        )[0]
        _assert_box_solution(result, [0.0, 0.0], 0.0)
        diagnostics = result.diagnostics
        assert diagnostics['basis_override_requested']
        assert diagnostics['basis_override_rejected']
        assert not diagnostics['basis_override_used']
        assert diagnostics['basis_override_rejection_reason'] == 'validation_failed'
        assert diagnostics['basis_override_recovery'] == 'previous_basis_reused'
        assert diagnostics['basis_reused']
    finally:
        service.close()


def test_override_setter_failure_restores_the_live_cpu_basis():
    service = RepeatedCpuLP(1)
    try:
        _assert_box_solution(
            service.solve_batch([_box_request([-1.0, -2.0])])[0],
            [3.0, 4.0],
            -11.0,
        )
        entry = next(iter(service.models.values()))
        proxy = _SetBasisProxy(
            entry['solver'], service.hp.HighsStatus, failed_set_calls=1
        )
        entry['solver'] = proxy
        result = service.solve_batch(
            [_box_request([1.0, 2.0], override=_override())]
        )[0]
        _assert_box_solution(result, [0.0, 0.0], 0.0)
        diagnostics = result.diagnostics
        assert proxy.set_calls == 2
        assert diagnostics['basis_override_rejected']
        assert diagnostics['basis_override_rejection_reason'] == 'setter_failed'
        assert diagnostics['basis_override_recovery'] == 'restored_live_basis'
        assert diagnostics['basis_reused']
        assert not diagnostics['basis_override_used']
    finally:
        service.close()


def test_failed_override_and_restore_clear_solver_for_an_exact_cold_run():
    service = RepeatedCpuLP(1)
    try:
        _assert_box_solution(
            service.solve_batch([_box_request([-1.0, -2.0])])[0],
            [3.0, 4.0],
            -11.0,
        )
        entry = next(iter(service.models.values()))
        proxy = _SetBasisProxy(
            entry['solver'], service.hp.HighsStatus, failed_set_calls=100
        )
        entry['solver'] = proxy
        result = service.solve_batch(
            [_box_request([1.0, 2.0], override=_override())]
        )[0]
        _assert_box_solution(result, [0.0, 0.0], 0.0)
        diagnostics = result.diagnostics
        assert proxy.set_calls >= 2
        assert proxy.clear_calls == 1
        assert diagnostics['basis_override_rejected']
        assert diagnostics['basis_override_recovery'] == 'cleared_solver'
        assert not diagnostics['basis_reused']
        assert not diagnostics['basis_override_used']
    finally:
        service.close()


def test_failure_to_restore_or_clear_fails_closed():
    service = RepeatedCpuLP(1)
    try:
        service.solve_batch([_box_request([-1.0, -2.0])])
        entry = next(iter(service.models.values()))
        proxy = _SetBasisProxy(
            entry['solver'], service.hp.HighsStatus,
            failed_set_calls=100, fail_clear=True,
        )
        entry['solver'] = proxy
        with pytest.raises(RuntimeError, match='cold recovery'):
            service.solve_batch(
                [_box_request([1.0, 2.0], override=_override())]
            )
    finally:
        service.close()


def test_rebuild_skips_override_and_preserves_current_initial_basis_behavior():
    service = RepeatedCpuLP(1)
    initial = _row_slack_basis()
    try:
        service.solve_batch([_box_request([-1.0, -2.0])])
        changed = _box_request(
            [1.0, 2.0],
            matrix=[[1.0, 1.0], [0.0, 1.0]],
            override=dict(basis={}, source='bad', age=0),
            initial=initial,
        )
        result = service.solve_batch([changed])[0]
        _assert_box_solution(result, [0.0, 0.0], 0.0)
        diagnostics = result.diagnostics
        assert diagnostics['model_rebuilt']
        assert diagnostics['initial_basis_used']
        assert diagnostics['basis_override_requested']
        assert not diagnostics['basis_override_used']
        assert not diagnostics['basis_override_rejected']
        assert diagnostics['basis_override_skipped_reason'] == 'model_rebuilt'
    finally:
        service.close()


def test_reuse_disabled_skips_override_and_keeps_the_cold_policy():
    service = RepeatedCpuLP(1, reuse_basis=False)
    try:
        service.solve_batch([_box_request([-1.0, -2.0])])
        result = service.solve_batch(
            [_box_request([1.0, 2.0], override=_override())]
        )[0]
        _assert_box_solution(result, [0.0, 0.0], 0.0)
        diagnostics = result.diagnostics
        assert diagnostics['basis_override_skipped_reason'] == 'basis_reuse_disabled'
        assert not diagnostics['basis_override_used']
        assert not diagnostics['basis_override_rejected']
        assert not diagnostics['basis_reused']
    finally:
        service.close()


def test_basis_override_is_isolated_by_stable_environment_id():
    service = RepeatedCpuLP(2)
    try:
        initial = [_box_request([-1.0, -2.0]), _box_request([-1.0, -2.0])]
        service.solve_batch(initial, environment_ids=[11, 29])
        requests = [
            _box_request([1.0, 2.0], override=_override()),
            _box_request([1.0, 2.0]),
        ]
        first, second = service.solve_batch(requests, environment_ids=[11, 29])
        _assert_box_solution(first, [0.0, 0.0], 0.0)
        _assert_box_solution(second, [0.0, 0.0], 0.0)
        assert first.diagnostics['basis_override_used']
        assert not first.diagnostics['basis_reused']
        assert not second.diagnostics['basis_override_requested']
        assert second.diagnostics['basis_reused']
        assert len(service.models) == 2
    finally:
        service.close()


def test_override_never_certifies_an_infeasible_updated_lp():
    service = RepeatedCpuLP(1)
    proposal = dict(col_status=np.array([1], dtype=np.int32),
                    row_status=np.array([0], dtype=np.int32))
    try:
        feasible = ([1.0], dict(
            A_eq=[[1.0]], b_eq=[0.5], bounds=[(0.0, 1.0)],
            method='highs-ds', _stage='maxmin'))
        assert service.solve_batch([feasible])[0].success
        infeasible_kwargs = dict(feasible[1], b_eq=[2.0],
            _basis_override=_override(basis=proposal))
        result = service.solve_batch([([1.0], infeasible_kwargs)])[0]
        assert result.diagnostics['basis_override_used']
        assert not result.success
        assert result.x is None
        assert not result.diagnostics['certificate_passed']
    finally:
        service.close()
