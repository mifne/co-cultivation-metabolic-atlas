"""Exact, opt-in exchange performance-row support update tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.cpu_repeated_lp import RepeatedCpuLP, _problem


def _exchange_request(performance_column, *, stage="exchange", x1_upper=2.0,
                      transfer_column=1, extra_row=False):
    """Small stage-three LP with one absolute-value exchange auxiliary."""
    matrix = np.zeros((3 + int(extra_row), 4), dtype=float)
    matrix[0, performance_column] = -1.0
    matrix[1, transfer_column] = 1.0
    matrix[1, 3] = -1.0
    matrix[2, transfer_column] = -1.0
    matrix[2, 3] = -1.0
    rhs = np.array([-1.0, 0.0, 0.0] + ([2.0] if extra_row else []))
    return np.array([0.0, 0.0, 0.0, 1.0]), dict(
        A_ub=csr_matrix(matrix),
        b_ub=rhs,
        bounds=[(0.0, 2.0), (0.0, x1_upper), (0.0, 0.0), (0.0, None)],
        method="highs-ds",
        _stage=stage,
    )


def _assert_certified(result, expected_fun):
    assert result.success
    assert result.fun == pytest.approx(expected_fun, abs=1e-9)
    assert result.diagnostics["certificate_passed"]


class _ChangeCoeffFailure:
    def __init__(self, solver, status):
        self.solver = solver
        self.status = status
        self.change_calls = 0

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def changeCoeff(self, row, column, value):
        self.change_calls += 1
        if self.change_calls == 1:
            # A rejected mutation is deliberately treated as potentially
            # side-effecting by the implementation.
            self.solver.changeCoeff(row, column, value)
            return self.status.kError
        return self.solver.changeCoeff(row, column, value)


class _InvalidBasisAndSetterFailure:
    def __init__(self, solver, status):
        self.solver = solver
        self.status = status
        self.changed = False
        self.set_calls = 0

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def changeCoeff(self, row, column, value):
        status = self.solver.changeCoeff(row, column, value)
        self.changed = True
        return status

    def getBasis(self):
        if self.changed:
            return SimpleNamespace(valid=False)
        return self.solver.getBasis()

    def setBasis(self, basis):
        self.set_calls += 1
        return self.status.kError


def test_default_off_keeps_the_conservative_structural_rebuild():
    service = RepeatedCpuLP(1, n_fluxes=2)
    try:
        service.solve_batch([_exchange_request(0)])
        changed = _exchange_request(1)
        result = service.solve_batch([changed])[0]
        _assert_certified(result, 1.0)
        assert result.diagnostics["model_rebuilt"]
        assert not result.diagnostics["exchange_support_updated"]
        assert not result.diagnostics["basis_reused"]
    finally:
        service.close()


def test_performance_support_union_zeros_stale_coefficient_and_reuses_basis():
    service = RepeatedCpuLP(
        1, n_fluxes=2, exchange_support_updates=True
    )
    try:
        initial = service.solve_batch([_exchange_request(0)])[0]
        assert initial.success and initial.fun == pytest.approx(0.0)
        changed = _exchange_request(1)
        arrays = _problem(*changed)
        entry = next(iter(service.models.values()))
        assert service.can_reuse_structure(
            arrays[0], entry, "exchange", arrays[-1]
        )

        result = service.solve_batch([changed])[0]
        _assert_certified(result, 1.0)
        # If the old -x0 coefficient were not explicitly set to zero, x0
        # could satisfy the new floor at zero auxiliary cost.  The exact
        # objective 1 therefore proves that the stale support was removed.
        assert result.fun == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(result.x[[0, 1, 3]], [0.0, 1.0, 1.0], atol=1e-9)
        diagnostics = result.diagnostics
        assert not diagnostics["model_rebuilt"]
        assert diagnostics["exchange_support_updated"]
        assert diagnostics["exchange_support_update_row"] == 0
        assert diagnostics["changed_coefficients"] == 2
        assert diagnostics["basis_reused"]
        assert diagnostics["exchange_support_update_recovery"] is None
    finally:
        service.close()


@pytest.mark.parametrize(
    ("changed", "expected_fun"),
    [
        (_exchange_request(1, stage="maxmin"), 1.0),
        (_exchange_request(1, transfer_column=0), 0.0),
        (_exchange_request(1, extra_row=True), 1.0),
    ],
)
def test_nonexchange_other_row_and_shape_changes_still_rebuild(
    changed, expected_fun
):
    service = RepeatedCpuLP(
        1, n_fluxes=2, exchange_support_updates=True
    )
    try:
        service.solve_batch([_exchange_request(0)], environment_ids=[8])
        result = service.solve_batch([changed], environment_ids=[8])[0]
        _assert_certified(result, expected_fun)
        assert result.diagnostics["model_rebuilt"]
        assert not result.diagnostics["exchange_support_updated"]
    finally:
        service.close()


def test_support_updates_remain_isolated_between_environments():
    service = RepeatedCpuLP(
        1, n_fluxes=2, exchange_support_updates=True
    )
    try:
        initial = [_exchange_request(0), _exchange_request(0)]
        service.solve_batch(initial, environment_ids=[11, 29])
        changed, unchanged = _exchange_request(1), _exchange_request(0)
        first, second = service.solve_batch(
            [changed, unchanged], environment_ids=[11, 29]
        )
        _assert_certified(first, 1.0)
        _assert_certified(second, 0.0)
        assert first.fun == pytest.approx(1.0)
        assert second.fun == pytest.approx(0.0)
        assert first.diagnostics["exchange_support_updated"]
        assert not second.diagnostics["exchange_support_updated"]
        assert second.diagnostics["basis_reused"]
        assert len(service.models) == 2
        assert len({id(entry["solver"]) for entry in service.models.values()}) == 2
    finally:
        service.close()


def test_change_coefficient_setter_failure_rebuilds_the_exact_current_lp():
    service = RepeatedCpuLP(
        1, n_fluxes=2, exchange_support_updates=True
    )
    try:
        service.solve_batch([_exchange_request(0)])
        entry = next(iter(service.models.values()))
        proxy = _ChangeCoeffFailure(entry["solver"], service.hp.HighsStatus)
        entry["solver"] = proxy
        changed = _exchange_request(1)
        result = service.solve_batch([changed])[0]
        _assert_certified(result, 1.0)
        assert proxy.change_calls == 1
        assert result.fun == pytest.approx(1.0)
        diagnostics = result.diagnostics
        assert diagnostics["model_rebuilt"]
        assert not diagnostics["exchange_support_updated"]
        assert not diagnostics["basis_reused"]
        assert diagnostics["exchange_support_update_recovery"] == "model_rebuilt"
        assert "coefficient update" in diagnostics["exchange_support_update_error"]
    finally:
        service.close()


def test_invalid_live_basis_and_failed_basis_setter_rebuild_exactly():
    service = RepeatedCpuLP(
        1, n_fluxes=2, exchange_support_updates=True
    )
    try:
        service.solve_batch([_exchange_request(0)])
        entry = next(iter(service.models.values()))
        proxy = _InvalidBasisAndSetterFailure(
            entry["solver"], service.hp.HighsStatus
        )
        entry["solver"] = proxy
        changed = _exchange_request(1)
        result = service.solve_batch([changed])[0]
        _assert_certified(result, 1.0)
        assert proxy.set_calls == 1
        diagnostics = result.diagnostics
        assert diagnostics["model_rebuilt"]
        assert not diagnostics["exchange_support_updated"]
        assert diagnostics["exchange_support_update_recovery"] == "model_rebuilt"
        assert "basis reuse" in diagnostics["exchange_support_update_error"]
    finally:
        service.close()


def test_support_update_cannot_accept_an_infeasible_original_lp():
    service = RepeatedCpuLP(
        1, n_fluxes=2, exchange_support_updates=True
    )
    try:
        service.solve_batch([_exchange_request(0, x1_upper=0.5)])
        infeasible = _exchange_request(1, x1_upper=0.5)
        result = service.solve_batch([infeasible])[0]
        assert not result.success and result.x is None and result.fun is None
        assert result.diagnostics["exchange_support_updated"]
        assert not result.diagnostics["certificate_passed"]
        assert result.diagnostics["cpu_solver_runs"] == 1
        entry = next(iter(service.models.values()))
        assert entry["basis"] is None and not entry["solver"].getBasis().valid
    finally:
        service.close()
