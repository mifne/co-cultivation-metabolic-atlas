from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.cpu_repeated_lp import RepeatedCpuLP, _problem
from src.temporal_basis_probe import (BasisSnapshot, TemporalBasisProbeBackend,
    reinterpret_basis, request_key, snapshot_from_solver)


def request(a, rhs, c, bounds):
    return np.asarray(c, dtype=float), dict(A_ub=csr_matrix(a), b_ub=np.asarray(rhs, dtype=float),
        bounds=bounds, method='highs-ds', options={'threads':1, 'parallel':False})


def capture(cpu, req, environment_id=0, n_fluxes=2):
    key, problem = request_key(*req, environment_id, n_fluxes)
    result = cpu.solve_batch([req], environment_ids=[environment_id])[0]
    assert result.success
    return problem, snapshot_from_solver(cpu.models[key]['solver'], key[1], problem[0].shape, problem[-1])


def test_previous_basis_certifies_rhs_update_and_records_basis_stability():
    initial = request([[1.,0.],[0.,1.]], [1.,2.], [-1.,-1.], [(0.,10.)]*2)
    changed = request([[1.,0.],[0.,1.]], [3.,4.], [-1.,-1.], [(0.,10.)]*2)
    cpu = RepeatedCpuLP(1, n_fluxes=2)
    try:
        _, old = capture(cpu, initial)
        record = reinterpret_basis(_problem(*changed), old)
        assert record['success'] and record['certificate_passed']
        assert record['primal_residual'] <= 1e-5
        assert record['dual_violation'] <= 1e-7
        assert record['relative_kkt_gap'] <= 1e-7
    finally:
        cpu.close()


def test_previous_basis_fails_closed_when_new_bound_excludes_basic_value():
    initial = request([[1.,1.]], [1.], [-1.,0.], [(0.,10.),(0.,10.)])
    changed = request([[1.,1.]], [1.], [-1.,0.], [(0.,.5),(0.,10.)])
    cpu = RepeatedCpuLP(1, n_fluxes=2)
    try:
        _, old = capture(cpu, initial)
        record = reinterpret_basis(_problem(*changed), old)
        assert not record['success']
        assert record['failure'] == 'original_lp_certificate_failed'
        assert record['primal_residual'] > 1e-5
    finally:
        cpu.close()


def test_previous_basis_cost_change_fails_original_dual_gate():
    initial = request([[1.,0.],[0.,1.]], [1.,2.], [-1.,-1.], [(0.,10.)]*2)
    changed = request([[1.,0.],[0.,1.]], [1.,2.], [1.,1.], [(0.,10.)]*2)
    cpu = RepeatedCpuLP(1, n_fluxes=2)
    try:
        _, old = capture(cpu, initial)
        record = reinterpret_basis(_problem(*changed), old)
        assert not record['success'] and not record['certificate_passed']
        assert record['dual_violation'] > 1e-7 or record['relative_kkt_gap'] > 1e-7
    finally:
        cpu.close()


def test_singular_current_basis_is_diagnostic_failure():
    req = request([[1.,1.],[2.,2.]], [1.,2.], [0.,0.], [(0.,10.)]*2)
    previous = BasisSnapshot(np.array([1,1],np.int8), np.array([2,2],np.int8),
        'exchange', (2,2), 0)
    record = reinterpret_basis(_problem(*req), previous)
    assert not record['success']
    assert record['failure'] == 'singular_or_invalid_current_basis'


def test_infinite_lower_and_inappropriate_zero_row_statuses_are_rejected():
    req = request([[1.]], [1.], [0.], [(0.,10.)])
    lower = BasisSnapshot(np.array([1],np.int8), np.array([0],np.int8),
        'exchange', (1,1), 0)
    assert reinterpret_basis(_problem(*req), lower)['failure'] == 'unsupported_infinite_lower_row_status'
    negative = request([[1.]], [-1.], [0.], [(0.,10.)])
    zero = BasisSnapshot(np.array([1],np.int8), np.array([3],np.int8),
        'exchange', (1,1), 0)
    assert reinterpret_basis(_problem(*negative), zero)['failure'] == 'zero_row_status_outside_current_bounds'


def test_valid_zero_row_status_uses_zero_activity_not_rhs():
    req = request([[1.]], [2.], [0.], [(0.,10.)])
    zero = BasisSnapshot(np.array([1],np.int8), np.array([3],np.int8),
        'exchange', (1,1), 0)
    record = reinterpret_basis(_problem(*req), zero)
    assert record['success'] and record['objective'] == 0.


def test_failed_probe_drops_stale_basis_and_duplicate_ids_fail_before_cpu():
    feasible = request([[1.]], [1.], [-1.], [(0.,10.)])
    infeasible = request([[1.]], [-1.], [-1.], [(0.,10.)])
    backend = TemporalBasisProbeBackend(RepeatedCpuLP(1,n_fluxes=1),
        stages=('exchange',),workers=1)
    try:
        assert backend.solve_batch([feasible],environment_ids=[7])[0].success
        key, _ = request_key(*feasible, 7, 1)
        assert key in backend.previous
        assert not backend.solve_batch([infeasible],environment_ids=[7])[0].success
        assert key not in backend.previous
        before = len(backend.cpu.history)
        with pytest.raises(ValueError, match='unique stable'):
            backend.solve_batch([feasible,feasible],environment_ids=[7,7])
        assert len(backend.cpu.history) == before
    finally:
        backend.close()


def test_reset_clears_previous_and_cpu_models():
    req = request([[1.]], [1.], [-1.], [(0.,10.)])
    backend = TemporalBasisProbeBackend(RepeatedCpuLP(1,n_fluxes=1),
        stages=('exchange',),workers=1)
    try:
        assert backend.solve_batch([req])[0].success
        assert backend.previous and backend.cpu.models
        backend.reset_trajectory()
        assert not backend.previous and not backend.cpu.models
        assert backend.solve_batch([req])[0].success
        assert not backend.history[-1]['rows'][0]['probe']['available']
    finally:
        backend.close()


def test_probe_success_or_failure_never_changes_exact_cpu_trajectory():
    sequence = [
        request([[1.,0.],[0.,1.]], [1.,2.], [-1.,-1.], [(0.,10.)]*2),
        request([[1.,0.],[0.,1.]], [3.,4.], [-1.,-1.], [(0.,10.)]*2),
        request([[1.,0.],[0.,1.]], [3.,4.], [1.,1.], [(0.,10.)]*2),
    ]
    direct = RepeatedCpuLP(1,n_fluxes=2)
    probed = TemporalBasisProbeBackend(RepeatedCpuLP(1,n_fluxes=2),
        stages=('exchange',),workers=1)
    try:
        expected = [direct.solve_batch([req])[0] for req in sequence]
        actual = [probed.solve_batch([req])[0] for req in sequence]
        assert [r.success for r in actual] == [r.success for r in expected]
        np.testing.assert_allclose([r.fun for r in actual], [r.fun for r in expected], rtol=0., atol=0.)
        for left, right in zip(actual, expected):
            np.testing.assert_allclose(left.x, right.x, rtol=0., atol=0.)
        assert probed.history[1]['rows'][0]['probe']['success']
        assert not probed.history[2]['rows'][0]['probe']['success']
    finally:
        direct.close(); probed.close()
