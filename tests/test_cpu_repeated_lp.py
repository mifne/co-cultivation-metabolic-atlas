import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from src.cpu_repeated_lp import RepeatedCpuLP


def _assert_matches(service, requests, **kwargs):
    results = service.solve_batch(requests, **kwargs)
    for result, (c, arguments) in zip(results, requests):
        arguments = {key: value for key, value in arguments.items() if key not in ('_stage', '_initial_basis')}
        arguments['options'] = dict(arguments.get('options', {}), threads=1, parallel=False)
        reference = linprog(c, **arguments)
        assert result.success == reference.success
        if result.success:
            np.testing.assert_allclose(result.fun, reference.fun, atol=1e-8)
            assert result.diagnostics['certificate_passed']
    return results


def test_persistent_updates_matrix_rhs_bounds_and_objective():
    service = RepeatedCpuLP(2)
    try:
        for iteration in range(4):
            requests = []
            for environment in range(2):
                scale = 1.+iteration*.2+environment*.1
                requests.append((np.array([-1.-iteration*.1, -.3]), dict(
                    A_ub=csr_matrix([[scale, 1.], [1., 2.]]), b_ub=[3.-iteration*.1, 4.+environment],
                    bounds=[(0., 1.+iteration*.2), (0., None)], method='highs-ds',
                    options={'presolve': True, 'threads': 1})))
            results = _assert_matches(service, requests)
            assert all(row.diagnostics['basis_reused'] == (iteration > 0) for row in results)
            assert all(row.diagnostics['changed_coefficients'] == (1 if iteration else 0) for row in results)
            assert all(row.diagnostics['model_rebuilt'] == (iteration == 0) for row in results)
    finally:
        service.close()


def test_structural_change_rebuilds_and_stage_cache_is_independent():
    service = RepeatedCpuLP(1)
    try:
        arguments = dict(A_ub=[[1., 0.]], b_ub=[2.], bounds=(0., 3.), _stage='a')
        initial = _assert_matches(service, [([-1., -1.], arguments)])[0]
        assert initial.diagnostics['model_rebuilt']
        changed = dict(arguments, A_ub=[[1., 1.]])
        assert _assert_matches(service, [([-1., -1.], changed)])[0].diagnostics['model_rebuilt']
        stage_b = dict(changed, _stage='b')
        assert _assert_matches(service, [([-1., -1.], stage_b)])[0].diagnostics['model_rebuilt']
        assert _assert_matches(service, [([-1., -1.], changed)])[0].diagnostics['basis_reused']
        assert len(service.models) == 2
    finally:
        service.close()


def test_equalities_free_variables_infeasible_and_unbounded():
    service = RepeatedCpuLP(2)
    try:
        _assert_matches(service, [([1., 2.], dict(A_eq=[[1., 1.]], b_eq=[2.],
            bounds=[(None, None), (0., 3.)], method='highs-ds'))])
        _assert_matches(service, [([1.], dict(A_eq=[[1.]], b_eq=[2.], bounds=[(0., 1.)]))])
        _assert_matches(service, [([-1.], dict(bounds=[(0., None)]))])
    finally:
        service.close()


def test_subset_routing_uses_stable_environment_ids_and_cold_option():
    service = RepeatedCpuLP(2, reuse_basis=False)
    try:
        requests = [([-1.], dict(bounds=(0., 2.))), ([-1.], dict(bounds=(0., 4.)))]
        _assert_matches(service, requests, environment_ids=[10, 20])
        result = _assert_matches(service, [requests[1]], environment_ids=[20])[0]
        assert not result.diagnostics['model_rebuilt']
        assert not result.diagnostics['basis_reused']
        assert len(service.models) == 2
        with pytest.raises(ValueError, match='unique'):
            service.solve_batch(requests, environment_ids=[10, 10])
    finally:
        service.close()
    with pytest.raises(RuntimeError, match='closed'):
        service.solve_batch([])


def test_repeated_equalities_cost_support_and_free_bound_changes():
    service = RepeatedCpuLP(1)
    try:
        for iteration in range(5):
            c = [0., -1.] if iteration % 2 else [1., 0.]
            arguments = dict(A_eq=[[1., 1.+iteration*.1]], b_eq=[2.+iteration*.1],
                bounds=[(None, 4.-iteration*.1), (-2.+iteration*.1, 3.)],
                integrality=np.zeros(2, dtype=int), method='highs-ds')
            result = _assert_matches(service, [(c, arguments)])[0]
            assert result.diagnostics['basis_reused'] == (iteration > 0)
        with pytest.raises(ValueError, match='continuous'):
            service.solve_batch([([1., 0.], dict(arguments, integrality=np.array([1, 0])))])
    finally:
        service.close()


def test_zero_to_nonzero_coefficient_changes_rebuild_but_never_stale_solve():
    service = RepeatedCpuLP(1)
    try:
        for value in (1., 0., 2., 2.):
            request = ([-1., -.5], dict(A_ub=csr_matrix([[1., value]]),
                b_ub=[1.], bounds=(0., 2.), method='highs-ds'))
            _assert_matches(service, [request])
        assert [row['rows'][0]['model_rebuilt'] for row in service.history] == [True, True, True, False]
    finally:
        service.close()


def test_shared_gpu_test_cpu_reference_does_not_change_highs_thread_scheduler():
    # Reproduce the mixed-suite order without running or allocating on a GPU.
    from tests.test_gpu_revised_basis import cpu, problem
    query = problem()
    assert cpu(query).success
    service = RepeatedCpuLP(1)
    try:
        _assert_matches(service, [(query.c, dict(A_ub=query.a, b_ub=query.rhs,
            bounds=list(zip(query.lower, query.upper)), method='highs-ds'))])
    finally:
        service.close()


def test_initial_basis_is_only_a_proposal_and_existing_native_basis_has_priority():
    service = RepeatedCpuLP(1)
    # Both original variables start basic at their upper row bounds. The
    # changed minimization objective needs reoptimization to reach zero.
    proposal = dict(col_status=np.array([1, 1]), row_status=np.array([2, 2]))
    arguments = dict(A_ub=np.eye(2), b_ub=[3., 4.], bounds=(0., 10.),
                     method='highs-ds', _initial_basis=proposal)
    try:
        first = _assert_matches(service, [([1., 2.], arguments)])[0]
        assert first.diagnostics['initial_basis_used']
        assert first.diagnostics['simplex_iterations'] > 0
        np.testing.assert_allclose(first.x, [0., 0.], atol=1e-8)
        # Even a malformed new proposal must not displace the native basis
        # of an existing, valid model. It is not consumed in this branch.
        second = _assert_matches(service, [([-1., -2.], dict(arguments, _initial_basis={}))])[0]
        assert second.diagnostics['basis_reused'] and not second.diagnostics['initial_basis_used']
        np.testing.assert_allclose(second.x, [3., 4.], atol=1e-8)
        changed = dict(arguments, A_ub=[[1., 1.], [0., 1.]])
        third = _assert_matches(service, [([1., 2.], changed)])[0]
        assert third.diagnostics['model_rebuilt'] and third.diagnostics['initial_basis_used']
    finally:
        service.close()


@pytest.mark.parametrize('proposal', [
    {}, dict(col_status=[1], row_status=[2, 2]),
    dict(col_status=[1, 1], row_status=[2]),
    dict(col_status=[1.5, 1.], row_status=[2, 2]),
    dict(col_status=[1, 8], row_status=[2, 2]),
    dict(col_status=[0, 0], row_status=[2, 2]),
    dict(col_status=[1, 1], row_status=[1, 1]),
])
def test_initial_basis_validates_statuses_shapes_and_number_of_basic_variables(proposal):
    service = RepeatedCpuLP(1)
    try:
        request = ([1., 2.], dict(A_ub=np.eye(2), b_ub=[3., 4.], bounds=(0., 10.),
                                 _initial_basis=proposal))
        with pytest.raises(ValueError, match='Initial basis'):
            service.solve_batch([request])
        assert not service.models
    finally:
        service.close()


def test_initial_basis_does_not_certify_an_infeasible_problem():
    service = RepeatedCpuLP(1)
    try:
        request = ([1.], dict(A_eq=[[1.]], b_eq=[2.], bounds=(0., 1.),
            _initial_basis=dict(col_status=[1], row_status=[0])))
        result = _assert_matches(service, [request])[0]
        assert not result.success and result.x is None
        assert result.diagnostics['initial_basis_used'] and not result.diagnostics['certificate_passed']
    finally:
        service.close()
