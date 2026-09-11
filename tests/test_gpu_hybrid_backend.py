"""Tiny exact LPs exercise routing; no large benchmark is performed here."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import eye

from src.gpu_hybrid_lp import HybridCertifiedBackend, observable_projection, route_rejected


def test_route_rejected_acceptance_precedes_dispersion_and_unknown_is_explicit():
    direct, repair, unknown = route_rejected(
        [True, False, False, False, False], [np.inf, .1, 2., .1, np.nan], [0, 3, 3, 1, 4], 1.)
    np.testing.assert_array_equal(direct, [False, False, True, True, True])
    np.testing.assert_array_equal(repair, [False, True, False, False, False])
    np.testing.assert_array_equal(unknown, [True, False, False, True, True])
    with pytest.raises(ValueError, match='same shape'):
        route_rejected([False], [], [1], 1.)
    with pytest.raises(ValueError, match='Nonnegative'):
        route_rejected([False], [0.], [2], -1.)


def test_observable_projection_retains_species_outputs_and_excludes_cycles_auxiliaries():
    metadata = dict(growth_terms={'a': (0, 1.), 'b': (1, 2.)},
        exchange_terms={'glucose': [('a', 2, -1., None), ('b', 3, -2., None)]},
        reaction_ids=['biomass_a', 'biomass_b', 'EX_a', 'EX_b', 'PHB_syn', 'cycle'],
        reaction_species=['a', 'b', 'a', 'b', 'a', 'a'])
    matrix, scales, labels = observable_projection(metadata, 8)
    assert matrix.shape == (5, 8)
    np.testing.assert_array_equal(matrix @ np.arange(8.), [0., 2., -2., -6., 4.])
    np.testing.assert_array_equal(scales, np.ones(5))
    assert matrix[:, 5:].nnz == 0
    assert labels[-1] == 'product:a:PHB_syn'


@pytest.fixture
def tiny_hybrid_factory(tmp_path, monkeypatch):
    pytest.importorskip('cupy')
    import src.gpu_hybrid_lp as hybrid_module
    from src.gpu_basis_bank import GpuBasisBank
    from src.gpu_certified_basis import NormalizedLP, compile_basis
    from src.gpu_compact_basis import CompactBank, project_basis
    from tests.test_gpu_revised_basis import problem
    p, alternate, q = problem(), problem(cost=(-1., -2.)), problem(cost=(2., 1.))
    queries = [p, alternate, q, problem(cost=(1., 2.)), problem(rhs=(-1., 3.)), problem(rhs=(.1, 3.))]
    anchors = [compile_basis(v.a, v.rhs, v.lower, v.upper, v.c, v.neq) for v in (p, alternate)]
    full = GpuBasisBank(anchors, [0])
    data = {key: value.get() for key, value in full.prepare_host(queries).items()}
    entries = [project_basis(anchor, data, np.zeros((1, 1, 2)), [0]) for anchor in anchors]
    bank = CompactBank(dict(a=p.a, neq=0), [0], entries, np.zeros((2, 1)),
        np.array([0]), np.ones(1), candidate_ranking='count', full_batch_candidates=True)
    bank.configure_observables(eye(2, format='csr'), np.ones(2))
    for index, anchor in enumerate(anchors):
        path = tmp_path / ('inverse_' + str(index) + '.npz'); inverse = anchor['inverse']
        np.savez(path, inverse_data=inverse.data, inverse_indices=inverse.indices,
                 inverse_indptr=inverse.indptr, inverse_shape=inverse.shape)
        bank.evaluators[index].repair_path = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    key = ('exchange', 2, 2, 0)
    coords = SimpleNamespace(n_fluxes=2, normalize=lambda a, r, lo, hi, c, neq:
        NormalizedLP(a, r, lo, hi, c, neq, np.ones(2), np.ones(2)))
    monkeypatch.setattr(hybrid_module, 'stage_key', lambda *args: key)
    services = []

    def factory(**options):
        service = HybridCertifiedBackend(coords, {key: bank}, cpu_workers=2,
            repair_columns=options.pop('repair_columns', 2), repair_pivots=options.pop('repair_pivots', 32),
            max_repair_batch=2, **options)
        services.append(service)
        return service, bank, p, q
    yield factory
    for service in services: service.close()
    bank.clear_graph_cache()


def _request(problem):
    return problem.c, dict(A_ub=problem.a, b_ub=problem.rhs,
                          bounds=list(zip(problem.lower, problem.upper)), method='highs-ds')


def _frozen_bank(bank):
    return [[(key, value.get().copy()) for key, value in evaluator.d.items()]
            for evaluator in bank.evaluators]


def _assert_bank_unchanged(bank, before):
    assert len(bank.evaluators) == len(before)
    for evaluator, fields in zip(bank.evaluators, before):
        assert set(evaluator.d) == {key for key, _ in fields}
        for key, value in fields: np.testing.assert_array_equal(evaluator.d[key].get(), value)


def test_direct_cpu_stage_bypasses_gpu_bank_and_preserves_exact_certificate(tiny_hybrid_factory, monkeypatch):
    from tests.test_gpu_revised_basis import cpu
    service,bank,p,q=tiny_hybrid_factory(gpu_stages=('maxmin',),repair_rounds=0)
    expected=[cpu(v).fun for v in (p,q)]
    def forbidden(*args,**kwargs):
        raise AssertionError('Disabled GPU stage must not prepare or screen the bank')
    monkeypatch.setattr(bank,'prepare_host',forbidden)
    monkeypatch.setattr(bank,'evaluate_device',forbidden)
    for _ in range(2):
        rows=service.solve_batch([_request(p),_request(q)])
        assert all(row.success and row.diagnostics['certificate_passed'] for row in rows)
        np.testing.assert_allclose([row.fun for row in rows],expected,atol=1e-8)
        record=service.history[-1]
        assert record['cpu_lp_calls']==2 and record['candidate_evaluations']==0
        assert record['routes']==['cpu_stage_policy']*2
        assert record['bank_seconds']==0. and record['seconds']>0.
    assert service.direct_cpu.cpu is service.cpu
    assert all(row['basis_reused'] for row in service.cpu.history[-1]['rows'])
    service.direct_cpu.close()  # A borrowing dictionary must not close its owner.
    assert not service.cpu.closed
    service.reset_trajectory()
    assert not service.cpu.models


def test_direct_cpu_stage_cannot_adopt_infeasible_lp(tiny_hybrid_factory):
    from tests.test_gpu_revised_basis import problem
    service,bank,p,q=tiny_hybrid_factory(gpu_stages=('maxmin',),repair_rounds=0)
    result=service.solve_batch([_request(problem(rhs=(-1.,3.)))])[0]
    assert not result.success and result.x is None
    assert service.history[-1]['accepted']==[False]
    assert service.history[-1]['cpu_lp_calls']==1


def test_cpu_selected_stage_still_solves_when_gpu_bank_shape_is_absent(tiny_hybrid_factory,monkeypatch):
    import src.gpu_hybrid_lp as module
    from scipy.sparse import vstack,csr_matrix
    service,bank,p,q=tiny_hybrid_factory(gpu_stages=('maxmin',),repair_rounds=0)
    monkeypatch.setattr(module,'stage_key',lambda *args:('exchange',3,2,0))
    objective,kwargs=_request(p)
    kwargs.update(A_ub=vstack((p.a,csr_matrix([[1.,0.]])),format='csr'),b_ub=np.r_[p.rhs,100.])
    result=service.solve_batch([(objective,kwargs)])[0]
    assert result.success and result.diagnostics['certificate_passed']
    assert result.diagnostics['dictionary_reason']=='missing_bank'
    assert service.history[-1]['routes']==['cpu_stage_policy']


def test_high_dispersion_cpu_fallback_is_certified_and_preserves_environment_ids(tiny_hybrid_factory):
    from tests.test_gpu_revised_basis import cpu, problem
    service, bank, p, q = tiny_hybrid_factory(dispersion_threshold=.1, repair_rounds=2)
    before = _frozen_bank(bank)
    other = problem(cost=(1., 2.))
    result = service.solve_batch([_request(p), _request(q), _request(other)])
    assert all(row.success for row in result)
    assert service.history[-1]['routes'] == ['gpu_dictionary', 'cpu_high_dispersion', 'cpu_high_dispersion']
    assert service.history[-1]['cpu_lp_calls'] == 2
    cpu_rows = service.cpu.history[-1]['rows']
    assert [row['environment_id'] for row in cpu_rows] == [1, 2]
    assert all(row['certificate_passed'] for row in cpu_rows)
    for row, query in zip(result, (p, q, other)):
        np.testing.assert_allclose(row.fun, cpu(query).fun, atol=1e-8)
    result = service.solve_batch([_request(p), _request(p), _request(q)])
    assert all(row.success for row in result)
    assert [row['environment_id'] for row in service.cpu.history[-1]['rows']] == [2]
    assert service.cpu.history[-1]['rows'][0]['basis_reused']
    assert {key[0] for key in service.cpu.models} == {1, 2}
    _assert_bank_unchanged(bank, before)


def test_unknown_dispersion_does_not_mean_safe_gpu_solution(tiny_hybrid_factory):
    service, bank, p, q = tiny_hybrid_factory(dispersion_threshold=np.inf, candidate_limit=1)
    before = _frozen_bank(bank)
    result = service.solve_batch([_request(p), _request(q)])
    assert all(row.success for row in result)
    assert service.history[-1]['routes'] == ['gpu_dictionary', 'cpu_unknown_dispersion']
    assert service.history[-1]['candidate_count'] == [1, 1]
    _assert_bank_unchanged(bank, before)
    service.reset_trajectory()
    assert not service.previous_candidates and not service.cpu.models


def test_low_dispersion_can_finish_on_gpu_without_any_cpu_lp(tiny_hybrid_factory, monkeypatch):
    import highspy
    from tests.test_gpu_revised_basis import cpu
    service, bank, p, q = tiny_hybrid_factory(dispersion_threshold=np.inf, repair_rounds=2, repair_columns=1)
    expected = cpu(q).fun
    monkeypatch.setattr(highspy.Highs, 'run', lambda *args, **kwargs:
        (_ for _ in ()).throw(AssertionError('Unexpected online CPU fallback')))
    result = service.solve_batch([_request(p), _request(q)])
    assert all(row.success for row in result)
    np.testing.assert_allclose(result[1].fun, expected, atol=1e-8)
    assert service.history[-1]['routes'] == ['gpu_dictionary', 'gpu_restricted_repair']
    assert service.history[-1]['cpu_lp_calls'] == 0 and not service.cpu.history
    groups = service.history[-1]['groups']
    assert groups and all(group['ids'] == [1] for group in groups)
    assert all(value <= 1e-7 for value in service.history[-1]['relative_kkt_gap'])


def test_low_dispersion_budget_exhaustion_still_uses_certified_cpu_safety(tiny_hybrid_factory):
    service, bank, p, q = tiny_hybrid_factory(dispersion_threshold=np.inf, repair_rounds=1,
        repair_columns=1, repair_pivots=1)
    result = service.solve_batch([_request(p), _request(q)])
    assert all(row.success for row in result)
    assert service.history[-1]['routes'] == ['gpu_dictionary', 'cpu_gpu_budget_exhausted']
    assert service.history[-1]['cpu_lp_calls'] == 1
    repair = [group for group in service.history[-1]['groups'] if group['route'] == 'gpu_restricted_repair']
    assert repair and repair[0]['accepted'] == [False]
    assert service.cpu.history[-1]['rows'][0]['certificate_passed']


def test_infeasible_cpu_fallback_never_returns_a_flux(tiny_hybrid_factory):
    from tests.test_gpu_revised_basis import problem
    service, bank, p, q = tiny_hybrid_factory(candidate_limit=1, repair_rounds=0)
    result = service.solve_batch([_request(problem(rhs=(-1., 3.)))])[0]
    assert not result.success and result.x is None and result.fun is None
    assert service.history[-1]['cpu_lp_calls'] == 1
    assert not service.cpu.history[-1]['rows'][0]['certificate_passed']


@pytest.mark.parametrize('replay', [False, True])
def test_heterogeneous_hybrid_preserves_certified_cpu_gpu_routes(tiny_hybrid_factory, replay):
    from tests.test_gpu_revised_basis import cpu
    service, bank, p, q = tiny_hybrid_factory(heterogeneous_candidates=True,
        heterogeneous_replay=replay, repair_rounds=0, candidate_limit=2, cpu_basis_proposals=True)
    expected = [cpu(query).fun for query in (p, q)]
    for queries in ((p, q), (q, p)):
        result = service.solve_batch([_request(query) for query in queries])
        assert all(row.success for row in result)
        np.testing.assert_allclose([row.fun for row in result], expected if queries[0] is p else expected[::-1], atol=1e-8)
        record = service.history[-1]
        assert record['candidate_engine'] == 'heterogeneous'
        assert record['candidate_evaluations'] == 4
        assert record['cpu_lp_calls'] == 1
        assert all(v <= 1e-5 for v in record['primal_residual'])
        assert all(v <= 1e-7 for v in record['dual_violation'])
        assert all(v <= 1e-7 for v in record['relative_kkt_gap'])
    if replay:
        assert len(next(iter(service.heterogeneous_banks.values())).graph_cache) == 1
