"""CPU dictionary setup may download metadata; timed solves must not use GPU."""
import builtins
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from src.cpu_dictionary_lp import CpuDictionaryLP, cpu_features
from src.gpu_certified_basis import NormalizedLP, compile_basis


class _SetupDeviceArray:
    __cuda_array_interface__ = {}

    def __init__(self, values):
        self.values = np.array(values)
        self.allowed = True
        self.downloads = 0

    def get(self):
        if not self.allowed:
            raise AssertionError('GPU download inside a timed CPU-only solve')
        self.downloads += 1
        return self.values.copy()


class _Coordinates:
    n_fluxes = 1

    def normalize(self, a, rhs, lower, upper, c, neq):
        return NormalizedLP(a, rhs, lower, upper, c, neq, np.ones(len(c)), np.ones(len(rhs)))


def _request(rhs=5., first_coefficient=1.):
    return np.array([-2., -1.]), dict(
        A_ub=csr_matrix([[first_coefficient, 1.], [1., 0.]]),
        b_ub=np.array([rhs, 3.]), bounds=[(0., 10.), (0., 10.)], method='highs-ds',
    )


def _fixture():
    requests = [_request(5.), _request(0.1)]
    anchors = []
    for c, kwargs in requests:
        anchors.append(compile_basis(kwargs['A_ub'], kwargs['b_ub'], np.zeros(2), np.full(2, 10.), c, 0))
    wrapped = []

    def device(values):
        array = _SetupDeviceArray(values)
        wrapped.append(array)
        return array

    bank = SimpleNamespace(
        host_a=requests[0][1]['A_ub'], neq=0, variable_rows=np.array([0]),
        centers=device(np.array([[np.log1p(5.)], [np.log1p(0.1)]], dtype=np.float32)),
        feature_indices=device(np.array([0], dtype=np.int64)), feature_scale=device(np.ones(1, np.float32)),
        evaluators=[SimpleNamespace(d=dict(kind=device(anchor['kind']), active=device(anchor['active']))) for anchor in anchors],
    )
    return _Coordinates(), {('aggregate', 2, 2, 0): bank}, wrapped, requests


def _reference(request):
    objective, kwargs = request
    return linprog(objective, **dict(kwargs, options={'threads': 1, 'parallel': False}))


def test_cpu_features_match_field_order_finite_encoding_and_dtype():
    inputs = dict(rhs=np.array([0., -1.]), lower=np.array([-np.inf]), upper=np.array([np.inf]),
        c=np.array([np.nan]), delta=np.array([[2., -3.]]), col_scale=np.array([4.]), row_scale=np.array([5.]))
    result = cpu_features(inputs)
    raw = np.array([0., -1., -1e13, 1e13, 0., 2., -3., 4., 5.])
    expected = (np.sign(raw)*np.log1p(np.abs(raw))).astype(np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32 and np.isfinite(result).all()


def test_cpu_nearest_basis_initialization_and_warm_reuse_never_call_gpu(monkeypatch):
    coordinates, banks, wrapped, requests = _fixture()
    service = CpuDictionaryLP(coordinates, banks, workers=2)
    expected = [_reference(request) for request in requests]
    assert all(array.downloads == 1 for array in wrapped)
    for array in wrapped:
        array.allowed = False
    original_import = builtins.__import__

    def cpu_only_import(name, *args, **kwargs):
        if name.split('.')[0] in {'cupy', 'cupyx', 'cuda'}:
            raise AssertionError('GPU library used inside CPU comparator')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', cpu_only_import)
    captured = []
    original_solve = service.cpu.solve_batch

    def capture(prepared, **kwargs):
        captured.append(prepared)
        return original_solve(prepared, **kwargs)

    monkeypatch.setattr(service.cpu, 'solve_batch', capture)
    try:
        results = service.solve_batch(requests)
        assert [row.diagnostics['dictionary_candidate'] for row in results] == [0, 1]
        assert all(row.diagnostics['initial_basis_used'] for row in results)
        assert all(row.success and row.diagnostics['certificate_passed'] for row in results)
        np.testing.assert_allclose([row.fun for row in results], [row.fun for row in expected], atol=1e-8)
        for request in requests:
            assert '_initial_basis' not in request[1]  # User kwargs were not mutated.
        proposals = [kwargs['_initial_basis'] for _, kwargs in captured[0]]
        np.testing.assert_array_equal(proposals[0]['col_status'], [1, 1])
        np.testing.assert_array_equal(proposals[0]['row_status'], [2, 2])
        np.testing.assert_array_equal(proposals[1]['col_status'], [1, 0])
        np.testing.assert_array_equal(proposals[1]['row_status'], [2, 1])
        # New nearest candidates are still only proposals; the existing native
        # model/basis takes priority over reinitializing from the dictionary.
        repeated = service.solve_batch(requests[::-1])
        assert [row.diagnostics['dictionary_candidate'] for row in repeated] == [None, None]
        assert [row.diagnostics['dictionary_reason'] for row in repeated] == ['existing_cpu_basis']*2
        assert all(row.diagnostics['basis_reused'] and not row.diagnostics['initial_basis_used'] for row in repeated)
        assert all('_initial_basis' not in kwargs for _, kwargs in captured[1])
        np.testing.assert_allclose([row.fun for row in repeated], [row.fun for row in expected[::-1]], atol=1e-8)
        assert len(service.history) == 2
        assert service.history[-1]['dictionary_seconds'] >= 0.
        assert service.history[-1]['cpu_lp_calls'] == 2
    finally:
        service.close()
    with pytest.raises(RuntimeError, match='closed'):
        service.solve_batch(requests)


def test_missing_bank_and_unsupported_coordinates_still_solve_original_lp():
    coordinates, banks, _, requests = _fixture()
    missing = CpuDictionaryLP(coordinates, {}, workers=1)
    try:
        result = missing.solve_batch([requests[0]])[0]
        assert result.success and result.diagnostics['dictionary_reason'] == 'missing_bank'
        assert not result.diagnostics['initial_basis_used']
    finally:
        missing.close()

    class Unsupported(_Coordinates):
        def normalize(self, *args):
            raise ValueError('Unsupported positive coordinate scaling')

    service = CpuDictionaryLP(Unsupported(), banks, workers=1)
    try:
        result = service.solve_batch([requests[0]])[0]
        assert result.success and result.diagnostics['dictionary_reason'] == 'unsupported_coordinates'
        assert not result.diagnostics['initial_basis_used']
    finally:
        service.close()


def test_changed_sparsity_can_reapply_proposal_and_reset_clears_trajectory():
    coordinates, banks, _, requests = _fixture()
    service = CpuDictionaryLP(coordinates, banks, workers=1)
    try:
        service.solve_batch([requests[0]], environment_ids=[23])
        changed = _request(5., first_coefficient=0.)
        result = service.solve_batch([changed], environment_ids=[23])[0]
        assert result.success and result.diagnostics['model_rebuilt']
        assert result.diagnostics['initial_basis_used']
        np.testing.assert_allclose(result.fun, _reference(changed).fun, atol=1e-8)
        service.reset_trajectory()
        assert not service.cpu.models
        restarted = service.solve_batch([requests[0]])[0]
        assert restarted.diagnostics['model_rebuilt'] and restarted.diagnostics['initial_basis_used']
    finally:
        service.close()


def test_setup_copies_metadata_and_rejects_invalid_feature_scale():
    coordinates, banks, wrapped, _ = _fixture()
    service = CpuDictionaryLP(coordinates, banks, workers=1)
    try:
        centers = service.banks[('aggregate', 2, 2, 0)]['centers'].copy()
        wrapped[0].values[:] = 99.
        np.testing.assert_array_equal(service.banks[('aggregate', 2, 2, 0)]['centers'], centers)
    finally:
        service.close()
    banks[('aggregate', 2, 2, 0)].feature_scale = np.zeros(1)
    with pytest.raises(ValueError, match='positive and finite'):
        CpuDictionaryLP(coordinates, banks, workers=1)


class _CpuCoverageRouter:
    xp = np
    candidate_count = 2
    input_dim = 1

    def __init__(self, order=(1, 0)):
        self.order = np.asarray([order])
        self.calls = 0

    def rank(self, features, k=None):
        assert isinstance(features, np.ndarray) and features.shape == (1, 1)
        assert k == 2
        self.calls += 1
        return self.order.copy()


def test_same_numpy_coverage_router_is_used_only_for_cold_or_rebuilt_cpu_models():
    coordinates, banks, _, requests = _fixture()
    key = next(iter(banks)); router = _CpuCoverageRouter()
    service = CpuDictionaryLP(coordinates, banks, workers=1,
        coverage_routers={key:router})
    try:
        first = service.solve_batch([requests[0]], environment_ids=[17])[0]
        assert first.success and first.diagnostics['dictionary_candidate'] == 1
        assert first.diagnostics['dictionary_reason'] == 'learned_coverage'
        assert router.calls == 1
        # A live HiGHS basis remains stronger than any new proposal; even the
        # feature encoder/router is skipped on an ordinary RHS update.
        warm = service.solve_batch([_request(4.)], environment_ids=[17])[0]
        assert warm.success and warm.diagnostics['dictionary_candidate'] is None
        assert warm.diagnostics['dictionary_reason'] == 'existing_cpu_basis'
        assert router.calls == 1
        rebuilt = service.solve_batch([_request(4., first_coefficient=0.)],
            environment_ids=[17])[0]
        assert rebuilt.success and rebuilt.diagnostics['model_rebuilt']
        assert rebuilt.diagnostics['dictionary_reason'] == 'learned_coverage'
        assert router.calls == 2
    finally:
        service.close()


@pytest.mark.parametrize('mutation', ['device', 'candidates', 'features', 'unknown_key'])
def test_cpu_coverage_router_setup_rejects_unmatched_or_device_router(mutation):
    coordinates, banks, _, _ = _fixture(); key = next(iter(banks))
    router = _CpuCoverageRouter(); mapping = {key:router}
    if mutation == 'device':
        router.xp = object()
    elif mutation == 'candidates':
        router.candidate_count = 1
    elif mutation == 'features':
        router.input_dim = 2
    else:
        mapping = {('maxmin', 2, 2, 0):router}
    with pytest.raises(ValueError, match='CPU coverage router'):
        CpuDictionaryLP(coordinates, banks, workers=1, coverage_routers=mapping)


def test_cpu_coverage_router_malformed_order_fails_before_basis_is_selected():
    coordinates, banks, _, requests = _fixture(); key = next(iter(banks))
    service = CpuDictionaryLP(coordinates, banks, workers=1,
        coverage_routers={key:_CpuCoverageRouter((1, 1))})
    try:
        with pytest.raises(ValueError, match='full candidate permutation'):
            service.solve_batch([requests[0]])
        assert not service.cpu.history
    finally:
        service.close()
