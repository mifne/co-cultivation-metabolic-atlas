"""CPU reference/mocks for fixed device payload gathers; no GPU execution."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix

from src.gpu_batched_ipm import constraint_form
from src.gpu_ipm_device_payload import DeviceIPMPayloadBinder
from src.gpu_ipm_numeric_update import _existing_host_state, _payloads, _target, prepare_host_rebind
from src.gpu_lp_numeric import DeviceLPBatch
from tests import test_gpu_ipm_numeric_update as fixture


def _states(state):
    face = tuple(p.reduced for p in state.face_plans)
    return dict(full=state.full_problems, forest=state.forest_problems, face=face,
        secondary=(tuple(p.problem for p in state.secondary_forest_reductions)
                   if state.secondary_forest_reductions else face), core=state.problems)


def _batches(state):
    return {phase: DeviceLPBatch(np.concatenate([p[0].data for p in ps]),
        *(np.stack([p[j] for p in ps]) for j in (1, 2, 3, 4)))
        for phase, ps in _states(state).items()}


def _witnesses(state):
    def triple(reductions):
        return tuple(np.stack([getattr(p, name) for p in reductions]).astype(np.int64)
            for name in ('lower_witness', 'upper_witness', 'fallback_witness'))
    return (triple(state.forest_reductions),
            triple(state.secondary_forest_reductions) if state.secondary_forest_reductions else None)


def _solver(*, second=False, exact=True, condensed=False):
    solver = fixture._mock_solver(second_forest=second, exact_equalities=exact)
    mock_cp = solver.cp
    solver.factor.stream.ptr = 0
    solver.factor.indptr = np.array(solver.factor.host_pattern.indptr, dtype=np.int32)
    solver.factor.indices = np.array(solver.factor.host_pattern.indices, dtype=np.int32)
    pattern = solver.factor.host_pattern
    rows = np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr))
    keys = rows*pattern.shape[0]+pattern.indices
    solver.factor.transpose_positions = np.searchsorted(keys, pattern.indices*pattern.shape[0]+rows)
    solver.batch_index = np.arange(solver.batch, dtype=np.int64)[:, None]
    solver._forest_batch = solver.batch_index.copy()
    if second:
        current = solver.secondary_forest_map
        host = current._host
        for name in ('transform', 'compression', 'dual_lift', 'original_at'):
            setattr(current, '_'+name, fixture._CSR(getattr(host, name), mock_cp))
        current._transform_t = fixture._CSR(host.transform.T.tocsr(), mock_cp)
        for name in ('objective', 'kept_rows', 'eliminated_rows', 'weights',
                     'lower_witness', 'upper_witness', 'fallback_witness'):
            setattr(current, '_'+name, np.array(getattr(host, name), copy=True))
        current._batch_rows = np.arange(solver.batch, dtype=np.int64)[:, None]
        current.device, current.stream = 0, solver.factor.stream
        current._context = solver.factor._context
    if condensed:
        forms = [constraint_form(p) for p in solver.problems]
        matrix = block_diag([f[2][:solver.q] for f in forms], format='csr')
        solver._condensed_h = fixture._CSR(matrix, mock_cp)
        solver._condensed_ht = fixture._CSR(matrix.T.tocsr(), mock_cp)
    solver.cp = np  # Explicit reference backend, not a CuPy success fallback.
    return solver


def _assert_legacy_equal(solver, state, staged):
    expected_sparse, expected_arrays = _payloads(solver, state, _direct_kkt_payload=True)
    actual = {path: source for path, _, source in staged}
    static = {path: _target(solver, path) for path in expected_arrays if path not in actual}
    for path, matrix in expected_sparse.items():
        np.testing.assert_array_equal(actual.get(path+('data',), _target(solver, path).data), matrix.data)
    for path, value in expected_arrays.items():
        np.testing.assert_array_equal(actual.get(path, static.get(path)), value)
    if state.secondary_forest_reductions:
        face = tuple(p.reduced for p in state.face_plans)
        np.testing.assert_array_equal(actual[('secondary_forest_map', '_original_at', 'data')],
            block_diag([p[0] for p in face], format='csr').T.tocsr().data)
        np.testing.assert_array_equal(actual[('secondary_forest_map', '_objective')],
                                     np.stack([p[4] for p in face]))
        for i, name in enumerate(('lower_witness', 'upper_witness', 'fallback_witness')):
            np.testing.assert_array_equal(actual[('secondary_forest_map', '_'+name)], _witnesses(state)[1][i])


@pytest.mark.parametrize('second', [False, True])
@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('condensed', [False, True])
def test_all_numeric_payloads_match_host_reference_without_target_mutation(second, exact, condensed):
    solver = _solver(second=second, exact=exact, condensed=condensed)
    old, new = prepare_host_rebind(solver, fixture._problems(1))
    binder = DeviceIPMPayloadBinder(solver, _states(old))
    before = {path: target.copy() for path, target in binder.numeric_targets}
    inputs = _batches(new)
    input_before = {k: tuple(v.copy() for v in b.arrays()) for k, b in inputs.items()}
    solver.regularization = 2e-6  # Numeric KKT diagonal uses current scalar.
    staged = binder.stage(inputs, *_witnesses(new))
    _assert_legacy_equal(solver, new, staged)
    assert len({p for p, _, _ in staged}) == len(staged) == len(binder.numeric_targets)
    for path, target, source in staged:
        np.testing.assert_array_equal(target, before[path])
        assert source is not target and not np.shares_memory(source, target)
        for b in inputs.values():
            assert not any(np.shares_memory(source, a) for a in b.arrays())
    for k, b in inputs.items():
        for a, previous in zip(b.arrays(), input_before[k]):
            np.testing.assert_array_equal(a, previous)
    assert solver.factor.factored and solver.factor.analysis_count == 1
    for path, target, snapshot in binder.static_targets:
        np.testing.assert_array_equal(target, snapshot)
        assert not np.shares_memory(target, snapshot)
    assert any(p[:2] == ('factor', 'indices') for p, _, _ in binder.static_targets)
    assert any(p[0] == '__binder__' for p, _, _ in binder.static_targets)


@pytest.mark.parametrize('case', ['empty', 'equality_only', 'heterogeneous'])
def test_empty_equality_inequality_and_heterogeneous_patterns(monkeypatch, case):
    def problems(revision=0, batch=2):
        result = []
        for lane in range(batch):
            if case == 'empty':
                a, neq = csr_matrix((0, 4), dtype=np.float64), 0
            elif case == 'equality_only':
                a, neq = csr_matrix([[1., 1., 1., 0.]]), 1
            else:
                a = csr_matrix([[1., 0. if lane else 2., 2. if lane else 0., 1.],
                                [0., 1., 1., 0.]])
                neq = 0
            result.append((a, np.zeros(a.shape[0]), np.full(4, -2.-revision),
                           np.full(4, 3.+revision), np.array([1., -1., 2., 0.])+revision, neq))
        return result
    monkeypatch.setattr(fixture, '_problems', problems)
    solver = _solver(exact=False, condensed=True)
    old, new = prepare_host_rebind(solver, problems(1))
    binder = DeviceIPMPayloadBinder(solver, _states(old))
    _assert_legacy_equal(solver, new, binder.stage(_batches(new), *_witnesses(new)))


def test_stage_uses_no_host_payload_builder_or_numeric_download(monkeypatch):
    solver = _solver(second=True, condensed=True)
    old, new = prepare_host_rebind(solver, fixture._problems(1))
    binder = DeviceIPMPayloadBinder(solver, _states(old))
    import src.gpu_ipm_device_payload as module
    def forbidden(*a, **kw): raise AssertionError('Host construction in stage')
    for name in ('_payloads', 'constraint_form', 'block_diag', 'uniform_kkt_pattern', 'bmat'):
        monkeypatch.setattr(module, name, forbidden)
    binder.stage(_batches(new), *_witnesses(new))


@pytest.mark.parametrize('field,kind', [(f, k) for f in ('data', 'rhs', 'lower', 'upper', 'c')
                                     for k in ('shape', 'dtype', 'noncontiguous')])
def test_invalid_input_shape_dtype_or_layout_rejects_without_mutation(field, kind):
    solver = _solver()
    state = _existing_host_state(solver)
    binder = DeviceIPMPayloadBinder(solver, _states(state))
    batches = _batches(state)
    value = getattr(batches['core'], field)
    if kind == 'shape': value = value.reshape(-1, 1, 1)
    elif kind == 'dtype': value = value.astype(np.float32)
    else:
        wide = np.empty(value.shape[:-1]+(value.shape[-1]*2,))
        value = wide[..., ::2]
    batches['core'] = replace(batches['core'], **{field: value})
    before = [a.copy() for _, a in binder.numeric_targets]
    with pytest.raises(ValueError): binder.stage(batches, *_witnesses(state))
    for (_, target), expected in zip(binder.numeric_targets, before):
        np.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize('change', ['replace', 'shape', 'dtype', 'map', 'map_alias', 'cache', 'witness', 'batch'])
def test_inventory_and_identity_replacements_fail_closed(change):
    solver = _solver()
    state = _existing_host_state(solver)
    binder = DeviceIPMPayloadBinder(solver, _states(state))
    batches, witnesses = _batches(state), _witnesses(state)
    if change == 'replace': solver.c = solver.c.copy()
    elif change == 'shape': solver.c.shape = (solver.c.size,)
    elif change == 'dtype': solver.c.dtype = np.int64
    elif change == 'map': binder._maps['fixed'] = binder._maps['fixed'].copy()
    elif change == 'map_alias': binder._il = binder._il.copy()
    elif change == 'cache': solver._condensed_h = SimpleNamespace()
    elif change == 'witness': witnesses = (tuple(a.astype(np.int32) for a in witnesses[0]), None)
    else: batches.pop('face')
    with pytest.raises(ValueError): binder.stage(batches, *witnesses)


@pytest.mark.parametrize('change', ['csr_data', 'csr_indices', 'proof', 'native', 'host', 'extra_kkt'])
def test_constructor_rejects_stale_initial_device_or_host_state(change):
    solver = _solver()
    state = _existing_host_state(solver)
    hosts = _states(state)
    if change == 'csr_data': solver.g.data[0] += 1.
    elif change == 'csr_indices': solver.g.indices[0] += 1
    elif change == 'proof': solver._forest_weights[0, 0] += 1.
    elif change == 'native': solver.factor.indices[0] += 1
    elif change == 'host': hosts['face'] = hosts['full']
    else:
        pattern = solver.factor.host_pattern.tolil()
        pattern[0, 1] = pattern[1, 0] = 1.
        solver.factor.host_pattern = pattern.tocsr()
    with pytest.raises(ValueError): DeviceIPMPayloadBinder(solver, hosts)


def test_changed_current_values_are_not_bound_to_initial_host_snapshot():
    solver = _solver(second=True, condensed=True)
    old, new = prepare_host_rebind(solver, fixture._problems(1))
    binder = DeviceIPMPayloadBinder(solver, _states(old))
    for path, target, source in binder.stage(_batches(new), *_witnesses(new)):
        np.copyto(target, source)
    # Staging another current batch never reads the old host values back.
    _, later = prepare_host_rebind(solver, fixture._problems(2))
    _assert_legacy_equal(solver, later, binder.stage(_batches(later), *_witnesses(later)))


def test_wrong_device_is_not_silently_copied_or_cast():
    solver = _solver()
    state = _existing_host_state(solver)
    binder = DeviceIPMPayloadBinder(solver, _states(state))
    binder.cp = SimpleNamespace(ndarray=np.ndarray)
    binder.device = 1
    value = np.zeros((2, 3), dtype=np.float64).view(fixture._Array)  # device 0
    with pytest.raises(ValueError, match='device'):
        binder._array(value, (2, 3), np.float64, 'wrong device')


@pytest.mark.parametrize('kind', ['duplicate', 'unsorted', 'indptr', 'nonfinite', 'float_indices'])
def test_host_raw_csr_validation_does_not_trust_cached_flags(kind):
    solver = _solver()
    state = _existing_host_state(solver)
    hosts = _states(state)
    p = hosts['full'][0]
    a = p[0].copy()
    assert a.has_canonical_format
    if kind == 'duplicate': a.indices[1] = a.indices[0]
    elif kind == 'unsorted': a.indices[:2] = a.indices[:2][::-1]
    elif kind == 'indptr': a.indptr[-1] -= 1
    elif kind == 'nonfinite': a.data[0] = np.inf
    else: a.indices = a.indices.astype(np.float64)
    hosts['full'] = ((a, *p[1:]), *hosts['full'][1:])
    with pytest.raises(ValueError): DeviceIPMPayloadBinder(solver, hosts)
