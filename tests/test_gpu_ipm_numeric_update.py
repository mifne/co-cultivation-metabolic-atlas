"""CPU transactional mocks and explicitly selected CUDA numeric rebind tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import bmat, csr_matrix, diags

from src.gpu_batched_ipm import constraint_form, uniform_kkt_pattern
from src.gpu_forest_ipm import ForestGpuBatchedIPM, prepare_forest_batch
from src.gpu_ipm_numeric_update import (
    NumericRebindRejected, _existing_host_state, _payloads, _target,
    prepare_host_rebind, rebind_forest_ipm,
)
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_exact_equalities import ExactEqualityReduction
from src.lp_trace import problem_hash
from src.lp_zero_face import ZeroFaceReduction


def _problems(revision=0, batch=2):
    matrix = np.array([
        [1., -1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 1., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 1., 2., 0.],
        [0., 0., 2., 0., 0., 2., 4., 0.],
        [1., 0., 0., 0., 0., 1.+.5*revision, 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., -1.],
    ])
    result = []
    for lane in range(batch):
        lo = np.array([0., 0., 0., 0., 0., -2., -2., -2.])
        hi = np.array([4.+2*revision, 5., 4., 4., 4., 2., 2., 2.])
        rhs = np.array([0., 0., 0., 0., 3.-.2*revision+.01*lane, 2.-.1*revision])
        cost = np.array([-1., -.1, .3, .4, .5, .2+.1*revision, -.3, .4])
        result.append((csr_matrix(matrix), rhs, lo, hi, cost, 4))
    return result


class _Array(np.ndarray):
    @property
    def device(self): return SimpleNamespace(id=0)


class _CP:
    ndarray = _Array
    def __init__(self):
        self.copy_count = 0
        self.fail_copy_at = None
        self.fail_staging = False
    def asarray(self, value, dtype=None):
        if self.fail_staging: raise MemoryError('Deliberate staging allocation failure')
        return np.array(value, dtype=dtype, copy=True).view(_Array)
    def asnumpy(self, value): return np.array(value, copy=True)
    def copyto(self, target, source):
        self.copy_count += 1
        if self.copy_count == self.fail_copy_at: raise RuntimeError('Deliberate midcommit copy failure')
        np.copyto(target, source)


class _CSR:
    def __init__(self, matrix, cp):
        self.shape = matrix.shape
        self.data = cp.asarray(matrix.data)
        self.indices = cp.asarray(matrix.indices)
        self.indptr = cp.asarray(matrix.indptr)


def _set_target(solver, path, value):
    parent = solver
    for key in path[:-1]:
        parent = parent[key] if isinstance(key, int) else getattr(parent, key)
    key = path[-1]
    if isinstance(key, int): parent[key] = value
    else: setattr(parent, key, value)


def _mock_solver(*, second_forest=False, exact_equalities=True):
    solver = object.__new__(ForestGpuBatchedIPM)
    solver.full_problems, solver.forest_plans, solver.forest_reductions = prepare_forest_batch(_problems())
    solver.forest_problems = tuple(r.problem for r in solver.forest_reductions)
    solver.face_plans = [ZeroFaceReduction(p) for p in solver.forest_problems]
    reduced = [p.reduced for p in solver.face_plans]
    solver.secondary_forest_plans = []
    solver.secondary_forest_reductions = []
    if second_forest:
        solver.secondary_forest_plans = [HomogeneousEqualityReduction.from_problem(p) for p in reduced]
        solver.secondary_forest_reductions = [plan.reduce(p) for plan, p in zip(solver.secondary_forest_plans, reduced)]
        reduced = [r.problem for r in solver.secondary_forest_reductions]
    solver.equality_plans = [ExactEqualityReduction(p) for p in reduced] if exact_equalities else []
    solver.problems = [p.reduced for p in solver.equality_plans] if exact_equalities else reduced
    solver.original_problems = [p.original for p in solver.face_plans]
    solver.full_m, solver.full_n = solver.full_problems[0][0].shape
    solver.full_neq = solver.full_problems[0][-1]
    solver.original_m, solver.original_n = solver.original_problems[0][0].shape
    solver.m, solver.n = solver.problems[0][0].shape
    solver.neq = solver.problems[0][-1]
    solver.batch = len(solver.problems)
    forms = [constraint_form(p) for p in solver.problems]
    solver.ne, solver.ng = forms[0][0].shape[0], forms[0][2].shape[0]
    solver.q = solver.m-solver.neq
    solver.size = solver.n+solver.ne+solver.ng
    solver.regularization = 1e-6
    solver.globalized = True
    solver.equality_row_scaling = False
    solver.near_equality_plans = []
    solver.second_forest = second_forest
    solver.exact_equalities = exact_equalities
    solver.closed = False
    solver.cp = _CP()
    solver._last_internal_state = object()
    solver.failure_snapshot = object()
    solver.problem_hashes = tuple(problem_hash(p) for p in solver.full_problems)
    matrices = [bmat([[diags(np.ones(solver.n)), f[0].T, f[2][:solver.q].T],
        [f[0], diags(-np.ones(solver.ne)), None],
        [f[2][:solver.q], None, diags(-np.ones(solver.q))]], format='csr') for f in forms]
    pattern, _, _ = uniform_kkt_pattern(matrices)
    solver.factor = SimpleNamespace(host_pattern=pattern, device=0, analysis_count=1,
        factored=True, failed=False, stream=SimpleNamespace(synchronize=lambda: None))
    def context():
        if solver.factor.failed or solver.closed: raise RuntimeError('Workspace failed or closed')
    solver.factor._context = context
    solver.assembled = [None]*6
    solver.original_assembled = [None]*6
    solver._full_assembled = [None]*6
    solver.proof_arrays = [None]*5
    sparse, arrays = _payloads(solver, _existing_host_state(solver))
    for path, matrix in sparse.items(): _set_target(solver, path, _CSR(matrix, solver.cp))
    for path, value in arrays.items(): _set_target(solver, path, solver.cp.asarray(value))
    if second_forest:
        from src.gpu_forest_map import _prepare_host_maps
        host = _prepare_host_maps(solver.secondary_forest_plans, solver.secondary_forest_reductions)
        solver.secondary_forest_map = SimpleNamespace(_host=host, **{name: getattr(host, name)
            for name in ('batch', 'original_n', 'original_m', 'reduced_n', 'reduced_m')})
    return solver


def _device_snapshot(solver):
    sparse, arrays = _payloads(solver, _existing_host_state(solver))
    snapshot = {}
    for path in sparse:
        target = _target(solver, path)
        snapshot[(path, 'data')] = np.array(target.data, copy=True)
    for path in arrays: snapshot[(path, 'array')] = np.array(_target(solver, path), copy=True)
    return snapshot


def _assert_snapshot(solver, snapshot):
    for (path, kind), expected in snapshot.items():
        target = _target(solver, path)
        np.testing.assert_array_equal(target.data if kind == 'data' else target, expected)


@pytest.mark.parametrize('exact', [False, True])
def test_transaction_updates_all_operators_certificates_and_witnesses_without_new_factor(exact):
    solver = _mock_solver(exact_equalities=exact)
    before_factor = solver.factor
    original_state = solver._last_internal_state
    original_witness = solver._forest_upper_witness.copy()
    sparse, arrays = _payloads(solver, _existing_host_state(solver))
    identities = {path: id(_target(solver, path)) for path in sparse.keys() | arrays.keys()}
    result = rebind_forest_ipm(solver, _problems(1))
    assert solver.factor is before_factor and solver.factor.analysis_count == 1
    assert not solver.factor.factored and not solver.factor.failed
    assert solver._last_internal_state is None and original_state is not None
    assert solver.failure_snapshot is None
    assert result['native_factor_object_preserved'] and result['exact_equality_qr_calls'] == 0
    assert result['payload_assembly_and_staging_seconds'] >= 0.
    assert 'device_staging_seconds' not in result
    assert not np.array_equal(solver._forest_upper_witness, original_witness)
    sparse, arrays = _payloads(solver, _existing_host_state(solver))
    for path, expected in sparse.items():
        target = _target(solver, path)
        np.testing.assert_array_equal(target.data, expected.data)
        assert id(target) == identities[path]
    for path, expected in arrays.items():
        np.testing.assert_array_equal(_target(solver, path), expected)
        assert id(_target(solver, path)) == identities[path]
    assert solver.problem_hashes == tuple(problem_hash(p) for p in _problems(1))
    assert all(p.summary['reused_proofs'] for p in solver.equality_plans)


def test_rebind_owned_snapshots_do_not_alias_user_inputs_or_old_snapshots():
    solver = _mock_solver()
    old = solver.full_problems
    old_hashes = solver.problem_hashes
    supplied = _problems(1)
    rebind_forest_ipm(solver, supplied)
    supplied[0][1][-1] += 10.
    assert solver.problem_hashes == tuple(problem_hash(p) for p in solver.full_problems)
    assert old_hashes == tuple(problem_hash(p) for p in old)
    assert all(not p[1].flags.writeable for p in solver.full_problems)


@pytest.mark.parametrize('change', ['equality', 'pattern', 'finite', 'fixed', 'count', 'hash',
                                   'host_map', 'stale_device', 'near', 'scaling', 'nonglobal'])
def test_before_commit_rejection_leaves_old_workspace_intact(change):
    solver = _mock_solver()
    incoming = _problems(1)
    before = _device_snapshot(solver)
    state = solver._last_internal_state
    old_hashes = solver.problem_hashes
    if change == 'equality': incoming[0][0].data[0] += .25
    elif change == 'pattern': incoming[0][0][4, 7] = .1
    elif change == 'finite': incoming[0][3][7] = np.inf
    elif change == 'fixed': incoming[0][3][7] = incoming[0][2][7]
    elif change == 'count': incoming = incoming[:1]
    elif change == 'hash': solver.problem_hashes = ('invalid',)*solver.batch
    elif change == 'host_map': solver.forest_plans[0].original_to_reduced[0] += 1
    elif change == 'stale_device':
        solver._full_assembled[5][0] += 1.
        before = _device_snapshot(solver)
    elif change == 'near': solver.near_equality_plans = [object()]
    elif change == 'scaling': solver.equality_row_scaling = True
    else: solver.globalized = False
    hashes_at_entry = solver.problem_hashes
    with pytest.raises(NumericRebindRejected): rebind_forest_ipm(solver, incoming)
    _assert_snapshot(solver, before)
    assert solver._last_internal_state is state
    assert solver.problem_hashes == hashes_at_entry
    assert solver.factor.factored and not solver.factor.failed


def test_staging_allocation_failure_is_nonmutating():
    solver = _mock_solver()
    before = _device_snapshot(solver)
    state = solver._last_internal_state
    solver.cp.fail_staging = True
    with pytest.raises(MemoryError): rebind_forest_ipm(solver, _problems(1))
    _assert_snapshot(solver, before)
    assert solver._last_internal_state is state
    assert solver.factor.factored and not solver.factor.failed


def test_public_host_prepare_classifies_underlying_plain_value_error(monkeypatch):
    import src.gpu_ipm_numeric_update as update
    solver = _mock_solver()
    original = ValueError('Invalid source LP')
    def reject(*_args, **_kwargs): raise original
    monkeypatch.setattr(update, '_validated_problem', reject)
    with pytest.raises(NumericRebindRejected, match='Invalid source LP') as caught:
        prepare_host_rebind(solver, _problems(1))
    assert caught.value.__cause__ is original
    assert solver.factor.factored and not solver.factor.failed


@pytest.mark.parametrize('phase', ['prepare_host_rebind', '_payloads', '_stage_updates', 'secondary_map'])
def test_plain_precommit_value_error_is_rejection_and_nonmutating(monkeypatch, phase):
    import src.gpu_ipm_numeric_update as update
    import src.gpu_forest_map as maps
    solver = _mock_solver(second_forest=phase == 'secondary_map')
    before = _device_snapshot(solver)
    state, hashes = solver._last_internal_state, solver.problem_hashes
    original = ValueError('Invalid '+phase)
    def reject(*_args, **_kwargs): raise original
    if phase == 'secondary_map': monkeypatch.setattr(maps, 'GpuForestMap', reject)
    else: monkeypatch.setattr(update, phase, reject)
    with pytest.raises(NumericRebindRejected, match='Invalid '+phase) as caught:
        rebind_forest_ipm(solver, _problems(1))
    assert caught.value.__cause__ is original
    _assert_snapshot(solver, before)
    assert solver._last_internal_state is state and solver.problem_hashes == hashes
    assert solver.factor.factored and not solver.factor.failed


def test_precommit_runtime_failure_retains_type_without_mutation(monkeypatch):
    import src.gpu_ipm_numeric_update as update
    solver = _mock_solver()
    before = _device_snapshot(solver)
    state = solver._last_internal_state
    original = RuntimeError('Device staging runtime failure')
    def fail(*_args, **_kwargs): raise original
    monkeypatch.setattr(update, '_stage_updates', fail)
    with pytest.raises(RuntimeError) as caught:
        rebind_forest_ipm(solver, _problems(1))
    assert caught.value is original
    _assert_snapshot(solver, before)
    assert solver._last_internal_state is state
    assert solver.factor.factored and not solver.factor.failed


def test_commit_value_error_retains_type_and_fails_closed(monkeypatch):
    solver = _mock_solver()
    original = ValueError('Unexpected commit failure')
    def fail(*_args, **_kwargs): raise original
    monkeypatch.setattr(solver.cp, 'copyto', fail)
    with pytest.raises(ValueError) as caught:
        rebind_forest_ipm(solver, _problems(1))
    assert caught.value is original and not isinstance(caught.value, NumericRebindRejected)
    assert solver.factor.failed and not solver.factor.factored
    assert solver._last_internal_state is None


def test_partial_commit_failure_invalidates_workspace_instead_of_silent_rollback():
    solver = _mock_solver()
    solver.cp.fail_copy_at = 3
    with pytest.raises(RuntimeError, match='midcommit'): rebind_forest_ipm(solver, _problems(1))
    assert solver.factor.failed and not solver.factor.factored
    assert solver._last_internal_state is None
    with pytest.raises(RuntimeError, match='failed'): solver.factor._context()


def test_secondary_forest_rebind_stages_new_bound_witness_map(monkeypatch):
    import src.gpu_forest_map as maps
    solver = _mock_solver(second_forest=True)
    old_map = solver.secondary_forest_map
    def cpu_map(plans, reductions, cp):
        host = maps._prepare_host_maps(plans, reductions)
        return SimpleNamespace(_host=host, **{name: getattr(host, name)
            for name in ('batch', 'original_n', 'original_m', 'reduced_n', 'reduced_m')})
    monkeypatch.setattr(maps, 'GpuForestMap', cpu_map)
    result = rebind_forest_ipm(solver, _problems(1))
    assert result['secondary_forest_map_rebuilt']
    assert solver.secondary_forest_map is not old_map
    assert solver.secondary_forest_map.original_n == old_map.original_n


def test_optional_cached_condensed_operators_are_updated_too():
    solver = _mock_solver()
    from scipy.sparse import block_diag
    forms = [constraint_form(p) for p in solver.problems]
    h = block_diag([f[2][:solver.q] for f in forms], format='csr')
    solver._condensed_h = _CSR(h, solver.cp)
    solver._condensed_ht = _CSR(h.T.tocsr(), solver.cp)
    before = solver._condensed_h.data.copy()
    rebind_forest_ipm(solver, _problems(1))
    assert not np.array_equal(solver._condensed_h.data, before)


def test_identical_rebind_still_invalidates_factor_and_old_acceptance():
    solver = _mock_solver()
    hashes = solver.problem_hashes
    result = rebind_forest_ipm(solver, _problems())
    assert result['new_problem_hashes'] == hashes
    assert result['unchanged_device_payloads_skipped'] > 0
    assert not solver.factor.factored and solver._last_internal_state is None


@pytest.mark.parametrize('second', [False, True])
def test_cuda_rebound_cold_solve_matches_fresh_current_lp_certificates(second):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1: pytest.skip('CUDA required')
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    options = dict(globalized=True, forcing_eta=.1, regularization=1e-6,
        newton_krylov_iterations=8, predictor_corrector=True,
        predictor_affine_fraction=.995, ipm_initialization='balanced',
        exact_equalities=True, second_forest=second, allow_box_dual=True,
        device_checked_solves=True, factor_refinements=0)
    current = _problems(1)
    with ForestGpuBatchedIPM(_problems(), **options) as solver:
        factor = solver.factor
        addresses = (solver.values.data.ptr, solver.c.data.ptr, solver._full_assembled[0].data.data.ptr)
        update = rebind_forest_ipm(solver, current)
        assert update['native_factor_object_preserved'] and solver.factor is factor
        assert addresses == (solver.values.data.ptr, solver.c.data.ptr, solver._full_assembled[0].data.data.ptr)
        rebound = solver.solve(iterations=120)
        rx, ry = rebound['x'].get(), rebound['y'].get()
        assert all(rebound['accepted']), rebound
        assert all(paired_certificate(p, x, y)['certificate_passed'] for p, x, y in zip(current, rx, ry))
    with ForestGpuBatchedIPM(current, **options) as fresh:
        result = fresh.solve(iterations=120)
        fx, fy = result['x'].get(), result['y'].get()
        assert all(result['accepted']), result
        for p, x, y, oldx in zip(current, fx, fy, rx):
            assert paired_certificate(p, x, y)['certificate_passed']
            np.testing.assert_allclose(p[4]@x, p[4]@oldx, atol=1e-6, rtol=1e-7)
