"""Speed options must preserve original-Newton and original-LP rejection gates.

CPU certificate tests execute the production arithmetic through an explicitly
test-only NumPy/SciPy adapter; no CUDA runtime or LP optimizer is initialized.
CUDA tests are separately named and belong to the exclusive GPU test owner.
"""

import ast
import inspect
import sys
import textwrap
from types import ModuleType

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_batched_ipm import GpuBatchedIPM
from src.gpu_block_lp import assemble_blocks, certify_blocks_device
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from src.gpu_globalized_ipm import solve_globalized_ipm
from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from src.lp_bounded_near_equality import BoundedNearEqualityReduction, NearEqualityCandidate
from src.lp_trace import problem_hash
from src.lp_zero_face import ZeroFaceReduction


@pytest.fixture
def numpy_certificate(monkeypatch):
    """No backend fallback: this adapter exists only inside these CPU tests."""
    cp = ModuleType('cupy')
    cp.__getattr__ = lambda name: getattr(np, name)
    cp.asnumpy = np.asarray
    root, scipy_module, sparse_module = map(ModuleType,
        ['cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse'])
    root.scipy = scipy_module
    scipy_module.sparse = sparse_module
    sparse_module.csr_matrix = csr_matrix
    for name, value in [('cupy', cp), ('cupyx', root), ('cupyx.scipy', scipy_module),
                        ('cupyx.scipy.sparse', sparse_module)]:
        monkeypatch.setitem(sys.modules, name, value)
    return cp


def _certify(problems, x, y, cp, **kwargs):
    return certify_blocks_device(problems, assemble_blocks(problems),
        cp.asarray(x, dtype=cp.float64).ravel(), cp.asarray(y, dtype=cp.float64).ravel(),
        cp=cp, **kwargs)


def _large_multiplier_problem():
    # Tiny equality residual is multiplied by y=1e8, while c-A.T*y=0.
    return (csr_matrix([[1e-8, 0.]]), np.zeros(1), np.full(2, -2.),
            np.full(2, 2.), np.array([1., 0.]), 1)


def _near_problem(cost=False, defect_upper=2.):
    # The original zero equality disappears BEFORE the candidate is applied.
    # Row 2 is approximately four times row 1, not a duplicate sign row.
    return (csr_matrix([[0., 0., 0.], [1., 1., 0.], [4., 4., 2.**-50], [0., 0., 1.]]),
            np.array([0., 1., 4., 1.5]), np.full(3, -2.), np.array([2., 2., defect_upper]),
            np.array([1., 1., 0.]) if cost else np.zeros(3), 3)


def _near_candidate(problem):
    face = ZeroFaceReduction(problem)
    # Derive positions from an independently constructed face map: do not
    # silently reuse original row numbers after zero-face preprocessing.
    target = int(np.flatnonzero(face.rows == 2)[0])
    support = int(np.flatnonzero(face.rows == 1)[0])
    return NearEqualityCandidate(target, (support,), (4,))


@pytest.mark.parametrize('sign', [-1., 1.])
def test_direct_gap_rejects_tiny_equality_residual_times_huge_dual(numpy_certificate, sign):
    p = _large_multiplier_problem()
    x, y = [[sign, 0.]], [[1e8]]
    old = _certify([p], x, y, numpy_certificate)[0]
    strict = _certify([p], x, y, numpy_certificate, require_direct_dual=True)[0]
    assert old['certificate_passed'] and old['relative_kkt_gap'] == 0.
    assert old['primal_residual'] == pytest.approx(1e-8)
    assert 'relative_direct_gap' not in old  # default remains unchanged
    assert strict['relative_direct_gap'] == pytest.approx(1.)
    assert strict['equality_dual_residual_contribution'] == pytest.approx(sign)
    assert not strict['direct_dual_gate_passed'] and not strict['certificate_passed']


def test_direct_gate_is_per_lane_and_does_not_replace_primal_check(numpy_certificate):
    p = _large_multiplier_problem()
    result = _certify([p, p, p], [[1., 0.], [0., 0.], [0., 3.]], [[1e8]]*3,
                      numpy_certificate, require_direct_dual=True)
    assert [row['certificate_passed'] for row in result] == [False, True, False]
    assert result[2]['direct_dual_gate_passed']  # original bound violation still rejects
    assert result[2]['primal_residual'] == 1.


def test_direct_gate_uses_actual_analytic_box_dual_when_selected(numpy_certificate):
    p = (csr_matrix([[1., 1.]]), np.array([2.]), np.zeros(2), np.ones(2), -np.ones(2), 1)
    result = _certify([p], [[1., 1.]], [[-100.]], numpy_certificate,
                      allow_box_dual=True, require_direct_dual=True)[0]
    assert result['certificate_source'] == 'analytic_box_bound'
    assert result['certificate_passed'] and result['direct_dual_gate_passed']
    assert result['direct_dual_objective'] == -2.
    assert result['relative_direct_gap'] == 0.
    assert result['solver_dual_relative_kkt_gap'] > 1.


def test_near_working_certificate_cannot_hide_original_omitted_row(numpy_certificate):
    p = _near_problem()
    face = ZeroFaceReduction(p)
    assert face.rows.tolist() == [1, 2, 3]
    plan = BoundedNearEqualityReduction(face.reduced, _near_candidate(p))
    x = np.array([.5 + 4e-6, .5, 0.])
    y = np.zeros(len(plan.rows))
    working = _certify([plan.reduced], [x], [y], numpy_certificate, require_direct_dual=True)[0]
    full_x, full_y = face.lift(plan.lift_primal(x), plan.lift_dual(y))
    original = _certify([p], [full_x], [full_y], numpy_certificate, require_direct_dual=True)[0]
    assert working['certificate_passed'] and working['primal_residual'] < 1e-5
    assert not original['certificate_passed'] and original['primal_residual'] > 1e-5
    assert original['direct_dual_gate_passed']  # full-row primal gate is independently necessary
    assert float(plan.defect_bound) < 1e-12


def test_near_dual_zero_lift_and_caller_copies_survive_row_remapping():
    p = _near_problem(cost=True)
    before = problem_hash(p)
    face = ZeroFaceReduction(p)
    plan = BoundedNearEqualityReduction(face.reduced, _near_candidate(p))
    full_initial_y = np.array([7., 1., 13., 0.])
    saved = full_initial_y.copy()
    compressed = plan.compress_dual(full_initial_y[face.rows])
    near_y = plan.lift_dual(compressed)
    x = np.array([.5, .5, 0.])
    primal_copy = plan.lift_primal(x)
    full_x, full_y = face.lift(primal_copy, near_y)
    np.testing.assert_array_equal(full_initial_y, saved)
    np.testing.assert_array_equal(full_y, [0., 1., 0., 0.])
    np.testing.assert_array_equal(full_x, x)
    assert not np.shares_memory(primal_copy, x)
    assert not np.shares_memory(compressed, full_initial_y)
    assert not np.shares_memory(near_y, compressed)
    assert problem_hash(p) == before
    assert not plan.summary['exact_feasible_set_equivalence_claimed']
    assert plan.summary['requires_original_full_row_certificate']
    assert plan.summary['requires_direct_dual_objective_and_equality_residual_check']


def test_near_none_is_default_and_does_not_build_a_relaxation(monkeypatch):
    received = []
    monkeypatch.setattr(GpuCondensedBatchedIPM, '__init__',
                        lambda self, problems, **kwargs: received.append(problems))
    monkeypatch.setattr(ZeroFaceGpuBatchedIPM, '_setup_postsolve', lambda *args: None)
    p = _near_problem()
    implicit = ZeroFaceGpuBatchedIPM([p])
    explicit = ZeroFaceGpuBatchedIPM([p], bounded_near_equality=None)
    assert implicit.near_equality_plans == explicit.near_equality_plans == []
    assert problem_hash(received[0][0]) == problem_hash(received[1][0])
    assert received[0][0][0].shape[0] == 3


def test_changed_current_box_rejects_near_proposal_before_gpu_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid current-bound proof must reject before GPU setup')

    monkeypatch.setattr(GpuCondensedBatchedIPM, '__init__', forbidden)
    with pytest.raises(ValueError, match='exceeds'):
        ZeroFaceGpuBatchedIPM([_near_problem(defect_upper=1e6)],
                             bounded_near_equality=_near_candidate(_near_problem()))


@pytest.mark.parametrize('value', [-1, 3, .5, True, None])
def test_invalid_factor_refinement_budget_is_rejected_before_numeric_setup(numpy_certificate, value):
    with pytest.raises(ValueError, match='refinement budget'):
        GpuBatchedIPM([], factor_refinements=value)


def _tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def test_defaults_factor_forwarding_and_globalized_metadata_are_explicit():
    assert inspect.signature(GpuBatchedIPM.__init__).parameters['factor_refinements'].default == 2
    assert inspect.signature(ZeroFaceGpuBatchedIPM.__init__).parameters['bounded_near_equality'].default is None
    calls = [node for node in ast.walk(_tree(GpuBatchedIPM.__init__)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'UniformCudssFactor']
    assert len(calls) == 1
    forwarded = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(forwarded['refinement_steps'], ast.Name)
    assert forwarded['refinement_steps'].id == 'factor_refinements'
    keys = [node for node in ast.walk(_tree(solve_globalized_ipm)) if isinstance(node, ast.keyword)
            and node.arg == 'factor_refinements']
    assert len(keys) == 1
    assert ast.literal_eval(keys[0].value.args[-1]) == 2


@pytest.mark.parametrize('function', [ZeroFaceGpuBatchedIPM.certificate,
    ForestGpuBatchedIPM.certificate, ForestGpuBatchedIPM._materialize_box_certificate])
def test_all_original_postsolve_certificates_enable_additional_gate_for_near_mode(function):
    calls = [node for node in ast.walk(_tree(function)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'certify_blocks_device']
    assert calls
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        gate = keywords['require_direct_dual']
        assert isinstance(gate, ast.Call) and isinstance(gate.func, ast.Name) and gate.func.id == 'bool'
        condition=gate.args[0]
        assert isinstance(condition,ast.BoolOp) and isinstance(condition.op,ast.Or)
        assert {value.attr for value in condition.values if isinstance(value,ast.Attribute)} == {
            'near_equality_plans','second_forest','fix_singleton_equalities'}


def test_cuda_direct_gap_rejects_huge_equality_multiplier():
    import cupy as cp
    p = _large_multiplier_problem()
    old = _certify([p], [[1., 0.]], [[1e8]], cp)[0]
    strict = _certify([p], [[1., 0.]], [[1e8]], cp, require_direct_dual=True)[0]
    assert old['certificate_passed'] and old['relative_kkt_gap'] == 0.
    assert not strict['certificate_passed'] and strict['relative_direct_gap'] == pytest.approx(1.)


@pytest.mark.parametrize('solver_type', [ZeroFaceGpuBatchedIPM, ForestGpuBatchedIPM])
def test_cuda_near_warm_start_returns_zero_lifted_actual_dual_without_mutation(solver_type):
    import cupy as cp
    p = _near_problem(cost=True)
    x = cp.asarray([[.5, .5, 0.]], dtype=cp.float64)
    y = cp.asarray([[7., 1., 13., 0.]], dtype=cp.float64)
    before_x, before_y, before_hash = x.copy(), y.copy(), problem_hash(p)
    with solver_type([p], bounded_near_equality=_near_candidate(p),
                     exact_equalities=True, factor_refinements=0, globalized=True) as solver:
        result = solver.solve(initial_x=x, initial_y=y, iterations=0)
        assert result['accepted'].all(), result['metrics']
        assert result['factor_count'] == 0 and result['cpu_lp_calls'] == 0
        assert result['bounded_near_equality']['enabled']
        assert result['factor_refinements'] == 0
        np.testing.assert_array_equal(result['y'].get(), [[0., 1., 0., 0.]])
        np.testing.assert_array_equal(x.get(), before_x.get())
        np.testing.assert_array_equal(y.get(), before_y.get())
        assert _certify([p], result['x'], result['y'], cp, require_direct_dual=True)[0]['certificate_passed']
        assert problem_hash(p) == before_hash


@pytest.mark.parametrize('solver_type', [ZeroFaceGpuBatchedIPM, ForestGpuBatchedIPM])
def test_cuda_near_reduced_feasible_pair_is_rejected_by_full_original_rows(solver_type):
    import cupy as cp
    p = _near_problem()
    with solver_type([p], bounded_near_equality=_near_candidate(p), factor_refinements=0) as solver:
        x = cp.asarray([[.5 + 4e-6, .5, 0.]], dtype=cp.float64)
        y = cp.zeros((1, solver.m), dtype=cp.float64)
        working = certify_blocks_device(solver.problems, solver.assembled, x.ravel(), y.ravel(),
                                        cp=cp, require_direct_dual=True)[0]
        assert working['certificate_passed']
        original = solver.certificate(x, y)[0]
        assert not original['certificate_passed'] and original['primal_residual'] > 1e-5


@pytest.mark.parametrize('refinements', [0, 1, 2])
def test_cuda_factor_refinements_do_not_change_original_lp_or_newton_target(refinements):
    import cupy as cp
    p = (csr_matrix([[1., 1.]]), np.array([1.]), np.zeros(2), np.full(2, 2.), np.array([1., 2.]), 1)
    before = problem_hash(p)
    with GpuCondensedBatchedIPM([p], globalized=True, factor_refinements=refinements,
                              newton_krylov_iterations=8, krylov_coordinates='condensed') as solver:
        assert solver.original_newton_target and solver.factor_refinements == refinements
        result = solver.solve(iterations=100)
        assert result['accepted'].all(), result['metrics']
        assert result['factor_count'] > 0 and result['cpu_lp_calls'] == 0
        assert result['factor_refinements'] == refinements
        assert result['original_newton_target']
        assert _certify([p], result['x'], result['y'], cp, require_direct_dual=True)[0]['certificate_passed']
        assert problem_hash(p) == before
