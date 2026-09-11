"""Causal GPU-state proposal transfer; original acceptance is never reused."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_forest_ipm import ForestGpuBatchedIPM
from src.gpu_ipm_warm_state import coordinate_signature, _sequence
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem(cap=1., row_cap=2.):
    return (csr_matrix(np.eye(2)), np.array([row_cap, 1.]), np.zeros(2),
            np.array([cap, 2.]), np.array([-1., 0.]), 0)


def fake_signature_solver(p):
    return SimpleNamespace(problems=[p], batch=1, n=2, m=2, ne=0, ng=6, neq=0)


def test_cpu_signature_allows_numeric_change_but_rejects_topology():
    base = coordinate_signature(fake_signature_solver(problem()))
    assert coordinate_signature(fake_signature_solver(problem(.9, 2.1))) == base
    assert coordinate_signature(fake_signature_solver(problem(0.))) != base
    p = list(problem()); p[0] = csr_matrix([[1., .2], [0., 1.]])
    assert coordinate_signature(fake_signature_solver(tuple(p))) != base


@pytest.mark.parametrize('ids,stage,step', [([1, 1], 'maxmin', 1),
    ([1], 'maxmin', 1), ([True, 2], 'maxmin', 1), ([1, 2], 'x', 1), ([1, 2], 'maxmin', -1)])
def test_cpu_sequence_rejects_ambiguous_identity(ids, stage, step):
    with pytest.raises(ValueError): _sequence(ids, stage, step, 2)


def exported(cp):
    solver = ForestGpuBatchedIPM([problem()], allow_box_dual=True, globalized=True,
        ipm_initialization='balanced', retain_internal_state=True)
    result = solver.solve(initial_x=cp.asarray([[1., .5]], dtype=cp.float64), iterations=0)
    assert result['accepted'].all() and result['internal_state_retained']
    state = solver.export_internal_state(environment_ids=[7], stage='maxmin', step=2)
    return solver, state


def test_cuda_warm_state_reuses_interior_values_and_owns_independent_arrays():
    import cupy as cp
    source, state = exported(cp)
    before = [v.copy() for v in state._arrays]
    source.close()
    with ForestGpuBatchedIPM([problem(row_cap=2.1)], allow_box_dual=True,
            globalized=True, retain_internal_state=True) as target:
        bound = state.bind(target, environment_ids=[7], stage='maxmin', step=3)
        result = target.solve(internal_warm_start=bound, iterations=0)
        assert result['accepted'].all() and result['factor_count'] == 0
        assert result['internal_warm_start']['source_step'] == 2
        assert not result['internal_warm_start']['current_CPU_solution_used']
        assert not result['internal_warm_start']['native_interior_dual_certification_implied']
        assert result['internal_warm_start']['source_original_certificate_basis']==('analytic_box_bound',)
        for got, expected in zip(target._last_internal_state, before):
            np.testing.assert_array_equal(got.get(), expected.get())
            if got.size:
                assert got.data.ptr != expected.data.ptr
        assert paired_certificate(problem(row_cap=2.1), result['x'][0].get(), result['y'][0].get())['certificate_passed']
        bound._arrays[0][0, 0] = 99.
    for got, expected in zip(state._arrays, before):
        np.testing.assert_array_equal(got.get(), expected.get())


def test_cuda_changed_current_bound_cannot_inherit_previous_acceptance():
    import cupy as cp
    source, state = exported(cp)
    source.close()
    with ForestGpuBatchedIPM([problem(.9)], allow_box_dual=True,
            globalized=True, retain_internal_state=True) as target:
        bound = state.bind(target, environment_ids=[7], stage='maxmin', step=3)
        result = target.solve(internal_warm_start=bound, iterations=0)
        assert not result['accepted'].any()
        assert not result['internal_state_retained']
        with pytest.raises(ValueError, match='No retained'):
            target.export_internal_state(environment_ids=[7], stage='maxmin', step=3)


def test_cuda_environment_stage_time_and_target_are_bound():
    import cupy as cp
    source, state = exported(cp)
    with source, ForestGpuBatchedIPM([problem()], allow_box_dual=True, globalized=True) as target:
        for ids, stage, step in (([8], 'maxmin', 3), ([7], 'exchange', 3),
                                  ([7], 'maxmin', 2), ([7], 'maxmin', 4)):
            with pytest.raises(ValueError, match='causal'):
                state.bind(target, environment_ids=ids, stage=stage, step=step)
        bound = state.bind(target, environment_ids=[7], stage='maxmin', step=3)
        with pytest.raises(ValueError, match='another target'):
            bound.initialize(source)
        bound._arrays[2][0, 0] = cp.nan
        with pytest.raises(ValueError, match='Finite'):
            target.solve(internal_warm_start=bound, iterations=0)


def test_cuda_topology_change_rejects_and_floor_is_only_an_internal_proposal():
    import cupy as cp
    source, state = exported(cp)
    with source, ForestGpuBatchedIPM([problem(0.)], allow_box_dual=True, globalized=True) as target:
        with pytest.raises(ValueError, match='coordinate'):
            state.bind(target, environment_ids=[7], stage='maxmin', step=3)
    with ForestGpuBatchedIPM([problem()], allow_box_dual=True, globalized=True) as target:
        bound = state.bind(target, environment_ids=[7], stage='maxmin', step=3, interior_floor=1e-4)
        assert all(bool(cp.all(v >= 1e-4)) for v in bound._arrays[2:])
        np.testing.assert_array_equal(target.full_problems[0][3], problem()[3])
        with pytest.raises(ValueError, match='floor'):
            state.bind(target, environment_ids=[7], stage='maxmin', step=3, interior_floor=1.)


def test_cuda_default_does_not_retain_and_arbitrary_state_is_rejected():
    with ForestGpuBatchedIPM([problem()], globalized=True) as solver:
        result = solver.solve(iterations=0)
        assert not result['internal_state_retained']
        with pytest.raises(ValueError, match='No retained'):
            solver.export_internal_state(environment_ids=[7], stage='maxmin', step=2)
        with pytest.raises(ValueError, match='causally bound'):
            solver.solve(internal_warm_start={}, iterations=0)


def test_cuda_slack_repair_uses_current_problem_without_repairing_infeasible_x():
    import cupy as cp
    source, state = exported(cp)
    source.close()
    with ForestGpuBatchedIPM([problem(.9, 2.1)], allow_box_dual=True, globalized=True) as target:
        bound = state.bind(target, environment_ids=[7], stage='maxmin', step=3, repair_slacks=True)
        expected = cp.maximum(target.h-target._mv(target.g, state._arrays[0], target.ng), 1e-8)
        np.testing.assert_array_equal(bound._arrays[3].get(), expected.get())
        np.testing.assert_array_equal(bound._arrays[0].get(), state._arrays[0].get())
        assert not target.solve(internal_warm_start=bound, iterations=0)['accepted'].any()
        with pytest.raises(ValueError, match='bool'):
            state.bind(target, environment_ids=[7], stage='maxmin', step=3, repair_slacks=1)


@pytest.mark.parametrize('identical', [False, True])
def test_cuda_numeric_rebind_invalidates_already_bound_warm_proposal(identical):
    import cupy as cp
    from src.gpu_ipm_numeric_update import rebind_forest_ipm
    source, state = exported(cp)
    source.close()
    with ForestGpuBatchedIPM([problem()], allow_box_dual=True, globalized=True) as target:
        bound = state.bind(target, environment_ids=[7], stage='maxmin', step=3)
        rebind_forest_ipm(target, [problem(row_cap=2. if identical else 2.1)])
        with pytest.raises(ValueError, match='Target LP changed'):
            target.solve(internal_warm_start=bound, iterations=0)
        rebound = state.bind(target, environment_ids=[7], stage='maxmin', step=3)
        assert target.solve(internal_warm_start=rebound, iterations=0)['accepted'].all()


def test_cuda_current_bound_restart_is_only_an_initial_proposal():
    import cupy as cp
    source, state = exported(cp)
    old = tuple(v.copy() for v in state._arrays)
    source.close()
    with ForestGpuBatchedIPM([problem(.9, 2.1)], allow_box_dual=True, globalized=True) as target:
        bound = state.bind(target, environment_ids=[7], stage='maxmin', step=3, restart_mu=1e-4)
        np.testing.assert_array_equal(bound._arrays[0].get(), state._arrays[0].get())
        assert bound.metadata['bound_dual_restart'] is not None
        assert all(bool(cp.all(v > 0.)) for v in bound._arrays[2:])
        assert not target.solve(internal_warm_start=bound, iterations=0)['accepted'].any()
        for options in ({'interior_floor':1e-4}, {'repair_slacks':True}):
            with pytest.raises(ValueError, match='separate proposal'):
                state.bind(target, environment_ids=[7], stage='maxmin', step=3, restart_mu=1e-4, **options)
    for before, after in zip(old, state._arrays):
        np.testing.assert_array_equal(before.get(), after.get())
