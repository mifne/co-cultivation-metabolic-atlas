"""Device regression tests for opt-in restricted-LP fixed-cost reductions.

These are intentionally separate from the CPU-only capacity proof checks.
"""
import numpy as np
import pytest

from src.gpu_certified_basis import NormalizedLP, compile_basis
from src.gpu_revised_basis import GpuRevisedBasis
from src.gpu_restricted_basis import GpuRestrictedBasis
from tests.test_gpu_revised_basis import problem


def _base(anchor, variable_rows):
    return GpuRevisedBasis(anchor, variable_rows, max_pivots=16, capture_safe=True,
        compact_updates=True, reuse_small_factor=True)


def _anchor(p):
    return compile_basis(p.a, p.rhs, p.lower, p.upper, p.c, p.neq)


def _forbid_cpu(monkeypatch):
    import highspy
    def forbidden(*args, **kwargs):
        raise AssertionError('No online CPU LP is allowed in a GPU correction')
    monkeypatch.setattr(highspy.Highs, 'run', forbidden)


def _snapshot(out):
    names = ('accepted', 'values', 'objective', 'primal_residual', 'dual_violation',
        'relative_kkt_gap', 'restricted_lifted_dual_violation', 'restricted_pivots',
        'restricted_failed', 'selected_columns')
    result = {key: out[key].get().copy() for key in names}
    result.update({f'warm_{key}': out['warm_state'][key].get().copy()
        for key in ('basis', 'kind', 'x', 'y', 'reduced', 'used_rank')})
    return result


def _same(left, right, exact=False):
    assert left.keys() == right.keys()
    for key in left:
        if exact or left[key].dtype.kind in 'biu':
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        else:
            np.testing.assert_allclose(left[key], right[key], rtol=1e-12, atol=1e-12,
                equal_nan=True, err_msg=key)


def _stored_width(solver, out, captured):
    # Non-bucket run_device trims its public warm result after the lifting
    # work. Inspect the replay's untrimmed allocation to test that fixed work.
    raw = next(iter(solver.graph_calls.values())).result if captured else out
    return raw['warm_state']['u'].shape[2]


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('bucket', [False, True])
def test_expansion_optout_preserves_certificate_results_and_warm_state(monkeypatch, captured, bucket):
    pytest.importorskip('cupy')
    p = problem(); base = _base(_anchor(p), [0])
    solver = GpuRestrictedBasis(base, max_pivots=8, rank_bucket=bucket)
    queries = [p, problem(cost=(2., 1.)), problem(rhs=(.1, 3.)), problem(rhs=(-1., 3.))]
    _forbid_cpu(monkeypatch)
    try:
        method = solver.run_device if captured else solver.solve_device
        inputs = base.prepare_host(queries)
        default = method(**inputs, columns=2)
        expected = _snapshot(default)
        assert default['expansion_scores'] is not None
        explicit = method(**inputs, columns=2, compute_expansion=True)
        _same(expected, _snapshot(explicit), exact=True)
        omitted = method(**inputs, columns=2, compute_expansion=False)
        assert omitted['expansion_scores'] is None
        _same(expected, _snapshot(omitted), exact=True)
        assert omitted['accepted'].get().tolist() == [True, True, True, False]
        assert np.isnan(omitted['values'].get()[-1]).all()
    finally:
        solver.clear_graph_cache(); base.clear_graph_cache()


def _diagonal_problem(size=8, positive=False):
    from scipy.sparse import eye
    return NormalizedLP(eye(size, format='csr'), np.ones(size), np.zeros(size),
        np.full(size, 10.), np.full(size, 1. if positive else -1.), 0,
        np.ones(size), np.ones(size))


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('bucket', [False, True])
@pytest.mark.parametrize('variable_rows', [[], [0]])
def test_small_pivot_budget_reduces_cold_lift_capacity_without_false_accept(monkeypatch, captured, bucket, variable_rows):
    pytest.importorskip('cupy')
    p = _diagonal_problem(); q = _diagonal_problem(positive=True); anchor = _anchor(p)
    bases = [_base(anchor, variable_rows), _base(anchor, variable_rows)]
    solvers = [GpuRestrictedBasis(base, max_pivots=1, rank_bucket=bucket,
        pivot_aware_capacity=enabled) for base, enabled in zip(bases, (False, True))]
    _forbid_cpu(monkeypatch)
    try:
        results = []; widths = []
        for base, solver in zip(bases, solvers):
            method = solver.run_device if captured else solver.solve_device
            out = method(**base.prepare_host([q]), columns=8, compute_expansion=False)
            results.append(_snapshot(out)); widths.append(_stored_width(solver, out, captured))
            assert not out['accepted'].get()[0]
            assert np.isnan(out['values'].get()).all()
            assert out['restricted_pivots'].get()[0] == 1
            assert out['warm_state']['used_rank'].get()[0] == len(variable_rows)+1
        _same(*results)
        assert widths == [len(variable_rows)+8, len(variable_rows)+1]
    finally:
        for solver, base in zip(solvers, bases):
            solver.clear_graph_cache(); base.clear_graph_cache()


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('variable_rows', [[], [0]])
def test_zero_pivot_cold_nonbucket_matches_legacy_without_basis_updates(monkeypatch, captured, variable_rows):
    pytest.importorskip('cupy')
    p = problem(); q = problem(cost=(2., 1.)); anchor = _anchor(p)
    bases = [_base(anchor, variable_rows), _base(anchor, variable_rows)]
    solvers = [GpuRestrictedBasis(base, max_pivots=0, rank_bucket=False,
        pivot_aware_capacity=enabled) for base, enabled in zip(bases, (False, True))]
    _forbid_cpu(monkeypatch)
    try:
        results = []
        for base, solver in zip(bases, solvers):
            method = solver.run_device if captured else solver.solve_device
            # The unchanged objective is already certified. The changed
            # objective requires pivots and must remain rejected at budget 0.
            out = method(**base.prepare_host([p, q]), columns=2, compute_expansion=False)
            results.append(_snapshot(out))
            assert out['accepted'].get().tolist() == [True, False]
            np.testing.assert_array_equal(out['restricted_pivots'].get(), [0, 0])
            np.testing.assert_array_equal(out['warm_state']['used_rank'].get(),
                [len(variable_rows), len(variable_rows)])
            np.testing.assert_array_equal(out['warm_state']['basis'].get(),
                np.broadcast_to(base.basis0.get(), (2, base.m)))
            assert np.isnan(out['values'].get()[1]).all()
            assert out['expansion_scores'] is None
        _same(*results)
    finally:
        for solver, base in zip(solvers, bases):
            solver.clear_graph_cache(); base.clear_graph_cache()


def test_bucketed_pivot_capacity_preserves_warm_updates_changed_columns_and_replay(monkeypatch):
    cp = pytest.importorskip('cupy')
    p = _diagonal_problem(); q = _diagonal_problem(positive=True); anchor = _anchor(p)
    bases = [_base(anchor, [0]), _base(anchor, [0])]
    solvers = [GpuRestrictedBasis(base, max_pivots=1, rank_bucket=True,
        pivot_aware_capacity=enabled) for base, enabled in zip(bases, (False, True))]
    _forbid_cpu(monkeypatch)
    try:
        inputs = [base.prepare_host([q]) for base in bases]
        warm = [None, None]
        for round_index in range(12):
            rows = []; next_warm = []
            width = 2 if round_index in (0, 2) else 8
            for index, (base, solver) in enumerate(zip(bases, solvers)):
                prior = warm[index]
                kinds = base.kind0 if prior is None else prior['kind'][0]
                selected = cp.flatnonzero(kinds != 2).astype(cp.int64)
                if round_index % 2:
                    selected = selected[::-1].copy()
                selected = selected[:width][None]
                out = solver.run_device(**inputs[index], selected_columns=selected,
                    warm_start=prior, reuse_lifted_state=prior is not None, compute_expansion=False)
                rows.append(_snapshot(out)); next_warm.append(out['warm_state'])
                assert out['expansion_scores'] is None
                assert out['warm_state']['u'].shape[2] >= int(out['warm_state']['used_rank'].max().get())
                assert out['restricted_pivots'].get()[0] <= 1
            _same(*rows)
            warm = next_warm
        assert rows[0]['accepted'].tolist() == [True]
        np.testing.assert_allclose(rows[0]['values'], 0., atol=1e-12)
        # Once rank and array shapes stabilize, changed direction VALUES
        # must replay the same graph. Continue one owner-bound warm state.
        solver, base, state = solvers[1], bases[1], warm[1]
        selected = cp.flatnonzero(state['kind'][0] != 2).astype(cp.int64)[None]
        first = solver.run_device(**inputs[1], selected_columns=selected,
            warm_start=state, reuse_lifted_state=True, compute_expansion=False)
        compiled = solver.graph_compilation_seconds
        again = solver.run_device(**inputs[1], selected_columns=selected[:, ::-1].copy(),
            warm_start=first['warm_state'], reuse_lifted_state=True, compute_expansion=False)
        assert again['accepted'].get()[0]
        assert solver.graph_compilation_seconds == compiled
        assert again['warm_state']['used_rank'].get()[0] == 9
    finally:
        for solver, base in zip(solvers, bases):
            solver.clear_graph_cache(); base.clear_graph_cache()


@pytest.mark.parametrize('captured', [False, True])
def test_optimized_restricted_warm_and_column_validation_remains_fail_closed(monkeypatch, captured):
    cp = pytest.importorskip('cupy')
    p = problem(); q = problem(cost=(2., 1.)); base = _base(_anchor(p), [0])
    solver = GpuRestrictedBasis(base, max_pivots=4, rank_bucket=True, pivot_aware_capacity=True)
    _forbid_cpu(monkeypatch)
    try:
        method = solver.run_device if captured else solver.solve_device
        good = method(**base.prepare_host([q]), columns=2, compute_expansion=False)
        assert good['accepted'].get()[0]
        state = good['warm_state']
        for changed in (problem(rhs=(.1, 3.), cost=(2., 1.)),
                problem(coef=2., cost=(2., 1.)), problem(cost=(1., 2.)),
                problem(lower=(.1, 0.), cost=(2., 1.))):
            out = method(**base.prepare_host([changed]), columns=2, warm_start=state,
                reuse_lifted_state=True, compute_expansion=False)
            assert not out['accepted'].get()[0]
            assert np.isnan(out['values'].get()).all()
        for selected in (cp.array([[-1]], dtype=cp.int64), cp.array([[0, 0]], dtype=cp.int64),
                state['basis'][:, :1].copy().astype(cp.int64)):
            out = method(**base.prepare_host([q]), selected_columns=selected, warm_start=state,
                reuse_lifted_state=True, compute_expansion=False)
            assert not out['accepted'].get()[0]
            assert np.isnan(out['values'].get()).all()
    finally:
        solver.clear_graph_cache(); base.clear_graph_cache()
