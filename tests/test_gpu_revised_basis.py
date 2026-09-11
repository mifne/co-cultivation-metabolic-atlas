import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from src.gpu_certified_basis import NormalizedLP, compile_basis
from src.gpu_revised_basis import GpuRevisedBasis


def problem(rhs=(5., 3.), coef=1., cost=(-2., -1.), lower=(0., 0.), upper=(10., 10.)):
    return NormalizedLP(csr_matrix([[coef, 1.], [1., 0.]]), np.array(rhs),
        np.array(lower), np.array(upper), np.array(cost), 0, np.ones(2), np.ones(2))


def cpu(p):
    # SciPy HiGHS owns a process-global scheduler. Keep all reference solves
    # consistent with the one-thread persistent CPU backend in combined runs.
    return linprog(p.c, A_eq=p.a[:p.neq] if p.neq else None,
        b_eq=p.rhs[:p.neq] if p.neq else None, A_ub=p.a[p.neq:], b_ub=p.rhs[p.neq:],
        bounds=list(zip(p.lower, p.upper)), method="highs-ds", options={'threads': 1, 'parallel': False})


@pytest.mark.parametrize('pivot_refinement',[0,1])
@pytest.mark.parametrize('compact_updates',[False,True])
def test_repairs_changed_active_set_without_online_cpu(monkeypatch,pivot_refinement,compact_updates):
    pytest.importorskip("cupy")
    import highspy
    p = problem()
    anchor = compile_basis(p.a, p.rhs, p.lower, p.upper, p.c, p.neq)
    solver = GpuRevisedBasis(anchor, [0], max_pivots=12,pivot_refinement=pivot_refinement,compact_updates=compact_updates)
    queries = [p, problem(rhs=(.1, 3.)), problem(cost=(2., 1.)),
        problem(upper=(1., 10.)), problem(coef=2.), problem(lower=(0., 1.), cost=(2., -1.))]
    expected = [cpu(q).fun for q in queries]
    def forbidden(*args, **kwargs): raise AssertionError("Online CPU solver forbidden")
    monkeypatch.setattr(highspy.Highs, "run", forbidden)
    out = solver.solve_device(**solver.prepare_host(queries))
    assert out["accepted"].get().all(), {k: v.get() for k,v in out.items() if hasattr(v,"get")}
    np.testing.assert_allclose(out["objective"].get(), expected, atol=1e-7)
    assert out["cpu_lp_calls"] == 0


def test_infeasible_and_nan_are_not_accepted():
    cp = pytest.importorskip("cupy")
    p = problem()
    anchor = compile_basis(p.a, p.rhs, p.lower, p.upper, p.c, p.neq)
    solver = GpuRevisedBasis(anchor, [0], max_pivots=12)
    queries = [p, problem(rhs=(-1., 3.)), p]
    data = solver.prepare_host(queries)
    data["c"][2, 0] = cp.nan
    out = solver.solve_device(**data)
    assert out["accepted"].get().tolist() == [True, False, False]
    assert np.isnan(out["values"].get()[1:]).all()


@pytest.mark.parametrize('reuse_small_factor',[False,True])
def test_compact_update_capture_and_trimmed_warm_continuation(reuse_small_factor):
    pytest.importorskip('cupy')
    p=problem();q=problem(rhs=(.1,3.))
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=12,
        capture_safe=True,compact_updates=True,pivot_refinement=1,reuse_small_factor=reuse_small_factor)
    try:
        first=solver.run_device(**solver.prepare_host([q]),pivot_budget=1)
        warm=first['warm_state']
        assert warm['v'].shape[1]==1
        assert warm['u'].shape[2]==int(warm['used_rank'].max().get())
        out=solver.run_device(**solver.prepare_host([q]),warm_start=warm,pivot_budget=11)
        assert out['accepted'].get()[0]
        np.testing.assert_allclose(out['objective'].get()[0],cpu(q).fun,atol=1e-7)
    finally:solver.clear_graph_cache()


@pytest.mark.parametrize('compact_updates',[False,True])
def test_online_warm_matrix_and_rhs_updates_are_recertified(compact_updates):
    pytest.importorskip('cupy')
    p=problem()
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=8,compact_updates=compact_updates)
    first=solver.solve_device(**solver.prepare_host([p]))
    q=problem(rhs=(.2,3.),coef=1.7,cost=(-1.,-2.))
    changed=solver.solve_device(**solver.prepare_host([q]),warm_start=first['warm_state'],update_warm_matrix=True)
    assert changed['accepted'].get()[0]
    np.testing.assert_allclose(changed['objective'].get()[0],cpu(q).fun,atol=1e-8)
    invalid=problem(rhs=(-1.,3.),coef=1.7)
    failed=solver.solve_device(**solver.prepare_host([invalid]),warm_start=changed['warm_state'],update_warm_matrix=True)
    assert not failed['accepted'].get()[0]


def test_compact_temporal_update_after_pivots_and_captured_replay(monkeypatch):
    pytest.importorskip('cupy');import highspy
    p=problem();q=problem(rhs=(.1,3.));r=problem(rhs=(.2,3.),coef=1.7,cost=(-1.,-2.))
    s=problem(rhs=(1.,3.),coef=.7,cost=(-2.,-.2));expected=[cpu(v).fun for v in (r,s)]
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=12,
        capture_safe=True,compact_updates=True,pivot_refinement=1,reuse_small_factor=True)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    try:
        first=solver.run_device(**solver.prepare_host([q]))
        assert first['accepted'].get()[0] and first['pivots'].get()[0]>0
        for query,value in zip((r,s),expected):
            prior=first['warm_state']['u'].shape[2]
            out=solver.run_device(**solver.prepare_host([query]),warm_start=first['warm_state'],update_warm_matrix=True)
            assert out['accepted'].get()[0]
            assert out['warm_state']['v'].shape[1]==1
            np.testing.assert_allclose(out['objective'].get(),[value],atol=1e-8)
            first=out
    finally:solver.clear_graph_cache()


def test_equalities_free_and_fixed_variables():
    pytest.importorskip("cupy")
    p = NormalizedLP(csr_matrix([[1., 1., 0.], [1., 0., 1.]]), np.array([2., 3.]),
        np.array([-np.inf, 0., 1.]), np.array([np.inf, 10., 1.]), np.array([-1., 0., 0.]),
        1, np.ones(3), np.ones(2))
    anchor = compile_basis(p.a, p.rhs, p.lower, p.upper, p.c, p.neq)
    solver = GpuRevisedBasis(anchor, [], max_pivots=12)
    q = NormalizedLP(p.a, np.array([2., .5]), p.lower, p.upper, p.c, p.neq, p.col_scale, p.row_scale)
    out = solver.solve_device(**solver.prepare_host([p, q]))
    assert out["accepted"].get().all()
    np.testing.assert_allclose(out["objective"].get(), [cpu(p).fun, cpu(q).fun], atol=1e-8)


def test_budget_exhaustion_and_unbounded_query_fail_closed():
    pytest.importorskip("cupy")
    p=problem()
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    solver=GpuRevisedBasis(anchor,[0],max_pivots=1)
    query=problem(cost=(2.,1.))
    result=solver.solve_device(**solver.prepare_host([query]))
    # One pivot is insufficient to move both positive primal variables to zero.
    assert not result["accepted"].get()[0]
    assert np.isnan(result["values"].get()).all()
    unbounded=problem(cost=(2.,1.),lower=(-np.inf,-np.inf))
    solver=GpuRevisedBasis(anchor,[0],max_pivots=12)
    result=solver.solve_device(**solver.prepare_host([unbounded]))
    assert not result["accepted"].get()[0]


@pytest.mark.parametrize('skip_unused_dual',[False,True])
def test_devex_pricing_eager_and_replayed_match_reference(monkeypatch,skip_unused_dual):
    pytest.importorskip('cupy')
    import highspy
    p=problem()
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    queries=[problem(rhs=(.1,3.)),problem(rhs=(.2,3.),coef=1.7),problem(cost=(2.,1.))]
    expected=[cpu(q).fun for q in queries]
    solver=GpuRevisedBasis(anchor,[0],max_pivots=12,capture_safe=True,dual_edge='devex',skip_unused_dual=skip_unused_dual)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU LP')))
    inputs=solver.prepare_host(queries)
    eager=solver.solve_device(**inputs)
    for _ in range(2):
        out=solver.run_device(**inputs)
        assert out['accepted'].get().all()
        np.testing.assert_allclose(out['objective'].get(),expected,atol=1e-7)
        np.testing.assert_array_equal(out['pivots'].get(),eager['pivots'].get())
        np.testing.assert_allclose(out['warm_state']['edge_weights'].get(),eager['warm_state']['edge_weights'].get(),rtol=1e-8)
    invalid=solver.run_device(**solver.prepare_host([problem(rhs=(-1.,3.))]))
    assert not invalid['accepted'].get()[0]
    assert np.isnan(invalid['values'].get()).all()


def test_graph_eviction_preserves_a_referenced_warm_state(monkeypatch):
    pytest.importorskip('cupy')
    import highspy
    p=problem();q=problem(cost=(2.,1.));expected=cpu(q).fun
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=8,capture_safe=True)
    result=solver.run_device(**solver.prepare_host([p]))
    assert result['accepted'].get()[0]
    solver.clear_graph_cache();assert not solver.graph_calls
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU LP')))
    continued=solver.run_device(**solver.prepare_host([q]),warm_start=result['warm_state'])
    assert continued['accepted'].get()[0]
    np.testing.assert_allclose(continued['objective'].get(),[expected],atol=1e-8)


def test_adaptive_refinement_nested_device_branches_preserve_certificate(monkeypatch):
    pytest.importorskip('cupy')
    import highspy
    p=problem();queries=[problem(rhs=(.1,3.)),problem(cost=(2.,1.)),problem(coef=1.7)]
    expected=[cpu(q).fun for q in queries]
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=12,
        capture_safe=True,compact_updates=True,pivot_refinement=1,reuse_small_factor=True,
        skip_unused_dual=True,adaptive_refinement_tolerance=1e-12)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    try:
        for rows,values in ((queries,expected),(queries[::-1],expected[::-1])):
            out=solver.run_device(**solver.prepare_host(rows))
            assert out['accepted'].get().all()
            np.testing.assert_allclose(out['objective'].get(),values,atol=1e-8)
        failed=solver.run_device(**solver.prepare_host([problem(rhs=(-1.,3.))]))
        assert not failed['accepted'].get()[0]
    finally:solver.clear_graph_cache()
