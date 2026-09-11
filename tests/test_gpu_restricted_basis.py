import numpy as np
import pytest
from tests.test_gpu_revised_basis import problem,cpu
from src.gpu_certified_basis import compile_basis
from src.gpu_revised_basis import GpuRevisedBasis
from src.gpu_restricted_basis import GpuRestrictedBasis


@pytest.mark.parametrize('captured',[False,True])
def test_restricted_full_span_matches_cpu_changed_bounds_matrix_objective(monkeypatch,captured):
    pytest.importorskip('cupy');import highspy
    p=problem();queries=[p,problem(rhs=(.1,3.)),problem(cost=(2.,1.)),
        problem(coef=1.7,cost=(-1.,-2.)),problem(lower=(0.,1.),cost=(2.,-1.))]
    expected=[cpu(q).fun for q in queries]
    base=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=32,
        capture_safe=True,compact_updates=True,reuse_small_factor=True)
    solver=GpuRestrictedBasis(base,max_pivots=32)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    try:
        method=solver.run_device if captured else solver.solve_device
        for rows,values in ((queries,expected),(queries[::-1],expected[::-1])):
            out=method(**base.prepare_host(rows),columns=2)
            assert out['accepted'].get().all(), {k:v.get().tolist() for k,v in out.items() if hasattr(v,'get')}
            np.testing.assert_allclose(out['objective'].get(),values,atol=1e-7)
        bad=method(**base.prepare_host([problem(rhs=(-1.,3.))]),columns=2)
        assert not bad['accepted'].get()[0] and np.isnan(bad['values'].get()).all()
    finally:solver.clear_graph_cache();base.clear_graph_cache()


def test_restricted_one_direction_is_not_mistaken_for_global_optimum(monkeypatch):
    pytest.importorskip('cupy');import highspy
    p=problem();q=problem(cost=(2.,1.));expected=cpu(q).fun
    base=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=32,
        capture_safe=True,compact_updates=True,reuse_small_factor=True)
    solver=GpuRestrictedBasis(base,max_pivots=32)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    try:
        narrow=solver.run_device(**base.prepare_host([q]),columns=1)
        assert not narrow['accepted'].get()[0]
        wider=solver.run_device(**base.prepare_host([q]),columns=2)
        assert wider['accepted'].get()[0]
        np.testing.assert_allclose(wider['objective'].get(),[expected],atol=1e-8)
    finally:solver.clear_graph_cache();base.clear_graph_cache()


@pytest.mark.parametrize('captured',[False,True])
def test_fixed_width_column_generation_reuses_basis_and_rejects_changed_rhs(monkeypatch,captured):
    pytest.importorskip('cupy');import highspy
    p=problem();q=problem(cost=(2.,1.));expected=cpu(q).fun
    base=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=32,
        capture_safe=True,compact_updates=True,reuse_small_factor=True)
    solver=GpuRestrictedBasis(base,max_pivots=32)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    try:
        method=solver.run_device if captured else solver.solve_device
        first=method(**base.prepare_host([q]),columns=1)
        assert not first['accepted'].get()[0]
        second=method(**base.prepare_host([q]),columns=1,warm_start=first['warm_state'])
        assert second['accepted'].get()[0]
        assert second['restricted_columns']==1
        np.testing.assert_allclose(second['objective'].get(),[expected],atol=1e-8)
        bad=method(**base.prepare_host([problem(rhs=(.1,3.),cost=(2.,1.))]),columns=1,warm_start=second['warm_state'])
        assert not bad['accepted'].get()[0] and np.isnan(bad['values'].get()).all()
    finally:solver.clear_graph_cache();base.clear_graph_cache()


@pytest.mark.parametrize('continued',[False,True])
def test_backend_enriches_only_rejected_rows_without_cpu_lp(monkeypatch,continued):
    cp=pytest.importorskip('cupy');import highspy
    from types import SimpleNamespace
    import src.gpu_batched_compiled_backend as module
    from src.gpu_certified_basis import NormalizedLP
    from src.gpu_basis_bank import GpuBasisBank
    p=problem();q=problem(cost=(2.,1.));expected=cpu(q).fun
    bank=GpuBasisBank([compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)],[0])
    key=('exchange',2,2,0)
    coords=SimpleNamespace(n_fluxes=2,normalize=lambda a,r,l,u,c,n:NormalizedLP(a,r,l,u,c,n,np.ones(2),np.ones(2)))
    monkeypatch.setattr(module,'stage_key',lambda *a:key)
    service=module.BatchedCompiledBackend(coords,{key:bank},pivots=32,compact_updates=True,reuse_small_factor=True,
        require_tie=False,restricted_columns=(1,) if continued else (1,2),restricted_pivots=32,max_repair_batch=2,
        restricted_rounds=2 if continued else 1)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    def request(q):return q.c,dict(A_ub=q.a,b_ub=q.rhs,bounds=list(zip(q.lower,q.upper)))
    out=service.solve_batch([request(p),request(q)])
    assert all(v.success for v in out)
    np.testing.assert_allclose(out[1].fun,expected,atol=1e-8)
    reduced=[g for g in service.history[-1]['groups'] if g['route']=='restricted_GPU_columns']
    assert [g['columns'] for g in reduced]==([1,1] if continued else [1,2])
    assert all(g['ids']==[1] for g in reduced)
    assert reduced[0]['accepted']==[False] and reduced[1]['accepted']==[True]


@pytest.mark.parametrize('bucket',[False,True])
def test_random_bounded_full_span_and_no_physical_matrix_update(monkeypatch,bucket):
    pytest.importorskip('cupy');import highspy
    from scipy.sparse import csr_matrix
    from scipy.optimize import linprog
    from src.gpu_certified_basis import NormalizedLP
    rng=np.random.default_rng(946)
    a=csr_matrix(rng.uniform(.1,1.,(7,5)))
    lo=np.zeros(5);hi=np.full(5,4.);rhs=np.full(7,3.);c=-rng.uniform(.5,2.,5)
    anchor=compile_basis(a,rhs,lo,hi,c,0)
    queries=[];expected=[]
    for _ in range(5):
        b=rhs*rng.uniform(.4,1.5,7);cost=rng.normal(size=5)
        queries.append(NormalizedLP(a,b,lo,hi,cost,0,np.ones(5),np.ones(7)))
        expected.append(linprog(cost,A_ub=a,b_ub=b,bounds=list(zip(lo,hi)),method='highs').fun)
    base=GpuRevisedBasis(anchor,[],max_pivots=64,capture_safe=True,compact_updates=True,reuse_small_factor=True)
    solver=GpuRestrictedBasis(base,max_pivots=128,rank_bucket=bucket)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    try:
        out=solver.run_device(**base.prepare_host(queries),columns=5)
        assert out['accepted'].get().all()
        np.testing.assert_allclose(out['objective'].get(),expected,atol=1e-7)
    finally:solver.clear_graph_cache();base.clear_graph_cache()


def test_aggregate_reuses_certified_stage1_with_changed_objective_and_bound(monkeypatch):
    pytest.importorskip('cupy');import highspy
    from types import SimpleNamespace
    import src.gpu_batched_compiled_backend as module
    from src.gpu_certified_basis import NormalizedLP
    from src.gpu_basis_bank import GpuBasisBank
    p=problem();q=problem(cost=(2.,1.),lower=(0.1,0.))
    expected=cpu(q).fun
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    keys={name:(name,2,2,0) for name in ('maxmin','aggregate')}
    banks={key:GpuBasisBank([anchor],[0]) for key in keys.values()}
    coords=SimpleNamespace(n_fluxes=2,normalize=lambda a,r,l,u,c,n:NormalizedLP(a,r,l,u,c,n,np.ones(2),np.ones(2)))
    stage=['maxmin'];monkeypatch.setattr(module,'stage_key',lambda *a:keys[stage[0]])
    service=module.BatchedCompiledBackend(coords,banks,pivots=32,compact_updates=True,reuse_small_factor=True,
        require_tie=False,restricted_columns=(1,),restricted_rounds=4,restricted_stage1_warm=True)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    def request(q):return q.c,dict(A_ub=q.a,b_ub=q.rhs,bounds=list(zip(q.lower,q.upper)))
    assert service.solve_batch([request(p)])[0].success
    stage[0]='aggregate';out=service.solve_batch([request(q)])[0]
    assert out.success;np.testing.assert_allclose(out.fun,expected,atol=1e-8)
    assert service.history[-1]['groups'] and all(g['route']=='restricted_stage1_GPU_columns' for g in service.history[-1]['groups'])


def test_bucketed_rounds_reuse_graph_and_validate_lifted_state_inputs(monkeypatch):
    pytest.importorskip('cupy');import highspy
    p=problem();q=problem(cost=(2.,1.));expected=cpu(q).fun
    base=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=32,
        capture_safe=True,compact_updates=True,reuse_small_factor=True)
    solver=GpuRestrictedBasis(base,max_pivots=32,rank_bucket=True)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    try:
        first=solver.run_device(**base.prepare_host([q]),columns=1)
        second=solver.run_device(**base.prepare_host([q]),columns=1,warm_start=first['warm_state'],reuse_lifted_state=True)
        assert second['accepted'].get()[0]
        np.testing.assert_allclose(second['objective'].get(),[expected],atol=1e-8)
        third=solver.run_device(**base.prepare_host([q]),columns=1,warm_start=second['warm_state'],reuse_lifted_state=True)
        compiled=solver.graph_compilation_seconds
        fourth=solver.run_device(**base.prepare_host([q]),columns=1,warm_start=third['warm_state'],reuse_lifted_state=True)
        assert fourth['accepted'].get()[0] and solver.graph_compilation_seconds==compiled
        for changed in (problem(cost=(1.,2.)),problem(cost=(2.,1.),lower=(.1,0.))):
            invalid=solver.run_device(**base.prepare_host([changed]),columns=1,warm_start=second['warm_state'],reuse_lifted_state=True)
            assert not invalid['accepted'].get()[0] and np.isnan(invalid['values'].get()).all()
    finally:solver.clear_graph_cache();base.clear_graph_cache()
