import numpy as np
import pytest
from types import SimpleNamespace
from greenlet import greenlet,getcurrent
from src.gpu_batched_compiled_backend import YieldingLPBackend
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend


def test_portable_growth_state_releases_owner_and_checks_provenance():
    import weakref,gc
    from src.gpu_batched_compiled_backend import portable_result,bind_portable_result,BatchedCompiledBackend
    class Owner:pass
    old=Owner();ref=weakref.ref(old)
    result=dict(warm_state=dict(owner=old,basis=np.array([[2,3]])))
    packet=portable_result(result)
    del old,result;gc.collect()
    assert ref() is None and packet['warm_state']['owner'] is None
    new=Owner();new.operator_identity=('maxmin',7,'immutable-hash',False)
    bound=bind_portable_result(packet,new,new.operator_identity)
    assert bound['warm_state']['owner'] is new
    assert packet['warm_state']['owner'] is None
    with pytest.raises(ValueError,match='operator changed'):
        bind_portable_result(packet,new,('maxmin',0,'other',False))
    service=BatchedCompiledBackend(None,{},max_repair_batch=3)
    assert service.chunks(list(range(8)))==[[0,1,2],[3,4,5],[6,7]]


def test_temporal_gpu_basis_updates_changed_matrix_and_fails_closed(monkeypatch):
    pytest.importorskip('cupy');import highspy
    from scipy.sparse import csr_matrix
    import src.gpu_batched_compiled_backend as module
    from src.gpu_certified_basis import compile_basis,NormalizedLP
    from src.gpu_basis_bank import GpuBasisBank
    a=csr_matrix([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    rhs=np.array([3.,3.,-1.,-1.]);lower=np.zeros(2);upper=np.full(2,10.)
    anchor=compile_basis(a,rhs,lower,upper,-np.ones(2),0)
    bank=GpuBasisBank([anchor],[2]);key=('aggregate',4,2,0)
    coords=SimpleNamespace(n_fluxes=2,normalize=lambda a,r,l,u,c,n:NormalizedLP(a,r,l,u,c,n,np.ones(2),np.ones(4)))
    monkeypatch.setattr(module,'stage_key',lambda *a:key)
    service=module.BatchedCompiledBackend(coords,{key:bank},pivots=12,compact_updates=True,
        pivot_refinement=1,reuse_small_factor=True,temporal_pivots=8,max_repair_batch=1,require_tie=False)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Online CPU LP')))
    def request(matrix,b):return (np.ones(2),dict(A_ub=matrix,b_ub=b,bounds=list(zip(lower,upper))))
    first=service.solve_batch([request(a,rhs)])[0]
    assert first.success and abs(first.fun-2.)<1e-8 and service.temporal_cache[key]
    assert service.temporal_cache[key][0]['result']['warm_state']['owner'] is None
    changed=a.copy();changed[2,0]=-2.
    second=service.solve_batch([request(changed,np.array([3.,3.,-3.,-1.]))])[0]
    assert second.success and abs(second.fun-2.5)<1e-8
    assert any(g['route']=='temporal_GPU_basis_update' and all(g['accepted']) for g in service.history[-1]['groups'])
    failed=service.solve_batch([request(changed,np.array([3.,3.,-8.,-1.]))])[0]
    assert not failed.success and failed.x is None


def test_environment_bridge_preserves_history_and_secondary_stage():
    from scipy.sparse import csr_matrix
    bridge=YieldingLPBackend(getcurrent())
    layout=SimpleNamespace(n_fluxes=1,_exchange_terms={'h2o_e':[('s',0,-1.,None)]})
    outer=GpuExchangeTieBreakBackend(layout,inner=bridge)
    objective=np.array([0.,0.,1.])
    worker=greenlet(lambda:outer.solve(objective,A_ub=csr_matrix((0,3)),b_ub=np.zeros(0),bounds=[(0,None)]*3))
    request=worker.switch()
    np.testing.assert_array_equal(request[0],objective)
    result=SimpleNamespace(success=True,x=np.array([0.,0.,2.]),fun=2.,message='test',
        diagnostics=dict(success=True,max_original_residual=0.,total_seconds=.1,objective=2.,cpu_lp_calls=0))
    second=worker.switch(result)
    assert not worker.dead
    np.testing.assert_array_equal(second[0],np.zeros(3))
    np.testing.assert_array_equal(second[1]['A_ub'].toarray(),objective[None])
    final=worker.switch(result)
    assert worker.dead and final.success and len(bridge.history)==2
    assert outer.history[0]['actual_gpu_lp_calls']==2


@pytest.mark.parametrize('use_reserve',[False,True])
def test_short_alternative_basis_is_certified_and_infeasible_row_rejected(monkeypatch,use_reserve):
    import pytest
    pytest.importorskip('cupy')
    import highspy
    import src.gpu_batched_compiled_backend as module
    from src.gpu_basis_bank import GpuBasisBank
    from src.gpu_certified_basis import compile_basis,NormalizedLP
    from src.gpu_neural_basis_proposal import ConstantBasisProposal,NeuralRoutedBasisBank
    from scipy.sparse import csr_matrix
    def problem(cost=(-1.,-1.),rhs=(3.,3.,-1.,-1.)):
        return NormalizedLP(csr_matrix([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]),np.array(rhs),
            np.zeros(2),np.full(2,10.),np.array(cost),0,np.ones(2),np.ones(4))
    # Move both coordinates from 3 to 1; one pivot is insufficient.
    first=problem();second=problem(cost=(1.,1.))
    bank=GpuBasisBank([compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq) for p in (first,second)],[])
    key=('aggregate',4,2,0)
    coordinates=SimpleNamespace(n_fluxes=2,normalize=lambda a,r,l,u,c,n:NormalizedLP(a,r,l,u,c,n,np.ones(2),np.ones(len(r))))
    monkeypatch.setattr(module,'stage_key',lambda *a:key)
    service=module.BatchedCompiledBackend(coordinates,{key:NeuralRoutedBasisBank(bank,ConstantBasisProposal(),True)},
        pivots=1,aggregate_portfolio=0 if use_reserve else 1,reserve_banks={key:bank} if use_reserve else None)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU LP')))
    def request(p):return (p.c,dict(A_ub=p.a,b_ub=p.rhs,bounds=list(zip(p.lower,p.upper))))
    result=service.solve_batch([request(second),request(problem(rhs=(3.,3.,-4.,-1.)))])
    assert result[0].success and abs(result[0].fun-2.)<1e-8
    assert not result[1].success and result[1].x is None
    if use_reserve:assert service.history[-1]['reserve_direct_accepts']==1
    else:assert any(g['route']=='short_alternative_basis' for g in service.history[-1]['groups'])


@pytest.mark.parametrize('budget,success',[(48,True),(36,False)])
def test_reserve_continuation_keeps_gpu_progress_and_budget_gate(monkeypatch,budget,success):
    pytest.importorskip('cupy')
    import highspy
    from scipy.sparse import eye,vstack
    import src.gpu_batched_compiled_backend as module
    from src.gpu_certified_basis import compile_basis,NormalizedLP
    from src.gpu_basis_bank import GpuBasisBank
    from src.gpu_neural_basis_proposal import ConstantBasisProposal,NeuralRoutedBasisBank
    n=40;a=vstack([eye(n),-eye(n)],format='csr');rhs=np.r_[np.full(n,3.),np.full(n,-1.)]
    lower=np.zeros(n);upper=np.full(n,10.)
    anchor=compile_basis(a,rhs,lower,upper,-np.ones(n),0)
    bank=GpuBasisBank([anchor],[]);key=('aggregate',2*n,n,0)
    coordinates=SimpleNamespace(n_fluxes=n,normalize=lambda a,r,l,u,c,k:NormalizedLP(a,r,l,u,c,k,np.ones(n),np.ones(2*n)))
    monkeypatch.setattr(module,'stage_key',lambda *a:key)
    service=module.BatchedCompiledBackend(coordinates,{key:NeuralRoutedBasisBank(bank,ConstantBasisProposal(),True)},
        pivots=budget,reserve_banks={key:bank},dual_edge='devex')
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU LP')))
    result=service.solve_batch([(np.ones(n),dict(A_ub=a,b_ub=rhs,bounds=list(zip(lower,upper))))])[0]
    assert result.success==success
    assert any(g['route']=='reserve_basis_continuation' for g in service.history[-1]['groups'])
    if success:np.testing.assert_allclose(result.x,np.ones(n),atol=1e-8)
    else:assert result.x is None
