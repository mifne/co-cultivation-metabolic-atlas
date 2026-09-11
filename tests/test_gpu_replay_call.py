import numpy as np
import pytest
from tests.test_gpu_revised_basis import problem,cpu


def test_close_releases_private_temporaries_but_preserves_external_results():
    cp=pytest.importorskip('cupy')
    from src.gpu_replay_call import GpuReplayCall
    class TemporarySolver:
        def __init__(self):self.cp=cp
        def solve_device(self,values,conditional=None,_warm_iterations=None):
            temporary=cp.arange(1024*1024,dtype=cp.float64)+values[0]
            return {'value':temporary[-1:].copy()}
    inputs={'values':cp.asarray([3.])};call=GpuReplayCall(TemporarySolver(),inputs)
    result=call.run(inputs);cp.cuda.get_current_stream().synchronize()
    before=call.pool.total_bytes();call.close();after=call.pool.total_bytes()
    assert after<before//2
    np.testing.assert_allclose(result['value'].get(),[1024*1024+2.])
    assert call.inputs is None and call.result is None and call.solver is None
    call.close()  # Idempotent cleanup must not double-destroy the graph.


def test_replay_batch_changes_and_invalid_rows_never_use_cpu(monkeypatch):
    pytest.importorskip('cupy')
    import highspy
    from src.gpu_certified_basis import compile_basis
    from src.gpu_revised_basis import GpuRevisedBasis
    p=problem()
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=12,capture_safe=True)
    first=[p,problem(rhs=(.2,3.)),problem(cost=(2.,1.))]
    second=[problem(rhs=(5.2,3.)),problem(coef=1.7),problem(rhs=(-1.,3.))]
    expected=[[cpu(q).fun for q in first],[cpu(q).fun for q in second[:2]]]
    def forbidden(*args,**kwargs):raise AssertionError('CPU optimization forbidden')
    monkeypatch.setattr(highspy.Highs,'run',forbidden)
    for i,queries in enumerate((first,second,first)):
        result=solver.run_device(**solver.prepare_host(queries))
        flags=result['accepted'].get()
        assert flags.tolist()==([True,True,False] if i==1 else [True]*3)
        np.testing.assert_allclose(result['objective'].get()[:len(expected[i%2])],expected[i%2],atol=1e-8)
        if i==1:assert np.isnan(result['values'].get()[2]).all()
    assert len(solver.graph_calls)==1


def test_failed_short_budget_can_continue_without_accepting_old_flux(monkeypatch):
    cp=pytest.importorskip('cupy')
    import highspy
    from src.gpu_certified_basis import compile_basis
    from src.gpu_revised_basis import GpuRevisedBasis
    p=problem();query=problem(cost=(2.,1.))
    expected=cpu(query).fun
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=8,capture_safe=True)
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU LP')))
    inputs=solver.prepare_host([query])
    short=solver.run_device(**inputs,pivot_budget=1)
    assert not short['accepted'].get()[0] and cp.isnan(short['values']).all()
    continued=solver.run_device(**inputs,pivot_budget=7,warm_start=short['warm_state'])
    assert continued['accepted'].get()[0]
    np.testing.assert_allclose(continued['objective'].get()[0],expected,atol=1e-8)


def test_cross_objective_warm_basis_keeps_original_lp_certificate():
    pytest.importorskip('cupy')
    from src.gpu_certified_basis import compile_basis
    from src.gpu_revised_basis import GpuRevisedBasis
    p=problem(cost=(-1.,0.))
    solver=GpuRevisedBasis(compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq),[0],max_pivots=12,capture_safe=True)
    primary=solver.run_device(**solver.prepare_host([p]),pivot_budget=0)
    query=problem(cost=(0.,-1.),lower=(2.97,0.),upper=(3.,10.))
    result=solver.run_device(**solver.prepare_host([query]),warm_start=primary['warm_state'])
    assert result['accepted'].get()[0]
    np.testing.assert_allclose(result['objective'].get()[0],cpu(query).fun,atol=1e-8)
    changed=problem(rhs=(4.,3.),cost=(0.,-1.))
    invalid=solver.run_device(**solver.prepare_host([changed]),warm_start=primary['warm_state'])
    assert not invalid['accepted'].get()[0]  # No undeclared matrix/RHS reuse.
