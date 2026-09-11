import numpy as np
import pytest
from src.gpu_compact_basis import CompactBank,project_basis
from src.gpu_certified_basis import compile_basis
from src.gpu_basis_bank import GpuBasisBank
from tests.test_gpu_revised_basis import problem,cpu


def test_candidate_ranking_keeps_order_within_dual_classes():
    cp=pytest.importorskip('cupy')
    bank=CompactBank.__new__(CompactBank);bank.cp=cp;bank.candidate_ranking='count'
    bank.last_candidate_scores=cp.array([[100.,200.,2.1,1.1]],dtype=cp.float64)
    bank.last_candidate_dual=cp.array([[False,True,False,False]])
    assert bank.rank().get().tolist()==[[1,3,2,0]]
    bank.candidate_ranking='count_only'
    assert bank.rank().get().tolist()==[[3,2,0,1]]


def test_compact_maps_match_full_inverse_and_cpu_without_online_lp(monkeypatch):
    pytest.importorskip('cupy')
    import highspy
    p=problem();anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    queries=[p,problem(rhs=(6.,3.)),problem(coef=1.2),problem(upper=(4.,10.))]
    full=GpuBasisBank([anchor],[0]);device=full.prepare_host(queries)
    data={k:v.get() for k,v in device.items()}
    projected=project_basis(anchor,data,np.zeros((1,1,2)),[0])
    bank=CompactBank(dict(a=p.a,neq=0),[0],[projected],np.zeros((1,1)),np.array([0]),np.ones(1))
    expected=[cpu(q).fun for q in queries]
    monkeypatch.setattr(highspy.Highs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU LP')))
    out=bank.evaluate_device(bank.prepare_host(queries))
    assert out['accepted'].get().all()
    np.testing.assert_allclose(out['objective'].get(),expected,atol=1e-7)
    assert out['cpu_lp_calls']==0


def test_compact_unsupported_family_and_infeasible_inputs_fail_closed():
    cp=pytest.importorskip('cupy')
    p=problem();anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    full=GpuBasisBank([anchor],[0]);data={k:v.get() for k,v in full.prepare_host([p]).items()}
    projected=project_basis(anchor,data,np.zeros((1,1,2)),[0])
    bank=CompactBank(dict(a=p.a,neq=0),[0],[projected],np.zeros((1,1)),np.array([0]),np.ones(1))
    inputs=bank.prepare_host([problem(rhs=(-1.,3.)),p]);inputs['rhs'][1,0]=cp.nan
    out=bank.evaluate_device(inputs)
    assert not out['accepted'].get().any()
    assert np.isnan(out['values'].get()).all()


def test_same_basis_secondary_certificate_respects_primary_allowance():
    cp=pytest.importorskip('cupy')
    from src.gpu_compact_backend import certify_same_basis_tie
    p=problem();q=problem(cost=(1.,2.))
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    full=GpuBasisBank([anchor],[0]);data={k:v.get() for k,v in full.prepare_host([p]).items()}
    projected=project_basis(anchor,data,np.zeros((1,1,2)),[0],[q.c])
    bank=CompactBank(dict(a=p.a,neq=0),[0],[projected],np.zeros((1,1)),np.array([0]),np.ones(1))
    pin=bank.prepare_host([p]);qin=bank.prepare_host([q])
    primary=bank.evaluate_device(pin);secondary=bank.evaluate_device(qin,order=np.zeros((1,1),int))
    # Secondary optimum differs without the retained primary objective.
    assert not secondary['accepted'].get()[0]
    tight=certify_same_basis_tie(bank,primary,pin,qin,secondary,primary['objective']+1e-10)
    assert tight['accepted'].get()[0]
    loose=certify_same_basis_tie(bank,primary,pin,qin,secondary,primary['objective']+1.)
    assert not loose['accepted'].get()[0]


def test_zero_column_projection_and_unseen_matrix_direction():
    pytest.importorskip('cupy')
    p=problem();anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    full=GpuBasisBank([anchor],[0]);data={k:v.get() for k,v in full.prepare_host([p]).items()}
    projected=project_basis(anchor,data,np.zeros((1,1,2)),[0])
    assert projected['p_delta'].shape[1]==0
    bank=CompactBank(dict(a=p.a,neq=0),[0],[projected],np.zeros((1,1)),np.array([0]),np.ones(1))
    out=bank.evaluate_device(bank.prepare_host([p,problem(coef=1.1)]))
    assert out['accepted'].get().tolist()==[True,False]


def test_offline_operator_repairs_miss_without_online_factorization(tmp_path,monkeypatch):
    pytest.importorskip('cupy')
    import hashlib,highspy,scipy.sparse.linalg
    from src.gpu_revised_basis import GpuRevisedBasis
    p=problem();q=problem(rhs=(.1,3.))
    anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    expected=cpu(q).fun
    full=GpuBasisBank([anchor],[0]);data={k:v.get() for k,v in full.prepare_host([p,q]).items()}
    projected=project_basis(anchor,data,np.zeros((1,1,2)),[0])
    bank=CompactBank(dict(a=p.a,neq=0),[0],[projected],np.zeros((1,1)),np.array([0]),np.ones(1))
    file=tmp_path/'inverse.npz';inv=anchor['inverse']
    np.savez(file,inverse_data=inv.data,inverse_indices=inv.indices,inverse_indptr=inv.indptr,inverse_shape=inv.shape)
    bank.evaluators[0].repair_path=(file,hashlib.sha256(file.read_bytes()).hexdigest())
    def forbidden(*a,**kw):raise AssertionError('Online CPU factorization/optimization')
    monkeypatch.setattr(highspy.Highs,'run',forbidden)
    monkeypatch.setattr(scipy.sparse.linalg,'splu',forbidden)
    assert not bank.evaluate_device(bank.prepare_host([q]))['accepted'].get()[0]
    adapter=bank.evaluators[0].repair_adapter()
    solver=GpuRevisedBasis(adapter.anchor,adapter.variable_rows,max_pivots=12,adapter=adapter)
    out=solver.run_device(**solver.prepare_host([q]))
    assert out['accepted'].get()[0]
    np.testing.assert_allclose(out['objective'].get()[0],expected,atol=1e-7)


def test_compact_cuda_replay_uses_updated_inputs_and_same_certificates():
    pytest.importorskip('cupy')
    p=problem();q=problem(rhs=(6.,3.));anchor=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    full=GpuBasisBank([anchor],[0]);data={k:v.get() for k,v in full.prepare_host([p,q]).items()}
    projected=project_basis(anchor,data,np.zeros((1,1,2)),[0])
    bank=CompactBank(dict(a=p.a,neq=0),[0],[projected],np.zeros((1,1)),np.array([0]),np.ones(1),capture=True,full_batch_candidates=True)
    try:
        for queries in ([p,q],[q,p]):
            out=bank.evaluate_device(bank.prepare_host(queries))
            assert out['accepted'].get().all()
            np.testing.assert_allclose(out['objective'].get(),[cpu(v).fun for v in queries],atol=1e-7)
        assert len(bank.graph_cache)==1
    finally:bank.clear_graph_cache()
