import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_sparse_factor import CudssError
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem():
    return (csr_matrix([[1.,-1.,0.],[1.,0.,0.],[2.,0.,0.],[0.,0.,-1.]]),
        np.array([0.,0.,0.,-1.]),np.zeros(3),np.array([3.,3.,2.]),np.array([-1.,-9.,1.]),3)


def test_gpu_lift_matches_cpu_reverse_proof_and_original_lp_certificate():
    import cupy as cp
    ps=[problem()]*2
    with ZeroFaceGpuBatchedIPM(ps) as solver:
        x=cp.array([[1.],[1.]],dtype=cp.float64);y=cp.array([[-1.],[-1.]],dtype=cp.float64)
        xx,yy=solver.lift_device(x,y)
        for i,p in enumerate(solver.face_plans):
            cx,cy=p.lift(x[i].get(),y[i].get())
            np.testing.assert_allclose(xx[i].get(),cx,atol=1e-12)
            np.testing.assert_allclose(yy[i].get(),cy,atol=1e-12)
            assert paired_certificate(ps[i],xx[i].get(),yy[i].get())['certificate_passed']


def test_end_to_end_reduced_gpu_solve_returns_original_dimensions_and_gates():
    with ZeroFaceGpuBatchedIPM([problem()]*2) as solver:
        r=solver.solve(iterations=30)
        assert r['accepted'].all(),r['metrics']
        assert r['x'].shape==(2,3) and r['y'].shape==(2,4)
        assert r['cpu_lp_calls']==0 and r['analysis_count']==1
        for x,y in zip(r['x'].get(),r['y'].get()):
            assert paired_certificate(problem(),x,y)['certificate_passed']


def test_certified_warm_dual_on_removed_duplicates_is_aggregated_on_gpu():
    import cupy as cp
    p=(csr_matrix([[1.,1.],[-1.,-1.],[1.,1.]]),np.array([1.,-1.,1.]),
       np.zeros(2),np.full(2,np.inf),np.array([1.,2.]),3)
    x=cp.asarray([[1.,0.],[1.,0.]])
    y=cp.asarray([[0.,-1.,0.],[0.,0.,1.]])
    original_y=y.copy()
    with ZeroFaceGpuBatchedIPM([p,p]) as solver:
        r=solver.solve(initial_x=x,initial_y=y,iterations=0)
        assert r['accepted'].all() and r['factor_count']==0
        np.testing.assert_array_equal(y.get(),original_y.get())
        for xx,yy in zip(r['x'].get(),r['y'].get()):
            assert paired_certificate(p,xx,yy)['certificate_passed']


def test_cpu_original_neq_mismatch_rejected_before_gpu_allocation(monkeypatch):
    called=[]
    def unexpected(*args,**kwargs):
        called.append(True)
        raise AssertionError('GPU base initialization must not run')
    monkeypatch.setattr(GpuCondensedBatchedIPM,'__init__',unexpected)
    p=(csr_matrix([[1.,1.],[0.,0.]]),np.array([1.,0.]),np.zeros(2),
       np.full(2,2.),np.ones(2),1)
    # Both reductions have one equality, so checking only reduced neq would
    # allow allocation before the original assembler rejected the batch.
    with pytest.raises(ValueError,match='original LP equality count'):
        ZeroFaceGpuBatchedIPM([p,(*p[:-1],2)])
    assert not called


@pytest.mark.parametrize('cleanup_fails',[False,True])
def test_cpu_postsolve_setup_exception_closes_factor_and_preserves_error(monkeypatch,cleanup_fails):
    closed=[]
    monkeypatch.setattr(GpuCondensedBatchedIPM,'__init__',lambda *args,**kwargs:None)
    def fail_setup(*args):
        raise ValueError('postsolve setup failed')
    def close(*args):
        closed.append(True)
        if cleanup_fails:
            raise RuntimeError('cleanup failed')
    monkeypatch.setattr(ZeroFaceGpuBatchedIPM,'_setup_postsolve',fail_setup)
    monkeypatch.setattr(ZeroFaceGpuBatchedIPM,'close',close)
    with pytest.raises(ValueError,match='postsolve setup failed') as error:
        ZeroFaceGpuBatchedIPM([problem()])
    assert closed==[True]
    if cleanup_fails:
        assert 'cleanup failed' in ' '.join(error.value.__notes__)


def test_public_gpu_lift_rejects_broadcast_dtype_host_and_wrong_stream():
    import cupy as cp
    with ZeroFaceGpuBatchedIPM([problem()]*2) as solver:
        x=cp.ones((2,1),dtype=cp.float64)
        y=-cp.ones((2,1),dtype=cp.float64)
        for bad_x,bad_y in ((x[:1],y),(x,y[:1]),(x.astype(cp.float32),y),
                             (x,y.astype(cp.float32)),(np.ones((2,1)),y)):
            with pytest.raises(ValueError,match='Exact-shape FP64'):
                solver.lift_device(bad_x,bad_y)
        with cp.cuda.Stream(non_blocking=True):
            with pytest.raises(CudssError,match='stream'):
                solver.lift_device(x,y)
            with pytest.raises(CudssError,match='stream'):
                solver.solve(iterations=0)
    with pytest.raises(CudssError,match='closed'):
        solver.lift_device(x,y)
    with pytest.raises(CudssError,match='closed'):
        solver.solve(iterations=0)


def test_nonfinite_gpu_lift_is_rejected_by_original_certificate_without_extra_guard_sync():
    import cupy as cp
    with ZeroFaceGpuBatchedIPM([problem()]) as solver:
        x=cp.array([[np.nan]],dtype=cp.float64)
        y=cp.array([[-1.]],dtype=cp.float64)
        assert not solver.certificate(x,y)[0]['certificate_passed']
        x[:]=1.
        y[:]=cp.inf
        assert not solver.certificate(x,y)[0]['certificate_passed']


def test_gpu_lift_keeps_distinct_environment_column_mappings():
    import cupy as cp
    p=problem()
    order=np.array([2,1,0])
    second=(p[0][:,order].tocsr(),p[1].copy(),p[2][order],p[3][order],p[4][order],p[-1])
    with ZeroFaceGpuBatchedIPM([p,second]) as solver:
        assert solver.face_plans[0].columns.tolist()==[2]
        assert solver.face_plans[1].columns.tolist()==[0]
        x=cp.ones((2,1),dtype=cp.float64)
        y=-cp.ones((2,1),dtype=cp.float64)
        xx,yy=solver.lift_device(x,y)
        for i,plan in enumerate(solver.face_plans):
            cx,cy=plan.lift([1.],[-1.])
            np.testing.assert_allclose(xx[i].get(),cx,atol=1e-12)
            np.testing.assert_allclose(yy[i].get(),cy,atol=1e-12)
            assert paired_certificate(plan.original,xx[i].get(),yy[i].get())['certificate_passed']
