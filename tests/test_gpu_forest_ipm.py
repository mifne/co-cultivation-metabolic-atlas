import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_forest_ipm import ForestGpuBatchedIPM, prepare_forest_batch
from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from src.lp_trace import problem_hash
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem(*, negative=False):
    return (csr_matrix([[1.,1. if negative else -1.,0.],[0.,0.,1.]]),
        np.array([0.,1.]),np.array([0.,-2. if negative else 0.,0.]),
        np.array([3. if negative else 2.,0. if negative else 2.,2.]),
        np.array([-1.,0. if negative else -1.,1.]),1)


def test_cpu_independent_forests_preserve_original_certificate_and_bound_witness():
    inputs=[problem(),problem(negative=True)]
    original_hashes=[problem_hash(p) for p in inputs]
    full,plans,reductions=prepare_forest_batch(inputs)
    assert plans[0].equality_fingerprint!=plans[1].equality_fingerprint
    assert reductions[1].upper_witness[0]==1
    for p,plan,reduction in zip(full,plans,reductions):
        z=np.array([2.,0.]);y=np.array([0.])
        cost=reduction.problem[4]-reduction.problem[0].T@y
        x=plan.expand_primal(z)
        original_y=reduction.lift_dual(y,cost)
        assert paired_certificate(p,x,original_y)['certificate_passed']
        np.testing.assert_allclose(plan.compress_primal(x),z,atol=1e-12)
    assert [problem_hash(p) for p in inputs]==original_hashes
    inputs[0][0].data[:]=9.
    assert problem_hash(full[0])==original_hashes[0]
    with pytest.raises(ValueError):
        full[0][4][0]=0.


def test_cpu_invalid_batch_is_rejected_before_parent_gpu_allocation(monkeypatch):
    called=[]
    def unexpected(*args,**kwargs):
        called.append(True)
        raise AssertionError('GPU allocation must not run')
    monkeypatch.setattr(ZeroFaceGpuBatchedIPM,'__init__',unexpected)
    p=problem()
    for ps in ([],[p,(*p[:-1],2)], [p,(csr_matrix([[1.,0.,0.],[0.,0.,1.]]),*p[1:])]):
        with pytest.raises(ValueError):
            ForestGpuBatchedIPM(ps)
    assert not called


@pytest.mark.parametrize('cleanup_fails',[False,True])
def test_cpu_forest_setup_exception_closes_parent_and_preserves_error(monkeypatch,cleanup_fails):
    closed=[]
    def init(self,*args,**kwargs):self.setup_seconds=0.
    def setup(*args):raise ValueError('forest setup failed')
    def close(*args):
        closed.append(True)
        if cleanup_fails:raise RuntimeError('cleanup failed')
    monkeypatch.setattr(ZeroFaceGpuBatchedIPM,'__init__',init)
    monkeypatch.setattr(ForestGpuBatchedIPM,'_setup_forest_device',setup)
    monkeypatch.setattr(ForestGpuBatchedIPM,'close',close)
    with pytest.raises(ValueError,match='forest setup failed') as error:
        ForestGpuBatchedIPM([problem()])
    assert closed==[True]
    if cleanup_fails:assert 'cleanup failed' in ' '.join(error.value.__notes__)


def test_gpu_different_forests_and_negative_weight_match_cpu_postsolve():
    import cupy as cp
    ps=[problem(),problem(negative=True)]
    with ForestGpuBatchedIPM(ps) as solver:
        assert solver.problem_hashes==tuple(problem_hash(p) for p in ps)
        assert solver.full_n==3 and solver.original_n==2
        z=cp.asarray([[2.,0.],[2.,0.]],dtype=cp.float64)
        y=cp.zeros((2,1),dtype=cp.float64)
        x,yy=solver.expand_forest_device(z,y)
        for i,reduction in enumerate(solver.forest_reductions):
            cost=reduction.problem[4]-reduction.problem[0].T@y[i].get()
            cy=reduction.lift_dual(y[i].get(),cost)
            np.testing.assert_allclose(x[i].get(),reduction.expand_primal(z[i].get()),atol=1e-12)
            np.testing.assert_allclose(yy[i].get(),cy,atol=1e-12)
            assert paired_certificate(ps[i],x[i].get(),yy[i].get())['certificate_passed']
        result=solver.solve(initial_x=x,initial_y=yy,iterations=0)
        assert result['accepted'].all() and result['factor_count']==0
        assert result['x'].shape==(2,3) and result['y'].shape==(2,2)


def test_gpu_forest_and_zero_face_composition_cold_solve_uses_full_certificate():
    p=(csr_matrix([[1.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,-1.],[0.,0.,1.,0.]]),
        np.array([0.,0.,0.,1.]),np.zeros(4),np.array([3.,3.,2.,2.]),
        np.array([-1.,-9.,-1.,-1.]),3)
    with ForestGpuBatchedIPM([p,p]) as solver:
        result=solver.solve(iterations=40)
        assert result['accepted'].all(),result['metrics']
        assert solver.full_n==4 and solver.original_n==2 and solver.n==1
        assert result['cpu_lp_calls']==0
        for x,y in zip(result['x'].get(),result['y'].get()):
            assert paired_certificate(p,x,y)['certificate_passed']


def test_gpu_full_original_gate_rejects_corrupted_forest_lift(monkeypatch):
    import cupy as cp
    with ForestGpuBatchedIPM([problem()]) as solver:
        original=solver.expand_forest_device
        def corrupt(z,y):
            xx,yy=original(z,y)
            xx[:,1]+=0.1
            return xx,yy
        monkeypatch.setattr(solver,'expand_forest_device',corrupt)
        z=cp.asarray([[2.,0.]],dtype=cp.float64)
        y=cp.zeros((1,1),dtype=cp.float64)
        assert not solver.certificate(z,y)[0]['certificate_passed']


def test_gpu_forest_public_shape_and_finite_gate():
    import cupy as cp
    with ForestGpuBatchedIPM([problem()]) as solver:
        z=cp.asarray([[2.,0.]],dtype=cp.float64)
        y=cp.zeros((1,1),dtype=cp.float64)
        with pytest.raises(ValueError):solver.expand_forest_device(z[0],y)
        with pytest.raises(ValueError):solver.expand_forest_device(z.astype(cp.float32),y)
        with pytest.raises(ValueError):solver.solve(initial_x=z,iterations=0)
        z[:]=cp.nan
        assert not solver.certificate(z,y)[0]['certificate_passed']
