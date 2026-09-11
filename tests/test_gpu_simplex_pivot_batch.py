import numpy as np
import pytest


@pytest.mark.parametrize("upper",[np.inf,3.,2.])
def test_device_pivots_toy_lp(upper):
    cp=pytest.importorskip("cupy")
    from src.gpu_simplex_pivot_batch import PivotBatch
    tableau=cp.asarray([[1.,1.,1.,0.],[1.,0.,0.,1.]])
    x=cp.asarray([0.,0.,5.,3.]); basis=cp.asarray([2,3],dtype=cp.int64)
    reduced=cp.asarray([-2.,-1.,0.,0.]); limits=cp.asarray([upper,4.,cp.inf,cp.inf])
    batch=PivotBatch(tableau,x,basis,reduced,limits)
    state,_=batch.run()
    assert state[0]==1
    expected=[min(3.,upper),5.-min(3.,upper)]
    np.testing.assert_allclose(batch.x.get()[:2],expected,atol=1e-8)
    np.testing.assert_allclose(tableau.get()@batch.x.get(),[5.,3.],atol=1e-8)


def test_device_pivots_unbounded():
    cp=pytest.importorskip("cupy")
    from src.gpu_simplex_pivot_batch import PivotBatch
    batch=PivotBatch(cp.asarray([[-1.,1.]]),cp.asarray([0.,1.]),cp.asarray([1],dtype=cp.int64),
        cp.asarray([-1.,0.]),cp.asarray([cp.inf,cp.inf]))
    assert batch.run()[0][0]==2


@pytest.mark.parametrize("seed",range(8))
def test_full_device_simplex_against_cpu(seed):
    pytest.importorskip("cupy")
    from scipy.optimize import linprog
    from src.gpu_device_bounded_simplex import GpuDeviceBoundedSimplex
    rng=np.random.default_rng(seed)
    matrix=rng.normal(size=(8,12)); feasible=rng.uniform(-1,1,12)
    kw=dict(A_eq=matrix[:3],b_eq=matrix[:3]@feasible,A_ub=matrix[3:],
        b_ub=matrix[3:]@feasible+rng.uniform(0,1,5),bounds=[(-2,2)]*12)
    c=rng.normal(size=12)
    cpu=linprog(c,**kw)
    gpu=GpuDeviceBoundedSimplex().solve(c,**kw)
    assert gpu.success,gpu.message
    assert abs(gpu.fun-cpu.fun)<1e-5
