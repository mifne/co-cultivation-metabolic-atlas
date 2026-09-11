import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_block_affine_feasibility import GpuBlockAffineFeasibility


def problems(rhs=2.):
    return [(csr_matrix([[1.,-1.,0.],[0.,1.,1.]]),np.array([0.,rhs]),
             np.zeros(3),np.array([2.,2.,1.]),np.array([0.,0.,-1.]),1)]


def test_block_projection_current_point_and_equality():
    import cupy as cp
    with GpuCondensedBatchedIPM(problems()) as s:
        p=GpuBlockAffineFeasibility(s,block_size=3)
        x,report=p.propose(cp.array([[1.5,1.5,1.]]),iterations=30,chunk=5)
        assert float(cp.max(cp.abs(s.e@x.ravel()-s.b.ravel())))<1e-10
        assert float(x[0,2])==pytest.approx(1.,abs=1e-10)
        violation=float(cp.max(s.g@x.ravel()-s.h.ravel()))
        assert violation<1e-7
        assert report['violation_history'][-1][1]==pytest.approx(violation)
        assert s.factor.factor_count==0 and report['original_certificate_required']
        assert 'accepted' not in report


def test_auxiliary_failure_is_not_original_infeasibility():
    with GpuCondensedBatchedIPM(problems(.5)) as s:
        p=GpuBlockAffineFeasibility(s,block_size=3)
        _,r=p.propose(iterations=30,chunk=5)
        assert r['violation_history'][-1][1]>.1
        assert r['auxiliary_failure_does_not_prove_infeasibility']
        assert 'accepted' not in r


def test_current_rhs_not_stale_template():
    import cupy as cp
    with GpuCondensedBatchedIPM(problems()) as s:
        p=GpuBlockAffineFeasibility(s,block_size=3)
        s.h[:,0]=1.25
        x,_=p.propose(cp.array([[1.5,1.5,1.]]),iterations=30,chunk=5)
        assert float(x[0,1])<=.2500001


@pytest.mark.parametrize('corrupt',['index','projection','equality','inverse','cost'])
def test_corruption_rejected_before_kernel(corrupt):
    with GpuCondensedBatchedIPM(problems()) as s:
        p=GpuBlockAffineFeasibility(s,block_size=3)
        if corrupt=='index':s.g.indices[0]=999999
        elif corrupt=='projection':p.p[0,0]+=.1
        elif corrupt=='equality':s.e.data[0]+=.1
        elif corrupt=='inverse':p.inverse[0,0]+=.1
        else:s.c[0,0]+=.1
        with pytest.raises(ValueError,match='static layout'):
            p.propose(iterations=1)


@pytest.mark.parametrize('options',[dict(block_size=0),dict(block_size=999),
    dict(block_size=2.5),dict(dual_sweeps=0),dict(dual_sweeps=True)])
def test_invalid_budgets(options):
    with GpuCondensedBatchedIPM(problems()) as s:
        with pytest.raises(ValueError):GpuBlockAffineFeasibility(s,**options)
