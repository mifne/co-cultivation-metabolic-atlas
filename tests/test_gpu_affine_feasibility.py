import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_affine_feasibility import GpuAffineFeasibility


def inputs(rhs=2.):
    # x0=x1, x1+x2 <= rhs; maximize x2 up to one.
    return [(csr_matrix([[1.,-1.,0.],[0.,1.,1.]]),np.array([0.,rhs]),
        np.zeros(3),np.array([2.,2.,1.]),np.array([0.,0.,-1.]),1)]


def test_cuda_affine_projection_proposes_feasible_box_optimum():
    import cupy as cp
    with GpuCondensedBatchedIPM(inputs()) as solver:
        projector=GpuAffineFeasibility(solver)
        x,report=projector.propose(cp.array([[1.5,1.5,1.]]),iterations=100,chunk=10)
        assert np.max(np.abs((solver.e@x.ravel()).get()-solver.b.get().ravel()))<1e-10
        assert x.get()[0,2]==pytest.approx(1.,abs=1e-10)
        assert np.max((solver.g@x.ravel()).get()-solver.h.get().ravel())<1e-6
        assert solver.factor.factor_count==0 and report['original_certificate_required']


def test_cuda_infeasible_auxiliary_does_not_emit_acceptance():
    with GpuCondensedBatchedIPM(inputs(.5)) as solver:
        projector=GpuAffineFeasibility(solver)
        _,report=projector.propose(iterations=20,chunk=10)
        assert report['violation_history'][-1][1]>.1
        assert 'accepted' not in report and report['auxiliary_failure_does_not_prove_infeasibility']


def test_cuda_projection_rejects_corrupt_index_before_execution():
    with GpuCondensedBatchedIPM(inputs()) as solver:
        projector=GpuAffineFeasibility(solver)
        solver.g.indices[0]=999999
        with pytest.raises(ValueError,match='static layout'):projector.propose(iterations=1)
