import numpy as np
import pytest
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_dr_feasibility import GpuDRFeasibility
from tests.test_gpu_dual_schur import inputs,options


def test_cuda_dr_proposal_has_one_factor_and_keeps_original_operator():
    import cupy as cp
    with GpuCondensedBatchedIPM(inputs(),**options()) as s:
        before=[v.copy() for v in (s.e.data,s.g.data,s.b,s.h,s.c)]
        projector=GpuDRFeasibility(s)
        x,report=projector.propose(iterations=200,chunk=10)
        assert report['factor_count']==1 and report['original_certificate_required']
        assert 'accepted' not in report
        assert float(cp.max(cp.abs((s.e@x.ravel()).reshape(s.batch,s.ne)-s.b)))<1e-6
        assert float(cp.max((s.g@x.ravel()).reshape(s.batch,s.ng)-s.h))<1e-6
        for old,new in zip(before,(s.e.data,s.g.data,s.b,s.h,s.c)):
            np.testing.assert_array_equal(old.get(),new.get())


def test_cuda_dr_auxiliary_failure_is_not_original_infeasibility():
    ps=inputs()
    ps=[(p[0],np.array([0.,.5]),*p[2:]) for p in ps]
    with GpuCondensedBatchedIPM(ps,**options()) as s:
        _,report=GpuDRFeasibility(s).propose(iterations=50,chunk=10)
        assert report['violation_history'][-1][1]>.1
        assert report['auxiliary_failure_does_not_prove_infeasibility']
        assert 'accepted' not in report


def test_cuda_dr_rejects_cost_change():
    with GpuCondensedBatchedIPM(inputs(),**options()) as s:
        projector=GpuDRFeasibility(s)
        s.c[0,0]=1.
        with pytest.raises(ValueError,match='Objective changed'):projector.propose(iterations=1)
