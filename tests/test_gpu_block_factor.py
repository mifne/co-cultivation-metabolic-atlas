import numpy as np
import pytest
from scipy.sparse import csr_matrix,block_diag
from src.gpu_sparse_factor import UniformCudssFactor,CudssError


@pytest.mark.parametrize('method',['solve','_solve_device_checked','_solve_device_checked_fused'])
def test_cuda_block_execution_preserves_independent_lanes_across_refactors(method):
    import cupy as cp
    base=np.array([[4.,1.,0.],[1.,3.,1.],[0.,1.,2.]])
    pattern=csr_matrix(base)
    with UniformCudssFactor(pattern,batch_size=4,execution_layout='block_diagonal',refinement_steps=0) as solver:
        expected=block_diag([pattern]*4,format='csr')
        np.testing.assert_array_equal(solver._native_indptr.get(),expected.indptr)
        np.testing.assert_array_equal(solver._native_indices.get(),expected.indices)
        for revision in (0,1):
            matrices=[base+(i+revision)*np.eye(3) for i in range(4)]
            solver.factor(cp.asarray(np.stack([csr_matrix(m).data for m in matrices])))
            rhs=np.arange(12,dtype=float).reshape(4,3,1)+revision
            result=getattr(solver,method)(cp.asarray(rhs)).get()
            expected=np.stack([np.linalg.solve(a,b) for a,b in zip(matrices,rhs)])
            np.testing.assert_allclose(result,expected,rtol=1e-13,atol=1e-13)
        assert solver.analysis_count==1 and solver.factor_count==2


def test_cuda_block_layout_change_rejected():
    with UniformCudssFactor(csr_matrix(np.eye(3)),batch_size=2,execution_layout='block_diagonal') as solver:
        solver.execution_layout='uniform'
        with pytest.raises(CudssError,match='layout changed'):solver._context()


def test_cuda_block_forest_numeric_updates_keep_current_original_certificates():
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_device_update import DeviceNumericUpdatePlan
    from src.gpu_ipm_numeric_update import NumericRebindRejected
    from tests.test_gpu_ipm_device_update import options
    from tests.test_gpu_ipm_numeric_update import _problems
    from scripts.probe_ipm_restart_mu_matched import _verify
    with ForestGpuBatchedIPM(_problems(),factor_layout='block_diagonal',**options()) as solver:
        plan=DeviceNumericUpdatePlan(solver)
        for revision in (1,2):
            current=_problems(revision)
            plan.rebind(current)
            result=solver.solve(iterations=120)
            assert _verify(current,result,2)['qualified']
        before=solver.c.get().copy()
        solver.factor._native_indices[0]=999999
        with pytest.raises(NumericRebindRejected):plan.rebind(_problems(3))
        np.testing.assert_array_equal(solver.c.get(),before)
