"""CUDA regression: each factor must use the current preconditioner delta."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_batched_ipm import GpuBatchedIPM
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.lp_trace import problem_hash


@pytest.mark.parametrize('solver_type',[GpuBatchedIPM,GpuCondensedBatchedIPM])
def test_refactor_refreshes_both_regularization_blocks_without_changing_lp(solver_type):
    import cupy as cp
    problem=(csr_matrix([[1.,1.,0.],[0.,1.,1.]]),np.array([2.,3.]),
        np.array([0.,0.,-np.inf]),np.array([10.,np.inf,4.]),np.array([1.,2.,-.5]),1)
    original_hash=problem_hash(problem)
    with solver_type([problem]*2,regularization=1e-4,
                     original_newton_target=True) as solver:
        rng=np.random.default_rng(20260906)
        expected=cp.asarray(rng.normal(size=(2,solver.size)))
        ratios=cp.asarray(np.exp(rng.normal(size=(2,solver.ng))))
        for index,delta in enumerate((1e-4,1e-7)):
            solver.regularization=delta
            ratio=ratios*(1.+index*.3)
            # RHS is manufactured using the actual current K_delta target,
            # not assembled from a possibly stale stored factor diagonal.
            rhs=solver._kkt_mv(expected,ratio,regularized=True)
            scaling=solver._factor_newton(ratio)
            actual=solver._solve_newton(rhs,ratio,scaling)
            residual=rhs-solver._kkt_mv(actual,ratio,regularized=True)
            assert float(cp.max(cp.abs(residual)))<1e-10
            np.testing.assert_allclose(actual.get(),expected.get(),rtol=1e-8,atol=1e-8)
            np.testing.assert_array_equal(
                solver.values[:,solver.diagonal[solver.n:solver.n+solver.ne]].get(),
                np.full((2,solver.ne),-delta))
            diagonal=cp.full((2,solver.n),delta,dtype=cp.float64)
            if solver._condense_bounds:
                diagonal[:,solver.il]+=1./ratio[:,solver.q:solver.q+len(solver.il)]
                diagonal[:,solver.iu]+=1./ratio[:,solver.q+len(solver.il):]
            np.testing.assert_array_equal(
                solver.values[:,solver.diagonal[:solver.n]].get(),diagonal.get())
            # Regularized linear accuracy is not an original K_0 certificate.
            assert float(cp.max(cp.abs(rhs-solver._kkt_mv(actual,ratio,regularized=False))))>1e-9
            assert problem_hash(problem)==original_hash
            assert solver.problem_hashes==(original_hash,original_hash)
        assert solver.factor.factor_count==2
        assert solver.factor.analysis_count==1
