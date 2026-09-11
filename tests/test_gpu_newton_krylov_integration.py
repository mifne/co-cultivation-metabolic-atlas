import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from scripts.probe_downstream_gpu_coverage import paired_certificate


def example(rhs=1.,cost=(1.,2.)):
    return (csr_matrix([[1.,1.]]),np.array([rhs]),np.zeros(2),np.full(2,np.inf),
            np.array(cost),1)


def test_gpu_krylov_corrects_original_kkt_with_strong_preconditioner_regularization():
    import cupy as cp
    from src.gpu_newton_krylov import batched_gmres
    with GpuCondensedBatchedIPM([example(),example()],regularization=1e-3) as solver:
        ratio=cp.ones((2,solver.ng))
        scaling=solver._factor_newton(ratio)
        exact=cp.asarray([[1.,2.,3.,4.,5.],[0.,0.,0.,0.,0.]])
        rhs=solver._kkt_mv(exact,ratio,regularized=False)
        initial=solver._solve_newton(rhs,ratio,scaling)
        assert float(solver._direction_error(rhs,rhs-solver._kkt_mv(initial,ratio,regularized=False))[0])>1e-8
        answer,diagnostic=batched_gmres(rhs,initial,
            lambda v:solver._kkt_mv(v,ratio,regularized=False),
            lambda v:solver._solve_newton(v,ratio,scaling),
            solver._direction_error,xp=cp,max_iterations=12,tolerance=1e-8)
        assert bool(diagnostic['converged'].all())
        assert int(diagnostic['iterations'][1])==0
        np.testing.assert_allclose(answer.get(),exact.get(),atol=1e-7)


def test_original_newton_gpu_krylov_lp_solve_is_certified_without_cpu_optimizer():
    problems=[example(),example(2.,(3.,1.))]
    with GpuCondensedBatchedIPM(problems,regularization=1e-3,
            original_newton_target=True,newton_refinements=0,newton_krylov_iterations=12) as solver:
        result=solver.solve(iterations=35)
        assert result['accepted'].all(),result['metrics']
        assert np.sum(result['krylov_iterations_by_direction'])>0
        assert result['cpu_lp_calls']==0 and result['analysis_count']==1
        for p,x,y in zip(problems,result['x'].get(),result['y'].get()):
            assert paired_certificate(p,x,y)['certificate_passed']


def test_krylov_never_implicitly_changes_newton_target():
    with pytest.raises(ValueError,match='explicitly target original'):
        GpuCondensedBatchedIPM([example()],newton_krylov_iterations=12)


def test_scaled_krylov_claim_cannot_replace_better_original_equation_direction(monkeypatch):
    import cupy as cp
    from src import gpu_newton_krylov
    with GpuCondensedBatchedIPM([example()],regularization=1e-3,
            original_newton_target=True,newton_refinements=0) as solver:
        baseline=solver.solve(iterations=1,capture_failure=True)
        assert baseline['status']=='unreliable_affine_direction'
        expected=solver.failure_snapshot['answer'].copy()
        def misleading(rhs,initial,*args,**kwargs):
            return initial+1e6,dict(iterations=cp.ones(1,dtype=cp.int32),
                                    termination=cp.zeros(1,dtype=cp.int32))
        monkeypatch.setattr(gpu_newton_krylov,'batched_gmres',misleading)
        solver.newton_krylov_iterations=12
        result=solver.solve(iterations=1,capture_failure=True)
        assert result['status']=='unreliable_affine_direction'
        np.testing.assert_array_equal(solver.failure_snapshot['answer'].get(),expected.get())
