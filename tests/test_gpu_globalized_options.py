"""Integration gates for optional initialization and condensed corrections."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_batched_ipm import GpuBatchedIPM
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem():
    return (csr_matrix([[1.,1.]]),np.array([1.]),np.zeros(2),
            np.full(2,1e6),np.array([1.,2.]),1)


@pytest.mark.parametrize('coordinates',['full','condensed'])
@pytest.mark.parametrize('pc',[False,True])
def test_cuda_balanced_initialization_and_both_correction_paths_keep_original_gate(coordinates,pc):
    p=problem()
    with GpuCondensedBatchedIPM([p],globalized=True,ipm_initialization='balanced',
            newton_krylov_iterations=8,krylov_coordinates=coordinates,
            predictor_corrector=pc,predictor_affine_fraction=.995 if pc else 1.) as solver:
        r=solver.solve(iterations=100)
        assert r['accepted'].all(),r['metrics']
        assert r['cpu_lp_calls']==0
        assert r['ipm_initialization']=='balanced'
        assert r['krylov_width']==(solver.condensed_size if coordinates=='condensed' else solver.size)
        # Lower-bound c=2 retains its positive dual proposal; large upper
        # bounds contribute unit, not million-sized, complementarity.
        assert r['initialization_diagnostics'][0][0]==pytest.approx(1.25)
        assert paired_certificate(p,r['x'].get()[0],r['y'].get()[0])['certificate_passed']


def test_cuda_balanced_initialization_preserves_certified_warm_pair_without_newton():
    import cupy as cp
    x,y=cp.array([[1.,0.]]),cp.array([[1.]])
    with GpuCondensedBatchedIPM([problem()],globalized=True,ipm_initialization='balanced') as solver:
        r=solver.solve(initial_x=x,initial_y=y,iterations=4)
        assert r['accepted'].all() and r['factor_count']==0
        np.testing.assert_array_equal(r['x'].get(),x.get())
        np.testing.assert_array_equal(r['y'].get(),y.get())


def test_cuda_condensed_proxy_success_cannot_bypass_actual_full_forcing(monkeypatch):
    import cupy as cp
    import src.gpu_globalized_condensed as helper
    with GpuCondensedBatchedIPM([problem()],globalized=True,
            newton_krylov_iterations=8,krylov_coordinates='condensed') as solver:
        calls=[]
        monkeypatch.setattr(solver,'_solve_newton',lambda rhs,ratio,scaling:cp.zeros_like(rhs))
        def false_success(solver,rhs,answer,ratio,scaling,requested,weights,target,eta,*,z):
            calls.append(True)
            return cp.zeros_like(rhs),dict(iterations=cp.ones(1,dtype=cp.int32),
                termination=cp.zeros(1,dtype=cp.int32)),cp.zeros(1,dtype=cp.bool_)
        monkeypatch.setattr(helper,'globalized_condensed_gmres',false_success)
        r=solver.solve(iterations=3)
        assert calls and not r['accepted'].any()
        assert r['failed_environments']==[2]
        np.testing.assert_array_equal(r['x'].get(),np.zeros((1,2)))


@pytest.mark.parametrize('kw',[dict(ipm_initialization='balanced'),
    dict(predictor_affine_fraction=.995),dict(predictor_affine_fraction=True),
    dict(globalized=True,ipm_initialization='unknown'),
    dict(globalized=True,krylov_coordinates='condensed')])
def test_incompatible_options_are_explicitly_rejected(kw):
    with pytest.raises(ValueError):
        GpuBatchedIPM([problem()],**kw)
