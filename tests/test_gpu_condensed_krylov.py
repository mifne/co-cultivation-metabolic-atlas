"""Condensation algebra uses NumPy; the final smoke test requires CUDA.

The NumPy factor below is only a deterministic test double. The production
condensed implementation always uses its existing GPU cuDSS factor.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import block_diag,csr_matrix

from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
from src.gpu_newton_krylov import GMRES_ITERATION_LIMIT


class _DenseTestFactor:
    def __init__(self,matrices):
        self.matrices=matrices
        self.n=matrices.shape[1]
        self.calls=0
        self.wrong_context=False

    def _context(self):
        if self.wrong_context:
            raise RuntimeError('wrong test context')

    def _array(self,value,shape):
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float64
                or value.shape!=shape or not np.isfinite(value).all()):
            raise ValueError('Finite FP64 input with exact shape required')
        return value

    def solve(self,rhs):
        self.calls+=1
        return np.linalg.solve(self.matrices,rhs)


def _workspace(*,q=1,ne=1):
    solver=object.__new__(GpuCondensedBatchedIPM)
    solver.cp=np
    solver.batch,solver.n,solver.ne,solver.q=2,3,ne,q
    solver.il,solver.iu=np.array([0,1]),np.array([0,2])
    solver.ng=q+4
    solver.size=solver.n+ne+solver.ng
    solver.condensed_size=solver.n+ne+q
    solver.regularization=.1
    solver.original_newton_target=True
    solver.krylov_coordinates='condensed'
    solver.newton_krylov_iterations=16
    solver.newton_relative_tolerance=1e-11
    solver.closed=False
    es=[np.array([[1.,1.,0.]])[:ne],np.array([[1.,2.,0.]])[:ne]]
    hs=[np.array([[0.,1.,1.]])[:q],np.array([[1.,0.,2.]])[:q]]
    bound=np.vstack((-np.eye(3)[solver.il],np.eye(3)[solver.iu]))
    gs=[np.vstack((h,bound)) for h in hs]
    solver.e=block_diag(es,format='csr')
    solver.et=solver.e.T.tocsr()
    solver.g=block_diag(gs,format='csr')
    solver.gt=solver.g.T.tocsr()
    rng=np.random.default_rng(71)
    ratio=np.exp(rng.normal(size=(2,solver.ng)))
    full=[]
    condensed=[]
    for e,h,g,d in zip(es,hs,gs,ratio):
        full.append(np.block([[np.zeros((3,3)),e.T,g.T],
            [e,np.zeros((ne,ne)),np.zeros((ne,solver.ng))],
            [g,np.zeros((solver.ng,ne)),-np.diag(d)]]))
        condensed.append(np.block([[bound.T@np.diag(1./d[q:])@bound,e.T,h.T],
            [e,np.zeros((ne,ne)),np.zeros((ne,q))],
            [h,np.zeros((q,ne)),-np.diag(d[:q])]]))
    condensed=np.stack(condensed)
    regularized=condensed.copy()
    regularized[:,:3,:3]+=.1*np.eye(3)
    regularized[:,3:3+ne,3:3+ne]-=.1*np.eye(ne)
    solver.factor=_DenseTestFactor(regularized)
    return solver,ratio,np.stack(full),condensed


@pytest.mark.parametrize('q,ne',[(1,1),(0,1),(1,0),(0,0)])
def test_condensed_operator_and_rhs_match_full_newton_algebra(q,ne):
    solver,ratio,full,condensed=_workspace(q=q,ne=ne)
    rng=np.random.default_rng(11)
    reduced=rng.normal(size=(2,solver.condensed_size))
    rhs=rng.normal(size=(2,solver.size))
    expanded=solver._expand_direction(reduced,rhs,ratio)
    direct=np.einsum('bij,bj->bi',condensed,reduced)
    np.testing.assert_allclose(solver._condensed_mv(reduced,ratio),direct,atol=2e-14)
    np.testing.assert_allclose(
        (rhs-np.einsum('bij,bj->bi',full,expanded))[:,:solver.condensed_size],
        solver._pack_rhs(rhs,ratio)-direct,atol=2e-14)
    np.testing.assert_allclose(
        (rhs-np.einsum('bij,bj->bi',full,expanded))[:,solver.condensed_size:],0.,atol=2e-14)
    np.testing.assert_allclose(solver._condensed_mv(reduced,ratio,regularized=True),
        np.einsum('bij,bj->bi',solver.factor.matrices,reduced),atol=2e-14)


@pytest.mark.parametrize('q,ne',[(1,1),(0,1),(1,0),(0,0)])
def test_condensed_fgmres_targets_unregularized_full_system(q,ne):
    solver,ratio,full,_=_workspace(q=q,ne=ne)
    rhs=np.random.default_rng(16).normal(size=(2,solver.size))
    scaling=np.ones((2,solver.condensed_size))
    initial=solver._solve_newton(rhs,ratio,scaling)
    previous=solver._direction_error(rhs,rhs-np.einsum('bij,bj->bi',full,initial))
    assert previous.min()>1e-3
    answer,diagnostic=solver._krylov_direction(rhs,initial,ratio,scaling,np.ones(2,dtype=bool))
    error=solver._direction_error(rhs,rhs-np.einsum('bij,bj->bi',full,answer))
    assert (error<solver.newton_relative_tolerance).all()
    assert diagnostic['converged'].all(),diagnostic
    np.testing.assert_allclose(diagnostic['error'],error,atol=2e-14)
    np.testing.assert_allclose(answer,np.linalg.solve(full,rhs[...,None])[...,0],atol=2e-11)
    assert (diagnostic['iterations']<=solver.condensed_size).all()


def test_inactive_environment_is_preserved_and_not_iterated():
    solver,ratio,_,_=_workspace()
    rhs=np.ones((2,solver.size))
    scaling=np.ones((2,solver.condensed_size))
    initial=solver._solve_newton(rhs,ratio,scaling)
    answer,diagnostic=solver._krylov_direction(rhs,initial,ratio,scaling,np.array([True,False]))
    np.testing.assert_array_equal(answer[1],initial[1])
    assert diagnostic['iterations'][1]==0
    assert diagnostic['iterations'][0]>0


def test_arnoldi_never_reconstructs_bound_duals_and_final_gate_checks_them():
    solver,ratio,_,_=_workspace()
    rhs=np.ones((2,solver.size))
    scaling=np.ones((2,solver.condensed_size))
    initial=solver._solve_newton(rhs,ratio,scaling)
    original=solver._expand_direction
    counts=SimpleNamespace(expand=0,full=0)
    original_full=solver._kkt_mv

    def corrupted_expansion(*args):
        counts.expand+=1
        value=original(*args)
        value[:,solver.condensed_size:]+=1.
        return value

    def full_check(*args,**kwargs):
        counts.full+=1
        return original_full(*args,**kwargs)

    solver._expand_direction=corrupted_expansion
    solver._kkt_mv=full_check
    answer,diagnostic=solver._krylov_direction(rhs,initial,ratio,scaling,np.ones(2,dtype=bool))
    assert counts.expand==1
    assert counts.full==2
    assert diagnostic['condensed_converged'].all()
    assert not diagnostic['converged'].any()
    np.testing.assert_array_equal(diagnostic['termination'],GMRES_ITERATION_LIMIT)
    np.testing.assert_array_equal(answer,initial)


@pytest.mark.parametrize('invalid',[
    'rhs_shape','rhs_dtype','rhs_nan','answer_inf','ratio_zero','ratio_negative',
    'ratio_inf','scaling_zero','scaling_shape','mask_shape','mask_dtype','context'])
def test_rejects_invalid_inputs_before_arnoldi(invalid):
    solver,ratio,_,_=_workspace()
    rhs=np.ones((2,solver.size))
    answer=np.zeros_like(rhs)
    scaling=np.ones((2,solver.condensed_size))
    need=np.ones(2,dtype=bool)
    if invalid=='rhs_shape':rhs=rhs[:,:-1]
    elif invalid=='rhs_dtype':rhs=rhs.astype(np.float32)
    elif invalid=='rhs_nan':rhs[0,0]=np.nan
    elif invalid=='answer_inf':answer[0,0]=np.inf
    elif invalid=='ratio_zero':ratio[0,0]=0.
    elif invalid=='ratio_negative':ratio[0,0]=-1.
    elif invalid=='ratio_inf':ratio[0,0]=np.inf
    elif invalid=='scaling_zero':scaling[0,0]=0.
    elif invalid=='scaling_shape':scaling=scaling[:,:-1]
    elif invalid=='mask_shape':need=need[:,None]
    elif invalid=='mask_dtype':need=need.astype(np.int32)
    elif invalid=='context':solver.factor.wrong_context=True
    with pytest.raises((ValueError,RuntimeError)):
        solver._krylov_direction(rhs,answer,ratio,scaling,need)
    assert solver.factor.calls==0


def test_full_mode_delegates_without_changing_base_algorithm(monkeypatch):
    from src.gpu_batched_ipm import GpuBatchedIPM
    solver,_,_,_=_workspace()
    solver.krylov_coordinates='full'
    sentinel=(object(),{'full_mode':True})
    monkeypatch.setattr(GpuBatchedIPM,'_krylov_direction',lambda *_:sentinel,raising=False)
    assert solver._krylov_direction(None,None,None,None,None) is sentinel


def test_cuda_condensed_fgmres_lp_preserves_original_certificate():
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    p=(csr_matrix([[1.,1.,0.],[0.,1.,1.]]),np.array([2.,3.]),
       np.array([0.,0.,-np.inf]),np.array([10.,np.inf,4.]),np.array([1.,2.,-.5]),1)
    with GpuCondensedBatchedIPM([p]*2,regularization=1e-4,
            original_newton_target=True,newton_krylov_iterations=12,
            krylov_coordinates='condensed') as solver:
        result=solver.solve(iterations=60)
        assert result['accepted'].all(),result['metrics']
        assert result['cpu_lp_calls']==0
        assert result['analysis_count']==1
        assert result['factored_dimension']<result['kkt_dimension']
        for x,y in zip(result['x'].get(),result['y'].get()):
            assert paired_certificate(p,x,y)['certificate_passed']
