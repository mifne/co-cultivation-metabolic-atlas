import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.gpu_condensed_ipm import GpuCondensedBatchedIPM


def inputs():
    a=csr_matrix([[1.,-1.,0.],[0.,1.,1.]])
    return [(a,np.array([0.,bound]),np.zeros(3),np.array([2.,2.,1.]),np.array([0.,0.,-1.]),1)
            for bound in (2.,1.8)]


def options():
    return dict(newton_backend='dual_schur',globalized=True,newton_krylov_iterations=16,
        regularization=1e-6,device_checked_solves=True,predictor_corrector=True,
        predictor_affine_fraction=.995,ipm_initialization='balanced')


@pytest.mark.parametrize('layout',['uniform','block_diagonal'])
def test_cuda_schur_matches_current_augmented_linear_equations(layout):
    import cupy as cp
    with GpuCondensedBatchedIPM(inputs(),factor_layout=layout,**options()) as s:
        for scale in (1.,1.7):
            # Actual device coefficients, not saved host values, must enter M.
            for lane in range(s.batch):
                first=int(s.g.indptr[lane*s.ng]);last=int(s.g.indptr[lane*s.ng+s.q])
                s.g.data[first:last]*=scale
            s.gt=s.g.T.tocsr()
            if hasattr(s,'_condensed_h'):
                del s._condensed_h,s._condensed_ht
            ratio=cp.asarray(np.linspace(.5,2.,s.batch*s.ng).reshape(s.batch,s.ng))
            rhs=cp.asarray(np.linspace(-1.,2.,s.batch*s.size).reshape(s.batch,s.size))
            scaling=s._factor_newton(ratio)
            answer=s._solve_newton(rhs,ratio,scaling)
            # Check the complete bound-condensed K_delta, including all
            # recovered bound multipliers; bound G must stay +/- identity.
            packed=s._pack_rhs(rhs,ratio)
            residual=packed-s._condensed_mv(answer[:,:s.condensed_size],ratio,regularized=True)
            assert float(cp.max(cp.abs(residual)))<1e-8
            helper=s._dual_schur
            h=helper.h.get().toarray();d=helper.d.get().ravel()
            r=np.concatenate((np.full((s.batch,s.ne),s.regularization),ratio.get()[:,:s.q]),axis=1).ravel()
            expected=(h/d[None,:])@h.T+np.diag(r)
            native=helper.values.get()
            for lane in range(s.batch):
                block=helper.factor.host_pattern.copy();block.data[:]=native[lane]
                np.testing.assert_allclose(block.toarray(),expected[lane*helper.m:(lane+1)*helper.m,
                    lane*helper.m:(lane+1)*helper.m],rtol=1e-12,atol=1e-12)
        assert s.factor.factor_count==0 and s.numerical_factor.factor_count==2
        assert s.numerical_factor.n==s.ne+s.q


def test_cuda_schur_certifies_original_lp():
    with GpuCondensedBatchedIPM(inputs(),factor_layout='block_diagonal',**options()) as s:
        result=s.solve(iterations=100)
        assert result['accepted'].all()
        assert result['newton_backend']=='dual_schur'
        assert result['factored_dimension']==s.ne+s.q
        assert result['factor_count']==s.numerical_factor.factor_count>0
        factor=s.numerical_factor
    assert factor.closed


@pytest.mark.parametrize('kind',['recipe','g_index','dimension','numeric_buffer'])
def test_cuda_schur_corruption_rejected_before_numeric_factor(kind):
    import cupy as cp
    with GpuCondensedBatchedIPM(inputs(),**options()) as s:
        if kind=='recipe':s._dual_schur.left[0]=999999
        elif kind=='g_index':s.g.indices[0]=999999
        elif kind=='dimension':s._dual_schur.h._shape=(1,1)
        else:s._dual_schur.h.data=s._dual_schur.h.data.astype(cp.float32)
        with pytest.raises(ValueError,match='Schur'):s._factor_newton(cp.ones((s.batch,s.ng)))
        assert s.numerical_factor.factor_count==0


@pytest.mark.parametrize('change',[dict(globalized=False),dict(krylov_coordinates='condensed'),
    dict(factor_precision='float32'),dict(factor_ordering='amd')])
def test_schur_rejects_unsupported_compound_modes(change):
    opts=options();opts.update(change)
    with pytest.raises(ValueError,match='Dual Schur requires'):GpuCondensedBatchedIPM(inputs(),**opts)
