"""Globalized condensed arithmetic tests; one optional root-run CUDA smoke."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_globalized_condensed import globalized_condensed_gmres
from src.gpu_globalized_ipm import weighted_blocks,relative_forcing
from tests.test_gpu_condensed_krylov import _workspace


def fixture(*,q=1,ne=1):
    solver,ratio,_,_=_workspace(q=q,ne=ne)
    rng=np.random.default_rng(81)
    rhs=rng.normal(size=(2,solver.size))
    z=np.exp(rng.normal(size=(2,solver.ng)))
    rg=rng.normal(size=(2,solver.ng))
    rd=-rhs[:,:solver.n]
    rp=-rhs[:,solver.n:solver.n+solver.ne]
    rc=z*(rhs[:,solver.n+solver.ne:]+rg)
    weights=tuple(np.array([.2+i*.2,.3+i*.15]) for i in range(4))
    target=weighted_blocks((rd,rp,rg,rc),weights,xp=np)
    scaling=np.ones((2,solver.condensed_size))
    answer=solver._solve_newton(rhs,ratio,scaling)
    return solver,rhs,answer,ratio,scaling,weights,target,z,rg,rc


def actual_forcing(solver,rhs,answer,ratio,weights,target,z,rg,rc):
    dx=answer[:,:solver.n]
    dy=answer[:,solver.n:solver.n+solver.ne]
    dz=answer[:,solver.n+solver.ne:]
    gdx=solver._mv(solver.g,dx,solver.ng)
    ds=-rg-gdx
    defect=weighted_blocks((
        -rhs[:,:solver.n]+solver._mv(solver.et,dy,solver.n)+solver._mv(solver.gt,dz,solver.n),
        -rhs[:,solver.n:solver.n+solver.ne]+solver._mv(solver.e,dx,solver.ne),
        rg+gdx+ds,rc+z*ds+(z*ratio)*dz),weights,xp=np)
    return relative_forcing(defect,target,xp=np)


@pytest.mark.parametrize('q,ne',[(1,1),(0,1),(1,0),(0,0)])
def test_condensed_proxy_matches_actual_full_four_block_forcing(q,ne):
    solver,rhs,answer,ratio,scaling,weights,target,z,rg,rc=fixture(q=q,ne=ne)
    eta=1e-10
    improved,diagnostic,invalid=globalized_condensed_gmres(solver,rhs,answer,ratio,
        scaling,np.ones(2,dtype=bool),weights,target,eta,z=z)
    forcing=actual_forcing(solver,rhs,improved,ratio,weights,target,z,rg,rc)
    assert not invalid.any()
    assert diagnostic['converged'].all()
    assert (forcing<=eta).all()
    np.testing.assert_allclose(diagnostic['error'],forcing,atol=2e-14)
    np.testing.assert_allclose(diagnostic['full_target_norm'],np.linalg.norm(target,axis=1),rtol=1e-15)
    assert diagnostic['krylov_width']==solver.condensed_size
    assert diagnostic['full_newton_width']==solver.size
    assert diagnostic['full_target_width']==target.shape[1]
    assert (diagnostic['iterations']<=solver.condensed_size).all()


def test_denominator_remains_full_target_not_magnified_packed_rhs():
    solver,rhs,answer,ratio,scaling,weights,target,z,rg,rc=fixture()
    ratio[:,solver.q:]=1e-8
    solver.newton_krylov_iterations=0
    answer=np.zeros_like(answer)
    candidate,diagnostic,_=globalized_condensed_gmres(solver,rhs,answer,ratio,
        scaling,np.ones(2,dtype=bool),weights,target,.1,z=z)
    packed=solver._pack_rhs(rhs,ratio)
    wc=np.concatenate((np.broadcast_to(weights[0][:,None],(2,solver.n)),
        np.broadcast_to(weights[1][:,None],(2,solver.ne)),weights[3][:,None]*z[:,:solver.q]),axis=1)
    expected=np.linalg.norm(wc*packed,axis=1)/np.linalg.norm(target,axis=1)
    np.testing.assert_allclose(diagnostic['error'],expected,rtol=2e-15)
    assert (diagnostic['error']>1e6).all()
    assert solver.factor.calls==1  # Fixture's initial solve only.
    np.testing.assert_allclose(actual_forcing(solver,rhs,candidate,ratio,weights,target,z,rg,rc),
                               diagnostic['error'],rtol=2e-15)


def test_arnoldi_never_expands_bounds_and_condensed_convergence_is_not_final_acceptance():
    solver,rhs,answer,ratio,scaling,weights,target,z,rg,rc=fixture()
    original=solver._expand_direction
    count=0
    def corrupt_final(*args):
        nonlocal count
        count+=1
        value=original(*args)
        value[:,solver.condensed_size:]+=1.
        return value
    def forbidden_full_operator(*args,**kwargs):
        raise AssertionError('Full Newton matvec must not be used inside condensed Arnoldi')
    solver._expand_direction=corrupt_final
    solver._kkt_mv=forbidden_full_operator
    candidate,diagnostic,invalid=globalized_condensed_gmres(solver,rhs,answer,ratio,
        scaling,np.ones(2,dtype=bool),weights,target,1e-10,z=z)
    assert count==1 and not invalid.any()
    assert diagnostic['converged'].all()
    # This is exactly why the caller's products()/actual forcing gate remains
    # authoritative rather than trusting the condensed convergence flag.
    assert (actual_forcing(solver,rhs,candidate,ratio,weights,target,z,rg,rc)>1e-10).all()


@pytest.mark.parametrize('invalid',['weight_nan','weight_zero','z_zero','z_inf','target_nan'])
def test_invalid_requested_lane_is_reported_without_poisoning_other_environment(invalid):
    solver,rhs,answer,ratio,scaling,weights,target,z,rg,rc=fixture()
    if invalid=='weight_nan':weights[3][0]=np.nan
    elif invalid=='weight_zero':weights[2][0]=0.
    elif invalid=='z_zero':z[0,0]=0.
    elif invalid=='z_inf':z[0,0]=np.inf
    else:target[0,0]=np.nan
    candidate,diagnostic,bad=globalized_condensed_gmres(solver,rhs,answer,ratio,
        scaling,np.ones(2,dtype=bool),weights,target,1e-10,z=z)
    assert bad.tolist()==[True,False]
    np.testing.assert_array_equal(candidate[0],answer[0])
    assert diagnostic['iterations'][0]==0
    assert not diagnostic['converged'][0]
    assert diagnostic['converged'][1]


def test_inactive_lane_keeps_original_direction_and_does_not_require_finite_target_weights():
    solver,rhs,answer,ratio,scaling,weights,target,z,rg,rc=fixture()
    weights[3][1]=np.nan
    z[1]=np.nan
    target[1]=np.nan
    candidate,diagnostic,bad=globalized_condensed_gmres(solver,rhs,answer,ratio,
        scaling,np.array([True,False]),weights,target,1e-10,z=z)
    assert not bad.any()
    np.testing.assert_array_equal(candidate[1],answer[1])
    assert diagnostic['iterations'][1]==0
    assert diagnostic['converged'][0]


@pytest.mark.parametrize('bad',['shape','dtype','rhs_nan','answer_nan','ratio_zero','scaling_zero',
    'mask','target_shape','z_shape','weights_length','context','not_condensed','regularized_target'])
def test_bad_interface_inputs_are_rejected(bad):
    solver,rhs,answer,ratio,scaling,weights,target,z,rg,rc=fixture()
    requested=np.ones(2,dtype=bool)
    if bad=='shape':rhs=rhs[:,:-1]
    elif bad=='dtype':answer=answer.astype(np.float32)
    elif bad=='rhs_nan':rhs[0,0]=np.nan
    elif bad=='answer_nan':answer[0,0]=np.nan
    elif bad=='ratio_zero':ratio[0,0]=0.
    elif bad=='scaling_zero':scaling[0,0]=0.
    elif bad=='mask':requested=requested.astype(np.int32)
    elif bad=='target_shape':target=target[:,:-1]
    elif bad=='z_shape':z=z[:,:-1]
    elif bad=='weights_length':weights=weights[:3]
    elif bad=='context':solver.factor.wrong_context=True
    elif bad=='not_condensed':solver._condense_bounds=False
    else:solver.original_newton_target=False
    calls=solver.factor.calls
    with pytest.raises((ValueError,RuntimeError)):
        globalized_condensed_gmres(solver,rhs,answer,ratio,scaling,requested,weights,target,.1,z=z)
    assert solver.factor.calls==calls


def test_cuda_globalized_condensed_lp_returns_original_certified_pair():
    from src.gpu_condensed_ipm import GpuCondensedBatchedIPM
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    p=(csr_matrix([[1.,1.,0.],[0.,1.,1.]]),np.array([2.,3.]),
       np.array([0.,0.,-np.inf]),np.array([10.,np.inf,4.]),np.array([1.,2.,-.5]),1)
    with GpuCondensedBatchedIPM([p]*2,regularization=1e-6,globalized=True,
            forcing_eta=.1,original_newton_target=True,newton_krylov_iterations=16,
            krylov_coordinates='condensed') as solver:
        result=solver.solve(iterations=120)
        assert result['accepted'].all(),result['metrics']
        assert result['cpu_lp_calls']==0
        for x,y in zip(result['x'].get(),result['y'].get()):
            assert paired_certificate(p,x,y)['certificate_passed']
