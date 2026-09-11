"""CUDA fixed-layout updates: current numeric equality, rejection and solve."""
import numpy as np
import pytest
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from src.gpu_ipm_device_update import DeviceNumericUpdatePlan
from src.gpu_ipm_numeric_update import prepare_host_rebind, rebind_forest_ipm, NumericRebindRejected
from src.lp_trace import problem_hash
from tests.test_gpu_ipm_numeric_update import _problems


def options(second=True):
    return dict(globalized=True,forcing_eta=.1,regularization=1e-6,
        newton_krylov_iterations=8,predictor_corrector=True,predictor_affine_fraction=.995,
        ipm_initialization='balanced',exact_equalities=True,second_forest=second,
        allow_box_dual=True,device_checked_solves=True,factor_refinements=0)


@pytest.mark.parametrize('second',[False,True])
def test_cuda_pipeline_and_commit_match_current_host_reductions(second):
    import cupy as cp
    with ForestGpuBatchedIPM(_problems(),**options(second)) as solver:
        plan=DeviceNumericUpdatePlan(solver)
        for revision in (1,2):
            current=_problems(revision)
            # The reference is separate and never supplied to the GPU path.
            with ForestGpuBatchedIPM(current,**options(second)) as reference:
                reference_plan=DeviceNumericUpdatePlan(reference)
                full,packed,_=plan.pack_current_host(current)
                batches,first,other,flags,_=plan.pipeline.stage(packed)
                assert bool(cp.all(cp.stack(flags)))
                hosts=dict(full=reference.full_problems,forest=reference.forest_problems,
                    face=[p.reduced for p in reference.face_plans],
                    secondary=[r.problem for r in reference.secondary_forest_reductions]
                        if second else [p.reduced for p in reference.face_plans],core=reference.problems)
                for name,problems in hosts.items():
                    expected=(np.concatenate([p[0].data for p in problems]),
                              *(np.stack([p[i] for p in problems]) for i in (1,2,3,4)))
                    for actual,wanted in zip(batches[name].arrays(),expected):
                        np.testing.assert_allclose(actual.get(),wanted,rtol=2e-15,atol=1e-14)
                update=plan.rebind(current)
                assert update['generation']==revision and update['cpu_lp_calls']==0
                assert tuple(solver.problem_hashes)==tuple(problem_hash(p) for p in current)
                assert not solver.factor.factored and solver._last_internal_state is None
                for path,target in plan.binder.numeric_targets:
                    if path in [('values',),('factor','values')]:continue
                    from src.gpu_ipm_numeric_update import _target
                    np.testing.assert_allclose(target.get(),_target(reference,path).get(),rtol=2e-15,atol=1e-14)
        with pytest.raises(NumericRebindRejected,match='layout-only'):
            rebind_forest_ipm(solver,_problems(3))


@pytest.mark.parametrize('change',['equality','eq_rhs','support','zero_sign','fixed','nan','static_index','numeric'])
def test_cuda_device_update_rejection_preserves_current_solver(change):
    with ForestGpuBatchedIPM(_problems(),**options()) as solver:
        plan=DeviceNumericUpdatePlan(solver)
        current=_problems(1)
        if change=='equality':current[0][0].data[0]+=.1
        elif change=='eq_rhs':current[0][1][0]=1.
        elif change=='support':current[0][0].indices[-1]-=1
        elif change=='zero_sign':current[0][2][5]=.1
        elif change=='fixed':current[0][3][0]=current[0][2][0]
        elif change=='nan':current[0][4][0]=np.nan
        elif change=='static_index':plan.pipeline.first.linear.static_targets[1][1][0]=999999
        elif change=='numeric':solver.c[0,0]+=1.
        before=solver.c.get().copy();hashes=solver.problem_hashes
        with pytest.raises(NumericRebindRejected):plan.rebind(current)
        np.testing.assert_array_equal(solver.c.get(),before)
        assert solver.problem_hashes==hashes and not solver.factor.failed


def test_cuda_device_updated_lp_solves_and_certifies_current_original_inputs():
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    with ForestGpuBatchedIPM(_problems(),**options()) as solver:
        plan=DeviceNumericUpdatePlan(solver)
        for revision in (1,2):
            current=_problems(revision)
            plan.rebind(current)
            result=solver.solve(iterations=120)
            assert result['accepted'].all(), result['metrics']
            assert result['device_numeric_update']['generation']==revision
            assert all(paired_certificate(p,x,y)['certificate_passed']
                       for p,x,y in zip(current,result['x'].get(),result['y'].get()))


@pytest.mark.parametrize('initial_coefficient,current_coefficient', [(1., 2.), (2., 1.)])
def test_cuda_rejects_cancelled_forest_group_support_changes(initial_coefficient,current_coefficient):
    def inputs(coefficient):
        result=[]
        for a,rhs,lo,hi,c,neq in _problems():
            a=a.tolil()
            # x0=x1: this group disappears exactly when coefficient is one.
            # Original CSR support is identical in both versions.
            a[4,0]=coefficient; a[4,1]=-1.
            result.append((a.tocsr(),rhs,lo,hi,c,neq))
        return result
    with ForestGpuBatchedIPM(inputs(initial_coefficient),**options()) as solver:
        plan=DeviceNumericUpdatePlan(solver)
        before=solver.c.get().copy(); hashes=solver.problem_hashes
        with pytest.raises(NumericRebindRejected):
            plan.rebind(inputs(current_coefficient))
        np.testing.assert_array_equal(solver.c.get(),before)
        assert solver.problem_hashes==hashes and not solver.factor.failed


def test_cuda_nonzero_fixed_substitution_and_removed_zero_row_are_current():
    import cupy as cp
    from scipy.sparse import csr_matrix
    from src.lp_zero_face import ZeroFaceReduction
    from src.gpu_ipm_device_update import _Layout, _RowColumn, _pack
    a=csr_matrix([[1.,1.,0.],[-1.,-1.,0.],[4.,0.,2.],[1.,0.,0.]])
    original=(a,np.array([5.,-5.,20.,3.]),np.array([3.,0.,0.]),
              np.array([3.,10.,10.]),np.array([7.,1.,2.]),2)
    face=ZeroFaceReduction(original)
    transform=_RowColumn(_Layout([original],cp),_Layout([face.reduced],cp),
        [face.rows],[face.columns],cp,face_plans=[face])
    changed=[v.copy() if hasattr(v,'copy') else v for v in original]
    changed[0].data[changed[0].indptr[2]]=5.
    changed[1][2]=23.; changed[4][0]=11.
    current=tuple(changed)
    result,flags,offsets=transform.stage(_pack([current],cp))
    reference=ZeroFaceReduction(current)
    assert bool(cp.all(cp.stack(flags)))
    np.testing.assert_array_equal(result.rhs.get()[0],reference.reduced[1])
    assert 8. in result.rhs.get()[0]  # 23 - 5 * 3, not the old coefficient.
    assert offsets.get().tolist()==[33.]
    changed[1][3]=np.nextafter(3.,-np.inf)
    _,flags,_=transform.stage(_pack([tuple(changed)],cp))
    assert not bool(cp.all(cp.stack(flags)))
    changed[1][3]=4.
    _,flags,_=transform.stage(_pack([tuple(changed)],cp))
    assert bool(cp.all(cp.stack(flags)))
