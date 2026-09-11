"""Analytic box duals must materialize as independently valid full LP pairs."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_forest_ipm import ForestGpuBatchedIPM
from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from src.lp_trace import problem_hash
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem(*,row_cap=2.,box_cap=1.):
    return (csr_matrix(np.eye(2)),np.array([row_cap,1.]),np.zeros(2),
            np.array([box_cap,2.]),np.array([-1.,0.]),0)


def test_cpu_option_requires_explicit_boolean_before_gpu_allocation(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No GPU initialization expected')
    monkeypatch.setattr(ZeroFaceGpuBatchedIPM,'__init__',forbidden)
    for value in (1,'true',None,np.bool_(True)):
        with pytest.raises(ValueError,match='boolean'):
            ForestGpuBatchedIPM([problem()],allow_box_dual=value)


def test_gpu_box_certificate_returns_zero_full_dual_and_unchanged_original_lp():
    import cupy as cp
    p=problem()
    digest=problem_hash(p)
    x=cp.asarray([[1.,0.]],dtype=cp.float64)
    bad_y=cp.asarray([[-1.,0.]],dtype=cp.float64)
    assert not paired_certificate(p,x[0].get(),bad_y[0].get())['certificate_passed']
    with ForestGpuBatchedIPM([p],allow_box_dual=True) as solver:
        result=solver.solve(initial_x=x,initial_y=bad_y,iterations=0)
        assert result['accepted'].tolist()==[True]
        assert result['accepted_iteration'].tolist()==[0]
        assert result['metrics'][0]['certificate_source']=='analytic_box_bound'
        assert result['analytic_box_dual_materialized']==[True]
        assert result['original_certified_warm_pair_preserved']==[False]
        np.testing.assert_array_equal(result['y'].get(),np.zeros((1,2)))
        assert paired_certificate(p,result['x'][0].get(),result['y'][0].get())['certificate_passed']
        assert result['returned_pair_certificates_without_alternate'][0]['certificate_passed']
        assert result['metrics'][0]['solver_dual_relative_kkt_gap']>1e-7
        assert result['factor_count']==0 and result['cpu_lp_calls']==0
        assert 'diagnostic' in result['intermediate_dual_scope']
        assert not paired_certificate(p,result['forest_x'][0].get(),
            result['forest_y'][0].get())['certificate_passed']
        assert solver.problem_hashes==(digest,)
    assert problem_hash(p)==digest


def test_gpu_default_does_not_enable_alternate_certificate():
    import cupy as cp
    with ForestGpuBatchedIPM([problem()]) as solver:
        result=solver.solve(initial_x=cp.asarray([[1.,0.]],dtype=cp.float64),
            initial_y=cp.asarray([[-1.,0.]],dtype=cp.float64),iterations=0)
        assert result['accepted'].tolist()==[False]
        assert result['allow_box_dual'] is False
        assert result['metrics'][0]['certificate_source']=='solver_row_dual'
        np.testing.assert_array_equal(result['y'].get(),[[-1.,0.]])
        assert 'analytic_box_dual_materialized' not in result


@pytest.mark.parametrize('row_cap,box_cap,x',[(1.,2.,1.),(.5,1.,1.),(1.,np.inf,1.)])
def test_gpu_nonattainable_unbounded_box_or_infeasible_primal_is_never_accepted(row_cap,box_cap,x):
    import cupy as cp
    p=problem(row_cap=row_cap,box_cap=box_cap)
    with ForestGpuBatchedIPM([p],allow_box_dual=True) as solver:
        result=solver.solve(initial_x=cp.asarray([[x,0.]],dtype=cp.float64),
            initial_y=cp.zeros((1,2),dtype=cp.float64),iterations=0)
        assert result['accepted'].tolist()==[False]
        assert result['analytic_box_dual_materialized']==[False]
        assert not result['metrics'][0]['certificate_passed']
        assert not paired_certificate(p,result['x'][0].get(),result['y'][0].get())['certificate_passed']


def test_gpu_mixed_batch_preserves_valid_nonbox_dual():
    import cupy as cp
    problems=[problem(),problem(row_cap=1.,box_cap=2.)]
    x=cp.asarray([[1.,0.],[1.,0.]],dtype=cp.float64)
    y=cp.asarray([[-1.,0.],[-1.,0.]],dtype=cp.float64)
    with ForestGpuBatchedIPM(problems,allow_box_dual=True) as solver:
        result=solver.solve(initial_x=x,initial_y=y,iterations=0)
        assert result['accepted'].all()
        assert result['analytic_box_dual_materialized']==[True,False]
        assert result['original_certified_warm_pair_preserved']==[False,True]
        np.testing.assert_array_equal(result['y'].get(),[[0.,0.],[-1.,0.]])
        assert result['metrics'][1]['certificate_source']=='solver_row_dual'
        for p,xx,yy in zip(problems,result['x'].get(),result['y'].get()):
            assert paired_certificate(p,xx,yy)['certificate_passed']


def test_gpu_certified_full_warm_pair_retains_eliminated_equality_multiplier():
    import cupy as cp
    p=(csr_matrix([[1.,-1.],[0.,1.]]),np.array([0.,2.]),np.zeros(2),
       np.ones(2),np.array([-1.,-1.]),1)
    x=cp.asarray([[1.,1.]],dtype=cp.float64)
    y=cp.asarray([[.3,0.]],dtype=cp.float64)
    assert paired_certificate(p,x[0].get(),y[0].get())['certificate_passed']
    with ForestGpuBatchedIPM([p],allow_box_dual=True) as solver:
        result=solver.solve(initial_x=x,initial_y=y,iterations=0)
        assert solver.full_n>solver.original_n
        assert result['accepted'].all()
        assert result['original_certified_warm_pair_preserved']==[True]
        assert result['analytic_box_dual_materialized']==[False]
        np.testing.assert_array_equal(result['x'].get(),x.get())
        np.testing.assert_array_equal(result['y'].get(),y.get())
        assert paired_certificate(p,result['x'][0].get(),result['y'][0].get())['certificate_passed']


def test_gpu_final_return_recheck_fails_closed_on_corrupted_postsolve(monkeypatch):
    import cupy as cp
    with ForestGpuBatchedIPM([problem()],allow_box_dual=True) as solver:
        expand=solver.expand_forest_device
        calls=0
        def corrupt_final(z,y):
            nonlocal calls
            calls+=1
            full_x,full_y=expand(z,y)
            if calls==2:full_x[:,0]+=.25
            return full_x,full_y
        monkeypatch.setattr(solver,'expand_forest_device',corrupt_final)
        result=solver.solve(initial_x=cp.asarray([[1.,0.]],dtype=cp.float64),
            initial_y=cp.asarray([[-1.,0.]],dtype=cp.float64),iterations=0)
        assert result['accepted'].tolist()==[False]
        assert result['accepted_iteration'].tolist()==[-1]
        assert result['status']=='returned_pair_certificate_failed'
        assert not result['metrics'][0]['returned_pair_certificate_passed']
