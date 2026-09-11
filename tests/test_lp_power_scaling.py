import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.lp_power_scaling import power_equilibrate,ScaledGpuBatchedIPM
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problem():
    return (csr_matrix([[2.**20,2.**20],[2.**-10,0.]]),np.array([2.**20,2.**-10]),
        np.zeros(2),np.full(2,np.inf),np.array([1.,2.]),1)


def test_scaling_is_reversible_for_all_rows_and_duals():
    p=problem();s=power_equilibrate(p)
    x=np.array([1.,0.]);y=np.array([2.**-20,0.])
    scaled_x=x/s.column_scale;scaled_y=y/(s.row_scale*s.objective_scale)
    np.testing.assert_array_equal(s.problem[0]@scaled_x,s.row_scale*(p[0]@x))
    np.testing.assert_array_equal(s.problem[4]@scaled_x,p[4]@x/s.objective_scale)
    assert paired_certificate(p,x,y)['certificate_passed']
    assert paired_certificate(s.problem,scaled_x,scaled_y)['certificate_passed']
    p[0].data[:]=0
    assert np.any(s.problem[0].data)


def test_scaled_solver_public_warm_start_and_outputs_are_original_units():
    import cupy as cp
    p=problem()
    with ScaledGpuBatchedIPM([p]) as solver:
        x=cp.array([[1.,0.]],dtype=cp.float64);y=cp.array([[2.**-20,0.]],dtype=cp.float64)
        r=solver.solve(initial_x=x,initial_y=y,iterations=0)
        assert r['accepted'].all()
        np.testing.assert_array_equal(r['x'].get(),x.get())
        np.testing.assert_array_equal(r['y'].get(),y.get())


@pytest.mark.parametrize('rounds',[True,-1,13,1.5])
def test_invalid_rounds_are_rejected(rounds):
    with pytest.raises(ValueError):power_equilibrate(problem(),rounds=rounds)
