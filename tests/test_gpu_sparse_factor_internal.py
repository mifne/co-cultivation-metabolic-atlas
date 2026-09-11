"""Internal device-checked proposals retain strict public cuDSS boundaries.

Run ``-k 'not cuda'`` for CPU-only structural/metadata checks. CUDA tests are
deliberately named so the experiment owner can serialize GPU execution.
"""
import ast
import inspect
import textwrap
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_sparse_factor import CudssError, UniformCudssFactor


def test_internal_numerical_path_has_no_host_finite_materialization():
    tree=ast.parse(textwrap.dedent(inspect.getsource(UniformCudssFactor._solve_device_checked)))
    guarded=next(node for node in ast.walk(tree) if isinstance(node,ast.Try))
    normal=ast.Module(body=guarded.body,type_ignores=[])
    calls=[node.func for node in ast.walk(normal) if isinstance(node,ast.Call)]
    assert not any(isinstance(func,ast.Name) and func.id in ('bool','float','int') for func in calls)
    assert not any(isinstance(func,ast.Attribute) and func.attr in
                   ('get','item','asnumpy','synchronize','_array') for func in calls)


@pytest.mark.parametrize('rhs',[np.ones((2,2,1),dtype=np.float32),np.ones((2,3,1)),None])
def test_internal_metadata_checks_precede_numerical_work(rhs):
    factor=SimpleNamespace(_context=lambda:None,factored=True,cp=np,batch=2,n=2,nrhs=1)
    with pytest.raises(ValueError,match='exact batch shape'):
        UniformCudssFactor._solve_device_checked(factor,rhs)


def test_internal_context_check_precedes_any_array_access():
    def rejected_context():
        raise CudssError('wrong context')
    factor=SimpleNamespace(_context=rejected_context)
    with pytest.raises(CudssError,match='wrong context'):
        UniformCudssFactor._solve_device_checked(factor,None)


def _matrix():
    return csr_matrix([[4.,1.],[1.,-3.]])


def test_cuda_internal_bad_rhs_lane_is_sanitized_and_cannot_look_like_valid_zero(monkeypatch):
    import cupy as cp
    matrix=_matrix()
    with UniformCudssFactor(matrix,batch_size=3,nrhs=2) as factor:
        factor.factor(cp.asarray(np.stack([matrix.data]*3)))
        rhs=cp.asarray(np.arange(12,dtype=float).reshape(3,2,2)+1.)
        rhs[1,0,0]=cp.nan
        saved=rhs.copy()
        seen=[]
        original_execute=factor._execute
        def record_input(phase):
            if phase==1008:seen.append(factor.rhs.copy())
            return original_execute(phase)
        monkeypatch.setattr(factor,'_execute',record_input)
        result=factor._solve_device_checked(rhs)
        np.testing.assert_array_equal(rhs.get(),saved.get())
        assert np.isfinite(seen[0].get()).all()
        np.testing.assert_array_equal(seen[0].get()[1],0.)
        assert np.isnan(result.get()[1]).all()  # Mandatory fail-closed marker.
        for lane in (0,2):
            np.testing.assert_allclose(result.get()[lane],
                np.linalg.solve(matrix.toarray(),rhs.get()[lane]),atol=1e-12,rtol=1e-12)
        assert not factor.failed
        diagnostic=factor.internal_diagnostics()
        assert diagnostic['solve_count']==factor.solve_count==1
        np.testing.assert_array_equal(diagnostic['invalid_rhs_count'].get(),[0,1,0])
        np.testing.assert_array_equal(diagnostic['nonfinite_output_count'].get(),[0,0,0])
        diagnostic['invalid_rhs_count'][:]=100
        np.testing.assert_array_equal(factor.internal_diagnostics()['invalid_rhs_count'].get(),[0,1,0])


def test_cuda_internal_results_own_storage_across_internal_and_public_solves():
    import cupy as cp
    matrix=_matrix()
    with UniformCudssFactor(matrix,batch_size=2,nrhs=2) as factor:
        factor.factor(cp.asarray(np.stack([matrix.data,2.*matrix.data])))
        rhs=cp.asarray(np.arange(8,dtype=float).reshape(2,2,2)+1.)
        result=factor._solve_device_checked(rhs)
        saved=result.copy()
        next_result=factor._solve_device_checked(2.*rhs)
        public=factor.solve(3.*rhs)
        np.testing.assert_array_equal(result.get(),saved.get())
        np.testing.assert_allclose(next_result.get(),2.*saved.get(),atol=1e-12)
        np.testing.assert_allclose(public.get(),3.*saved.get(),atol=1e-12)
        assert not cp.shares_memory(result,factor.solution)
        assert not cp.shares_memory(result,next_result)
        result[:]=0.
        np.testing.assert_allclose(next_result.get(),2.*saved.get(),atol=1e-12)
        assert factor.solve_count==3 and factor.internal_diagnostics()['solve_count']==2


def test_cuda_internal_nonfinite_output_isolated_without_accepting_other_lane_values(monkeypatch):
    import cupy as cp
    matrix=_matrix()
    with UniformCudssFactor(matrix,batch_size=2) as factor:
        factor.factor(cp.asarray(np.stack([matrix.data]*2)))
        original_execute=factor._execute
        def corrupt_one_lane(phase):
            original_execute(phase)
            if phase==1008:factor.solution[1,0,0]=cp.inf
        monkeypatch.setattr(factor,'_execute',corrupt_one_lane)
        result=factor._solve_device_checked(cp.ones((2,2,1)))
        assert np.isfinite(result.get()[0]).all() and np.isnan(result.get()[1]).all()
        assert not factor.failed
        np.testing.assert_array_equal(factor.internal_diagnostics()['nonfinite_output_count'].get(),[0,1])


def test_cuda_internal_shape_dtype_stream_thread_and_factor_state_guards():
    import cupy as cp
    matrix=_matrix()
    with UniformCudssFactor(matrix) as factor:
        rhs=cp.ones((1,2,1))
        with pytest.raises(CudssError,match='No current'):
            factor._solve_device_checked(rhs)
        factor.factor(cp.asarray(matrix.data[None]))
        with pytest.raises(ValueError):factor._solve_device_checked(cp.ones((1,3,1)))
        with pytest.raises(ValueError):factor._solve_device_checked(rhs.astype(cp.float32))
        with cp.cuda.Stream():
            with pytest.raises(CudssError,match='thread, CUDA device, or stream'):
                factor._solve_device_checked(rhs)
        errors=[]
        def other_thread():
            try:factor._solve_device_checked(rhs)
            except BaseException as error:errors.append(error)
        worker=threading.Thread(target=other_thread)
        worker.start();worker.join(timeout=10)
        assert not worker.is_alive()
        assert len(errors)==1 and isinstance(errors[0],CudssError)
        assert factor.solve_count==0 and not factor.failed
        assert np.isfinite(factor._solve_device_checked(rhs).get()).all()
    with pytest.raises(CudssError):factor._solve_device_checked(rhs)


def test_cuda_internal_wrong_device_array_is_rejected():
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<2:
        pytest.skip('A second CUDA device is required for the cross-device array check')
    original_device=cp.cuda.runtime.getDevice()
    other_device=1 if original_device==0 else 0
    with UniformCudssFactor(_matrix()) as factor:
        factor.factor(cp.asarray(_matrix().data[None]))
        with cp.cuda.Device(other_device):rhs=cp.ones((1,2,1))
        with pytest.raises(ValueError,match='Wrong-device'):
            factor._solve_device_checked(rhs)


def test_cuda_internal_cudss_failure_marks_workspace_failed_and_drains(monkeypatch):
    import cupy as cp
    with UniformCudssFactor(_matrix()) as factor:
        factor.factor(cp.asarray(_matrix().data[None]))
        actual_stream=factor.stream
        drains=[]
        def drained():
            drains.append(None)
            actual_stream.synchronize()
        monkeypatch.setattr(factor,'stream',SimpleNamespace(ptr=actual_stream.ptr,synchronize=drained))
        def failed_execute(phase):
            raise CudssError('manufactured cuDSS status error')
        monkeypatch.setattr(factor,'_execute',failed_execute)
        with pytest.raises(CudssError,match='manufactured'):
            factor._solve_device_checked(cp.ones((1,2,1)))
        assert factor.failed and len(drains)==1
        assert factor.solve_count==0 and factor.internal_solve_count==0
        with pytest.raises(CudssError,match='closed or failed'):
            factor._solve_device_checked(cp.ones((1,2,1)))


def test_cuda_public_solve_retains_synchronous_nonfinite_exceptions(monkeypatch):
    import cupy as cp
    with UniformCudssFactor(_matrix()) as factor:
        factor.factor(cp.asarray(_matrix().data[None]))
        with pytest.raises(ValueError,match='Nonfinite'):
            factor.solve(cp.full((1,2,1),cp.nan))
        assert not factor.failed and factor.solve_count==0
        original_execute=factor._execute
        def nonfinite_output(phase):
            original_execute(phase)
            if phase==1008:factor.solution[:]=cp.inf
        monkeypatch.setattr(factor,'_execute',nonfinite_output)
        with pytest.raises(CudssError,match='nonfinite solution'):
            factor.solve(cp.ones((1,2,1)))
        assert factor.failed
