from types import SimpleNamespace
import numpy as np
import pytest
from src.gpu_candidate_transfer import download_candidate,METRICS


class Array(np.ndarray):
    gets=0
    def get(self):
        Array.gets+=1
        return np.array(self,copy=True)


def device(v):return np.asarray(v).view(Array)


CP=SimpleNamespace(float64=np.float64,where=lambda *args:device(np.where(*args)),
    concatenate=lambda *args,**kwargs:device(np.concatenate(*args,**kwargs)))


def candidate():
    return dict(accepted=device([True,False]),candidate_index=device([2,-1]),
        values=device([[1.,2.],[np.nan,np.nan]]),objective=device([3.,np.nan]),
        **{name:device([0.,np.inf]) for name in METRICS})


def test_packed_download_preserves_host_inputs_with_one_transfer():
    c=candidate();Array.gets=0
    expected=download_candidate(CP,c,2,2,3)
    assert Array.gets==7
    Array.gets=0;got=download_candidate(CP,c,2,2,3,packed=True)
    assert Array.gets==1
    for left,right in zip(got[:4],expected[:4]):np.testing.assert_array_equal(left,right)
    for name in METRICS:np.testing.assert_array_equal(got[4][name],expected[4][name])
    got[2][0,0]=999
    assert c['values'][0,0]==1.


@pytest.mark.parametrize('index',[3,2**53+1,np.iinfo(np.int64).max,np.iinfo(np.int64).min])
def test_bad_integer_id_is_rejected_before_float_packing(index):
    c=candidate();c['candidate_index'][0]=index
    out=download_candidate(CP,c,2,2,3,packed=True)
    assert out[1][0]==-1


@pytest.mark.parametrize('name',METRICS)
@pytest.mark.parametrize('bad',[np.nan,np.inf,-1.,1.])
def test_bad_metric_is_not_sanitized(name,bad):
    c=candidate();c[name][0]=bad
    out=download_candidate(CP,c,2,2,3,packed=True)
    np.testing.assert_equal(out[4][name][0],bad)


@pytest.mark.parametrize('name,value',[('accepted',[1,0]),('candidate_index',[1.,2.]),
    ('values',[[1.,2.,3.],[4.,5.,6.]]),('objective',[[1.],[2.]]),('primal_residual',[[0.],[0.]])])
def test_malformed_fields_fail_before_download(name,value):
    c=candidate();c[name]=device(value);Array.gets=0
    with pytest.raises(ValueError):download_candidate(CP,c,2,2,3,packed=True)
    assert Array.gets==0


def test_float64_contract_rejects_implicit_downcast():
    c=candidate();c['values']=c['values'].astype(np.float32)
    with pytest.raises(ValueError,match='float64'):download_candidate(CP,c,2,2,3,packed=True)


def test_actual_cuda_packed_transfer_matches_separate_fp64_downloads():
    cp=pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA device required')
    c={name:cp.asarray(value) for name,value in candidate().items()}
    expected=download_candidate(cp,c,2,2,3)
    got=download_candidate(cp,c,2,2,3,packed=True)
    for left,right in zip(got[:4],expected[:4]):np.testing.assert_array_equal(left,right)
    for name in METRICS:np.testing.assert_array_equal(got[4][name],expected[4][name])


def test_deferred_packing_enqueues_all_device_work_before_download():
    c=candidate();Array.gets=0
    ready=download_candidate(CP,c,2,2,3,packed=True,deferred=True)
    assert callable(ready) and Array.gets==0
    # Packing has already copied candidate values into a distinct buffer.
    c['values'][0,0]=123
    out=ready()
    assert Array.gets==1 and out[2][0,0]==1.
