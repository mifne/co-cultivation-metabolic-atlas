from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.benchmark_compact_gpu import save_failed_cpu_lps, cpu_solver_run_count, json_finite_values


def test_only_failed_original_lp_is_saved_without_overwriting(tmp_path):
    a=csr_matrix([[1.,2.],[-1.,0.]])
    request=(np.array([-1.,1.]),dict(A_ub=a,b_ub=np.array([3.,0.]),bounds=[(0.,1.),(0.,None)]))
    rows=[dict(success=True),dict(success=False,environment_id=7,stage='aggregate')]
    prefix=tmp_path/'failed.json'
    files=save_failed_cpu_lps(prefix,[request,request],rows)
    assert len(files)==1 and files[0]['environment_id']==7 and files[0]['batch_index']==1
    with np.load(files[0]['path'],allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload['a_data'],a.data)
        np.testing.assert_array_equal(payload['c'],request[0])
        np.testing.assert_array_equal(payload['upper'],[1.,np.inf])
        assert int(payload['neq'])==0
    with pytest.raises(FileExistsError):save_failed_cpu_lps(prefix,[request,request],rows)
    assert not Path(prefix.with_suffix('.cpu_failure_0.npz')).exists()


def test_optimizer_attempts_are_not_confused_with_cpu_fallback_requests():
    cpu=[dict(cpu_lp_calls=2,rows=[dict(cpu_solver_runs=2),dict(cpu_solver_runs=1)])]
    hybrid=[dict(cpu_lp_calls=2,groups=[dict(route='cpu_fallback',rows=cpu[0]['rows']),
        dict(route='gpu_restricted_repair',rows=[dict(cpu_solver_runs=100)])])]
    assert cpu_solver_run_count(cpu)==3
    assert cpu_solver_run_count(hybrid)==3
    assert cpu_solver_run_count([dict(cpu_lp_calls=2)])==2
    assert cpu_solver_run_count([dict(cpu_lp_calls=0,groups=[])])==0


def test_nonfinite_library_diagnostics_are_null_in_strict_json():
    import json
    values=dict(valid=1.2,success=False,attempts=[np.inf,np.nan,-np.inf],nested={'unused':None})
    out=json_finite_values(values)
    assert json.loads(json.dumps(out,allow_nan=False))==dict(valid=1.2,success=False,
        attempts=[None,None,None],nested={'unused':None})
    assert np.isinf(values['attempts'][0])
