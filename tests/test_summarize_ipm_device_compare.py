from copy import deepcopy
import pytest
from scripts.summarize_ipm_device_compare import summarize


def records():
    base=dict(completed=True,all_requested_steps_qualified=True,
        current_reference_vectors_loaded=False,current_CPU_solutions_passed_to_GPU=False,
        configuration=dict(mode='gpu',batch=32,stage='maxmin',steps=[2],warm=True,
            second_forest=True,fix_singleton_equalities=True,device_numeric_updates=True),
        sequence_identity={'trace':'same'},original_certificate_limits={'primal':1e-5},
        source_sha256={'solver':'same'},cpu_lp_calls=0,sequence_lifecycle_seconds=3.,
        steps=[dict(step=2,problem_sha256=['same']*32,all_original_certificates_passed=True,
            full_step_lifecycle_seconds=2.,solve_api_seconds=1.,numeric_update_wall_seconds=.1,
            constructor_seconds=.2,solver_result={'factor_count':4})])
    cpu=deepcopy(base)
    cpu['configuration'].update(mode='cpu',workers=16)
    cpu.update(highspy_version='1.15.1',numerical_retry_attempts=0)
    cpu['steps'][0]['solve_plus_independent_verification_seconds']=1.
    return base,cpu


def test_summary_does_not_promote_environments_to_repetitions():
    result=summarize(*records())
    assert result['repetitions']==1 and result['confidence_interval'] is None
    assert not result['steps'][0]['gpu_faster_than_cpu']
    assert result['steps'][0]['cpu_over_gpu_speed_ratio']==.5


@pytest.mark.parametrize('change',['hash','source','certificate','workers','reference','precision','retry'])
def test_comparison_rejects_unfair_or_unqualified_inputs(change):
    gpu,cpu=records()
    if change=='hash':cpu['steps'][0]['problem_sha256'][0]='different'
    elif change=='source':cpu['source_sha256']['solver']='different'
    elif change=='certificate':gpu['steps'][0]['all_original_certificates_passed']=False
    elif change=='workers':cpu['configuration']['workers']=1
    elif change=='reference':gpu['current_CPU_solutions_passed_to_GPU']=True
    elif change=='precision':cpu['original_certificate_limits']['primal']=.1
    elif change=='retry':cpu['numerical_retry_attempts']=1
    with pytest.raises(ValueError):summarize(gpu,cpu)
