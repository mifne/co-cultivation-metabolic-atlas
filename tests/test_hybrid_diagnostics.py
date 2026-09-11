import copy
import pytest

from scripts.plot_hybrid_diagnostics import diagnostics


def example():
    stages = [dict(stage=name,batch=2,bank_accepts=i,cpu_lp_calls=2-i,
        seconds=2.,preparation_seconds=.2,bank_seconds=.3)
        for _ in range(2) for i,name in enumerate(('maxmin','aggregate','exchange'))]
    return dict(status='completed',configuration=dict(steps=2,environments=2),runs=[dict(
        repeat=0,all_endpoint_gates_passed=True,failure=None,gpu_completed_steps=[2,2],
        cpu_seconds=14.,gpu_seconds=15.,gpu_history=stages)])


def test_partition_and_coverage_are_exact():
    out=diagnostics(example())
    assert out['coverage']['exchange']==[1.,1.]
    assert out['cpu_calls']==6 and out['lp_requests']==12
    assert sum(out['workflow_seconds'].values())==pytest.approx(15.)
    assert out['workflow_seconds']['host_dfba']==3.


def test_pipeline_spans_cannot_be_reported_as_additive_workflow_time():
    report=example()
    report['configuration']['pipeline_cpu_stages']=True
    with pytest.raises(ValueError,match='overlap'):
        diagnostics(report)


@pytest.mark.parametrize('change', ['incomplete','failed','steps','stage','batch','overlap'])
def test_invalid_benchmarks_are_not_plotted_as_success(change):
    report=copy.deepcopy(example());row=report['runs'][0]
    if change=='incomplete':report['status']='running'
    elif change=='failed':row['all_endpoint_gates_passed']=False
    elif change=='steps':row['gpu_completed_steps']=[1,2]
    elif change=='stage':row['gpu_history'][0]['stage']='exchange_tie'
    elif change=='batch':row['gpu_history'][0]['batch']=1
    elif change=='overlap':row['gpu_seconds']=1.
    with pytest.raises(ValueError):diagnostics(report)
