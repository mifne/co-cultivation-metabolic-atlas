import copy
import pytest
from scripts.summarize_hybrid_benchmark import (
    FROZEN_CONFIGURATION, FROZEN_STAGES, LP_LIMITS, summarize, validate_frozen_protocol,
)


def example():
    row=dict(all_endpoint_gates_passed=True,failure=None,gpu_completed_steps=[2],
        cpu_seconds=2.,gpu_seconds=1.,gpu_history=[dict(batch=6,cpu_lp_calls=2)],
        errors=[dict(pha_relative=0.,biomass_g_l=0.,phv_fraction=0.)])
    return dict(status='completed',configuration=dict(steps=2,environments=1),
        runs=[dict(copy.deepcopy(row),repeat=i,cold_first_use_included=i==0) for i in range(4)])


def test_summary_keeps_cold_and_warm_separate_and_counts_fallbacks():
    report=example();report['runs'][0]['gpu_seconds']=4.
    result=summarize(report)
    assert result['warm_trials']==3 and result['geometric_mean_ratio']==2.
    assert result['geometric_mean_ratio_95_ci']==[2.,2.]
    assert result['cold_trials'][0]['ratio']==.5
    assert result['cpu_fallback_fraction']==pytest.approx(1/3)


@pytest.mark.parametrize('change',[lambda r:r.update(status='benchmark'),
    lambda r:r['runs'][1].update(all_endpoint_gates_passed=False),
    lambda r:r['runs'][1].update(gpu_completed_steps=[1]),
    lambda r:r['runs'][1].update(gpu_seconds=0.)])
def test_summary_never_silently_drops_bad_trials(change):
    report=example();change(report)
    with pytest.raises(ValueError):summarize(report)


def frozen_example():
    certificate = {key:0. for key in LP_LIMITS}
    stages = [dict(stage=stage, entries=[{} for _ in range(count)])
              for stage,count in zip(FROZEN_STAGES, [20,28,20])]
    models = {'example_model':'a'*64}
    report = dict(status='completed', configuration=copy.deepcopy(FROZEN_CONFIGURATION),
        dt_hours=.2, simulated_hours_per_environment=.4, model_fingerprints=models,
        source_hashes={'example.py':'b'*64},
        offline_bank_manifest=dict(status='completed', model_fingerprints=copy.deepcopy(models),
            train_seeds=[123], stages=stages), runs=[])
    for repeat in range(6):
        first = FROZEN_CONFIGURATION['seed']+repeat*32
        hybrid_history = [dict(stage=stage,batch=32,accepted=[True]*32,cpu_lp_calls=4,
            **{key:[0.]*32 for key in LP_LIMITS}) for stage in FROZEN_STAGES*2]
        cpu_history = [dict(batch=32,rows=[dict(stage=stage,success=True,
            certificate_passed=True,**certificate) for _ in range(32)])
            for stage in FROZEN_STAGES*2]
        report['runs'].append(dict(repeat=repeat,cold_first_use_included=repeat==0,
            seeds=list(range(first,first+32)),execution_order='cpu-first' if repeat%2==0 else 'gpu-first',
            all_endpoint_gates_passed=True,failure=None,gpu_completed_steps=[2]*32,
            cpu_seconds=2.,gpu_seconds=1.,gpu_history=hybrid_history,cpu_history=cpu_history,
            errors=[dict(pha_relative=0.,biomass_g_l=0.,phv_fraction=0.) for _ in range(32)]))
    return report


def test_frozen_protocol_checks_six_pairs_and_does_not_mutate_report():
    report=frozen_example();original=copy.deepcopy(report)
    result=summarize(report,strict_frozen_protocol=True)
    assert result['frozen_protocol']['validated']
    assert result['frozen_protocol']['distinct_evaluation_seeds']==192
    assert result['frozen_protocol']['original_lp_limits']==LP_LIMITS
    assert result['warm_trials']==5
    assert result['two_sided_sign_test_pvalue']==.0625
    assert result['positive_warm_pairs']==5 and result['tied_warm_pairs']==0
    assert report==original


@pytest.mark.parametrize('change',[
    lambda r:r.update(status='benchmark'),
    lambda r:r['configuration'].update(repeats=5),
    lambda r:r['configuration'].update(tie_policy='secondary'),
    lambda r:r['configuration'].update(tie_relative_allowance=1e-6),
    lambda r:r['configuration'].update(hybrid_rounds=1),
    lambda r:r['configuration'].update(dispersion_threshold=2.),
    lambda r:r['runs'].pop(),
    lambda r:r['runs'][1].update(repeat=0),
    lambda r:r['runs'][1].update(cold_first_use_included=True),
    lambda r:r['runs'][1].update(seeds=r['runs'][0]['seeds']),
    lambda r:r['offline_bank_manifest']['train_seeds'].append(r['runs'][0]['seeds'][0]),
    lambda r:r['runs'][1].update(execution_order='cpu-first'),
    lambda r:r.update(dt_hours=.1),
    lambda r:r.update(model_fingerprints={'changed':'c'*64}),
    lambda r:r.update(source_hashes={}),
    lambda r:r['offline_bank_manifest']['stages'][0]['entries'].pop(),
    lambda r:r['runs'][0].update(cpu_seconds=float('nan')),
    lambda r:r['runs'][0].update(gpu_seconds=0.),
    lambda r:r['runs'][2]['errors'][3].update(pha_relative=.011),
    lambda r:r['runs'][2]['errors'][3].update(biomass_g_l=float('nan')),
    lambda r:r['runs'][2]['errors'].pop(),
    lambda r:r['runs'][2]['gpu_history'][0].update(stage='exchange_tie'),
    lambda r:r['runs'][2]['gpu_history'][0].update(accepted=[True]*31),
    lambda r:r['runs'][2]['gpu_history'][0].update(primal_residual=[2e-5]*32),
    lambda r:r['runs'][2]['gpu_history'][0].update(dual_violation=[2e-7]*32),
    lambda r:r['runs'][2]['gpu_history'][0].update(relative_kkt_gap=[float('nan')]*32),
    lambda r:r['runs'][2]['cpu_history'].pop(),
    lambda r:r['runs'][2]['cpu_history'][0]['rows'].pop(),
    lambda r:r['runs'][2]['cpu_history'][0]['rows'][0].update(stage='aggregate'),
    lambda r:r['runs'][2]['cpu_history'][0]['rows'][0].update(certificate_passed=False),
    lambda r:r['runs'][2]['cpu_history'][0]['rows'][0].update(relative_kkt_gap=2e-7),
])
def test_strict_protocol_rejects_bad_evidence_even_with_success_flag(change):
    report=frozen_example();change(report)
    with pytest.raises(ValueError):validate_frozen_protocol(report)


def test_protocol_validation_is_opt_in_for_existing_development_reports():
    report=example()
    assert summarize(report)['frozen_protocol']==dict(validated=False)
    with pytest.raises(ValueError):summarize(report,strict_frozen_protocol=True)


def test_sign_test_counts_ties_and_does_not_claim_five_wins_when_mixed():
    report=frozen_example()
    for row,gpu_time in zip(report['runs'][1:],[1.,1.,1.,2.,4.]):
        row['gpu_seconds']=gpu_time
    result=summarize(report,strict_frozen_protocol=True)
    assert result['positive_warm_pairs']==3 and result['negative_warm_pairs']==1
    assert result['tied_warm_pairs']==1 and result['sign_test_non_tied_pairs']==4
    assert result['two_sided_sign_test_pvalue']==.625


def test_sign_test_all_ties_is_uninformative():
    report=example()
    for row in report['runs']:row['gpu_seconds']=row['cpu_seconds']
    result=summarize(report)
    assert result['two_sided_sign_test_pvalue']==1.
    assert result['sign_test_non_tied_pairs']==0
