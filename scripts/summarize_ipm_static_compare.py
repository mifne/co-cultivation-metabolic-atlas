"""Summarize immutable before/after input-matched reoptimization records.

No graph or speed claim extrapolates these maxmin saved-input timings to PPO.
The previous-version comparison is chronological, not a randomized crossover.
"""
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'


def main():
    output=RESULTS/'pf_ipm_static_compare_summary_20260906.json'
    if output.exists(): raise FileExistsError('Do not overwrite an earlier summary')
    names={
        'previous_gpu':[
            'pf_ipm_gpu_bound_restart_mu1e4_32_234_20260906.json',
            'pf_ipm_gpu_bound_restart_mu1e4_32_234_20260906_repeat.json',
            'pf_ipm_gpu_bound_restart_mu1e4_32_234_20260906_repeat3.json'],
        'current_gpu':[f'pf_ipm_gpu_direct_kkt_mu1e5_32_234_20260906_r{i}.json' for i in (1,2,3)],
        'current_cpu':[f'pf_ipm_cpu1151_static_compare32_234_20260906_r{i}.json' for i in (1,2,3)]}
    records={group:[json.loads((RESULTS/name).read_text()) for name in paths]
             for group,paths in names.items()}
    reference=records['current_gpu'][0]
    input_hashes=[row['problem_sha256'] for row in reference['steps']]
    for group,rows in records.items():
        for record in rows:
            assert record['completed'] and record['all_requested_steps_qualified']
            assert record['configuration']['batch']==32 and record['configuration']['stage']=='maxmin'
            assert record['configuration']['steps']==[2,3,4]
            assert record['sequence_identity']==reference['sequence_identity']
            assert record['original_certificate_limits']==reference['original_certificate_limits']
            assert [row['problem_sha256'] for row in record['steps']]==input_hashes
            assert all(row['all_original_certificates_passed'] for row in record['steps'])
            assert not record['current_reference_vectors_loaded']
            assert not record['current_CPU_solutions_passed_to_GPU']
            if group=='current_cpu':
                assert record['configuration']['workers']==16 and record['configuration']['warm']
                assert record['highspy_version']=='1.15.1'
                assert record['actual_cpu_optimizer_runs']==96 and record['numerical_retry_attempts']==0
            else:
                assert record['cpu_lp_calls']==0 and record['numeric_update_successes']==2
    def distribution(values):
        return dict(raw=values,median=median(values),minimum=min(values),maximum=max(values))
    summary=dict(role='maxmin_saved_input_comparison_NOT_closed_loop_dfba_or_PPO',
        files=names,n=3,all_original_inputs_identical=True,
        previous_comparison_is_chronological_not_randomized=True,
        current_gpu_cpu_interleaved_order=['GPU1','CPU1','GPU2','CPU2','GPU3','CPU3'],
        precision_relaxed=False,cpu_reference_or_fallback_used_by_gpu=False,
        source_code_fixed_within_current_group={g:all(r['source_sha256']==rows[0]['source_sha256']
            for r in rows) for g,rows in records.items() if g!='previous_gpu'},steps=[],
        total_certified_LP_instances={g:len(rows)*96 for g,rows in records.items()})
    assert all(summary['source_code_fixed_within_current_group'].values())
    for index,step in enumerate([2,3,4]):
        trial=dict(step=step)
        for group,rows in records.items():
            fields=({'solve':'solve_batch_wall_seconds',
                     'update_solve_verify':'solve_plus_independent_verification_seconds'}
                    if group=='current_cpu' else
                    {'solve':'solve_api_seconds','update_solve_verify':'full_step_lifecycle_seconds',
                     'numeric_update':'numeric_update_wall_seconds'})
            trial[group]={label:distribution([row['steps'][index][key] for row in rows])
                          for label,key in fields.items()}
            if group!='current_cpu':
                trial[group]['factors']=[r['steps'][index]['solver_result']['factor_count'] for r in rows]
        old=trial['previous_gpu']['update_solve_verify']['median']
        new=trial['current_gpu']['update_solve_verify']['median']
        cpu=trial['current_cpu']['update_solve_verify']['median']
        trial.update(previous_gpu_over_current_gpu=old/new,
                     current_cpu_over_current_gpu=cpu/new,
                     current_gpu_faster_than_cpu=new<cpu)
        summary['steps'].append(trial)
    summary['sequence_lifecycle']={group:distribution([r['sequence_lifecycle_seconds'] for r in rows])
                                    for group,rows in records.items()}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__': main()
