"""One input-matched development trial, not a statistical speedup claim."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(gpu,cpu):
    def require(condition,message):
        if not condition:raise ValueError(message)
    for record in (gpu,cpu):
        require(record['completed'] and record['all_requested_steps_qualified'],'Incomplete or unqualified run')
        require(not record['current_reference_vectors_loaded'] and
            not record['current_CPU_solutions_passed_to_GPU'],'Reference leakage')
        require(all(s['all_original_certificates_passed'] for s in record['steps']),'Certificate failed')
    require(gpu['configuration']['mode']=='gpu' and cpu['configuration']['mode']=='cpu','Wrong comparator roles')
    for field in ('batch','stage','steps','warm','second_forest','fix_singleton_equalities'):
        require(gpu['configuration'][field]==cpu['configuration'][field],f'Mismatched {field}')
    for field in ('sequence_identity','original_certificate_limits','source_sha256'):
        require(gpu[field]==cpu[field],f'Mismatched {field}')
    require(gpu['configuration']['device_numeric_updates'] and gpu['cpu_lp_calls']==0,'Not device numeric updates')
    require(cpu['configuration']['workers']==16 and cpu['configuration']['warm'],'Strong warm CPU16 baseline required')
    require(cpu['highspy_version']=='1.15.1','Unexpected CPU version')
    require(cpu['numerical_retry_attempts']==0,'CPU retries must be analysed explicitly')
    require(len(gpu['steps'])==len(cpu['steps'])==len(gpu['configuration']['steps']),'Incomplete steps')
    result=dict(scope='saved maxmin LP input replay; NOT closed-loop dFBA or PPO',
        repetitions=1,confidence_interval=None,independent_environment_count=gpu['configuration']['batch'],
        source_code_identical=True,precision_relaxed=False,CPU_optimizer_calls_in_GPU=0,
        lifecycle_includes_input_file_load=False,steps=[])
    for gs,cs in zip(gpu['steps'],cpu['steps']):
        require(gs['step']==cs['step'] and gs['problem_sha256']==cs['problem_sha256'],'Different current LP inputs')
        g=gs['full_step_lifecycle_seconds']; c=cs['solve_plus_independent_verification_seconds']
        result['steps'].append(dict(step=gs['step'],gpu_solve_seconds=gs['solve_api_seconds'],
            gpu_update_solve_verify_seconds=g,cpu16_update_solve_verify_seconds=c,
            gpu_numeric_update_seconds=gs['numeric_update_wall_seconds'],
            gpu_constructor_seconds=gs['constructor_seconds'],
            gpu_factors=gs['solver_result']['factor_count'],cpu_over_gpu_speed_ratio=c/g,
            gpu_faster_than_cpu=g<c))
    result['sequence_lifecycle_seconds']=dict(gpu=gpu['sequence_lifecycle_seconds'],cpu16=cpu['sequence_lifecycle_seconds'])
    result['certified_LP_instances_per_backend']=len(gpu['steps'])*gpu['configuration']['batch']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(json.loads(args.gpu.read_text()),json.loads(args.cpu.read_text()))
    result['artifacts']={name:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name,path in [('gpu',args.gpu),('cpu',args.cpu)]}
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
