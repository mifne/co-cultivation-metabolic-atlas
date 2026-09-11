"""Input-verified medians of the three serialized layout comparison trials."""
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_ipm_device_compare import summarize


def main():
    groups={}
    paths={}
    for group in ('uniform','block_diagonal','cpu16'):
        paths[group]=[ROOT/f'results/pf_ipm_layout_{group}_32_2to8_20260907_r{i}.json' for i in (1,2,3)]
        groups[group]=[json.loads(p.read_text()) for p in paths[group]]
    reference=groups['cpu16'][0]
    for group,records in groups.items():
        for index,record in enumerate(records):
            for field in ('sequence_identity','source_sha256','original_certificate_limits'):
                if record[field]!=reference[field]:raise ValueError(f'Mismatched {field}')
            if [r['problem_sha256'] for r in record['steps']] != [r['problem_sha256'] for r in reference['steps']]:
                raise ValueError('Inputs differ across repetitions')
            if group!='cpu16':
                summarize(record,groups['cpu16'][index])
                config=record['configuration']
                if (config['factor_layout']!=group or config['factor_reuse_interval']!=1
                        or config['fused_solve_guards'] or config['krylov_coordinates']!='full'):
                    raise ValueError('Unexpected compound optimization')
    def distribution(values):
        return dict(raw=values,median=median(values),minimum=min(values),maximum=max(values))
    result=dict(scope='saved maxmin LP replay, not closed-loop dFBA/PPO',n=3,
        order=['uniform','block_diagonal','cpu16']*3,randomized=False,
        original_input_and_source_hashes_matched=True,precision_relaxed=False,
        cpu_lp_calls_in_gpu=0,certified_lp_instances_per_group=672,steps=[],
        files={g:[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in pp]
               for g,pp in paths.items()})
    for index,step in enumerate(reference['configuration']['steps']):
        row=dict(step=step)
        for group,records in groups.items():
            fields=({'solve_seconds':'solve_batch_wall_seconds','full_seconds':'solve_plus_independent_verification_seconds'}
                if group=='cpu16' else {'solve_seconds':'solve_api_seconds','full_seconds':'full_step_lifecycle_seconds',
                    'numeric_update_seconds':'numeric_update_wall_seconds','constructor_seconds':'constructor_seconds'})
            row[group]={label:distribution([r['steps'][index][key] for r in records]) for label,key in fields.items()}
            if group!='cpu16':
                for key in ('factor_seconds','triangular_solve_and_update_seconds','factor_count','solve_count'):
                    row[group][key]=distribution([r['steps'][index]['solver_result'][key] for r in records])
        old=row['uniform']['full_seconds']['median'];new=row['block_diagonal']['full_seconds']['median']
        cpu=row['cpu16']['full_seconds']['median']
        row.update(uniform_over_block_ratio=old/new,block_reduction_percent=100*(1-new/old),
            cpu_over_block_ratio=cpu/new,block_faster_than_cpu=new<cpu)
        result['steps'].append(row)
    result['sequence_lifecycle_seconds']={g:distribution([r['sequence_lifecycle_seconds'] for r in rr])
        for g,rr in groups.items()}
    result['hot_steps_total_seconds']={g:distribution([sum(s['solve_plus_independent_verification_seconds']
        if g=='cpu16' else s['full_step_lifecycle_seconds'] for s in r['steps'][1:]) for r in rr])
        for g,rr in groups.items()}
    output=ROOT/'results/pf_ipm_layout_compare_summary_20260907.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(steps=[dict(step=r['step'],uniform=r['uniform']['full_seconds']['median'],
        block=r['block_diagonal']['full_seconds']['median'],cpu16=r['cpu16']['full_seconds']['median'],
        block_reduction_percent=r['block_reduction_percent']) for r in result['steps']],
        hot_total=result['hot_steps_total_seconds'],lifecycle=result['sequence_lifecycle_seconds']),indent=2))


if __name__=='__main__':main()
