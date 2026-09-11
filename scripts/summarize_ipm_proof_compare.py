"""Matched-input, fixed-source medians; startup and hot claims kept separate."""
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_ipm_device_compare import summarize


def main():
    paths={g:[ROOT/f'results/pf_ipm_proof_{g}_32_2to8_20260907_r{i}.json' for i in (1,2,3)]
           for g in ('baseline','proof_reuse','cpu16')}
    groups={g:[json.loads(p.read_text()) for p in pp] for g,pp in paths.items()}
    ref=groups['cpu16'][0]
    for group,records in groups.items():
        for i,r in enumerate(records):
            for field in ('sequence_identity','source_sha256','original_certificate_limits'):
                if r[field]!=ref[field]:raise ValueError(f'Mismatch: {field}')
            if [s['problem_sha256'] for s in r['steps']]!=[s['problem_sha256'] for s in ref['steps']]:
                raise ValueError('Different inputs')
            if group!='cpu16':
                summarize(r,groups['cpu16'][i])
                c=r['configuration']
                if (c['reuse_equality_proofs']!=(group=='proof_reuse') or c['factor_precision']!='float64'
                    or c['factor_ordering']!='default' or c['factor_algorithm']!='default'
                    or c['factor_layout']!='block_diagonal' or c['newton_regularization']!=1e-6
                    or c['factor_reuse_interval']!=1 or c['fused_solve_guards']):
                    raise ValueError('Unexpected compound optimization')
    def distribution(values):return dict(raw=values,median=median(values),minimum=min(values),maximum=max(values))
    output=dict(scope='32 distinct environments, saved maxmin input replay steps 2..8; not closed-loop PPO',
        n=3,randomized=False,order=[['baseline','proof_reuse','cpu16'],['proof_reuse','cpu16','baseline'],
                                 ['cpu16','baseline','proof_reuse']],
        certified_lp_instances_per_group=672,precision_relaxed=False,current_CPU_solutions_used=False,
        cpu_lp_calls_in_gpu=0,input_source_precision_identities_matched=True,
        files={g:[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in pp]
               for g,pp in paths.items()})
    getters=dict(sequence_lifecycle_seconds=lambda r,g:r['sequence_lifecycle_seconds'],
        first_step_seconds=lambda r,g:r['steps'][0]['solve_plus_independent_verification_seconds'
            if g=='cpu16' else 'full_step_lifecycle_seconds'],
        hot_total_seconds=lambda r,g:sum(s['solve_plus_independent_verification_seconds'
            if g=='cpu16' else 'full_step_lifecycle_seconds'] for s in r['steps'][1:]))
    for name,get in getters.items():
        output[name]={g:distribution([get(r,g) for r in rr]) for g,rr in groups.items()}
        old=output[name]['baseline']['median'];new=output[name]['proof_reuse']['median']
        output[name]['reduction_percent']=100*(1-new/old)
        output[name]['halved']=new<=old*.5
    output['constructor_seconds']={g:distribution([r['steps'][0]['constructor_seconds'] for r in groups[g]])
                                   for g in ('baseline','proof_reuse')}
    path=ROOT/'results/pf_ipm_proof_compare_summary_20260907.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(output,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k!='files'},indent=2))


if __name__=='__main__':main()
