"""Report qualified runtime and correction rounds without hiding failed arms."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(record):
    arms={}
    for name in record['configuration']['arms']:
        trials=[r for r in record['trials'] if r['arm']==name]
        completed=[r for r in trials if r['completed']]
        rows=[s for r in trials for s in r['steps'][1:]]
        a=dict(trials=len(trials),completed_trials=len(completed),
            certified_lp=sum(len(s['accepted_iteration']) for s in rows if s['qualified']),
            requested_hot_lp=record['configuration']['repeats']*(len(record['configuration']['steps'])-1)*record['configuration']['batch'],
            raw_candidate_passed=sum(s['raw_candidate_passed'] for s in rows),
            hot_original_max_primal=max((c['primal_residual'] for s in rows for c in s['independent_host_certificates']),default=None))
        if len(completed)==record['configuration']['repeats']:
            times=[r['hot_seconds'] for r in completed]
            a.update(hot_seconds_median=statistics.median(times),hot_seconds_min=min(times),hot_seconds_max=max(times),
                batched_factor_rounds_median=statistics.median(r['hot_factor_count'] for r in completed),
                newton_iterations_per_lp_median=statistics.median(
                    sum(sum(s['accepted_iteration']) for s in r['steps'][1:])/
                    sum(len(s['accepted_iteration']) for s in r['steps'][1:]) for r in completed),
                graph_inference_seconds_median=statistics.median(sum(s['inference_and_graph_seconds'] for s in r['steps'][1:]) for r in completed))
        arms[name]=a
    base=arms.get('previous',{}).get('hot_seconds_median')
    for name,a in arms.items():
        if base and 'hot_seconds_median' in a:a['speedup_vs_previous']=base/a['hot_seconds_median']
    return dict(scope=record['scope'],configuration=record['configuration'],arms=arms,
        cpu_lp_calls=record['cpu_lp_calls'],current_reference_vectors_loaded=record['current_reference_vectors_loaded'],
        notes=['First common cold step excluded from hot measurements; initialization separately recorded.',
            'Full time includes CPU input/graph staging, neural inference, GPU restart/solve, original certificate and independent audit.',
            'Factor rounds are BATCHED factorizations, not total individual LP factorizations.',
            'Newton iterations per LP use actual certified accepted_iteration per lane.',
            'Only saved maxmin LP inputs are replayed. Not closed-loop dFBA, three-stage LP, PPO or PHA endpoint validation.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing overwrite')
    result=summarize(json.loads(args.input.read_text()))
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
