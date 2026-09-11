"""Plot a completed trajectory's GPU coverage and measured wall-time partition.

Component intervals are workflow timers, NOT GPU kernel/utilization timings.
The last repeat is displayed explicitly; no best-time selection is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def diagnostics(report):
    if report.get('configuration',{}).get('pipeline_cpu_stages'):
        raise ValueError('Pipelined stage spans overlap; use pipeline wall-time records instead')
    if report.get('status') != 'completed' or not report['runs']:
        raise ValueError('A completed benchmark is required')
    if any(not row.get('all_endpoint_gates_passed') or row.get('failure') for row in report['runs']):
        raise ValueError('Every reported trial must pass accuracy')
    config = report['configuration']; row = report['runs'][-1]
    steps, batch = config['steps'], config['environments']
    if row['gpu_completed_steps'] != [steps]*batch:
        raise ValueError('Incomplete trajectory')
    history = row['gpu_history']; names = ('maxmin', 'aggregate', 'exchange')
    if len(history) != steps*3 or [stage['stage'] for stage in history] != list(names)*steps:
        raise ValueError('Expected three original LP stages per step')
    if any(stage['batch'] != batch for stage in history):
        raise ValueError('Changing environment count is not supported')
    total = float(row['gpu_seconds'])
    service = sum(stage['seconds'] for stage in history)
    prepare = sum(stage['preparation_seconds'] for stage in history)
    screening = sum(stage['bank_seconds'] for stage in history)
    parts = dict(preparation=prepare, screening=screening,
                 other_lp_work=service-prepare-screening, host_dfba=total-service)
    if not all(np.isfinite(value) and value >= 0 for value in parts.values()):
        raise ValueError('Invalid or overlapping workflow timers')
    coverage = {name:[history[step*3+i]['bank_accepts']/batch for step in range(steps)]
                for i, name in enumerate(names)}
    if any(not 0 <= value <= 1 for values in coverage.values() for value in values):
        raise ValueError('Invalid dictionary acceptance counts')
    return dict(repeat=row['repeat'], environments=batch, steps=steps,
        cpu_seconds=row['cpu_seconds'], hybrid_seconds=total,
        coverage=coverage, workflow_seconds=parts,
        cpu_calls=sum(stage['cpu_lp_calls'] for stage in history), lp_requests=batch*steps*3,
        scope='Last completed repeat, not best repeat; workflow intervals are not device utilization measurements')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.output_prefix.with_suffix(suffix) for suffix in ('.json', '.svg', '.png')]
    if any(path.exists() for path in paths):
        raise FileExistsError(args.output_prefix)
    report = json.loads(args.input.read_text()); result = diagnostics(report)
    result.update(source=str(args.input), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'xtick.major.width':.7,'ytick.major.width':.7})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
    x = np.arange(1, result['steps']+1)
    for name, color, marker in zip(result['coverage'], ('#0072B2','#D55E00','#009E73'), ('o','s','^')):
        axes[0].plot(x, 100*np.array(result['coverage'][name]), color=color, marker=marker,
                     ms=3, lw=1, label=name.capitalize())
    axes[0].set(xlabel='Time step', ylabel='GPU dictionary acceptance (%)', ylim=(-2, 102), xticks=x)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title('(a)', loc='left', fontsize=10)
    labels = ['LP preparation', 'Candidate screening', 'Other LP work', 'Host dFBA']
    values = list(result['workflow_seconds'].values())
    axes[1].barh(np.arange(4), values, height=.55, color=['#56B4E9','#009E73','#D55E00','#999999'])
    axes[1].set(yticks=np.arange(4), yticklabels=labels, xlabel='Wall time (s)', xlim=(0,max(values)*1.2))
    axes[1].invert_yaxis()
    axes[1].set_title('(b)', loc='left', fontsize=10)
    for i,value in enumerate(values):axes[1].text(value+.25,i,f'{value:.2f}',va='center',fontsize=8)
    for path in paths[1:]:fig.savefig(path,dpi=300)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
