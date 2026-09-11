"""Plot a strictly validated completed pipeline run; never extrapolate timing."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_pipeline_run import NONOVERLAP_FIELDS, summarize_pipeline_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args()
    targets = [args.output_prefix.with_suffix(suffix) for suffix in ('.json', '.svg', '.png')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Refusing to overwrite an existing analysis artifact')
    payload = args.report.read_bytes()
    report = json.loads(payload)
    summary = summarize_pipeline_report(report)
    if not 0 <= args.repeat < len(summary['runs']):
        raise ValueError('repeat must select a completed run')
    row, result = report['runs'][args.repeat], summary['runs'][args.repeat]

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'svg.fonttype': 'none', 'axes.linewidth': .7,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.05), layout='constrained')
    side_colors = {'cpu': '#777777', 'hybrid': '#0072B2'}
    side_labels = {'cpu': 'CPU', 'hybrid': 'GPU/CPU hybrid'}
    for side, key in (('cpu', 'cpu_pipeline_history'), ('hybrid', 'gpu_pipeline_history')):
        cumulative = np.r_[0., np.cumsum([cycle['total_seconds'] for cycle in row[key]])]
        axes[0].plot(np.arange(len(cumulative)), cumulative, color=side_colors[side],
                     linewidth=1.5, label=side_labels[side])
    axes[0].set(xlabel='Simulation step', ylabel='Cumulative cycle time (s)',
                xlim=(0, summary['configuration']['steps']), ylim=(0, None))
    axes[0].legend(frameon=False, loc='upper left', fontsize=7)
    axes[0].set_title('(a) Continuous execution', loc='left', fontsize=9)

    bottom = np.zeros(2)
    part_labels = ('Maxmin service', 'Host-state interval', 'CPU request preparation',
                   'CPU completion wait', 'Scheduler')
    part_colors = ('#0072B2', '#E69F00', '#56B4E9', '#009E73', '#999999')
    for field, label, color in zip(NONOVERLAP_FIELDS, part_labels, part_colors):
        values = np.array([sum(block[field] for block in result['step_blocks'][side])
                           for side in ('cpu', 'hybrid')])
        axes[1].bar([0, 1], values, bottom=bottom, width=.52, color=color, label=label)
        bottom += values
    axes[1].set(xticks=[0, 1], xticklabels=['CPU', 'Hybrid'],
                ylabel='Non-overlapping time (s)', ylim=(0, None))
    axes[1].set_title('(b) Main-thread intervals', loc='left', fontsize=9)
    axes[1].legend(frameon=False, fontsize=6.5, loc='upper left',
                   bbox_to_anchor=(0, -.18), ncol=2)

    accepted, transitions = [], []
    maxmin = [record for record in row['gpu_history'] if record['stage'] == 'maxmin']
    for block in result['step_blocks']['hybrid']:
        selected = maxmin[block['start_step'] - 1:block['end_step']]
        accepted.append(100 * sum(record['bank_accepts'] for record in selected)
                        / sum(record['batch'] for record in selected))
        transitions.append(str(block['end_step']))
    axes[2].bar(np.arange(len(accepted)), accepted, width=.65, color='#0072B2')
    axes[2].set(xticks=np.arange(len(accepted)), xticklabels=transitions, ylim=(0, 100),
                xlabel='Block-ending simulation step', ylabel='Maxmin GPU bank acceptance (%)')
    axes[2].set_title('(c) GPU candidate reuse', loc='left', fontsize=9)

    summary['figure'] = dict(repeat=args.repeat, block_size_steps=20,
        source_path=str(args.report), source_sha256=hashlib.sha256(payload).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caption=('One completed paired run. Cumulative cycle time excludes small outer-timer '
                 'intervals. Panel b partitions main-thread elapsed time, not hardware utilization; '
                 'worker LP computation overlaps host-state and preparation intervals. Async '
                 'stage spans are not added. Panel c covers only maxmin dictionary acceptance, '
                 'not all LPs. No confidence intervals or statistical superiority are inferred.'))
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    for path in targets[1:]:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(json.dumps({'artifacts': [str(path) for path in targets],
                      'overall': summary['overall']}, allow_nan=False))


if __name__ == '__main__':
    main()
