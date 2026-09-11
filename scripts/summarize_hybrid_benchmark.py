"""Summarize a completed, paired benchmark without hiding failed trials.

The first cold-use trial is reported separately. The remaining paired log
ratios receive a Student-t confidence interval; this describes these runs,
not arbitrary cultures, GPUs, longer horizons, or reinforcement learning.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t


FROZEN_CONFIGURATION = dict(repeats=6, environments=32, steps=2, cpu_workers=4,
    seed=20289801, seed_stride=32, execution_order='alternate', cpu_backend='dictionary',
    bank='results/pf_compact_prefix4x4_20260904', hybrid=True,
    dispersion_threshold=1., candidate_limit=0, hybrid_rounds=0,
    cohort_candidates=True, cohort_replay=True, cpu_basis_proposals=True,
    temporal_pivots=0, restricted_columns=[], candidate_ranking='count',
    tie_policy='original3', tie_relative_allowance=1e-10)
LP_LIMITS = dict(primal_residual=1e-5, dual_violation=1e-7, relative_kkt_gap=1e-7)
ENDPOINT_LIMITS = dict(pha_relative=.01, biomass_g_l=.01, phv_fraction=.01)
FROZEN_STAGES = ('maxmin', 'aggregate', 'exchange')


def _within_limits(row, limits, *, shape, description):
    for key, limit in limits.items():
        values = np.asarray(row.get(key), dtype=float)
        if (values.shape != shape or not np.isfinite(values).all()
                or np.any(values < 0.) or np.any(values > limit)):
            raise ValueError(f'{description}: invalid or excessive {key}')


def validate_frozen_protocol(report):
    """Check the recorded Rev5 protocol, without rerunning or repairing a trial.

    This validates JSON evidence, not a cryptographic attestation of execution.
    The old report schema does not store LP tolerances as configuration fields;
    every recorded original-LP residual is therefore checked against the fixed
    protocol limits rather than inferring tolerances from a success flag.
    """
    if report.get('status') != 'completed':
        raise ValueError('Frozen-protocol validation requires a completed benchmark')
    config = report.get('configuration', {})
    for key, expected in FROZEN_CONFIGURATION.items():
        if key not in config or config[key] != expected:
            raise ValueError(f'Frozen protocol: configuration mismatch for {key}')
    if report.get('dt_hours') != .2 or report.get('simulated_hours_per_environment') != .4:
        raise ValueError('Frozen protocol: simulation time mismatch')
    runs = report.get('runs', [])
    if len(runs) != 6 or [row.get('repeat') for row in runs] != list(range(6)):
        raise ValueError('Frozen protocol: expected exactly six ordered trial records')
    if [row.get('cold_first_use_included') for row in runs] != [True]+[False]*5:
        raise ValueError('Frozen protocol: expected one first-use and five warm pairs')
    manifest = report.get('offline_bank_manifest', {})
    if manifest.get('status') != 'completed':
        raise ValueError('Frozen protocol: incomplete dictionary manifest')
    models = report.get('model_fingerprints')
    if not models or models != manifest.get('model_fingerprints'):
        raise ValueError('Frozen protocol: model/dictionary fingerprints differ')
    stages = manifest.get('stages', [])
    if ([stage.get('stage') for stage in stages] != list(FROZEN_STAGES)
            or [len(stage.get('entries', [])) for stage in stages] != [20, 28, 20]):
        raise ValueError('Frozen protocol: expected the smaller 20/28/20-entry dictionary')
    train_seeds = manifest.get('train_seeds')
    if not isinstance(train_seeds, list) or not train_seeds:
        raise ValueError('Frozen protocol: missing dictionary training-seed provenance')
    if not report.get('source_hashes'):
        raise ValueError('Frozen protocol: missing implementation provenance')
    seen = set(); training = set(train_seeds)
    for index, row in enumerate(runs):
        seeds = row.get('seeds', [])
        expected_seeds = list(range(config['seed']+index*32, config['seed']+(index+1)*32))
        if seeds != expected_seeds or seen.intersection(seeds) or training.intersection(seeds):
            raise ValueError('Frozen protocol: seed schedule mismatch or training/test overlap')
        seen.update(seeds)
        order = 'cpu-first' if index % 2 == 0 else 'gpu-first'
        if row.get('execution_order') != order:
            raise ValueError('Frozen protocol: execution order is not the frozen alternating order')
        if (row.get('all_endpoint_gates_passed') is not True or row.get('failure')
                or row.get('gpu_completed_steps') != [2]*32):
            raise ValueError('Frozen protocol: failed or incomplete trajectory')
        errors = row.get('errors', [])
        if len(errors) != 32:
            raise ValueError('Frozen protocol: expected 32 endpoint comparisons per pair')
        for error in errors:
            _within_limits(error, ENDPOINT_LIMITS, shape=(), description='Endpoint')
        histories = row.get('gpu_history', [])
        if [history.get('stage') for history in histories] != list(FROZEN_STAGES)*2:
            raise ValueError('Frozen protocol: original three-stage GPU history is incomplete')
        for history in histories:
            if history.get('batch') != 32 or history.get('accepted') != [True]*32:
                raise ValueError('Frozen protocol: incomplete GPU/CPU hybrid LP batch')
            _within_limits(history, LP_LIMITS, shape=(32,), description='Hybrid LP')
        cpu_histories = row.get('cpu_history', [])
        if len(cpu_histories) != 6:
            raise ValueError('Frozen protocol: expected six CPU LP batches')
        for history, stage in zip(cpu_histories, list(FROZEN_STAGES)*2):
            entries = history.get('rows', [])
            if history.get('batch') != 32 or len(entries) != 32:
                raise ValueError('Frozen protocol: incomplete CPU LP batch')
            for entry in entries:
                if (entry.get('stage') != stage or entry.get('success') is not True
                        or entry.get('certificate_passed') is not True):
                    raise ValueError('Frozen protocol: CPU stage or certificate mismatch')
                _within_limits(entry, LP_LIMITS, shape=(), description='CPU LP')
        timings = np.asarray([row.get('cpu_seconds'), row.get('gpu_seconds')], dtype=float)
        if not np.isfinite(timings).all() or np.any(timings <= 0):
            raise ValueError('Frozen protocol: invalid paired wall times')
    return dict(name='2026-09-04 Rev5 frozen 32x2 protocol', validated=True,
        cold_pairs=1, warm_pairs=5, distinct_evaluation_seeds=len(seen),
        original_lp_limits=LP_LIMITS.copy(), endpoint_limits=ENDPOINT_LIMITS.copy(),
        tolerance_evidence='Recorded residuals checked directly; original schema has no LP tolerance configuration fields',
        limitation='Checks recorded provenance/configuration, not external attestation of immutable execution')


def summarize(report, *, strict_frozen_protocol=False):
    if report['status'] != 'completed':
        raise ValueError('Incomplete or failed benchmarks cannot be summarized as a speedup')
    runs = report['runs']
    if not runs or any(not r.get('all_endpoint_gates_passed') or r.get('failure') for r in runs):
        raise ValueError('All trials must pass the declared accuracy gates')
    expected = [report['configuration']['steps']]*report['configuration']['environments']
    if any(r['gpu_completed_steps'] != expected for r in runs):
        raise ValueError('A trajectory is incomplete')
    protocol = validate_frozen_protocol(report) if strict_frozen_protocol else dict(validated=False)
    cold = [r for r in runs if r.get('cold_first_use_included')]
    warm = [r for r in runs if not r.get('cold_first_use_included')]
    if len(warm) < 2:
        raise ValueError('At least two warm paired runs are required for an interval')
    cpu = np.array([r['cpu_seconds'] for r in warm])
    hybrid = np.array([r['gpu_seconds'] for r in warm])
    if not (np.isfinite(cpu).all() and np.isfinite(hybrid).all() and (cpu > 0).all() and (hybrid > 0).all()):
        raise ValueError('Invalid timings')
    ratio = cpu/hybrid; logs = np.log(ratio)
    positive, negative = int(np.count_nonzero(logs > 0)), int(np.count_nonzero(logs < 0))
    non_tied = positive+negative
    sign_pvalue = float(binomtest(positive, non_tied, p=.5).pvalue) if non_tied else 1.
    margin = float(t.ppf(.975, len(warm)-1)*logs.std(ddof=1)/np.sqrt(len(warm)))
    center = float(logs.mean())
    stages = [h for r in warm for h in r['gpu_history']]
    lp_count = sum(h['batch'] for h in stages)
    cpu_calls = sum(h.get('cpu_lp_calls', 0) for h in stages)
    error = {key:max(e[key] for r in runs for e in r['errors'])
             for key in ('pha_relative', 'biomass_g_l', 'phv_fraction')}
    return dict(frozen_protocol=protocol,
        cold_trials=[dict(repeat=r['repeat'],cpu_seconds=r['cpu_seconds'],
                    hybrid_seconds=r['gpu_seconds'],ratio=r['cpu_seconds']/r['gpu_seconds']) for r in cold],
        warm_trials=len(warm),warm_cpu_seconds=cpu.tolist(),warm_hybrid_seconds=hybrid.tolist(),
        warm_ratios=ratio.tolist(),geometric_mean_ratio=float(np.exp(center)),
        geometric_mean_ratio_95_ci=[float(np.exp(center-margin)),float(np.exp(center+margin))],
        mean_cpu_seconds=float(cpu.mean()),mean_hybrid_seconds=float(hybrid.mean()),
        total_time_reduction_fraction=float(1.-hybrid.sum()/cpu.sum()),
        cpu_fallback_calls=cpu_calls,total_lp_requests=lp_count,
        cpu_fallback_fraction=cpu_calls/lp_count if lp_count else None,
        positive_warm_pairs=positive, negative_warm_pairs=negative,
        tied_warm_pairs=len(warm)-non_tied, two_sided_sign_test_pvalue=sign_pvalue,
        sign_test_non_tied_pairs=non_tied,
        maximum_errors=error,scope='This hardware, configuration, trajectory length and seed batches only',
        interval_method='Two-sided 95% Student-t interval on paired log timing ratios; cold use excluded and separately reported',
        interval_assumptions='Independent, approximately normal paired log ratios; n counts seed batches, not environments. Non-overlapping seeds do not establish independence of timing noise.',
        sign_test_scope='Secondary descriptive robustness check, not the prespecified primary interval; exact-zero log ratios are ties. With five non-tied pairs all on one side, two-sided p=0.0625.',
        cold_warm_scope='First-use includes graph compilation, not all startup costs; warm means reused immutable graphs with fresh trajectory/CPU states.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--strict-frozen-protocol', action='store_true',
        help='Require the completed 2026-09-04 Rev5 six-pair protocol and verify all recorded residuals')
    args=parser.parse_args()
    output=args.output_prefix
    paths=[output.with_suffix(suffix) for suffix in ('.json','.md','.svg','.png')]
    if any(path.exists() for path in paths):raise FileExistsError(output)
    report=json.loads(args.input.read_text());result=summarize(report, strict_frozen_protocol=args.strict_frozen_protocol)
    result.update(source=str(args.input),source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  configuration=report['configuration'])
    output.parent.mkdir(parents=True,exist_ok=True)
    paths[0].write_text(json.dumps(result,indent=2))
    config=report['configuration'];ci=result['geometric_mean_ratio_95_ci']
    lines=['# GPU-first hybrid benchmark: paired results','',
        f"Scope: {config['environments']} environments × {config['steps']} time steps per trajectory; "
        f"{report['simulated_hours_per_environment']:g} simulated hours. CPU comparator: {config['cpu_backend']}, "
        f"{config['cpu_workers']} workers, one thread per LP.",'',
        '| Trial | CPU (s) | Hybrid (s) | CPU / hybrid |','|---|---:|---:|---:|']
    for row in report['runs']:
        label=('cold' if row['cold_first_use_included'] else 'warm')+f" {row['repeat']}"
        lines.append(f"| {label} | {row['cpu_seconds']:.3f} | {row['gpu_seconds']:.3f} | {row['cpu_seconds']/row['gpu_seconds']:.4f} |")
    lines+=['',f"Warm geometric mean ratio: {result['geometric_mean_ratio']:.4f} "
        f"(95% CI {ci[0]:.4f}–{ci[1]:.4f}; n={result['warm_trials']} paired seed batches).",
        'This small-sample Student-t interval assumes independent, approximately normal paired log ratios. '
        'The sample size is the number of seed batches, not the number of environments; '
        'distinct seeds do not exclude shared thermal/clock-related timing noise.',
        f"Secondary two-sided sign test: p={result['two_sided_sign_test_pvalue']:.5g}; "
        f"{result['positive_warm_pairs']} ratios above one, {result['negative_warm_pairs']} below, "
        f"{result['tied_warm_pairs']} ties. Five non-tied pairs all on one side give p=0.0625; "
        'this descriptive check does not replace the prespecified paired-log interval.',
        f"Warm mean CPU/hybrid times: {result['mean_cpu_seconds']:.3f} / {result['mean_hybrid_seconds']:.3f} s.",
        f"CPU fallback: {result['cpu_fallback_calls']} / {result['total_lp_requests']} LP requests "
        f"({result['cpu_fallback_fraction']:.2%}). This is a HYBRID, not GPU-only execution.",'',
        'Maximum endpoint discrepancies (including the cold trial):','',
        f"- PHA relative: {result['maximum_errors']['pha_relative']:.5g}",
        f"- Biomass absolute: {result['maximum_errors']['biomass_g_l']:.5g} g/L",
        f"- PHV fraction absolute: {result['maximum_errors']['phv_fraction']:.5g}",'',
        'Figure: paired warm online wall times (left), paired speed ratios and their geometric mean/95% interval (right). '
        'Input assembly, transfers, candidate screening, failed proposals, CPU fallback and final state updates are timed. '
        'Offline dictionary construction and environment construction are separately reported in the source JSON. '
        'Shared static setup and CPU service setup are also excluded from these online times. '
        'First-use graph compilation is not silently discarded; it is the first table row, not a full-startup benchmark. '
        'Warm pairs reuse graphs, not solutions from earlier trajectories. '
        'No 120-step, full-training, multi-GPU, or universal speedup claim follows from a short trajectory.',
        '',f"Frozen protocol validation: {'passed' if result['frozen_protocol']['validated'] else 'not requested'}.",
        'Recorded residuals are checked directly in strict mode; the source schema does not independently attest '
        'runtime tolerance settings or immutability. Geometric mean speed ratio and aggregate time reduction '
        'are different statistics and are not interchangeable.',
        '',f"Source: `{args.input}`",f"SHA-256: `{result['source_sha256']}`"]
    paths[1].write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,
        'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.8),layout='constrained')
    cpu=np.array(result['warm_cpu_seconds']);gpu=np.array(result['warm_hybrid_seconds'])
    offsets=np.linspace(-.05,.05,len(cpu))
    for i,offset in enumerate(offsets):
        axes[0].plot([offset,1+offset],[cpu[i],gpu[i]],color='#A5A5A5',lw=.65,zorder=1)
    axes[0].scatter(offsets,cpu,color='#0072B2',s=22,zorder=2)
    axes[0].scatter(1+offsets,gpu,color='#D55E00',s=25,marker='^',zorder=2)
    axes[0].set(xticks=[0,1],xticklabels=['CPU','Hybrid'],ylabel='Online wall time (s)',xlim=(-.3,1.3))
    axes[0].set_title('(a)',loc='left',fontsize=10)
    ratios=np.array(result['warm_ratios']);x=np.arange(1,len(ratios)+1)
    axes[1].axhline(1.,color='#737373',lw=.7,ls='--')
    axes[1].scatter(x,ratios,color='#009E73',s=22)
    mean=result['geometric_mean_ratio']
    axes[1].errorbar(len(x)+1,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='s',color='#222222',
        markersize=4,capsize=3,lw=.8)
    axes[1].set(xticks=list(x)+[len(x)+1],xticklabels=[str(v) for v in x]+['GM'],
        ylabel='CPU / hybrid time',xlabel='Paired seed batch')
    axes[1].set_title('(b)',loc='left',fontsize=10)
    for suffix in ('.svg','.png'):fig.savefig(output.with_suffix(suffix),dpi=300)
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
