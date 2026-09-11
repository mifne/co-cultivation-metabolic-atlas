"""Measure whether a preceding exact CPU basis certifies the next dFBA LP."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from greenlet import getcurrent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_basis_bank_rollout import environment, snapshot
from scripts.microbatch_comparison_support import drive_microbatch
from src.cpu_repeated_lp import RepeatedCpuLP
from src.fba_surrogate import model_fingerprint
from src.temporal_basis_probe import TemporalBasisProbeBackend


def finite_json(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key:finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def summarize(history, stages):
    report = {}
    for stage in stages:
        rows = [row for batch in history for row in batch['rows']
            if row['stage'] == stage and row['probe']['available']]
        passed = [row for row in rows if row['probe']['success']]
        report[stage] = dict(attempts=len(rows), certificate_passes=len(passed),
            certificate_pass_rate=(len(passed)/len(rows) if rows else None),
            factor_seconds=sum(row['probe'].get('factor_seconds', 0.) for row in rows),
            solve_seconds=sum(row['probe'].get('solve_seconds', 0.) for row in rows),
            certificate_seconds=sum(row['probe'].get('certificate_seconds', 0.) for row in rows),
            cpu_simplex_iterations_after_probe=sum(row['cpu_simplex_iterations'] for row in rows),
            zero_cpu_pivots_after_probe=sum(row['cpu_simplex_iterations'] == 0 for row in rows),
            failures=dict(sorted({reason:sum(row['probe'].get('failure') == reason for row in rows)
                for reason in {row['probe'].get('failure') for row in rows}}.items(), key=lambda item:str(item[0]))))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--environments', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20292001)
    parser.add_argument('--stages', nargs='+', default=['aggregate', 'exchange'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.steps, args.environments, args.workers) < 1 or args.steps > 120:
        raise ValueError('Invalid probe size')
    if len(set(args.stages)) != len(args.stages) or not set(args.stages) <= {'maxmin','aggregate','exchange','exchange_tie'}:
        raise ValueError('Invalid probe stages')
    seeds = list(range(args.seed, args.seed+args.environments))
    actions = [np.random.default_rng(seed).uniform(.05, .95, (120, 5)).astype(np.float32)[:args.steps]
        for seed in seeds]
    sample, _ = environment(args.seed)
    envs = []
    for seed in seeds:
        env = copy.deepcopy(sample); env.reset(seed=seed)
        model = env.simulator._cooperative_solver
        model._linprog_options.update(threads=1, parallel=False)
        envs.append((env, model))
    service = TemporalBasisProbeBackend(
        RepeatedCpuLP(args.workers,
            n_fluxes=sample.simulator._cooperative_solver.n_fluxes),
        stages=args.stages, workers=args.workers)
    sources = [Path(__file__), ROOT/'src/temporal_basis_probe.py', ROOT/'src/cpu_repeated_lp.py',
        ROOT/'src/community_solver.py']
    report = dict(status='running', configuration=vars(args) | {'output':str(args.output)},
        seeds=seeds, scope='Diagnostic only: previous exact CPU basis is interpreted on the current LP; '
        'the candidate never advances the trajectory and all states use a fresh exact CPU solve.',
        source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources}, history=[])
    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False))
        temp.replace(args.output)
    save(); started = time.perf_counter(); parent = getcurrent()
    try:
        with patch('src.community_solver.linprog', lambda c, **kwargs:parent.switch((c, kwargs))):
            endpoints = drive_microbatch(envs, actions, service, snapshot)
        report.update(status='completed', wall_seconds=time.perf_counter()-started,
            completed_steps=[args.steps]*args.environments, endpoints=endpoints,
            summary=summarize(service.history, args.stages), history=service.history,
            probe_parallel_wall_seconds=sum(batch['probe_wall_seconds'] for batch in service.history),
            probe_sum_row_seconds=sum(batch['probe_sum_seconds'] for batch in service.history),
            model_fingerprints={name:model_fingerprint(model)
                for name, model in sample.simulator.models.items()})
    except Exception as error:
        report.update(status='failed', wall_seconds=time.perf_counter()-started,
            error_type=type(error).__name__, error=str(error), history=service.history)
        raise
    finally:
        service.close(); save()


if __name__ == '__main__':
    main()
