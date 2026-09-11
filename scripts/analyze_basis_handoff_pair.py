#!/usr/bin/env python3
"""Create a provenance-rich paired summary of CPU-basis handoff runs.

This is an analysis utility only. It does not import or execute the simulator,
COBRA, HiGHS, or GPU code.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_pipeline_run import summarize_pipeline_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _cpu_rows(run):
    rows = {}
    for ordinal, history in enumerate(run['gpu_history']):
        stage = history['stage']
        cycle = history.get('pipeline_cycle', ordinal // 3)
        for group in history.get('groups', []):
            if group.get('route') != 'cpu_fallback':
                continue
            if len(group.get('ids', [])) != len(group.get('rows', [])):
                raise ValueError('CPU group IDs and diagnostic rows must be aligned')
            for environment_id, row in zip(
                group.get('ids', []), group.get('rows', [])
            ):
                key = (stage, int(cycle), int(environment_id))
                if key in rows:
                    raise ValueError(f'Duplicate CPU row {key}')
                rows[key] = row
    return rows


def _sum(rows, name):
    values = [float(row[name]) for row in rows]
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError(f'Nonfinite or negative row metric {name}')
    return float(sum(values))


def _action_sha256(seeds, steps):
    actions = np.stack([
        np.random.default_rng(seed).uniform(.05, .95, (120, 5))
        .astype(np.float32)[:steps]
        for seed in seeds
    ])
    return hashlib.sha256(actions.tobytes(order='C')).hexdigest()


def _paired_summary(off, on):
    # A shared status flag or zero error array alone is not sufficient:
    # recompute endpoint errors and validate every original LP certificate.
    for item in (off, on):
        summarize_pipeline_report(item)
        if item['configuration'].get('hybrid_rounds') != 0:
            raise ValueError('Handoff comparison requires zero GPU repair rounds')
    if off.get('status') != 'completed' or on.get('status') != 'completed':
        raise ValueError('Both handoff runs must be completed')
    config_keys = set(off['configuration']) | set(on['configuration'])
    config_differences = {
        key: [off['configuration'].get(key), on['configuration'].get(key)]
        for key in sorted(config_keys)
        if off['configuration'].get(key) != on['configuration'].get(key)
    }
    expected = {'output', 'cpu_basis_handoff'}
    if set(config_differences) != expected:
        raise ValueError(f'Unexpected configuration differences: {config_differences}')
    if config_differences['cpu_basis_handoff'] != [False, True]:
        raise ValueError('The pair must be ordered handoff-off, handoff-on')
    for field in (
        'source_hashes', 'model_fingerprints', 'offline_bank_manifest',
        'repair_operator_manifest',
    ):
        if not isinstance(off.get(field), dict) or not off[field]:
            raise ValueError(f'Missing paired provenance in {field}')
        if off.get(field) != on.get(field):
            raise ValueError(f'Paired provenance differs in {field}')
    if len(off['runs']) != len(on['runs']):
        raise ValueError('Paired reports contain different repeat counts')

    paired_rows = []
    run_summaries = []
    all_off_cpu_rows = []
    all_on_cpu_rows = []
    for repeat, (off_run, on_run) in enumerate(zip(off['runs'], on['runs'])):
        for field in ('seeds', 'execution_order'):
            if off_run.get(field) != on_run.get(field):
                raise ValueError(f'Repeat {repeat} differs in {field}')
        for field in ('cpu_rows', 'gpu_rows', 'errors'):
            if off_run.get(field) != on_run.get(field):
                raise ValueError(f'Repeat {repeat} endpoints differ in {field}')
        off_signature = [
            (row['stage'], row.get('pipeline_cycle'), row['routes'],
             row['accepted'], row['cpu_lp_calls'])
            for row in off_run['gpu_history']
        ]
        on_signature = [
            (row['stage'], row.get('pipeline_cycle'), row['routes'],
             row['accepted'], row['cpu_lp_calls'])
            for row in on_run['gpu_history']
        ]
        if off_signature != on_signature:
            raise ValueError(f'Repeat {repeat} routing or acceptance differs')

        off_rows, on_rows = _cpu_rows(off_run), _cpu_rows(on_run)
        if set(off_rows) != set(on_rows):
            raise ValueError(f'Repeat {repeat} CPU fallback rows differ')
        all_off_cpu_rows.extend(off_rows.values())
        all_on_cpu_rows.extend(on_rows.values())
        flags = Counter()
        used = []
        for key, on_row in on_rows.items():
            for name in ('requested', 'used', 'rejected'):
                flags[name] += int(bool(on_row.get(f'basis_override_{name}')))
            reason = on_row.get('basis_override_skipped_reason')
            if reason is not None:
                flags[f'skipped_{reason}'] += 1
            if not on_row.get('basis_override_used'):
                continue
            if on_row.get('certificate_passed') is not True:
                raise ValueError('An applied override lacks an original-LP certificate')
            off_row = off_rows[key]
            record = {
                'repeat': repeat,
                'stage': key[0],
                'cycle': key[1],
                'environment_id': key[2],
                'seed': on_run['seeds'][key[2]],
                'source': on_row.get('basis_override_source'),
                'age': on_row.get('basis_override_age'),
                'certificate_passed': bool(on_row['certificate_passed']),
                'numerical_retries': int(on_row['numerical_retry_count']),
            }
            for name in ('simplex_iterations', 'setup_seconds', 'solve_seconds'):
                record[f'off_{name}'] = off_row[name]
                record[f'on_{name}'] = on_row[name]
                record[f'delta_{name}'] = on_row[name] - off_row[name]
            record['off_setup_plus_solve_seconds'] = (
                off_row['setup_seconds'] + off_row['solve_seconds']
            )
            record['on_setup_plus_solve_seconds'] = (
                on_row['setup_seconds'] + on_row['solve_seconds']
            )
            record['delta_setup_plus_solve_seconds'] = (
                record['on_setup_plus_solve_seconds']
                - record['off_setup_plus_solve_seconds']
            )
            used.append(record)
            paired_rows.append(record)
        run_summaries.append({
            'repeat': repeat,
            'seeds': on_run['seeds'],
            'execution_order': on_run['execution_order'],
            'reconstructed_action_sha256': _action_sha256(
                on_run['seeds'], on['configuration']['steps']
            ),
            'cpu_fallback_rows': len(on_rows),
            'override_flags': dict(flags),
            'override_used_rows': len(used),
            'online_cpu_lp_calls': [
                off_run['online_cpu_lp_calls'], on_run['online_cpu_lp_calls']
            ],
            'online_cpu_solver_runs': [
                off_run['online_cpu_solver_runs'], on_run['online_cpu_solver_runs']
            ],
            'hybrid_wall_seconds': [off_run['gpu_seconds'], on_run['gpu_seconds']],
            'cpu_reference_wall_seconds': [
                off_run['cpu_seconds'], on_run['cpu_seconds']
            ],
            'endpoints_exactly_equal_between_off_and_on': True,
            'endpoint_exact_by_seed': {
                str(seed): True for seed in on_run['seeds']
            },
            'all_endpoint_gates_passed': bool(
                off_run['all_endpoint_gates_passed']
                and on_run['all_endpoint_gates_passed']
            ),
        })

    off_iterations = int(sum(row['off_simplex_iterations'] for row in paired_rows))
    on_iterations = int(sum(row['on_simplex_iterations'] for row in paired_rows))
    off_setup = _sum(paired_rows, 'off_setup_seconds')
    on_setup = _sum(paired_rows, 'on_setup_seconds')
    off_solve = _sum(paired_rows, 'off_solve_seconds')
    on_solve = _sum(paired_rows, 'on_solve_seconds')
    off_combined = off_setup + off_solve
    on_combined = on_setup + on_solve
    off_wall = sum(row['gpu_seconds'] for row in off['runs'])
    on_wall = sum(row['gpu_seconds'] for row in on['runs'])
    return {
        'scope': (
            'Paired development diagnostic. Per-row seconds are elapsed worker '
            'sums and are not additive to concurrent pipeline wall time. This '
            'report is not a speedup claim.'
        ),
        'configuration_differences': config_differences,
        'source_hashes_equal': True,
        'model_fingerprints_equal': True,
        'offline_bank_manifest_equal': True,
        'repair_operator_manifest_equal': True,
        'runs': run_summaries,
        'applied_override_rows': paired_rows,
        'aggregate_applied_override': {
            'count': len(paired_rows),
            'certificate_passed_count': sum(
                row['certificate_passed'] for row in paired_rows
            ),
            'numerical_retry_count': sum(
                row['numerical_retries'] for row in paired_rows
            ),
            'simplex_iterations': {
                'off': off_iterations,
                'on': on_iterations,
                'delta': on_iterations - off_iterations,
                'improved_rows': sum(
                    row['delta_simplex_iterations'] < 0 for row in paired_rows
                ),
                'equal_rows': sum(
                    row['delta_simplex_iterations'] == 0 for row in paired_rows
                ),
                'worsened_rows': sum(
                    row['delta_simplex_iterations'] > 0 for row in paired_rows
                ),
            },
            'elapsed_worker_seconds': {
                'setup': {'off': off_setup, 'on': on_setup,
                          'delta': on_setup - off_setup},
                'solve': {'off': off_solve, 'on': on_solve,
                          'delta': on_solve - off_solve},
                'setup_plus_solve': {
                    'off': off_combined,
                    'on': on_combined,
                    'delta': on_combined - off_combined,
                    'improved_rows': sum(
                        row['delta_setup_plus_solve_seconds'] < 0
                        for row in paired_rows
                    ),
                    'worsened_rows': sum(
                        row['delta_setup_plus_solve_seconds'] > 0
                        for row in paired_rows
                    ),
                },
            },
        },
        'aggregate_all_cpu_fallback': {
            'count': [len(all_off_cpu_rows), len(all_on_cpu_rows)],
            'simplex_iterations': {
                'off': int(_sum(all_off_cpu_rows, 'simplex_iterations')),
                'on': int(_sum(all_on_cpu_rows, 'simplex_iterations')),
                'delta': int(
                    _sum(all_on_cpu_rows, 'simplex_iterations')
                    - _sum(all_off_cpu_rows, 'simplex_iterations')
                ),
            },
            'elapsed_worker_seconds': {
                'setup': {
                    'off': _sum(all_off_cpu_rows, 'setup_seconds'),
                    'on': _sum(all_on_cpu_rows, 'setup_seconds'),
                },
                'solve': {
                    'off': _sum(all_off_cpu_rows, 'solve_seconds'),
                    'on': _sum(all_on_cpu_rows, 'solve_seconds'),
                },
            },
            'numerical_retries': [
                int(_sum(all_off_cpu_rows, 'numerical_retry_count')),
                int(_sum(all_on_cpu_rows, 'numerical_retry_count')),
            ],
        },
        'hybrid_wall_seconds': {
            'off': off_wall,
            'on': on_wall,
            'delta': on_wall - off_wall,
            'off_over_on_ratio': off_wall / on_wall,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing handoff analysis')
    with args.off.open(encoding='utf-8') as stream:
        off = json.load(stream)
    with args.on.open(encoding='utf-8') as stream:
        on = json.load(stream)
    report = _paired_summary(off, on)
    report['provenance'] = {
        'off': {'path': str(args.off), 'sha256': _sha256(args.off)},
        'on': {'path': str(args.on), 'sha256': _sha256(args.on)},
        'analysis_script': {
            'path': str(Path(__file__)), 'sha256': _sha256(Path(__file__))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(args.output)


if __name__ == '__main__':
    main()
