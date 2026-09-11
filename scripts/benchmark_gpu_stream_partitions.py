"""Same distinct LPs, sequential cuDSS setup, cooperative nonblocking streams.

This is a frozen-input stage benchmark, NOT causal dFBA/PPO or complete-device
execution. All returned full-original pairs are independently host-certified.
No CPU LP is solved and no stored reference solution is read.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.probe_downstream_gpu_coverage import paired_certificate
from src.gpu_stream_partition import partition_indices, solve_cooperatively
from src.lp_direct_dual_audit import audit_direct_dual


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value): return float(value)
        return 'NaN' if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def _ordered_rows(shard_rows, partitions, batch):
    """Reject missing/duplicate/mislabeled outputs, preserving exact env order."""
    expected = partition_indices(batch, partitions)
    if len(shard_rows) != partitions:
        raise ValueError('Missing shard output')
    rows = [None] * batch
    for group, indices in zip(shard_rows, expected):
        if len(group) != len(indices):
            raise ValueError('Incomplete shard output')
        for row, environment in zip(group, indices):
            if row['environment_id'] != environment or rows[environment] is not None:
                raise ValueError('Missing, duplicate or mislabeled original environment')
            rows[environment] = row
    return rows


def _independent_direct_gate(audit):
    """Mirror the additional backend direct-gap gate, including its scale."""
    primal = np.asarray(audit['primal_objective'])
    dual = np.asarray(audit['dual_objective'])
    gap = np.asarray(audit['signed_gap'])
    ratio = np.abs(gap) / np.maximum(1., np.abs(primal))
    return bool(np.all(np.isfinite(dual) & np.isfinite(ratio) & (ratio <= 1e-7)))


def run_trial(problems, *, partitions, mode, options, iterations, check_interval,
              yield_mode, index):
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    indices = partition_indices(len(problems), partitions)
    if mode not in ('direct', 'cooperative') or (mode == 'direct' and partitions != 1):
        raise ValueError('Direct control uses one batch/stream')
    streams, solvers, shard_results = [], [], []
    lifecycle_start = time.perf_counter()
    cleanup_errors = []
    try:
        setup_start = time.perf_counter()
        # All analysis and constructors execute serially, before any LP solve.
        for group in indices:
            stream = cp.cuda.Stream(non_blocking=True)
            streams.append(stream)
            with stream:
                solver = ForestGpuBatchedIPM([problems[i] for i in group], **options)
                solvers.append(solver)
        for stream in streams:
            stream.synchronize()
        setup_seconds = time.perf_counter() - setup_start
        kwargs = dict(iterations=iterations, check_interval=check_interval)
        solve_start = time.perf_counter()
        if mode == 'cooperative':
            shard_results, scheduling = solve_cooperatively(solvers, streams,
                solve_kwargs=kwargs, yield_mode=yield_mode)
        else:
            with streams[0]:
                shard_results = [solvers[0].solve(**kwargs)]
            scheduling = dict(execution_mode='direct_single_workspace_control', partitions=1,
                owner_thread_id=solvers[0].factor.thread, stream_pointers=[streams[0].ptr],
                per_solver_timings_include_yield_time=False)
        for stream in streams:
            stream.synchronize()
        solve_seconds = time.perf_counter() - solve_start
        transfer_start = time.perf_counter()
        host_pairs = []
        for result, stream in zip(shard_results, streams):
            with stream:
                host_pairs.append((result.pop('x').get(), result.pop('y').get()))
                for key in ('reduced_x', 'reduced_y', 'forest_x', 'forest_y'):
                    result.pop(key, None)
        transfer_seconds = time.perf_counter() - transfer_start
        verification_start = time.perf_counter()
        shard_rows = []
        require_direct = bool(options.get('second_forest') or options.get('bounded_near_equality')
                              or options.get('fix_singleton_equalities'))
        from src.lp_trace import problem_hash
        for group, result, (x, y) in zip(indices, shard_results, host_pairs):
            if len(x) != len(group) or len(y) != len(group):
                raise ValueError('Incomplete full-original returned pair')
            rows = []
            for lane, environment in enumerate(group):
                p = problems[environment]
                certificate = paired_certificate(p, x[lane], y[lane])
                direct = audit_direct_dual(p, x[lane], y[lane], xp=np)
                direct_host = {key: np.asarray(value).tolist() for key, value in direct.items()}
                backend = result['metrics'][lane]
                host_direct_passed = _independent_direct_gate(direct)
                extra_direct_passed = (not require_direct or
                    (bool(backend.get('direct_dual_gate_passed', False)) and host_direct_passed))
                qualified = bool(result['accepted'][lane] and backend['certificate_passed']
                    and certificate['certificate_passed'] and extra_direct_passed
                    and np.isfinite(x[lane]).all() and np.isfinite(y[lane]).all())
                rows.append(dict(environment_id=environment, problem_sha256=problem_hash(p),
                    qualified=qualified, backend_accepted=bool(result['accepted'][lane]),
                    backend_original_certificate=backend,
                    independent_host_original_certificate=certificate,
                    extra_direct_gap_gate_required=require_direct,
                    independent_host_direct_gap_passed=host_direct_passed,
                    independent_host_direct_dual_audit=direct_host))
            shard_rows.append(rows)
        rows = _ordered_rows(shard_rows, partitions, len(problems))
        verification_seconds = time.perf_counter() - verification_start
        qualified = all(row['qualified'] for row in rows)
        trial = dict(index=index, partitions=partitions, batch=len(problems),
            shard_environment_ids=indices, mode=mode,
            all_original_certificates_passed=qualified,
            original_pair_count=len(rows), rows=rows,
            setup_all_shards_seconds=setup_seconds,
            solve_all_shards_synchronized_wall_seconds=solve_seconds,
            original_pair_d2h_seconds=transfer_seconds,
            independent_host_verification_seconds=verification_seconds,
            solve_transfer_verification_seconds=solve_seconds+transfer_seconds+verification_seconds,
            certified_lp_per_solve_second=(len(problems)/solve_seconds if qualified else None),
            scheduling=scheduling, shard_solver_results=shard_results,
            source_of_solver_phase_times=('inclusive wall with time yielded to other shards; do not sum'
                if mode == 'cooperative' else 'single workspace wall diagnostics'),
            cudss_versions=[solver.factor.version for solver in solvers],
            factor_analyses=[solver.factor.analysis_count for solver in solvers],
            factor_dimensions=[solver.factor.n for solver in solvers],
            factor_nnz=[solver.factor.nnz for solver in solvers],
            cupy_pool_used_bytes_before_close=cp.get_default_memory_pool().used_bytes(),
            cupy_pool_reserved_bytes_before_close=cp.get_default_memory_pool().total_bytes())
    finally:
        close_start = time.perf_counter()
        for solver, stream in zip(solvers, streams):
            try:
                with stream:
                    solver.close()
            except BaseException as error:
                cleanup_errors.append(error)
        for stream in streams:
            try:
                stream.synchronize()
            except BaseException as error:
                cleanup_errors.append(error)
        close_seconds = time.perf_counter() - close_start
        if cleanup_errors:
            raise BaseExceptionGroup('Partition trial cleanup failed', cleanup_errors)
    trial['close_seconds'] = close_seconds
    trial['solver_lifecycle_wall_seconds'] = time.perf_counter()-lifecycle_start
    trial['lifecycle_scope'] = ('all sequential construction/analysis, cooperative or direct solve, '
        'full pair D2H, independent host verification and close; excludes imports/input load/serialization')
    return trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--stage', choices=['maxmin', 'aggregate', 'exchange'], default='maxmin')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--batch', type=int, choices=[1, 2, 4, 8, 16, 32], default=32)
    parser.add_argument('--partitions', nargs='+', type=int, choices=[1, 2, 4], default=[1, 2, 4])
    parser.add_argument('--mode', choices=['direct', 'cooperative'], default='cooperative')
    parser.add_argument('--yield-mode', choices=['factor-only', 'factor-and-solve'], default='factor-and-solve')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=240)
    parser.add_argument('--check-interval', type=int, default=1)
    parser.add_argument('--krylov', type=int, default=16)
    parser.add_argument('--factor-refinements', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--krylov-defer-lane-checks', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--second-forest', action='store_true')
    parser.add_argument('--fix-singleton-equalities', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Never overwrite earlier experiments')
    if (not 1 <= args.repeats <= 20 or args.iterations < 1 or args.check_interval < 1
            or len(set(args.partitions)) != len(args.partitions)
            or (args.mode == 'direct' and args.partitions != [1])):
        raise ValueError('Bounded repeats, valid budgets, unique partitions; direct control uses --partitions 1')
    for partitions in args.partitions:
        partition_indices(args.batch, partitions)
    load_start = time.perf_counter()
    problems, provenance = load_inputs(args.trace, args.stage, args.step, args.batch)
    input_load_seconds = time.perf_counter()-load_start
    # Only the common validated input loader is reused, not CPU trial labels.
    provenance.pop('cpu_solution_source', None)
    provenance.pop('gpu_calls', None)
    provenance['solution_source'] = 'current_GPU_trial_only_no_CPU_reference_or_optimizer'
    import cupy as cp
    options = dict(regularization=1e-6, matrix_type='symmetric', globalized=True,
        forcing_eta=.1, regularization_retries=1, newton_krylov_iterations=args.krylov,
        krylov_coordinates='full', krylov_microkernels='all', device_checked_solves=True,
        factor_refinements=args.factor_refinements,
        krylov_defer_lane_checks=args.krylov_defer_lane_checks,
        predictor_corrector=True, predictor_affine_fraction=.995,
        ipm_initialization='balanced', exact_equalities=True, allow_box_dual=True,
        second_forest=args.second_forest, fix_singleton_equalities=args.fix_singleton_equalities)
    source_paths = ('scripts/benchmark_gpu_stream_partitions.py', 'src/gpu_stream_partition.py',
        'scripts/benchmark_ipm_cpu_inputs.py', 'scripts/analyze_coverage_temporal_routing.py',
        'scripts/probe_downstream_gpu_coverage.py', 'src/gpu_sparse_factor.py',
        'src/gpu_batched_ipm.py', 'src/gpu_condensed_ipm.py', 'src/gpu_newton_krylov.py',
        'src/gpu_krylov_microkernels.py', 'src/gpu_globalized_ipm.py',
        'src/gpu_globalized_condensed.py', 'src/gpu_forest_ipm.py', 'src/gpu_zero_face_ipm.py',
        'src/gpu_forest_map.py',
        'src/gpu_ipm_initialization.py', 'src/lp_zero_face.py', 'src/lp_exact_equalities.py',
        'src/lp_equality_reduction.py', 'src/gpu_block_lp.py', 'src/lp_direct_dual_audit.py')
    record = dict(role='development_same_input_GPU_stream_partition_not_dfba_or_PPO',
        stage=args.stage, step=args.step, batch=args.batch,
        configuration=dict(solver=options, mode=args.mode, yield_mode=args.yield_mode,
            partitions=args.partitions, repeats=args.repeats, iterations=args.iterations,
            check_interval=args.check_interval),
        environment=dict(device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            cupy=cp.__version__, python=sys.version, greenlet=version('greenlet'),
            thread_environment={k: os.environ.get(k) for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}),
        current_reference_vectors_loaded=False, cpu_lp_calls=0,
        original_certificate_limits=dict(primal_residual=1e-5, dual_violation=1e-7, relative_kkt_gap=1e-7),
        timing_caution='Fresh workspaces, not cold OS/kernel cache. No solution warm starts. Setup reported separately.',
        sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        source_snapshots={p: (ROOT/p).read_text() for p in source_paths},
        input_load_seconds=input_load_seconds, input_provenance=provenance, trials=[])
    # Reverse configuration order on alternating rounds to expose order effects.
    for repeat in range(args.repeats):
        order = args.partitions if repeat % 2 == 0 else list(reversed(args.partitions))
        for partitions in order:
            print(json.dumps(dict(status='starting', repeat=repeat, partitions=partitions,
                stage=args.stage, batch=args.batch)), flush=True)
            trial = run_trial(problems, partitions=partitions, mode=args.mode, options=options,
                iterations=args.iterations, check_interval=args.check_interval,
                yield_mode=args.yield_mode, index=repeat)
            record['trials'].append(trial)
            print(json.dumps(dict(status='trial_complete', repeat=repeat, partitions=partitions,
                seconds=trial['solve_all_shards_synchronized_wall_seconds'],
                all_certified=trial['all_original_certificates_passed'])), flush=True)
    record['groups'] = []
    for partitions in args.partitions:
        trials = [t for t in record['trials'] if t['partitions'] == partitions]
        times = [t['solve_all_shards_synchronized_wall_seconds'] for t in trials]
        qualified = all(t['all_original_certificates_passed'] for t in trials)
        record['groups'].append(dict(partitions=partitions, shard_batch=args.batch//partitions,
            solve_wall_seconds=times, median_solve_seconds=float(np.median(times)),
            all_trials_qualified=qualified,
            certified_lp_per_median_second=args.batch/float(np.median(times)) if qualified else None))
    record['completed'] = True
    serialized = json.dumps(_json_safe(record), indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(serialized)
    print('Saved '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
