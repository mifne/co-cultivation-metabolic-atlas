"""Fresh/rebound GPU workspaces from ONE matched certified prior state.

Saved inputs follow a CPU-generated trajectory; this is a bounded stage-level
numerical-control diagnostic, not causal dFBA/PPO or a CPU speed comparison.
No saved reference x/y is read and no CPU optimizer is called. The same prior
GPU interior proposal is copied into both current targets. Prior acceptance
is never copied. Exact pre-solve input/operator/proof agreement is mandatory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.benchmark_ipm_sequence import validate_sequence_provenance, _workspace_identity
from scripts.benchmark_gpu_stream_partitions import _json_safe, _independent_direct_gate
from scripts.probe_downstream_gpu_coverage import paired_certificate
from src.gpu_ipm_numeric_update import _existing_host_state, _payloads, _target
from src.gpu_ipm_warm_state import coordinate_signature
from src.lp_direct_dual_audit import audit_direct_dual
from src.lp_trace import problem_hash


class MatchedStateMismatch(RuntimeError):
    """Stop before target solves when the controlled comparison is invalid."""


def array_fingerprint(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=digest.hexdigest())


def assert_exact_array(left, right, label, report, *, require_fp64=False, require_finite=False):
    """Byte equality, including signed zero; only small summaries are saved."""
    left, right = np.asarray(left), np.asarray(right)
    valid = (left.shape == right.shape and left.dtype == right.dtype)
    if require_fp64:
        valid &= left.dtype == np.float64 and right.dtype == np.float64
    if require_finite:
        valid &= bool(np.isfinite(left).all() and np.isfinite(right).all())
    equal = bool(valid and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes())
    row = dict(label=label, exact_equal=equal,
        left=array_fingerprint(left), right=array_fingerprint(right),
        require_fp64=require_fp64, require_finite=require_finite)
    report.append(row)
    if not equal:
        raise MatchedStateMismatch('Pre-solve arrays differ: '+label)


def array_difference(left, right):
    """Post-solve difference is evidence, never an acceptance criterion."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(compatible=False, left=array_fingerprint(left), right=array_fingerprint(right))
    with np.errstate(over='ignore', invalid='ignore'):
        difference = np.abs(left-right)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    return dict(compatible=True, all_finite=finite,
        bitwise_equal=np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes(),
        max_absolute_difference=float(np.max(difference, initial=0.)),
        per_environment_max_absolute_difference=(np.max(difference, axis=1).tolist()
            if difference.ndim == 2 and difference.shape[1] else [0.]*len(left)),
        left=array_fingerprint(left), right=array_fingerprint(right))


def _host(cp, value):
    return cp.asnumpy(value)


def _actual_payloads(solver):
    """Current device operators, certificates and both postsolve maps.

    The numeric-update payload inventory supplies independently reconstructed
    current host values too. Native factor RHS/solution/values are unfactored
    work buffers, not current-model data; they are recorded separately.
    """
    sparse, arrays = _payloads(solver, _existing_host_state(solver))
    actual, expected = {}, {}
    for path, host_matrix in sparse.items():
        device_matrix = _target(solver, path)
        for part in ('indptr', 'indices', 'data'):
            key = repr(path)+'.'+part
            actual[key], expected[key] = getattr(device_matrix, part), getattr(host_matrix, part)
    for path, host_value in arrays.items():
        if path == ('factor', 'values'):
            continue
        actual[repr(path)], expected[repr(path)] = _target(solver, path), host_value
    if solver.secondary_forest_plans:
        mapper = solver.secondary_forest_map
        host = mapper._host
        for name in ('transform', 'compression', 'dual_lift', 'original_at', 'transform_t'):
            device_matrix = getattr(mapper, '_'+name)
            host_matrix = host.transform.T.tocsr() if name == 'transform_t' else getattr(host, name)
            for part in ('indptr', 'indices', 'data'):
                key = 'secondary_forest.'+name+'.'+part
                actual[key], expected[key] = getattr(device_matrix, part), getattr(host_matrix, part)
        for name in ('objective', 'kept_rows', 'eliminated_rows', 'lower_witness',
                     'upper_witness', 'fallback_witness', 'weights'):
            key = 'secondary_forest.'+name
            actual[key], expected[key] = getattr(mapper, '_'+name), getattr(host, name)
    return actual, expected


def compare_targets(fresh, rebound, fresh_bound, rebound_bound, source_state, problems, report):
    """Fail closed before any target solve; no previous acceptance is used."""
    cp = fresh.cp
    fresh.factor._context()
    rebound.factor._context()
    report.update(completed=False, array_checks=[])
    if fresh.factor.factored or rebound.factor.factored:
        raise MatchedStateMismatch('Both current targets must start with invalidated numeric factors')
    if fresh.factor.device != rebound.factor.device or fresh.factor.stream.ptr != rebound.factor.stream.ptr:
        raise MatchedStateMismatch('Targets must share device and bound stream for this controlled diagnostic')
    hashes = tuple(problem_hash(p) for p in problems)
    if tuple(fresh.problem_hashes) != hashes or tuple(rebound.problem_hashes) != hashes:
        raise MatchedStateMismatch('Target original LP hashes do not match current inputs')
    signatures = [coordinate_signature(solver) for solver in (fresh, rebound)]
    if signatures[0] != signatures[1]:
        raise MatchedStateMismatch('Target coordinate signatures differ')
    if (fresh_bound.metadata['source_problem_sha256'] != rebound_bound.metadata['source_problem_sha256']
            or fresh_bound.metadata['source_step'] != rebound_bound.metadata['source_step']
            or fresh_bound.metadata['target_step'] != rebound_bound.metadata['target_step']
            or fresh_bound.metadata['environment_ids'] != rebound_bound.metadata['environment_ids']):
        raise MatchedStateMismatch('Bound proposals do not share the same causal source/target identity')
    report.update(original_problem_sha256=hashes, coordinate_signature=signatures[0],
        fresh_bound_metadata=dict(fresh_bound.metadata), rebound_bound_metadata=dict(rebound_bound.metadata))
    left = fresh_bound.initialize(fresh)
    right = rebound_bound.initialize(rebound)
    for name, a, b, source_array in zip(('x', 'y', 'z', 's'), left, right, source_state._arrays):
        if cp.may_share_memory(a, b):
            raise MatchedStateMismatch('Bound initial states must own independent arrays')
        assert_exact_array(_host(cp, a), _host(cp, b), 'initial.'+name,
            report['array_checks'], require_fp64=True, require_finite=True)
        assert_exact_array(_host(cp, a), _host(cp, source_array), 'unchanged_source.'+name,
            report['array_checks'], require_fp64=True, require_finite=True)
    for first, second in ((fresh.full_problems, rebound.full_problems),
                           (fresh.forest_problems, rebound.forest_problems),
                           (fresh.problems, rebound.problems)):
        if tuple(problem_hash(p) for p in first) != tuple(problem_hash(p) for p in second):
            raise MatchedStateMismatch('A full/forest/core LP host snapshot differs')
    fa, fe = _actual_payloads(fresh)
    ra, re = _actual_payloads(rebound)
    if fa.keys() != ra.keys() or fe.keys() != re.keys():
        raise MatchedStateMismatch('Current operator/proof payload inventories differ')
    for key in fa:
        host_fresh, host_rebound = _host(cp, fa[key]), _host(cp, ra[key])
        numerical_fp64 = np.asarray(fe[key]).dtype.kind == 'f'
        assert_exact_array(host_fresh, host_rebound, 'device_pair.'+key, report['array_checks'],
                           require_fp64=numerical_fp64)
        assert_exact_array(host_fresh, np.asarray(fe[key], dtype=host_fresh.dtype),
                           'fresh_current_host.'+key, report['array_checks'])
        assert_exact_array(host_rebound, np.asarray(re[key], dtype=host_rebound.dtype),
                           'rebound_current_host.'+key, report['array_checks'])
    # solver.values was compared above in full, including all off-diagonals.
    # Fresh cuDSS factor.values starts as pattern.ones; rebind stages new KKT
    # there. Neither is usable until the first NEW numeric factorization.
    report['unfactored_native_values_scratch_difference'] = array_difference(
        _host(cp, fresh.factor.values), _host(cp, rebound.factor.values))
    report['native_scratch_scope'] = ('factor.values/RHS/solution are not matched initial model operators; '
        'both factors are invalidated and each solve must perform new numerical factorization if needed')
    report['completed'] = True


def _state_summary(arrays):
    x, y, z, s = arrays
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        product, curvature = s*z, z/s
    return dict(fingerprints={name:array_fingerprint(value) for name,value in zip(('x','y','z','s'), arrays)},
        s_min=np.min(s, axis=1).tolist(), s_max=np.max(s, axis=1).tolist(),
        z_min=np.min(z, axis=1).tolist(), z_max=np.max(z, axis=1).tolist(),
        mean_complementarity=np.mean(product, axis=1).tolist(),
        z_over_s_max=np.max(curvature, axis=1).tolist(), z_over_s_min=np.min(curvature, axis=1).tolist())


def solve_and_audit(solver, problems, *, iterations, bound=None):
    cp = solver.cp
    cp.cuda.get_current_stream().synchronize()
    stamp = time.perf_counter()
    result = solver.solve(iterations=iterations, internal_warm_start=bound)
    cp.cuda.get_current_stream().synchronize()
    wall = time.perf_counter()-stamp
    stamp = time.perf_counter()
    x, y = result['x'].get(), result['y'].get()
    certificates = [paired_certificate(p, xx, yy) for p,xx,yy in zip(problems,x,y)]
    direct = [{k:np.asarray(v).tolist() for k,v in audit_direct_dual(p, xx, yy, xp=np).items()}
              for p,xx,yy in zip(problems,x,y)]
    qualified = bool(result['accepted'].all() and all(row['certificate_passed'] for row in certificates)
                     and all(_independent_direct_gate(row) for row in direct))
    internal = getattr(solver, '_last_internal_state', None)
    internal = None if internal is None else tuple(v.get() for v in internal)
    report = dict(solve_api_seconds=result['total_seconds'], externally_synchronized_solve_wall_seconds=wall,
        all_original_certificates_passed=qualified, independent_original_certificates=certificates,
        independent_direct_dual_audits=direct, current_problem_sha256=tuple(solver.problem_hashes),
        internal_state_summary=None if internal is None else _state_summary(internal),
        D2H_and_independent_audit_seconds=time.perf_counter()-stamp)
    report['solver_result'] = {key:value for key,value in result.items() if not isinstance(value, cp.ndarray)}
    return report, dict(x=x, y=y, internal=internal)


def solver_options(args):
    return dict(regularization=1e-6, globalized=True, forcing_eta=.1, regularization_retries=1,
        newton_krylov_iterations=16, predictor_corrector=True, predictor_affine_fraction=.995,
        ipm_initialization='balanced', exact_equalities=True, allow_box_dual=True,
        second_forest=args.second_forest, fix_singleton_equalities=args.fix_singleton_equalities,
        device_checked_solves=True, factor_refinements=0, krylov_microkernels='all',
        krylov_defer_lane_checks=True, reuse_gmres_workspace=args.reuse_gmres_workspace,
        retain_internal_state=True)


def _own_solver(stack, factory, problems, options, record, label):
    solver = factory(problems, **options)
    def close():
        try:
            solver.close()
        finally:
            record['close_attempted_solver_labels'].append(label)
    stack.callback(close)
    record['created_solver_labels'].append(label)
    return solver


def run_matched(args, inputs, record):
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    from src.gpu_ipm_numeric_update import rebind_forest_ipm
    options = solver_options(args)
    record['solver_options'] = options
    record['device'] = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name'].decode()
    order = ['fresh', 'rebind'] if args.order == 'fresh-first' else ['rebind', 'fresh']
    record.update(execution_order=order, source_steps=[], targets={},
        created_solver_labels=[], close_attempted_solver_labels=[], pre_solve_match={})
    with ExitStack() as owners:
        first_problems = inputs[0][0]
        stamp = time.perf_counter()
        source = _own_solver(owners, ForestGpuBatchedIPM, first_problems, options, record, 'source_rebind')
        record['source_constructor_seconds'] = time.perf_counter()-stamp
        previous = None
        for step, (problems, provenance) in zip(range(2, args.target_step), inputs[:-1]):
            trial = dict(step=step, original_problem_sha256=provenance['problem_sha256'])
            if previous is not None:
                stamp = time.perf_counter()
                trial['numeric_update'] = rebind_forest_ipm(source, problems)
                trial['numeric_update_wall_seconds'] = time.perf_counter()-stamp
                bound = previous.bind(source, environment_ids=list(range(args.batch)), stage=args.stage,
                    step=step, interior_floor=0., repair_slacks=False)
            else:
                bound = None
            result, _pairs = solve_and_audit(source, problems, iterations=args.iterations, bound=bound)
            trial.update(result)
            record['source_steps'].append(trial)
            print(json.dumps(dict(role='source', step=step, qualified=trial['all_original_certificates_passed'],
                seconds=trial['solve_api_seconds'])), flush=True)
            if not trial['all_original_certificates_passed']:
                raise RuntimeError('Source LP was not certified; matched target comparison is prohibited')
            stamp = time.perf_counter()
            previous = source.export_internal_state(environment_ids=list(range(args.batch)),
                stage=args.stage, step=step)
            trial['export_seconds'] = time.perf_counter()-stamp
        target_problems = inputs[-1][0]
        solvers = {}
        for method in order:
            stamp = time.perf_counter()
            if method == 'fresh':
                solvers[method] = _own_solver(owners, ForestGpuBatchedIPM, target_problems,
                    options, record, 'fresh_target')
                metadata = dict(kind='fresh_constructor')
            else:
                metadata = dict(kind='numeric_rebind', update=rebind_forest_ipm(source, target_problems))
                solvers[method] = source
            metadata.update(preparation_seconds=time.perf_counter()-stamp,
                workspace_identity=_workspace_identity(solvers[method]))
            record['targets'][method] = metadata
        stamp = time.perf_counter()
        bounds = {method: previous.bind(solver, environment_ids=list(range(args.batch)), stage=args.stage,
            step=args.target_step, interior_floor=0., repair_slacks=False) for method,solver in solvers.items()}
        record['both_bindings_seconds'] = time.perf_counter()-stamp
        stamp = time.perf_counter()
        compare_targets(solvers['fresh'], solvers['rebind'], bounds['fresh'], bounds['rebind'],
                        previous, target_problems, record['pre_solve_match'])
        record['pre_solve_match_seconds'] = time.perf_counter()-stamp
        outcomes = {}
        for method in order:
            report, outcomes[method] = solve_and_audit(solvers[method], target_problems,
                iterations=args.iterations, bound=bounds[method])
            record['targets'][method].update(report)
            print(json.dumps(dict(role='target', method=method, step=args.target_step,
                qualified=report['all_original_certificates_passed'], seconds=report['solve_api_seconds'],
                factors=report['solver_result']['factor_count'])), flush=True)
        record['post_solve_differences'] = {name:array_difference(outcomes['fresh'][name], outcomes['rebind'][name])
                                           for name in ('x', 'y')}
        if all(outcomes[method]['internal'] is not None for method in order):
            record['post_solve_differences']['internal'] = {name:array_difference(a, b)
                for name,a,b in zip(('x','y','z','s'),outcomes['fresh']['internal'],outcomes['rebind']['internal'])}
        record['all_target_original_certificates_passed'] = all(
            record['targets'][method]['all_original_certificates_passed'] for method in order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--stage', choices=['maxmin', 'aggregate', 'exchange'], default='maxmin')
    parser.add_argument('--batch', type=int, choices=[4], default=4)
    parser.add_argument('--target-step', type=int, choices=[3, 4], default=3)
    parser.add_argument('--order', choices=['fresh-first', 'rebind-first'], default='fresh-first')
    parser.add_argument('--iterations', type=int, default=240)
    parser.add_argument('--second-forest', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fix-singleton-equalities', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reuse-gmres-workspace', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Matched-state results are never overwritten')
    if not 0 <= args.iterations <= 1000:
        raise ValueError('Bounded iteration count 0..1000 required')
    steps = list(range(2, args.target_step+1))
    stamp = time.perf_counter()
    inputs = [load_inputs(args.trace, args.stage, step, args.batch) for step in steps]
    raw = (args.trace/'manifest.json').read_bytes()
    identity = validate_sequence_provenance([p for _,p in inputs], steps, args.stage, args.batch, raw)
    record = dict(role='matched_GPU_interior_fresh_vs_rebind_numerical_control_NOT_CPU_speed_or_closed_loop',
        configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source_start_step=2, sequence_identity=identity, input_provenance=[p for _,p in inputs],
        input_load_seconds=time.perf_counter()-stamp, reference_vectors_loaded=False, cpu_lp_calls=0,
        previous_acceptance_copied=False, interior_floor=0., repair_slacks=False,
        completed=False, all_target_original_certificates_passed=False,
        timing_scope='Setup, numeric update, matching checks, solve APIs and independent audits are separate; single diagnostic, no CPU speed claim')
    paths = ['scripts/probe_ipm_rebind_matched_state.py', 'scripts/benchmark_ipm_sequence.py',
        'scripts/benchmark_ipm_cpu_inputs.py', 'scripts/benchmark_gpu_stream_partitions.py',
        'scripts/analyze_coverage_temporal_routing.py', 'scripts/probe_downstream_gpu_coverage.py',
        'src/gpu_ipm_warm_state.py', 'src/gpu_ipm_numeric_update.py', 'src/gpu_batched_ipm.py',
        'src/gpu_condensed_ipm.py', 'src/gpu_globalized_ipm.py', 'src/gpu_globalized_condensed.py',
        'src/gpu_newton_krylov.py', 'src/gpu_zero_face_ipm.py', 'src/gpu_forest_ipm.py',
        'src/gpu_forest_map.py', 'src/lp_zero_face.py', 'src/lp_equality_reduction.py',
        'src/lp_exact_equalities.py', 'src/gpu_sparse_factor.py', 'src/gpu_block_lp.py',
        'src/gpu_ipm_initialization.py', 'src/gpu_krylov_microkernels.py',
        'src/lp_direct_dual_audit.py', 'src/cpu_repeated_lp.py', 'src/lp_trace.py']
    record['source_snapshots'] = {path:(ROOT/path).read_text() for path in paths}
    record['source_sha256'] = {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
    started = time.perf_counter()
    failure = None
    try:
        run_matched(args, inputs, record)
        record['completed'] = True
    except Exception as error:
        failure = error
        record['failure'] = dict(type=type(error).__name__, message=str(error),
                                notes=list(getattr(error, '__notes__', ())))
    record['full_diagnostic_lifecycle_seconds'] = time.perf_counter()-started
    serialized = json.dumps(_json_safe(record), indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(serialized)
    print('Saved '+str(args.output), flush=True)
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
