"""Matched GPU restart-mu experiment, using one certified step-2 source.

This replays recorded LP inputs, not a closed-loop dFBA/PPO trajectory. Every
step-3 arm receives the SAME GPU-generated interior bundle; no current CPU
solution is loaded. Only qualified step-3 arms continue with their OWN state
to step 4. Setup, numeric update, binding, solve and verification are separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_ipm_cpu_inputs import load_inputs
from scripts.benchmark_ipm_sequence import _GpuSequenceWorkspace, validate_sequence_provenance
from scripts.benchmark_gpu_stream_partitions import _json_safe, _independent_direct_gate
from scripts.probe_downstream_gpu_coverage import paired_certificate
from src.lp_direct_dual_audit import audit_direct_dual


SOURCE_PATHS = (
    'scripts/probe_ipm_restart_mu_matched.py', 'scripts/benchmark_ipm_sequence.py',
    'scripts/benchmark_ipm_cpu_inputs.py', 'scripts/benchmark_gpu_stream_partitions.py',
    'scripts/probe_downstream_gpu_coverage.py', 'src/lp_direct_dual_audit.py',
    'src/gpu_ipm_warm_state.py', 'src/gpu_ipm_numeric_update.py',
    'src/gpu_ipm_reoptimization.py', 'src/gpu_ipm_staging.py',
    'src/csr_block_assembly.py', 'src/gpu_ipm_kkt_payload.py',
    'src/gpu_batched_ipm.py', 'src/gpu_condensed_ipm.py',
    'src/gpu_globalized_ipm.py', 'src/gpu_globalized_condensed.py',
    'src/gpu_newton_krylov.py', 'src/gpu_zero_face_ipm.py',
    'src/gpu_forest_ipm.py', 'src/gpu_forest_map.py', 'src/lp_zero_face.py',
    'src/lp_equality_reduction.py', 'src/lp_exact_equalities.py',
    'src/gpu_sparse_factor.py', 'src/gpu_block_lp.py',
    'src/gpu_ipm_initialization.py', 'src/gpu_krylov_microkernels.py',
    'src/gpu_primal_extrapolation.py')


class SourceStateChanged(RuntimeError):
    """Matched-arm validity is lost; never restore and silently continue."""


class _SourceGuard:
    """Owned device snapshots; compare all four arrays without vector D2H."""

    def __init__(self, state, cp):
        self.state, self.cp = state, cp
        self.arrays = tuple(state._arrays)
        if len(self.arrays) != 4:
            raise ValueError('Exactly four exported interior arrays required')
        self.snapshots = tuple(value.copy() for value in self.arrays)
        self.metadata = self._metadata()

    def _metadata(self):
        return (self.state.signature, self.state.environment_ids, self.state.stage,
                self.state.step, self.state.source_problem_hashes)

    def check(self):
        arrays = self.state._arrays
        if (len(arrays) != 4 or any(a is not b for a, b in zip(arrays, self.arrays))
                or self._metadata() != self.metadata):
            raise SourceStateChanged('The shared source bundle identity or provenance changed')
        same = [bool(self.cp.array_equal(a, b)) for a, b in zip(arrays, self.snapshots)]
        if not all(same):
            raise SourceStateChanged('The shared GPU source x/y/z/s was mutated')
        return dict(x_unchanged=same[0], y_unchanged=same[1], z_unchanged=same[2],
                    s_unchanged=same[3], vector_D2H_copies=0)


def _phase(trial, name, operation, synchronize):
    before = time.perf_counter()
    try:
        value = operation()
        synchronize()  # Attribute asynchronous device work to THIS phase.
        return value
    finally:
        trial[name] = time.perf_counter() - before


def _options():
    # Match the successful sequence control, without changing any gate.
    return dict(regularization=1e-6, globalized=True, forcing_eta=.1,
        regularization_retries=1, newton_krylov_iterations=16,
        predictor_corrector=True, predictor_affine_fraction=.995,
        ipm_initialization='balanced', exact_equalities=True, allow_box_dual=True,
        second_forest=True, fix_singleton_equalities=True, device_checked_solves=True,
        factor_refinements=0, krylov_microkernels='all', krylov_defer_lane_checks=True,
        reuse_gmres_workspace=True, retain_internal_state=True)


def _verify(problems, result, batch):
    x, y = result['x'].get(), result['y'].get()
    accepted = result['accepted']
    accepted = np.asarray(accepted.get() if hasattr(accepted, 'get') else accepted)
    if (len(problems) != batch or accepted.shape != (batch,)
            or x.shape != (batch, problems[0][4].size)
            or y.shape != (batch, problems[0][0].shape[0])):
        raise ValueError('Incomplete or mis-shaped original-coordinate GPU batch')
    certificates = [paired_certificate(p, xx, yy) for p, xx, yy in zip(problems, x, y)]
    direct = [{key: np.asarray(value).tolist() for key, value in
               audit_direct_dual(p, xx, yy, xp=np).items()}
              for p, xx, yy in zip(problems, x, y)]
    original_pass = all(row['certificate_passed'] for row in certificates)
    direct_pass = all(_independent_direct_gate(row) for row in direct)
    return dict(all_original_certificates_passed=bool(original_pass),
        all_direct_dual_gates_passed=bool(direct_pass),
        all_gpu_acceptance_flags_passed=bool(accepted.all()),
        qualified=bool(accepted.all() and original_pass and direct_pass),
        independent_host_certificates=certificates, independent_direct_dual_audits=direct)


def _stage(trial, workspace, args, item, *, cp, synchronize, previous=None,
           source_guard=None, mu=None, export=False):
    problems, provenance = item
    trial.update(status='running', qualified=False, successful_solve_seconds=None,
                 problem_sha256=provenance['problem_sha256'])
    solver, metadata = _phase(trial, 'workspace_prepare_wall_seconds',
        lambda: workspace.prepare(problems), synchronize)
    trial.update(metadata)
    if tuple(solver.problem_hashes) != tuple(provenance['problem_sha256']):
        raise RuntimeError('Prepared solver does not contain the CURRENT original LP inputs')
    bound = None
    trial['binding_seconds'] = trial['source_guard_seconds'] = 0.
    if previous is not None:
        trial['previous_bundle_object_id'] = id(previous)
        # Rejected binding is an experiment failure, NOT a cold-solve arm.
        bound = _phase(trial, 'binding_seconds', lambda: previous.bind(solver,
            environment_ids=list(range(args.batch)), stage=args.stage, step=trial['step'],
            interior_floor=0., repair_slacks=False, restart_mu=mu), synchronize)
        trial['warm_binding'] = bound.metadata
    if source_guard is not None:
        trial['source_after_bind'] = _phase(trial, 'source_guard_seconds',
            source_guard.check, synchronize)
    trial['solve_called'] = True
    result = _phase(trial, 'solve_wall_seconds', lambda: solver.solve(
        iterations=args.iterations, internal_warm_start=bound), synchronize)
    trial['solve_api_seconds'] = result['total_seconds']
    # Keep unsuccessful solver diagnostics too, without retaining device arrays.
    trial['solver_result'] = {key: value for key, value in result.items()
                              if not isinstance(value, cp.ndarray)}
    if source_guard is not None:
        before = trial['source_guard_seconds']
        trial['source_after_solve'] = _phase(trial, 'source_guard_seconds',
            source_guard.check, synchronize)
        trial['source_guard_seconds'] += before
    verification = _phase(trial, 'D2H_and_independent_verification_seconds',
        lambda: _verify(problems, result, args.batch), synchronize)
    trial.update(verification)
    owned = None
    trial['export_recertification_and_snapshot_seconds'] = 0.
    if trial['qualified'] and export:
        owned = _phase(trial, 'export_recertification_and_snapshot_seconds',
            lambda: solver.export_internal_state(environment_ids=list(range(args.batch)),
                stage=args.stage, step=trial['step']), synchronize)
    if source_guard is not None:
        before = trial['source_guard_seconds']
        trial['source_after_export'] = _phase(trial, 'source_guard_seconds',
            source_guard.check, synchronize)
        trial['source_guard_seconds'] += before
    trial['symbolic_analysis_count_after_solve'] = int(solver.factor.analysis_count)
    trial['cumulative_symbolic_analysis_count'] = workspace.observe_analysis()
    trial['status'] = 'qualified' if trial['qualified'] else 'unqualified'
    if trial['qualified']:
        trial['successful_solve_seconds'] = trial['solve_wall_seconds']
    return owned


def _workspace_summary(workspace):
    return dict(workspace_creations=workspace.created, workspace_close_calls=workspace.closed,
        workspace_rebuilds=max(0, workspace.created - 1),
        numeric_update_attempts=workspace.update_attempts,
        numeric_update_successes=workspace.update_successes,
        numeric_update_rejections=workspace.rejections,
        actual_symbolic_analysis_count=workspace.analysis_total)


def _run_matched(record, args, inputs, *, cp, solver_factory, rebind_workspace, synchronize):
    """Dependency-injected orchestration, also exercised by CPU-only contracts."""
    if len(inputs) != 3:
        raise ValueError('Exactly steps 2, 3 and 4 are required')
    record.update(source=dict(step=2), arms=[], cpu_lp_calls=0,
                  source_solve_calls=0, completed=False, all_requested_arms_qualified=False)
    source, guard = None, None
    active_trial = record['source']
    options = _options()
    def update(solver, problems):
        updates = dict(reuse_static_forest=True, device_staging=args.device_staging)
        if getattr(args, 'direct_kkt_payload', False):
            updates['direct_kkt_payload'] = True
        return rebind_workspace(solver, problems, **updates)
    def workspace():
        return _GpuSequenceWorkspace(solver_factory, options, reuse=True, rebind=update)
    experiment_start = time.perf_counter()
    try:
        source_start = time.perf_counter()
        owner = workspace()
        try:
            source = _stage(active_trial, owner, args, inputs[0], cp=cp,
                synchronize=synchronize, export=True)
            if source is not None:
                guard = _phase(active_trial, 'source_immutable_snapshot_seconds',
                    lambda: _SourceGuard(source, cp), synchronize)
                record['matched_source'] = dict(bundle_object_id=id(source),
                    step=source.step, coordinate_signature=source.signature,
                    source_problem_sha256=source.source_problem_hashes,
                    original_certificate_basis=source.source_certificate_basis,
                    native_interior_duals_are_not_assumed_certified=True,
                    immutable_arrays_checked=['x', 'y', 'z', 's'])
        finally:
            try:
                active_trial['workspace_close_seconds'] = owner.close()
            finally:
                active_trial['full_lifecycle_seconds'] = time.perf_counter() - source_start
                active_trial['workspace_summary'] = _workspace_summary(owner)
                record['source_solve_calls'] = int(active_trial.get('solve_called', False))
        if source is None:
            record['abort_reason'] = 'Source step 2 failed an independent gate; no arms started'
            return
        for mu in args.mus:
            arm = dict(mu=float(mu), status='running', qualified=False, steps=[],
                       successful_sequence_solve_seconds=None)
            record['arms'].append(arm)
            arm_start = time.perf_counter()
            owner = workspace()  # Independent native workspace for EVERY arm.
            previous = source
            try:
                for step, item in zip((3, 4), inputs[1:]):
                    active_trial = dict(step=step)
                    arm['steps'].append(active_trial)
                    stamp = time.perf_counter()
                    try:
                        previous = _stage(active_trial, owner, args, item, cp=cp,
                            synchronize=synchronize, previous=previous, source_guard=guard,
                            mu=mu, export=step == 3)
                    finally:
                        active_trial['full_step_seconds_excluding_workspace_close'] = (
                            time.perf_counter() - stamp)
                    if not active_trial['qualified']:
                        break
                arm['qualified'] = (len(arm['steps']) == 2 and
                                    all(row['qualified'] for row in arm['steps']))
                arm['status'] = 'qualified' if arm['qualified'] else 'unqualified'
                arm['source_after_arm'] = _phase(arm, 'source_guard_seconds',
                    guard.check, synchronize)
                if arm['qualified']:
                    arm['successful_sequence_solve_seconds'] = sum(
                        row['solve_wall_seconds'] for row in arm['steps'])
            finally:
                try:
                    arm['workspace_close_seconds'] = owner.close()
                finally:
                    arm['full_lifecycle_seconds'] = time.perf_counter() - arm_start
                    arm['workspace_summary'] = _workspace_summary(owner)
            print(json.dumps(dict(mu=mu, status=arm['status'],
                factor_counts=[s.get('solver_result', {}).get('factor_count') for s in arm['steps']],
                successful_solve_seconds=arm['successful_sequence_solve_seconds'])), flush=True)
        record['completed'] = True
        record['all_requested_arms_qualified'] = all(arm['qualified'] for arm in record['arms'])
    except Exception as error:
        # Native/runtime failure can poison the context. Close, record, abort;
        # do NOT silently retry or time later arms in a questionable context.
        record['error'] = dict(type=type(error).__name__, message=str(error))
        active_trial.update(status='error', qualified=False, successful_solve_seconds=None)
        if record['arms']:
            record['arms'][-1].update(status='error', qualified=False,
                                     successful_sequence_solve_seconds=None)
        record['abort_reason'] = 'Exception; remaining arms were not attempted'
    finally:
        record['experiment_lifecycle_seconds'] = time.perf_counter() - experiment_start
        record['unattempted_mus'] = list(args.mus[len(record['arms']):])
        record['matched_source_preservation_verified'] = bool(
            guard is not None and not record.get('error') and record['completed'])


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT/'results/pf_lp_trace_dev32x41_20260905')
    parser.add_argument('--batch', type=int, choices=[4, 32], default=4)
    parser.add_argument('--stage', choices=['maxmin', 'aggregate', 'exchange'], default='maxmin')
    parser.add_argument('--mus', type=float, nargs='+', default=[1e-3, 1e-4, 1e-5, 1e-6])
    parser.add_argument('--iterations', type=int, default=240)
    parser.add_argument('--device-staging', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--direct-kkt-payload', action='store_true',
                        help='Opt-in numeric update payload path; identical for all mu arms')
    parser.add_argument('--output', type=Path, required=True)
    return parser


def validate_args(args):
    if (not args.mus or len(set(args.mus)) != len(args.mus)
            or any(not np.isfinite(mu) or not 1e-8 <= mu <= 1e-2 for mu in args.mus)
            or type(args.iterations) is not int or not 1 <= args.iterations <= 1000):
        raise ValueError('Distinct finite mu values in [1e-8, 1e-2] and 1..1000 iterations required')


def _write_exclusive(path, record):
    serialized = json.dumps(_json_safe(record), indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as output:
        output.write(serialized)


def main():
    args = make_parser().parse_args()
    validate_args(args)
    if args.output.exists():
        raise FileExistsError('Never overwrite previous measurements')
    stamp = time.perf_counter()
    inputs = [load_inputs(args.trace, args.stage, step, args.batch) for step in (2, 3, 4)]
    identity = validate_sequence_provenance([p for _, p in inputs], [2, 3, 4],
        args.stage, args.batch, (args.trace/'manifest.json').read_bytes())
    record = dict(role='matched_GPU_restart_mu_saved_inputs_NOT_closed_loop_dfba_or_PPO',
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        input_load_seconds=time.perf_counter() - stamp, sequence_identity=identity,
        input_provenance=[p for _, p in inputs], current_reference_vectors_loaded=False,
        current_CPU_solutions_passed_to_GPU=False, CPU_optimizer_fallback_allowed=False,
        mu_order_randomized=False, repeat_processes_and_counterbalance_order_for_performance_claims=True,
        original_certificate_limits=dict(primal_residual=1e-5, dual_violation=1e-7,
                                         relative_kkt_gap=1e-7),
        additional_direct_dual_relative_gap_limit=1e-7,
        source_setup_is_shared_and_reported_separately=True,
        failed_timings_are_diagnostics_not_successful_performance=True)
    # Hash exactly the bytes snapshotted, rather than racing two source reads.
    sources = {name: (ROOT/name).read_bytes() for name in SOURCE_PATHS}
    record['source_snapshots'] = {name: value.decode() for name, value in sources.items()}
    record['source_sha256'] = {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()}
    try:
        import cupy as cp
        from src.gpu_forest_ipm import ForestGpuBatchedIPM
        from src.gpu_ipm_numeric_update import rebind_forest_ipm
        record['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        _run_matched(record, args, inputs, cp=cp, solver_factory=ForestGpuBatchedIPM,
            rebind_workspace=rebind_forest_ipm,
            synchronize=cp.cuda.get_current_stream().synchronize)
    except Exception as error:
        record.update(completed=False, all_requested_arms_qualified=False,
                      error=dict(type=type(error).__name__, message=str(error)))
    _write_exclusive(args.output, record)
    print('Saved '+str(args.output), flush=True)
    return 0 if record.get('completed') and record.get('all_requested_arms_qualified') else 1


if __name__ == '__main__':
    raise SystemExit(main())
