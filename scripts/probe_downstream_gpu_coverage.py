"""Development-only candidate-hull existence oracle for downstream LP stages.

Original held-out LP inputs are replayed without reading stored CPU x/y. GPU
compact maps supply all proposed x/y. A CPU LP then asks whether their convex
hull (optionally affine hull) contains an original-LP feasible/optimal solution.
This is NOT an online GPU corrector, a closed-loop run, training, or a speed
comparison. Feasibility alone is not optimality: one complete original-LP
certificate for a single paired x/y must pass.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from src.compact_training_data import checked_npz, sha256_file
from src.cpu_repeated_lp import _certificate
from src.gpu_certified_basis import CommunityCoordinates
from src.gpu_compact_basis import CompactBank
from src.gpu_heterogeneous_compact import HeterogeneousCompactBank

DEFAULT_BANK_SHA256 = '6c192b1c5fe26980942f363ecc6ccbe133798445b76149e0a30842e35e42dde6'
DEFAULT_TRACE_SHA256 = 'af7b6b92ca27487590c67158294c390072bd89e1c84d0a0aebccdcec64e532bc'
LIMITS = dict(primal_residual=1e-5, dual_violation=1e-7, relative_kkt_gap=1e-7)
WEIGHT_TOLERANCE = 1e-8


def select_entries(manifest, stage, environment_id, steps):
    """Validate labels before selecting input-only diagnostic rows."""
    if (manifest.get('status') != 'completed'
            or manifest.get('role') != 'development_diagnostic_not_training'):
        raise ValueError('Completed development diagnostic trace required')
    seeds = manifest.get('seeds', [])
    completed = manifest.get('completed_steps', [])
    if (len(set(seeds)) != len(seeds) or len(completed) != len(seeds)
            or not seeds or not 0 <= environment_id < len(seeds)
            or not steps or len(set(steps)) != len(steps)
            or any(not isinstance(step, int) or isinstance(step, bool)
                   or not 1 <= step <= completed[environment_id] for step in steps)
            or stage not in ('maxmin', 'aggregate', 'exchange')):
        raise ValueError('Invalid trace selection')
    entries = {}
    for entry in manifest.get('entries', []):
        if entry.get('stage') != stage or entry.get('environment_id') != environment_id:
            continue
        step = entry.get('step')
        if step in entries:
            raise ValueError('Duplicate LP identity')
        entries[step] = entry
    chosen = []
    for step in steps:
        entry = entries.get(step)
        expected = f'lp_{step:03d}_{("maxmin", "aggregate", "exchange").index(stage)}_{environment_id:03d}.npz'
        if entry is None or entry.get('filename') != expected:
            raise ValueError('Missing or mislabeled original LP input')
        chosen.append(entry)
    return chosen


def original_candidates(raw_x, raw_y, col_scale, row_scale):
    """Undo z=col_scale*x and row scaling, including the dual transform."""
    raw_x, raw_y = np.asarray(raw_x), np.asarray(raw_y)
    cs, rs = np.asarray(col_scale), np.asarray(row_scale)
    if (raw_x.ndim != 2 or raw_y.ndim != 2 or len(raw_x) != len(raw_y)
            or cs.shape != (raw_x.shape[1],) or rs.shape != (raw_y.shape[1],)
            or not np.isfinite(cs).all() or not np.isfinite(rs).all()
            or np.any(cs <= 0) or np.any(rs <= 0)):
        raise ValueError('Invalid candidate coordinate arrays')
    return raw_x / cs, raw_y * rs


def paired_certificate(problem, x, y):
    """Never mix primal/dual/gap statistics from different solution pairs."""
    a, _, _, _, c, _ = problem
    solution = SimpleNamespace(col_value=x, row_dual=y, col_dual=c-a.T@y,
                               value_valid=True, dual_valid=True)
    return _certificate(*problem, solution)


def coefficient_range(values):
    """Magnitude spread is not a matrix condition-number estimate."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    absolute = np.abs(values[finite])
    nonzero = absolute[absolute > 0]
    maximum = float(absolute.max(initial=0.))
    minimum = float(nonzero.min()) if len(nonzero) else None
    with np.errstate(over='ignore'):
        spread = float(np.float64(maximum)/minimum) if minimum else None
    return dict(elements=int(values.size), nonfinite_elements=int((~finite).sum()),
                zero_elements=int((values == 0).sum()), maximum_absolute=maximum,
                minimum_nonzero_absolute=minimum,
                magnitude_dynamic_range_not_condition_number=spread)


def scale_reduced_problem(eq, eq_rhs, ub, ub_rhs, cost):
    """Positive row/objective equilibration; no row or variable is dropped.

    Scaling changes numerical solver tolerances in original units. Therefore
    a scaled solver success is still only a proposal and all unscaled original
    constraints/certificates plus the alpha domain must be checked afterward.
    """
    for array in (eq, eq_rhs, ub, ub_rhs, cost):
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite reduced coefficients cannot be rescaled')
    equality_scale = np.maximum(1., np.maximum(np.max(np.abs(eq), axis=1), np.abs(eq_rhs)))
    inequality_scale = np.maximum(1., np.maximum(np.max(np.abs(ub), axis=1), np.abs(ub_rhs)))
    objective_scale = max(1., float(np.max(np.abs(cost), initial=0.)))
    scaled = (eq/equality_scale[:, None], eq_rhs/equality_scale,
              ub/inequality_scale[:, None], ub_rhs/inequality_scale, cost/objective_scale)
    metadata = dict(method='Positive max-absolute row/RHS and objective equilibration, minimum divisor 1',
                    equality_divisors=coefficient_range(equality_scale),
                    inequality_divisors=coefficient_range(inequality_scale),
                    objective_divisor=objective_scale,
                    rows_removed=0, original_certificate_thresholds_changed=False,
                    caution='Scaled solver tolerance alone does not establish original-space feasibility')
    return scaled, metadata


def hull_oracle(problem, candidate_x, candidate_y, *, candidate_ids=None,
                family_valid=None, affine=False, scale_reduced=False):
    """Solve only for candidate weights; all original constraints remain.

    A solver's success flag alone does not establish feasibility. Reconstruct
    x and test original unscaled LP constraints at the unchanged threshold.
    Test both weight-matched y and each eligible dictionary y. Failing these
    finitely many dual tests does NOT prove no dual combination exists.
    """
    x, y = np.asarray(candidate_x, dtype=float), np.asarray(candidate_y, dtype=float)
    a, rhs, lower, upper, c, neq = problem
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != len(c)
            or y.shape != (len(x), len(rhs))):
        raise ValueError('Candidate shapes do not match the original LP')
    ids = np.arange(len(x)) if candidate_ids is None else np.asarray(candidate_ids)
    family = np.ones(len(x), dtype=bool) if family_valid is None else np.asarray(family_valid)
    if (ids.shape != (len(x),) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
            or family.shape != (len(x),) or family.dtype != np.bool_):
        raise ValueError('Invalid candidate identity or family flags')
    eligible = family & np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    report = dict(mode='affine' if affine else 'convex', total_candidates=len(x),
                  family_valid_candidates=int(family.sum()),
                  eligible_candidates=int(eligible.sum()),
                  eligible_candidate_ids=ids[eligible].tolist(),
                  solver_success=False, original_primal_feasible=False,
                  full_original_certificate_passed=False,
                  hull_membership_valid=False, hull_certified_solution_found=False,
                  alpha_sum_absolute_tolerance=WEIGHT_TOLERANCE,
                  alpha_negative_tolerance=WEIGHT_TOLERANCE,
                  failure_interpretation='Numerical solver failure/infeasibility is not a mathematical impossibility proof; conditioning and rounding can matter.',
                  dual_search_scope='Each eligible dictionary y plus alpha-weighted y; not exhaustive dual-hull optimization')
    if not eligible.any():
        report['status'] = 'no_finite_family_valid_candidates'
        return report
    x, y, ids = x[eligible], y[eligible], ids[eligible]
    # Columns of X correspond to the candidate coefficients alpha.
    before = time.perf_counter()
    matrix = x.T
    activity = a @ matrix
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    eq = np.vstack((activity[:neq], np.ones((1, len(x)))))
    eq_rhs = np.r_[rhs[:neq], 1.]
    ub = np.vstack((activity[neq:], -matrix[finite_lower], matrix[finite_upper]))
    ub_rhs = np.r_[rhs[neq:], -lower[finite_lower], upper[finite_upper]]
    cost = matrix.T@c
    report['reduced_problem'] = dict(variables=len(x), equalities=len(eq), inequalities=len(ub),
                                    original_variables=len(c), original_rows=len(rhs))
    report['coefficient_ranges'] = dict(candidate_x=coefficient_range(x), candidate_y=coefficient_range(y),
        equality_matrix=coefficient_range(eq), equality_rhs=coefficient_range(eq_rhs),
        inequality_matrix=coefficient_range(ub), inequality_rhs=coefficient_range(ub_rhs),
        objective=coefficient_range(cost))
    if not all(np.isfinite(array).all() for array in (eq, eq_rhs, ub, ub_rhs, cost)):
        report.update(status='nonfinite_reduced_problem', reduced_assembly_seconds=time.perf_counter()-before)
        return report
    report['positive_reduced_scaling_enabled'] = bool(scale_reduced)
    if scale_reduced:
        (eq, eq_rhs, ub, ub_rhs, cost), report['positive_reduced_scaling'] = scale_reduced_problem(
            eq, eq_rhs, ub, ub_rhs, cost)
        report['scaled_coefficient_ranges'] = dict(equality_matrix=coefficient_range(eq),
            equality_rhs=coefficient_range(eq_rhs), inequality_matrix=coefficient_range(ub),
            inequality_rhs=coefficient_range(ub_rhs), objective=coefficient_range(cost))
    report['reduced_assembly_seconds'] = time.perf_counter()-before
    before = time.perf_counter()
    result = linprog(cost, A_ub=ub, b_ub=ub_rhs, A_eq=eq, b_eq=eq_rhs,
                     bounds=(None, None) if affine else (0., None), method='highs-ds',
                     options=dict(primal_feasibility_tolerance=1e-9,
                                  dual_feasibility_tolerance=1e-9))
    report.update(cpu_oracle_seconds=time.perf_counter()-before,
                  solver_success=bool(result.success), solver_status=int(result.status),
                  solver_message=str(result.message), status='cpu_oracle_finished')
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        return report
    alpha = np.asarray(result.x)
    reconstructed = matrix @ alpha
    membership_valid = (abs(float(alpha.sum())-1.) <= WEIGHT_TOLERANCE
                        and (affine or float(alpha.min()) >= -WEIGHT_TOLERANCE))
    report.update(alpha=alpha.tolist(), alpha_l1=float(np.abs(alpha).sum()),
                  alpha_max_abs=float(np.abs(alpha).max()), alpha_sum=float(alpha.sum()),
                  alpha_min=float(alpha.min()), hull_membership_valid=bool(membership_valid),
                  original_objective=float(c@reconstructed))
    before = time.perf_counter()
    tests = [dict(dual_source='alpha_weighted', certificate=paired_certificate(problem, reconstructed, alpha@y))]
    tests.extend(dict(dual_source='dictionary', candidate_id=int(index),
                      certificate=paired_certificate(problem, reconstructed, vector))
                 for index, vector in zip(ids, y))
    def score(row):
        cert = row['certificate']
        return max(cert[key]/limit for key, limit in LIMITS.items())
    best = min(tests, key=score)
    report.update(original_primal_feasible=bool(best['certificate']['primal_residual'] <= LIMITS['primal_residual']),
                  full_original_certificate_passed=any(t['certificate']['certificate_passed'] for t in tests),
                  best_complete_pair=best, paired_certificates=tests,
                  original_certificate_seconds=time.perf_counter()-before)
    report['hull_certified_solution_found'] = bool(membership_valid and report['full_original_certificate_passed'])
    return report


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--trace-sha256', default=DEFAULT_TRACE_SHA256)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--bank-sha256', default=DEFAULT_BANK_SHA256)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--environment-id', type=int, default=0)
    parser.add_argument('--candidate-chunk', type=int, default=8)
    parser.add_argument('--affine', action='store_true')
    parser.add_argument('--scale-reduced', action='store_true')
    parser.add_argument('--existing-oracle', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.candidate_chunk < 1:
        raise ValueError('Positive candidate chunk required')
    for path, expected in ((args.trace/'manifest.json', args.trace_sha256),
                           (args.bank/'manifest.json', args.bank_sha256)):
        if sha256_file(path) != expected:
            raise ValueError(f'Pinned manifest SHA mismatch: {path}')
    trace = json.loads((args.trace/'manifest.json').read_text())
    manifest = json.loads((args.bank/'manifest.json').read_text())
    if (manifest.get('status') != 'completed'
            or manifest.get('model_fingerprints') != trace.get('model_fingerprints')
            or set(manifest.get('train_seeds', [])) & set(trace.get('seeds', []))):
        raise ValueError('Bank status, GEM fingerprints or train/diagnostic separation mismatch')
    selected = {stage: select_entries(trace, stage, args.environment_id, args.steps)
                for stage in ('aggregate', 'exchange')}
    maxmin_entry = select_entries(trace, 'maxmin', args.environment_id, [args.steps[0]])[0]
    maxmin = _load_problem_without_reference(args.trace, maxmin_entry)
    report = dict(status='running', scope=__doc__,
                  trace_sha256=args.trace_sha256, bank_sha256=args.bank_sha256,
                  input_selection=dict(environment_id=args.environment_id,
                                       seed=trace['seeds'][args.environment_id], steps=args.steps),
                  reference_vectors_read=False, training_performed=False,
                  online_cpu_lp_calls=0, cpu_diagnostic_oracle_calls=0,
                  speed_comparison_valid=False, original_certificate_limits=LIMITS,
                  positive_reduced_scaling_enabled=args.scale_reduced,
                  nonfinite_encoding='Nonfinite certificate metrics are null; acceptance remains explicit.',
                  source_hashes={name: sha256_file(ROOT/name) for name in (
                      'scripts/probe_downstream_gpu_coverage.py', 'scripts/analyze_coverage_temporal_routing.py',
                      'src/gpu_compact_basis.py', 'src/gpu_heterogeneous_compact.py',
                      'src/gpu_certified_basis.py', 'src/cpu_repeated_lp.py', 'src/lp_bounds.py')},
                  stages=[])
    if args.existing_oracle:
        previous = json.loads(args.existing_oracle.read_text())
        older_stages = {s['stage']: s for s in previous['offline_bank_manifest']['stages']}
        equal = {s['stage']: older_stages.get(s['stage']) == s for s in manifest['stages']
                 if s['stage'] in selected}
        report['existing_oracle'] = dict(path=str(args.existing_oracle),
            sha256=sha256_file(args.existing_oracle), identical_stage_manifests=equal,
            note='Prior all-candidate acceptance diagnostic is not repeated as a performance experiment. Different input seeds do not establish equal coverage.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False))
    save()
    try:
        with ExitStack() as cleanup:
            _run_gpu_diagnostic(args, trace, manifest, selected, maxmin, report, save, cleanup)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        save()
        raise
    report['status'] = 'completed'
    save()


def _run_gpu_diagnostic(args, trace, manifest, selected, maxmin, report, save, cleanup):
    import cupy as cp
    from scripts.benchmark_basis_bank_rollout import environment
    from src.fba_surrogate import model_fingerprint
    sample, layout = environment(trace['seeds'][args.environment_id])
    if hasattr(sample, 'close'):
        cleanup.callback(sample.close)
    if {name: model_fingerprint(model) for name, model in sample.simulator.models.items()} != trace['model_fingerprints']:
        raise ValueError('Runtime GEM fingerprints mismatch')
    coordinates = CommunityCoordinates(dict(growth_terms=layout._growth_terms,
        exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for model in sample.simulator.models.values() for r in model.reactions],
        reaction_species=[name for name, model in sample.simulator.models.items() for r in model.reactions]),
        maxmin[0], maxmin[-1])
    for stage_name, entries in selected.items():
        stage = next(s for s in manifest['stages'] if s['stage'] == stage_name)
        directory = args.bank/stage_name
        root = checked_npz(directory, 'root.npz', stage['root_sha256'])
        router = checked_npz(directory, 'router.npz', stage['router_sha256'])
        dictionaries = [checked_npz(directory, e['filename'], e['sha256']) for e in stage['entries']]
        a = csr_matrix((root['a_data'], root['a_indices'], root['a_indptr']), shape=tuple(root['a_shape']))
        bank = CompactBank(dict(a=a, neq=int(root['neq'])), root['variable_rows'], dictionaries,
                           router['centers'], router['indices'], router['scale'], capture=True,
                           selected_features=True, candidate_ranking='count')
        if bank.math is not None:
            cleanup.callback(bank.math.close)
        cleanup.callback(bank.clear_graph_cache)
        # Full diagnostics expose input_family_valid and raw x/y. No timing
        # claim is made; online certificate-only fastpath is not changed.
        engine = HeterogeneousCompactBank(bank, certificate_only=False)
        cleanup.callback(engine.close)
        stage_report = dict(stage=stage_name, candidates=len(dictionaries), rows=[])
        report['stages'].append(stage_report)
        try:
            for entry in entries:
                problem = _load_problem_without_reference(args.trace, entry)
                before = time.perf_counter()
                normalized = coordinates.normalize(*problem)
                inputs = bank.prepare_host([normalized])
                cp.cuda.get_current_stream().synchronize()
                prepare_seconds = time.perf_counter()-before
                vectors_x, vectors_y, family, accepted = [], [], [], []
                before = time.perf_counter()
                for start in range(0, len(dictionaries), args.candidate_chunk):
                    order = cp.arange(start, min(start+args.candidate_chunk, len(dictionaries)), dtype=cp.int32)[None]
                    out, _, _, _ = engine._candidates(inputs, order)
                    vectors_x.append(out['raw_values'].get())
                    vectors_y.append(out['raw_y'].get())
                    family.append(out['input_family_valid'].get())
                    accepted.append(out['accepted'].get())
                elapsed = time.perf_counter()-before
                original_x, original_y = original_candidates(np.concatenate(vectors_x), np.concatenate(vectors_y),
                                                             normalized.col_scale, normalized.row_scale)
                family = np.concatenate(family)
                accepted = np.concatenate(accepted)
                row = dict(step=entry['step'], environment_id=entry['environment_id'],
                           problem_sha256=entry['problem_sha256'], input_file_sha256=entry['sha256'],
                           input_preparation_seconds=prepare_seconds,
                           gpu_raw_candidate_diagnostic_including_transfer_seconds=elapsed,
                           family_valid_candidate_ids=np.flatnonzero(family).tolist(),
                           original_gpu_certificate_candidate_ids=np.flatnonzero(accepted).tolist(),
                           candidate_hulls=[])
                stage_report['rows'].append(row)
                for affine in ([False, True] if args.affine else [False]):
                    outcome = hull_oracle(problem, original_x, original_y, family_valid=family,
                                          affine=affine, scale_reduced=args.scale_reduced)
                    report['cpu_diagnostic_oracle_calls'] += int('solver_status' in outcome)
                    row['candidate_hulls'].append(outcome)
                save()
                print(json.dumps(dict(stage=stage_name, step=entry['step'],
                    family_candidates=int(family.sum()), directly_certified=int(accepted.sum()),
                    hulls=[dict(mode=r['mode'], solver_success=r['solver_success'],
                                primal_feasible=r['original_primal_feasible'],
                                hull_membership_valid=r['hull_membership_valid'],
                                certified=r['hull_certified_solution_found']) for r in row['candidate_hulls']])), flush=True)
        finally:
            engine.close()
            bank.clear_graph_cache()
            if bank.math is not None:
                bank.math.close()


if __name__ == '__main__':
    main()
