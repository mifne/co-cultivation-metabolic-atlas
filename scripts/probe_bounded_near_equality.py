"""Input-only diagnostic of one near-dependent equality, not an LP solver.

One bounded pivoted QR proposes a relation. Fraction.from_float then records
the EXACT defect of the stored binary64 matrix/RHS and encloses that defect
over the current finite box. A tiny enclosure can justify considering an
explicit working relaxation; it does NOT prove an exactly redundant row,
preserve the exact feasible set, or bound policy error/objective regret.

No reference primal/dual vectors are loaded. No GPU or LP optimizer is used.
The original problem and runtime solver are never changed by this script.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.linalg import qr, solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import select_entries
from src.gpu_batched_ipm import constraint_form
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_trace import problem_hash
from src.lp_zero_face import ZeroFaceReduction


EXPECTED_INPUT = '503cd51e57fa4c6a6ba2213c8929bfe0bbfb415f3975489ac6ec9c9e939b608d'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fraction(value):
    return dict(numerator=value.numerator, denominator=value.denominator,
                binary64_approximation=float(value))


def exact_row_defect(matrix, rhs, row, support, weights):
    """Return exact stored-data a[row]-sum(w*a[k]) and b[row]-sum(w*b[k])."""
    defect = {}
    for index, weight in [(row, Fraction(1)), *zip(support, (-w for w in weights))]:
        begin, end = matrix.indptr[index:index+2]
        for column, value in zip(matrix.indices[begin:end], matrix.data[begin:end]):
            defect[int(column)] = defect.get(int(column), Fraction(0)) + weight*Fraction.from_float(float(value))
    rhs_defect = Fraction.from_float(float(rhs[row])) - sum(
        (w*Fraction.from_float(float(rhs[k])) for k, w in zip(support, weights)), Fraction(0))
    return {j: value for j, value in defect.items() if value}, rhs_defect


def finite_box_enclosure(defect, rhs_defect, lower, upper):
    """Exact interval of r*x-rho over finite residual-support endpoints.

    Infinite endpoints fail closed. Other variables may have infinite bounds
    because they have identically zero coefficients in the exact defect.
    """
    low = high = -rhs_defect
    for column, coefficient in defect.items():
        lo, hi = float(lower[column]), float(upper[column])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError('Finite ordered bounds required on every defect-support variable')
        endpoints = (coefficient*Fraction.from_float(lo), coefficient*Fraction.from_float(hi))
        low += min(endpoints)
        high += max(endpoints)
    return low, high


def run_probe(trace, *, target_row=1871, primal_tolerance=1e-5):
    """Run precisely one QR on the SHA-bound exchange/env0/step1 LP input."""
    if type(target_row) is not int or target_row < 0:
        raise ValueError('Nonnegative integer target row required')
    if not math.isfinite(primal_tolerance) or primal_tolerance <= 0.:
        raise ValueError('Positive finite explicit primal tolerance required')
    names = ('scripts/probe_bounded_near_equality.py',
             'scripts/analyze_coverage_temporal_routing.py',
             'scripts/probe_downstream_gpu_coverage.py',
             'src/lp_equality_reduction.py', 'src/lp_zero_face.py',
             'src/gpu_batched_ipm.py', 'src/lp_trace.py')
    hashes = {name: _sha(ROOT/name) for name in names}
    trace = Path(trace).resolve()
    manifest_path = trace/'manifest.json'
    manifest_hash = _sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entry = select_entries(manifest, 'exchange', 0, [1])[0]
    original = _load_problem_without_reference(trace, entry)
    if problem_hash(original) != EXPECTED_INPUT:
        raise ValueError('This bounded diagnostic is restricted to the agreed input SHA')
    forest = HomogeneousEqualityReduction.from_problem(original)
    forest_lp = forest.reduce(original).problem
    face = ZeroFaceReduction(forest_lp)
    equality, rhs, _, _, fixed, _, _ = constraint_form(face.reduced)
    if np.any(fixed) or target_row >= face.reduced[-1]:
        raise ValueError('Expected an existing zero-face equality without extra fixed-bound rows')
    nonzero = np.flatnonzero(np.diff(equality.indptr))
    rows, columns = len(nonzero), equality.shape[1]
    estimated_bytes = 8*(4*rows*columns + 3*min(rows, columns)**2 + 3*columns)
    if max(rows, columns) > 6000 or estimated_bytes > 1024**3:
        raise ValueError('Bounded dense-QR memory/dimension guard exceeded')
    normalized = equality[nonzero].copy()
    divisor = np.asarray(abs(normalized).max(axis=1).toarray()).ravel()
    normalized.data /= np.repeat(divisor, np.diff(normalized.indptr))
    if not np.isfinite(normalized.data).all() or np.any(normalized.data == 0.):
        raise ValueError('Normalization loses a nonzero coefficient or is nonfinite')
    print(json.dumps(dict(status='one_CPU_QR_starting', shape=list(equality.shape),
                          cpu_lp_calls=0, gpu_calls=0)), flush=True)
    started = time.perf_counter()
    r, pivots = qr(normalized.T.toarray(order='F'), mode='r', pivoting=True,
                   overwrite_a=True, check_finite=False)
    qr_seconds = time.perf_counter()-started
    r = r[:min(rows, columns)]
    diagonal = np.abs(np.diag(r))
    if not np.isfinite(r).all() or diagonal[0] <= 0.:
        raise ValueError('Invalid QR result')
    relative = diagonal/diagonal[0]
    rank = int(np.count_nonzero(relative > 1e-12))
    if np.any(relative[:rank] <= 1e-12):
        raise ValueError('Ambiguous non-prefix QR rank')
    position = np.flatnonzero(nonzero[pivots] == target_row)
    if position.size != 1 or position[0] < rank:
        raise ValueError('Agreed target was not proposed as a dependent row')
    position = int(position[0])
    numeric = solve_triangular(r[:rank, :rank], r[:rank, position],
                               lower=False, check_finite=False)
    basis = pivots[:rank]
    candidate = pivots[position]
    proposal = numeric*(divisor[candidate]/divisor[basis])
    if not np.isfinite(proposal).all():
        raise ValueError('Nonfinite QR proposal')
    rational = [Fraction.from_float(float(v)).limit_denominator(1) for v in proposal]
    selected = [i for i, value in enumerate(rational) if value]
    if not 1 <= len(selected) <= 256:
        raise ValueError('Unexpected or oversized integer support proposal')
    support = [int(nonzero[basis[i]]) for i in selected]
    weights = [rational[i] for i in selected]
    defect, rhs_defect = exact_row_defect(equality, rhs, target_row, support, weights)
    if (defect != {4057: Fraction(-7, 2**56)} or rhs_defect != 0
            or len(support) != 11 or any(w != -1 for w in weights)):
        raise ValueError('Reconstructed exact defect differs from the previously observed relation')
    low, high = finite_box_enclosure(defect, rhs_defect, face.reduced[2], face.reduced[3])
    bound = max(abs(low), abs(high))
    l1 = sum(map(abs, weights), Fraction(0))
    tolerance = Fraction.from_float(float(primal_tolerance))
    equal_support_budget = max(Fraction(0), tolerance-bound)/l1
    source_rows = forest.kept_rows[face.rows[:face.reduced[-1]]]
    defect_records = []
    for column, coefficient in sorted(defect.items()):
        forest_column = int(face.columns[column])
        transform_column = forest.transform[:, forest_column].tocoo()
        defect_records.append(dict(zero_face_column=column, forest_column=forest_column,
            coefficient=_fraction(coefficient),
            lower_bound=_fraction(Fraction.from_float(float(face.reduced[2][column]))),
            upper_bound=_fraction(Fraction.from_float(float(face.reduced[3][column]))),
            original_column_map=[dict(original_column=int(row), forest_weight=float(weight))
                                 for row, weight in zip(transform_column.row, transform_column.data)]))
    if (_sha(manifest_path) != manifest_hash or problem_hash(original) != EXPECTED_INPUT
            or _sha(trace/entry['filename']) != entry['sha256']
            or hashes != {name: _sha(ROOT/name) for name in names}):
        raise ValueError('Input or diagnostic dependencies changed during inspection')
    return dict(status='completed', role='input_only_finite_bound_working_relaxation_diagnostic',
        trace=str(trace), input_entry={key: entry[key] for key in
            ('filename', 'sha256', 'problem_sha256', 'stage', 'step', 'environment_id')},
        input_manifest_sha256=manifest_hash, sources=hashes,
        model_fingerprints=manifest.get('model_fingerprints'),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        source_problem_unchanged=True, reference_vectors_loaded=False,
        cpu_lp_calls=0, gpu_calls=0, qr_calls=1, rows_deleted=0,
        preprocessing=dict(original_shape=list(original[0].shape), forest_shape=list(forest_lp[0].shape),
            zero_face_shape=list(face.reduced[0].shape), equality_shape=list(equality.shape),
            forest_problem_sha256=problem_hash(forest_lp), zero_face_problem_sha256=face.reduced_hash),
        proposal=dict(method='row-normalized pivoted QR; original-unit weights rounded to integers, then exact defect verification',
            qr_seconds=qr_seconds, threshold=1e-12, numerical_rank=rank, target_pivot_position=position,
            estimated_dense_working_bytes=estimated_bytes),
        exact_stored_data_relation=dict(target_zero_face_row=target_row,
            target_original_row=int(source_rows[target_row]),
            support=[dict(zero_face_row=row, original_row=int(source_rows[row]), coefficient=_fraction(weight))
                     for row, weight in zip(support, weights)],
            support_l1=_fraction(l1), matrix_defect=defect_records, rhs_defect=_fraction(rhs_defect),
            convention='a_target = sum(w_i*a_support_i) + r; b_target = sum(w_i*b_support_i) + rho',
            exactly_redundant=False,
            exact_feasible_set_implication='With every support equality exact, the retained target forces zero_face_x[4057]=0; deleting it changes the exact feasible set'),
        finite_bound_certificate=dict(defect_interval_lower=_fraction(low), defect_interval_upper=_fraction(high),
            defect_absolute_upper_bound=_fraction(bound), explicit_primal_tolerance=primal_tolerance,
            defect_below_explicit_primal_tolerance=bool(bound < tolerance),
            target_residual_identity='a_target*x-b_target = sum(w_i*(a_support_i*x-b_support_i)) + r*x-rho',
            worst_target_residual_if_each_support_at_global_tolerance=_fraction(l1*tolerance+bound),
            sufficient_uniform_support_residual_budget=_fraction(equal_support_budget),
            candidate_requires_dynamic_original_row_recheck=True,
            support_error_note='A tiny defect bound alone does not cover accumulated support residuals or floating-point evaluation error; full original-unit recomputation is required'),
        dual_bound_implications=dict(
            convention='min c*x, A_ineq*x<=b_ineq, y_ineq<=0, q=c-A.T*y',
            direct_dual_bound='D=b.T*y + sum(q_j*lower_j when q_j>=0 else q_j*upper_j), with finite selected bounds/sign validity',
            signed_gap_identity='c.T*x-D = y.T*(A*x-b) + q.T*(x-selected_bound)',
            historical_gap_limitation='Current historical bound/inequality complementarity metric omits equality y.T*(A*x-b); it is not a rigorous regret bound for approximate primal feasibility',
            candidate_dual_lift='When solving the relaxed working LP, inject zero at the omitted equality; do not describe approximate support compression as an exact dual identity',
            optional_compression_identity='If compressing an existing y: y_support_new=y_support+w*y_target; y_target_new=0, then q_new=q+r*y_target and b.T*y_new=b.T*y-rho*y_target',
            optional_compression_bound='For the current box, abs(D_new-D_old)<=abs(y_target)*defect_absolute_upper_bound; a huge multiplier can amplify a tiny row defect',
            actual_dual_multipliers_loaded=False, actual_objective_or_policy_error_estimated=False,
            exact_lp_optimum_preservation_claimed=False),
        proposed_next_experiment=dict(
            method='Explicit bounded near-equality working relaxation only; preserve the immutable original LP and all original certificate gates',
            required_checks=['Revalidate exact matrix/RHS relation and current finite-bound enclosure on every distinct input identity',
                'Use retained-support residuals plus original full-row recomputation; no unconditional acceptance from the static box bound',
                'Add signed equality-dual residual and direct dual objective diagnostics before claiming original-LP optimality',
                'Compare convergence and accepted-solution runtime on all three stages; failed stop time is not speedup',
                'Then check full physical-step/closed-loop trajectory and action-ranking accuracy; this algebraic screen is not PPO qualification'],
            no_runtime_change_in_this_probe=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT/'results/pf_coverage_holdout4x120_20260905')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--primal-tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('This diagnostic never overwrites an existing result')
    report = run_probe(args.trace, primal_tolerance=args.primal_tolerance)
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(payload)
    print(json.dumps(dict(status='completed', output=str(args.output),
        support_rows=[item['zero_face_row'] for item in report['exact_stored_data_relation']['support']],
        defect_bound=report['finite_bound_certificate']['defect_absolute_upper_bound'],
        cpu_lp_calls=0, gpu_calls=0, qr_calls=1, rows_deleted=0)), flush=True)


if __name__ == '__main__':
    main()
