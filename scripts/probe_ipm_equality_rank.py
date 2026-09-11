"""Read-only CPU rank diagnostic of the actual forest/zero-face IPM equalities.

Only SHA-verified LP inputs are loaded, never CPU reference x/y. Pivoted QR is
a finite-precision diagnostic, NOT an exact dependency proof, a row-deletion
instruction, an LP solve, a fallback, or a CPU/GPU performance comparison.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.linalg import qr, solve_triangular
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import structural_rank

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.audit_lp_equalities import _sha, coefficient_statistics, equality_hash
from scripts.probe_downstream_gpu_coverage import select_entries
from src.gpu_batched_ipm import constraint_form
from src.gpu_pdhg_corrector import _validated_problem
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_trace import problem_hash
from src.lp_zero_face import ZeroFaceReduction


def _finite_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def diagnose_equality_rank(matrix, rhs, *, thresholds=(1e-10, 1e-12, 1e-14),
                           nominated_threshold=1e-12, max_dimension=6000,
                           max_memory_mb=1024., max_candidates=32):
    """Normalize rows, compute structural rank and bounded pivoted QR of E.T.

    For nominated rank r, R11*w=R[:r,j] reconstructs a candidate dependent
    normalized row from the pivot basis. We report matrix and RHS backward
    residuals, not a certificate of exact dependence or LP infeasibility.
    All reported row indices refer to input E; the caller supplies provenance.
    """
    started = time.perf_counter()
    if (type(max_dimension) is not int or not 1 <= max_dimension <= 6000
            or not math.isfinite(max_memory_mb) or max_memory_mb <= 0.
            or type(max_candidates) is not int or not 1 <= max_candidates <= 128):
        raise ValueError('Bounded dimension <=6000, positive memory budget, and 1..128 candidates required')
    thresholds = tuple(thresholds)
    if (not thresholds or len(set(thresholds)) != len(thresholds)
            or any(not math.isfinite(t) or not 0. < t < 1. for t in thresholds)
            or nominated_threshold not in thresholds):
        raise ValueError('Unique thresholds in (0,1), including the nominated threshold, required')
    if np.iscomplexobj(matrix.data if hasattr(matrix, 'data') else matrix) or np.iscomplexobj(rhs):
        raise ValueError('Real-valued equalities and RHS required')
    equality = csr_matrix(matrix, dtype=np.float64, copy=True)
    equality.sum_duplicates()
    equality.eliminate_zeros()
    equality.sort_indices()
    b = np.asarray(rhs, dtype=np.float64).copy()
    rows, columns = equality.shape
    if (rows < 1 or columns < 1 or b.shape != (rows,)
            or not np.isfinite(equality.data).all() or not np.isfinite(b).all()):
        raise ValueError('Nonempty finite equality matrix and matching one-dimensional RHS required')
    if max(rows, columns) > max_dimension:
        raise ValueError('Equality dimensions exceed the configured dense-QR guard')
    small = min(rows, columns)
    # Conservative allowance: dense E.T, LAPACK copies/output/workspace,
    # triangular blocks, plus sparse copies and bounded candidate residuals.
    estimated_bytes = (8*(4*rows*columns + 3*small*small + 3*columns*max_candidates)
                       + 4*(equality.data.nbytes + equality.indices.nbytes + equality.indptr.nbytes))
    if estimated_bytes > max_memory_mb*1024**2:
        raise ValueError(f'Estimated dense-QR working memory {estimated_bytes} bytes exceeds guard')
    row_max = np.asarray(abs(equality).max(axis=1).toarray()).ravel()
    divisor = np.where(row_max > 0., row_max, 1.)
    normalized = equality.copy()
    normalized.data /= np.repeat(divisor, np.diff(normalized.indptr))
    with np.errstate(over='ignore', under='ignore'):
        normalized_b = b/divisor
    if (not np.isfinite(normalized.data).all() or not np.isfinite(normalized_b).all()
            or np.any((equality.data != 0.) & (normalized.data == 0.))
            or np.any((b != 0.) & (normalized_b == 0.))):
        raise ValueError('Row normalization loses nonzero data or creates a nonfinite value')
    shape_rank = int(structural_rank(equality))
    dense = normalized.T.toarray(order='F')
    qr_started = time.perf_counter()
    # No Q matrix is requested. Pivots select ROWS of the original E because
    # columns of E.T are pivoted. Caller-owned sparse arrays are unchanged.
    r_full, pivots = qr(dense, mode='r', pivoting=True, overwrite_a=True, check_finite=False)
    r = r_full[:small]
    qr_seconds = time.perf_counter()-qr_started
    if not np.isfinite(r).all():
        raise ValueError('Pivoted QR returned nonfinite coefficients')
    diagonal = np.abs(np.diag(r))
    leading = float(diagonal[0])
    relative_diagonal = diagonal/leading if leading > 0. else np.zeros_like(diagonal)
    ranks = {f'{t:.0e}': int(np.count_nonzero(relative_diagonal > t)) for t in thresholds}
    rank = int(np.count_nonzero(relative_diagonal > nominated_threshold))
    if np.any(relative_diagonal[:rank] <= nominated_threshold):
        raise ValueError('Non-prefix QR rank estimate; do not form an ambiguous R11 reconstruction')
    independent = pivots[:rank]
    dependent = pivots[rank:]
    chosen = dependent[:max_candidates]
    if rank:
        coefficients = solve_triangular(r[:rank, :rank], r[:rank, rank:rank+len(chosen)],
                                        lower=False, check_finite=False)
    else:
        coefficients = np.zeros((0, len(chosen)))
    if not np.isfinite(coefficients).all():
        raise ValueError('Candidate row reconstruction is nonfinite')
    basis = normalized[independent]
    reconstruction = np.asarray(basis.T@coefficients)
    absolute_activity = np.asarray(abs(basis).T@np.abs(coefficients))
    target_rows = normalized[chosen].toarray().T
    difference = target_rows-reconstruction
    predicted_b = coefficients.T@normalized_b[independent]
    absolute_b_activity = np.abs(coefficients).T@np.abs(normalized_b[independent])
    candidates = []
    for j, row in enumerate(chosen):
        residual_inf = float(np.max(np.abs(difference[:, j]), initial=0.))
        denominator = float(np.max(np.abs(target_rows[:, j])+absolute_activity[:, j], initial=0.))
        matrix_backward = residual_inf/denominator if denominator else (0. if residual_inf == 0. else math.inf)
        b_error = float(normalized_b[row]-predicted_b[j])
        b_denominator = max(1., abs(float(normalized_b[row]))+float(absolute_b_activity[j]))
        rhs_backward = abs(b_error)/b_denominator
        candidates.append(dict(row=int(row), pivot_position=rank+j,
            row_divisor=float(divisor[row]),
            scaled_matrix_reconstruction_inf=residual_inf,
            scaled_matrix_backward_error_inf=_finite_scalar(matrix_backward),
            original_row_reconstruction_inf=_finite_scalar(residual_inf*divisor[row]),
            coefficient_max_abs=float(np.max(np.abs(coefficients[:, j]), initial=0.)),
            coefficient_l1=float(np.sum(np.abs(coefficients[:, j]))),
            rhs_original=float(b[row]), rhs_scaled=float(normalized_b[row]),
            rhs_scaled_prediction=_finite_scalar(predicted_b[j]), rhs_scaled_error=_finite_scalar(b_error),
            rhs_original_error=_finite_scalar(b_error*divisor[row]),
            rhs_backward_error=_finite_scalar(rhs_backward),
            matrix_consistent_at_nominated_threshold=bool(matrix_backward <= nominated_threshold),
            rhs_consistent_at_nominated_threshold=bool(rhs_backward <= nominated_threshold)))
    smallest = np.argsort(relative_diagonal)[:min(32, len(diagonal))]
    return dict(rows=rows, columns=columns, nonzeros=int(equality.nnz),
        equality_sha256=equality_hash((equality, b, None, None, None, rows)),
        structural_rank=shape_rank, structural_rank_is_only_an_upper_bound=True,
        original_coefficients=coefficient_statistics(equality),
        normalized_coefficients=coefficient_statistics(normalized),
        row_normalization=dict(method='E[i,:]/max(abs(E[i,:])); zero-row divisor=1',
            divisor_min=float(divisor.min()), divisor_max=float(divisor.max()),
            divisors=divisor.tolist(), zero_rows=np.flatnonzero(row_max == 0.).tolist()),
        estimated_dense_working_bytes=estimated_bytes,
        memory_estimate_note='Conservative allocation estimate, not measured peak RSS or runtime prediction',
        configured_memory_limit_mb=float(max_memory_mb), configured_dimension_limit=max_dimension,
        qr_mode='scipy.linalg.qr(E_normalized.T, mode=r, pivoting=True); no Q requested',
        qr_seconds=qr_seconds, diagnostic_seconds=time.perf_counter()-started,
        rank_threshold_definition='count(abs(diag(R))/abs(R[0,0]) > threshold); rank=0 when R[0,0]=0',
        rank_estimates=ranks, nominated_threshold=float(nominated_threshold), nominated_rank=rank,
        leading_r_diagonal=leading,
        smallest_r_diagonals=[dict(pivot_position=int(i), row=int(pivots[i]),
            absolute=float(diagonal[i]), relative=float(relative_diagonal[i])) for i in smallest],
        independent_row_indices=independent.tolist(), candidate_dependent_row_indices=dependent.tolist(),
        reconstructed_candidate_count=len(candidates), omitted_candidate_count=len(dependent)-len(candidates),
        candidate_reconstructions=candidates,
        rhs_backward_error_definition='abs(b_scaled[row]-w.T*b_scaled[basis])/max(1,abs(b_scaled[row])+abs(w).T*abs(b_scaled[basis]))',
        exact_dependency_proven=False, rows_deleted=0, cpu_lp_calls=0, gpu_calls=0,
        interpretation='Numerical rank and reconstruction diagnostics only; near-dependence or RHS disagreement is not an exact proof and cannot authorize deleting rows')


def prepare_ipm_equalities(problem):
    """Use exactly the CPU preprocessing functions underlying the GPU IPM."""
    original = _validated_problem(problem)
    initial_hash = problem_hash(original)
    forest = HomogeneousEqualityReduction.from_problem(original)
    forest_problem = forest.reduce(original).problem
    face = ZeroFaceReduction(forest_problem)
    e, b, g, h, fixed, il, iu = constraint_form(face.reduced)
    kept_original_rows = forest.kept_rows[face.rows[:face.reduced[-1]]]
    row_sources = [dict(kind='original_lp_row_after_column_transforms', original_row=int(row))
                   for row in kept_original_rows]
    row_sources += [dict(kind='fixed_bound_equality', zero_face_reduced_column=int(column),
                         forest_reduced_column=int(face.columns[column]))
                    for column in np.flatnonzero(fixed)]
    if len(row_sources) != e.shape[0] or problem_hash(original) != initial_hash:
        raise ValueError('Preprocessing provenance mismatch or input mutation')
    return e, b, dict(original_lp_sha256=initial_hash,
        forest_lp_sha256=problem_hash(forest_problem), zero_face_lp_sha256=face.reduced_hash,
        original_shape=list(original[0].shape), forest_shape=list(forest_problem[0].shape),
        zero_face_shape=list(face.reduced[0].shape), standard_equality_shape=list(e.shape),
        standard_inequality_shape=list(g.shape),
        explicit_fixed_count=len(face.explicit_fixed_columns), forced_zero_count=len(face.forced_zero_columns),
        removed_duplicate_equalities=len(face.removed_duplicate_rows),
        standard_equality_row_sources=row_sources,
        preprocessing='HomogeneousEqualityReduction -> ZeroFaceReduction(default exact duplicate removal) -> constraint_form; no GPU workspace')


def run_probe(trace, *, stage='exchange', step=1, environment_id=0, **rank_options):
    trace = Path(trace)
    if type(environment_id) is not int or environment_id < 0:
        raise ValueError('Nonnegative integer environment id required')
    source_names = ('scripts/probe_ipm_equality_rank.py', 'scripts/audit_lp_equalities.py',
        'scripts/analyze_coverage_temporal_routing.py', 'scripts/probe_downstream_gpu_coverage.py',
        'src/lp_equality_reduction.py', 'src/lp_zero_face.py', 'src/gpu_batched_ipm.py',
        'src/gpu_pdhg_corrector.py', 'src/lp_trace.py')
    hashes = {name: _sha(ROOT/name) for name in source_names}
    snapshots = {name: (ROOT/name).read_text() for name in source_names}
    manifest_path = trace/'manifest.json'
    manifest_hash = _sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entry = select_entries(manifest, stage, environment_id, [step])[0]
    problem = _load_problem_without_reference(trace, entry)
    e, b, preparation = prepare_ipm_equalities(problem)
    print(json.dumps(dict(status='CPU_QR_starting', stage=stage, step=step,
                          environment_id=environment_id, equality_shape=list(e.shape))), flush=True)
    rank_report = diagnose_equality_rank(e, b, **rank_options)
    if (_sha(manifest_path) != manifest_hash or hashes != {name: _sha(ROOT/name) for name in source_names}
            or _sha(trace/entry['filename']) != entry['sha256']):
        raise ValueError('Input or implementation changed during read-only diagnostic')
    return dict(status='completed', role='development_equality_rank_diagnostic_not_training_or_rollout',
        trace=str(trace.resolve()), input_manifest_sha256=manifest_hash,
        input_entry={key: entry[key] for key in ('filename', 'sha256', 'problem_sha256',
                    'stage', 'step', 'environment_id', 'rows', 'columns', 'equalities')},
        seed=manifest['seeds'][environment_id], model_fingerprints=manifest.get('model_fingerprints'),
        sources=hashes, source_snapshots=snapshots, numpy_version=np.__version__, scipy_version=scipy.__version__,
        current_reference_vectors_loaded=False, cpu_lp_calls=0, gpu_calls=0,
        preprocessing=preparation, equality_rank=rank_report,
        scope='CPU linear algebra diagnostic only; no LP solve, numerical fallback, row deletion, solver change, or CPU/GPU speed comparison',
        primary_references=[
            'https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html',
            'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.structural_rank.html'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT/'results/pf_coverage_holdout4x120_20260905')
    parser.add_argument('--stage', choices=['maxmin', 'aggregate', 'exchange'], default='exchange')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--environment-id', type=int, default=0)
    parser.add_argument('--max-dimension', type=int, default=6000)
    parser.add_argument('--max-memory-mb', type=float, default=1024.)
    parser.add_argument('--max-candidates', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Rank diagnostics never overwrite existing results')
    report = run_probe(args.trace, stage=args.stage, step=args.step, environment_id=args.environment_id,
        max_dimension=args.max_dimension, max_memory_mb=args.max_memory_mb, max_candidates=args.max_candidates)
    contents = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(contents)
    print(json.dumps(dict(status='completed', output=str(args.output),
        structural_rank=report['equality_rank']['structural_rank'],
        numerical_rank_estimates=report['equality_rank']['rank_estimates'],
        exact_dependency_proven=False, rows_deleted=0)), flush=True)


if __name__ == '__main__':
    main()
