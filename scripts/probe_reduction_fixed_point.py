"""Input-only alternating forest/zero-face presolve diagnostic.

No CPU LP, GPU operation, QR, numerical rank decision, or saved x/y is used.
This measures exact-sign structural opportunities, not solution accuracy or
solver speed. The forest implementation itself uses floating-point algebra;
its local null-map residual is therefore reported rather than called a proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import eye

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import select_entries
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_trace import problem_hash
from src.lp_zero_face import ZeroFaceReduction


def max_abs_sparse(matrix):
    return float(np.max(np.abs(matrix.data), initial=0.))


def structure(problem):
    a, rhs, lo, hi, _c, neq = problem
    counts = np.diff(a.indptr)[:neq]
    homogeneous = rhs[:neq] == 0.
    finite_width = np.isfinite(lo) & np.isfinite(hi)
    widths = hi[finite_width] - lo[finite_width]
    return dict(rows=a.shape[0], columns=a.shape[1], nonzeros=a.nnz,
        equalities=neq, homogeneous_two_entry_rows=int(np.sum(homogeneous & (counts == 2))),
        homogeneous_single_entry_rows=int(np.sum(homogeneous & (counts == 1))),
        homogeneous_zero_rows=int(np.sum(homogeneous & (counts == 0))),
        fixed_bounds=int(np.sum(lo == hi)),
        finite_bound_width_min=float(np.min(widths, initial=np.inf)) if len(widths) else None,
        finite_bound_width_max=float(np.max(widths, initial=0.)),
        problem_sha256=problem_hash(problem))


def singletons(problem, original_rows):
    a, rhs, lo, hi, _c, neq = problem
    rows = np.flatnonzero((np.diff(a.indptr)[:neq] == 1) & (rhs[:neq] == 0.))
    entries = []
    for row in rows:
        pos = a.indptr[row]
        col, value = int(a.indices[pos]), float(a.data[pos])
        entries.append(dict(row=int(row), original_row=int(original_rows[row]), column=col,
            coefficient=value, lower=float(lo[col]), upper=float(hi[col]),
            zero_strictly_inside_bounds=bool(lo[col] < 0. < hi[col])))
    return entries


def diagnose(problem, max_cycles=6, *, fix_singleton_equalities=False):
    """Alternate conservative maps, preserving the caller's input."""
    original_sha = problem_hash(problem)
    original_a, original_rhs, _lo, _hi, original_c, _neq = problem
    current = problem
    kept_original_rows = np.arange(original_a.shape[0])
    transform = eye(original_a.shape[1], format='csr', dtype=np.float64)
    offset = np.zeros(original_a.shape[1])
    cycles = []
    reached_fixed_point = False
    for cycle in range(1, max_cycles + 1):
        before = structure(current)
        started = time.perf_counter()
        forest = HomogeneousEqualityReduction(current)
        reduced_forest = forest.reduce(current)
        after_forest = reduced_forest.problem
        null_residual = max_abs_sparse(current[0][forest.eliminated_rows] @ forest.transform)
        eliminated_original_forest_rows = kept_original_rows[forest.eliminated_rows]
        transform = (transform @ forest.transform).tocsr()
        kept_original_rows = kept_original_rows[forest.kept_rows]
        forest_seconds = time.perf_counter() - started

        started = time.perf_counter()
        zero = ZeroFaceReduction(after_forest, fix_singleton_equalities=fix_singleton_equalities)
        original_fixed_contribution = np.asarray(transform[:, zero.fixed_columns] @ zero.fixed_values).ravel()
        offset += original_fixed_contribution
        transform = transform[:, zero.columns].tocsr()
        removed_original_zero_rows = kept_original_rows[zero.removed_zero_rows]
        removed_original_duplicate_rows = kept_original_rows[zero.removed_duplicate_rows]
        kept_original_rows = kept_original_rows[zero.rows]
        current = zero.reduced
        zero_seconds = time.perf_counter() - started
        cycles.append(dict(cycle=cycle, before=before, after_forest=structure(after_forest),
            after_zero_face=structure(current), forest_seconds=forest_seconds,
            zero_face_seconds=zero_seconds,
            forest_eliminated_equalities=len(forest.eliminated_rows),
            forest_null_map_max_abs_residual=null_residual,
            forest_eliminated_original_rows=eliminated_original_forest_rows.tolist(),
            zero_face_explicit_fixed_columns=len(zero.explicit_fixed_columns),
            zero_face_forced_zero_columns=len(zero.forced_zero_columns),
            zero_face_witness_rows=len(zero.witnesses),
            zero_face_singleton_witness_rows=sum(w.orientation == 'equality' for w in zero.witnesses),
            zero_face_skipped_small_singleton_rows=zero.skipped_small_singleton_rows.tolist(),
            zero_face_removed_original_zero_rows=removed_original_zero_rows.tolist(),
            zero_face_removed_original_duplicate_rows=removed_original_duplicate_rows.tolist()))
        if problem_hash(current) == before['problem_sha256']:
            reached_fixed_point = True
            break
    # These compare only retained transformed rows. Removed duplicate rows
    # are not all null constraints; no blanket eliminated-row claim is made.
    expected_a = (original_a[kept_original_rows] @ transform).tocsr()
    expected_rhs = original_rhs[kept_original_rows] - original_a[kept_original_rows] @ offset
    expected_c = np.asarray(transform.T @ original_c).ravel()
    if problem_hash(problem) != original_sha:
        raise AssertionError('Diagnostic mutated original input')
    return dict(original=structure(problem), final=structure(current), cycles=cycles,
        fix_singleton_equalities=fix_singleton_equalities,
        final_homogeneous_singletons=singletons(current, kept_original_rows),
        reached_fixed_point=reached_fixed_point,
        original_input_unchanged=True,
        composite_map=dict(shape=list(transform.shape), nonzeros=transform.nnz,
            original_columns_now_fixed=int(np.sum(np.diff(transform.indptr) == 0)),
            offset_nonzeros=int(np.count_nonzero(offset)),
            original_objective_offset=float(original_c @ offset),
            retained_row_matrix_max_abs_error=max_abs_sparse(current[0] - expected_a),
            retained_rhs_max_abs_error=float(np.max(np.abs(current[1]-expected_rhs), initial=0.)),
            objective_max_abs_error=float(np.max(np.abs(current[4]-expected_c), initial=0.))),
        limitations=['Structural presolve only; no primal/dual solution or speedup is claimed.',
            'No numerical rank or nearly dependent equality is removed.',
            'Forest floating-point null-map residual is reported; original-LP certification remains mandatory.',
            'The composed primal map does not replace reverse ordered bound-aware dual lifting.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, default=ROOT/'results/pf_coverage_holdout4x120_20260905')
    parser.add_argument('--cycles', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--fix-singleton-equalities', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Diagnostic results are never overwritten')
    manifest_path = args.trace/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entry = select_entries(manifest, 'exchange', 0, [1])[0]
    problem = _load_problem_without_reference(args.trace, entry)
    record = diagnose(problem, args.cycles, fix_singleton_equalities=args.fix_singleton_equalities)
    record.update(role='input_only_structural_diagnostic', stage='exchange', environment=0,
        step=1, entry=entry, reference_vectors_loaded=False, cpu_lp_calls=0,
        gpu_calls=0, qr_calls=0, input_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), ROOT/'src/lp_equality_reduction.py', ROOT/'src/lp_zero_face.py')})
    with args.output.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(output=str(args.output), original=record['original'], final=record['final'],
        reached_fixed_point=record['reached_fixed_point'], cycles=[dict(cycle=c['cycle'],
            forest_removed=c['forest_eliminated_equalities'], forced_zero=c['zero_face_forced_zero_columns'],
            after=c['after_zero_face']) for c in record['cycles']])))


if __name__ == '__main__':
    main()
