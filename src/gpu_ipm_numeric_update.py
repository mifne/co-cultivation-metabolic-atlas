"""Transactional numeric rebind for an existing globalized forest GPU IPM.

This is host sparse algebra and staged device copies, never a CPU optimizer.
Coordinates, finite/fixed-bound topology and every CSR pattern must survive.
Old numerical factors and cached internal states are invalidated; only the
native handle, symbolic analysis, fixed layouts and compatible work buffers
are retained. Every later solve must re-certify the NEW full original LP.
"""

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse import bmat, diags
from .csr_block_assembly import block_diag

from .gpu_batched_ipm import constraint_form, uniform_kkt_pattern
from .gpu_block_lp import assemble_blocks
from .gpu_pdhg_corrector import _validated_problem
from .lp_exact_equalities import ExactEqualityReduction
from .lp_trace import problem_hash
from .lp_zero_face import _freeze_problem


class NumericRebindRejected(ValueError):
    """Before-commit rejection: use a fresh workspace, not stale state."""


@dataclass(frozen=True)
class _HostState:
    full_problems: tuple
    forest_reductions: tuple
    forest_problems: tuple
    face_plans: tuple
    secondary_forest_reductions: tuple
    equality_plans: tuple
    problems: tuple


def _pattern_equal(a, b):
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices))


def _same_coordinates(old, new, label):
    if old[-1] != new[-1] or not _pattern_equal(old[0], new[0]):
        raise NumericRebindRejected(f'{label} CSR pattern/equality coordinates changed; rebuild required')
    for old_bound, new_bound in zip(old[2:4], new[2:4]):
        if not np.array_equal(np.isfinite(old_bound), np.isfinite(new_bound)):
            raise NumericRebindRejected(f'{label} finite-bound topology changed; rebuild required')
    if not np.array_equal(old[2] == old[3], new[2] == new[3]):
        raise NumericRebindRejected(f'{label} fixed-bound topology changed; rebuild required')


def _existing_host_state(solver):
    return _HostState(tuple(solver.full_problems), tuple(solver.forest_reductions),
        tuple(solver.forest_problems), tuple(solver.face_plans),
        tuple(solver.secondary_forest_reductions), tuple(solver.equality_plans), tuple(solver.problems))


def _forest_coordinate_integrity(plan):
    """Cheap consistency checks for legacy mutable forest indexing metadata."""
    transform = plan.T
    if (not np.all(np.diff(transform.indptr) == 1)
            or not np.array_equal(transform.indices, plan.original_to_reduced)
            or not np.array_equal(transform.data, plan.weights)
            or np.any(np.asarray(plan.representatives) < 0)
            or np.any(np.asarray(plan.representatives) >= transform.shape[0])
            or not np.array_equal(transform.indices[plan.representatives],
                                  np.arange(transform.shape[1]))):
        raise NumericRebindRejected('Forest coordinate metadata is inconsistent with its transform')


def prepare_host_rebind(solver, problems):
    """Re-prove maps without mutation; classify any invalid input as rejection."""
    try:
        return _prepare_host_rebind(solver, problems)
    except NumericRebindRejected:
        raise
    except ValueError as error:
        raise NumericRebindRejected(str(error)) from error


def _prepare_host_rebind(solver, problems):
    """Re-prove saved maps and build fresh snapshots without touching solver."""
    from .gpu_forest_ipm import ForestGpuBatchedIPM
    if getattr(solver,'host_reduced_snapshots_are_layout_only',False):
        raise NumericRebindRejected('Device-updated intermediate host snapshots are layout-only; use the bound device plan or rebuild')
    if type(solver) is not ForestGpuBatchedIPM:
        raise NumericRebindRejected('Only the concrete ForestGpuBatchedIPM is supported')
    if (not solver.globalized or solver.near_equality_plans
            or getattr(solver, 'equality_row_scaling', False)):
        raise NumericRebindRejected('Require globalized mode, no near-equality relaxation or equality scaling')
    old = _existing_host_state(solver)
    full = tuple(_freeze_problem(_validated_problem(p)) for p in problems)
    if len(full) != solver.batch or len(old.full_problems) != solver.batch:
        raise NumericRebindRejected('The number/order of environment slots must remain unchanged')
    if tuple(problem_hash(p) for p in old.full_problems) != tuple(solver.problem_hashes):
        raise NumericRebindRejected('Current full-original snapshot/hash is stale or modified')
    for plan in tuple(solver.forest_plans)+tuple(solver.secondary_forest_plans):
        _forest_coordinate_integrity(plan)
    for i, (before, after) in enumerate(zip(old.full_problems, full)):
        _same_coordinates(before, after, f'full environment{i}')
    forests = tuple(plan.reduce(p) for plan, p in zip(solver.forest_plans, full))
    forest_problems = tuple(r.problem for r in forests)
    if len(forests) != solver.batch:
        raise NumericRebindRejected('Incomplete forest plan batch')
    for i, (before, after) in enumerate(zip(old.forest_problems, forest_problems)):
        _same_coordinates(before, after, f'forest environment{i}')
    faces = tuple(plan.rebind(p) for plan, p in zip(old.face_plans, forest_problems))
    if len(faces) != solver.batch:
        raise NumericRebindRejected('Incomplete zero-face plan batch')
    for before, after in zip(old.face_plans, faces):
        for name in ('rows', 'columns', 'fixed_columns', 'fixed_values', 'explicit_fixed_columns',
                     'forced_zero_columns', 'removed_duplicate_rows', 'removed_zero_rows'):
            if not np.array_equal(getattr(before, name), getattr(after, name)):
                raise NumericRebindRejected('Zero-face coordinate map changed: '+name)
        if before.witnesses != after.witnesses or before.duplicate_equalities != after.duplicate_equalities:
            raise NumericRebindRejected('Zero-face postsolve proof changed')
        _same_coordinates(before.reduced, after.reduced, 'zero-face')
    reduced = tuple(p.reduced for p in faces)
    secondary = ()
    if solver.secondary_forest_plans:
        secondary = tuple(plan.reduce(p) for plan, p in zip(solver.secondary_forest_plans, reduced))
        if len(secondary) != solver.batch:
            raise NumericRebindRejected('Incomplete secondary forest plan batch')
        for before, after in zip(old.secondary_forest_reductions, secondary):
            _same_coordinates(before.problem, after.problem, 'secondary forest')
        reduced = tuple(r.problem for r in secondary)
    equalities = ()
    if old.equality_plans:
        equalities = tuple(ExactEqualityReduction(p, reuse_from=plan)
                           for p, plan in zip(reduced, old.equality_plans))
        if len(equalities) != solver.batch:
            raise NumericRebindRejected('Incomplete exact equality plan batch')
        for before, after in zip(old.equality_plans, equalities):
            if (not np.array_equal(before.rows, after.rows)
                    or before.proof_fingerprint != after.proof_fingerprint
                    or before.compression_fingerprint != after.compression_fingerprint):
                raise NumericRebindRejected('Exact equality coordinate/dual map changed')
        reduced = tuple(p.reduced for p in equalities)
    for before, after in zip(old.problems, reduced):
        _same_coordinates(before, after, 'final IPM')
    return old, _HostState(full, forests, forest_problems, faces, secondary, equalities, reduced)


def _payloads(solver, state, *, _forms=None, _old_forms=None, _direct_kkt_payload=False):
    """Exhaustive mutable operator/certificate/postsolve numerical payloads.

    The private optional forms are freshly validated within the SAME rebind
    call. Sharing them avoids repeating sparse constraint construction; it is
    not a cross-update cache. The original two-argument inspection API still
    constructs and validates its own forms, and every path checks old/new
    fixed/lower/upper coordinate masks below.
    """
    sparse, arrays = {}, {}
    forms = [constraint_form(p) for p in state.problems] if _forms is None else _forms
    old_forms = [constraint_form(p) for p in solver.problems] if _old_forms is None else _old_forms
    for before, after in zip(old_forms, forms):
        for index in (4, 5, 6):
            if not np.array_equal(before[index], after[index]):
                raise NumericRebindRejected('Constraint fixed/lower/upper coordinate masks changed')
    sparse[('e',)] = block_diag([f[0] for f in forms], format='csr')
    sparse[('g',)] = block_diag([f[2] for f in forms], format='csr')
    sparse[('et',)] = sparse[('e',)].T.tocsr()
    sparse[('gt',)] = sparse[('g',)].T.tocsr()
    arrays[('b',)] = np.stack([f[1] for f in forms])
    arrays[('h',)] = np.stack([f[3] for f in forms])
    arrays[('c',)] = np.stack([p[4] for p in state.problems])
    arrays[('fixed',)] = np.flatnonzero(forms[0][4])
    arrays[('il',)], arrays[('iu',)] = forms[0][5], forms[0][6]
    for name, problems in (('assembled', state.problems),
                           ('original_assembled', state.forest_problems),
                           ('_full_assembled', state.full_problems)):
        packed = assemble_blocks(problems)
        sparse[(name, 0)] = packed[0]
        for index in range(1, 6): arrays[(name, index)] = packed[index]
    sparse[('original_at',)] = sparse[('original_assembled', 0)].T.tocsr()
    sparse[('_full_at',)] = sparse[('_full_assembled', 0)].T.tocsr()
    # These static maps are included so a mutated host plan cannot silently
    # propagate a transform different from the already prepared GPU map.
    for name, attr in (('_forest_t', 'T'), ('_forest_compression', 'compression'),
                       ('_forest_dual_lift', 'dual_lift_map')):
        sparse[(name,)] = block_diag([getattr(p, attr) for p in solver.forest_plans], format='csr')
    sparse[('_forest_tt',)] = sparse[('_forest_t',)].T.tocsr()
    for name, attr in (('_forest_kept_rows', 'kept_rows'),
                       ('_forest_eliminated_rows', 'eliminated_rows'), ('_forest_weights', 'weights')):
        arrays[(name,)] = np.stack([getattr(p, attr) for p in solver.forest_plans])
    for name in ('lower_witness', 'upper_witness', 'fallback_witness'):
        arrays[('_forest_'+name,)] = np.stack([getattr(r, name) for r in state.forest_reductions])
    faces = state.face_plans
    arrays[('columns',)] = np.stack([p.columns for p in faces])
    arrays[('rows',)] = np.stack([p.rows for p in faces])
    template = np.zeros((solver.batch, solver.original_n))
    duplicates, representatives, signs = [], [], []
    offsets, wrows, orientations, new_offsets, columns, values = [0], [], [], [0], [], []
    for i, plan in enumerate(faces):
        template[i, plan.fixed_columns] = plan.fixed_values
        for duplicate in plan.duplicate_equalities:
            duplicates.append(i*solver.original_m+duplicate.row)
            representatives.append(i*solver.original_m+duplicate.representative_row)
            signs.append(duplicate.sign)
        for witness in plan.witnesses:
            wrows.append(i*solver.original_m+witness.row)
            orientations.append({'min': -1, 'max': 1, 'equality': 0}[witness.orientation])
            columns.extend(i*solver.original_n+np.asarray(witness.newly_fixed_columns))
            values.extend(witness.newly_fixed_coefficients)
            new_offsets.append(len(columns))
        offsets.append(len(wrows))
    arrays[('fixed_template',)] = template
    arrays[('duplicate_rows',)] = np.asarray(duplicates, dtype=np.int64)
    arrays[('duplicate_representatives',)] = np.asarray(representatives, dtype=np.int64)
    arrays[('duplicate_signs',)] = np.asarray(signs, dtype=np.float64)
    for i, value in enumerate((offsets, wrows, orientations, new_offsets, columns)):
        arrays[('proof_arrays', i)] = np.asarray(value, dtype=np.int32)
    arrays[('proof_values',)] = np.asarray(values, dtype=np.float64)
    if state.equality_plans:
        arrays[('equality_keep',)] = np.stack([p.rows for p in state.equality_plans])
        sparse[('equality_dual_compression',)] = block_diag(
            [p.dual_compression for p in state.equality_plans], format='csr')
    if hasattr(solver, '_condensed_h') or hasattr(solver, '_condensed_ht'):
        if not (hasattr(solver, '_condensed_h') and hasattr(solver, '_condensed_ht')):
            raise NumericRebindRejected('Incomplete cached condensed operator')
        sparse[('_condensed_h',)] = block_diag([f[2][:solver.q] for f in forms], format='csr')
        sparse[('_condensed_ht',)] = sparse[('_condensed_h',)].T.tocsr()
    if _direct_kkt_payload:
        from .gpu_ipm_kkt_payload import build_condensed_kkt_payload
        pattern = solver.factor.host_pattern
        kkt_values, diagonal = build_condensed_kkt_payload(forms, pattern,
            n=solver.n, ne=solver.ne, q=solver.q, regularization=solver.regularization)
    else:
        matrices = [bmat([[diags(np.ones(solver.n)), f[0].T, f[2][:solver.q].T],
            [f[0], diags(-np.ones(solver.ne)), None],
            [f[2][:solver.q], None, diags(-np.ones(solver.q))]], format='csr') for f in forms]
        pattern, kkt_values, diagonal = uniform_kkt_pattern(matrices)
    if not _pattern_equal(pattern, solver.factor.host_pattern):
        raise NumericRebindRejected('cuDSS symbolic KKT pattern changed; rebuild required')
    kkt_values[:, diagonal[:solver.n]] = solver.regularization
    kkt_values[:, diagonal[solver.n:solver.n+solver.ne]] = -solver.regularization
    arrays[('values',)] = kkt_values
    arrays[('factor', 'values')] = kkt_values
    arrays[('diagonal',)] = diagonal
    arrays[('kkt_rows',)] = np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr))
    arrays[('kkt_columns',)] = pattern.indices
    return sparse, arrays


def _target(solver, path):
    value = solver
    for part in path:
        value = value[part] if isinstance(part, int) else getattr(value, part)
    return value


def _stage_updates(solver, old_payload, new_payload):
    """No target is modified until all source values and buffers are ready."""
    cp = solver.cp
    old_sparse, old_arrays = old_payload
    new_sparse, new_arrays = new_payload
    if old_sparse.keys() != new_sparse.keys() or old_arrays.keys() != new_arrays.keys():
        raise NumericRebindRejected('Prepared operator/certificate payload set changed')
    copies = []
    skipped = 0
    for path, new_matrix in new_sparse.items():
        old_matrix = old_sparse[path]
        target = _target(solver, path)
        if (not _pattern_equal(old_matrix, new_matrix) or target.shape != new_matrix.shape
                or not np.array_equal(cp.asnumpy(target.indptr), old_matrix.indptr)
                or not np.array_equal(cp.asnumpy(target.indices), old_matrix.indices)
                or not np.array_equal(cp.asnumpy(target.data), old_matrix.data)):
            raise NumericRebindRejected('Stale/changed sparse operator or map: '+str(path))
        if np.array_equal(old_matrix.data, new_matrix.data):
            skipped += 1
            continue
        source = cp.asarray(new_matrix.data, dtype=target.data.dtype)
        copies.append((str(path)+'.data', target.data, source))
    dynamic = {('values',), ('factor', 'values')}
    for path, value in new_arrays.items():
        target = _target(solver, path)
        old_value = np.asarray(old_arrays[path])
        value = np.asarray(value)
        if (target.shape != value.shape or old_value.shape != value.shape
                or target.device.id != solver.factor.device):
            raise NumericRebindRejected('Device vector shape/device changed: '+str(path))
        if path not in dynamic and not np.array_equal(cp.asnumpy(target), old_value):
            raise NumericRebindRejected('Stale current operator/certificate vector: '+str(path))
        if path not in dynamic and np.array_equal(old_value, value):
            skipped += 1
            continue
        source = cp.asarray(value, dtype=target.dtype)
        copies.append((str(path), target, source))
    for name, target, source in copies:
        if source.shape != target.shape or source.device.id != solver.factor.device:
            raise NumericRebindRejected('Staging mismatch: '+name)
    solver.factor.stream.synchronize()
    return copies, skipped


def rebind_forest_ipm(solver, new_problems, *, reuse_static_forest=False, device_staging=False,
                      direct_kkt_payload=False):
    """Update a compatible workspace transactionally; preserve symbolic factor.

    Preparation rejection leaves the old workspace intact. Any error after
    commit begins marks the factor failed and unusable; no mixed-state solve
    is allowed. Caller can close and build a fresh workspace explicitly.
    Compatible GMRES scratch arenas survive, but no old acceptance or internal
    warm-state object is automatically reused for the new LP.

    ``payload_assembly_and_staging_seconds`` includes host sparse assembly,
    old-device-value validation, optional secondary-map construction and
    device staging. It is deliberately NOT a GPU-only transfer timing.
    Before-commit ValueError becomes NumericRebindRejected; resource/runtime
    failures retain their original types, as do all failures after commit.
    """
    started = time.perf_counter()
    try:
        if any(type(value) is not bool for value in (reuse_static_forest, device_staging, direct_kkt_payload)):
            raise ValueError('Explicit boolean numeric update optimizations required')
        solver.factor._context()
        old, new = prepare_host_rebind(solver, new_problems)
        host_seconds = time.perf_counter()-started
        old_hashes = tuple(solver.problem_hashes)
        # No persistent payload cache: revalidate both current and incoming
        # final LPs once, then share these per-call sparse forms. Previously
        # the old forms were built three times and the new forms once.
        old_forms = tuple(constraint_form(p) for p in old.problems)
        new_forms = tuple(constraint_form(p) for p in new.problems)
        old_payload = _payloads(solver, old, _forms=old_forms, _old_forms=old_forms,
                                _direct_kkt_payload=direct_kkt_payload)
        new_payload = _payloads(solver, new, _forms=new_forms, _old_forms=old_forms,
                                _direct_kkt_payload=direct_kkt_payload)
        staged_map = None
        if new.secondary_forest_reductions:
            from .gpu_forest_map import GpuForestMap
            current = solver.secondary_forest_map
            staged_map = (current.rebind_copy(new.secondary_forest_reductions) if reuse_static_forest else
                GpuForestMap(solver.secondary_forest_plans, new.secondary_forest_reductions, cp=solver.cp))
            for name in ('original_n', 'original_m', 'reduced_n', 'reduced_m', 'batch'):
                if getattr(current, name) != getattr(staged_map, name):
                    raise NumericRebindRejected('Secondary forest map coordinates changed')
            for name in ('transform', 'compression', 'dual_lift'):
                before, after = getattr(current._host, name), getattr(staged_map._host, name)
                if not _pattern_equal(before, after) or not np.array_equal(before.data, after.data):
                    raise NumericRebindRejected('Secondary forest static map changed: '+name)
            for name in ('kept_rows', 'eliminated_rows', 'weights'):
                if not np.array_equal(getattr(current._host, name), getattr(staged_map._host, name)):
                    raise NumericRebindRejected('Secondary forest coordinate array changed: '+name)
        if device_staging:
            from .gpu_ipm_staging import stage_numeric_updates
            copies, skipped = stage_numeric_updates(solver, old_payload, new_payload)
        else:
            copies, skipped = _stage_updates(solver, old_payload, new_payload)
        staging_seconds = time.perf_counter()-started-host_seconds
        solver.factor._context()
        if tuple(solver.problem_hashes) != old_hashes:
            raise NumericRebindRejected('Workspace was rebound while preparation was in progress')
        factor = solver.factor
        analysis_count = factor.analysis_count
        factor_identity = id(factor)
    except NumericRebindRejected:
        raise
    except ValueError as error:
        raise NumericRebindRejected(str(error)) from error
    commit_started = time.perf_counter()
    factor.factored = False
    solver._last_internal_state = None
    solver.failure_snapshot = None
    try:
        for _name, target, source in copies:
            solver.cp.copyto(target, source)
        factor.stream.synchronize()
        solver.full_problems = new.full_problems
        solver.forest_reductions = new.forest_reductions
        solver.forest_problems = new.forest_problems
        solver.face_plans = list(new.face_plans)
        solver.original_problems = [p.original for p in new.face_plans]
        solver.secondary_forest_reductions = list(new.secondary_forest_reductions)
        solver.equality_plans = list(new.equality_plans)
        solver.problems = list(new.problems)
        if staged_map is not None: solver.secondary_forest_map = staged_map
        solver.problem_hashes = tuple(problem_hash(p) for p in new.full_problems)
        solver._forest_certificate_seconds = 0.
        solver.equality_setup_seconds = sum(p.summary['setup_seconds'] for p in new.equality_plans)
        generation = getattr(solver, 'numeric_update_generation', 0)+1
        solver.numeric_update_generation = generation
        result = dict(generation=generation, old_problem_hashes=old_hashes,
            new_problem_hashes=solver.problem_hashes,
            native_factor_object_preserved=id(solver.factor) == factor_identity,
            symbolic_analysis_count_before=analysis_count,
            symbolic_analysis_count_after=factor.analysis_count,
            numerical_factor_invalidated=not factor.factored,
            automatic_warm_state_reuse=False, cpu_lp_calls=0,
            copied_device_payloads=[name for name, _target_value, _source in copies],
            unchanged_device_payloads_skipped=skipped,
            secondary_forest_map_rebuilt=staged_map is not None,
            secondary_forest_static_maps_reused=bool(reuse_static_forest and staged_map is not None),
            device_staging=device_staging,
            direct_kkt_payload=direct_kkt_payload,
            exact_equality_qr_calls=0,
            host_reproof_seconds=host_seconds,
            payload_assembly_and_staging_seconds=staging_seconds,
            commit_seconds=time.perf_counter()-commit_started,
            total_seconds=time.perf_counter()-started,
            scope='Compatible numeric input update only; new original LP certificates required; no solve performed')
        solver.last_numeric_update = result
        return result
    except BaseException as error:
        factor.failed = True
        try: factor.stream.synchronize()
        except BaseException as draining:
            error.add_note(f'Failed numeric update stream drain also failed: {draining}')
        raise
