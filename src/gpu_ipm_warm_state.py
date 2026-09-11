"""Explicit, causal transfer of GPU IPM interior proposals between LP inputs.

This is not a solution cache: each new LP is independently certified and may
require Newton corrections. Only GPU-produced, originally certified previous
states can be exported. Original variables/row maps and bound topology must
match; numerical RHS/cost/bound values may differ. No CPU optimizer is used.
"""
from dataclasses import dataclass
import hashlib
import json
import threading
import weakref

import numpy as np


def coordinate_signature(solver):
    """Conservative host metadata key, excluding varying LP numerical data.

    Symbolic-factor reuse is a DIFFERENT contract. This signature only permits
    an old interior point to retain its coordinate meaning in a new workspace.
    New constructors still build/prove every current reduction independently.
    """
    digest = hashlib.sha256()
    def add(value):
        if hasattr(value, 'tocsr'):
            matrix = value.tocsr()
            add(matrix.shape)
            for part in (matrix.indptr, matrix.indices, matrix.data): add(part)
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    add([type(solver).__module__, type(solver).__name__, solver.batch,
         solver.n, solver.m, solver.ne, solver.ng, solver.neq])
    if getattr(solver, 'equality_row_scaling', False):
        raise ValueError('Internal warm transfer currently excludes equality row scaling')
    for a, rhs, lo, hi, c, neq in solver.problems:
        add(a.shape); add(a.indptr); add(a.indices); add(neq)
        add(np.isfinite(lo)); add(np.isfinite(hi)); add(lo == hi)
    for name in ('forest_plans', 'secondary_forest_plans'):
        plans = getattr(solver, name, ())
        add([name, len(plans)])
        for plan in plans:
            add(plan.kept_rows); add(plan.eliminated_rows); add(plan.T)
    for name in ('face_plans', 'equality_plans', 'near_equality_plans'):
        plans = getattr(solver, name, ())
        add([name, len(plans)])
        for plan in plans:
            add(plan.rows)
            if hasattr(plan, 'columns'): add(plan.columns)
    return digest.hexdigest()


def _sequence(environment_ids, stage, step, batch):
    ids = tuple(environment_ids)
    if (len(ids) != batch or any(type(i) is not int or i < 0 for i in ids)
            or len(set(ids)) != len(ids) or stage not in ('maxmin', 'aggregate', 'exchange')
            or type(step) is not int or step < 0):
        raise ValueError('Unique ordered integer environments, known stage and nonnegative step required')
    return ids


def _validated_copies(solver, arrays, *, interior_floor=0.):
    cp = solver.cp
    if (isinstance(interior_floor, bool) or not np.isfinite(interior_floor)
            or not 0. <= interior_floor <= 1e-2):
        raise ValueError('Finite interior proposal floor in [0, 1e-2] required')
    expected = ((solver.batch, solver.n), (solver.batch, solver.ne),
                (solver.batch, solver.ng), (solver.batch, solver.ng))
    if len(arrays) != 4:
        raise ValueError('Four internal x/y/z/s arrays required')
    for value, shape in zip(arrays, expected):
        if (not isinstance(value, cp.ndarray) or value.shape != shape or value.dtype != cp.float64
                or value.device.id != solver.factor.device):
            raise ValueError('Exact-shape FP64 interior arrays on the bound device required')
    valid = cp.all(cp.stack([cp.all(cp.isfinite(v)) for v in arrays]))
    valid &= cp.all(arrays[2] > 0.) & cp.all(arrays[3] > 0.)
    if not bool(valid):
        raise ValueError('Finite x/y and strictly positive finite z/s required')
    copies = tuple(v.copy() for v in arrays)
    if interior_floor:
        copies = (*copies[:2], cp.maximum(copies[2], interior_floor),
                  cp.maximum(copies[3], interior_floor))
    return copies


@dataclass(frozen=True)
class BoundGpuWarmState:
    _target: object
    _arrays: tuple
    metadata: dict
    _target_generation: int
    _target_hashes: tuple

    def initialize(self, solver):
        if self._target() is not solver:
            raise ValueError('Interior proposal is bound to another target workspace')
        solver.factor._context()
        if (getattr(solver, 'numeric_update_generation', 0) != self._target_generation
                or tuple(solver.problem_hashes) != self._target_hashes):
            raise ValueError('Target LP changed after binding the interior proposal')
        # Never return scratch views: the solver may replace/update its state.
        return _validated_copies(solver, self._arrays)


@dataclass(frozen=True)
class GpuIPMWarmState:
    _arrays: tuple
    signature: str
    environment_ids: tuple
    stage: str
    step: int
    device: int
    stream: int
    owner_thread: int
    source_problem_hashes: tuple
    source_certificate_basis: tuple
    source_native_dual_relative_kkt_gap: tuple

    def bind(self, solver, *, environment_ids, stage, step, interior_floor=0., repair_slacks=False,
             restart_mu=None):
        solver.factor._context()
        if type(repair_slacks) is not bool:
            raise ValueError('repair_slacks must be a bool')
        if restart_mu is not None and (interior_floor != 0. or repair_slacks):
            raise ValueError('A dual restart is a separate proposal mode; do not combine floor/slack repair')
        ids = _sequence(environment_ids, stage, step, solver.batch)
        if ids != self.environment_ids or stage != self.stage or step != self.step + 1:
            raise ValueError('Only the next step of the same ordered environments and stage is causal')
        if (solver.factor.device != self.device or solver.factor.stream.ptr != self.stream
                or threading.get_ident() != self.owner_thread):
            raise ValueError('Interior transfer must retain device, stream and owner thread')
        if coordinate_signature(solver) != self.signature:
            raise ValueError('Changed coordinate maps or bound topology require a fresh proposal')
        arrays = _validated_copies(solver, self._arrays, interior_floor=interior_floor)
        if repair_slacks:
            # Repair the internal initial iterate, never the model bounds or
            # acceptance thresholds. Infeasible x still requires a Newton solve.
            slack = solver.cp.maximum(solver.h-solver._mv(solver.g, arrays[0], solver.ng),
                                      max(1e-8, interior_floor))
            arrays = (*arrays[:3], slack)
        restart_metadata = None
        if restart_mu is not None:
            from .gpu_ipm_reoptimization import centered_bound_restart
            arrays, restart_metadata = centered_bound_restart(solver, arrays[0], mu=restart_mu)
        return BoundGpuWarmState(weakref.ref(solver), arrays, dict(
            source='previous_GPU_iterate_with_original_certified_output',
            source_original_certificate_basis=self.source_certificate_basis,
            source_native_dual_relative_kkt_gap=self.source_native_dual_relative_kkt_gap,
            native_interior_dual_certification_implied=False,
            source_step=self.step, target_step=step, environment_ids=ids, stage=stage,
            source_problem_sha256=self.source_problem_hashes,
            target_problem_sha256=tuple(solver.problem_hashes),
            coordinate_signature=self.signature, interior_floor=float(interior_floor),
            repair_slacks_from_current_activity=repair_slacks,
            bound_dual_restart=restart_metadata,
            current_CPU_solution_used=False, current_LP_requires_new_certificate=True),
            getattr(solver, 'numeric_update_generation', 0), tuple(solver.problem_hashes))


def export_internal_state(solver, *, environment_ids, stage, step):
    solver.factor._context()
    ids = _sequence(environment_ids, stage, step, solver.batch)
    arrays = getattr(solver, '_last_internal_state', None)
    if arrays is None:
        raise ValueError('No retained fully certified GPU state; enable retain_internal_state')
    owned = _validated_copies(solver, arrays)
    # Recompute the unchanged full-original certificate via dynamic dispatch.
    # An analytic box certificate remains legal only for a physically feasible
    # original x; the internal multipliers themselves are still just a proposal.
    metrics = solver.certificate(owned[0], solver._row_dual(owned[1], owned[2]))
    if not all(m['certificate_passed'] for m in metrics):
        raise ValueError('Cannot export an uncertified original GPU state')
    return GpuIPMWarmState(owned, coordinate_signature(solver), ids, stage, step,
        solver.factor.device, solver.factor.stream.ptr, threading.get_ident(), tuple(solver.problem_hashes),
        tuple(m.get('certificate_source', 'original_LP_gate') for m in metrics),
        tuple(m.get('solver_dual_relative_kkt_gap') for m in metrics))
