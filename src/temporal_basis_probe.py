"""Read-only diagnostics for interpreting a previous HiGHS basis on a new LP.

The probe never supplies its candidate to the simulation.  It reconstructs
the primal/dual solution implied by the old combinatorial basis, using a fresh
sparse LU of the *current* basis matrix, and applies the unchanged original-LP
certificate.  The real trajectory is advanced only by ``RepeatedCpuLP``.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
import time

import numpy as np
from scipy.sparse.linalg import splu

from .cpu_repeated_lp import _certificate, _problem
from .gpu_compiled_community_backend import stage_key


@dataclass(frozen=True)
class BasisSnapshot:
    """Host-only combinatorial basis from one certified HiGHS solve."""
    col_status: np.ndarray
    row_status: np.ndarray
    stage: str
    shape: tuple
    neq: int


def request_key(objective, kwargs, environment_id, n_fluxes):
    problem = _problem(objective, kwargs)
    a, _, _, _, c, neq = problem
    stage = kwargs.get('_stage')
    if stage is None:
        stage = stage_key(c, a, neq, n_fluxes)[0] if n_fluxes is not None else 'unspecified'
    return (environment_id, stage, a.shape[0], a.shape[1], neq), problem


def snapshot_from_solver(solver, stage, shape, neq):
    basis = solver.getBasis()
    if not basis.valid:
        raise ValueError('Cannot snapshot an invalid HiGHS basis')
    return BasisSnapshot(
        col_status=np.fromiter((int(value) for value in basis.col_status), dtype=np.int8),
        row_status=np.fromiter((int(value) for value in basis.row_status), dtype=np.int8),
        stage=str(stage), shape=tuple(shape), neq=int(neq))


def basis_change(previous, current):
    if previous.shape != current.shape or previous.neq != current.neq:
        return dict(comparable=False, changed_columns=None, changed_rows=None,
            basic_column_symmetric_difference=None, active_row_symmetric_difference=None)
    basic = 1  # highspy.HighsBasisStatus.kBasic
    return dict(comparable=True,
        changed_columns=int(np.count_nonzero(previous.col_status != current.col_status)),
        changed_rows=int(np.count_nonzero(previous.row_status != current.row_status)),
        basic_column_symmetric_difference=int(np.count_nonzero(
            (previous.col_status == basic) != (current.col_status == basic))),
        active_row_symmetric_difference=int(np.count_nonzero(
            (previous.row_status != basic) != (current.row_status != basic))))


def reinterpret_basis(problem, previous):
    """Solve the current LP at an old basis and certify it in original units."""
    started = time.perf_counter()
    a, rhs, lower, upper, c, neq = problem
    m, n = a.shape
    record = dict(available=True, success=False, stage=previous.stage,
        rows=m, columns=n, equalities=neq, failure=None)
    if previous.shape != a.shape or previous.neq != neq:
        record['failure'] = 'structure_changed'
        record['total_seconds'] = time.perf_counter()-started
        return record
    if previous.col_status.shape != (n,) or previous.row_status.shape != (m,):
        record['failure'] = 'invalid_basis_dimensions'
        record['total_seconds'] = time.perf_counter()-started
        return record
    # HiGHS enum values: lower=0, basic=1, upper=2, zero=3.  kNonbasic=4
    # has no unique bound/value interpretation and is deliberately rejected.
    lower_status, basic_status, upper_status, zero_status = 0, 1, 2, 3
    supported_columns = np.isin(previous.col_status,
        (lower_status, basic_status, upper_status, zero_status))
    supported_rows = np.isin(previous.row_status,
        (lower_status, basic_status, upper_status, zero_status))
    if not supported_columns.all() or not supported_rows.all():
        record['failure'] = 'unsupported_basis_status'
        record['total_seconds'] = time.perf_counter()-started
        return record
    basic_columns = np.flatnonzero(previous.col_status == basic_status)
    active_rows = np.flatnonzero(previous.row_status != basic_status)
    record.update(basic_columns=int(len(basic_columns)), active_rows=int(len(active_rows)))
    if not len(basic_columns) or len(basic_columns) != len(active_rows):
        record['failure'] = 'non_square_reduced_basis'
        record['total_seconds'] = time.perf_counter()-started
        return record
    x = np.zeros(n, dtype=np.float64)
    lower_ids = previous.col_status == lower_status
    upper_ids = previous.col_status == upper_status
    zero_ids = previous.col_status == zero_status
    x[lower_ids] = lower[lower_ids]; x[upper_ids] = upper[upper_ids]
    if (not np.isfinite(x[~(previous.col_status == basic_status)]).all()
            or np.any((zero_ids & ((lower > 0.) | (upper < 0.))))):
        record['failure'] = 'nonbasic_value_outside_current_finite_bound'
        record['total_seconds'] = time.perf_counter()-started
        return record
    # SciPy-style equalities have row lower == upper == rhs. Inequalities in
    # this backend have row lower == -inf and row upper == rhs. A nonbasic
    # inequality row at kLower therefore has no finite value to impose.
    # kZero means row activity zero, not the upper RHS; accept it only when
    # zero lies in the actual row interval and solve against zero explicitly.
    row_target = rhs.copy()
    if neq < m:
        inequality_status = previous.row_status[neq:]
        if np.any(inequality_status == lower_status):
            record['failure'] = 'unsupported_infinite_lower_row_status'
            record['total_seconds'] = time.perf_counter()-started
            return record
        zero_rows = np.flatnonzero(previous.row_status == zero_status)
        zero_inequalities = zero_rows[zero_rows >= neq]
        if len(zero_inequalities):
            if np.any(rhs[zero_inequalities] < 0.):
                record['failure'] = 'zero_row_status_outside_current_bounds'
                record['total_seconds'] = time.perf_counter()-started
                return record
            row_target[zero_inequalities] = 0.
    zero_equalities = np.flatnonzero(
        (previous.row_status[:neq] == zero_status) & (rhs[:neq] != 0.))
    if len(zero_equalities):
        record['failure'] = 'zero_row_status_outside_current_bounds'
        record['total_seconds'] = time.perf_counter()-started
        return record
    assembly_finished = time.perf_counter()
    try:
        factor_started = time.perf_counter()
        factor = splu(a[active_rows][:, basic_columns].tocsc())
        factor_seconds = time.perf_counter()-factor_started
        solve_started = time.perf_counter()
        residual = row_target[active_rows] - a[active_rows] @ x
        x[basic_columns] = factor.solve(np.asarray(residual))
        y = np.zeros(m, dtype=np.float64)
        y[active_rows] = factor.solve(c[basic_columns], trans='T')
        solve_seconds = time.perf_counter()-solve_started
    except (RuntimeError, ValueError) as error:
        record.update(failure='singular_or_invalid_current_basis', error=str(error),
            assembly_seconds=assembly_finished-started,
            factor_seconds=time.perf_counter()-factor_started,
            solve_seconds=0., certificate_seconds=0.,
            total_seconds=time.perf_counter()-started)
        return record
    reduced = c-a.T@y
    candidate = SimpleNamespace(col_value=x, row_dual=y, col_dual=reduced,
        value_valid=True, dual_valid=True)
    certificate_started = time.perf_counter()
    certificate = _certificate(a, rhs, lower, upper, c, neq, candidate)
    certificate_seconds = time.perf_counter()-certificate_started
    record.update(success=bool(certificate['certificate_passed']),
        failure=None if certificate['certificate_passed'] else 'original_lp_certificate_failed',
        assembly_seconds=assembly_finished-started, factor_seconds=factor_seconds,
        solve_seconds=solve_seconds, certificate_seconds=certificate_seconds,
        objective=float(c@x) if np.isfinite(x).all() else None,
        **certificate, total_seconds=time.perf_counter()-started)
    return record


class TemporalBasisProbeBackend:
    """Diagnostic wrapper around the exact persistent CPU trajectory backend."""
    name = 'cpu_persistent_highs_with_read_only_temporal_probe'

    def __init__(self, cpu_backend, *, stages=('aggregate', 'exchange'), workers=4):
        self.cpu = cpu_backend
        self.n_fluxes = cpu_backend.n_fluxes
        self.stages = frozenset(stages)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.previous = {}
        self.history = []
        self.closed = False

    def solve_batch(self, requests, *, environment_ids=None):
        if self.closed:
            raise RuntimeError('Temporal probe is closed')
        if environment_ids is None:
            environment_ids = list(range(len(requests)))
        if len(environment_ids) != len(requests) or len(set(environment_ids)) != len(environment_ids):
            raise ValueError('One unique stable environment ID is required per request')
        keyed = [request_key(c, kwargs, env, self.n_fluxes)
            for env, (c, kwargs) in zip(environment_ids, requests)]
        probe_started = time.perf_counter()
        jobs = [(problem, self.previous.get(key)) for key, problem in keyed]
        futures = [self.pool.submit(reinterpret_basis, problem, old)
            if old is not None and key[1] in self.stages else None
            for (key, problem), (problem, old) in zip(keyed, jobs)]
        probes = [future.result() if future is not None else dict(
            available=False, success=False, failure='no_previous_basis', total_seconds=0.)
            for future in futures]
        probe_wall_seconds = time.perf_counter()-probe_started
        # A failed interpretation must never be tried again merely because
        # the following exact CPU solve also fails. Preserve ``jobs`` below
        # for basis-change diagnostics, but remove stale live cache entries.
        for (key, _), probe in zip(keyed, probes):
            if probe['available'] and not probe['success']:
                self.previous.pop(key, None)
        exact_started = time.perf_counter()
        results = self.cpu.solve_batch(requests, environment_ids=environment_ids)
        exact_wall_seconds = time.perf_counter()-exact_started
        rows = []
        for (key, problem), (_, old), result, probe in zip(keyed, jobs, results, probes):
            current = None; change = dict(comparable=False, changed_columns=None,
                changed_rows=None, basic_column_symmetric_difference=None,
                active_row_symmetric_difference=None)
            if result.success:
                entry = self.cpu.models.get(key)
                if entry is None:
                    raise RuntimeError('Certified CPU model missing after solve')
                current = snapshot_from_solver(entry['solver'], key[1], problem[0].shape, problem[-1])
                if old is not None:
                    change = basis_change(old, current)
                self.previous[key] = current
            rows.append(dict(environment_id=key[0], stage=key[1],
                cpu_success=bool(result.success), cpu_simplex_iterations=result.diagnostics['simplex_iterations'],
                cpu_basis_reused=result.diagnostics['basis_reused'], probe=probe,
                previous_to_current_basis_change=change))
        self.history.append(dict(batch=len(requests), probe_wall_seconds=probe_wall_seconds,
            probe_sum_seconds=sum(row['probe']['total_seconds'] for row in rows),
            exact_cpu_wall_seconds=exact_wall_seconds, rows=rows))
        return results

    def reset_trajectory(self):
        """Clear diagnostic and exact-solver state between independent runs."""
        self.previous.clear()
        self.cpu.models.clear()

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True)
            self.cpu.close()
            self.previous.clear()
            self.closed = True
