"""Persistent, exact HiGHS simplex comparator for repeated independent LPs.

This is a CPU benchmark backend, not an approximation or training component.
Every environment/stage owns a separate HiGHS model. Unchanged sparsity permits
coefficient, RHS, bound and cost updates followed by basis reoptimization;
structural changes rebuild safely. Host updates and certification are timed.
``solve_batch`` calls must be serialized and keep stable environment ordering.
Each environment retains one native owner thread through creation, hot updates,
solve, output snapshots and destruction. This is lifecycle hardening, not a
claim that a prior native crash has been reproduced or its cause established.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import numpy as np
from scipy.sparse import csr_matrix, vstack

from .lp_bounds import split_variable_bounds


_NUMERICAL_OPTIONS = ('kkt_tolerance', 'primal_feasibility_tolerance',
    'dual_feasibility_tolerance', 'primal_residual_tolerance',
    'dual_residual_tolerance', 'optimality_tolerance', 'presolve')
_QUALITY_FIELDS = ('max_primal_infeasibility', 'max_dual_infeasibility',
    'max_primal_residual_error', 'max_dual_residual_error',
    'max_complementarity_violation', 'primal_dual_objective_error')


def _solution_snapshot(solution):
    """Detach numerical output while the native model is on its owner thread."""
    return SimpleNamespace(**{name:np.array(getattr(solution, name, []), dtype=np.float64, copy=True)
        for name in ('col_value', 'row_value', 'col_dual', 'row_dual')},
        value_valid=bool(solution.value_valid), dual_valid=bool(solution.dual_valid))


class _OwnerSolverHandle:
    """Compatibility view; the native Highs instance never leaves its lane.

    Internal calls are direct on the owner. Existing diagnostic callers may
    use the four read methods below from another thread, receiving detached
    host snapshots rather than a native solver/reference. External mutation
    is rejected; changing an LP must go through the serialized solve path.
    """
    def __init__(self, service, lane, native):
        self._service, self._lane, self._native = service, lane, native

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        def invoke(*args, **kwargs):
            owner = threading.get_ident() == self._service._lane_thread_ids[self._lane]
            if not owner and name not in ('getSolution', 'getBasis', 'getOptions', 'getInfo'):
                raise RuntimeError('Native HiGHS mutations/access require the owner thread')
            def call():
                if self._native is None:
                    raise RuntimeError('The owner-thread HiGHS model is closed')
                value = getattr(self._native, name)(*args, **kwargs)
                if owner:
                    return value
                if name == 'getSolution':
                    return _solution_snapshot(value)
                if name == 'getBasis':
                    return SimpleNamespace(valid=bool(value.valid),
                        col_status=np.array(value.col_status, dtype=np.int32, copy=True),
                        row_status=np.array(value.row_status, dtype=np.int32, copy=True))
                fields = _NUMERICAL_OPTIONS if name == 'getOptions' else _QUALITY_FIELDS
                return SimpleNamespace(**{field:getattr(value, field) for field in fields})
            return call() if owner else self._service._call_lane(self._lane, call)
        return invoke

    def _dispose_owner(self):
        if threading.get_ident() != self._service._lane_thread_ids[self._lane]:
            raise RuntimeError('Native HiGHS destruction requires its owner thread')
        native, self._native = self._native, None
        if native is None:
            return
        failures = []
        try:
            try:
                native.disableCallbacks()
            except Exception as error:
                failures.append('disable callbacks: '+str(error))
            try:
                self._service._check(native.clear(), 'owner-thread model clear')
            except Exception as error:
                failures.append('clear: '+str(error))
        finally:
            # Release the last service-owned native reference BEFORE the
            # executor thread exits and destroys its native thread-local data.
            del native
        if failures:
            raise RuntimeError('; '.join(failures))


class _OwnedModelRegistry(dict):
    """Preserve metadata lookups and route legacy ``models.clear`` safely."""
    def __init__(self, service):
        super().__init__()
        self._service = service

    def __setitem__(self, key, value):
        previous = self.get(key)
        if previous is not None and previous is not value:
            self._service._release_handle(previous.get('_owner_handle'))
        super().__setitem__(key, value)

    def pop(self, key, *default):
        previous = self.get(key)
        if previous is not None:
            self._service._release_handle(previous.get('_owner_handle'))
        return super().pop(key, *default)

    def clear(self):
        self._service.clear_models()


def _problem(objective, kwargs):
    """Validate the continuous SciPy-style LP subset used by the simulator."""
    c = np.asarray(objective, dtype=np.float64)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all():
        raise ValueError('A finite one-dimensional objective is required')
    n = len(c)
    integrality = kwargs.get('integrality')
    if integrality is not None and np.any(np.asarray(integrality)):
        raise ValueError('The repeated CPU comparator supports continuous LPs only')
    if kwargs.get('method', 'highs-ds') not in ('highs', 'highs-ds'):
        raise ValueError('The persistent comparator uses HiGHS simplex')
    blocks, rhs = [], []
    for matrix_name, rhs_name in (('A_eq', 'b_eq'), ('A_ub', 'b_ub')):
        supplied = kwargs.get(matrix_name)
        matrix = csr_matrix((0, n), dtype=float) if supplied is None else csr_matrix(supplied, dtype=float)
        if matrix.shape[1] != n or not np.isfinite(matrix.data).all():
            raise ValueError('Invalid LP matrix')
        values = kwargs.get(rhs_name)
        values = np.empty(0) if values is None and not matrix.shape[0] else np.asarray(values, dtype=float)
        if values.shape != (matrix.shape[0],) or not np.isfinite(values).all():
            raise ValueError('Each matrix row requires a finite RHS')
        blocks.append(matrix); rhs.append(values)
    a = vstack(blocks, format='csr'); a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    neq = blocks[0].shape[0]
    bounds = kwargs.get('bounds')
    lower, upper = split_variable_bounds(bounds, n, checked=True)
    if np.isnan(lower).any() or np.isnan(upper).any() or (lower > upper).any():
        raise ValueError('Invalid variable bounds')
    return a, np.concatenate(rhs), lower, upper, c.copy(), neq


def _certificate(a, rhs, lower, upper, c, neq, solution):
    """Original unscaled LP checks, matching the GPU's acceptance thresholds."""
    x = np.asarray(solution.col_value, dtype=float)
    y = np.asarray(solution.row_dual, dtype=float)
    if x.shape != c.shape or y.shape != rhs.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return dict(primal_residual=np.inf, dual_violation=np.inf, relative_kkt_gap=np.inf,
                    stationarity_residual=np.inf, certificate_passed=False,
                    bound_complementarity_gap=np.inf, row_complementarity_gap=np.inf,
                    absolute_kkt_gap=np.inf)
    activity = a @ x; reduced = c - a.T @ y
    row_error = activity - rhs
    primal = max(0., np.max(np.abs(row_error[:neq]), initial=0.),
                 np.max(row_error[neq:], initial=0.), np.max(lower-x, initial=0.),
                 np.max(x-upper, initial=0.))
    dual = max(0., np.max(y[neq:], initial=0.),
               np.max(np.where(~np.isfinite(lower), reduced, 0.), initial=0.),
               np.max(np.where(~np.isfinite(upper), -reduced, 0.), initial=0.))
    target = np.where(reduced >= 0., lower, upper)
    bound_gap = np.abs(reduced*(x-np.where(np.isfinite(target), target, x))).sum()
    row_gap = np.abs(y[neq:]*(rhs-activity)[neq:]).sum()
    gap = bound_gap + row_gap
    gap /= max(1., abs(float(c @ x)))
    stationarity = float(np.max(np.abs(np.asarray(solution.col_dual)-reduced), initial=0.))
    passed = bool(solution.value_valid and solution.dual_valid and np.isfinite(gap)
                  and primal <= 1e-5 and dual <= 1e-7 and gap <= 1e-7)
    return dict(primal_residual=float(primal), dual_violation=float(dual),
                relative_kkt_gap=float(gap), stationarity_residual=stationarity,
                certificate_passed=passed, bound_complementarity_gap=float(bound_gap),
                row_complementarity_gap=float(row_gap), absolute_kkt_gap=float(bound_gap+row_gap))


class RepeatedCpuLP:
    """``solve_batch(requests)`` comparator with persistent per-environment LPs.

    Set ``n_fluxes`` for the community solver's explicit stage identity. Other
    callers may supply a ``_stage`` string in each request's keyword arguments;
    otherwise matrices with equal shape/equality count share an environment's
    model, safely reoptimizing even when its objective changes. The optional
    ``environment_ids`` supports subset routing without conflating environments.
    A fresh/rebuilt model may receive ``_initial_basis`` with integer
    ``col_status`` and ``row_status`` arrays. This is only a simplex proposal;
    the current LP is still solved and certified before returning any flux.
    An opt-in ``_basis_override`` may replace a valid hot basis, but only for
    the explicitly identified previous certified-GPU handoff policy. It never
    supplies a solution or bypasses the same original-LP certificate.
    """
    name = 'cpu_persistent_highs_simplex'

    def __init__(self, workers=4, *, reuse_basis=True, n_fluxes=None,
                 max_numerical_retries=2, exchange_support_updates=False):
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('A positive CPU worker count is required')
        if not isinstance(max_numerical_retries, int) or not 0 <= max_numerical_retries <= 2:
            raise ValueError('At most two bounded numerical retries are supported')
        import highspy
        self.hp = highspy; self.reuse_basis = bool(reuse_basis); self.n_fluxes = n_fluxes
        # This remains deliberately opt-in.  Only the final exchange-stage
        # performance-floor row may change sparse support; every other
        # structural change retains the conservative rebuild policy.
        self.exchange_support_updates = bool(exchange_support_updates)
        self.max_numerical_retries = max_numerical_retries
        self._lanes = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'highs-owner-{i}')
                       for i in range(workers)]
        self._lane_thread_ids = [None]*workers
        self._lane_handles = [set() for _ in range(workers)]
        self._environment_lanes = {}
        self._routing_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        # Existing speculative callers submit orchestration jobs here. _solve
        # routes their native work to the same stable owner lane. Normal batch
        # calls submit straight to the lanes, so these threads stay unstarted.
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='highs-dispatch')
        self.models = _OwnedModelRegistry(self); self.history = []; self.closed = False

    def _lane_for(self, environment_id):
        with self._routing_lock:
            if environment_id not in self._environment_lanes:
                self._environment_lanes[environment_id] = len(self._environment_lanes)%len(self._lanes)
            return self._environment_lanes[environment_id]

    def _on_lane(self, lane, function, *args):
        identity = threading.get_ident()
        previous = self._lane_thread_ids[lane]
        if previous is not None and previous != identity:
            raise RuntimeError('Persistent HiGHS owner thread changed')
        self._lane_thread_ids[lane] = identity
        return function(*args)

    def _call_lane(self, lane, function, *args):
        if threading.get_ident() == self._lane_thread_ids[lane]:
            return function(*args)
        if self.closed:
            raise RuntimeError('The repeated CPU backend is closed')
        return self._lanes[lane].submit(self._on_lane, lane, function, *args).result()

    def _release_handle(self, handle):
        if handle is None:
            return
        def release():
            try:
                handle._dispose_owner()
            finally:
                self._lane_handles[handle._lane].discard(handle)
        self._call_lane(handle._lane, release)

    def _replace_entry_solver(self, entry, solver):
        self._release_handle(entry.get('_owner_handle'))
        entry.update(solver=solver, _owner_handle=solver)

    def clear_models(self):
        """Discard each native model on its owner, retaining stable lane IDs."""
        with self._lifecycle_lock:
            if self.closed:
                dict.clear(self.models)
                return
            failures = []
            def clear_lane(lane):
                errors = []
                for handle in tuple(self._lane_handles[lane]):
                    try:
                        handle._dispose_owner()
                    except Exception as error:
                        errors.append(str(error))
                self._lane_handles[lane].clear()
                return errors
            for lane in range(len(self._lanes)):
                if self._lane_thread_ids[lane] is not None:
                    failures.extend(self._call_lane(lane, clear_lane, lane))
            dict.clear(self.models)
            if failures:
                raise RuntimeError('Owner-thread cleanup failed: '+'; '.join(failures))

    def _check(self, status, operation):
        if status == self.hp.HighsStatus.kError:
            raise RuntimeError('HiGHS rejected ' + operation)

    @staticmethod
    def _options(kwargs):
        options = dict(kwargs.get('options') or {})
        options.pop('threads', None); options.pop('parallel', None)
        if 'disp' in options: options['output_flag'] = options.pop('disp')
        if 'maxiter' in options: options['simplex_iteration_limit'] = options.pop('maxiter')
        if isinstance(options.get('presolve'), (bool, np.bool_)):
            options['presolve'] = 'on' if options['presolve'] else 'off'
        options.update(threads=1, parallel='off', solver='simplex', simplex_strategy=1,
                       output_flag=False)
        return options

    def _new_solver(self, a, rhs, lower, upper, c, neq, options):
        try:
            lane = self._lane_thread_ids.index(threading.get_ident())
        except ValueError as error:
            raise RuntimeError('Native HiGHS creation requires an owner thread') from error
        solver = _OwnerSolverHandle(self, lane, self.hp.Highs())
        self._lane_handles[lane].add(solver)
        try:
            for name, value in options.items(): self._check(solver.setOptionValue(name, value), 'option ' + name)
            lp = self.hp.HighsLp(); lp.num_col_, lp.num_row_ = a.shape[1], a.shape[0]
            lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lower, upper
            lp.row_lower_, lp.row_upper_ = np.r_[rhs[:neq], np.full(a.shape[0]-neq, -np.inf)], rhs
            lp.a_matrix_.format_ = self.hp.MatrixFormat.kRowwise
            lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = a.indptr, a.indices, a.data
            self._check(solver.passModel(lp), 'model creation')
            return solver
        except BaseException:
            self._release_handle(solver)
            raise

    @staticmethod
    def _numerical_options(solver):
        options = solver.getOptions()
        return {name:getattr(options, name) for name in _NUMERICAL_OPTIONS}

    def _attempt(self, solver, problem, label, setup_seconds):
        """Record every actual solver call, including non-optimal/error results."""
        started = time.perf_counter()
        try:
            run_status = solver.run()
        except RuntimeError as error:
            elapsed = time.perf_counter()-started
            solution = SimpleNamespace(col_value=[], row_dual=[], value_valid=False, dual_valid=False)
            certificate = _certificate(*problem, solution)
            record = dict(kind=label, solver_run=True, run_status='exception',
                model_status='HighsModelStatus.kSolveError', optimal=False,
                value_valid=False, dual_valid=False, simplex_iterations=0,
                setup_seconds=setup_seconds, solve_seconds=elapsed, certificate_seconds=0.,
                total_seconds=setup_seconds+elapsed, error=str(error),
                effective_options=self._numerical_options(solver), **certificate)
            return (self.hp.HighsStatus.kError, self.hp.HighsModelStatus.kSolveError,
                    solution, None, certificate, record)
        solve_seconds = time.perf_counter()-started
        status, solution, info = solver.getModelStatus(), solver.getSolution(), solver.getInfo()
        before = time.perf_counter(); certificate = _certificate(*problem, solution)
        certificate_seconds = time.perf_counter()-before
        record = dict(kind=label, solver_run=True, run_status=str(run_status), model_status=str(status),
            optimal=status == self.hp.HighsModelStatus.kOptimal,
            value_valid=bool(solution.value_valid), dual_valid=bool(solution.dual_valid),
            simplex_iterations=int(info.simplex_iteration_count), setup_seconds=setup_seconds,
            solve_seconds=solve_seconds, certificate_seconds=certificate_seconds,
            total_seconds=setup_seconds+time.perf_counter()-started,
            effective_options=self._numerical_options(solver),
            highs_info={name:float(getattr(info, name)) for name in _QUALITY_FIELDS}, **certificate)
        return run_status, status, solution, info, certificate, record

    def _initial_basis(self, proposal, columns, rows):
        if not isinstance(proposal, dict) or set(proposal) != {'col_status', 'row_status'}:
            raise ValueError('Initial basis requires col_status and row_status')
        vectors = []
        for name, size in (('col_status', columns), ('row_status', rows)):
            values = np.asarray(proposal[name])
            if values.shape != (size,) or values.dtype.kind not in 'iu':
                raise ValueError('Initial basis ' + name + ' must be an integer vector of the correct length')
            if ((values < 0) | (values > 4)).any():
                raise ValueError('Initial basis statuses must be HiGHS enum values 0 through 4')
            vectors.append(values)
        basic = int(self.hp.HighsBasisStatus.kBasic)
        if sum(int(np.count_nonzero(values == basic)) for values in vectors) != rows:
            raise ValueError('Initial basis must contain exactly one basic variable per LP row')
        basis = self.hp.HighsBasis()
        basis.col_status = [self.hp.HighsBasisStatus(int(value)) for value in vectors[0]]
        basis.row_status = [self.hp.HighsBasisStatus(int(value)) for value in vectors[1]]
        basis.valid = True
        return basis

    def _validated_basis_override(self, override, columns, rows):
        """Validate an opt-in basis handoff without changing the solver."""
        if not isinstance(override, dict) or set(override) != {'basis', 'source', 'age'}:
            raise ValueError('Basis override requires basis, source and age')
        source = override['source']
        if source != 'previous_certified_gpu':
            raise ValueError('Basis override source must be previous_certified_gpu')
        age = override['age']
        if isinstance(age, (bool, np.bool_)) or not isinstance(age, (int, np.integer)) or age < 1:
            raise ValueError('Basis override age must be a positive integer')
        return self._initial_basis(override['basis'], columns, rows), source, int(age)

    def _exchange_performance_row(self, a, neq, rhs=None):
        """Return the audited stage-three performance row, or fail closed.

        The community exchange LP ends with two transfer rows per auxiliary
        absolute-value variable.  Each pair contains its own auxiliary column
        with coefficient -1 and, unless its physical coefficient is exactly
        zero, one flux column.  The optional aggregate-performance floor is
        the single row immediately before that immutable suffix.  Requiring
        this full signature prevents a generic CSR change from acquiring the
        support-update exception.
        """
        if not self.exchange_support_updates or self.n_fluxes is None:
            return None
        if (isinstance(self.n_fluxes, (bool, np.bool_))
                or not isinstance(self.n_fluxes, (int, np.integer))):
            return None
        n_fluxes = int(self.n_fluxes)
        rows, columns = a.shape
        n_aux = columns - n_fluxes - 1
        if n_fluxes < 1 or n_aux < 1 or rows < neq + 2*n_aux + 1:
            return None
        performance_row = rows - 2*n_aux - 1
        if performance_row < neq:
            return None
        if rhs is not None:
            rhs = np.asarray(rhs)
            if (rhs.shape != (rows,) or not rhs[performance_row] < 0.
                    or np.any(rhs[performance_row + 1:] != 0.)):
                return None
        start, end = a.indptr[performance_row:performance_row + 2]
        performance_columns = a.indices[start:end]
        if not len(performance_columns) or np.any(performance_columns >= n_fluxes):
            return None
        transfer_start = performance_row + 1
        for auxiliary in range(n_aux):
            expected_auxiliary = n_fluxes + 1 + auxiliary
            pair_flux_columns = []
            for row in (transfer_start + 2*auxiliary,
                        transfer_start + 2*auxiliary + 1):
                start, end = a.indptr[row:row + 2]
                row_columns = a.indices[start:end]
                row_values = a.data[start:end]
                auxiliary_positions = np.flatnonzero(
                    row_columns == expected_auxiliary
                )
                if (len(row_columns) not in (1, 2)
                        or len(auxiliary_positions) != 1
                        or row_values[auxiliary_positions[0]] != -1.):
                    return None
                flux_columns = row_columns[row_columns != expected_auxiliary]
                if len(flux_columns) > 1 or np.any(flux_columns >= n_fluxes):
                    return None
                pair_flux_columns.append(tuple(int(value) for value in flux_columns))
            if pair_flux_columns[0] != pair_flux_columns[1]:
                return None
        return performance_row

    def _exchange_support_update_row(self, a, entry, stage, neq, rhs=None):
        """Identify the sole support-changing row accepted for hot update."""
        if stage != 'exchange' or entry is None or 'a' not in entry:
            return None
        previous = entry['a']
        if previous.shape != a.shape:
            return None
        previous_row = self._exchange_performance_row(
            previous, neq, entry.get('rhs')
        )
        current_row = self._exchange_performance_row(a, neq, rhs)
        if previous_row is None or current_row != previous_row:
            return None
        row = current_row
        previous_counts = np.diff(previous.indptr)
        current_counts = np.diff(a.indptr)
        if (not np.array_equal(previous_counts[:row], current_counts[:row])
                or not np.array_equal(previous_counts[row + 1:],
                                       current_counts[row + 1:])):
            return None
        previous_start, previous_end = previous.indptr[row:row + 2]
        current_start, current_end = a.indptr[row:row + 2]
        if (not np.array_equal(previous.indices[:previous_start],
                               a.indices[:current_start])
                or not np.array_equal(previous.indices[previous_end:],
                                       a.indices[current_end:])):
            return None
        if np.array_equal(previous.indices[previous_start:previous_end],
                          a.indices[current_start:current_end]):
            return None
        return row

    def can_reuse_structure(self, a, entry, stage, neq):
        """Return whether a hot model can represent ``a`` exactly.

        This public predicate lets proposal routers avoid paying dictionary
        lookup cost when the same safe support-update path will retain an
        existing CPU basis.  The solve path repeats the check with the current
        RHS, so this preflight predicate can never authorize a solve itself.
        """
        if entry is None or 'a' not in entry or entry['a'].shape != a.shape:
            return False
        if (np.array_equal(a.indptr, entry['a'].indptr)
                and np.array_equal(a.indices, entry['a'].indices)):
            return True
        return self._exchange_support_update_row(
            a, entry, stage, neq
        ) is not None

    @staticmethod
    def _coefficient_updates(a, previous, support_update_row):
        """Return exact (row, column, new value) coefficient mutations."""
        if support_update_row is None:
            changed = np.flatnonzero(a.data != previous.data)
            rows = np.searchsorted(a.indptr, changed, side='right') - 1
            return rows, a.indices[changed], a.data[changed]
        # Sparse subtraction produces the union of changed coordinates.  The
        # fancy lookup obtains the requested LP's value, including an explicit
        # zero for a coefficient that disappeared from its CSR support.
        difference = (a - previous).tocsr()
        difference.sum_duplicates(); difference.eliminate_zeros(); difference.sort_indices()
        changed = difference.tocoo(copy=False)
        values = np.asarray(a[changed.row, changed.col]).ravel()
        return changed.row, changed.col, values

    def _solve(self, job):
        """Legacy single-request entry point, including external worker pools."""
        if self.closed:
            raise RuntimeError('The repeated CPU backend is closed')
        lane = self._lane_for(job[0])
        return self._call_lane(lane, self._solve_owner, job)

    def _solve_owner(self, job):
        environment_id, request = job; objective, kwargs = request
        started = time.perf_counter()
        a, rhs, lower, upper, c, neq = _problem(objective, kwargs)
        m, n = a.shape
        if '_stage' in kwargs:
            stage = kwargs['_stage']
        elif self.n_fluxes is not None:
            from .gpu_compiled_community_backend import stage_key
            stage = stage_key(c, a, neq, self.n_fluxes)[0]
        else:
            stage = 'unspecified'
        key = (environment_id, stage, m, n, neq)
        entry = self.models.get(key)
        exact_structure = (entry is not None
            and np.array_equal(a.indptr, entry['a'].indptr)
            and np.array_equal(a.indices, entry['a'].indices))
        support_update_row = (None if exact_structure else
            self._exchange_support_update_row(a, entry, stage, neq, rhs))
        same_structure = exact_structure or support_update_row is not None
        rebuilt = not same_structure; changed_coefficients = 0; basis_reused = False
        exchange_support_updated = False
        exchange_support_update_recovery = None
        exchange_support_update_error = None
        initial_basis_used = False
        override = kwargs.get('_basis_override')
        basis_override_requested = override is not None
        basis_override_used = False
        basis_override_rejected = False
        basis_override_skipped_reason = None
        basis_override_rejection_reason = None
        basis_override_recovery = None
        basis_override_error = None
        basis_override_source = override.get('source') if isinstance(override, dict) else None
        basis_override_age = override.get('age') if isinstance(override, dict) else None
        options = self._options(kwargs)
        if rebuilt:
            if basis_override_requested:
                # An override represents continuity from an already existing
                # CPU model. It must never acquire the distinct cold-start
                # semantics of ``_initial_basis``.
                basis_override_skipped_reason = 'model_rebuilt'
            proposal = kwargs.get('_initial_basis')
            initial_basis = self._initial_basis(proposal, n, m) if proposal is not None else None
            solver = self._new_solver(a, rhs, lower, upper, c, neq, options)
            if initial_basis is not None:
                self._check(solver.setBasis(initial_basis), 'initial basis proposal')
                initial_basis_used = True
            entry = dict(solver=solver, _owner_handle=solver, options=options, basis=None)
            self.models[key] = entry
        else:
            solver = entry['solver']
            support_update_failure = None
            try:
                for name, value in options.items():
                    if entry['options'].get(name) != value: self._check(solver.setOptionValue(name, value), 'option ' + name)
                rows, columns, values = self._coefficient_updates(
                    a, entry['a'], support_update_row
                )
                changed_coefficients = len(rows)
                for row, column, value in zip(rows, columns, values):
                    self._check(solver.changeCoeff(int(row), int(column), float(value)), 'coefficient update')
                for old_name, current, method in (
                    ('c', c, lambda ids: solver.changeColsCost(len(ids), ids, c[ids])),
                    ('rhs', rhs, lambda ids: solver.changeRowsBounds(len(ids), ids, np.where(ids < neq, rhs[ids], -np.inf), rhs[ids])),
                ):
                    ids = np.flatnonzero(current != entry[old_name]).astype(np.int32)
                    if len(ids): self._check(method(ids), old_name + ' update')
                ids = np.flatnonzero((lower != entry['lower']) | (upper != entry['upper'])).astype(np.int32)
                if len(ids): self._check(solver.changeColsBounds(len(ids), ids, lower[ids], upper[ids]), 'bound update')
            except RuntimeError as error:
                if support_update_row is None:
                    raise
                support_update_failure = error
            if support_update_failure is None:
                saved_basis = entry['basis']
                saved_basis_valid = saved_basis is not None and saved_basis.valid
                override_cold_recovery = False
                if basis_override_requested and not self.reuse_basis:
                    basis_override_skipped_reason = 'basis_reuse_disabled'
                elif basis_override_requested and not saved_basis_valid:
                    basis_override_skipped_reason = 'no_valid_existing_basis'
                elif basis_override_requested:
                    # Capture both representations before attempting the handoff.
                    # A rejected HiGHS setter is not assumed to be side-effect
                    # free, so an old basis is explicitly restored before solve.
                    live_basis = solver.getBasis()
                    try:
                        proposed_basis, basis_override_source, basis_override_age = (
                            self._validated_basis_override(override, n, m)
                        )
                    except (TypeError, ValueError) as error:
                        basis_override_rejected = True
                        basis_override_rejection_reason = 'validation_failed'
                        basis_override_recovery = 'previous_basis_unchanged'
                        basis_override_error = str(error)
                    else:
                        try:
                            self._check(solver.setBasis(proposed_basis), 'basis override')
                            if not solver.getBasis().valid:
                                raise RuntimeError('HiGHS did not retain a valid basis override')
                        except Exception as error:
                            basis_override_rejected = True
                            basis_override_rejection_reason = 'setter_failed'
                            basis_override_error = str(error)
                            recovery_candidates = []
                            if live_basis.valid:
                                recovery_candidates.append(('restored_live_basis', live_basis))
                            if saved_basis.valid:
                                recovery_candidates.append(('restored_saved_basis', saved_basis))
                            for recovery_name, recovery_basis in recovery_candidates:
                                try:
                                    self._check(solver.setBasis(recovery_basis), 'basis override recovery')
                                    if not solver.getBasis().valid:
                                        raise RuntimeError('HiGHS did not retain the recovered basis')
                                except Exception:
                                    continue
                                basis_override_recovery = recovery_name
                                basis_reused = True
                                break
                            if basis_override_recovery is None:
                                # clearSolver is the same exact cold-simplex path
                                # used by reuse_basis=False. If it also fails,
                                # _check raises and no result can be certified.
                                self._check(solver.clearSolver(), 'basis override cold recovery')
                                basis_override_recovery = 'cleared_solver'
                                override_cold_recovery = True
                        else:
                            basis_override_used = True
                            basis_override_recovery = 'not_needed'
                if (not basis_override_used and not override_cold_recovery
                        and self.reuse_basis and saved_basis_valid and not basis_reused):
                    # Coefficient changes invalidate a factorization, not the
                    # saved combinatorial basis proposal. HiGHS rechecks it.
                    # Bounds, costs and RHS updates normally preserve the live
                    # factorization. Do not reset a still-valid native basis.
                    try:
                        if not solver.getBasis().valid:
                            self._check(solver.setBasis(saved_basis), 'basis reuse')
                            if support_update_row is not None and not solver.getBasis().valid:
                                raise RuntimeError('HiGHS did not retain the reused basis')
                    except RuntimeError as error:
                        if support_update_row is None:
                            raise
                        support_update_failure = error
                    else:
                        basis_reused = True
                        if basis_override_rejected and basis_override_recovery == 'previous_basis_unchanged':
                            basis_override_recovery = 'previous_basis_reused'
                elif not self.reuse_basis and not override_cold_recovery:
                    self._check(solver.clearSolver(), 'cold simplex state')
            if support_update_failure is not None:
                # changeCoeff and setBasis error statuses are not assumed to be
                # side-effect free. Replace the possibly partial solver with a
                # fresh exact model; never solve or certify the damaged state.
                exchange_support_update_error = str(support_update_failure)
                try:
                    solver = self._new_solver(a, rhs, lower, upper, c, neq, options)
                except Exception:
                    self.models.pop(key, None)
                    raise
                entry = dict(solver=solver, _owner_handle=solver, options=options, basis=None)
                self.models[key] = entry
                rebuilt = True
                basis_reused = False
                exchange_support_update_recovery = 'model_rebuilt'
                if basis_override_requested and basis_override_skipped_reason is None:
                    basis_override_skipped_reason = 'model_rebuilt_after_support_update_failure'
                    basis_override_used = False
            else:
                exchange_support_updated = support_update_row is not None
        entry.update(a=a, rhs=rhs.copy(), lower=lower.copy(), upper=upper.copy(), c=c.copy())
        setup_seconds = time.perf_counter()-started
        problem = (a, rhs, lower, upper, c, neq)
        run_status, status, solution, info, certificate, attempt = self._attempt(solver, problem, 'initial', setup_seconds)
        attempts = [attempt]
        original_options = attempt['effective_options']
        options_touched = False
        # Only an alleged optimal LP that fails our unchanged, original-unit
        # certificate is eligible. Never turn infeasible/unbounded/limited
        # solves into success, and never change the mathematical LP.
        try:
            for retry in range(self.max_numerical_retries):
                if (run_status == self.hp.HighsStatus.kError
                        or status != self.hp.HighsModelStatus.kOptimal or certificate['certificate_passed']):
                    break
                before = time.perf_counter()
                tolerance = min(1e-9 if retry == 0 else 1e-10,
                    *(original_options[name] for name in _NUMERICAL_OPTIONS if name != 'presolve'))
                label = 'strict_basis_refactor' if retry == 0 else 'strict_cold_no_presolve'
                try:
                    if retry == 0:
                        basis = solver.getBasis()
                        options_touched = True
                        # Set every effective tolerance explicitly: a changed
                        # umbrella KKT option alone need not refresh existing
                        # simplex feasibility tolerances on a hot solver.
                        for name in _NUMERICAL_OPTIONS[:-1]:
                            self._check(solver.setOptionValue(name, tolerance), 'strict option ' + name)
                        # A plain run() could reuse the same updated numerical
                        # factors. Force a fresh factorization, retaining only
                        # the combinatorial basis as a reoptimization proposal.
                        self._check(solver.clearSolver(), 'numerical factor reset')
                        if basis.valid: self._check(solver.setBasis(basis), 'refactor basis proposal')
                    else:
                        retry_options = dict(options, **original_options)
                        retry_options.update({name:tolerance for name in _NUMERICAL_OPTIONS[:-1]})
                        retry_options['presolve'] = 'off'
                        solver = self._new_solver(*problem, retry_options)
                        self._replace_entry_solver(entry, solver)
                except RuntimeError as error:
                    attempts.append(dict(kind=label, solver_run=False, run_status='setup_error',
                        model_status='not_run', certificate_passed=False, error=str(error),
                        setup_seconds=time.perf_counter()-before, solve_seconds=0.,
                        total_seconds=time.perf_counter()-before, simplex_iterations=0))
                    continue
                run_status, status, solution, info, certificate, attempt = self._attempt(
                    solver, problem, label, time.perf_counter()-before)
                attempts.append(attempt)
        finally:
            if options_touched:
                # kkt_tolerance can override the other five tolerance fields.
                # Restore it first, then each original effective value. A
                # strict retry must not silently alter later timed LP solves.
                for name, value in original_options.items():
                    self._check(solver.setOptionValue(name, value), 'restore option ' + name)
        optimal = status == self.hp.HighsModelStatus.kOptimal
        success = bool(run_status != self.hp.HighsStatus.kError and optimal and certificate['certificate_passed'])
        entry['basis'] = solver.getBasis() if success else None
        if not success:
            self._check(solver.clearSolver(), 'discard uncertified solver state')
        solver_runs = sum(int(attempt['solver_run']) for attempt in attempts)
        simplex_iterations = sum(max(0, attempt['simplex_iterations']) for attempt in attempts)
        diagnostics = dict(environment_id=environment_id, stage=stage, success=success,
            native_owner_lane=self._environment_lanes[environment_id],
            native_owner_thread=threading.get_ident(),
            cpu_lp_calls=1, model_rebuilt=rebuilt, basis_reused=basis_reused,
            initial_basis_used=initial_basis_used,
            exchange_support_updated=exchange_support_updated,
            exchange_support_update_row=(int(support_update_row)
                if support_update_row is not None else None),
            exchange_support_update_recovery=exchange_support_update_recovery,
            exchange_support_update_error=exchange_support_update_error,
            basis_override_requested=basis_override_requested,
            basis_override_used=basis_override_used,
            basis_override_rejected=basis_override_rejected,
            basis_override_skipped_reason=basis_override_skipped_reason,
            basis_override_rejection_reason=basis_override_rejection_reason,
            basis_override_source=basis_override_source,
            basis_override_age=basis_override_age,
            basis_override_recovery=basis_override_recovery,
            basis_override_error=basis_override_error,
            changed_coefficients=changed_coefficients, setup_seconds=setup_seconds,
            solve_seconds=sum(attempt['solve_seconds'] for attempt in attempts),
            simplex_iterations=simplex_iterations, cpu_solver_runs=solver_runs,
            numerical_retry_count=len(attempts)-1,
            numerical_retry_seconds=sum(attempt['total_seconds'] for attempt in attempts[1:]),
            solver_attempts=attempts,
            model_status=str(status), run_status=str(run_status),
            effective_options_restored=self._numerical_options(solver) == original_options,
            **certificate, max_original_residual=certificate['primal_residual'],
            total_seconds=time.perf_counter()-started)
        x = np.asarray(solution.col_value, dtype=float).copy() if success else None
        return SimpleNamespace(success=success, x=x, fun=float(c @ x) if success else None,
            nit=simplex_iterations, diagnostics=diagnostics,
            solution_snapshot=_solution_snapshot(solution),
            message=solver.modelStatusToString(status) + ('' if certificate['certificate_passed'] else '; original LP certificate failed'))

    def solve_batch(self, requests, *, environment_ids=None):
        with self._lifecycle_lock:
            if self.closed: raise RuntimeError('The repeated CPU backend is closed')
            if environment_ids is None: environment_ids = list(range(len(requests)))
            if len(environment_ids) != len(requests) or len(set(environment_ids)) != len(environment_ids):
                raise ValueError('One unique stable environment ID is required per request')
            started = time.perf_counter()
            futures, results, failures = [], [], []
            try:
                for environment_id, request in zip(environment_ids, requests):
                    lane = self._lane_for(environment_id)
                    futures.append(self._lanes[lane].submit(self._on_lane, lane,
                        self._solve_owner, (environment_id, request)))
            except BaseException as error:
                failures.append(error)
            # Drain EVERY submitted owner before exposing an exception or
            # permitting reset/close. One failed row must not leave hidden work.
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as error:
                    failures.append(error)
                    # A KeyboardInterrupt while awaiting a result is not a
                    # native completion. Keep joining this accepted job.
                    while not future.done():
                        try:
                            future.result()
                        except BaseException as drain_error:
                            failures.append(drain_error)
            if failures:
                raise failures[0]
            self.history.append(dict(batch=len(requests), cpu_lp_calls=len(requests),
                cpu_solver_runs=sum(result.diagnostics['cpu_solver_runs'] for result in results),
                numerical_retry_count=sum(result.diagnostics['numerical_retry_count'] for result in results),
                seconds=time.perf_counter()-started, rows=[result.diagnostics for result in results]))
            return results

    def close(self):
        with self._lifecycle_lock:
            if not self.closed:
                # Stop optional legacy orchestrators first, leaving native
                # owner lanes alive until every model has been released there.
                self.pool.shutdown(wait=True)
                try:
                    self.clear_models()
                finally:
                    for lane in self._lanes:
                        lane.shutdown(wait=True)
                    self.closed = True
