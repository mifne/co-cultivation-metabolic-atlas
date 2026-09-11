"""Experimental independent-LP aggregation for cuOpt GPU iterations.

This is an exact block-diagonal formulation, not a shared nutrient pool.
Each original LP must pass its own primal/dual/complementarity certificate;
the aggregate objective and cuOpt termination status are insufficient.
Host assembly, cuOpt presolve, and returned host arrays remain explicit.
No CPU optimization, crossover, regularization, or reference warm start.
"""
import time
from types import SimpleNamespace

import numpy as np
from .csr_block_assembly import block_diag


def assemble_blocks(problems):
    if not problems:
        raise ValueError('At least one independent LP is required')
    shape, neq = problems[0][0].shape, problems[0][-1]
    if any(p[0].shape != shape or p[-1] != neq for p in problems):
        raise ValueError('A batch requires identical dimensions and equality counts')
    a = block_diag([p[0] for p in problems], format='csr')
    rhs, lower, upper, c = [np.concatenate([p[index] for p in problems])
                           for index in (1, 2, 3, 4)]
    row_lower = np.concatenate([np.r_[p[1][:neq],
                       np.full(shape[0]-neq, -np.inf)] for p in problems])
    if max(a.shape + (a.nnz,)) >= np.iinfo(np.int32).max:
        raise ValueError('cuOpt CSR int32 capacity exceeded')
    return a, row_lower, rhs, lower, upper, c


def box_face_feasibility(problem):
    """Try the box optimum face, without declaring the original LP infeasible.

    Only one objective variable is currently supported. If this face is
    feasible, y=0 certifies the original objective. If it is not feasible,
    the caller must reject this proposal and solve the unchanged original LP.
    """
    a, rhs, lower, upper, c, neq = problem
    positions = np.flatnonzero(c)
    if len(positions) != 1:
        raise ValueError('Box-face feasibility requires exactly one objective variable')
    index = int(positions[0])
    target = lower[index] if c[index] > 0 else upper[index]
    if not np.isfinite(target):
        raise ValueError('The objective variable needs a finite optimizing bound')
    lo, hi = lower.copy(), upper.copy()
    lo[index] = hi[index] = target
    return a, rhs, lo, hi, np.zeros_like(c), neq


def certify_blocks_device(problems, assembled, x, y, *, cp, allow_box_dual=False,
                          require_direct_dual=False):
    """Evaluate the unchanged original-LP gates independently on the GPU."""
    from cupyx.scipy.sparse import csr_matrix
    batch = len(problems); m, n = problems[0][0].shape; neq = problems[0][-1]
    if np.shape(x) != (batch*n,) or np.shape(y) != (batch*m,):
        raise ValueError('cuOpt solution dimensions do not match original blocks')
    a, _, rhs, lower, upper, c = assembled
    a = csr_matrix(a)
    x, y = cp.asarray(x, dtype=cp.float64), cp.asarray(y, dtype=cp.float64)
    lower, upper, c, rhs = [cp.asarray(v) for v in (lower, upper, c, rhs)]
    activity = a @ x; reduced = c - a.T @ y
    x, lower, upper, c, reduced = [v.reshape(batch, n) for v in (x, lower, upper, c, reduced)]
    y, rhs, activity = [v.reshape(batch, m) for v in (y, rhs, activity)]
    error = activity-rhs
    # Empty equality / inequality groups are legal in generic LP tests.
    zero = cp.zeros(batch, dtype=cp.float64)
    row_eq = cp.max(cp.abs(error[:, :neq]), axis=1) if neq else zero
    row_ub = cp.max(error[:, neq:], axis=1) if neq < m else zero
    primal = cp.maximum(zero, cp.maximum(cp.maximum(row_eq, row_ub),
                    cp.maximum(cp.max(lower-x, axis=1), cp.max(x-upper, axis=1))))
    dual_rows = cp.max(y[:, neq:], axis=1) if neq < m else zero
    dual = cp.maximum(zero, cp.maximum(dual_rows, cp.maximum(
        cp.max(cp.where(~cp.isfinite(lower), reduced, 0.), axis=1),
        cp.max(cp.where(~cp.isfinite(upper), -reduced, 0.), axis=1))))
    target = cp.where(reduced >= 0., lower, upper)
    bound_gap = cp.sum(cp.abs(reduced*(x-cp.where(cp.isfinite(target), target, x))), axis=1)
    row_gap = cp.sum(cp.abs(y[:, neq:]*(rhs-activity)[:, neq:]), axis=1)
    objective = cp.sum(c*x, axis=1)
    gap = (bound_gap+row_gap)/cp.maximum(1., cp.abs(objective))
    finite_primal = cp.all(cp.isfinite(x), axis=1) & cp.all(cp.isfinite(activity), axis=1) & cp.isfinite(objective)
    finite = finite_primal & cp.all(cp.isfinite(y), axis=1) & cp.all(cp.isfinite(reduced), axis=1)
    accepted = finite & cp.isfinite(gap) & (primal <= 1e-5) & (dual <= 1e-7) & (gap <= 1e-7)
    original_gap = gap.copy()
    box_used = cp.zeros(batch, dtype=bool)
    if allow_box_dual:
        # A second mathematically valid certificate, NOT clipped multipliers.
        # y=0 implies reduced cost=c. If the feasible primal attains the box
        # objective bound, it is optimal without needing any nonzero row dual.
        # This is useful for maxmin when the configured growth cap is attained.
        box_target = cp.where(c >= 0., lower, upper)
        box_gap = cp.sum(cp.abs(c*(x-cp.where(cp.isfinite(box_target), box_target, x))), axis=1)
        box_gap /= cp.maximum(1., cp.abs(objective))
        box_dual = cp.maximum(zero, cp.maximum(
            cp.max(cp.where(~cp.isfinite(lower), c, 0.), axis=1),
            cp.max(cp.where(~cp.isfinite(upper), -c, 0.), axis=1)))
        box_ok = finite_primal & cp.isfinite(box_gap) & (
            primal <= 1e-5) & (box_dual <= 1e-7) & (box_gap <= 1e-7)
        box_used = ~accepted & box_ok
        dual = cp.where(box_used, box_dual, dual)
        gap = cp.where(box_used, box_gap, gap)
        accepted = accepted | box_ok
    direct_columns = None
    if require_direct_dual:
        # This is an ADDITIONAL gate, never a substitute for complementarity.
        # Equality residuals times large free multipliers can dominate the
        # true objective difference despite tiny unweighted row residuals.
        certificate_y=cp.where(box_used[:,None],0.,y)
        certificate_rc=cp.where(box_used[:,None],c,reduced)
        support_bound=cp.where(certificate_rc>0.,lower,
                              cp.where(certificate_rc<0.,upper,0.))
        dual_objective=cp.sum(certificate_y*rhs,axis=1)+cp.sum(certificate_rc*support_bound,axis=1)
        signed_direct_gap=objective-dual_objective
        relative_direct_gap=cp.abs(signed_direct_gap)/cp.maximum(1.,cp.abs(objective))
        equality_pairing=cp.sum(certificate_y[:,:neq]*error[:,:neq],axis=1)
        direct_ok=cp.isfinite(dual_objective)&cp.isfinite(relative_direct_gap)&(relative_direct_gap<=1e-7)
        accepted &= direct_ok
        direct_columns=cp.asnumpy(cp.stack((dual_objective,signed_direct_gap,
                                            relative_direct_gap,equality_pairing,direct_ok),axis=1))
    metrics = cp.asnumpy(cp.stack((primal, dual, gap, objective, accepted, box_used, original_gap), axis=1))
    result = [dict(primal_residual=float(p), dual_violation=float(d),
                 relative_kkt_gap=float(g), objective=float(o), certificate_passed=bool(ok),
                 certificate_source='analytic_box_bound' if box else 'solver_row_dual',
                 solver_dual_relative_kkt_gap=float(raw_gap))
            for p, d, g, o, ok, box, raw_gap in metrics]
    if direct_columns is not None:
        for row,(d,s,r,e,ok) in zip(result,direct_columns):
            row.update(direct_dual_objective=float(d),signed_direct_gap=float(s),
                       relative_direct_gap=float(r),equality_dual_residual_contribution=float(e),
                       direct_dual_gate_passed=bool(ok))
    return result


class GpuBlockLP:
    """Cold sparse block solve; no environment state or oracle targets accepted."""
    name = 'cuopt_block_diagonal_gpu_experimental'

    def __init__(self, *, method='pdlp', time_limit=10., tolerance=1e-9, presolve=2,
                 box_dual_certificate=False, box_face=False):
        if method not in {'pdlp', 'barrier'} or presolve not in (0, 2):
            raise ValueError('Only explicit GPU methods and presolve 0/2 are supported')
        if not np.isfinite(time_limit) or time_limit <= 0 or not 0 < tolerance <= (1e-5 if box_face else 1e-7):
            raise ValueError('Invalid time limit or tolerance')
        if box_face and not box_dual_certificate:
            raise ValueError('Box-face feasibility must use the original-LP analytic dual certificate')
        import cupy as cp
        from cuopt import linear_programming as lp
        from cuopt.linear_programming import SolverMethod, PDLPSolverMode
        self.cp, self.lp = cp, lp
        self.settings = lp.SolverSettings()
        self.method = method
        self.box_dual_certificate = bool(box_dual_certificate)
        self.box_face = bool(box_face)
        self.parameters = dict(method=getattr(SolverMethod, 'PDLP' if method == 'pdlp' else 'Barrier'),
            crossover=False, presolve=presolve, time_limit=float(time_limit),
            log_to_console=False, num_cpu_threads=1)
        if method == 'pdlp':
            self.parameters.update(pdlp_precision=1, pdlp_solver_mode=PDLPSolverMode.Stable3)
        else:
            self.parameters.update(augmented=1, barrier_iterative_refinement=1,
                                   cudss_deterministic=True)
        for key, value in self.parameters.items():
            self.settings.set_parameter(key, value)
        self.settings.set_optimality_tolerance(tolerance)
        self.settings.set_parameter('relative_primal_tolerance', 0.)
        self.history = []

    def solve_batch(self, requests):
        from .cpu_repeated_lp import _problem
        started = time.perf_counter()
        problems = [_problem(c, kwargs) for c, kwargs in requests]
        normalized = time.perf_counter()-started
        results = self._solve_problems(problems)
        self.history[-1]['input_normalization_seconds'] = normalized
        self.history[-1]['total_seconds'] = time.perf_counter()-started
        return results

    def _solve_problems(self, problems):
        """Internal path: only inputs validated by solve_batch may enter."""
        started = time.perf_counter()
        packed = assemble_blocks(problems)
        solve_packed = assemble_blocks([box_face_feasibility(p) for p in problems]) if self.box_face else packed
        a, row_lower, rhs, lower, upper, c = solve_packed
        model = self.lp.DataModel()
        model.set_csr_constraint_matrix(a.data, a.indices.astype(np.int32), a.indptr.astype(np.int32))
        model.set_constraint_lower_bounds(row_lower)
        model.set_constraint_upper_bounds(rhs)
        model.set_variable_lower_bounds(lower)
        model.set_variable_upper_bounds(upper)
        model.set_objective_coefficients(c)
        model.set_maximize(False)
        setup_seconds = time.perf_counter()-started
        before = time.perf_counter()
        solution = self.lp.Solve(model, self.settings)
        self.cp.cuda.runtime.deviceSynchronize()
        solve_seconds = time.perf_counter()-before
        status = solution.get_termination_status().name.lower()
        solved_by = solution.get_solved_by().name.lower()
        before = time.perf_counter()
        x = np.asarray(solution.get_primal_solution(), dtype=np.float64)
        y = np.asarray(solution.get_dual_solution(), dtype=np.float64)
        valid_shape = x.shape == c.shape and y.shape == rhs.shape
        rows = certify_blocks_device(problems, packed, x, y, cp=self.cp,
            allow_box_dual=self.box_dual_certificate) if valid_shape else [
            dict(certificate_passed=False, reason='missing_primal_or_dual') for _ in problems]
        # A stopped/error solve must not silently be accepted as optimal.
        for row in rows:
            row['success'] = bool(status == 'optimal' and solved_by == self.method and row['certificate_passed'])
        certificate_seconds = time.perf_counter()-before
        record = dict(batch=len(problems), method=self.method, status=status, solved_by=solved_by,
            rows=rows, all_accepted=all(r['success'] for r in rows), cpu_lp_calls=0,
            variables=a.shape[1], constraints=a.shape[0], nonzeros=a.nnz,
            setup_seconds=setup_seconds, solve_seconds=solve_seconds,
            certificate_seconds=certificate_seconds, total_seconds=time.perf_counter()-started,
            box_dual_certificate=self.box_dual_certificate,
            proposal_formulation='box_optimum_face_feasibility' if self.box_face else 'original_objective',
            cuopt_solve_seconds=float(solution.get_solve_time()), cuopt_stats=solution.get_lp_stats(),
            scope='GPU LP iterations; host block assembly/presolve and host-return API; not fully device resident')
        n = problems[0][0].shape[1]
        results = [SimpleNamespace(success=row['success'],
            x=x[i*n:(i+1)*n].copy() if row['success'] else None,
            fun=row.get('objective') if row['success'] else None,
            diagnostics=row, message=str(record['status'])) for i, row in enumerate(rows)]
        record['total_seconds'] = time.perf_counter()-started
        self.history.append(record)
        return results
