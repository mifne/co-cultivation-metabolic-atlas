"""Optional cuOpt GPU LP backend for the unchanged cooperative three-stage LP.

No CPU LP fallback, Concurrent mode or crossover. Matrix assembly and cuOpt
presolve may use the host; optimization and original-matrix residual checks
use the GPU. This is not a surrogate or an endpoint correction.
"""
from collections import OrderedDict
import hashlib
import time
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, vstack


class PresolvedCuOptBackend:
    """Host algebraic presolve/postsolve; GPU LP iterations, no CPU optimizer.

    No basis is passed to Highs.postsolve: supplying a basis would enable
    simplex cleanup. We additionally set all CPU iteration limits to zero.
    This is NOT a fully device-resident pipeline.
    """
    name = "highs_presolve_cuopt_gpu_lexicographic"

    def __init__(self, method="barrier", **kwargs):
        if kwargs.get("reaction_support") is not None:
            raise ValueError("presolve backend requires the full reaction space")
        if kwargs.get("quadratic_regularization", 0):
            raise ValueError("LP presolve cannot preserve a changed quadratic objective")
        import highspy
        self.highspy = highspy
        self.method = method
        if method == "tableau":
            from src.gpu_bounded_simplex import GpuBoundedSimplex
            self.engine = GpuBoundedSimplex(**kwargs)
            self.name = "highs_presolve_gpu_simplex_experimental"
        else:
            self.engine = (HybridCuOptLexicographicBackend(**kwargs) if method == "hybrid"
                           else CuOptLexicographicBackend(method=method, **kwargs))
        self.validation_engine = self.engine.barrier if method == "hybrid" else self.engine
        self.history = []

    def solve(self, c, *, A_eq=None, b_eq=None, A_ub=None, b_ub=None, bounds=None, **ignored):
        started = time.perf_counter()
        hp = self.highspy
        c = np.asarray(c, dtype=float)
        n = len(c)
        eq = csr_matrix((0,n)) if A_eq is None else csr_matrix(A_eq)
        ub = csr_matrix((0,n)) if A_ub is None else csr_matrix(A_ub)
        matrix = vstack((eq,ub), format="csr")
        rhs = np.r_[[] if b_eq is None else b_eq, [] if b_ub is None else b_ub]
        bounds = [(0, None)]*n if bounds is None else bounds
        upper = np.array([np.inf if hi is None else hi for _,hi in bounds])
        # SciPy bounds None denotes an unbounded side (default bounds are 0,+inf).
        lower = np.array([-np.inf if lo is None else lo for lo,_ in bounds])
        lp = hp.HighsLp()
        lp.num_col_, lp.num_row_ = n, matrix.shape[0]
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lower, upper
        lp.col_names_ = [f"v{i}" for i in range(n)]
        lp.row_names_ = [f"r{i}" for i in range(matrix.shape[0])]
        lp.row_lower_ = np.r_[rhs[:eq.shape[0]], np.full(ub.shape[0], -np.inf)]
        lp.row_upper_ = rhs
        lp.a_matrix_.format_ = hp.MatrixFormat.kRowwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
        host = hp.Highs()
        for key,value in dict(output_flag=False, threads=1, presolve="on", run_crossover="off",
                              simplex_iteration_limit=0, ipm_iteration_limit=0).items():
            if host.setOptionValue(key,value) == hp.HighsStatus.kError:
                raise RuntimeError(f"unsupported safety option {key}")
        if host.passModel(lp) == hp.HighsStatus.kError or host.presolve() == hp.HighsStatus.kError:
            raise RuntimeError("HiGHS algebraic presolve failed")
        reduced = host.getPresolvedLp()
        record = dict(host_presolve_status=host.getModelPresolveStatus().name,
            original_variables=n, original_rows=matrix.shape[0],
            presolved_variables=reduced.num_col_, presolved_rows=reduced.num_row_,
            host_presolve_seconds=time.perf_counter()-started, cpu_optimization_allowed=False)
        if reduced.num_col_ == 0:
            record.update(success=False, status="empty_presolve_not_gpu_solved")
            self.history.append(record)
            return SimpleNamespace(success=False, x=None, fun=None, message=str(record))
        a = reduced.a_matrix_
        constructor = csc_matrix if a.format_ == hp.MatrixFormat.kColwise else csr_matrix
        mat = constructor((a.value_,a.index_,a.start_), shape=(reduced.num_row_,reduced.num_col_)).tocsr()
        rl,ru = np.asarray(reduced.row_lower_),np.asarray(reduced.row_upper_)
        equal = np.flatnonzero(rl == ru)
        up = np.flatnonzero((rl != ru) & np.isfinite(ru))
        low = np.flatnonzero((rl != ru) & np.isfinite(rl))
        result = self.engine.solve(reduced.col_cost_, A_eq=mat[equal], b_eq=ru[equal],
            A_ub=vstack((mat[up],-mat[low]), format="csr"), b_ub=np.r_[ru[up],-rl[low]],
            bounds=list(zip(reduced.col_lower_,reduced.col_upper_)),
            column_ids=list(reduced.col_names_),
            inequality_ids=["upper:"+reduced.row_names_[i] for i in up]+["lower:"+reduced.row_names_[i] for i in low],
            stage_key="maxmin" if np.count_nonzero(c)==1 and c[-1]==-1 else "aggregate" if np.any(c<0) else "parsimony")
        record.update(gpu=self.engine.history[-1], success=False)
        values = None
        if result.success:
            solution = hp.HighsSolution()
            solution.col_value = result.x
            solution.value_valid = True
            # Crucial: primal-only overload, NO basis, NO host.run().
            poststatus = host.postsolve(solution)
            recovered = host.getSolution()
            info = host.getInfo()
            counts = {key:int(getattr(info,key)) for key in
                      ("simplex_iteration_count", "ipm_iteration_count", "crossover_iteration_count")}
            if any(value > 0 for value in counts.values()):
                raise RuntimeError("CPU optimization detected during postsolve")
            record.update(cpu_iteration_counts=counts, postsolve_status=poststatus.name)
            if poststatus != hp.HighsStatus.kError and recovered.value_valid:
                cp = self.validation_engine.cp
                values = np.asarray(recovered.col_value)
                x = cp.asarray(values)
                activity = self.validation_engine.gpu_csr(matrix) @ x
                residuals = cp.concatenate((cp.abs(activity[:eq.shape[0]]-cp.asarray(rhs[:eq.shape[0]])),
                    cp.maximum(activity[eq.shape[0]:]-cp.asarray(rhs[eq.shape[0]:]),0),
                    cp.maximum(cp.asarray(lower)-x,0),cp.maximum(x-cp.asarray(upper),0)))
                violation = float(residuals.max().get())
                record.update(max_original_residual=violation, objective=float(c @ values),
                    success=bool(cp.isfinite(x).all().get()) and violation <= self.validation_engine.residual_tolerance)
        record["total_seconds"] = time.perf_counter()-started
        self.history.append(record)
        return SimpleNamespace(success=record["success"], x=values if record["success"] else None,
                               fun=record.get("objective"), message=str(record))


class HybridCuOptLexicographicBackend:
    """Use GPU PDLP for max-min, GPU barrier for the remaining objectives."""
    name, method = "cuopt_gpu_lexicographic", "hybrid"

    def __init__(self, **kwargs):
        self.pdlp = CuOptLexicographicBackend(method="pdlp", **dict(kwargs, presolve=2,
            rank_reduce=False, objective_scale=1.0))
        self.barrier = CuOptLexicographicBackend(method="barrier", **kwargs)
        self.history = []
        if kwargs.get("reaction_support") is not None:
            self.name = "cuopt_gpu_reaction_support_lexicographic"

    def solve(self, c, **kwargs):
        c = np.asarray(c)
        max_min = np.count_nonzero(c) == 1 and c[-1] == -1.
        engine = self.pdlp if max_min else self.barrier
        result = engine.solve(c, **kwargs)
        self.history.append(engine.history[-1])
        return result


class CuOptLexicographicBackend:
    name = "cuopt_gpu_lexicographic"

    def __init__(self, method="barrier", tolerance=1e-9, residual_tolerance=1e-5,
                 time_limit=30.0, presolve=2, log_to_console=False,
                 augmented=1, objective_scale=1.0, rank_reduce=False, reaction_support=None,
                 quadratic_regularization=0., pdlp_mode="Stable3"):
        if method not in {"barrier", "pdlp"}:
            raise ValueError("GPU-only LP requires barrier or pdlp, never Concurrent/DualSimplex")
        if any(not np.isfinite(v) or v <= 0 for v in (tolerance, residual_tolerance, time_limit)):
            raise ValueError("tolerances and time limit must be finite and positive")
        from cuopt.linear_programming import SolverMethod
        from cuopt.linear_programming.problem import Problem, LinearExpression, MINIMIZE
        from cuopt.linear_programming.solver_settings import SolverSettings
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        self.cp, self.gpu_csr = cp, gpu_csr
        self.Problem, self.Expression, self.MINIMIZE = Problem, LinearExpression, MINIMIZE
        self.method, self.tolerance, self.residual_tolerance = method, tolerance, residual_tolerance
        if not np.isfinite(objective_scale) or objective_scale <= 0:
            raise ValueError("objective scale must be positive and finite")
        self.objective_scale = float(objective_scale)
        if quadratic_regularization < 0 or not np.isfinite(quadratic_regularization):
            raise ValueError("quadratic regularization must be nonnegative")
        if quadratic_regularization and method != "barrier":
            raise ValueError("regularized QP requires the GPU barrier method")
        self.quadratic_regularization = float(quadratic_regularization)
        if quadratic_regularization:
            self.name = "cuopt_gpu_regularized_lexicographic_qp"
        self.rank_reduce = bool(rank_reduce)
        self.reaction_support = None if reaction_support is None else np.asarray(reaction_support, dtype=bool)
        if self.reaction_support is not None:
            self.name = "cuopt_gpu_reaction_support_lexicographic"
        self.rank_cache = OrderedDict()
        self.settings = SolverSettings()
        self.parameters = dict(method=getattr(SolverMethod, "Barrier" if method == "barrier" else "PDLP"),
            crossover=False, presolve=int(presolve), time_limit=float(time_limit),
            log_to_console=bool(log_to_console), num_cpu_threads=1)
        if method == "barrier":
            # Avoid squaring the condition number in ADAT for degenerate GEMs.
            self.parameters.update(augmented=int(augmented), barrier_iterative_refinement=1,
                                   cudss_deterministic=True)
        else:
            from cuopt.linear_programming import PDLPSolverMode
            self.parameters.update(pdlp_precision=1, pdlp_solver_mode=getattr(PDLPSolverMode,pdlp_mode))
        # Fail on unsupported settings: never silently enable a CPU method.
        for key, value in self.parameters.items():
            self.settings.set_parameter(key, value)
        self.settings.set_optimality_tolerance(tolerance)
        # Large non-limiting water/proton RHS must not hide small nutrient errors.
        self.settings.set_parameter("relative_primal_tolerance", 0.0)
        self.cache = OrderedDict()
        self.history = []

    def _independent_rows(self, eq, lower, upper):
        """Cached algebraic preprocessing, not an LP solve or bound relaxation.

        Select an independent row basis after fixed columns are accounted for.
        All ORIGINAL equations are audited afterwards. A relaxed-row optimum
        feasible for the original rows is also an optimum of the original LP.
        """
        if not self.rank_reduce:
            return np.arange(eq.shape[0])
        from scipy.linalg import qr
        from scipy.sparse import bmat
        from scipy.sparse.csgraph import connected_components
        structural = np.asarray(eq.getnnz(axis=0)).ravel() > 0
        free = np.flatnonzero(structural & (lower != upper))
        key = hashlib.sha256(eq.data.tobytes()+eq.indices.tobytes()+eq.indptr.tobytes()+free.tobytes()).hexdigest()
        if key not in self.rank_cache:
            reduced = eq[:, free].copy()
            reduced.eliminate_zeros()
            graph = bmat([[None, reduced], [reduced.T, None]], format="csr")
            _, labels = connected_components(graph, directed=False)
            selected = []
            for group in np.unique(labels[:eq.shape[0]]):
                row_ids = np.flatnonzero(labels[:eq.shape[0]] == group)
                col_ids = np.flatnonzero(labels[eq.shape[0]:] == group)
                if not len(col_ids):
                    continue
                block = reduced[row_ids][:, col_ids].toarray()
                r, pivots = qr(block.T, mode="r", pivoting=True, check_finite=False)
                diagonal = np.abs(np.diag(r))
                tolerance = max(block.shape)*np.finfo(float).eps*diagonal.max(initial=0)
                rank = np.count_nonzero(diagonal > tolerance)
                selected.extend(row_ids[np.asarray(pivots[:rank], dtype=int)])
            self.rank_cache[key] = np.sort(np.asarray(selected, dtype=int))
            if len(self.rank_cache) > 8:
                self.rank_cache.popitem(last=False)
        self.rank_cache.move_to_end(key)
        return self.rank_cache[key]

    def _problem(self, matrix, rhs, neq, lower, upper, objective):
        signature = hashlib.sha256(matrix.indptr.tobytes()+matrix.indices.tobytes()).hexdigest()
        key = (matrix.shape, neq, signature)
        if key not in self.cache:
            problem = self.Problem("cooperative_gpu_stage")
            variables = [problem.addVariable(lb=float(lo), ub=float(hi), obj=float(c), name=f"v{i}")
                         for i, (lo, hi, c) in enumerate(zip(lower, upper, objective))]
            constraints = []
            for row in range(matrix.shape[0]):
                a, b = matrix.indptr[row:row+2]
                expression = self.Expression([variables[j] for j in matrix.indices[a:b]],
                                             matrix.data[a:b].tolist(), 0.0)
                constraints.append(problem.addConstraint(expression == float(rhs[row]) if row < neq
                                                         else expression <= float(rhs[row])))
            problem.updateObjective(sense=self.MINIMIZE)
            self.cache[key] = dict(problem=problem, variables=variables, constraints=constraints,
                data=matrix.data.copy(), rhs=rhs.copy(), lower=lower.copy(), upper=upper.copy(),
                objective=objective.copy())
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
            state = self.cache[key]
            problem, variables = state["problem"], state["variables"]
            for index in np.flatnonzero((lower != state["lower"]) | (upper != state["upper"])):
                variables[index].setLowerBound(float(lower[index]))
                variables[index].setUpperBound(float(upper[index]))
            changed = np.flatnonzero(objective != state["objective"])
            if len(changed):
                problem.updateObjective(coeffs=[(variables[i], float(objective[i])) for i in changed],
                                        sense=self.MINIMIZE)
            coefficient_rows = np.unique(np.searchsorted(matrix.indptr[1:],
                np.flatnonzero(matrix.data != state["data"]), side="right"))
            for row in np.union1d(coefficient_rows, np.flatnonzero(rhs != state["rhs"])):
                a, b = matrix.indptr[row:row+2]
                problem.updateConstraint(state["constraints"][row],
                    coeffs=[(variables[j], float(v)) for j, v in zip(matrix.indices[a:b], matrix.data[a:b])],
                    rhs=float(rhs[row]))
            state.update(data=matrix.data.copy(), rhs=rhs.copy(), lower=lower.copy(),
                         upper=upper.copy(), objective=objective.copy())
        state = self.cache[key]
        if self.quadratic_regularization:
            from cuopt.linear_programming.problem import QuadraticExpression
            state["problem"].setObjective(QuadraticExpression(
                qvars1=state["variables"], qvars2=state["variables"],
                qcoefficients=[self.quadratic_regularization*self.objective_scale]*len(objective),
                vars=state["variables"], coefficients=objective.tolist()), sense=self.MINIMIZE)
        state["problem"].update()
        return state

    def solve(self, c, *, A_eq=None, b_eq=None, A_ub=None, b_ub=None, bounds=None, **ignored_highs):
        started = time.perf_counter()
        c = np.asarray(c, dtype=np.float64)
        n = len(c)
        eq = csr_matrix((0, n)) if A_eq is None else csr_matrix(A_eq, dtype=np.float64)
        ub = csr_matrix((0, n)) if A_ub is None else csr_matrix(A_ub, dtype=np.float64)
        eq.eliminate_zeros(); ub.eliminate_zeros()
        matrix = vstack((eq, ub), format="csr")
        matrix.sum_duplicates(); matrix.sort_indices()
        rhs = np.r_[np.zeros(0) if b_eq is None else b_eq, np.zeros(0) if b_ub is None else b_ub].astype(np.float64)
        bounds = [(0, None)]*n if bounds is None else bounds
        lower = np.array([-np.inf if lo is None else lo for lo, _ in bounds], dtype=np.float64)
        upper = np.array([np.inf if hi is None else hi for _, hi in bounds], dtype=np.float64)
        if (len(rhs) != matrix.shape[0] or len(bounds) != n or not np.isfinite(matrix.data).all()
                or not np.isfinite(c).all() or not np.isfinite(rhs).all()
                or np.isnan(lower).any() or np.isnan(upper).any() or (lower > upper).any()):
            raise ValueError("invalid original LP data")
        solve_lower, solve_upper = lower.copy(), upper.copy()
        if self.reaction_support is not None:
            if self.reaction_support.ndim != 1 or len(self.reaction_support) > n:
                raise ValueError("invalid reaction support shape")
            omitted = np.flatnonzero(~self.reaction_support)
            omitted = omitted[(lower[omitted] <= 0) & (upper[omitted] >= 0)]
            solve_lower[omitted] = 0.; solve_upper[omitted] = 0.
        else:
            omitted = np.empty(0, dtype=int)
        independent = self._independent_rows(eq, solve_lower, solve_upper)
        # Remove only interval-provably redundant inequalities. In particular,
        # the artificial 1e6 water/proton supplies must not dominate scaling.
        contributions = ub.data*np.where(ub.data > 0, upper[ub.indices], lower[ub.indices])
        row_maximum = np.asarray(csr_matrix((contributions, ub.indices, ub.indptr),
                                           shape=ub.shape).sum(axis=1)).ravel()
        kept_ub = np.flatnonzero(row_maximum > rhs[eq.shape[0]:])
        solve_matrix = vstack((eq[independent], ub[kept_ub]), format="csr")
        solve_rhs = np.r_[rhs[independent], rhs[eq.shape[0]+kept_ub]]
        state = self._problem(solve_matrix, solve_rhs, len(independent), solve_lower, solve_upper, c*self.objective_scale)
        problem = state["problem"]
        build_seconds = time.perf_counter()-started
        solution = problem.solve(self.settings)
        status = getattr(problem.Status, "name", str(problem.Status)).lower()
        solved_by = solution.get_solved_by().name.lower()
        record = dict(method=self.method, status=status, build_seconds=build_seconds,
                      solver_seconds=float(problem.SolveTime), variables=n, rows=matrix.shape[0],
                      solved_by=solved_by, solver_stats=solution.get_lp_stats(),
                      objective_scale=self.objective_scale, independent_equality_rows=len(independent),
                      original_equality_rows=eq.shape[0], restricted_reactions=len(omitted),
                      redundant_inequalities=ub.shape[0]-len(kept_ub),
                      effective_parameters={k:int(v) if hasattr(v, "name") else v for k,v in self.parameters.items()},
                      optimality_tolerance=self.tolerance)
        record["quadratic_regularization"] = self.quadratic_regularization
        # No primal-feasible/time-limit result is accepted as an LP optimum.
        success = status == "optimal" and solved_by == self.method
        values = np.asarray([v.Value for v in state["variables"]], dtype=np.float64) if success else None
        if success:
            cp = self.cp
            x = cp.asarray(values)
            activity = self.gpu_csr(matrix) @ x
            row_error = cp.concatenate((cp.abs(activity[:eq.shape[0]]-cp.asarray(rhs[:eq.shape[0]])),
                cp.maximum(activity[eq.shape[0]:]-cp.asarray(rhs[eq.shape[0]:]), 0)))
            bound_error = cp.maximum(cp.maximum(cp.asarray(lower)-x, x-cp.asarray(upper)), 0)
            row_residual = float(row_error.max().get()) if row_error.size else 0.0
            bound_residual = float(bound_error.max().get()) if bound_error.size else 0.0
            finite = bool(cp.isfinite(x).all().get())
            success = finite and max(row_residual, bound_residual) <= self.residual_tolerance
            record.update(max_original_row_residual=row_residual, max_bound_violation=bound_residual,
                          objective=float(cp.dot(cp.asarray(c), x).get()))
            # A support LP is an approximation unless its dual also certifies
            # the FULL original variable bounds. Report this separately.
            dual = cp.asarray(solution.get_dual_solution()) / self.objective_scale
            full_dual = cp.zeros(matrix.shape[0], dtype=cp.float64)
            full_dual[cp.asarray(independent)] = dual[:len(independent)]
            full_dual[cp.asarray(eq.shape[0]+kept_ub)] = cp.minimum(dual[len(independent):], 0.)
            gpu_matrix = self.gpu_csr(matrix)
            aty = gpu_matrix.T @ full_dual
            # All nonnegative, unbounded auxiliary variables have positive
            # costs. Shrink the dual if needed so their reduced costs >= 0.
            unbounded = cp.asarray(~np.isfinite(upper)) & (cp.asarray(c) > 0)
            if bool(unbounded.any().get()):
                ratio = cp.min(cp.asarray(c)[unbounded]/cp.maximum(aty[unbounded], 1e-300))
                full_dual *= cp.minimum(1., ratio*(1.-1e-12))
            reduced_cost = cp.asarray(c) - gpu_matrix.T @ full_dual
            contribution = cp.where(reduced_cost >= 0, cp.asarray(lower), cp.asarray(upper))*reduced_cost
            contribution = cp.where(reduced_cost == 0, 0., contribution)
            dual_bound = float((cp.asarray(rhs) @ full_dual + contribution.sum()).get())
            record.update(full_lp_dual_bound=dual_bound,
                full_lp_relative_gap=(record["objective"]-dual_bound)/max(1., abs(record["objective"])),
                optimality_scope=("regularized_qp_not_original_lp" if self.quadratic_regularization else
                                  "full_lp" if not len(omitted) else "training_support_restricted_lp"))
        record.update(success=success, total_seconds=time.perf_counter()-started)
        self.history.append(record)
        return SimpleNamespace(success=success, x=values if success else None,
            fun=record.get("objective"), message=str(record))
