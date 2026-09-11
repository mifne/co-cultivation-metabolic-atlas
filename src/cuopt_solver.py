"""Optional NVIDIA cuOpt backend for COBRApy FBA models.

The project keeps GLPK as the default because cuOpt is an optional NVIDIA
package and GPU LP performance depends strongly on model size and batching.
This adapter builds the stoichiometric LP once, then updates reaction bounds
and objective coefficients on every dFBA step.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from cobra.core import Model
from cobra.core.solution import Solution
from cobra.util.array import create_stoichiometric_matrix


class CuOptUnavailable(RuntimeError):
    """Raised when the optional cuOpt Python package is not installed."""


def cuopt_available() -> bool:
    """Return whether the cuOpt Python API can be imported."""

    try:
        import cuopt.linear_programming  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@dataclass
class CuOptConfig:
    """Settings chosen for repeated dFBA solves."""

    method: str = "barrier"
    time_limit: float = 3.0
    crossover: bool = True
    presolve: int = 2
    pdlp_precision: str = "double"
    log_to_console: bool = False
    # PDLP reaches this tolerance quickly; crossover then restores a basic,
    # high-accuracy FBA solution which is validated at 1e-5 on the original S.
    optimality_tolerance: float = 1e-5
    validation_tolerance: float = 1e-5
    remove_dependent_rows: bool = True
    max_dense_rank_elements: int = 10_000_000
    scale_constraint_rows: bool = False


@dataclass
class CuOptSolveStats:
    """Measured host/update/solve costs and numerical validity."""

    status: str = "not_started"
    method: str = ""
    update_seconds: float = 0.0
    solve_seconds: float = 0.0
    extract_seconds: float = 0.0
    changed_bounds: int = 0
    changed_objectives: int = 0
    max_equality_residual: float = float("inf")
    max_bound_violation: float = float("inf")
    objective_value: float = float("nan")
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    original_constraints: int = 0
    reduced_constraints: int = 0
    dependent_rows_removed: int = 0
    rank_reduction_seconds: float = 0.0
    initialization_seconds: float = 0.0
    crossover_enabled: bool = False
    retried_without_crossover: bool = False


class CuOptFbaSolver:
    """Persistent cuOpt representation of one COBRApy continuous LP.

    COBRApy represents FBA as ``S @ v == 0`` with reaction bounds and a
    linear objective.  Those bounds change during dFBA, so the problem is
    constructed once and only variable bounds/objective coefficients are
    updated before each solve.
    """

    def __init__(self, model: Model, config: Optional[CuOptConfig] = None):
        initialization_started = time.perf_counter()
        try:
            from cuopt.linear_programming.problem import (
                CONTINUOUS,
                MAXIMIZE,
                MINIMIZE,
                LinearExpression,
                Problem,
            )
            from cuopt.linear_programming import SolverMethod
            from cuopt.linear_programming.solver_settings import (
                PDLPSolverMode,
                SolverSettings,
            )
        except (ImportError, OSError) as exc:
            raise CuOptUnavailable(
                "cuOpt is not installed. Install the NVIDIA wheel for the CUDA "
                "version in use, e.g. cuopt-cu13 from pypi.nvidia.com."
            ) from exc

        self.config = config or CuOptConfig()
        self._LinearExpression = LinearExpression
        self._MAXIMIZE = MAXIMIZE
        self._MINIMIZE = MINIMIZE
        self._SolverMethod = SolverMethod
        self._PDLPSolverMode = PDLPSolverMode
        self._SolverSettings = SolverSettings
        self.problem = Problem(f"dFBA_{model.id}")
        self.reaction_ids = [reaction.id for reaction in model.reactions]
        self.variables = []

        for reaction in model.reactions:
            lb, ub = reaction.bounds
            self.variables.append(
                self.problem.addVariable(
                    lb=float(lb),
                    ub=float(ub),
                    obj=float(reaction.objective_coefficient),
                    vtype=CONTINUOUS,
                    name=reaction.id,
                )
            )

        matrix = create_stoichiometric_matrix(
            model, array_type="lil", dtype=np.float64
        ).tocsr()
        self._matrix = matrix
        nonempty_rows = np.flatnonzero(np.diff(matrix.indptr) != 0)
        reduced_matrix = matrix[nonempty_rows].tocsr()
        rank_reduction_seconds = 0.0
        if (
            self.config.remove_dependent_rows
            and reduced_matrix.shape[0] > 0
            and reduced_matrix.shape[0] * reduced_matrix.shape[1]
            <= int(self.config.max_dense_rank_elements)
        ):
            # FBA stoichiometric matrices contain conservation-law dependent
            # rows.  They are harmless to simplex, but make the normal/KKT
            # systems used by GPU first-order/interior methods singular or
            # extremely ill-conditioned.  Pivoted QR on S.T selects an
            # independent row basis.  Solutions are still checked against the
            # complete original S below, so an unsafe rank decision is rejected.
            from scipy.linalg import qr

            rank_started = time.perf_counter()
            r_factor, pivots = qr(
                reduced_matrix.T.toarray(),
                mode="r",
                pivoting=True,
                check_finite=False,
            )
            diagonal = np.abs(np.diag(r_factor))
            if diagonal.size:
                rank_tolerance = (
                    max(reduced_matrix.shape)
                    * np.finfo(np.float64).eps
                    * float(np.max(diagonal))
                )
                rank = int(np.count_nonzero(diagonal > rank_tolerance))
                independent_rows = np.sort(np.asarray(pivots[:rank], dtype=int))
                reduced_matrix = reduced_matrix[independent_rows].tocsr()
            rank_reduction_seconds = time.perf_counter() - rank_started
        if self.config.scale_constraint_rows and reduced_matrix.shape[0] > 0:
            row_maximum = np.asarray(
                np.abs(reduced_matrix).max(axis=1).toarray()
            ).ravel()
            row_scale = np.ones(reduced_matrix.shape[0], dtype=np.float64)
            row_scale[row_maximum > 0] = 1.0 / row_maximum[row_maximum > 0]
            reduced_matrix = reduced_matrix.multiply(row_scale[:, None]).tocsr()
        self._constraint_matrix = reduced_matrix

        for row_index in range(reduced_matrix.shape[0]):
            start, end = reduced_matrix.indptr[row_index : row_index + 2]
            vars_in_row = [
                self.variables[j] for j in reduced_matrix.indices[start:end]
            ]
            coeffs = reduced_matrix.data[start:end].tolist()
            expression = LinearExpression(vars_in_row, coeffs, 0.0)
            self.problem.addConstraint(expression == 0.0, name=f"mass_{row_index}")

        self._last_lower = np.asarray(
            [float(reaction.lower_bound) for reaction in model.reactions],
            dtype=np.float64,
        )
        self._last_upper = np.asarray(
            [float(reaction.upper_bound) for reaction in model.reactions],
            dtype=np.float64,
        )
        # Force the first solve to apply both the complete objective and its
        # sense. Problem.addVariable(obj=...) sets coefficients, but the
        # Problem default sense is not guaranteed to match COBRA's maximize.
        self._last_objective = np.full(len(model.reactions), np.nan)
        self._last_sense: Optional[str] = None
        self.stats = CuOptSolveStats(method=str(self.config.method))
        self.stats.original_constraints = int(matrix.shape[0])
        self.stats.reduced_constraints = int(reduced_matrix.shape[0])
        self.stats.dependent_rows_removed = int(
            matrix.shape[0] - reduced_matrix.shape[0]
        )
        self.stats.rank_reduction_seconds = rank_reduction_seconds
        self._method_key = str(self.config.method).strip().lower()
        self._warm_start_data = None
        self._configure_settings()
        self.stats.initialization_seconds = time.perf_counter() - initialization_started

    def _configure_settings(self):
        settings = self._SolverSettings()
        # Parameter names follow NVIDIA's Python API.  Keep this in one place
        # so version changes are easy to diagnose.
        method_names = {
            "concurrent": self._SolverMethod.Concurrent,
            "pdlp": self._SolverMethod.PDLP,
            "dual simplex": self._SolverMethod.DualSimplex,
            "dual_simplex": self._SolverMethod.DualSimplex,
            "dualsimplex": self._SolverMethod.DualSimplex,
            "barrier": self._SolverMethod.Barrier,
        }
        method = method_names.get(str(self.config.method).strip().lower())
        if method is None:
            raise ValueError(
                f"Unsupported cuOpt method {self.config.method!r}; "
                f"choose one of {sorted(method_names)}"
            )
        method_key = str(self.config.method).strip().lower()
        effective_crossover = bool(self.config.crossover)
        if method_key == "pdlp" and not self.config.remove_dependent_rows:
            effective_crossover = False
        params = {
            "method": method,
            "time_limit": float(self.config.time_limit),
            "crossover": effective_crossover,
            "presolve": int(self.config.presolve),
            "log_to_console": bool(self.config.log_to_console),
            # PDLP's infeasibility declaration is heuristic.  A dFBA step can
            # be validated directly from S@v and bounds instead.
            "infeasibility_detection": False,
            "save_best_primal_so_far": True,
        }
        self._effective_crossover = effective_crossover
        if method_key == "pdlp" and int(self.config.presolve) == 0:
            params["pdlp_solver_mode"] = self._PDLPSolverMode.Stable2
        for name, value in params.items():
            try:
                settings.set_parameter(name, value)
            except (KeyError, TypeError, ValueError, RuntimeError):
                # Older cuOpt wheels may not expose newer optional settings;
                # the core LP solve remains usable without them.
                continue
        try:
            settings.set_optimality_tolerance(
                float(self.config.optimality_tolerance)
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        self.settings = settings
        self.stats.crossover_enabled = effective_crossover

    def solve(self, model: Model) -> Optional[Solution]:
        """Update the persistent LP from *model* and solve it on cuOpt."""

        if len(model.reactions) != len(self.variables):
            raise ValueError("The COBRA model structure changed after cuOpt setup")

        self.stats.attempts += 1
        self.stats.status = "running"
        self.stats.max_equality_residual = float("inf")
        self.stats.max_bound_violation = float("inf")
        self.stats.objective_value = float("nan")
        self.stats.extract_seconds = 0.0
        update_started = time.perf_counter()
        lower = np.asarray(
            [float(reaction.lower_bound) for reaction in model.reactions],
            dtype=np.float64,
        )
        upper = np.asarray(
            [float(reaction.upper_bound) for reaction in model.reactions],
            dtype=np.float64,
        )
        objective = np.asarray(
            [float(reaction.objective_coefficient) for reaction in model.reactions],
            dtype=np.float64,
        )

        changed_bounds = np.flatnonzero(
            (lower != self._last_lower) | (upper != self._last_upper)
        )
        for index in changed_bounds:
            variable = self.variables[int(index)]
            variable.setLowerBound(float(lower[index]))
            variable.setUpperBound(float(upper[index]))

        changed_objective = np.flatnonzero(objective != self._last_objective)
        objective_coeffs = [
            (self.variables[int(index)], float(objective[index]))
            for index in changed_objective
        ]

        sense = (
            self._MAXIMIZE
            if model.objective.direction == "max"
            else self._MINIMIZE
        )
        sense_changed = model.objective.direction != self._last_sense
        if objective_coeffs or sense_changed:
            self.problem.updateObjective(coeffs=objective_coeffs, sense=sense)
        if changed_bounds.size or objective_coeffs or sense_changed:
            self.problem.update()
        self._last_lower = lower
        self._last_upper = upper
        self._last_objective = objective
        self._last_sense = model.objective.direction
        self.stats.changed_bounds = int(changed_bounds.size)
        self.stats.changed_objectives = int(changed_objective.size)
        self.stats.update_seconds = time.perf_counter() - update_started

        solve_started = time.perf_counter()
        if (
            self._method_key == "pdlp"
            and int(self.config.presolve) == 0
            and self._warm_start_data is not None
        ):
            try:
                self.settings.set_pdlp_warm_start_data(self._warm_start_data)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                self._warm_start_data = None
        try:
            self.problem.solve(self.settings)
        except (TypeError, ValueError, RuntimeError):
            # A stale/incompatible warm start must never make a training step
            # fail. Retry once from a cold start.
            if self._method_key != "pdlp" or self._warm_start_data is None:
                raise
            self._warm_start_data = None
            self.settings.pdlp_warm_start_data = None
            self.problem.solve(self.settings)
        status = getattr(self.problem.Status, "name", str(self.problem.Status)).lower()
        if (
            status == "numericalerror"
            and self._method_key == "pdlp"
            and self._effective_crossover
        ):
            # On highly degenerate zero-objective FBA states PDLP can converge
            # while crossover's basis construction fails. Retry PDLP without
            # crossover and accept it only after the original S/bounds checks.
            self.settings.set_parameter("crossover", False)
            self._effective_crossover = False
            self.stats.retried_without_crossover = True
            self.problem.solve(self.settings)
            status = getattr(
                self.problem.Status, "name", str(self.problem.Status)
            ).lower()
        self.stats.solve_seconds = time.perf_counter() - solve_started
        self.stats.status = status
        if status not in {"optimal", "primalfeasible", "primal_feasible"}:
            self.stats.failures += 1
            return None

        extract_started = time.perf_counter()
        values = np.asarray([variable.getValue() for variable in self.variables], dtype=float)
        if values.shape[0] != len(self.reaction_ids) or not np.all(np.isfinite(values)):
            self.stats.status = "invalid_nonfinite_solution"
            self.stats.failures += 1
            return None
        equality_residual = np.asarray(self._matrix @ values, dtype=np.float64)
        max_equality_residual = (
            float(np.max(np.abs(equality_residual)))
            if equality_residual.size
            else 0.0
        )
        max_bound_violation = float(
            max(
                np.max(np.maximum(lower - values, 0.0), initial=0.0),
                np.max(np.maximum(values - upper, 0.0), initial=0.0),
            )
        )
        self.stats.max_equality_residual = max_equality_residual
        self.stats.max_bound_violation = max_bound_violation
        tolerance = float(self.config.validation_tolerance)
        if max(max_equality_residual, max_bound_violation) > tolerance:
            self.stats.status = "invalid_residual"
            self.stats.failures += 1
            self.stats.extract_seconds = time.perf_counter() - extract_started
            return None
        objective_value = float(self.problem.ObjValue)
        self.stats.objective_value = objective_value
        if not np.isfinite(objective_value):
            self.stats.status = "invalid_objective"
            self.stats.failures += 1
            self.stats.extract_seconds = time.perf_counter() - extract_started
            return None
        self.stats.successes += 1
        self.stats.extract_seconds = time.perf_counter() - extract_started
        if self._method_key == "pdlp" and int(self.config.presolve) == 0:
            try:
                self._warm_start_data = self.problem.getWarmstartData()
            except (AttributeError, TypeError, ValueError, RuntimeError):
                self._warm_start_data = None
        return Solution(
            objective_value=objective_value,
            status="optimal",
            fluxes=pd.Series(values, index=self.reaction_ids, dtype=float),
        )
