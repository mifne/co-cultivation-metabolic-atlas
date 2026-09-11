"""Joint (block-diagonal) community FBA solver.

The dFBA model currently couples species through the shared medium between
time steps.  This module preserves that semantics while solving the three GEM
LPs as one larger sparse LP.  The block-diagonal construction is mathematically
equivalent to solving the species LPs independently, but reduces Python/solver
launch overhead and gives GPU solvers a larger sparse problem.

The implementation is optional: SciPy HiGHS is used for the CPU joint mode and
NVIDIA cuOpt is used for the GPU joint mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Mapping, Optional
import warnings

import numpy as np
import pandas as pd
from cobra.core import Model
from cobra.core.solution import Solution
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
from scipy.optimize import linprog
from scipy.sparse import block_diag, csr_matrix, hstack, vstack

from .cuopt_solver import CuOptConfig, CuOptUnavailable
from .fba_surrogate import model_fingerprint
from .frozen_community_inputs import FrozenCommunityStepInputs
from .metabolite_ids import canonical_metabolite_id


def _sparse_live_objective_coefficients(model):
    """Use COBRA's exact coefficient semantics on live nonzero candidates.

    The public helper normally visits every reaction and looks up both
    optlang variables, triggering solver synchronization each time. Only a
    forward variable present as a linear expression term can have a nonzero
    net-flux coefficient. Select that small set by its current reaction ID,
    then let COBRA check the actual forward/reverse coefficients. Re-read the
    expression on every call: in-place objectives, renames, added reactions
    and model contexts must never observe a cached vector.
    """
    try:
        terms = model.solver.objective.expression.as_coefficients_dict()
    except AttributeError:
        return linear_reaction_coefficients(model)
    reactions = model.reactions
    candidates = []
    seen = set()
    for term in terms:
        name = getattr(term, 'name', None)
        if isinstance(name, str) and name in reactions and name not in seen:
            candidates.append(reactions.get_by_id(name))
            seen.add(name)
    # Passing [] to COBRA means ALL reactions, not an empty selection.
    return linear_reaction_coefficients(model, reactions=candidates) if candidates else {}


def _freeze_array(array):
    """Mark private template storage read-only and return the same array."""
    array.setflags(write=False)
    return array


def _parsimonious_exchange_signature(n_fluxes, exchange_variables):
    """Fingerprint private stage-three metadata, including scalar semantics."""
    return (
        int(n_fluxes),
        tuple(
            (species, index, stoich, type(index), type(stoich))
            for species, index, stoich in exchange_variables
        ),
    )


def _build_parsimonious_exchange_template(a_eq, n_fluxes, exchange_variables):
    """Build immutable stage-three structure; no biomass value is captured."""
    exchange_variables = tuple(exchange_variables)
    n_aux = len(exchange_variables)
    n_columns = n_fluxes + 1 + n_aux
    indices = np.fromiter(
        (term[1] for term in exchange_variables), dtype=np.int64, count=n_aux
    )
    columns = np.empty((2 * n_aux, 2), dtype=np.int64)
    columns[:, 0] = np.repeat(indices, 2)
    columns[:, 1] = np.repeat(n_fluxes + 1 + np.arange(n_aux), 2)
    transfer_indptr = np.arange(0, 4 * n_aux + 1, 2, dtype=np.int64)
    transfer_coefficients = np.empty((2 * n_aux, 2), dtype=np.float64)
    transfer_coefficients[:, 0] = 0.0
    transfer_coefficients[:, 1] = -1.0

    extended_eq = a_eq.copy()
    extended_eq.resize((a_eq.shape[0], n_columns))
    for array in (extended_eq.data, extended_eq.indices, extended_eq.indptr):
        _freeze_array(array)
    objective = np.zeros(n_columns, dtype=np.float64)
    objective[n_fluxes + 1 :] = 1.0
    return {
        "signature": _parsimonious_exchange_signature(
            n_fluxes, exchange_variables
        ),
        "exchange_variables": exchange_variables,
        "columns": _freeze_array(columns.ravel()),
        "transfer_indptr": _freeze_array(transfer_indptr),
        "transfer_coefficients": _freeze_array(transfer_coefficients.ravel()),
        "extended_eq": extended_eq,
        "objective": _freeze_array(objective),
        "shape": (2 * n_aux, n_columns),
    }


def _assemble_parsimonious_exchange_lp(
    a_ub,
    b_ub,
    a_eq,
    n_fluxes,
    exchange_variables,
    biomass_g_l,
    global_objective,
    aggregate,
    optimize_live_objectives,
    objective_fraction,
    _template=None,
):
    """Assemble the unchanged stage-three LP directly in sparse form.

    Rows retain the previous order: existing inequalities, optional aggregate
    floor, then positive/negative transfer bounds for each exchange variable.
    This is a shared CPU/GPU preprocessing optimization, not an LP reduction.
    """
    exchange_variables = tuple(exchange_variables)
    n_aux = len(exchange_variables)
    n_columns = n_fluxes + 1 + n_aux
    if _template is None:
        template = _build_parsimonious_exchange_template(
            a_eq, n_fluxes, exchange_variables
        )
        # Standalone callers historically received independently mutable
        # arrays. Only the solver-owned template is shared and read-only.
        cached = False
    else:
        template = _template
        expected = _parsimonious_exchange_signature(
            n_fluxes, exchange_variables
        )
        if template["signature"] != expected:
            raise ValueError("Stage-three template metadata changed")
        if template["extended_eq"].shape != (a_eq.shape[0], n_columns):
            raise ValueError("Stage-three equality template shape changed")
        cached = True
    extended_ub = a_ub.copy()
    extended_ub.resize((a_ub.shape[0], n_columns))
    extended_eq = template["extended_eq"] if cached else template["extended_eq"].copy()

    include_performance = bool(optimize_live_objectives and aggregate > 1e-9)
    extra_rows = []
    extra_rhs = np.zeros(int(include_performance) + 2 * n_aux, dtype=np.float64)
    if include_performance:
        performance_indices = np.flatnonzero(global_objective)
        extra_rows.append(
            csr_matrix(
                (
                    -global_objective[performance_indices],
                    performance_indices,
                    np.array([0, len(performance_indices)], dtype=np.int32),
                ),
                shape=(1, n_columns),
            )
        )
        extra_rhs[0] = -aggregate * objective_fraction

    # Each transfer-bound row has at most two nonzeros. Building the entire
    # block avoids allocating a dense n_columns-vector and a CSR object for
    # every individual inequality.
    scales = np.fromiter(
        (
            max(1e-12, float(biomass_g_l[species])) * stoich
            for species, _, stoich in exchange_variables
        ),
        dtype=np.float64,
    )
    coefficients = template["transfer_coefficients"].copy().reshape(2 * n_aux, 2)
    coefficients[:, 0] = (scales[:, None] * np.array([1.0, -1.0])).ravel()
    transfers = csr_matrix(
        (
            coefficients.ravel(),
            template["columns"].copy(),
            template["transfer_indptr"].copy(),
        ),
        shape=template["shape"],
    )
    # Dense-to-CSR previously omitted zero stoichiometric coefficients.
    transfers.eliminate_zeros()
    extra_rows.append(transfers)
    extended_ub = vstack([extended_ub, *extra_rows], format="csr")
    extended_rhs = np.concatenate((b_ub, extra_rhs))
    objective = template["objective"] if cached else template["objective"].copy()
    return objective, extended_ub, extended_rhs, extended_eq


@dataclass
class JointSolveStats:
    """Last-solve diagnostics exposed to benchmark scripts."""

    backend: str
    status: str
    method: str = ""
    solve_seconds: float = 0.0
    build_seconds: float = 0.0
    stage1_seconds: float = 0.0
    stage2_seconds: float = 0.0
    stage3_seconds: float = 0.0


class JointCommunityFbaSolver:
    """Solve all species models in one block-diagonal LP.

    The models remain independent inside the LP (block diagonal ``S``). Their
    bounds and objectives are updated from the live dFBA models before every
    solve. The simulator still applies the resulting exchange fluxes to one
    shared extracellular state, exactly as in the existing per-species mode.
    """

    def __init__(
        self,
        models: Mapping[str, Model],
        backend: str = "scipy",
        config: Optional[CuOptConfig] = None,
    ):
        self.species_names = list(models.keys())
        if not self.species_names:
            raise ValueError("At least one species model is required")
        self.backend = backend.lower()
        if self.backend not in {"scipy", "cuopt"}:
            raise ValueError("Joint backend must be scipy or cuopt")
        self.config = config or CuOptConfig()
        self.stats = JointSolveStats(
            backend=self.backend,
            status="not_started",
            method=str(self.config.method),
        )

        self._offsets: Dict[str, tuple[int, int]] = {}
        matrices = []
        reaction_ids: list[str] = []
        cursor = 0
        for name in self.species_names:
            model = models[name]
            matrix = create_stoichiometric_matrix(
                model, array_type="lil", dtype=np.float64
            ).tocsr()
            matrices.append(matrix)
            start = cursor
            cursor += len(model.reactions)
            self._offsets[name] = (start, cursor)
            reaction_ids.extend([reaction.id for reaction in model.reactions])

        from scipy.sparse import block_diag

        self.a_eq = block_diag(matrices, format="csr")
        self.b_eq = np.zeros(self.a_eq.shape[0], dtype=np.float64)
        self.reaction_ids = reaction_ids
        self.n_variables = self.a_eq.shape[1]

        self._cuopt_problem = None
        self._cuopt_variables = None
        self._cuopt_maximize = None
        if self.backend == "cuopt":
            self._initialize_cuopt(models)

    def _initialize_cuopt(self, models: Mapping[str, Model]) -> None:
        try:
            from cuopt.linear_programming import SolverMethod
            from cuopt.linear_programming.problem import (
                CONTINUOUS,
                MAXIMIZE,
                LinearExpression,
                Problem,
            )
            from cuopt.linear_programming.solver_settings import SolverSettings
        except (ImportError, OSError) as exc:
            raise CuOptUnavailable(
                "cuOpt is not installed; install cuopt-cu13 or cuopt-cu12"
            ) from exc

        self._cuopt_problem = Problem("joint_dFBA_community")
        self._cuopt_variables = []
        for name in self.species_names:
            for reaction in models[name].reactions:
                lb, ub = reaction.bounds
                self._cuopt_variables.append(
                    self._cuopt_problem.addVariable(
                        lb=float(lb),
                        ub=float(ub),
                        obj=0.0,
                        vtype=CONTINUOUS,
                        name=f"{name}__{reaction.id}",
                    )
                )

        # Add S v = 0 rows. Empty rows are redundant and are omitted only from
        # cuOpt; the SciPy matrix retains them harmlessly.
        for row_index in range(self.a_eq.shape[0]):
            start, end = self.a_eq.indptr[row_index : row_index + 2]
            if start == end:
                continue
            variables = [
                self._cuopt_variables[j]
                for j in self.a_eq.indices[start:end]
            ]
            coeffs = self.a_eq.data[start:end].tolist()
            expression = LinearExpression(variables, coeffs, 0.0)
            self._cuopt_problem.addConstraint(
                expression == 0.0, name=f"community_mass_{row_index}"
            )

        self._cuopt_maximize = MAXIMIZE
        settings = SolverSettings()
        method_names = {
            "concurrent": SolverMethod.Concurrent,
            "pdlp": SolverMethod.PDLP,
            "dual simplex": SolverMethod.DualSimplex,
            "dual_simplex": SolverMethod.DualSimplex,
            "dualsimplex": SolverMethod.DualSimplex,
            "barrier": SolverMethod.Barrier,
        }
        method = method_names.get(str(self.config.method).strip().lower())
        if method is None:
            raise ValueError(f"Unsupported cuOpt method: {self.config.method}")
        # PDLP returns a high-quality interior solution directly. Running the
        # crossover phase on this large disconnected block matrix currently
        # triggers cuDSS CSR errors on RTX 4060/cuOpt 26.2, so disable it for
        # PDLP while keeping crossover enabled for barrier/dual-simplex.
        effective_crossover = bool(self.config.crossover)
        if str(self.config.method).strip().lower() == "pdlp":
            effective_crossover = False
        for key, value in {
            "method": method,
            "time_limit": float(self.config.time_limit),
            "crossover": effective_crossover,
            "presolve": int(self.config.presolve),
            "log_to_console": bool(self.config.log_to_console),
            "infeasibility_detection": False,
            "save_best_primal_so_far": True,
        }.items():
            try:
                settings.set_parameter(key, value)
            except (KeyError, TypeError, ValueError, RuntimeError):
                # Keep the core solve usable across cuOpt minor releases.
                continue
        try:
            settings.set_optimality_tolerance(
                float(self.config.optimality_tolerance)
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        self._cuopt_settings = settings

    @staticmethod
    def _objective_vector(models: Mapping[str, Model], offsets: Dict[str, tuple[int, int]], n: int):
        """Return a global maximize vector and original per-model vectors."""

        global_objective = np.zeros(n, dtype=np.float64)
        original: Dict[str, np.ndarray] = {}
        for name, (start, end) in offsets.items():
            coefficients = np.asarray(
                [float(r.objective_coefficient) for r in models[name].reactions],
                dtype=np.float64,
            )
            original[name] = coefficients
            # cuOpt/HiGHS use one global maximize direction here. A minimization
            # model is represented by negating its objective block.
            if models[name].objective.direction == "min":
                coefficients = -coefficients
            global_objective[start:end] = coefficients
        return global_objective, original

    def _make_solutions(
        self,
        models: Mapping[str, Model],
        values: np.ndarray,
        original_objectives: Dict[str, np.ndarray],
    ) -> Dict[str, Optional[Solution]]:
        solutions: Dict[str, Optional[Solution]] = {}
        for name, (start, end) in self._offsets.items():
            flux_values = np.asarray(values[start:end], dtype=np.float64)
            if not np.all(np.isfinite(flux_values)):
                solutions[name] = None
                continue
            fluxes = pd.Series(
                flux_values,
                index=[r.id for r in models[name].reactions],
                dtype=float,
            )
            objective_value = float(np.dot(original_objectives[name], flux_values))
            solutions[name] = Solution(
                objective_value=objective_value,
                status="optimal",
                fluxes=fluxes,
            )
        return solutions

    def solve(
        self, models: Mapping[str, Model]
    ) -> Dict[str, Optional[Solution]]:
        """Solve the current live bounds/objectives and split the result."""

        if list(models.keys()) != self.species_names:
            raise ValueError("Community model order/structure changed after setup")
        lower = np.empty(self.n_variables, dtype=np.float64)
        upper = np.empty(self.n_variables, dtype=np.float64)
        global_objective, original_objectives = self._objective_vector(
            models, self._offsets, self.n_variables
        )
        for name, (start, end) in self._offsets.items():
            bounds = np.asarray([r.bounds for r in models[name].reactions], dtype=float)
            lower[start:end] = bounds[:, 0]
            upper[start:end] = bounds[:, 1]

        import time

        started = time.perf_counter()
        if self.backend == "scipy":
            from scipy.optimize import linprog

            scipy_bounds = [
                (
                    None if not np.isfinite(lb) else float(lb),
                    None if not np.isfinite(ub) else float(ub),
                )
                for lb, ub in zip(lower, upper)
            ]
            result = linprog(
                -global_objective,
                A_eq=self.a_eq,
                b_eq=self.b_eq,
                bounds=scipy_bounds,
                method="highs",
            )
            self.stats.solve_seconds = time.perf_counter() - started
            self.stats.status = str(result.message)
            if not result.success:
                return {name: None for name in self.species_names}
            return self._make_solutions(models, result.x, original_objectives)

        assert self._cuopt_problem is not None
        assert self._cuopt_variables is not None
        objective_coeffs = []
        for index, variable in enumerate(self._cuopt_variables):
            variable.setLowerBound(float(lower[index]))
            variable.setUpperBound(float(upper[index]))
            objective_coeffs.append((variable, float(global_objective[index])))
        self._cuopt_problem.updateObjective(
            coeffs=objective_coeffs, sense=self._cuopt_maximize
        )
        self._cuopt_problem.update()
        self._cuopt_problem.solve(self._cuopt_settings)
        self.stats.solve_seconds = time.perf_counter() - started
        status_obj = getattr(self._cuopt_problem, "Status", "unknown")
        status = getattr(status_obj, "name", str(status_obj)).lower()
        self.stats.status = status
        if status not in {"optimal", "primalfeasible", "primal_feasible"}:
            return {name: None for name in self.species_names}
        values = np.asarray(
            [variable.getValue() for variable in self._cuopt_variables],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            self.stats.status = "invalid_nonfinite_solution"
            return {name: None for name in self.species_names}
        equality_residual = float(
            np.max(np.abs(self.a_eq @ values), initial=0.0)
        )
        bound_violation = float(
            max(
                np.max(np.maximum(lower - values, 0.0), initial=0.0),
                np.max(np.maximum(values - upper, 0.0), initial=0.0),
            )
        )
        if max(equality_residual, bound_violation) > float(
            self.config.validation_tolerance
        ):
            self.stats.status = "invalid_residual"
            return {name: None for name in self.species_names}
        return self._make_solutions(models, values, original_objectives)


def inverse_distance_flux_interpolation(
    candidates: list[tuple[float, np.ndarray]], power: float = 2.0
) -> np.ndarray:
    """Return a convex inverse-distance interpolation of feasible LP fluxes."""

    if not candidates:
        raise ValueError("at least one feasible candidate is required")
    if len(candidates) == 1:
        return np.asarray(candidates[0][1], dtype=np.float64)
    distances = np.asarray([item[0] for item in candidates], dtype=np.float64)
    if np.any(distances < 0) or not np.all(np.isfinite(distances)):
        raise ValueError("candidate distances must be finite and non-negative")
    exact = np.flatnonzero(distances <= 1e-12)
    if len(exact):
        return np.asarray(candidates[int(exact[0])][1], dtype=np.float64)
    weights = np.power(np.maximum(distances, 1e-12), -float(power))
    weights /= weights.sum()
    fluxes = np.stack([np.asarray(item[1], dtype=np.float64) for item in candidates])
    return np.tensordot(weights, fluxes, axes=(0, 0))


def surrogate_validation_status(path: str | Path | None) -> tuple[bool, str]:
    """Return whether an artifact has an explicit scientific qualification."""

    if path is None:
        return False, "validation_manifest_missing"
    manifest = Path(path)
    if not manifest.is_file():
        return False, "validation_manifest_missing"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False, "validation_manifest_invalid"
    status = str(payload.get("status", "")).strip().lower()
    if status != "qualified":
        return False, f"validation_status:{status or 'missing'}"
    return True, "qualified"


class CooperativeCommunityFbaSolver:
    """Exact shared-medium LP that permits simultaneous cross-feeding.

    Unlike :class:`JointCommunityFbaSolver`, this formulation couples each
    extracellular pool across species.  It first maximizes the minimum member
    growth, then maximizes the live model objectives while retaining a chosen
    fraction of that coexistence rate.  A final parsimonious stage minimizes
    gratuitous exchange without reducing the aggregate objective.

    This is a cooperative design objective, not a proof that unengineered
    organisms will secrete the predicted metabolites experimentally.
    """

    def __init__(
        self,
        models: Mapping[str, Model],
        original_exchange_bounds: Optional[
            Mapping[str, Mapping[str, tuple[float, float]]]
        ] = None,
        coexistence_fraction: float = 0.99,
        objective_fraction: float = 0.99,
        maximum_coexistence_growth: Optional[float] = None,
        optimize_live_objectives: bool = False,
        parsimonious_exchange: bool = True,
        highs_presolve: bool = True,
        highs_method: str = "highs",
        capture_training_snapshot: bool = False,
        surrogate_artifact: Optional[str] = None,
        surrogate_device: str = "cuda",
        surrogate_top_k: int = 16,
        surrogate_interpolation_k: int = 1,
        surrogate_interpolation_power: float = 2.0,
        surrogate_exact_interval: int = 128,
        surrogate_distance_threshold: Optional[float] = None,
        surrogate_require_qualified: bool = True,
        surrogate_validation_manifest: Optional[str] = None,
        gpu_qp_projection: bool = False,
        gpu_qp_only: bool = False,
        gpu_qp_candidates: int = 64,
        gpu_qp_rerank_pool: int | None = None,
        gpu_qp_decision_strength: float | None = None,
        gpu_qp_blend_candidates: int = 1,
        gpu_qp_blend_distance_power: float = 2.0,
        gpu_qp_max_iterations: int = 400,
        gpu_qp_normalized_tolerance: float = 2e-5,
        gpu_qp_match_pha: bool = False,
        gpu_qp_multioutput_strength: float = 0.0,
        gpu_qp_independent_species: bool = False,
        gpu_qp_service: Optional[object] = None,
        linear_program_backend: Optional[object] = None,
    ) -> None:
        self.species_names = list(models)
        if not self.species_names:
            raise ValueError("At least one species model is required")
        self.coexistence_fraction = float(np.clip(coexistence_fraction, 0.0, 1.0))
        self.objective_fraction = float(np.clip(objective_fraction, 0.0, 1.0))
        self.maximum_coexistence_growth = (
            None
            if maximum_coexistence_growth is None
            else max(0.0, float(maximum_coexistence_growth))
        )
        self.optimize_live_objectives = bool(optimize_live_objectives)
        self.parsimonious_exchange = bool(parsimonious_exchange)
        if highs_method not in {"highs", "highs-ds", "highs-ipm"}:
            raise ValueError("highs_method must be 'highs', 'highs-ds', or 'highs-ipm'")
        self.highs_presolve = bool(highs_presolve)
        self.highs_method = highs_method
        self.capture_training_snapshot = bool(capture_training_snapshot)
        self.last_training_snapshot: Optional[dict[str, object]] = None
        self.surrogate_top_k = max(1, int(surrogate_top_k))
        self.surrogate_interpolation_k = max(1, int(surrogate_interpolation_k))
        self.surrogate_interpolation_power = max(
            0.25, float(surrogate_interpolation_power)
        )
        self.surrogate_exact_interval = max(0, int(surrogate_exact_interval))
        self.surrogate_attempts = 0
        self.surrogate_accepts = 0
        self.surrogate_rejections = 0
        self.surrogate_rejection_reasons: Dict[str, int] = {}
        self.surrogate_exact_bypasses = 0
        self.cpu_cooperative_solve_calls = 0
        self.cpu_lp_stage_calls = 0
        self.linear_program_backend = linear_program_backend
        self.gpu_lp_stage_calls = 0
        if linear_program_backend is not None and (surrogate_artifact or gpu_qp_projection or gpu_qp_service):
            raise ValueError("Exact LP backend cannot be combined with surrogate/QP paths")
        self.gpu_qp_target_attempts = 0
        self.gpu_qp_target_successes = 0
        self.gpu_qp_projection = bool(gpu_qp_projection)
        self.gpu_qp_only = bool(gpu_qp_only)
        self.gpu_qp_candidates = max(2, int(gpu_qp_candidates))
        self.gpu_qp_rerank_pool = (
            None
            if gpu_qp_rerank_pool is None
            else max(self.gpu_qp_candidates, int(gpu_qp_rerank_pool))
        )
        self.gpu_qp_decision_strength = (
            None
            if gpu_qp_decision_strength is None
            else max(0.0, float(gpu_qp_decision_strength))
        )
        self.gpu_qp_blend_candidates = max(1, int(gpu_qp_blend_candidates))
        self.gpu_qp_blend_distance_power = max(
            0.0, float(gpu_qp_blend_distance_power)
        )
        self.gpu_qp_max_iterations = max(1, int(gpu_qp_max_iterations))
        self.gpu_qp_normalized_tolerance = float(gpu_qp_normalized_tolerance)
        self.gpu_qp_match_pha = bool(gpu_qp_match_pha)
        self.gpu_qp_multioutput_strength = float(gpu_qp_multioutput_strength)
        self.gpu_qp_independent_species = bool(gpu_qp_independent_species)
        if self.gpu_qp_independent_species and not self.gpu_qp_multioutput_strength > 0:
            raise ValueError("Independent species mixing requires multi-output QP")
        if not np.isfinite(self.gpu_qp_multioutput_strength) or self.gpu_qp_multioutput_strength < 0:
            raise ValueError("GPU multi-output strength must be finite and nonnegative")
        if self.gpu_qp_multioutput_strength > 0 and (not self.gpu_qp_projection or gpu_qp_service is not None):
            raise ValueError("Multi-output projection currently requires the local GPU QP path, not the GPU service")
        self._gpu_multioutput_projector = None
        self.gpu_multioutput_attempts = 0
        self.gpu_multioutput_improvements = 0
        self.gpu_qp_service = gpu_qp_service
        self.gpu_qp_attempts = 0
        self.gpu_qp_accepts = 0
        self.gpu_qp_failures = 0
        self.gpu_qp_iterations = 0
        self.gpu_qp_seconds = 0.0
        self.gpu_qp_last_diagnostics: Dict[str, float | int | bool] = {}
        self.gpu_qp_failure_diagnostics: list[Dict[str, float | int | bool]] = []
        self._gpu_qp_projector = None
        self._gpu_qp_pha_reaction_indices: list[int] = []
        self._gpu_qp_pha_decision_positions: list[int] = []
        self._gpu_qp_pha_weights: list[float] = []
        self.solve_calls = 0
        self._surrogate_dictionary = None
        self._surrogate_neural = None
        self.surrogate_disabled_reason: Optional[str] = None
        if surrogate_artifact:
            validation_path = surrogate_validation_manifest
            if validation_path is None:
                validation_path = str(Path(surrogate_artifact).with_suffix(".validation.json"))
            qualified, reason = surrogate_validation_status(validation_path)
            if qualified and surrogate_require_qualified and self.gpu_qp_multioutput_strength > 0:
                manifest_settings = json.loads(Path(validation_path).read_text(encoding="utf-8"))
                if (manifest_settings.get("gpu_qp_multioutput_strength") != self.gpu_qp_multioutput_strength
                    or bool(manifest_settings.get("gpu_qp_independent_species", False)) != self.gpu_qp_independent_species):
                    qualified, reason = False, "validation_multioutput_configuration_mismatch"
            if surrogate_require_qualified and not qualified:
                self.surrogate_disabled_reason = reason
                warnings.warn(
                    "Cooperative GPU surrogate is not scientifically qualified; "
                    f"using exact HiGHS ({reason}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                import torch

                payload = torch.load(
                    Path(surrogate_artifact), map_location="cpu", weights_only=False
                )
                formulation = str(payload.get("metadata", {}).get("formulation", ""))
                del payload
                if formulation == "neural_mechanistic_low_rank_cooperative":
                    from .cooperative_neural_surrogate import (
                        CooperativeNeuralMechanisticSurrogate,
                    )

                    surrogate = CooperativeNeuralMechanisticSurrogate(
                        surrogate_artifact, device=surrogate_device
                    )
                elif formulation == "neural_decision_reranked_feasible_dictionary":
                    from .cooperative_neural_surrogate import (
                        CooperativeNeuralDecisionReranker,
                    )

                    surrogate = CooperativeNeuralDecisionReranker(
                        surrogate_artifact, device=surrogate_device
                    )
                else:
                    from .cooperative_surrogate import CooperativeFluxDictionary

                    surrogate = CooperativeFluxDictionary(
                        surrogate_artifact, device=surrogate_device
                    )
                expected_reactions = [
                    reaction.id
                    for name in self.species_names
                    for reaction in models[name].reactions
                ]
                if surrogate.metadata.get("species") != self.species_names:
                    raise ValueError("cooperative surrogate species order mismatch")
                if surrogate.metadata.get("reaction_ids") != expected_reactions:
                    raise ValueError("cooperative surrogate reaction order mismatch")
                recorded_objective_mode = surrogate.metadata.get(
                    "cooperative_optimize_live_objectives"
                )
                if (
                    recorded_objective_mode is not None
                    and bool(recorded_objective_mode) != self.optimize_live_objectives
                ):
                    raise ValueError(
                        "cooperative surrogate live-objective mode mismatch"
                    )
                expected_fingerprints = {
                    name: model_fingerprint(models[name])
                    for name in self.species_names
                }
                artifact_fingerprints = surrogate.metadata.get(
                    "model_fingerprints"
                )
                if (
                    artifact_fingerprints is not None
                    and artifact_fingerprints != expected_fingerprints
                ):
                    raise ValueError(
                        "cooperative surrogate GEM stoichiometry fingerprint mismatch"
                    )
                if formulation == "neural_mechanistic_low_rank_cooperative":
                    self._surrogate_neural = surrogate
                else:
                    if surrogate_distance_threshold is not None:
                        surrogate.distance_threshold = float(surrogate_distance_threshold)
                    self._surrogate_dictionary = surrogate
        self._linprog_options = {"presolve": self.highs_presolve}
        self.stats = JointSolveStats(
            backend="scipy", status="not_started", method=self.highs_method
        )
        self._offsets: Dict[str, tuple[int, int]] = {}
        self._reaction_ids: Dict[str, tuple[str, ...]] = {}
        self._accepted_frozen_contract = None
        self._growth_terms: Dict[str, tuple[int, float]] = {}
        self._exchange_terms: Dict[str, list[tuple[str, int, float, str]]] = {}
        self._original_exchange_bounds = {
            species: dict(bounds)
            for species, bounds in (original_exchange_bounds or {}).items()
        }

        matrices = []
        cursor = 0
        for species, model in models.items():
            matrix = create_stoichiometric_matrix(
                model, array_type="lil", dtype=np.float64
            ).tocsr()
            matrices.append(matrix)
            start = cursor
            end = start + len(model.reactions)
            self._offsets[species] = (start, end)
            self._reaction_ids[species] = tuple(
                reaction.id for reaction in model.reactions
            )
            growth, coefficient = self._find_growth_reaction(model)
            self._growth_terms[species] = (
                start + model.reactions.index(growth),
                coefficient,
            )
            exchanges = set(model.exchanges)
            for local_index, reaction in enumerate(model.reactions):
                if reaction not in exchanges or len(reaction.metabolites) != 1:
                    continue
                metabolite, stoich = next(iter(reaction.metabolites.items()))
                canonical = canonical_metabolite_id(metabolite.id)
                compartment = str(getattr(metabolite, "compartment", "")).lower()
                if compartment not in {"e", "external", "extracellular"} and not canonical.endswith("_e"):
                    continue
                self._exchange_terms.setdefault(canonical, []).append(
                    (species, start + local_index, float(stoich), reaction.id)
                )
            cursor = end
        self.n_fluxes = cursor
        self.a_eq_flux = block_diag(matrices, format="csr")
        self.b_eq = np.zeros(self.a_eq_flux.shape[0], dtype=np.float64)
        # Stoichiometry and variable ordering are immutable for this solver's
        # lifetime. Build the common equality matrix and fixed objectives once;
        # dFBA only changes bounds, biomass-weighted rows, RHS and live costs.
        self._stage_a_eq = hstack(
            [self.a_eq_flux, csr_matrix((self.a_eq_flux.shape[0], 1))],
            format="csr",
        )
        for array in (
            self._stage_a_eq.data,
            self._stage_a_eq.indices,
            self._stage_a_eq.indptr,
        ):
            _freeze_array(array)
        self._stage1_objective = np.zeros(
            self.n_fluxes + 1, dtype=np.float64
        )
        self._stage1_objective[-1] = -1.0
        _freeze_array(self._stage1_objective)
        exchange_variables = tuple(
            (species, index, stoich)
            for terms in self._exchange_terms.values()
            for species, index, stoich, _ in terms
        )
        self._parsimonious_exchange_template = (
            _build_parsimonious_exchange_template(
                self._stage_a_eq, self.n_fluxes, exchange_variables
            )
        )
        if self.gpu_qp_projection:
            if self.gpu_qp_service is not None:
                pass
            elif self._surrogate_dictionary is None or not hasattr(
                self._surrogate_dictionary, "rank_device"
            ):
                self.surrogate_disabled_reason = "gpu_qp_requires_neural_reranker"
                if self.gpu_qp_only or self.gpu_qp_multioutput_strength > 0:
                    raise ValueError(
                        "GPU-only QP requires a neural-reranked cooperative artifact"
                    )
            else:
                from .gpu_batch_qp import BatchedCooperativeQpProjector

                if self.gpu_qp_rerank_pool is not None:
                    self._surrogate_dictionary.rerank_pool = min(
                        self.gpu_qp_rerank_pool,
                        self._surrogate_dictionary.dictionary.candidate_count,
                    )
                if self.gpu_qp_decision_strength is not None:
                    self._surrogate_dictionary.decision_strength = (
                        self.gpu_qp_decision_strength
                    )
                self._surrogate_dictionary.blend_candidates = (
                    self.gpu_qp_blend_candidates
                )
                self._surrogate_dictionary.blend_distance_power = (
                    self.gpu_qp_blend_distance_power
                )

                shared_ids = sorted(self._exchange_terms)
                species_position = {
                    name: index for index, name in enumerate(self.species_names)
                }
                shared_terms = []
                for metabolite_index, metabolite in enumerate(shared_ids):
                    for species, reaction_index, stoich, _ in self._exchange_terms[
                        metabolite
                    ]:
                        shared_terms.append(
                            (
                                metabolite_index,
                                species_position[species],
                                reaction_index,
                                float(stoich),
                            )
                        )
                self._gpu_qp_projector = BatchedCooperativeQpProjector(
                    self.species_names,
                    shared_ids,
                    shared_terms,
                    species_offsets=[
                        self._offsets[name] for name in self.species_names
                    ],
                    growth_reaction_indices=[
                        self._growth_terms[name][0] for name in self.species_names
                    ],
                    growth_coefficients=[
                        self._growth_terms[name][1] for name in self.species_names
                    ],
                    coexistence_fraction=self.coexistence_fraction,
                    # A global exchange-L1 rerank can select a physically
                    # feasible but context-distant dictionary row.  Keep the
                    # CPU-like objective available in the projector for
                    # controlled experiments, but use the learned local
                    # reference in the qualified online path.
                    parsimonious_reference=False,
                    device=surrogate_device,
                    maximum_iterations=self.gpu_qp_max_iterations,
                    normalized_tolerance=self.gpu_qp_normalized_tolerance,
                )
                reaction_ids = list(self._surrogate_dictionary.metadata["reaction_ids"])
                pha_candidates = [
                    index
                    for index, reaction_id in enumerate(reaction_ids)
                    if reaction_id.lower() in {"ex_pha_c", "ex_phb_c", "ex_phv_c"}
                ]
                decision_indices = (
                    self._surrogate_dictionary.decision_indices.detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )
                for reaction_index in pha_candidates:
                    if self.gpu_qp_match_pha and reaction_index in decision_indices:
                        self._gpu_qp_pha_reaction_indices.append(reaction_index)
                        self._gpu_qp_pha_decision_positions.append(
                            decision_indices.index(reaction_index)
                        )
                        self._gpu_qp_pha_weights.append(
                            0.10012 if reaction_ids[reaction_index].lower() == "ex_phv_c"
                            else 0.08609
                        )

    @staticmethod
    def _find_growth_reaction(model: Model):
        preferred = (
            "R_Growth",
            "Growth",
            "R_BIOMASS_LLA",
            "BIOMASS_LLA",
            "added_biomass_sink",
            "BIOMASS",
            "biomass",
        )
        for reaction_id in preferred:
            if reaction_id in model.reactions:
                return model.reactions.get_by_id(reaction_id), 1.0
        candidates = [
            reaction
            for reaction in model.reactions
            if "growth" in reaction.id.lower() or "biomass" in reaction.id.lower()
        ]
        if candidates:
            return candidates[0], 1.0
        coefficients = linear_reaction_coefficients(model)
        if coefficients:
            reaction, coefficient = max(
                coefficients.items(), key=lambda item: abs(float(item[1]))
            )
            return reaction, float(coefficient)
        raise ValueError(f"No biomass/growth reaction found in {model.id!r}")

    @staticmethod
    def _objective_vector(
        models: Mapping[str, Model],
        offsets: Mapping[str, tuple[int, int]],
        size: int,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        total = np.zeros(size, dtype=np.float64)
        per_species: Dict[str, np.ndarray] = {}
        for species, (start, end) in offsets.items():
            # Avoid checking every reaction's two optlang variables. Select
            # live linear candidates, then use COBRA's coefficient semantics;
            # do not cache across changing dFBA objectives.
            objective_coefficients = _sparse_live_objective_coefficients(models[species])
            coefficients = np.asarray(
                [float(objective_coefficients.get(reaction, 0.0)) for reaction in models[species].reactions],
                dtype=np.float64,
            )
            per_species[species] = coefficients.copy()
            if models[species].objective.direction == "min":
                coefficients = -coefficients
            total[start:end] = coefficients
        return total, per_species

    def _bounds(
        self,
        models: Mapping[str, Model],
        max_crossfeed_uptake: float,
        frozen_bounds_by_species: Optional[Mapping[str, np.ndarray]] = None,
    ) -> list[tuple[Optional[float], Optional[float]]]:
        bounds: list[tuple[Optional[float], Optional[float]]] = []
        exchange_lookup = {
            (species, index): reaction_id
            for terms in self._exchange_terms.values()
            for species, index, _, reaction_id in terms
        }
        for species, (start, end) in self._offsets.items():
            model = models[species]
            originals = self._original_exchange_bounds.get(species, {})
            # Pack live bounds once. Scalar np.isfinite for every reaction
            # dominated this routine; apply exactly the same finite-to-None
            # conversion to the array after the unchanged crossfeed rule.
            if frozen_bounds_by_species is None:
                live_bounds = np.asarray(
                    [reaction.bounds for reaction in model.reactions],
                    dtype=np.float64,
                ).reshape(-1, 2)
            else:
                supplied = np.asarray(
                    frozen_bounds_by_species[species], dtype=np.float64
                )
                if supplied.shape != (len(model.reactions), 2):
                    raise ValueError("Frozen bounds do not match reaction order")
                # Crossfeed relaxation below is solver-local; never alter the
                # read-only per-step input retained for diagnostics/tests.
                live_bounds = supplied.copy()
            for local_index in range(len(model.reactions)):
                reaction_id = exchange_lookup.get((species, start + local_index))
                if reaction_id is not None:
                    lower, upper = map(float, live_bounds[local_index])
                    original_lower = float(originals.get(reaction_id, (lower, upper))[0])
                    if original_lower < 0.0:
                        # A consumer may use material secreted by another
                        # member in the same interval. The shared-pool row
                        # below prevents uptake from an empty environment.
                        lower = min(lower, max(original_lower, -abs(max_crossfeed_uptake)))
                        live_bounds[local_index, 0] = lower
            converted = live_bounds.astype(object)
            converted[~np.isfinite(live_bounds)] = None
            bounds.extend(map(tuple, converted.tolist()))
        return bounds

    def validate_frozen_contract(
        self,
        models: Mapping[str, Model],
        contract,
    ) -> None:
        """Validate and prime one immutable array-input contract."""

        if contract is not self._accepted_frozen_contract:
            if contract.species_names != tuple(self.species_names):
                raise ValueError("Frozen species order does not match this solver")
            for species in self.species_names:
                if contract.reaction_ids[species] != self._reaction_ids[species]:
                    raise ValueError("Frozen reaction order does not match this solver")
                if model_fingerprint(models[species]) != contract.model_fingerprints[species]:
                    raise ValueError("Frozen stoichiometry does not match this solver")
            self._accepted_frozen_contract = contract

    def _validate_frozen_inputs(
        self,
        models: Mapping[str, Model],
        inputs: FrozenCommunityStepInputs,
    ) -> None:
        """Accept one fully validated contract, then check cheap array facts."""

        self.validate_frozen_contract(models, inputs.contract)
        if tuple(inputs.bounds_by_species) != tuple(self.species_names):
            raise ValueError("Frozen bounds species order changed")
        if tuple(inputs.objective_by_species) != tuple(self.species_names):
            raise ValueError("Frozen objective species order changed")
        if tuple(inputs.objective_direction_by_species) != tuple(self.species_names):
            raise ValueError("Frozen objective directions species order changed")
        for species, (start, end) in self._offsets.items():
            size = end - start
            bounds = np.asarray(inputs.bounds_by_species[species])
            objective = np.asarray(inputs.objective_by_species[species])
            if bounds.shape != (size, 2) or objective.shape != (size,):
                raise ValueError("Frozen LP arrays do not match solver dimensions")
            if np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("Frozen LP bounds are invalid")
            if not np.isfinite(objective).all():
                raise ValueError("Frozen LP objective is invalid")
            if inputs.objective_direction_by_species[species] not in {"min", "max"}:
                raise ValueError("Frozen objective direction must be min or max")

    def _frozen_objective_vector(
        self,
        inputs: FrozenCommunityStepInputs,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        total = np.zeros(self.n_fluxes, dtype=np.float64)
        per_species: Dict[str, np.ndarray] = {}
        for species, (start, end) in self._offsets.items():
            coefficients = np.asarray(
                inputs.objective_by_species[species], dtype=np.float64
            ).copy()
            per_species[species] = coefficients.copy()
            if inputs.objective_direction_by_species[species] == "min":
                coefficients = -coefficients
            total[start:end] = coefficients
        return total, per_species

    def _community_inequalities(
        self,
        biomass_g_l: Mapping[str, float],
        medium_mmol_l: Mapping[str, float],
        dt: float,
    ) -> tuple[csr_matrix, np.ndarray]:
        # Cache only structural metadata, never a biomass/medium-dependent
        # matrix. Fingerprint the actual terms, including their order, so an
        # in-place model-layout edit cannot silently retain a stale template.
        growth = tuple((species, index, coefficient, type(coefficient))
                       for species, (index, coefficient) in self._growth_terms.items())
        exchanges = tuple((metabolite, tuple((species, index, stoich, type(stoich), reaction_id)
                           for species, index, stoich, reaction_id in terms))
                          for metabolite, terms in sorted(self._exchange_terms.items()))
        signature = (self.n_fluxes, growth, exchanges)
        template = getattr(self, '_community_inequality_template', None)
        if template is None or template['signature'] != signature:
            from operator import index as integer_index
            n_columns = self.n_fluxes + 1
            indices, indptr, base_values = [], [0], []
            term_slots, term_values = [], []

            def column(value):
                # Preserve NumPy's scalar integer/negative-index semantics.
                value = integer_index(value)
                if value < 0: value += n_columns
                if value < 0 or value >= n_columns:
                    raise IndexError('Community reaction index outside LP columns')
                return value

            for _, index, coefficient, _ in growth:
                row = {column(index): -coefficient}
                row[n_columns-1] = 1.0  # Same final assignment as the dense row.
                for col in sorted(row):
                    indices.append(col); base_values.append(row[col])
                indptr.append(len(indices))
            for _, terms in exchanges:
                columns = [column(index) for _, index, _, _, _ in terms]
                positions = {}
                for col in sorted(set(columns)):
                    positions[col] = len(indices)
                    indices.append(col); base_values.append(0.0)
                for col, (species, _, stoich, _, _) in zip(columns, terms):
                    term_slots.append(positions[col]); term_values.append((species, stoich))
                indptr.append(len(indices))
            template = dict(signature=signature, shape=(len(growth)+len(exchanges), n_columns),
                indices=np.asarray(indices, dtype=np.int64), indptr=np.asarray(indptr, dtype=np.int64),
                base_values=np.asarray(base_values, dtype=np.float64),
                term_slots=np.asarray(term_slots, dtype=np.intp), term_values=tuple(term_values))
            self._community_inequality_template = template

        values = template['base_values'].copy()
        # Evaluate each scalar product in exactly the legacy term order/type.
        # add.at is unbuffered: repeated columns retain the original left-to-
        # right += accumulation, including cancellation and zero coefficients.
        products = np.fromiter((max(1e-12, float(biomass_g_l[species])) * stoich
            for species, stoich in template['term_values']), dtype=np.float64,
            count=len(template['term_values']))
        np.add.at(values, template['term_slots'], products)
        matrix = csr_matrix((values, template['indices'].copy(), template['indptr'].copy()),
                            shape=template['shape'])
        matrix.eliminate_zeros()  # Match dense-to-CSR omission of exact zeros.
        rhs = np.zeros(len(growth)+len(exchanges), dtype=np.float64)
        for row, (metabolite, _) in enumerate(exchanges, start=len(growth)):
            rhs[row] = (1e6 if metabolite in {"h2o_e", "h_e", "oh1_e"} else
                        max(0.0, float(medium_mmol_l.get(metabolite, 0.0))) / dt)
        return matrix, rhs

    def _solutions(
        self,
        models: Mapping[str, Model],
        values: np.ndarray,
        objectives: Mapping[str, np.ndarray],
    ) -> Dict[str, Optional[Solution]]:
        result: Dict[str, Optional[Solution]] = {}
        for species, (start, end) in self._offsets.items():
            flux = np.asarray(values[start:end], dtype=np.float64)
            result[species] = Solution(
                objective_value=float(np.dot(objectives[species], flux)),
                status="optimal",
                fluxes=pd.Series(
                    flux,
                    index=[reaction.id for reaction in models[species].reactions],
                    dtype=float,
                ),
            )
        return result

    def _stage_three_template(self):
        """Return current private exchange metadata and its static LP layout.

        ``_exchange_terms`` is setup metadata, but older tests and diagnostic
        callers may deliberately edit it. Retain their fail-safe behavior by
        rebuilding when ordering, indices, scalar types, or stoichiometry
        changes; ordinary dFBA calls reuse the original immutable template.
        """
        exchange_variables = tuple(
            (species, index, stoich)
            for terms in self._exchange_terms.values()
            for species, index, stoich, _ in terms
        )
        signature = _parsimonious_exchange_signature(
            self.n_fluxes, exchange_variables
        )
        template = self._parsimonious_exchange_template
        if template["signature"] != signature:
            if self._stage_a_eq.shape[1] != self.n_fluxes + 1:
                raise ValueError(
                    "Community flux structure changed after solver setup"
                )
            template = _build_parsimonious_exchange_template(
                self._stage_a_eq, self.n_fluxes, exchange_variables
            )
            self._parsimonious_exchange_template = template
        return exchange_variables, template

    def _solve_linear_program(self, objective, **kwargs):
        if self.linear_program_backend is None:
            self.cpu_lp_stage_calls += 1
            return linprog(objective, **kwargs)
        self.gpu_lp_stage_calls += 1
        return self.linear_program_backend.solve(objective, **kwargs)

    def solve(
        self,
        models: Mapping[str, Model],
        biomass_g_l: Mapping[str, float],
        medium_mmol_l: Mapping[str, float],
        dt: float,
        max_crossfeed_uptake: float = 20.0,
        frozen_inputs: Optional[FrozenCommunityStepInputs] = None,
    ) -> Dict[str, Optional[Solution]]:
        import time

        started = time.perf_counter()
        self.solve_calls += 1
        self.last_training_snapshot = None
        if self.gpu_qp_only and self.gpu_qp_service is None and (
            self._surrogate_dictionary is None or self._gpu_qp_projector is None
        ):
            raise RuntimeError("GPU-only mode has no usable GPU QP solver; refusing CPU LP fallback")
        if list(models) != self.species_names:
            raise ValueError("Community model order/structure changed after setup")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if frozen_inputs is not None:
            self._validate_frozen_inputs(models, frozen_inputs)
        flux_bounds = self._bounds(
            models,
            max_crossfeed_uptake,
            None if frozen_inputs is None else frozen_inputs.bounds_by_species,
        )
        a_ub, b_ub = self._community_inequalities(biomass_g_l, medium_mmol_l, dt)
        a_eq = self._stage_a_eq
        growth_upper = (
            10.0
            if self.maximum_coexistence_growth is None
            else self.maximum_coexistence_growth
        )
        bounds = flux_bounds + [(0.0, growth_upper)]
        if frozen_inputs is None:
            global_objective, per_species = self._objective_vector(
                models, self._offsets, self.n_fluxes
            )
        else:
            global_objective, per_species = self._frozen_objective_vector(
                frozen_inputs
            )

        if self.gpu_qp_service is not None:
            self.surrogate_attempts += 1
            self.gpu_qp_attempts += 1
            lower = np.asarray(
                [-np.inf if item[0] is None else item[0] for item in flux_bounds],
                dtype=np.float32,
            )
            upper = np.asarray(
                [np.inf if item[1] is None else item[1] for item in flux_bounds],
                dtype=np.float32,
            )
            biomass_row = np.asarray(
                [float(biomass_g_l[name]) for name in self.species_names],
                dtype=np.float32,
            )
            context = np.concatenate(
                (
                    biomass_row,
                    np.asarray(b_ub[len(self.species_names) :], dtype=np.float32),
                    lower,
                    upper,
                    global_objective.astype(np.float32, copy=False),
                )
            )
            response = self.gpu_qp_service.predict(context)
            self.gpu_qp_seconds += float(response.get("inference_seconds", 0.0))
            self.gpu_qp_iterations += int(response.get("iterations", 0))
            self.gpu_qp_last_diagnostics = {
                "feasible": bool(response["feasible"]),
                "iterations": int(response.get("iterations", 0)),
                "max_bound_violation": float(
                    response.get("max_bound_violation", float("inf"))
                ),
                "max_shared_violation": float(
                    response.get("max_shared_violation", float("inf"))
                ),
            }
            if bool(response["feasible"]):
                self.surrogate_accepts += 1
                self.gpu_qp_accepts += 1
                self.stats.backend = "cuda_batched_qp_service"
                self.stats.method = "microbatched_convex_hull_projection"
                self.stats.solve_seconds = time.perf_counter() - started
                self.stats.build_seconds = max(
                    0.0,
                    self.stats.solve_seconds
                    - float(response.get("inference_seconds", 0.0)),
                )
                self.stats.stage1_seconds = 0.0
                self.stats.stage2_seconds = 0.0
                self.stats.stage3_seconds = 0.0
                self.stats.status = "gpu_qp_service_feasible"
                return self._solutions(
                    models, np.asarray(response["fluxes"], dtype=np.float64), per_species
                )
            self.surrogate_rejections += 1
            self.gpu_qp_failures += 1
            self.surrogate_rejection_reasons["gpu_qp_service_infeasible"] = (
                self.surrogate_rejection_reasons.get(
                    "gpu_qp_service_infeasible", 0
                )
                + 1
            )
            if self.gpu_qp_only:
                self.stats.backend = "cuda_batched_qp_service"
                self.stats.status = "gpu_qp_service_infeasible_no_cpu_fallback"
                return {species: None for species in self.species_names}

        if self._surrogate_neural is not None:
            exact_bypass = (
                self.surrogate_exact_interval > 0
                and self.solve_calls % self.surrogate_exact_interval == 0
            )
            if exact_bypass:
                self.surrogate_exact_bypasses += 1
            else:
                self.surrogate_attempts += 1
                lower = np.asarray(
                    [-np.inf if item[0] is None else item[0] for item in flux_bounds],
                    dtype=np.float32,
                )
                upper = np.asarray(
                    [np.inf if item[1] is None else item[1] for item in flux_bounds],
                    dtype=np.float32,
                )
                context = np.concatenate(
                    (
                        np.asarray(
                            [float(biomass_g_l[name]) for name in self.species_names],
                            dtype=np.float32,
                        ),
                        np.asarray(
                            b_ub[len(self.species_names) :], dtype=np.float32
                        ),
                        lower,
                        upper,
                        global_objective.astype(np.float32, copy=False),
                    )
                )
                prediction = self._surrogate_neural.predict(context)[0]
                candidate = prediction.fluxes
                rejection_reason = None
                if not prediction.accepted_domain:
                    rejection_reason = prediction.reason
                elif np.any(candidate < lower - 1e-4) or np.any(
                    candidate > upper + 1e-4
                ):
                    rejection_reason = "reaction_bounds"
                else:
                    # The affine exact-flux basis should make this residual
                    # zero by construction.  Keep the numerical check active
                    # so corrupted or incompatible artifacts never propagate.
                    mass_residual = np.asarray(self.a_eq_flux @ candidate).reshape(-1)
                    if np.max(np.abs(mass_residual), initial=0.0) > 2e-3:
                        rejection_reason = "mass_balance"
                    else:
                        shared_matrix = a_ub[
                            len(self.species_names) :, : self.n_fluxes
                        ]
                        shared_rhs = b_ub[len(self.species_names) :]
                        shared_activity = np.asarray(
                            shared_matrix @ candidate
                        ).reshape(-1)
                        if np.any(shared_activity > shared_rhs + 2e-4):
                            rejection_reason = "shared_medium"
                        else:
                            growth_values = [
                                float(candidate[index] * coefficient)
                                for index, coefficient in self._growth_terms.values()
                            ]
                            if min(growth_values) < -1e-8:
                                rejection_reason = "growth"

                if rejection_reason is None:
                    self.surrogate_accepts += 1
                    self.stats.backend = "cuda_neural_mechanistic"
                    self.stats.method = "affine_exact_flux_subspace"
                    self.stats.solve_seconds = time.perf_counter() - started
                    self.stats.build_seconds = max(
                        0.0,
                        self.stats.solve_seconds - prediction.inference_seconds,
                    )
                    self.stats.stage1_seconds = 0.0
                    self.stats.stage2_seconds = 0.0
                    self.stats.stage3_seconds = 0.0
                    self.stats.status = (
                        "surrogate_optimal; "
                        f"ood_score={prediction.ood_score:.8g}"
                    )
                    return self._solutions(models, candidate, per_species)

                self.surrogate_rejections += 1
                self.surrogate_rejection_reasons[rejection_reason] = (
                    self.surrogate_rejection_reasons.get(rejection_reason, 0) + 1
                )

        if self._surrogate_dictionary is not None and self._gpu_qp_projector is not None:
            exact_bypass = (
                not self.gpu_qp_only
                and self.surrogate_exact_interval > 0
                and self.solve_calls % self.surrogate_exact_interval == 0
            )
            if exact_bypass:
                self.surrogate_exact_bypasses += 1
            else:
                self.surrogate_attempts += 1
                self.gpu_qp_attempts += 1
                lower = np.asarray(
                    [-np.inf if item[0] is None else item[0] for item in flux_bounds],
                    dtype=np.float32,
                )
                upper = np.asarray(
                    [np.inf if item[1] is None else item[1] for item in flux_bounds],
                    dtype=np.float32,
                )
                biomass_row = np.asarray(
                    [float(biomass_g_l[name]) for name in self.species_names],
                    dtype=np.float32,
                )
                context = np.concatenate(
                    (
                        biomass_row,
                        np.asarray(
                            b_ub[len(self.species_names) :], dtype=np.float32
                        ),
                        lower,
                        upper,
                        global_objective.astype(np.float32, copy=False),
                    )
                )
                device_candidates = self._surrogate_dictionary.rank_device(
                    context, top_k=self.gpu_qp_candidates
                )
                if self.capture_training_snapshot:
                    # Capture the *current* query even on failure. DAgger needs
                    # rejected states too; a previous step's flux is not a label.
                    self.last_training_snapshot = {
                        "context": context.copy(), "fluxes": None,
                        "source": "gpu_rejected_query",
                    }
                project = self._gpu_qp_projector.project
                multi_options = {}
                if self.gpu_qp_multioutput_strength > 0:
                    from .gpu_multioutput_qp import MultiOutputCooperativeQpProjector
                    if self._gpu_multioutput_projector is None:
                        self._gpu_multioutput_projector = MultiOutputCooperativeQpProjector(
                            self._gpu_qp_projector, self.gpu_qp_multioutput_strength,
                            independent_species=self.gpu_qp_independent_species)
                    project = self._gpu_multioutput_projector.project
                    multi_options = {
                        "decision_indices": self._surrogate_dictionary.decision_indices,
                        "decision_targets": device_candidates.predicted_decision_fluxes,
                        "decision_scales": self._surrogate_dictionary.decision_scale,
                        "decision_weights": self._surrogate_dictionary.decision_weight,
                    }
                projection = project(
                    device_candidates.fluxes,
                    lower,
                    upper,
                    biomass_row,
                    np.asarray(
                        b_ub[len(self.species_names) :], dtype=np.float32
                    ),
                    reference_weights=device_candidates.reference_weights,
                    target_reaction_indices=(self._gpu_qp_pha_reaction_indices or None),
                    target_reaction_weights=(self._gpu_qp_pha_weights or None),
                    enforce_target_in_hull=bool(self._gpu_qp_pha_reaction_indices),
                    target_flux=(
                        None
                        if not self._gpu_qp_pha_decision_positions
                        else (
                            device_candidates.predicted_decision_fluxes[
                                :, self._gpu_qp_pha_decision_positions
                            ]
                            * device_candidates.predicted_decision_fluxes.new_tensor(
                                self._gpu_qp_pha_weights
                            )[None, :]
                        ).sum(dim=1)
                    ),
                    **multi_options,
                )
                self.gpu_qp_iterations += projection.iterations
                self.gpu_qp_seconds += projection.inference_seconds
                self.gpu_multioutput_attempts += int(projection.multioutput_projection_attempted)
                if projection.multioutput_improved is not None:
                    self.gpu_multioutput_improvements += int(projection.multioutput_improved[0])
                self.gpu_qp_target_attempts += int(projection.target_projection_attempted)
                if projection.target_projection_success is not None:
                    self.gpu_qp_target_successes += int(projection.target_projection_success[0])
                self.gpu_qp_last_diagnostics = {
                    "multioutput_attempted": projection.multioutput_projection_attempted,
                    "multioutput_improved": bool(projection.multioutput_improved[0])
                        if projection.multioutput_improved is not None else None,
                    "multioutput_loss_before": float(projection.multioutput_loss_before[0])
                        if projection.multioutput_loss_before is not None else None,
                    "multioutput_loss_after": float(projection.multioutput_loss_after[0])
                        if projection.multioutput_loss_after is not None else None,
                    "multioutput_step_fraction": float(projection.multioutput_step_fraction[0])
                        if projection.multioutput_step_fraction is not None else None,
                    "target_projection_attempted": projection.target_projection_attempted,
                    "target_projection_success": (
                        bool(projection.target_projection_success[0])
                        if projection.target_projection_success is not None else None
                    ),
                    "feasible": bool(projection.feasible[0]),
                    "iterations": int(projection.iterations),
                    "max_bound_violation": float(
                        projection.max_bound_violation[0]
                    ),
                    "max_shared_violation": float(
                        projection.max_shared_violation[0]
                    ),
                    "max_shared_metabolite": (
                        self._gpu_qp_projector.shared_metabolite_ids[
                            int(projection.max_shared_violation_index[0])
                        ]
                    ),
                    "max_normalized_violation": float(
                        projection.max_normalized_violation[0]
                    ),
                }
                if bool(projection.feasible[0]):
                    self.gpu_qp_accepts += 1
                    self.surrogate_accepts += 1
                    selected_flux = projection.fluxes[0]
                    selected_distance = float(
                        device_candidates.distances[0, 0].detach().cpu()
                    )
                    self.stats.backend = "cuda_batched_qp"
                    self.stats.method = (
                        "multioutput_soft_admm_projection" if projection.multioutput_projection_attempted else
                        "pdhg_with_admm_target_projection"
                        if projection.target_projection_attempted else "pdhg_convex_hull_projection"
                    )
                    self.stats.solve_seconds = time.perf_counter() - started
                    self.stats.build_seconds = max(
                        0.0,
                        self.stats.solve_seconds - projection.inference_seconds,
                    )
                    self.stats.stage1_seconds = 0.0
                    self.stats.stage2_seconds = 0.0
                    self.stats.stage3_seconds = 0.0
                    self.stats.status = (
                        "gpu_qp_feasible; "
                        f"distance={selected_distance:.8g}; "
                        f"iterations={projection.iterations}"
                    )
                    if self.capture_training_snapshot:
                        self.last_training_snapshot = {
                            "context": context.copy(),
                            "fluxes": np.asarray(
                                selected_flux, dtype=np.float32
                            ).copy(),
                            "source": "gpu_rollout",
                        }
                    return self._solutions(models, selected_flux, per_species)

                self.gpu_qp_failures += 1
                if len(self.gpu_qp_failure_diagnostics) < 32:
                    self.gpu_qp_failure_diagnostics.append(
                        {
                            "solve_call": int(self.solve_calls),
                            **self.gpu_qp_last_diagnostics,
                        }
                    )
                self.surrogate_rejections += 1
                rejection_reason = "gpu_qp_infeasible"
                self.surrogate_rejection_reasons[rejection_reason] = (
                    self.surrogate_rejection_reasons.get(rejection_reason, 0) + 1
                )
                if self.gpu_qp_only:
                    self.stats.backend = "cuda_batched_qp"
                    self.stats.method = "pdhg_convex_hull_projection"
                    self.stats.solve_seconds = time.perf_counter() - started
                    self.stats.status = "gpu_qp_infeasible_no_cpu_fallback"
                    return {species: None for species in self.species_names}

        if self._surrogate_dictionary is not None and self._gpu_qp_projector is None:
            exact_bypass = (
                self.surrogate_exact_interval > 0
                and self.solve_calls % self.surrogate_exact_interval == 0
            )
            if exact_bypass:
                self.surrogate_exact_bypasses += 1
            else:
                self.surrogate_attempts += 1
                lower = np.asarray(
                    [-np.inf if item[0] is None else item[0] for item in flux_bounds],
                    dtype=np.float32,
                )
                upper = np.asarray(
                    [np.inf if item[1] is None else item[1] for item in flux_bounds],
                    dtype=np.float32,
                )
                context = np.concatenate(
                    (
                        np.asarray(
                            [float(biomass_g_l[name]) for name in self.species_names],
                            dtype=np.float32,
                        ),
                        np.asarray(
                            b_ub[len(self.species_names) :], dtype=np.float32
                        ),
                        lower,
                        upper,
                        global_objective.astype(np.float32, copy=False),
                    )
                )
                candidates = self._surrogate_dictionary.rank(
                    context, top_k=self.surrogate_top_k
                )
                threshold = self._surrogate_dictionary.distance_threshold
                shared_matrix = a_ub[len(self.species_names) :, : self.n_fluxes]
                shared_rhs = b_ub[len(self.species_names) :]
                feasible_candidates: list[tuple[float, np.ndarray]] = []
                selected_distance = float("inf")
                saw_in_domain = False
                saw_bound_feasible = False
                saw_shared_feasible = False
                for distance, candidate in zip(
                    candidates.distances[0], candidates.fluxes[0]
                ):
                    selected_distance = float(distance)
                    if selected_distance > threshold:
                        break
                    saw_in_domain = True
                    if np.any(candidate < lower - 1e-4) or np.any(
                        candidate > upper + 1e-4
                    ):
                        continue
                    saw_bound_feasible = True
                    activity = np.asarray(shared_matrix @ candidate).reshape(-1)
                    if np.any(activity > shared_rhs + 2e-4):
                        continue
                    saw_shared_feasible = True
                    growth_values = [
                        float(candidate[index] * coefficient)
                        for index, coefficient in self._growth_terms.values()
                    ]
                    if min(growth_values) < -1e-8:
                        continue
                    feasible_candidates.append((selected_distance, candidate))
                    if len(feasible_candidates) >= self.surrogate_interpolation_k:
                        break
                if feasible_candidates:
                    selected_distance = feasible_candidates[0][0]
                    selected_flux = inverse_distance_flux_interpolation(
                        feasible_candidates,
                        power=self.surrogate_interpolation_power,
                    )
                    self.surrogate_accepts += 1
                    neural_reranked = (
                        self._surrogate_dictionary.metadata.get("formulation")
                        == "neural_decision_reranked_feasible_dictionary"
                    )
                    self.stats.backend = (
                        "cuda_neural_reranked_dictionary"
                        if neural_reranked
                        else "cuda_dictionary"
                    )
                    if neural_reranked:
                        self.stats.method = "neural_phenotype_rerank_of_exact_fluxes"
                    else:
                        self.stats.method = (
                            "nearest_exact_cooperative_state"
                            if len(feasible_candidates) == 1
                            else f"convex_inverse_distance_k{len(feasible_candidates)}"
                        )
                    self.stats.solve_seconds = time.perf_counter() - started
                    self.stats.build_seconds = self.stats.solve_seconds - candidates.inference_seconds
                    self.stats.stage1_seconds = 0.0
                    self.stats.stage2_seconds = 0.0
                    self.stats.stage3_seconds = 0.0
                    self.stats.status = (
                        "surrogate_optimal; "
                        f"distance={selected_distance:.8g}; threshold={threshold:.8g}"
                    )
                    return self._solutions(
                        models, selected_flux, per_species
                    )
                self.surrogate_rejections += 1
                if not saw_in_domain:
                    rejection_reason = "out_of_distribution"
                elif not saw_bound_feasible:
                    rejection_reason = "reaction_bounds"
                elif not saw_shared_feasible:
                    rejection_reason = "shared_medium"
                else:
                    rejection_reason = "growth"
                self.surrogate_rejection_reasons[rejection_reason] = (
                    self.surrogate_rejection_reasons.get(rejection_reason, 0) + 1
                )

        stage1_started = time.perf_counter()
        self.stats.build_seconds = stage1_started - started
        if self.linear_program_backend is not None:
            self.stats.backend = self.linear_program_backend.name
            self.stats.method = self.linear_program_backend.method

        stage1_objective = self._stage1_objective
        self.cpu_cooperative_solve_calls += int(self.linear_program_backend is None)
        stage1 = self._solve_linear_program(
            stage1_objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=self.b_eq,
            bounds=bounds,
            method=self.highs_method,
            options=self._linprog_options,
        )
        self.stats.stage1_seconds = time.perf_counter() - stage1_started
        if not stage1.success or stage1.x is None:
            self.stats.solve_seconds = time.perf_counter() - started
            self.stats.status = f"coexistence_infeasible: {stage1.message}"
            return {species: None for species in self.species_names}
        coexistence = max(0.0, float(stage1.x[-1]))

        stage2_bounds = list(bounds)
        floor = coexistence * self.coexistence_fraction
        if self.optimize_live_objectives:
            stage2_started = time.perf_counter()
            stage2_bounds[-1] = (floor, coexistence)
            stage2_objective = np.concatenate((-global_objective, [0.0]))
            stage2 = self._solve_linear_program(
                stage2_objective,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=self.b_eq,
                bounds=stage2_bounds,
                method=self.highs_method,
                options=self._linprog_options,
            )
            if self.linear_program_backend is not None and not stage2.success:
                self.stats.status = f"gpu_stage2_failed: {stage2.message}"
                return {species: None for species in self.species_names}
            chosen = stage2 if stage2.success and stage2.x is not None else stage1
            chosen_stage = "aggregate_objective" if chosen is stage2 else "max_min"
            self.stats.stage2_seconds = time.perf_counter() - stage2_started
        else:
            # Cost-minimum maintenance mode: do not consume extra feed merely
            # to maximize one member after the coexistence requirement is met.
            stage2_bounds[-1] = (floor, floor)
            chosen = stage1
            chosen_stage = "max_min"
            self.stats.stage2_seconds = 0.0

        # Parsimonious stage: retain aggregate performance and minimize the
        # absolute environmental transfer rate of every member.
        aggregate = float(np.dot(global_objective, chosen.x[: self.n_fluxes]))
        exchange_variables, stage3_template = self._stage_three_template()
        n_aux = len(exchange_variables)
        self.stats.stage3_seconds = 0.0
        if n_aux and self.parsimonious_exchange:
            stage3_started = time.perf_counter()
            objective, extended_ub, extended_rhs, extended_eq = (
                _assemble_parsimonious_exchange_lp(
                    a_ub,
                    b_ub,
                    a_eq,
                    self.n_fluxes,
                    exchange_variables,
                    biomass_g_l,
                    global_objective,
                    aggregate,
                    self.optimize_live_objectives,
                    self.objective_fraction,
                    _template=stage3_template,
                )
            )
            stage3 = self._solve_linear_program(
                objective,
                A_ub=extended_ub,
                b_ub=extended_rhs,
                A_eq=extended_eq,
                b_eq=self.b_eq,
                bounds=stage2_bounds + [(0.0, None)] * n_aux,
                method=self.highs_method,
                options=self._linprog_options,
            )
            if self.linear_program_backend is not None and not stage3.success:
                self.stats.status = f"gpu_stage3_failed: {stage3.message}"
                return {species: None for species in self.species_names}
            if stage3.success and stage3.x is not None:
                chosen = stage3
                chosen_stage = "parsimonious_exchange"
            else:
                chosen_stage += f"; parsimonious_failed={stage3.message}"
            self.stats.stage3_seconds = time.perf_counter() - stage3_started

        self.stats.solve_seconds = time.perf_counter() - started
        self.stats.status = (
            f"optimal; max_min_growth={coexistence:.8g}; "
            f"retained_fraction={self.coexistence_fraction:.3f}; stage={chosen_stage}"
        )
        self.stats.backend = "scipy" if self.linear_program_backend is None else self.linear_program_backend.name
        self.stats.method = self.highs_method if self.linear_program_backend is None else self.linear_program_backend.method
        if self.capture_training_snapshot:
            lower = np.asarray(
                [-np.inf if item[0] is None else item[0] for item in flux_bounds],
                dtype=np.float32,
            )
            upper = np.asarray(
                [np.inf if item[1] is None else item[1] for item in flux_bounds],
                dtype=np.float32,
            )
            biomass = np.asarray(
                [float(biomass_g_l[name]) for name in self.species_names],
                dtype=np.float32,
            )
            shared_supply = np.asarray(
                b_ub[len(self.species_names) :], dtype=np.float32
            )
            context = np.concatenate(
                (
                    biomass,
                    shared_supply,
                    lower,
                    upper,
                    global_objective.astype(np.float32, copy=False),
                )
            )
            self.last_training_snapshot = {
                "context": context,
                "fluxes": np.asarray(
                    chosen.x[: self.n_fluxes], dtype=np.float32
                ).copy(),
                "common_growth": np.float32(coexistence),
                "aggregate_objective": np.float32(aggregate),
            }
        return self._solutions(models, chosen.x[: self.n_fluxes], per_species)
