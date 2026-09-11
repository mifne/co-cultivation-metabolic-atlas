"""FP64 GPU PDHG correction for batches of independent original LPs.

The mathematical problem is unchanged::

    minimize c.T @ x
    subject to A_eq @ x == b_eq
               A_ub @ x <= b_ub
               lower <= x <= upper

Row duals use the same convention as HiGHS and :mod:`src.gpu_block_lp`:
``y_ub <= 0`` and reduced costs ``c - A.T @ y``.  The only acceptance
criterion is the existing original-unit certificate.  Exhausting the fixed
iteration budget therefore fails closed; this class never calls a CPU
optimizer or substitutes a relaxed problem.

The diagonal metric follows the row/column absolute-sum preconditioner of
Pock and Chambolle (ICCV 2011, doi:10.1109/ICCV.2011.6126441):
``tau_j = eta / sum_i |A_ij|`` and ``sigma_i = eta / sum_j |A_ij|`` with
``eta < 1``.  Zero rows or columns use the finite neutral value one rather
than an inverse epsilon.  All sparse matrices, iterates and arithmetic remain
FP64 on the selected CUDA device.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from .gpu_block_lp import certify_blocks_device


def _validated_problem(problem: Any):
    """Copy and validate one already-normalized ``_problem`` tuple."""

    if not isinstance(problem, (tuple, list)) or len(problem) != 6:
        raise ValueError(
            "Each problem must be an (a, rhs, lower, upper, c, neq) tuple"
        )
    supplied_a, supplied_rhs, supplied_lower, supplied_upper, supplied_c, neq = problem
    a = csr_matrix(supplied_a, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    m, n = a.shape
    if n < 1 or not np.isfinite(a.data).all():
        raise ValueError("The LP matrix must have finite coefficients and columns")
    if isinstance(neq, (bool, np.bool_)) or not isinstance(neq, (int, np.integer)):
        raise ValueError("neq must be an integer")
    neq = int(neq)
    if not 0 <= neq <= m:
        raise ValueError("neq must identify an initial block of equality rows")

    vectors = []
    for name, supplied, shape, finite in (
        ("rhs", supplied_rhs, (m,), True),
        ("lower", supplied_lower, (n,), False),
        ("upper", supplied_upper, (n,), False),
        ("objective", supplied_c, (n,), True),
    ):
        value = np.asarray(supplied, dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"{name} has the wrong shape")
        if np.isnan(value).any() or (finite and not np.isfinite(value).all()):
            raise ValueError(f"{name} contains invalid values")
        vectors.append(value.copy())
    rhs, lower, upper, c = vectors
    if np.any(lower > upper):
        raise ValueError("Variable lower bounds cannot exceed upper bounds")
    return a, rhs, lower, upper, c, neq


class GpuPdhgCorrector:
    """Correct independent same-shape LP proposals on one GPU.

    Parameters
    ----------
    problems:
        Non-empty iterable of ``(a, rhs, lower, upper, c, neq)`` tuples as
        returned by :func:`src.cpu_repeated_lp._problem`.  A batch must have
        identical matrix dimensions and equality counts; coefficients, RHS,
        bounds and objectives may differ.
    step_safety:
        Strict contraction factor for the absolute-sum diagonal metric.
    theta:
        Primal extrapolation parameter.  ``1`` is standard PDHG.
    primal_weight:
        Positive relative primal/dual metric weight.  The implemented metric
        is ``tau = tau_0 / primal_weight`` and
        ``sigma = primal_weight * sigma_0``.  Their product, and therefore the
        diagonal preconditioner's contraction bound, is unchanged.  The
        default value one is the historical implementation exactly.
    """

    name = "gpu_fp64_block_pdhg_original_lp_corrector"

    def __init__(
        self,
        problems: Iterable[Any],
        *,
        step_safety: float = 0.99,
        theta: float = 1.0,
        primal_weight: float = 1.0,
    ) -> None:
        started = time.perf_counter()
        problems = list(problems)
        if not problems:
            raise ValueError("At least one independent LP is required")
        if not np.isfinite(step_safety) or not 0.0 < float(step_safety) < 1.0:
            raise ValueError("step_safety must be finite and strictly between zero and one")
        if not np.isfinite(theta) or not 0.0 <= float(theta) <= 1.0:
            raise ValueError("theta must be finite and between zero and one")
        if (
            isinstance(primal_weight, (bool, np.bool_))
            or not np.isfinite(primal_weight)
            or float(primal_weight) <= 0.0
        ):
            raise ValueError("primal_weight must be finite and strictly positive")

        normalized = tuple(_validated_problem(problem) for problem in problems)
        shape = normalized[0][0].shape
        neq = normalized[0][-1]
        if any(problem[0].shape != shape or problem[-1] != neq for problem in normalized):
            raise ValueError("A PDHG batch requires identical dimensions and equality counts")
        self.problems = normalized
        self.batch = len(normalized)
        self.m, self.n = shape
        self.neq = neq
        self.step_safety = float(step_safety)
        self.theta = float(theta)
        self.primal_weight = float(primal_weight)

        total_rows = self.batch * self.m
        total_columns = self.batch * self.n
        total_nonzeros = sum(problem[0].nnz for problem in normalized)
        if max(total_rows, total_columns, total_nonzeros) >= np.iinfo(np.int32).max:
            raise ValueError("The block-diagonal CSR exceeds int32 GPU sparse capacity")

        rhs = np.concatenate([problem[1] for problem in normalized])
        lower = np.concatenate([problem[2] for problem in normalized])
        upper = np.concatenate([problem[3] for problem in normalized])
        c = np.concatenate([problem[4] for problem in normalized])
        row_lower = np.concatenate(
            [
                np.r_[problem[1][:neq], np.full(self.m - neq, -np.inf)]
                for problem in normalized
            ]
        )
        inequality_mask = np.tile(
            np.r_[np.zeros(neq, dtype=bool), np.ones(self.m - neq, dtype=bool)],
            self.batch,
        )
        host_assembly_seconds = time.perf_counter() - started

        import cupy as cp
        from cupyx.scipy.sparse import block_diag, csr_matrix as device_csr_matrix

        self.cp = cp
        transfer_started = time.perf_counter()
        blocks = []
        for a, *_rest in normalized:
            blocks.append(
                device_csr_matrix(
                    (
                        cp.asarray(a.data, dtype=cp.float64),
                        cp.asarray(a.indices, dtype=cp.int32),
                        cp.asarray(a.indptr, dtype=cp.int32),
                    ),
                    shape=a.shape,
                )
            )
        self.rhs = cp.asarray(rhs, dtype=cp.float64)
        self.lower = cp.asarray(lower, dtype=cp.float64)
        self.upper = cp.asarray(upper, dtype=cp.float64)
        self.c = cp.asarray(c, dtype=cp.float64)
        self.row_lower = cp.asarray(row_lower, dtype=cp.float64)
        self.inequality_mask = cp.asarray(inequality_mask)
        cp.cuda.runtime.deviceSynchronize()
        gpu_transfer_seconds = time.perf_counter() - transfer_started

        setup_started = time.perf_counter()
        self.a = block_diag(blocks, format="csr", dtype=cp.float64)
        absolute = self.a.copy()
        absolute.data = cp.abs(absolute.data)
        row_sum = cp.asarray(absolute.sum(axis=1)).ravel()
        column_sum = cp.asarray(absolute.sum(axis=0)).ravel()
        base_sigma = cp.where(
            row_sum > 0.0, self.step_safety / row_sum, 1.0
        )
        base_tau = cp.where(
            column_sum > 0.0, self.step_safety / column_sum, 1.0
        )
        self.sigma = base_sigma * self.primal_weight
        self.tau = base_tau / self.primal_weight
        if not bool(
            (cp.all(cp.isfinite(self.sigma)) & cp.all(cp.isfinite(self.tau))).item()
        ):
            raise ValueError("primal_weight produced a non-finite diagonal metric")
        cp.cuda.runtime.deviceSynchronize()
        gpu_setup_seconds = time.perf_counter() - setup_started
        self._assembled = (
            self.a,
            self.row_lower,
            self.rhs,
            self.lower,
            self.upper,
            self.c,
        )
        self.setup_timing = {
            "host_assembly_seconds": host_assembly_seconds,
            "gpu_transfer_seconds": gpu_transfer_seconds,
            "gpu_setup_seconds": gpu_setup_seconds,
            "setup_total_seconds": time.perf_counter() - started,
            "primal_weight": self.primal_weight,
        }
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _valid_budget(value: Any, name: str, *, allow_zero: bool) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if value < (0 if allow_zero else 1):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}")
        return value

    def _warm_array(self, supplied, shape, name, default):
        cp = self.cp
        if supplied is None:
            return default()
        value = cp.asarray(supplied, dtype=cp.float64)
        flat_shape = (int(np.prod(shape)),)
        if value.shape == flat_shape:
            value = value.reshape(shape)
        elif value.shape != shape:
            raise ValueError(f"{name} must have shape {shape} or {flat_shape}")
        if not bool(cp.all(cp.isfinite(value)).item()):
            raise ValueError(f"{name} must contain only finite values")
        return value.copy()

    def _certificate(self, x, y):
        return certify_blocks_device(
            self.problems,
            self._assembled,
            x.ravel(),
            y.ravel(),
            cp=self.cp,
            allow_box_dual=False,
        )

    def _checkpoint(self, iteration, metrics, elapsed):
        accepted = [bool(row["certificate_passed"]) for row in metrics]
        return {
            "iteration": int(iteration),
            "accepted_count": int(sum(accepted)),
            "batch": len(metrics),
            "primal_residual_maximum": float(
                max((row["primal_residual"] for row in metrics), default=np.inf)
            ),
            "dual_violation_maximum": float(
                max((row["dual_violation"] for row in metrics), default=np.inf)
            ),
            "relative_kkt_gap_maximum": float(
                max((row["relative_kkt_gap"] for row in metrics), default=np.inf)
            ),
            "certificate_seconds": float(elapsed),
            "primal_weight": self.primal_weight,
        }

    def solve(
        self,
        initial_x=None,
        initial_y=None,
        *,
        iterations: int = 1000,
        check_interval: int = 50,
    ) -> dict[str, Any]:
        """Run a bounded correction and return device iterates plus strict gates.

        Warm starts may be NumPy or CuPy arrays, either flattened or shaped as
        ``(batch, n)`` / ``(batch, m)``.  A supplied primal is projected into
        its original box and inequality row duals are projected to ``y <= 0``.
        These are PDHG initialization operations, not acceptance relaxations.
        """

        iterations = self._valid_budget(iterations, "iterations", allow_zero=True)
        check_interval = self._valid_budget(
            check_interval, "check_interval", allow_zero=False
        )
        cp = self.cp
        solve_started = time.perf_counter()
        warm_started = time.perf_counter()
        x = self._warm_array(
            initial_x,
            (self.batch, self.n),
            "initial_x",
            lambda: cp.zeros((self.batch, self.n), dtype=cp.float64),
        )
        y = self._warm_array(
            initial_y,
            (self.batch, self.m),
            "initial_y",
            lambda: cp.zeros((self.batch, self.m), dtype=cp.float64),
        )
        flat_x = x.ravel()
        flat_y = y.ravel()
        flat_x = cp.minimum(cp.maximum(flat_x, self.lower), self.upper)
        if self.m > self.neq:
            flat_y = cp.where(self.inequality_mask, cp.minimum(flat_y, 0.0), flat_y)
        x = flat_x.reshape(self.batch, self.n)
        y = flat_y.reshape(self.batch, self.m)
        cp.cuda.runtime.deviceSynchronize()
        warm_start_transfer_seconds = time.perf_counter() - warm_started

        certificate_seconds = 0.0
        before = time.perf_counter()
        metrics = self._certificate(x, y)
        elapsed = time.perf_counter() - before
        certificate_seconds += elapsed
        checkpoints = [self._checkpoint(0, metrics, elapsed)]
        accepted = np.asarray(
            [bool(row["certificate_passed"]) for row in metrics], dtype=bool
        )
        accepted_iteration = np.where(accepted, 0, -1).astype(np.int64)
        done = cp.asarray(accepted)
        x_bar = x.copy()
        correction_seconds = 0.0
        iterations_run = 0

        iteration_budget = 0 if accepted.all() else iterations
        for iteration in range(1, iteration_budget + 1):
            segment_started = time.perf_counter()
            flat_x = x.ravel()
            flat_y = y.ravel()
            flat_x_bar = x_bar.ravel()

            proposed_y = flat_y + self.sigma * (self.rhs - self.a @ flat_x_bar)
            if self.m > self.neq:
                proposed_y = cp.where(
                    self.inequality_mask, cp.minimum(proposed_y, 0.0), proposed_y
                )
            reduced = self.c - self.a.T @ proposed_y
            proposed_x = cp.minimum(
                cp.maximum(flat_x - self.tau * reduced, self.lower), self.upper
            )

            if accepted.any():
                active_x = cp.repeat(~done, self.n)
                active_y = cp.repeat(~done, self.m)
                proposed_x = cp.where(active_x, proposed_x, flat_x)
                proposed_y = cp.where(active_y, proposed_y, flat_y)
            next_x = proposed_x.reshape(self.batch, self.n)
            next_y = proposed_y.reshape(self.batch, self.m)
            x_bar = next_x + self.theta * (next_x - x)
            x, y = next_x, next_y
            iterations_run = iteration

            if iteration % check_interval == 0 or iteration == iterations:
                cp.cuda.runtime.deviceSynchronize()
                correction_seconds += time.perf_counter() - segment_started
                before = time.perf_counter()
                metrics = self._certificate(x, y)
                elapsed = time.perf_counter() - before
                certificate_seconds += elapsed
                checkpoints.append(self._checkpoint(iteration, metrics, elapsed))
                now = np.asarray(
                    [bool(row["certificate_passed"]) for row in metrics], dtype=bool
                )
                newly_accepted = ~accepted & now
                accepted_iteration[newly_accepted] = iteration
                accepted |= now
                done = cp.asarray(accepted)
                if accepted.all():
                    break
            else:
                # Kernel launches are asynchronous.  The elapsed interval is
                # accumulated only at synchronized certificate checkpoints.
                correction_seconds += time.perf_counter() - segment_started

        # The per-iteration launch accounting above includes Python dispatch
        # but not necessarily device completion between checkpoints.  Each
        # checkpoint synchronizes, and an initial all-accepted batch performs
        # no correction work.
        for index, row in enumerate(metrics):
            row["accepted_iteration"] = int(accepted_iteration[index])
            row["success"] = bool(accepted[index])
            row["solver"] = self.name
        solve_total_seconds = time.perf_counter() - solve_started
        timing = dict(self.setup_timing)
        timing.update(
            warm_start_transfer_seconds=warm_start_transfer_seconds,
            correction_seconds=correction_seconds,
            certificate_seconds=certificate_seconds,
            host_control_seconds=max(
                0.0,
                solve_total_seconds
                - warm_start_transfer_seconds
                - correction_seconds
                - certificate_seconds,
            ),
            solve_total_seconds=solve_total_seconds,
        )
        result = {
            "x": x,
            "y": y,
            "accepted": accepted,
            "all_accepted": bool(accepted.all()),
            "iterations_run": int(iterations_run),
            "metrics": metrics,
            "checkpoints": checkpoints,
            "timing": timing,
            "cpu_lp_calls": 0,
            "primal_weight": self.primal_weight,
            "scope": (
                "FP64 GPU PDHG on unchanged original LPs; strict original-unit "
                "certificate; fixed-budget failures remain rejected"
            ),
        }
        self.history.append(
            {
                "batch": self.batch,
                "variables_per_problem": self.n,
                "constraints_per_problem": self.m,
                "nonzeros": int(self.a.nnz),
                "accepted": accepted.tolist(),
                "all_accepted": bool(accepted.all()),
                "iterations_run": int(iterations_run),
                "metrics": [dict(row) for row in metrics],
                "checkpoints": [dict(row) for row in checkpoints],
                "timing": dict(timing),
                "cpu_lp_calls": 0,
                "primal_weight": self.primal_weight,
            }
        )
        return result
