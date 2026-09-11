"""Diagnose equality-reduced PDHG residuals on independent saved LPs.

This is a cold, single-stage LP diagnostic on CPU-generated trace inputs.  It
is not a causal replay, closed-loop dFBA run, training evaluation, or speed
benchmark.  Saved CPU solutions are loaded only after selecting the original
LP and are used for objective/error scoring; they are never supplied to the
GPU corrector as warm starts.  Acceptance remains the unchanged original-LP
certificate implemented by :mod:`src.gpu_block_lp`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cpu_repeated_lp import _certificate
from src.gpu_reduced_pdhg import GpuReducedPdhgCorrector
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_trace import load_trace_lp


SCOPE = (
    "Independent cold equality-reduced GPU PDHG on saved exchange LP inputs at "
    "selected CPU-trajectory states. Not causal replay, not closed-loop dFBA, "
    "not a training or speed benchmark. CPU reference x/y are scoring-only and "
    "are never passed to the corrector. Every acceptance is the unchanged "
    "original-unit LP certificate."
)
THRESHOLDS = {"primal_residual": 1e-5, "dual_violation": 1e-7,
              "relative_kkt_gap": 1e-7}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
    """Convert NumPy values recursively and reject non-finite JSON numbers."""
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def _top_indices(values, count, *, positive=False):
    values = np.asarray(values, dtype=np.float64)
    candidates = np.flatnonzero(values > 0.0) if positive else np.arange(len(values))
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-values[candidates], kind="stable")[:count]
    return candidates[order]


def _row_record(problem, activity, y, index, value, *, original_index=None,
                eliminated=False):
    a, rhs, _lower, _upper, _c, neq = problem
    start, stop = a.indptr[index], a.indptr[index + 1]
    coefficients = np.abs(a.data[start:stop])
    return {
        "row_index": int(index),
        "original_row_index": int(index if original_index is None else original_index),
        "row_kind": "equality" if index < neq else "inequality",
        "eliminated_by_reduction": bool(eliminated),
        "value": float(value),
        "activity": float(activity[index]),
        "rhs": float(rhs[index]),
        "activity_minus_rhs": float(activity[index] - rhs[index]),
        "row_dual": float(y[index]),
        "nonzeros": int(stop - start),
        "coefficient_abs_min_nonzero": (
            float(np.min(coefficients)) if len(coefficients) else None
        ),
        "coefficient_abs_max": float(np.max(coefficients, initial=0.0)),
    }


def _column_record(problem, x, reduced_cost, index, value, *, plan=None):
    _a, _rhs, lower, upper, c, _neq = problem
    row = {
        "column_index": int(index),
        "value": float(value),
        "x": float(x[index]),
        "lower": float(lower[index]) if np.isfinite(lower[index]) else None,
        "upper": float(upper[index]) if np.isfinite(upper[index]) else None,
        "objective_coefficient": float(c[index]),
        "reduced_cost": float(reduced_cost[index]),
    }
    if plan is not None:
        group = int(plan.original_to_reduced[index])
        row.update(
            reduced_group=group,
            transform_weight=float(plan.weights[index]),
            group_representative=int(plan.representatives[group]),
        )
    return row


def residual_breakdown(problem, x, y, *, top=8, plan=None, kept_rows=None):
    """Decompose the exact certificate quantities without changing its gates."""
    a, rhs, lower, upper, c, neq = problem
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != c.shape or y.shape != rhs.shape:
        raise ValueError("Primal/dual dimensions do not match the LP")
    activity = np.asarray(a @ x).ravel()
    reduced_cost = c - np.asarray(a.T @ y).ravel()
    row_error = activity - rhs
    equality_error = np.abs(row_error[:neq])
    inequality_violation = np.maximum(row_error[neq:], 0.0)
    lower_violation = np.maximum(lower - x, 0.0)
    upper_violation = np.maximum(x - upper, 0.0)
    inequality_dual_violation = np.maximum(y[neq:], 0.0)
    lower_infinite_dual_violation = np.where(
        ~np.isfinite(lower), np.maximum(reduced_cost, 0.0), 0.0
    )
    upper_infinite_dual_violation = np.where(
        ~np.isfinite(upper), np.maximum(-reduced_cost, 0.0), 0.0
    )
    target = np.where(reduced_cost >= 0.0, lower, upper)
    finite_target = np.where(np.isfinite(target), target, x)
    bound_complementarity = np.abs(reduced_cost * (x - finite_target))
    row_complementarity = np.abs(y[neq:] * (rhs - activity)[neq:])

    solution = SimpleNamespace(
        col_value=x,
        row_dual=y,
        col_dual=reduced_cost,
        value_valid=bool(np.isfinite(x).all()),
        dual_valid=bool(np.isfinite(y).all()),
    )
    certificate = _certificate(*problem, solution)

    removed = set() if plan is None else set(map(int, plan.eliminated_rows))
    equality_rows = [
        _row_record(
            problem, activity, y, int(index), equality_error[index],
            eliminated=int(index) in removed,
        )
        for index in _top_indices(equality_error, top)
    ]
    inequality_rows = [
        _row_record(
            problem, activity, y, int(index + neq), inequality_violation[index],
            eliminated=False,
        )
        for index in _top_indices(inequality_violation, top)
    ]
    row_complementarity_rows = [
        _row_record(
            problem, activity, y, int(index + neq), row_complementarity[index],
            eliminated=False,
        )
        for index in _top_indices(row_complementarity, top, positive=True)
    ]
    # For a reduced problem, map retained row positions back to original IDs.
    if kept_rows is not None:
        kept_rows = np.asarray(kept_rows)
        for rows in (equality_rows, inequality_rows, row_complementarity_rows):
            for row in rows:
                row["original_row_index"] = int(kept_rows[row["row_index"]])

    def columns(values, positive=False):
        return [
            _column_record(problem, x, reduced_cost, int(index), values[index], plan=plan)
            for index in _top_indices(values, top, positive=positive)
        ]

    primal_components = {
        "equality_abs_max": float(np.max(equality_error, initial=0.0)),
        "inequality_violation_max": float(np.max(inequality_violation, initial=0.0)),
        "lower_bound_violation_max": float(np.max(lower_violation, initial=0.0)),
        "upper_bound_violation_max": float(np.max(upper_violation, initial=0.0)),
    }
    dual_components = {
        "inequality_row_dual_violation_max": float(
            np.max(inequality_dual_violation, initial=0.0)
        ),
        "no_finite_lower_reduced_cost_violation_max": float(
            np.max(lower_infinite_dual_violation, initial=0.0)
        ),
        "no_finite_upper_reduced_cost_violation_max": float(
            np.max(upper_infinite_dual_violation, initial=0.0)
        ),
    }
    gate_ratios = {
        name: float(certificate[name] / threshold)
        for name, threshold in THRESHOLDS.items()
    }
    return {
        "certificate": certificate,
        "gate_to_threshold_ratios": gate_ratios,
        "dominant_gate": max(gate_ratios, key=gate_ratios.get),
        "primal_components": primal_components,
        "dual_components": dual_components,
        "bound_complementarity_sum": float(np.sum(bound_complementarity)),
        "row_complementarity_sum": float(np.sum(row_complementarity)),
        "top_equality_residual_rows": equality_rows,
        "top_inequality_violation_rows": inequality_rows,
        "top_row_complementarity_rows": row_complementarity_rows,
        "top_lower_bound_violation_columns": columns(lower_violation, positive=True),
        "top_upper_bound_violation_columns": columns(upper_violation, positive=True),
        "top_bound_complementarity_columns": columns(bound_complementarity, positive=True),
        "top_lower_infinite_dual_violation_columns": columns(
            lower_infinite_dual_violation, positive=True
        ),
        "top_upper_infinite_dual_violation_columns": columns(
            upper_infinite_dual_violation, positive=True
        ),
    }


def _device_provenance(cp):
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {
        "device_id": int(device.id),
        "name": str(name),
        "compute_capability": [
            int(properties.get("major", 0)), int(properties.get("minor", 0))
        ],
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    }


def run_diagnostic(trace, *, steps=(1, 41), environments=4, iterations=2048,
                   check_interval=256, primal_weight=1.0, top=8):
    trace = Path(trace)
    steps = tuple(map(int, steps))
    if (
        not steps
        or len(set(steps)) != len(steps)
        or min(steps) < 1
        or not isinstance(environments, int)
        or environments < 1
        or not isinstance(iterations, int)
        or iterations < 0
        or not isinstance(check_interval, int)
        or check_interval < 1
        or not isinstance(top, int)
        or top < 1
    ):
        raise ValueError("Invalid diagnostic cohort, budget, or top-row count")
    manifest_path = trace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("A completed LP trace is required")
    seeds = manifest.get("seeds", [])
    if environments > len(seeds) or len(set(seeds[:environments])) != environments:
        raise ValueError("Insufficient or duplicate requested environment seeds")
    entries = {}
    for entry in manifest.get("entries", []):
        if (
            entry.get("stage") == "exchange"
            and entry.get("step") in steps
            and entry.get("environment_id", environments) < environments
        ):
            key = (entry["step"], entry["environment_id"])
            if key in entries:
                raise ValueError("Duplicate selected LP trace entry")
            entries[key] = entry
    requested = {(step, env) for step in steps for env in range(environments)}
    if set(entries) != requested:
        raise ValueError("Selected exchange LP cohort is incomplete")

    loaded = {
        key: load_trace_lp(trace, entries[key]) for key in sorted(requested)
    }
    first_problem = loaded[min(requested)][0]
    before = time.perf_counter()
    plan = HomogeneousEqualityReduction.from_problem(first_problem)
    plan_setup_seconds = time.perf_counter() - before

    import cupy as cp

    report = {
        "status": "running",
        "scope": SCOPE,
        "configuration": {
            "trace": str(trace),
            "stage": "exchange",
            "steps": list(steps),
            "environments": environments,
            "iterations": iterations,
            "check_interval": check_interval,
            "primal_weight": float(primal_weight),
            "top": top,
            "warm_start": None,
            "cpu_reference_role": "scoring_only_never_solver_input",
        },
        "certificate_thresholds": dict(THRESHOLDS),
        "trace_manifest_sha256": _sha256(manifest_path),
        "trace_role": manifest.get("role"),
        "evaluation_seeds": seeds[:environments],
        "model_fingerprints": manifest.get("model_fingerprints"),
        "selected_entries": [
            {
                "step": step,
                "environment_id": env,
                "filename": entries[(step, env)]["filename"],
                "file_sha256": entries[(step, env)]["sha256"],
                "problem_sha256": entries[(step, env)]["problem_sha256"],
            }
            for step, env in sorted(requested)
        ],
        "reduction": {
            "equality_fingerprint": plan.equality_fingerprint,
            "original_rows": plan.original_shape[0],
            "original_variables": plan.original_variables,
            "reduced_rows": plan.original_shape[0] - len(plan.eliminated_rows),
            "reduced_variables": plan.reduced_variables,
            "eliminated_equalities": len(plan.eliminated_rows),
            "transform_nonzeros": int(plan.T.nnz),
            "dual_lift_nonzeros": int(plan.dual_lift_map.nnz),
            "plan_setup_seconds": plan_setup_seconds,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gpu": _device_provenance(cp),
        },
        "steps": [],
    }

    for step in steps:
        problems = [loaded[(step, env)][0] for env in range(environments)]
        references = [loaded[(step, env)][1:] for env in range(environments)]
        setup_started = time.perf_counter()
        corrector = GpuReducedPdhgCorrector(
            problems,
            plan=plan,
            primal_weight=primal_weight,
        )
        backend_setup_seconds = time.perf_counter() - setup_started
        solve_started = time.perf_counter()
        # Intentionally cold.  In particular, neither reference_x nor
        # reference_y is passed here or retained between selected states.
        result = corrector.solve(
            iterations=iterations,
            check_interval=check_interval,
        )
        solve_wall_seconds = time.perf_counter() - solve_started
        original_x = cp.asnumpy(result["x"])
        original_y = cp.asnumpy(result["y"])
        reduced_x = cp.asnumpy(result["reduced_x"])
        reduced_y = cp.asnumpy(result["reduced_y"])

        rows = []
        for env, (problem, reduction, reference) in enumerate(
            zip(problems, corrector.reductions, references)
        ):
            reference_x, reference_y = reference
            original = residual_breakdown(
                problem,
                original_x[env],
                original_y[env],
                top=top,
                plan=plan,
            )
            reduced = residual_breakdown(
                reduction.problem,
                reduced_x[env],
                reduced_y[env],
                top=top,
                kept_rows=plan.kept_rows,
            )
            reference_certificate = residual_breakdown(
                problem, reference_x, reference_y, top=1, plan=plan
            )["certificate"]
            if not reference_certificate["certificate_passed"]:
                raise ValueError("A scoring-only CPU reference failed its original certificate")
            if bool(original["certificate"]["certificate_passed"]) != bool(
                result["metrics"][env]["certificate_passed"]
            ):
                raise ArithmeticError("Host and device original certificates disagree")
            rows.append(
                {
                    "environment_id": env,
                    "seed": int(seeds[env]),
                    "problem_sha256": entries[(step, env)]["problem_sha256"],
                    "accepted": bool(result["accepted"][env]),
                    "original": original,
                    "reduced": reduced,
                    "device_original_certificate": dict(result["metrics"][env]),
                    "reference_scoring": {
                        "role": "scoring_only_never_warm_start",
                        "certificate": reference_certificate,
                        "objective": float(problem[4] @ reference_x),
                        "candidate_objective": float(problem[4] @ original_x[env]),
                        "objective_abs_error": float(
                            abs(problem[4] @ (original_x[env] - reference_x))
                        ),
                        "primal_abs_error_max": float(
                            np.max(np.abs(original_x[env] - reference_x), initial=0.0)
                        ),
                    },
                }
            )

        report["steps"].append(
            {
                "step": step,
                "backend_setup_seconds": backend_setup_seconds,
                "solve_wall_seconds": solve_wall_seconds,
                "accepted": int(np.count_nonzero(result["accepted"])),
                "requested": environments,
                "iterations_run": int(result["iterations_run"]),
                "timing": dict(result["timing"]),
                "checkpoints": [dict(item) for item in result["checkpoints"]],
                "rows": rows,
            }
        )
        del result, corrector

    all_rows = [row for step in report["steps"] for row in step["rows"]]
    gate_maxima = {
        name: max(row["original"]["certificate"][name] for row in all_rows)
        for name in THRESHOLDS
    }
    gate_ratios = {
        name: gate_maxima[name] / THRESHOLDS[name] for name in THRESHOLDS
    }
    report["summary"] = {
        "accepted": sum(row["accepted"] for row in all_rows),
        "requested": len(all_rows),
        "original_gate_maxima": gate_maxima,
        "original_gate_maximum_to_threshold_ratios": gate_ratios,
        "dominant_original_gate": max(gate_ratios, key=gate_ratios.get),
        "backend_setup_seconds_sum": sum(
            step["backend_setup_seconds"] for step in report["steps"]
        ),
        "solve_wall_seconds_sum": sum(
            step["solve_wall_seconds"] for step in report["steps"]
        ),
        "correction_seconds_sum": sum(
            step["timing"]["correction_seconds"] for step in report["steps"]
        ),
        "certificate_seconds_sum": sum(
            step["timing"]["certificate_seconds"] for step in report["steps"]
        ),
        "inference_or_cpu_optimizer_used": False,
        "interpretation": (
            "Dominant gate is selected only by its unchanged threshold ratio. "
            "Timing is diagnostic single-stage cost and is not a speedup claim."
        ),
    }
    report["source_hashes"] = {
        name: _sha256(ROOT / name)
        for name in (
            "scripts/diagnose_reduced_pdhg_residuals.py",
            "src/gpu_reduced_pdhg.py",
            "src/gpu_pdhg_corrector.py",
            "src/lp_equality_reduction.py",
            "src/gpu_block_lp.py",
            "src/lp_trace.py",
            "src/cpu_repeated_lp.py",
        )
    }
    report["status"] = "completed"
    return _finite(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 41])
    parser.add_argument("--environments", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2048)
    parser.add_argument("--check-interval", type=int, default=256)
    parser.add_argument("--primal-weight", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_diagnostic(
        args.trace,
        steps=args.steps,
        environments=args.environments,
        iterations=args.iterations,
        check_interval=args.check_interval,
        primal_weight=args.primal_weight,
        top=args.top,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
