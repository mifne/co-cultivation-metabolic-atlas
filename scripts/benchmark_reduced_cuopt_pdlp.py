"""Cold reduced-cuOpt convergence probe on recorded exchange LPs.

This script answers one narrow question: after an exact homogeneous-equality
reduction, can cuOpt's mature PDLP implementation return a solution that passes
the unchanged ORIGINAL-LP certificate quickly enough to merit integration?

The recorded HiGHS primal/dual vectors are scoring references only.  They are
never supplied to cuOpt.  This is a single-stage development diagnostic, not a
dFBA trajectory benchmark and not evidence of a fully device-resident path:
cuOpt's direct Python API accepts/returns host arrays in this installed version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cpu_repeated_lp import RepeatedCpuLP, _certificate
from src.gpu_block_lp import assemble_blocks, certify_blocks_device
from src.lp_equality_reduction import HomogeneousEqualityReduction
from src.lp_trace import load_trace_lp, problem_request


THRESHOLDS = {
    "primal_residual": 1e-5,
    "dual_violation": 1e-7,
    "relative_kkt_gap": 1e-7,
}

SCOPE = (
    "Cold, independent, same-time exchange-LP development diagnostic. Exact "
    "host equality reduction; one manually block-diagonal cuOpt PDLP solve; "
    "host primal/dual postsolve; strict ORIGINAL-LP device certificate. No "
    "trajectory update, temporal warm start, neural input, CPU optimizer in "
    "the GPU path, or relaxed acceptance threshold. The installed direct "
    "cuOpt Python API crosses the host/device boundary and is not fully "
    "device resident."
)

PRIMARY_SOURCES = [
    {
        "title": "Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient",
        "url": "https://papers.neurips.cc/paper_files/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf",
        "relevance": (
            "PDLP combines PDHG with presolve, diagonal preconditioning, "
            "adaptive steps, primal weighting, and adaptive restarts."
        ),
    },
    {
        "title": "NVIDIA cuOpt LP/QP Features 26.08",
        "url": "https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html",
        "relevance": (
            "Official definitions of PDLP, PSLP presolve, solver modes, "
            "precision, warm starts, and Python/API timing boundaries."
        ),
    },
    {
        "title": "NVIDIA cuOpt FAQ 26.08",
        "url": "https://docs.nvidia.com/cuopt/user-guide/latest/faq.html",
        "relevance": (
            "Official advice that tolerance, solver mode, and presolve require "
            "problem-class measurement; LP batch mode is deprecated."
        ),
    },
    {
        "title": "cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming",
        "url": "https://arxiv.org/abs/2507.14051",
        "relevance": (
            "Recent primary evidence that restart and primal-weight control, "
            "rather than raw fixed PDHG iteration count alone, matter on GPU."
        ),
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
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
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    # cuOpt statistics occasionally contain enum-like scalar objects.
    return str(value)


def _enum_name(value) -> str:
    return str(getattr(value, "name", value)).lower()


def _host_certificate(problem, x, y):
    a, rhs, _lower, _upper, c, _neq = problem
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    solution = SimpleNamespace(
        col_value=x,
        row_dual=y,
        col_dual=np.asarray(c - a.T @ y).ravel(),
        value_valid=True,
        dual_valid=True,
    )
    return _certificate(*problem, solution)


def _selected_entries(manifest, *, step: int, environments: int):
    selected = sorted(
        (
            entry
            for entry in manifest.get("entries", [])
            if entry.get("stage") == "exchange"
            and entry.get("step") == step
            and entry.get("environment_id", environments) < environments
        ),
        key=lambda entry: entry["environment_id"],
    )
    if [entry.get("environment_id") for entry in selected] != list(range(environments)):
        raise ValueError("The selected exchange LP cohort is incomplete or duplicated")
    return selected


def _cpu_cold_reference(requests, *, workers: int):
    initialize_started = time.perf_counter()
    backend = RepeatedCpuLP(workers=workers)
    initialize_seconds = time.perf_counter() - initialize_started
    close_error = None
    try:
        started = time.perf_counter()
        results = backend.solve_batch(requests)
        solve_seconds = time.perf_counter() - started
        history = backend.history[-1]
    finally:
        close_started = time.perf_counter()
        try:
            backend.close()
        except BaseException as error:  # preserve diagnostics before propagating
            close_error = error
        close_seconds = time.perf_counter() - close_started
    if close_error is not None:
        raise close_error
    if not all(result.success for result in results):
        raise RuntimeError("The matching cold HiGHS reference did not certify every original LP")
    return results, {
        "backend_initialization_seconds": initialize_seconds,
        "solve_batch_wall_seconds": solve_seconds,
        "cleanup_seconds_excluded": close_seconds,
        "history": history,
        "all_original_certificates_passed": True,
        "scope": (
            "One fresh RepeatedCpuLP backend, one model per stable environment, "
            "workers fixed by CLI; first-call cold model construction and solve "
            "included, backend construction and executor teardown reported separately."
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
            int(properties.get("major", 0)),
            int(properties.get("minor", 0)),
        ],
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    }


def _make_settings(lp, *, time_limit: float, tolerance: float, presolve: int, mode: str):
    from cuopt.linear_programming import PDLPSolverMode, SolverMethod

    mode_value = {
        "stable3": PDLPSolverMode.Stable3,
        "methodical1": PDLPSolverMode.Methodical1,
        "stable2": PDLPSolverMode.Stable2,
        "fast1": PDLPSolverMode.Fast1,
    }[mode]
    settings = lp.SolverSettings()
    parameters = {
        "method": SolverMethod.PDLP,
        "crossover": False,
        "presolve": int(presolve),
        "time_limit": float(time_limit),
        "log_to_console": False,
        "num_cpu_threads": 1,
        # Installed cuOpt's value 1 is the explicit FP64 PDLP mode, matching
        # the already-qualified GpuBlockLP configuration in this repository.
        "pdlp_precision": 1,
        "pdlp_solver_mode": mode_value,
        "dual_postsolve": True,
    }
    for name, value in parameters.items():
        settings.set_parameter(name, value)
    settings.set_optimality_tolerance(float(tolerance))
    # The acceptance gate is absolute in original units.  Do not let a large
    # RHS norm turn the solver's primal stopping threshold into a loose proxy.
    settings.set_parameter("relative_primal_tolerance", 0.0)
    return settings, {
        key: (_enum_name(value) if key in {"method", "pdlp_solver_mode"} else value)
        for key, value in parameters.items()
    }


def _solve_reduced_cuopt(
    original_problems,
    reductions,
    *,
    cp,
    lp,
    time_limit: float,
    tolerance: float,
    presolve: int,
    mode: str,
):
    pipeline_started = time.perf_counter()
    reduced_problems = [reduction.problem for reduction in reductions]

    before = time.perf_counter()
    reduced_packed = assemble_blocks(reduced_problems)
    original_packed = assemble_blocks(original_problems)
    block_assembly_seconds = time.perf_counter() - before
    a, row_lower, rhs, lower, upper, c = reduced_packed

    before = time.perf_counter()
    model = lp.DataModel()
    model.set_csr_constraint_matrix(
        a.data,
        a.indices.astype(np.int32, copy=False),
        a.indptr.astype(np.int32, copy=False),
    )
    model.set_constraint_lower_bounds(row_lower)
    model.set_constraint_upper_bounds(rhs)
    model.set_variable_lower_bounds(lower)
    model.set_variable_upper_bounds(upper)
    model.set_objective_coefficients(c)
    model.set_maximize(False)
    settings, parameters = _make_settings(
        lp,
        time_limit=time_limit,
        tolerance=tolerance,
        presolve=presolve,
        mode=mode,
    )
    model_and_settings_seconds = time.perf_counter() - before

    cp.cuda.runtime.deviceSynchronize()
    before = time.perf_counter()
    solution = lp.Solve(model, settings)
    cp.cuda.runtime.deviceSynchronize()
    solve_wall_seconds = time.perf_counter() - before
    status = _enum_name(solution.get_termination_status())
    solved_by = _enum_name(solution.get_solved_by())

    before = time.perf_counter()
    try:
        reduced_x = np.asarray(solution.get_primal_solution(), dtype=np.float64)
        reduced_y = np.asarray(solution.get_dual_solution(), dtype=np.float64)
    except BaseException:
        reduced_x = np.empty(0, dtype=np.float64)
        reduced_y = np.empty(0, dtype=np.float64)
    solution_download_seconds = time.perf_counter() - before
    valid_shape = reduced_x.shape == c.shape and reduced_y.shape == rhs.shape

    reduced_rows = []
    original_x_rows = []
    original_y_rows = []
    lift_errors = []
    before = time.perf_counter()
    if valid_shape:
        m, n = reduced_problems[0][0].shape
        for environment_id, (reduction, problem) in enumerate(
            zip(reductions, reduced_problems)
        ):
            z = reduced_x[environment_id * n : (environment_id + 1) * n]
            y = reduced_y[environment_id * m : (environment_id + 1) * m]
            reduced_rows.append(_host_certificate(problem, z, y))
            try:
                reduced_cost = np.asarray(problem[4] - problem[0].T @ y).ravel()
                original_x_rows.append(reduction.expand_primal(z))
                original_y_rows.append(reduction.lift_dual(y, reduced_cost))
                lift_errors.append(None)
            except BaseException as error:
                original_x_rows.append(
                    np.full(reduction.plan.original_variables, np.nan, dtype=np.float64)
                )
                original_y_rows.append(
                    np.full(reduction.plan.original_shape[0], np.nan, dtype=np.float64)
                )
                lift_errors.append(f"{type(error).__name__}: {error}")
    else:
        for reduction in reductions:
            reduced_rows.append(
                {
                    "certificate_passed": False,
                    "reason": "missing_or_malformed_reduced_primal_or_dual",
                }
            )
            original_x_rows.append(
                np.full(reduction.plan.original_variables, np.nan, dtype=np.float64)
            )
            original_y_rows.append(
                np.full(reduction.plan.original_shape[0], np.nan, dtype=np.float64)
            )
            lift_errors.append("missing_or_malformed_reduced_primal_or_dual")
    postsolve_seconds = time.perf_counter() - before

    original_x = np.concatenate(original_x_rows)
    original_y = np.concatenate(original_y_rows)
    cp.cuda.runtime.deviceSynchronize()
    before = time.perf_counter()
    original_device_rows = certify_blocks_device(
        original_problems,
        original_packed,
        original_x,
        original_y,
        cp=cp,
        allow_box_dual=False,
    )
    cp.cuda.runtime.deviceSynchronize()
    original_device_certificate_seconds = time.perf_counter() - before

    before = time.perf_counter()
    original_host_rows = [
        _host_certificate(problem, x, y)
        for problem, x, y in zip(original_problems, original_x_rows, original_y_rows)
    ]
    original_host_certificate_seconds = time.perf_counter() - before

    rows = []
    for environment_id, (
        reduced_certificate,
        device_certificate,
        host_certificate,
        lift_error,
    ) in enumerate(
        zip(reduced_rows, original_device_rows, original_host_rows, lift_errors)
    ):
        parity = {}
        for key in THRESHOLDS:
            host_value = host_certificate.get(key, np.inf)
            device_value = device_certificate.get(key, np.inf)
            parity[key] = (
                float(abs(host_value - device_value))
                if np.isfinite(host_value) and np.isfinite(device_value)
                else None
            )
        certificate_parity = bool(
            host_certificate.get("certificate_passed", False)
            == device_certificate.get("certificate_passed", False)
        )
        accepted = bool(
            status == "optimal"
            and solved_by == "pdlp"
            and lift_error is None
            and certificate_parity
            and device_certificate.get("certificate_passed", False)
        )
        rows.append(
            {
                "environment_id": environment_id,
                "accepted": accepted,
                "lift_error": lift_error,
                "reduced_host_certificate": reduced_certificate,
                "original_device_certificate": device_certificate,
                "original_host_certificate": host_certificate,
                "host_device_certificate_pass_agreement": certificate_parity,
                "host_device_metric_abs_differences": parity,
            }
        )

    stats = solution.get_lp_stats()
    record = {
        "method": "pdlp",
        "mode": mode,
        "presolve": presolve,
        "requested_tolerance": tolerance,
        "time_limit_seconds": time_limit,
        "status": status,
        "solved_by": solved_by,
        "parameters": parameters,
        "reduced_block": {
            "batch": len(reduced_problems),
            "variables": int(a.shape[1]),
            "constraints": int(a.shape[0]),
            "nonzeros": int(a.nnz),
        },
        "valid_solution_shapes": valid_shape,
        "cuopt_stats": stats,
        "cuopt_reported_solve_seconds": float(solution.get_solve_time()),
        "timing": {
            "block_assembly_seconds": block_assembly_seconds,
            "model_and_settings_seconds": model_and_settings_seconds,
            "solve_wall_seconds": solve_wall_seconds,
            "solution_download_seconds": solution_download_seconds,
            "host_exact_postsolve_seconds": postsolve_seconds,
            "original_device_certificate_seconds": original_device_certificate_seconds,
            "original_host_certificate_parity_seconds": original_host_certificate_seconds,
            "pipeline_total_seconds": time.perf_counter() - pipeline_started,
        },
        "rows": rows,
        "accepted": sum(row["accepted"] for row in rows),
        "requested": len(rows),
        "all_original_certificates_passed": all(row["accepted"] for row in rows),
        "cpu_lp_calls_in_gpu_path": 0,
        "scope": SCOPE,
    }
    return record, original_x_rows


def run_probe(
    trace: Path,
    *,
    step: int = 1,
    environments: int = 4,
    workers: int = 4,
    time_limit: float = 3.0,
    tolerance: float = 1e-9,
    presolve: int = 2,
    mode: str = "stable3",
):
    if (
        not isinstance(step, int)
        or step < 1
        or not isinstance(environments, int)
        or environments < 1
        or not isinstance(workers, int)
        or workers < 1
        or not np.isfinite(time_limit)
        or time_limit <= 0
        or not np.isfinite(tolerance)
        or not 0 < tolerance <= 1e-7
        or presolve not in (0, 1, 2)
        or mode not in {"stable3", "methodical1", "stable2", "fast1"}
    ):
        raise ValueError("Invalid cohort or solver configuration")

    trace = Path(trace)
    manifest_path = trace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "completed"
        or manifest.get("role") != "development_diagnostic_not_training"
    ):
        raise ValueError("A completed development-only LP trace is required")
    seeds = manifest.get("seeds", [])
    if environments > len(seeds) or len(set(seeds[:environments])) != environments:
        raise ValueError("The requested environment IDs need unique trace seeds")
    entries = _selected_entries(manifest, step=step, environments=environments)

    load_started = time.perf_counter()
    loaded = [load_trace_lp(trace, entry) for entry in entries]
    trace_load_and_reference_validation_seconds = time.perf_counter() - load_started
    problems = [item[0] for item in loaded]
    references = [item[1:] for item in loaded]
    requests = [problem_request(problem, stage="exchange") for problem in problems]

    plan_started = time.perf_counter()
    plan = HomogeneousEqualityReduction.from_problem(problems[0])
    equality_plan_setup_seconds = time.perf_counter() - plan_started
    reduction_started = time.perf_counter()
    reductions = [plan.reduce(problem) for problem in problems]
    dynamic_equality_reduction_seconds = time.perf_counter() - reduction_started

    cpu_results, cpu_record = _cpu_cold_reference(requests, workers=workers)

    library_started = time.perf_counter()
    import cupy as cp
    import cuopt
    from cuopt import linear_programming as lp

    cp.cuda.Device().use()
    cp.cuda.runtime.free(0)
    cp.cuda.runtime.deviceSynchronize()
    gpu_library_and_context_setup_seconds = time.perf_counter() - library_started
    gpu_record, gpu_original_x = _solve_reduced_cuopt(
        problems,
        reductions,
        cp=cp,
        lp=lp,
        time_limit=time_limit,
        tolerance=tolerance,
        presolve=presolve,
        mode=mode,
    )

    scoring_rows = []
    for environment_id, (problem, reference, cpu_result, candidate_x) in enumerate(
        zip(problems, references, cpu_results, gpu_original_x)
    ):
        reference_x, reference_y = reference
        reference_certificate = _host_certificate(problem, reference_x, reference_y)
        if not reference_certificate["certificate_passed"]:
            raise ValueError("A scoring-only trace reference lost its original certificate")
        candidate_objective = (
            float(problem[4] @ candidate_x)
            if np.isfinite(candidate_x).all()
            else None
        )
        reference_objective = float(problem[4] @ reference_x)
        scoring_rows.append(
            {
                "environment_id": environment_id,
                "seed": int(seeds[environment_id]),
                "problem_sha256": entries[environment_id]["problem_sha256"],
                "reference_role": "scoring_only_never_solver_input",
                "reference_original_certificate": reference_certificate,
                "reference_objective": reference_objective,
                "cold_cpu_objective": float(cpu_result.fun),
                "cold_cpu_objective_abs_error_from_trace_reference": float(
                    abs(cpu_result.fun - reference_objective)
                ),
                "gpu_candidate_objective": candidate_objective,
                "gpu_candidate_objective_abs_error_from_trace_reference": (
                    float(abs(candidate_objective - reference_objective))
                    if candidate_objective is not None
                    else None
                ),
            }
        )

    original_shape = problems[0][0].shape
    reduced_shape = reductions[0].problem[0].shape
    report = {
        "status": "completed",
        "scope": SCOPE,
        "configuration": {
            "trace": str(trace),
            "stage": "exchange",
            "step": step,
            "environments": environments,
            "workers": workers,
            "time_limit_seconds": time_limit,
            "tolerance": tolerance,
            "presolve": presolve,
            "pdlp_mode": mode,
            "warm_start": None,
            "execution_order": "cold_cpu_first_then_cold_gpu",
        },
        "certificate_thresholds": dict(THRESHOLDS),
        "trace_manifest_sha256": _sha256(manifest_path),
        "trace_role": manifest.get("role"),
        "evaluation_seeds": seeds[:environments],
        "model_fingerprints": manifest.get("model_fingerprints"),
        "selected_entries": [
            {
                "step": entry["step"],
                "environment_id": entry["environment_id"],
                "filename": entry["filename"],
                "file_sha256": entry["sha256"],
                "problem_sha256": entry["problem_sha256"],
            }
            for entry in entries
        ],
        "reduction": {
            "equality_fingerprint": plan.equality_fingerprint,
            "original_rows": original_shape[0],
            "original_variables": original_shape[1],
            "original_nonzeros_per_lp": int(problems[0][0].nnz),
            "reduced_rows": reduced_shape[0],
            "reduced_variables": reduced_shape[1],
            "reduced_nonzeros_per_lp": int(reductions[0].problem[0].nnz),
            "eliminated_equalities": len(plan.eliminated_rows),
            "transform_nonzeros": int(plan.T.nnz),
            "dual_lift_nonzeros": int(plan.dual_lift_map.nnz),
            "equality_plan_setup_seconds_excluded_as_fixed_offline_work": equality_plan_setup_seconds,
            "dynamic_equality_reduction_seconds_included_in_gpu_online_total": dynamic_equality_reduction_seconds,
        },
        "timing_scope": {
            "trace_load_and_reference_validation_seconds_excluded": trace_load_and_reference_validation_seconds,
            "gpu_library_and_context_setup_seconds_excluded": gpu_library_and_context_setup_seconds,
            "fixed_equality_plan_setup_excluded": equality_plan_setup_seconds,
            "dynamic_reduction_in_gpu_pipeline": True,
            "gpu_model_build_transfer_solve_download_lift_and_certificate_in_gpu_pipeline": True,
            "cpu_backend_constructor_and_cleanup_reported_but_excluded_from_cpu_solve_batch_wall": True,
            "inference": "Descriptive cold same-stage timing only; no trajectory or purchase speedup claim.",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cuopt": cuopt.__version__,
            "gpu": _device_provenance(cp),
        },
        "cpu": cpu_record,
        "gpu": gpu_record,
        "scoring": scoring_rows,
        "methodological_sources": PRIMARY_SOURCES,
    }
    # Dynamic reduction is measured immediately before CPU so it cannot be
    # nested in _solve_reduced_cuopt's timer; add it exactly once here.
    gpu_online = (
        dynamic_equality_reduction_seconds
        + gpu_record["timing"]["pipeline_total_seconds"]
    )
    report["summary"] = {
        "cold_cpu_original_certified": environments,
        "gpu_original_certified": int(gpu_record["accepted"]),
        "requested": environments,
        "gpu_online_total_seconds_including_dynamic_reduction": gpu_online,
        "cold_cpu_solve_batch_wall_seconds": cpu_record["solve_batch_wall_seconds"],
        "qualified_cold_cpu_over_gpu_ratio": (
            float(cpu_record["solve_batch_wall_seconds"] / gpu_online)
            if gpu_record["all_original_certificates_passed"]
            else None
        ),
        "qualified_for_speed_integration": bool(
            gpu_record["all_original_certificates_passed"]
            and gpu_online < cpu_record["solve_batch_wall_seconds"]
        ),
        "interpretation": (
            "A speed ratio is emitted only after every row passes the unchanged "
            "original certificate. Failure or timeout is evidence against this "
            "configuration, not permission to relax the gate."
        ),
    }
    source_paths = (
        "scripts/benchmark_reduced_cuopt_pdlp.py",
        "src/lp_equality_reduction.py",
        "src/gpu_block_lp.py",
        "src/lp_trace.py",
        "src/cpu_repeated_lp.py",
    )
    report["source_hashes"] = {name: _sha256(ROOT / name) for name in source_paths}
    return _finite(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("results/pf_lp_trace_dev4x60_20260905"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--environments", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--presolve", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--mode",
        choices=("stable3", "methodical1", "stable2", "fast1"),
        default="stable3",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    phase = "initialization"
    started = time.perf_counter()
    try:
        phase = "probe"
        report = run_probe(
            args.trace,
            step=args.step,
            environments=args.environments,
            workers=args.workers,
            time_limit=args.time_limit,
            tolerance=args.tolerance,
            presolve=args.presolve,
            mode=args.mode,
        )
        report["diagnostic_wall_seconds"] = time.perf_counter() - started
    except BaseException as error:
        report = {
            "status": "failed",
            "scope": SCOPE,
            "error_phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "diagnostic_wall_seconds": time.perf_counter() - started,
        }
        with args.output.open("x") as handle:
            json.dump(_finite(report), handle, indent=2, allow_nan=False)
        raise
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
