"""Benchmark separate versus joint 3-GEM FBA on identical dFBA inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dfba_simulator import dFBASimulator
from src.gpu_assignment import assign_gpu_for_worker
from src.utils import get_initial_params, load_sbml_models, select_consortium_models


def run_case(sbml_dir: str, backend: str, fba_mode: str, steps: int, gpu_id: str | None):
    if gpu_id is not None:
        assign_gpu_for_worker(0, gpu_id)
    models = select_consortium_models(load_sbml_models(Path(sbml_dir)))
    initial_biomass, initial_metabolites = get_initial_params(models)
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        dt=0.2,
        solver_backend=backend,
        cuopt_method="pdlp",
        fba_mode=fba_mode,
    )
    started = time.perf_counter()
    for _ in range(steps):
        # Fixed inputs make CPU/GPU and separate/joint runs directly comparable.
        simulator.step(
            rubber_degradation_rates={
                "OR16": 0.01,
                "NS21": 0.01,
                "LP": 0.005,
            },
            nutrient_supplementation={
                "sn_or16": 0.02,
                "sn_ns21": 0.02,
                "sn_lp": 0.01,
            },
        )
    elapsed = time.perf_counter() - started
    final_state = simulator.get_state_vector().astype(float).tolist()
    growth = {
        name: float(state.growth_rate)
        for name, state in simulator.state.species.items()
    }
    stats = None
    if simulator._joint_solver is not None:
        stats = {
            "backend": simulator._joint_solver.stats.backend,
            "status": simulator._joint_solver.stats.status,
            "method": simulator._joint_solver.stats.method,
            "last_solve_seconds": simulator._joint_solver.stats.solve_seconds,
            "matrix_rows": int(simulator._joint_solver.a_eq.shape[0]),
            "matrix_cols": int(simulator._joint_solver.a_eq.shape[1]),
            "matrix_nnz": int(simulator._joint_solver.a_eq.nnz),
        }
    return {
        "backend_requested": backend,
        "fba_mode": fba_mode,
        "assigned_gpu": simulator.assigned_gpu,
        "assigned_gpu_slot": simulator.assigned_gpu_slot,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(steps, 1),
        "steps_per_second": steps / max(elapsed, 1e-12),
        "growth_rates": growth,
        "final_state": final_state,
        "joint_solver": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sbml-dir",
        default="models/sbml/final_consortium",
        help="directory containing the three final GEM SBML files",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument(
        "--cases",
        choices=["all", "cpu", "gpu"],
        default="all",
        help="run all cases, CPU-only cases, or only the cuOpt case",
    )
    parser.add_argument(
        "--output", default="results/benchmark_joint_fba.json"
    )
    args = parser.parse_args()

    cases = [("glpk", "separate"), ("glpk", "joint")]
    if args.cases in {"all", "gpu"}:
        cases.append(("cuopt", "joint"))
    if args.cases == "gpu":
        cases = [("cuopt", "joint")]
    results = {}
    for backend, fba_mode in cases:
        key = f"{fba_mode}_{backend}"
        try:
            results[key] = run_case(
                args.sbml_dir, backend, fba_mode, args.steps, args.gpu_id
            )
        except Exception as exc:
            results[key] = {
                "backend_requested": backend,
                "fba_mode": fba_mode,
                "error": f"{type(exc).__name__}: {exc}",
            }

    baseline = results.get("separate_glpk", {}).get("final_state")
    if baseline is not None:
        for value in results.values():
            state = value.get("final_state")
            if state is not None:
                value["max_abs_state_diff_vs_separate_glpk"] = float(
                    np.max(np.abs(np.asarray(state) - np.asarray(baseline)))
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
