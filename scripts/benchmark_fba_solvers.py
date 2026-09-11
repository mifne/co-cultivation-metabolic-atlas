"""End-to-end CPU/GPU FBA benchmark.

Run this script once per backend on the same machine and compare the resulting
JSON files. It measures the full simulator path, including Python state updates
and model/host transfer overhead, rather than cuOpt's internal solve time only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.gpu_assignment import assign_gpu_for_worker


def run(backend: str, steps: int, method: str, gpu_id: str | None = None) -> dict:
    assigned_gpu = assign_gpu_for_worker(0, gpu_id)
    model_dir = Path("models/sbml/final_consortium")
    models = select_consortium_models(load_sbml_models(model_dir))
    biomass, metabolites = get_initial_params(models)
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2,
        solver_backend=backend,
        cuopt_method=method,
    )
    env = ConsortiumEnv(simulator=simulator, max_time=max(steps * 0.2, 0.2))
    observation, _ = env.reset(seed=0)
    action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
    start = time.perf_counter()
    completed = 0
    for _ in range(steps):
        observation, _, terminated, truncated, _ = env.step(action)
        completed += 1
        if terminated or truncated:
            observation, _ = env.reset(seed=0)
    elapsed = time.perf_counter() - start
    return {
        "backend_requested": backend,
        "backend_used": simulator.solver_backend,
        "assigned_gpu": simulator.assigned_gpu or assigned_gpu,
        "cuopt_method": method,
        "steps": completed,
        "elapsed_seconds": elapsed,
        "steps_per_second": completed / elapsed if elapsed else 0.0,
        "seconds_per_step": elapsed / completed if completed else 0.0,
        "solver_diagnostics": simulator.get_solver_diagnostics(),
        "final_state_vector": simulator.get_state_vector().astype(float).tolist(),
        "final_growth_rates": {
            name: float(state.growth_rate)
            for name, state in simulator.state.species.items()
        },
        "final_rubber_concentration": float(
            simulator.state.rubber_concentration
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["glpk", "cuopt", "auto"], required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cuopt-method", choices=["barrier", "pdlp", "concurrent", "dual simplex"], default="pdlp")
    parser.add_argument("--gpu-id", type=str, default=None, help="GPU ID visible to this benchmark process")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.backend, args.steps, args.cuopt_method, args.gpu_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
