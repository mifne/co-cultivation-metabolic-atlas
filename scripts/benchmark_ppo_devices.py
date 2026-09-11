"""Compare PPO rollout/training on CPU versus CUDA with parallel dFBA envs.

The environment workers remain separate CPU processes (COBRApy/GLPK), while
the PPO policy/value network is placed on the requested device. This isolates
the practical question: whether moving a small MLP to the RTX 4060 helps when
parallel environments are the dominant workload.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import make_env
from src.ppo_agent import ConsortiumPPOAgent


def run(
    device: str,
    n_envs: int,
    timesteps: int,
    solver_backend: str,
    gpu_ids: str | None = None,
    fba_mode: str = "separate",
    gpu_slots_per_device: int = 1,
) -> dict:
    env_params = {
        "max_time": 32.0,
        "solver_backend": solver_backend,
        "gpu_ids": gpu_ids,
        "cuopt_method": "pdlp",
        "fba_mode": fba_mode,
        "gpu_slots_per_device": gpu_slots_per_device,
    }
    env_fns = [
        make_env(
            "models/sbml/final_consortium",
            env_params,
            data_log_path=None,
            rank=rank,
        )
        for rank in range(n_envs)
    ]
    agent = ConsortiumPPOAgent(
        env=env_fns,
        n_steps=min(32, max(8, timesteps // n_envs)),
        batch_size=32,
        n_epochs=1,
        device=device,
        verbose=0,
        n_envs=n_envs,
    )
    actual_device = str(agent.model.device)
    start = time.perf_counter()
    agent.model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.perf_counter() - start
    agent.env.close()
    return {
        "device_requested": device,
        "device_used": actual_device,
        "n_envs": n_envs,
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "timesteps_per_second": timesteps / elapsed if elapsed else 0.0,
        "seconds_per_timestep": elapsed / timesteps if timesteps else 0.0,
        "solver_backend": solver_backend,
        "fba_mode": fba_mode,
        "gpu_ids_requested": gpu_ids,
        "gpu_slots_per_device": gpu_slots_per_device,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], required=True)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--solver-backend", choices=["glpk", "cuopt", "auto"], default="glpk")
    parser.add_argument("--fba-mode", choices=["separate", "joint"], default="separate")
    parser.add_argument("--gpu-ids", type=str, default=None, help="worker GPU IDs, e.g. 0,1,2")
    parser.add_argument("--gpu-slots-per-device", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(
        args.device,
        args.n_envs,
        args.timesteps,
        args.solver_backend,
        args.gpu_ids,
        args.fba_mode,
        args.gpu_slots_per_device,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
