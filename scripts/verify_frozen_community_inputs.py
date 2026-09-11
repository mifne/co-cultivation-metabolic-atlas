"""Compare live-COBRA and frozen-array cooperative LP inputs and host profiles.

This is a diagnostic equivalence run, not a CPU/GPU performance benchmark.
The profiler is disabled while the exact LP service and request hashing run.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from greenlet import getcurrent

from scripts.benchmark_basis_bank_rollout import environment, snapshot
from scripts.benchmark_compact_gpu import json_finite_values, matched_actions
from scripts.microbatch_comparison_support import drive_microbatch
from src.cpu_repeated_lp import RepeatedCpuLP, _problem


def _request_digest(request):
    objective, kwargs = request
    a, rhs, lower, upper, c, neq = _problem(objective, kwargs)
    digest = hashlib.sha256()
    for value in (
        np.asarray(a.shape, dtype=np.int64),
        a.data,
        a.indices,
        a.indptr,
        rhs,
        lower,
        upper,
        c,
        np.asarray([neq], dtype=np.int64),
    ):
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _case(sample, seeds, actions, workers, frozen, profile_path):
    environments = []
    for seed in seeds:
        env = copy.deepcopy(sample)
        env.reset(seed=seed)
        simulator = env.simulator
        if frozen:
            simulator.enable_frozen_community_inputs()
        model = simulator._cooperative_solver
        model._linprog_options.update(threads=1, parallel=False)
        environments.append((env, model))

    service = RepeatedCpuLP(workers, n_fluxes=environments[0][1].n_fluxes)
    original = service.solve_batch
    profiler = cProfile.Profile()
    service_seconds = []
    request_hashes = []

    def solve(requests):
        profiler.disable()
        started = time.perf_counter()
        try:
            request_hashes.append([_request_digest(request) for request in requests])
            return original(requests)
        finally:
            service_seconds.append(time.perf_counter() - started)
            profiler.enable()

    service.solve_batch = solve
    progress = [0] * len(environments)
    parent = getcurrent()
    started = time.perf_counter()
    try:
        with patch(
            "src.community_solver.linprog",
            lambda objective, **kwargs: parent.switch((objective, kwargs)),
        ):
            profiler.enable()
            endpoints = drive_microbatch(
                environments, actions, service, snapshot, progress
            )
    finally:
        profiler.disable()
        wall = time.perf_counter() - started
        service.close()
    profiler.dump_stats(str(profile_path))
    stats = pstats.Stats(profiler)
    functions = [
        dict(
            file=file,
            line=line,
            function=name,
            primitive_calls=primitive_calls,
            calls=calls,
            self_seconds=self_seconds,
            cumulative_seconds=cumulative_seconds,
        )
        for (file, line, name), (
            primitive_calls,
            calls,
            self_seconds,
            cumulative_seconds,
            _callers,
        ) in stats.stats.items()
    ]
    functions.sort(key=lambda row: row["self_seconds"], reverse=True)
    return dict(
        mode="frozen_array" if frozen else "live_cobra",
        wall_seconds=wall,
        lp_service_seconds=sum(service_seconds),
        profiled_host_wall_seconds=wall - sum(service_seconds),
        completed_steps=progress,
        request_hashes=request_hashes,
        endpoints=endpoints,
        cpu_history=service.history,
        profile_path=str(profile_path),
        functions=functions,
    )


def _endpoint_error(left, right):
    values = [abs(float(left["pha"]) - float(right["pha"]))]
    values.append(abs(float(left["phv_fraction"]) - float(right["phv_fraction"])))
    values.extend(
        abs(float(left["biomass"][key]) - float(right["biomass"][key]))
        for key in left["biomass"]
    )
    values.extend(
        abs(float(left["metabolites"].get(key, 0.0)) - float(right["metabolites"].get(key, 0.0)))
        for key in set(left["metabolites"]) | set(right["metabolites"])
    )
    return max(values, default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environments", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20293301)
    parser.add_argument("--cpu-workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.environments, args.steps, args.cpu_workers) < 1:
        raise ValueError("Positive environments, steps and workers are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    live_profile = args.output.with_suffix(".live.pstats")
    frozen_profile = args.output.with_suffix(".frozen.pstats")
    if live_profile.exists() or frozen_profile.exists():
        raise FileExistsError("Profile output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sample, _layout = environment(args.seed)
    seeds = list(range(args.seed, args.seed + args.environments))
    actions = matched_actions(seeds, args.steps)
    live = _case(
        sample, seeds, actions, args.cpu_workers, False, live_profile
    )
    frozen = _case(
        sample, seeds, actions, args.cpu_workers, True, frozen_profile
    )
    hashes_equal = live["request_hashes"] == frozen["request_hashes"]
    endpoint_maximum = max(
        (_endpoint_error(a, b) for a, b in zip(live["endpoints"], frozen["endpoints"])),
        default=0.0,
    )
    report = dict(
        status=(
            "completed_equivalent"
            if hashes_equal and endpoint_maximum == 0.0
            else "completed_difference"
        ),
        scope=(
            "CPU-only diagnostic; cProfile disabled in LP service; timings are "
            "not a CPU/GPU benchmark or a general speed claim"
        ),
        configuration={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        seeds=seeds,
        original_lp_request_hashes_equal=hashes_equal,
        endpoint_absolute_maximum=endpoint_maximum,
        live=live,
        frozen=frozen,
        source_hashes={
            filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            for filename in (
                "src/frozen_community_inputs.py",
                "src/dfba_simulator.py",
                "src/community_solver.py",
                "scripts/verify_frozen_community_inputs.py",
            )
        },
    )
    args.output.write_text(
        json.dumps(json_finite_values(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "original_lp_request_hashes_equal": hashes_equal,
                "endpoint_absolute_maximum": endpoint_maximum,
                "live_profiled_host_wall_seconds": live[
                    "profiled_host_wall_seconds"
                ],
                "frozen_profiled_host_wall_seconds": frozen[
                    "profiled_host_wall_seconds"
                ],
            },
            indent=2,
        )
    )
    if not hashes_equal or endpoint_maximum != 0.0:
        raise RuntimeError("Frozen/live equivalence failed")


if __name__ == "__main__":
    main()
