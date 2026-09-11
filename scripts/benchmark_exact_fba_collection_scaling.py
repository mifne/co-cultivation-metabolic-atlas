#!/usr/bin/env python3
"""Benchmark CPU scaling of exact three-GEM FBA label collection.

This measures a second important workstation workload: producing the exact
HiGHS labels used to build and audit the GPU candidate library.  Every sample
advances one consortium environment step and solves all three GEMs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

from train_fba_surrogate import collect_exact_dataset_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbml-dir", type=Path, default=Path("models/sbml"))
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exact_fba_collection_scaling_current_cpu.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for worker_count in args.workers:
        started = time.perf_counter()
        dataset = collect_exact_dataset_parallel(
            sbml_dir=args.sbml_dir,
            samples=args.samples,
            episode_steps=args.episode_steps,
            seed=args.seed,
            workers=worker_count,
        )
        elapsed = time.perf_counter() - started
        species_rows = {
            species: int(values["features"].shape[0])
            for species, values in dataset.items()
        }
        result = {
            "workers": int(worker_count),
            "environment_steps": int(args.samples),
            "exact_lp_solves": int(sum(species_rows.values())),
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": args.samples / elapsed,
            "exact_lp_solves_per_second": sum(species_rows.values()) / elapsed,
            "species_rows": species_rows,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    mem_total_kib = 0
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                mem_total_kib = int(line.split()[1])
                break
    payload = {
        "benchmark": "exact_three_gem_fba_label_collection_scaling",
        "scope": (
            "One environment step solves all three consortium GEMs with the "
            "exact HiGHS backend; process workers collect independent shards."
        ),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_ram_gib_visible_to_wsl": mem_total_kib / 2**20,
        "samples_per_measurement": args.samples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
