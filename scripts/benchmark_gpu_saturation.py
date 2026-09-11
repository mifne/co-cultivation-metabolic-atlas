#!/usr/bin/env python3
"""Measure sustained utilization of the real 3-GEM CUDA solve stage.

Unlike a synthetic stress test, every iteration executes the production
candidate optimizer and bound/mass-balance guards for all three GEMs.  COBRA
Solution construction is intentionally left to environment workers, matching
the production service architecture.  Environment state updates are excluded
to show the GPU-stage ceiling and the batch size required to reach it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_resource_usage import ResourceSampler, summarize_resource
from src.fba_surrogate import FbaSurrogateBackend, safe_species_filename
from src.utils import load_sbml_models, select_consortium_models


def run_batch(backend, archive, batch_size: int, duration: float, interval: float) -> dict:
    feature_batches = {}
    for species in backend.species:
        features = archive[f"{safe_species_filename(species)}__features"]
        indices = np.arange(batch_size) % len(features)
        feature_batches[species] = np.asarray(features[indices], dtype=np.float32)

    # Warm every species and preallocate CUDA kernels before resource sampling.
    with ThreadPoolExecutor(max_workers=len(feature_batches)) as executor:
        list(
            executor.map(
                lambda item: backend.predict_feature_payload_batch(*item),
                feature_batches.items(),
            )
        )
    torch.cuda.synchronize()

    sampler = ResourceSampler(interval, f"batch_{batch_size}")
    sampler.start()
    started = time.perf_counter()
    batches = 0
    predictions = 0
    accepted = 0
    with ThreadPoolExecutor(max_workers=len(feature_batches)) as executor:
        while time.perf_counter() - started < duration:
            result_groups = list(
                executor.map(
                    lambda item: backend.predict_feature_payload_batch(*item),
                    feature_batches.items(),
                )
            )
            for results in result_groups:
                predictions += len(results)
                accepted += sum(bool(item["accepted"]) for item in results)
            batches += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    sampler.stop()
    return {
        "batch_size_per_species": batch_size,
        "gem_batches": batches,
        "predictions": predictions,
        "accepted": accepted,
        "acceptance_rate": accepted / predictions if predictions else 0.0,
        "elapsed_seconds": elapsed,
        "predictions_per_second": predictions / elapsed,
        "resource": summarize_resource(sampler.rows),
        "resource_samples": sampler.rows,
    }


def make_plot(report: dict, output: Path) -> None:
    rows = report["rows"]
    batches = [row["batch_size_per_species"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(batches, [row["resource"]["gpu_utilization_percent"]["mean"] for row in rows], "o-", label="Mean")
    axes[0].plot(batches, [row["resource"]["gpu_utilization_percent"]["p95"] for row in rows], "s--", label="P95")
    axes[0].plot(batches, [row["resource"]["gpu_utilization_percent"]["max"] for row in rows], "^:", label="Max")
    axes[0].axhline(100, color="gray", linewidth=1)
    axes[0].set_title("Sustained NVIDIA GPU utilization")
    axes[0].set_ylabel("Percent (%)")
    axes[0].legend()
    axes[1].plot(batches, [row["resource"]["gpu_memory_used_mib"]["max"] for row in rows], "o-")
    axes[1].set_title("Peak VRAM")
    axes[1].set_ylabel("MiB")
    axes[2].plot(batches, [row["predictions_per_second"] for row in rows], "o-")
    axes[2].set_title("Guarded 3-GEM solve throughput")
    axes[2].set_ylabel("Predictions / second")
    for axis in axes:
        axis.set_xlabel("Concurrent environments / GEM batch")
        axis.set_xticks(batches)
        axis.grid(alpha=0.25)
    fig.suptitle(f"GPU-stage saturation: {report['gpu_name']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "models" / "fba_surrogate_gpu_lp_2048")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "gpu_saturation_2048_rtx4060.json")
    parser.add_argument("--plot", type=Path, default=PROJECT_ROOT / "results" / "gpu_saturation_2048_rtx4060.png")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.dataset is None:
        args.dataset = args.artifact_dir / "exact_training_data.npz"
    models = select_consortium_models(load_sbml_models(PROJECT_ROOT / "models" / "sbml"))
    backend = FbaSurrogateBackend(models, args.artifact_dir, device="cuda")
    archive = np.load(args.dataset)
    report = {
        "measurement_type": "production 3-GEM CUDA candidate solve + physical guards; environment update excluded",
        "gpu_name": torch.cuda.get_device_name(0),
        "artifact_dir": str(args.artifact_dir),
        "duration_per_batch_seconds": args.seconds,
        "sample_interval_seconds": args.interval,
        "rows": [],
    }
    for batch_size in args.batch_sizes:
        row = run_batch(backend, archive, batch_size, args.seconds, args.interval)
        report["rows"].append(row)
        utilization = row["resource"]["gpu_utilization_percent"]
        print(
            f"batch={batch_size}: mean={utilization['mean']:.1f}%, "
            f"p95={utilization['p95']:.1f}%, max={utilization['max']:.1f}%, "
            f"throughput={row['predictions_per_second']:.1f}/s"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(report, args.plot)
    print(f"saved: {args.output}\nsaved: {args.plot}")


if __name__ == "__main__":
    main()
