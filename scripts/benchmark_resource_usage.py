#!/usr/bin/env python3
"""Short CPU/GPU rollout with system-resource sampling.

The benchmark intentionally measures steady-state rollout work (after model and
CUDA warm-up).  It records GPU metrics through nvidia-smi and CPU/process
metrics through psutil, so the resulting data can be cited as a short-run
measurement rather than a hardware specification.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_rollout_cpu_gpu import make_environment, synchronize_cuda


def query_gpus() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
        "utilization.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=2)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_used_mib": number(fields[2]),
                "memory_total_mib": number(fields[3]),
                "gpu_utilization_percent": number(fields[4]),
                "memory_utilization_percent": number(fields[5]),
                "power_w": number(fields[6]),
                "temperature_c": number(fields[7]),
            }
        )
    return rows


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.2, label: str = "") -> None:
        self.interval_seconds = interval_seconds
        self.label = label
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._root = psutil.Process(os.getpid())

    def _process_tree(self) -> list[psutil.Process]:
        processes = [self._root]
        try:
            processes.extend(self._root.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return processes

    def _sample(self) -> dict:
        cpu_values = []
        rss_bytes = 0
        pss_bytes = 0
        for process in self._process_tree():
            try:
                cpu_values.append(process.cpu_percent(None))
                memory = process.memory_info()
                rss_bytes += memory.rss
                try:
                    full_memory = process.memory_full_info()
                    pss_bytes += getattr(full_memory, "pss", memory.rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied, NotImplementedError):
                    pss_bytes += memory.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        system_memory = psutil.virtual_memory()
        return {
            "elapsed_seconds": time.perf_counter() - self._started,
            "label": self.label,
            "system_cpu_percent": psutil.cpu_percent(None),
            "process_cpu_percent_sum": float(sum(cpu_values)),
            "process_rss_mib": rss_bytes / (1024 * 1024),
            "process_pss_mib": pss_bytes / (1024 * 1024),
            "system_memory_used_mib": system_memory.used / (1024 * 1024),
            "system_memory_available_mib": system_memory.available / (1024 * 1024),
            "system_memory_total_mib": system_memory.total / (1024 * 1024),
            "system_memory_percent": float(system_memory.percent),
            "gpus": query_gpus(),
        }

    def _run(self) -> None:
        # Prime psutil counters before the first reported sample.
        psutil.cpu_percent(None)
        for process in self._process_tree():
            try:
                process.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        while not self._stop.is_set():
            self.rows.append(self._sample())
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))


def run_once(
    sbml_dir: Path,
    artifact_dir: Path,
    backend: str,
    actions: np.ndarray,
    interval_seconds: float,
) -> tuple[dict, list[dict]]:
    env = make_environment(sbml_dir, backend, artifact_dir)
    env.reset(seed=0)
    # Exclude model loading and first-use CUDA setup from measured work.
    env.step(actions[0])
    env.reset(seed=0)
    synchronize_cuda()
    sampler = ResourceSampler(interval_seconds, backend)
    sampler.start()
    started = time.perf_counter()
    steps = 0
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    synchronize_cuda()
    elapsed = time.perf_counter() - started
    sampler.stop()
    diagnostics = env.simulator.get_solver_diagnostics()
    env.close()
    return (
        {
            "backend": backend,
            "steps": steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / elapsed if elapsed else 0.0,
            "diagnostics": diagnostics,
        },
        sampler.rows,
    )


def metric_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        for gpu in row["gpus"]:
            value = gpu.get(key)
            if value is not None:
                values.append(float(value))
    return values


def summarize_resource(rows: list[dict]) -> dict:
    gpu_util = metric_values(rows, "gpu_utilization_percent")
    vram = metric_values(rows, "memory_used_mib")
    memory_util = metric_values(rows, "memory_utilization_percent")
    cpu = [float(row["system_cpu_percent"]) for row in rows]
    process_cpu = [float(row["process_cpu_percent_sum"]) for row in rows]
    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "mean": None, "max": None, "p95": None}
        return {
            "n": len(values),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
        }
    return {
        "samples": len(rows),
        "gpu_utilization_percent": stats(gpu_util),
        "gpu_memory_used_mib": stats(vram),
        "gpu_memory_utilization_percent": stats(memory_util),
        "system_cpu_percent": stats(cpu),
        "process_cpu_percent_sum": stats(process_cpu),
        "process_rss_mib": stats([float(row["process_rss_mib"]) for row in rows]),
        "process_pss_mib": stats(
            [float(row["process_pss_mib"]) for row in rows if "process_pss_mib" in row]
        ),
        "system_memory_used_mib": stats(
            [float(row["system_memory_used_mib"]) for row in rows if "system_memory_used_mib" in row]
        ),
        "system_memory_available_mib": stats(
            [float(row["system_memory_available_mib"]) for row in rows if "system_memory_available_mib" in row]
        ),
        "system_memory_total_mib": stats(
            [float(row["system_memory_total_mib"]) for row in rows if "system_memory_total_mib" in row]
        ),
        "system_memory_percent": stats(
            [float(row["system_memory_percent"]) for row in rows if "system_memory_percent" in row]
        ),
    }


def make_plot(report: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    colors = {"highs": "#2563eb", "surrogate_cuda": "#dc2626"}
    labels = {"highs": "CPU / HiGHS", "surrogate_cuda": "CUDA surrogate"}
    for backend, color in colors.items():
        runs = report["runs"].get(backend, [])
        for run in runs:
            rows = run["resource_samples"]
            x = [row["elapsed_seconds"] for row in rows]
            gpu = [
                (row["gpus"][0].get("gpu_utilization_percent") if row["gpus"] else np.nan)
                for row in rows
            ]
            vram = [
                (row["gpus"][0].get("memory_used_mib") if row["gpus"] else np.nan)
                for row in rows
            ]
            system_cpu = [row["system_cpu_percent"] for row in rows]
            process_cpu = [row["process_cpu_percent_sum"] for row in rows]
            axes[0, 0].plot(x, gpu, color=color, alpha=0.55, label=labels[backend])
            axes[0, 1].plot(x, vram, color=color, alpha=0.55, label=labels[backend])
            axes[1, 0].plot(x, system_cpu, color=color, alpha=0.35, linestyle="--")
            axes[1, 0].plot(x, process_cpu, color=color, alpha=0.65)
    axes[0, 0].set_title("GPU utilization")
    axes[0, 0].set_ylabel("Percent (%)")
    axes[0, 1].set_title("VRAM used")
    axes[0, 1].set_ylabel("MiB")
    axes[1, 0].set_title("CPU utilization")
    axes[1, 0].set_ylabel("Percent (%)")
    axes[1, 0].set_xlabel("Measured time (s)")
    axes[0, 0].set_xlabel("Measured time (s)")
    axes[0, 1].set_xlabel("Measured time (s)")
    backends = ["highs", "surrogate_cuda"]
    means = [report["summary"][b]["steps_per_second_mean"] for b in backends]
    bars = axes[1, 1].bar([labels[b] for b in backends], means, color=[colors[b] for b in backends])
    axes[1, 1].set_title("Rollout throughput")
    axes[1, 1].set_ylabel("Steps / second")
    for bar, value in zip(bars, means):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    axes[0, 0].legend(loc="best")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("Short-run resource profile: CPU vs CUDA", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "resource_usage_short_rtx4060.json")
    parser.add_argument("--plot", type=Path, default=PROJECT_ROOT / "results" / "resource_usage_short_rtx4060.png")
    args = parser.parse_args()
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA is required for the CPU/CUDA comparison")
    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(0.05, 0.95, size=(args.steps, 5)).astype(np.float32)
    sbml_dir = PROJECT_ROOT / "models" / "sbml"
    artifact_dir = PROJECT_ROOT / "models" / "fba_surrogate_gpu_lp"
    raw: dict[str, list[dict]] = {"highs": [], "surrogate_cuda": []}
    resource_rows: dict[str, list[list[dict]]] = {"highs": [], "surrogate_cuda": []}
    for _ in range(args.repeats):
        for backend in ("highs", "surrogate"):
            key = "highs" if backend == "highs" else "surrogate_cuda"
            result, samples = run_once(sbml_dir, artifact_dir, backend, actions, args.interval)
            raw[key].append(result)
            resource_rows[key].append(samples)
    summary = {}
    for key in raw:
        summary[key] = {
            "steps_per_second_mean": float(np.mean([row["steps_per_second"] for row in raw[key]])),
            "elapsed_seconds_mean": float(np.mean([row["elapsed_seconds"] for row in raw[key]])),
            "resource": summarize_resource([sample for run in resource_rows[key] for sample in run]),
        }
    report = {
        "measurement_type": "short steady-state rollout; resource samples are not historical PPO logs",
        "gpu_name": __import__("torch").cuda.get_device_name(0),
        "steps_per_run": args.steps,
        "repeats": args.repeats,
        "sample_interval_seconds": args.interval,
        "seed": args.seed,
        "summary": summary,
        "runs": {
            key: [dict(result, resource_samples=samples) for result, samples in zip(raw[key], resource_rows[key])]
            for key in raw
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(report, args.plot)
    print(json.dumps({"output": str(args.output), "plot": str(args.plot), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
