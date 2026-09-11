"""Cross-process micro-batching service for GPU FBA surrogate inference.

Stable-Baselines3 ``SubprocVecEnv`` normally gives every environment its own
Python process.  Loading a CUDA model in each worker both duplicates VRAM and
turns inference into batch-size-one launches.  This manager-hosted service owns
one CUDA context, waits a short bounded window, and fuses aligned requests from
all environment workers into one inference batch per species.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from typing import Any, Mapping

import numpy as np

from .fba_surrogate import FbaSurrogateBackend


class _PendingRequest:
    def __init__(self, species: str, features: np.ndarray) -> None:
        self.species = species
        self.features = np.asarray(features, dtype=np.float32)
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None


class BatchedSurrogateService:
    """Thread-safe request collector hosted in a ``BaseManager`` process."""

    def __init__(
        self,
        models: Mapping[str, Any],
        artifact_dir: str,
        device: str = "cuda",
        batch_window_ms: float = 2.0,
        max_batch_size: int = 64,
        ood_threshold: float = 8.0,
        bound_tolerance: float = 1e-4,
        residual_tolerance: float = 2e-4,
    ) -> None:
        self.backend = FbaSurrogateBackend(
            models,
            artifact_dir,
            device=device,
            ood_threshold=ood_threshold,
            bound_tolerance=bound_tolerance,
            residual_tolerance=residual_tolerance,
        )
        self.batch_window_seconds = max(0.0, float(batch_window_ms) / 1000.0)
        self.max_batch_size = max(1, int(max_batch_size))
        self.requests: queue.Queue[_PendingRequest | None] = queue.Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, len(self.backend.species)),
            thread_name_prefix="fba-gem-stream",
        )
        self.running = True
        self.lock = threading.Lock()
        self.counters = {
            "requests": 0,
            "batches": 0,
            "accepted": 0,
            "max_observed_batch": 0,
            "inference_seconds": 0.0,
        }
        self.batch_size_histogram: dict[int, int] = defaultdict(int)
        self.species_counters: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "requests": 0,
                "batches": 0,
                "accepted": 0,
                "max_observed_batch": 0,
                "inference_seconds": 0.0,
            }
        )
        self.worker = threading.Thread(
            target=self._batch_loop, name="fba-surrogate-batcher", daemon=True
        )
        self.worker.start()

    def _enqueue(self, species: str, features: np.ndarray) -> _PendingRequest:
        if not self.running:
            raise RuntimeError("FBA surrogate service is stopped")
        request = _PendingRequest(species, features)
        self.requests.put(request)
        return request

    @staticmethod
    def _wait(request: _PendingRequest) -> dict[str, Any]:
        request.event.wait()
        if request.error is not None:
            raise request.error
        assert request.result is not None
        return request.result

    def predict(self, species: str, features: np.ndarray) -> dict[str, Any]:
        return self._wait(self._enqueue(species, features))

    def predict_many(
        self, features_by_species: Mapping[str, np.ndarray]
    ) -> dict[str, dict[str, Any]]:
        """Submit all GEM requests for one environment before blocking.

        This removes three manager round trips per environment step and lets
        the batcher see aligned requests for every species from every worker.
        """

        pending = {
            species: self._enqueue(species, features)
            for species, features in features_by_species.items()
        }
        return {species: self._wait(request) for species, request in pending.items()}

    def diagnostics(self) -> dict[str, Any]:
        with self.lock:
            values = dict(self.counters)
        values["mean_batch_size"] = (
            values["requests"] / values["batches"] if values["batches"] else 0.0
        )
        with self.lock:
            values["batch_size_histogram"] = {
                str(size): count
                for size, count in sorted(self.batch_size_histogram.items())
            }
            values["species"] = {
                name: dict(counters)
                for name, counters in sorted(self.species_counters.items())
            }
        for counters in values["species"].values():
            counters["mean_batch_size"] = (
                counters["requests"] / counters["batches"]
                if counters["batches"]
                else 0.0
            )
        values["load_errors"] = dict(self.backend.load_errors)
        values["loaded_species"] = sorted(self.backend.species)
        return values

    def close(self) -> None:
        self.running = False
        self.requests.put(None)
        self.worker.join(timeout=5.0)
        self.executor.shutdown(wait=True)

    def _solve_group(
        self, species: str, requests: list[_PendingRequest]
    ) -> tuple[str, list[_PendingRequest], list[dict[str, Any]] | None, BaseException | None, float]:
        started = time.perf_counter()
        try:
            batch = np.stack([item.features for item in requests])
            results = self.backend.predict_feature_payload_batch(species, batch)
            return species, requests, results, None, time.perf_counter() - started
        except BaseException as exc:
            return species, requests, None, exc, time.perf_counter() - started

    def _batch_loop(self) -> None:
        while self.running:
            first = self.requests.get()
            if first is None:
                return
            pending = [first]
            deadline = time.perf_counter() + self.batch_window_seconds
            while len(pending) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.running = False
                    break
                pending.append(item)

            groups: dict[str, list[_PendingRequest]] = defaultdict(list)
            for item in pending:
                groups[item.species].append(item)
            futures = [
                self.executor.submit(self._solve_group, species, requests)
                for species, requests in groups.items()
            ]
            for future in futures:
                species, requests, results, error, elapsed = future.result()
                if error is not None:
                    for item in requests:
                        item.error = error
                else:
                    assert results is not None
                    for item, result in zip(requests, results):
                        item.result = result
                try:
                    with self.lock:
                        self.counters["requests"] += len(requests)
                        self.counters["batches"] += 1
                        self.counters["max_observed_batch"] = max(
                            self.counters["max_observed_batch"], len(requests)
                        )
                        self.counters["inference_seconds"] += elapsed
                        self.counters["accepted"] += sum(
                            item.result is not None
                            and bool(item.result.get("accepted", False))
                            for item in requests
                        )
                        self.batch_size_histogram[len(requests)] += 1
                        species_values = self.species_counters[species]
                        species_values["requests"] += len(requests)
                        species_values["batches"] += 1
                        species_values["max_observed_batch"] = max(
                            int(species_values["max_observed_batch"]), len(requests)
                        )
                        species_values["inference_seconds"] += elapsed
                        species_values["accepted"] += sum(
                            item.result is not None
                            and bool(item.result.get("accepted", False))
                            for item in requests
                        )
                finally:
                    for item in requests:
                        item.event.set()


class SurrogateManager(BaseManager):
    pass


SurrogateManager.register(
    "BatchedSurrogateService",
    BatchedSurrogateService,
    exposed=("predict", "predict_many", "diagnostics", "close"),
)


def start_surrogate_service(
    models: Mapping[str, Any],
    artifact_dir: str,
    **kwargs: Any,
):
    """Start the single GPU owner and return ``(manager, service_proxy)``."""

    # CUDA cannot be safely initialized in a process forked from a parent that
    # has already touched torch.cuda (which SB3 and benchmark code commonly do).
    # A spawned manager owns a fresh interpreter and therefore one valid CUDA
    # context regardless of parent initialization order.
    manager = SurrogateManager(ctx=mp.get_context("spawn"))
    manager.start()
    service = manager.BatchedSurrogateService(models, artifact_dir, **kwargs)
    return manager, service
