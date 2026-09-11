"""Cross-process micro-batching service for cooperative neural + GPU QP FBA."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.managers import BaseManager
import queue
import threading
import time
from typing import Any, Mapping

import numpy as np

from .community_solver import CooperativeCommunityFbaSolver


class _Request:
    def __init__(self, context: np.ndarray) -> None:
        self.context = np.asarray(context, dtype=np.float32)
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None


def _layout(metadata: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    result = {}
    cursor = 0
    for name, width in metadata["context_layout"]:
        result[str(name)] = (cursor, cursor + int(width))
        cursor += int(width)
    return result


class CooperativeGpuQpService:
    """Own one CUDA context and fuse requests from SB3 environment workers."""

    def __init__(
        self,
        models: Mapping[str, Any],
        original_exchange_bounds: Mapping[str, Mapping[str, tuple[float, float]]],
        artifact: str,
        device: str = "cuda",
        candidates: int = 128,
        rerank_pool: int | None = None,
        decision_strength: float | None = None,
        blend_candidates: int = 1,
        blend_distance_power: float = 2.0,
        retry_candidates: int = 2048,
        qp_max_iterations: int = 1200,
        block_composition_candidates: int = 6,
        batch_window_ms: float = 2.0,
        max_batch_size: int = 64,
        optimize_live_objectives: bool | None = None,
        multioutput_strength: float = 0.0,
        independent_species: bool = False,
        retry_batch_size: int = 1,
    ) -> None:
        if optimize_live_objectives is None:
            import torch
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            optimize_live_objectives = bool(payload["metadata"].get("cooperative_optimize_live_objectives", False))
            del payload
        self.solver = CooperativeCommunityFbaSolver(
            models,
            original_exchange_bounds=original_exchange_bounds,
            maximum_coexistence_growth=0.005,
            optimize_live_objectives=optimize_live_objectives,
            surrogate_artifact=artifact,
            surrogate_device=device,
            surrogate_require_qualified=False,
            surrogate_exact_interval=0,
            gpu_qp_projection=True,
            gpu_qp_only=True,
            gpu_qp_candidates=candidates,
            gpu_qp_rerank_pool=rerank_pool,
            gpu_qp_decision_strength=decision_strength,
            gpu_qp_blend_candidates=blend_candidates,
            gpu_qp_blend_distance_power=blend_distance_power,
            gpu_qp_max_iterations=qp_max_iterations,
            gpu_qp_multioutput_strength=multioutput_strength,
            gpu_qp_independent_species=independent_species,
        )
        if self.solver._surrogate_dictionary is None or self.solver._gpu_qp_projector is None:
            raise RuntimeError("cooperative GPU QP backend was not constructed")
        self.backend = self.solver._surrogate_dictionary
        self.projector = self.solver._gpu_qp_projector
        self.projector.block_composition_candidates = max(
            1, int(block_composition_candidates)
        )
        self.multioutput_strength = float(multioutput_strength)
        self.independent_species = bool(independent_species)
        if multioutput_strength > 0:
            from .gpu_multioutput_qp import MultiOutputCooperativeQpProjector
            self.projector = MultiOutputCooperativeQpProjector(self.projector, multioutput_strength,
                                                            independent_species=independent_species)
        self.layout = _layout(self.backend.metadata)
        self.candidates = max(2, int(candidates))
        self.retry_candidates = max(self.candidates, int(retry_candidates))
        self.retry_batch_size = max(1, int(retry_batch_size))
        self.batch_window_seconds = max(0.0, float(batch_window_ms) / 1000.0)
        self.max_batch_size = max(1, int(max_batch_size))
        self.requested_max_batch_size = self.max_batch_size
        if self.solver._gpu_qp_projector.device.type == "cuda":
            import torch
            free_bytes, _ = torch.cuda.mem_get_info(self.solver._gpu_qp_projector.device)
            species_factor = len(self.backend.metadata["species"]) if independent_species else 1
            # Conservative transient-buffer allowance, not a VRAM utilization target.
            per_row_bytes = (self.candidates + 256)*len(self.backend.metadata["reaction_ids"])*4*(10+6*species_factor)
            self.max_batch_size = min(self.max_batch_size, max(1, int(.5*free_bytes/per_row_bytes)))
        self.requests: queue.Queue[_Request | None] = queue.Queue()
        self.running = True
        self.lock = threading.Lock()
        self.counters = {
            "requests": 0,
            "batches": 0,
            "feasible": 0,
            "max_observed_batch": 0,
            "inference_seconds": 0.0,
            "retry_requests": 0,
            "retry_recovered": 0,
        }
        self.batch_size_histogram: dict[int, int] = {}
        self.worker = threading.Thread(target=self._batch_loop, daemon=True)
        self.worker.start()

    def predict(self, context: np.ndarray) -> dict[str, Any]:
        if not self.running:
            raise RuntimeError("cooperative GPU QP service is stopped")
        request = _Request(context)
        self.requests.put(request)
        request.event.wait()
        if request.error is not None:
            raise request.error
        assert request.result is not None
        return request.result

    def _solve(self, requests: list[_Request]) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        contexts = np.stack([request.context for request in requests])
        biomass = slice(*self.layout["biomass_g_l"])
        supply = slice(*self.layout["shared_supply_mmol_l_h"])
        lower = slice(*self.layout["flux_lower_bounds"])
        upper = slice(*self.layout["flux_upper_bounds"])
        candidate_batch = self.backend.rank_device(contexts, top_k=self.candidates)
        def project(batch, query):
            options = {}
            if self.multioutput_strength > 0:
                options = dict(decision_indices=self.backend.decision_indices,
                    decision_targets=batch.predicted_decision_fluxes,
                    decision_scales=self.backend.decision_scale,
                    decision_weights=self.backend.decision_weight)
            return self.projector.project(batch.fluxes, query[:, lower], query[:, upper],
                query[:, biomass], query[:, supply], reference_weights=batch.reference_weights, **options)
        projection = project(candidate_batch, contexts)
        initial_feasible = np.asarray(projection.feasible, dtype=bool).copy()
        failed = np.flatnonzero(~initial_feasible)
        retry_results = {}
        if len(failed) and self.retry_candidates > self.candidates:
            for start in range(0, len(failed), self.retry_batch_size):
                chunk = failed[start:start+self.retry_batch_size]
                retry_contexts = contexts[chunk]
                retry_batch = self.backend.rank_device(retry_contexts, top_k=self.retry_candidates)
                retry_projection = project(retry_batch, retry_contexts)
                for local, index in enumerate(chunk):
                    retry_results[int(index)] = (retry_projection, local)
        elapsed = time.perf_counter() - started
        results = []
        for index in range(len(requests)):
            source = projection
            source_index = index
            if index in retry_results:
                source, source_index = retry_results[index]
            results.append(
                {
                    "feasible": bool(source.feasible[source_index]),
                    "fluxes": source.fluxes[source_index],
                    "iterations": int(source.iterations),
                    "max_bound_violation": float(
                        source.max_bound_violation[source_index]
                    ),
                    "max_shared_violation": float(
                        source.max_shared_violation[source_index]
                    ),
                    "max_shared_violation_index": int(
                        source.max_shared_violation_index[source_index]
                    ),
                    "retry_used": index in retry_results,
                    "multioutput_improved": bool(source.multioutput_improved[source_index])
                        if source.multioutput_improved is not None else False,
                    "inference_seconds": float(elapsed / len(requests)),
                }
            )
        return results, elapsed

    def _batch_loop(self) -> None:
        while self.running:
            first = self.requests.get()
            if first is None:
                return
            pending = [first]
            deadline = time.perf_counter() + self.batch_window_seconds
            while len(pending) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    item = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.running = False
                    break
                pending.append(item)
            try:
                results, elapsed = self._solve(pending)
                for request, result in zip(pending, results):
                    request.result = result
                with self.lock:
                    self.counters["requests"] += len(pending)
                    self.counters["batches"] += 1
                    self.counters["feasible"] += sum(
                        int(result["feasible"]) for result in results
                    )
                    self.counters["max_observed_batch"] = max(
                        self.counters["max_observed_batch"], len(pending)
                    )
                    self.counters["inference_seconds"] += elapsed
                    self.counters["retry_requests"] += sum(
                        int(result["retry_used"]) for result in results
                    )
                    self.counters["retry_recovered"] += sum(
                        int(result["retry_used"] and result["feasible"])
                        for result in results
                    )
                    self.batch_size_histogram[len(pending)] = (
                        self.batch_size_histogram.get(len(pending), 0) + 1
                    )
            except BaseException as exc:
                for request in pending:
                    request.error = exc
            finally:
                for request in pending:
                    request.event.set()

    def diagnostics(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.counters)
            result["batch_size_histogram"] = {
                str(key): value for key, value in sorted(self.batch_size_histogram.items())
            }
        result["mean_batch_size"] = (
            result["requests"] / result["batches"] if result["batches"] else 0.0
        )
        result["feasible_rate"] = (
            result["feasible"] / result["requests"] if result["requests"] else 0.0
        )
        result["online_cpu_lp_stage_calls"] = self.solver.cpu_lp_stage_calls
        result["optimize_live_objectives"] = self.solver.optimize_live_objectives
        result["multioutput_strength"] = self.multioutput_strength
        result["independent_species"] = self.independent_species
        result["requested_max_batch_size"] = self.requested_max_batch_size
        result["memory_limited_max_batch_size"] = self.max_batch_size
        result["retry_batch_size"] = self.retry_batch_size
        result["species"] = list(self.solver.species_names) if hasattr(self.solver, "species_names") else []
        result["device"] = str(self.solver._gpu_qp_projector.device)
        return result

    def close(self) -> None:
        self.running = False
        self.requests.put(None)
        self.worker.join(timeout=5.0)


class CooperativeGpuManager(BaseManager):
    pass


CooperativeGpuManager.register(
    "CooperativeGpuQpService",
    CooperativeGpuQpService,
    exposed=("predict", "diagnostics", "close"),
)


def start_cooperative_gpu_qp_service(
    models: Mapping[str, Any],
    original_exchange_bounds: Mapping[str, Mapping[str, tuple[float, float]]],
    artifact: str,
    **kwargs: Any,
):
    manager = CooperativeGpuManager(ctx=mp.get_context("spawn"))
    manager.start()
    service = manager.CooperativeGpuQpService(
        models, original_exchange_bounds, artifact, **kwargs
    )
    return manager, service
