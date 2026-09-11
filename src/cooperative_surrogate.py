"""GPU-resident nearest-state dictionary for aligned cooperative FBA fluxes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CooperativeCandidateBatch:
    indices: np.ndarray
    distances: np.ndarray
    fluxes: np.ndarray
    inference_seconds: float


class CooperativeFluxDictionary:
    """Load aligned community candidates and rank them on one CUDA device."""

    def __init__(self, artifact: str | Path, device: str = "cuda") -> None:
        import torch

        payload = torch.load(Path(artifact), map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if int(metadata.get("format_version", -1)) != 1:
            raise ValueError("unsupported cooperative surrogate artifact version")
        if metadata.get("formulation") != "exact_cooperative_shared_medium":
            raise ValueError("artifact is not an aligned cooperative-FBA dictionary")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
        self.metadata: dict[str, Any] = dict(metadata)
        self.feature_indices = torch.as_tensor(
            payload["feature_indices"], dtype=torch.int64, device=self.device
        )
        self.feature_mean = torch.as_tensor(
            payload["feature_mean"], dtype=torch.float32, device=self.device
        )
        self.feature_scale = torch.as_tensor(
            payload["feature_scale"], dtype=torch.float32, device=self.device
        )
        self.feature_weight = torch.as_tensor(
            payload.get(
                "feature_weight",
                np.ones(len(payload["feature_indices"]), dtype=np.float32),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if self.feature_weight.shape != self.feature_mean.shape:
            raise ValueError("cooperative surrogate feature-weight shape mismatch")
        if not torch.all(torch.isfinite(self.feature_weight)) or torch.any(
            self.feature_weight <= 0
        ):
            raise ValueError("cooperative surrogate feature weights must be finite and positive")
        self.feature_weight_sqrt = torch.sqrt(self.feature_weight)
        self.contexts = torch.as_tensor(
            payload["normalized_contexts"], dtype=torch.float32, device=self.device
        ).contiguous()
        self.fluxes = torch.as_tensor(
            payload["fluxes"], dtype=torch.float32, device=self.device
        ).contiguous()
        self.distance_threshold = float(metadata["distance_threshold"])

    @property
    def candidate_count(self) -> int:
        return int(self.fluxes.shape[0])

    @property
    def selected_feature_count(self) -> int:
        return int(self.feature_indices.numel())

    def rank(self, contexts: np.ndarray, top_k: int = 16) -> CooperativeCandidateBatch:
        import torch

        values = np.asarray(contexts, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != int(self.metadata["context_dimension"]):
            raise ValueError("cooperative surrogate context dimension mismatch")
        started = time.perf_counter()
        with torch.inference_mode():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            selected = tensor.index_select(1, self.feature_indices)
            selected = (
                (selected - self.feature_mean)
                / self.feature_scale
                * self.feature_weight_sqrt
            )
            # Squared Euclidean distance avoids materializing [B, N, D].
            distance = (
                torch.sum(selected.square(), dim=1, keepdim=True)
                + torch.sum(self.contexts.square(), dim=1)[None, :]
                - 2.0 * selected @ self.contexts.T
            ).clamp_min_(0.0)
            k = min(max(1, int(top_k)), self.candidate_count)
            distances, indices = torch.topk(
                distance, k=k, dim=1, largest=False, sorted=True
            )
            selected_fluxes = self.fluxes[indices]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        return CooperativeCandidateBatch(
            indices=indices.cpu().numpy(),
            distances=np.sqrt(distances.cpu().numpy()),
            fluxes=selected_fluxes.cpu().numpy(),
            inference_seconds=elapsed,
        )
