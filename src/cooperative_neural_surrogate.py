"""GPU neural-mechanistic emulator for the cooperative three-GEM LP.

The decoder is affine in a flux basis learned only from exact feasible
cooperative-FBA solutions::

    v_hat = v_mean + B z(x)

Both ``v_mean`` and every row of ``B`` lie in the block-diagonal null space of
the three stoichiometric matrices.  Consequently the neural network cannot
invent mass-imbalanced intracellular fluxes (apart from floating-point and
offline factorisation error).  State-specific reaction bounds and shared
medium constraints are intentionally checked by the caller before accepting a
prediction; rejected or out-of-domain rows fall back to exact HiGHS.

This is inspired by artificial metabolic networks (AMNs), but is specialised
to the aligned, shared-medium community LP used by this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np


ARTIFACT_VERSION = 1
FORMULATION = "neural_mechanistic_low_rank_cooperative"
RERANK_FORMULATION = "neural_decision_reranked_feasible_dictionary"


@dataclass(frozen=True)
class CooperativeCandidateDeviceBatch:
    """CUDA-resident candidate batch used by the downstream QP layer."""

    indices: Any
    distances: Any
    fluxes: Any
    predicted_decision_fluxes: Any
    reference_weights: Any
    inference_seconds: float


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def make_cooperative_network(
    input_dim: int,
    latent_dim: int,
    hidden_dims: Sequence[int],
):
    """Construct the context-to-latent network used by training and runtime."""

    torch, nn = _torch()

    class CooperativeLatentNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            previous = int(input_dim)
            for width in hidden_dims:
                layers.extend(
                    (
                        nn.Linear(previous, int(width)),
                        nn.SiLU(),
                    )
                )
                previous = int(width)
            layers.append(nn.Linear(previous, int(latent_dim)))
            self.encoder = nn.Sequential(*layers)

        def forward(self, values):
            return self.encoder(values)

    return CooperativeLatentNetwork()


class CooperativeNeuralDecisionReranker:
    """Neural phenotype prediction followed by exact-feasible flux retrieval.

    The network predicts only fluxes consumed by the dynamic environment
    (growth, PHA and exchange reactions).  It does not emit the accepted flux
    vector.  Instead, it reorders a local pool from an exact feasible-flux
    dictionary, so every returned candidate retains intracellular mass
    balance by construction.
    """

    def __init__(self, artifact: str | Path, device: str = "cuda") -> None:
        torch, _ = _torch()
        from .cooperative_surrogate import CooperativeFluxDictionary

        artifact_path = Path(artifact)
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        metadata = dict(payload["metadata"])
        if int(metadata.get("format_version", -1)) != ARTIFACT_VERSION:
            raise ValueError("unsupported neural reranker artifact version")
        if metadata.get("formulation") != RERANK_FORMULATION:
            raise ValueError("artifact is not a neural decision reranker")
        dictionary_path = Path(str(metadata["base_dictionary"]))
        if not dictionary_path.is_absolute():
            dictionary_path = artifact_path.parent / dictionary_path
        self.dictionary = CooperativeFluxDictionary(dictionary_path, device=device)
        self.device = self.dictionary.device
        self.metadata: dict[str, Any] = metadata
        self.feature_indices = torch.as_tensor(
            payload["feature_indices"], dtype=torch.int64, device=self.device
        )
        self.feature_mean = torch.as_tensor(
            payload["feature_mean"], dtype=torch.float32, device=self.device
        )
        self.feature_scale = torch.as_tensor(
            payload["feature_scale"], dtype=torch.float32, device=self.device
        )
        self.decision_indices = torch.as_tensor(
            payload["decision_indices"], dtype=torch.int64, device=self.device
        )
        self.decision_mean = torch.as_tensor(
            payload["decision_mean"], dtype=torch.float32, device=self.device
        )
        self.decision_scale = torch.as_tensor(
            payload["decision_scale"], dtype=torch.float32, device=self.device
        )
        self.decision_weight = torch.as_tensor(
            payload["decision_weight"], dtype=torch.float32, device=self.device
        )
        self.network = make_cooperative_network(
            len(self.feature_indices),
            len(self.decision_indices),
            tuple(int(value) for value in metadata["hidden_dims"]),
        )
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval().to(self.device)
        # The immutable dictionary is queried at every dFBA step. Cache only
        # decision columns for reranking, not a pool x all-reaction temporary.
        # For the PHBV consortium this reduces each 2048-row gather from 6733
        # reaction columns to 155 decision columns; full fluxes are gathered
        # only after top-k selection for the QP.
        with torch.inference_mode():
            self._context_norm_squared = self.dictionary.contexts.square().sum(dim=1)
            self._normalized_dictionary_decisions = (
                self.dictionary.fluxes.index_select(1, self.decision_indices)
                - self.decision_mean
            ) / self.decision_scale
        self.distance_threshold = self.dictionary.distance_threshold
        self.rerank_pool = int(metadata.get("rerank_pool", 128))
        self.decision_strength = float(metadata.get("decision_strength", 4.0))
        self.blend_candidates = int(metadata.get("blend_candidates", 1))
        self.blend_distance_power = float(
            metadata.get("blend_distance_power", 2.0)
        )

    def rank_device(
        self, contexts: np.ndarray, top_k: int = 16
    ) -> CooperativeCandidateDeviceBatch:
        """Return reordered candidates without copying them back to the host."""

        torch, _ = _torch()

        values = np.asarray(contexts, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        started = time.perf_counter()
        with torch.inference_mode():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            selected = tensor.index_select(1, self.dictionary.feature_indices)
            selected = (
                (selected - self.dictionary.feature_mean)
                / self.dictionary.feature_scale
                * self.dictionary.feature_weight_sqrt
            )
            distance_squared = (
                torch.sum(selected.square(), dim=1, keepdim=True)
                + self._context_norm_squared[None, :]
                - 2.0 * selected @ self.dictionary.contexts.T
            ).clamp_min_(0.0)
            pool = min(
                max(int(top_k), self.rerank_pool),
                self.dictionary.candidate_count,
            )
            pool_distance, pool_indices = torch.topk(
                distance_squared, k=pool, dim=1, largest=False, sorted=True
            )

            neural_x = tensor.index_select(1, self.feature_indices)
            neural_x = (neural_x - self.feature_mean) / self.feature_scale
            predicted_decision = self.network(neural_x)
            candidate_decision = self._normalized_dictionary_decisions[pool_indices]
            decision_error = torch.mean(
                (candidate_decision - predicted_decision[:, None, :]).square()
                * self.decision_weight[None, None, :],
                dim=2,
            )
            # Normalise context distance per varying feature so the tunable
            # decision term has a stable scale across dictionary versions.
            combined = (
                pool_distance / max(1, self.dictionary.selected_feature_count)
                + self.decision_strength * decision_error
            )
            keep = min(max(1, int(top_k)), pool)
            _, order = torch.topk(
                combined, k=keep, dim=1, largest=False, sorted=True
            )
            selected_indices = torch.gather(pool_indices, 1, order)
            selected_distance = torch.sqrt(torch.gather(pool_distance, 1, order))
            selected_fluxes = self.dictionary.fluxes[selected_indices]
            blend = min(max(1, self.blend_candidates), keep)
            reference_weights = torch.zeros_like(selected_distance)
            if blend == 1:
                reference_weights[:, 0] = 1.0
            else:
                local_distance = selected_distance[:, :blend].clamp_min(1e-4)
                local_weight = local_distance.pow(-self.blend_distance_power)
                local_weight = local_weight / local_weight.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-12)
                reference_weights[:, :blend] = local_weight
        elapsed = time.perf_counter() - started
        return CooperativeCandidateDeviceBatch(
            indices=selected_indices,
            distances=selected_distance,
            fluxes=selected_fluxes,
            predicted_decision_fluxes=(
                predicted_decision * self.decision_scale + self.decision_mean
            ),
            reference_weights=reference_weights,
            inference_seconds=elapsed,
        )

    def rank(self, contexts: np.ndarray, top_k: int = 16):
        """Return exact flux candidates reordered by predicted phenotype."""

        from .cooperative_surrogate import CooperativeCandidateBatch

        device_batch = self.rank_device(contexts, top_k=top_k)
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)
        return CooperativeCandidateBatch(
            indices=device_batch.indices.cpu().numpy(),
            distances=device_batch.distances.cpu().numpy(),
            fluxes=device_batch.fluxes.cpu().numpy(),
            inference_seconds=device_batch.inference_seconds,
        )


@dataclass(frozen=True)
class CooperativeNeuralPrediction:
    fluxes: np.ndarray
    accepted_domain: bool
    reason: str
    ood_score: float
    inference_seconds: float


class CooperativeNeuralMechanisticSurrogate:
    """Load and execute a self-describing community flux emulator on CUDA."""

    def __init__(self, artifact: str | Path, device: str = "cuda") -> None:
        torch, _ = _torch()
        requested = device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda is unavailable")

        payload = torch.load(Path(artifact), map_location="cpu", weights_only=False)
        metadata = dict(payload["metadata"])
        if int(metadata.get("format_version", -1)) != ARTIFACT_VERSION:
            raise ValueError("unsupported cooperative neural artifact version")
        if metadata.get("formulation") != FORMULATION:
            raise ValueError("artifact is not a cooperative neural-mechanistic model")

        self.metadata: dict[str, Any] = metadata
        self.feature_indices = torch.as_tensor(
            payload["feature_indices"], dtype=torch.int64, device=self.device
        )
        self.feature_mean = torch.as_tensor(
            payload["feature_mean"], dtype=torch.float32, device=self.device
        )
        self.feature_scale = torch.as_tensor(
            payload["feature_scale"], dtype=torch.float32, device=self.device
        )
        self.feature_min = torch.as_tensor(
            payload["feature_min"], dtype=torch.float32, device=self.device
        )
        self.feature_max = torch.as_tensor(
            payload["feature_max"], dtype=torch.float32, device=self.device
        )
        self.latent_mean = torch.as_tensor(
            payload["latent_mean"], dtype=torch.float32, device=self.device
        )
        self.latent_scale = torch.as_tensor(
            payload["latent_scale"], dtype=torch.float32, device=self.device
        )
        self.flux_mean = torch.as_tensor(
            payload["flux_mean"], dtype=torch.float32, device=self.device
        )
        self.flux_basis = torch.as_tensor(
            payload["flux_basis"], dtype=torch.float32, device=self.device
        )
        self.network = make_cooperative_network(
            len(self.feature_indices),
            int(self.flux_basis.shape[0]),
            tuple(int(value) for value in metadata["hidden_dims"]),
        )
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval().to(self.device)
        self.ood_tolerance = float(metadata.get("ood_tolerance", 0.25))

    def predict(self, contexts: np.ndarray) -> list[CooperativeNeuralPrediction]:
        """Predict one or more aligned community flux vectors on one device."""

        torch, _ = _torch()
        values = np.asarray(contexts, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != int(self.metadata["context_dimension"]):
            raise ValueError("cooperative neural context dimension mismatch")

        started = time.perf_counter()
        with torch.inference_mode():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            selected = tensor.index_select(1, self.feature_indices)
            normalized = (selected - self.feature_mean) / self.feature_scale
            below = torch.relu(self.feature_min - selected) / self.feature_scale
            above = torch.relu(selected - self.feature_max) / self.feature_scale
            ood = torch.amax(torch.maximum(below, above), dim=1)
            normalized_latent = self.network(normalized)
            latent = normalized_latent * self.latent_scale + self.latent_mean
            fluxes = self.flux_mean + latent @ self.flux_basis
            finite = torch.all(torch.isfinite(fluxes), dim=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started

        flux_rows = fluxes.detach().cpu().numpy().astype(np.float64, copy=False)
        ood_rows = ood.detach().cpu().numpy().astype(float, copy=False)
        finite_rows = finite.detach().cpu().numpy().astype(bool, copy=False)
        per_row = elapsed / max(1, len(values))
        result = []
        for index, flux in enumerate(flux_rows):
            if not bool(finite_rows[index]):
                reason = "non_finite"
            elif float(ood_rows[index]) > self.ood_tolerance:
                reason = "out_of_distribution"
            else:
                reason = "accepted_domain"
            result.append(
                CooperativeNeuralPrediction(
                    fluxes=flux,
                    accepted_domain=reason == "accepted_domain",
                    reason=reason,
                    ood_score=float(ood_rows[index]),
                    inference_seconds=per_row,
                )
            )
        return result
