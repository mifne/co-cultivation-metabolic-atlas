"""Physics-constrained GPU surrogate for repeated genome-scale FBA solves.

The neural network predicts coordinates in the null space of the
stoichiometric matrix rather than reaction fluxes directly.  The fixed
decoder ``v = N z`` therefore satisfies ``S v = 0`` by construction (up to
floating-point round-off).  Runtime guards reject out-of-distribution inputs,
bound violations, and excessive residuals so that callers can fall back to an
exact LP solver.

Artifacts are deliberately model-specific: reaction order, objective vector,
and the stoichiometric matrix fingerprint are checked when they are loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from cobra.core.solution import Solution
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients


ARTIFACT_VERSION = 1


class SurrogateUnavailable(RuntimeError):
    """Raised when a surrogate artifact cannot safely serve a model."""


@dataclass(frozen=True)
class SurrogatePrediction:
    solution: Solution | None
    accepted: bool
    reason: str
    ood_score: float
    max_bound_violation: float
    max_mass_balance_residual: float
    max_relative_mass_balance_residual: float
    inference_seconds: float


def safe_species_filename(species_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", species_name).strip("_")


def model_fingerprint(model: Any) -> str:
    """Return a stable hash of reaction order and stoichiometry."""

    matrix = create_stoichiometric_matrix(
        model, array_type="lil", dtype=np.float64
    ).tocsr()
    digest = hashlib.sha256()
    digest.update("\0".join(reaction.id for reaction in model.reactions).encode())
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.data.astype(np.float64, copy=False).tobytes())
    return digest.hexdigest()


def extract_lp_features(model: Any) -> np.ndarray:
    """Encode every LP quantity that can change the optimum.

    The feature vector is ``[lower bounds, upper bounds, objective, sense]``.
    Keeping all bounds (rather than selected extracellular concentrations)
    also captures project-specific regulatory constraints such as ``R_LCP``.
    """

    reactions = list(model.reactions)
    lower = np.fromiter((r.lower_bound for r in reactions), dtype=np.float32)
    upper = np.fromiter((r.upper_bound for r in reactions), dtype=np.float32)
    objective = np.zeros(len(reactions), dtype=np.float32)
    reaction_index = {reaction: index for index, reaction in enumerate(reactions)}
    for reaction, coefficient in linear_reaction_coefficients(model).items():
        objective[reaction_index[reaction]] = float(coefficient)
    sense = np.array(
        [1.0 if model.objective.direction == "max" else -1.0],
        dtype=np.float32,
    )
    return np.concatenate((lower, upper, objective, sense))


def compute_nullspace_basis(model: Any, rcond: float | None = None) -> np.ndarray:
    """Compute an orthonormal basis ``N`` for ``null(S)`` offline."""

    from scipy.linalg import null_space

    matrix = create_stoichiometric_matrix(
        model, array_type="dense", dtype=np.float64
    )
    basis = null_space(np.asarray(matrix), rcond=rcond)
    if basis.shape[1] == 0:
        raise ValueError(f"model {model.id!r} has an empty stoichiometric null space")
    return np.asarray(basis, dtype=np.float32, order="C")


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - torch is an SB3 dependency
        raise SurrogateUnavailable(
            "PyTorch is required for the FBA surrogate backend"
        ) from exc
    return torch, nn


def make_network(
    basis: np.ndarray,
    input_dim: int,
    hidden_dims: Sequence[int],
    decoder_mode: str = "nullspace",
):
    """Create the model without importing torch when the backend is unused."""

    torch, nn = _torch()

    if decoder_mode not in {
        "nullspace",
        "feasible_dictionary",
        "feasible_dictionary_optimizer",
    }:
        raise ValueError(f"unsupported surrogate decoder mode: {decoder_mode}")

    class ConstrainedFluxNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend((nn.Linear(previous, width), nn.SiLU()))
                previous = width
            layers.append(nn.Linear(previous, basis.shape[1]))
            self.encoder = nn.Sequential(*layers)
            self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
            self.register_buffer("x_mean", torch.zeros(input_dim, dtype=torch.float32))
            self.register_buffer("x_scale", torch.ones(input_dim, dtype=torch.float32))
            self.register_buffer("z_mean", torch.zeros(basis.shape[1], dtype=torch.float32))
            self.register_buffer("z_scale", torch.ones(basis.shape[1], dtype=torch.float32))
            self.register_buffer("x_min", torch.full((input_dim,), -torch.inf))
            self.register_buffer("x_max", torch.full((input_dim,), torch.inf))

        def latent(self, features):
            normalized = (features - self.x_mean) / self.x_scale
            encoded = self.encoder(normalized)
            if decoder_mode == "feasible_dictionary":
                # A convex combination of exact feasible flux vectors remains
                # in null(S). Runtime guards still check state-specific bounds.
                return torch.softmax(encoded, dim=-1)
            return encoded * self.z_scale + self.z_mean

        def forward(self, features):
            if decoder_mode == "feasible_dictionary_optimizer":
                candidates = self.basis.T
                n_reactions = candidates.shape[1]
                lower = features[:, :n_reactions]
                upper = features[:, n_reactions : 2 * n_reactions]
                objective = features[:, 2 * n_reactions : 3 * n_reactions]
                sense = features[:, -1]
                feasible = torch.all(
                    (candidates[None, :, :] >= lower[:, None, :] - 1e-5)
                    & (candidates[None, :, :] <= upper[:, None, :] + 1e-5),
                    dim=2,
                )
                scores = objective @ candidates.T
                scores = scores * sense[:, None]
                scores = torch.where(
                    feasible, scores, torch.full_like(scores, -torch.inf)
                )
                selected = torch.argmax(scores, dim=1)
                return candidates[selected]
            latent = self.latent(features)
            return latent @ self.basis.T

    return ConstrainedFluxNetwork()


class SpeciesFluxSurrogate:
    """Loaded, validated surrogate artifact for one GEM."""

    def __init__(
        self,
        model: Any,
        artifact_path: str | Path,
        device: str = "cuda",
        ood_threshold: float = 8.0,
        bound_tolerance: float = 1e-4,
        residual_tolerance: float = 2e-4,
    ) -> None:
        import time

        del time  # keeps the import local convention explicit
        torch, _ = _torch()
        self.model = model
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise SurrogateUnavailable("CUDA was requested but torch.cuda is unavailable")
        self.ood_threshold = float(ood_threshold)
        self.bound_tolerance = float(bound_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.matrix = create_stoichiometric_matrix(
            model, array_type="lil", dtype=np.float64
        ).tocsr()
        self.absolute_matrix = abs(self.matrix)
        self.reaction_ids = tuple(reaction.id for reaction in model.reactions)

        payload = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise SurrogateUnavailable("unsupported FBA surrogate artifact version")
        if tuple(metadata["reaction_ids"]) != self.reaction_ids:
            raise SurrogateUnavailable("surrogate reaction order does not match the GEM")
        if metadata["model_fingerprint"] != model_fingerprint(model):
            raise SurrogateUnavailable("surrogate stoichiometry fingerprint does not match")
        basis = np.asarray(payload["basis"], dtype=np.float32)
        self.network = make_network(
            basis,
            int(metadata["input_dim"]),
            tuple(int(v) for v in metadata["hidden_dims"]),
            str(metadata.get("decoder_mode", "nullspace")),
        )
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval().to(self.device)
        self.cuda_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.metadata = metadata
        self.decoder_mode = str(metadata.get("decoder_mode", "nullspace"))

        # Keep the fixed stoichiometric operators beside the network.  Runtime
        # guards are applied to every environment, so evaluating them on the
        # host one row at a time otherwise becomes the bottleneck once GPU
        # inference is micro-batched.
        def torch_csr(matrix):
            return torch.sparse_csr_tensor(
                torch.as_tensor(matrix.indptr, dtype=torch.int64, device=self.device),
                torch.as_tensor(matrix.indices, dtype=torch.int64, device=self.device),
                torch.as_tensor(matrix.data, dtype=torch.float32, device=self.device),
                size=matrix.shape,
                dtype=torch.float32,
                device=self.device,
            )

        self.torch_matrix = torch_csr(self.matrix)
        self.torch_absolute_matrix = torch_csr(self.absolute_matrix)

    def predict(self, model: Any | None = None) -> SurrogatePrediction:
        return self.predict_features(extract_lp_features(model or self.model))[0]

    def predict_feature_payloads(self, features: np.ndarray) -> list[dict[str, Any]]:
        """Run a batch and return lightweight arrays without pandas objects."""

        import time

        torch, _ = _torch()
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != int(self.metadata["input_dim"]):
            raise ValueError("surrogate feature dimension mismatch")

        started = time.perf_counter()
        stream_context = (
            torch.cuda.stream(self.cuda_stream)
            if self.cuda_stream is not None
            else nullcontext()
        )
        with stream_context, torch.inference_mode():
            tensor = torch.as_tensor(values, device=self.device)
            n = len(self.reaction_ids)
            scale = torch.clamp(self.network.x_scale, min=1e-6)
            normalized_distance = torch.abs((tensor - self.network.x_mean) / scale)
            ood = torch.amax(normalized_distance, dim=1)
            fluxes = self.network(tensor)

            lower = tensor[:, :n]
            upper = tensor[:, n : 2 * n]
            objective = tensor[:, 2 * n : 3 * n]
            bound_violation = torch.maximum(
                torch.amax(lower - fluxes, dim=1),
                torch.amax(fluxes - upper, dim=1),
            ).clamp_min(0.0)
            residual_rows = torch.sparse.mm(
                self.torch_matrix, fluxes.transpose(0, 1)
            ).transpose(0, 1).abs()
            residual = torch.amax(residual_rows, dim=1)
            row_activity = torch.sparse.mm(
                self.torch_absolute_matrix, fluxes.abs().transpose(0, 1)
            ).transpose(0, 1)
            relative_residual = torch.amax(
                residual_rows / torch.clamp(row_activity, min=1.0), dim=1
            )
            finite = torch.all(torch.isfinite(fluxes), dim=1)
            objective_value = torch.sum(objective * fluxes, dim=1)
        if self.cuda_stream is not None:
            self.cuda_stream.synchronize()
        elapsed = time.perf_counter() - started
        flux_rows = fluxes.detach().cpu().numpy().astype(np.float64, copy=False)
        ood_rows = ood.detach().cpu().numpy().astype(float, copy=False)
        bound_rows = bound_violation.detach().cpu().numpy().astype(float, copy=False)
        residual_max_rows = residual.detach().cpu().numpy().astype(float, copy=False)
        relative_residual_rows = (
            relative_residual.detach().cpu().numpy().astype(float, copy=False)
        )
        finite_rows = finite.detach().cpu().numpy().astype(bool, copy=False)
        objective_rows = objective_value.detach().cpu().numpy().astype(float, copy=False)
        sense = values[:, -1].astype(np.float64, copy=False)
        payloads: list[dict[str, Any]] = []
        per_item_seconds = elapsed / max(1, len(values))
        for index, flux in enumerate(flux_rows):
            row_bound_violation = float(bound_rows[index])
            row_residual = float(residual_max_rows[index])
            row_relative_residual = float(relative_residual_rows[index])
            if not bool(finite_rows[index]):
                reason = "non_finite"
            # The dictionary optimizer does not extrapolate a learned mapping:
            # it evaluates the current objective over stored feasible flux
            # candidates and applies the current bounds explicitly.  Training
            # feature distance is therefore not a validity criterion for that
            # decoder.  Physical bound and mass-balance guards remain active.
            elif (
                self.decoder_mode != "feasible_dictionary_optimizer"
                and float(ood_rows[index]) > self.ood_threshold
            ):
                reason = "out_of_distribution"
            elif row_bound_violation > self.bound_tolerance:
                reason = "bound_violation"
            elif row_relative_residual > self.residual_tolerance:
                reason = "mass_balance_residual"
            else:
                reason = "accepted"
            accepted = reason == "accepted"
            solution_payload = None
            if accepted:
                # The sense is included in the features/learned mapping.  COBRA
                # Solution always reports the un-negated biological objective.
                _ = sense[index]
                solution_payload = {
                    "objective_value": float(objective_rows[index]),
                    "status": "surrogate_optimal",
                    "fluxes": flux,
                    "reaction_ids": self.reaction_ids,
                }
            payloads.append(
                {
                    "solution": solution_payload,
                    "accepted": accepted,
                    "reason": reason,
                    "ood_score": float(ood_rows[index]),
                    "max_bound_violation": row_bound_violation,
                    "max_mass_balance_residual": row_residual,
                    "max_relative_mass_balance_residual": row_relative_residual,
                    "inference_seconds": per_item_seconds,
                }
            )
        return payloads

    def predict_features(self, features: np.ndarray) -> list[SurrogatePrediction]:
        """Run one or more LP feature rows and construct COBRA solutions."""

        predictions = []
        for payload in self.predict_feature_payloads(features):
            solution_payload = payload["solution"]
            solution = None
            if solution_payload is not None:
                solution = Solution(
                    objective_value=float(solution_payload["objective_value"]),
                    status=str(solution_payload["status"]),
                    fluxes=pd.Series(
                        np.asarray(solution_payload["fluxes"], dtype=np.float64),
                        index=solution_payload["reaction_ids"],
                        dtype=float,
                    ),
                )
            predictions.append(
                SurrogatePrediction(
                    solution=solution,
                    accepted=bool(payload["accepted"]),
                    reason=str(payload["reason"]),
                    ood_score=float(payload["ood_score"]),
                    max_bound_violation=float(payload["max_bound_violation"]),
                    max_mass_balance_residual=float(
                        payload["max_mass_balance_residual"]
                    ),
                    max_relative_mass_balance_residual=float(
                        payload["max_relative_mass_balance_residual"]
                    ),
                    inference_seconds=float(payload["inference_seconds"]),
                )
            )
        return predictions


class FbaSurrogateBackend:
    """Artifact loader and batched dispatch for all consortium species."""

    def __init__(
        self,
        models: Mapping[str, Any],
        artifact_dir: str | Path,
        device: str = "cuda",
        **guard_kwargs: Any,
    ) -> None:
        artifact_root = Path(artifact_dir)
        manifest_path = artifact_root / "manifest.json"
        manifest: Dict[str, str] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "species", {}
            )
        self.species: Dict[str, SpeciesFluxSurrogate] = {}
        self.load_errors: Dict[str, str] = {}
        for name, model in models.items():
            filename = manifest.get(name, f"{safe_species_filename(name)}.pt")
            path = artifact_root / filename
            try:
                self.species[name] = SpeciesFluxSurrogate(
                    model, path, device=device, **guard_kwargs
                )
            except (OSError, KeyError, ValueError, SurrogateUnavailable) as exc:
                self.load_errors[name] = str(exc)

    def predict(self, species_name: str, model: Any) -> SurrogatePrediction:
        surrogate = self.species.get(species_name)
        if surrogate is None:
            return SurrogatePrediction(
                solution=None,
                accepted=False,
                reason="artifact_unavailable",
                ood_score=float("inf"),
                max_bound_violation=float("inf"),
                max_mass_balance_residual=float("inf"),
                max_relative_mass_balance_residual=float("inf"),
                inference_seconds=0.0,
            )
        return surrogate.predict(model)

    def predict_feature_batch(
        self, species_name: str, features: np.ndarray
    ) -> list[SurrogatePrediction]:
        surrogate = self.species.get(species_name)
        if surrogate is None:
            rows = 1 if np.asarray(features).ndim == 1 else len(features)
            return [
                SurrogatePrediction(
                    solution=None,
                    accepted=False,
                    reason="artifact_unavailable",
                    ood_score=float("inf"),
                    max_bound_violation=float("inf"),
                    max_mass_balance_residual=float("inf"),
                    max_relative_mass_balance_residual=float("inf"),
                    inference_seconds=0.0,
                )
                for _ in range(rows)
            ]
        return surrogate.predict_features(features)

    def predict_feature_payload_batch(
        self, species_name: str, features: np.ndarray
    ) -> list[dict[str, Any]]:
        """Return service-ready payloads without serial pandas construction."""

        surrogate = self.species.get(species_name)
        if surrogate is None:
            rows = 1 if np.asarray(features).ndim == 1 else len(features)
            return [
                {
                    "solution": None,
                    "accepted": False,
                    "reason": "artifact_unavailable",
                    "ood_score": float("inf"),
                    "max_bound_violation": float("inf"),
                    "max_mass_balance_residual": float("inf"),
                    "max_relative_mass_balance_residual": float("inf"),
                    "inference_seconds": 0.0,
                }
                for _ in range(rows)
            ]
        return surrogate.predict_feature_payloads(features)


def save_surrogate_artifact(
    path: str | Path,
    model: Any,
    network: Any,
    basis: np.ndarray,
    hidden_dims: Iterable[int],
    metrics: Mapping[str, float] | None = None,
    decoder_mode: str = "nullspace",
) -> None:
    """Persist a self-validating, model-specific surrogate checkpoint."""

    torch, _ = _torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "species": model.id,
        "reaction_ids": [reaction.id for reaction in model.reactions],
        "model_fingerprint": model_fingerprint(model),
        "input_dim": 3 * len(model.reactions) + 1,
        "latent_dim": int(np.asarray(basis).shape[1]),
        "hidden_dims": [int(v) for v in hidden_dims],
        "conservation": "hard_nullspace_decoder",
        "decoder_mode": decoder_mode,
        "metrics": dict(metrics or {}),
    }
    torch.save(
        {
            "metadata": metadata,
            "basis": np.asarray(basis, dtype=np.float32),
            "state_dict": {
                key: value.detach().cpu() for key, value in network.state_dict().items()
            },
        },
        target,
    )
