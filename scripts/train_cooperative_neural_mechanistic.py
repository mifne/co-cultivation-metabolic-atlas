#!/usr/bin/env python3
"""Train an AMN-like GPU emulator from exact cooperative-FBA trajectories.

The low-rank decoder is fitted to exact community fluxes.  Because centring
and PCA are linear operations on feasible fluxes, the decoder remains in the
block stoichiometric null space.  The neural encoder learns only the mapping
from the live LP context to decoder coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.extmath import randomized_svd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cooperative_neural_surrogate import (
    ARTIFACT_VERSION,
    FORMULATION,
    make_cooperative_network,
)


def _load_dataset(path: Path):
    if not path.is_dir():
        raise ValueError("the neural trainer currently requires an npy-directory dataset")
    arrays = {
        name: np.load(path / f"{name}.npy", mmap_mode="r")
        for name in ("contexts", "fluxes", "common_growth", "aggregate_objective")
    }
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return arrays, metadata


def _chunk_range(array, chunk: int = 1024):
    minimum = np.full(array.shape[1], np.inf, dtype=np.float32)
    maximum = np.full(array.shape[1], -np.inf, dtype=np.float32)
    for start in range(0, len(array), chunk):
        values = np.asarray(array[start : start + chunk], dtype=np.float32)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
    return minimum, maximum


def _layout_offsets(metadata: dict) -> dict[str, tuple[int, int]]:
    result = {}
    cursor = 0
    for name, width in metadata["context_layout"]:
        result[str(name)] = (cursor, cursor + int(width))
        cursor += int(width)
    return result


def _relevant_indices(reaction_ids: list[str]) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, reaction_id in enumerate(reaction_ids)
            if reaction_id.startswith("EX_")
            or any(
                token in reaction_id.lower()
                for token in ("growth", "biomass", "pha", "phb", "lcp", "roxa", "roxb")
            )
        ],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "models" / "cooperative_surrogate" / "production_32768",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--pca-samples", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ood-tolerance", type=float, default=0.25)
    parser.add_argument("--feature-epsilon", type=float, default=1e-8)
    parser.add_argument("--flux-epsilon", type=float, default=1e-7)
    args = parser.parse_args()

    arrays, metadata = _load_dataset(args.dataset)
    contexts = arrays["contexts"]
    fluxes = arrays["fluxes"]
    if len(contexts) != len(fluxes) or len(contexts) < 64:
        raise ValueError("at least 64 aligned exact context/flux rows are required")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(contexts))
    validation_count = max(64, int(round(0.2 * len(order))))
    validation_indices = np.sort(order[:validation_count])
    training_indices = np.sort(order[validation_count:])

    print("scanning varying context and flux columns", flush=True)
    context_min_full, context_max_full = _chunk_range(contexts)
    flux_min, flux_max = _chunk_range(fluxes)
    feature_indices = np.flatnonzero(
        context_max_full - context_min_full > args.feature_epsilon
    )
    varying_flux_indices = np.flatnonzero(flux_max - flux_min > args.flux_epsilon)
    if not len(feature_indices) or not len(varying_flux_indices):
        raise ValueError("dataset has no varying inputs or outputs")

    training_x = np.asarray(contexts[training_indices][:, feature_indices], dtype=np.float32)
    validation_x = np.asarray(contexts[validation_indices][:, feature_indices], dtype=np.float32)
    feature_mean = training_x.mean(axis=0)
    feature_scale = np.maximum.reduce(
        (
            training_x.std(axis=0),
            np.ptp(training_x, axis=0) / 4.0,
            np.full(len(feature_indices), 1e-6, dtype=np.float32),
        )
    ).astype(np.float32)
    feature_min = training_x.min(axis=0)
    feature_max = training_x.max(axis=0)
    training_x = ((training_x - feature_mean) / feature_scale).astype(np.float32)
    validation_x = ((validation_x - feature_mean) / feature_scale).astype(np.float32)

    pca_count = min(int(args.pca_samples), len(training_indices))
    pca_indices = np.sort(rng.choice(training_indices, pca_count, replace=False))
    pca_values = np.asarray(
        fluxes[pca_indices][:, varying_flux_indices], dtype=np.float32
    )
    varying_mean = pca_values.mean(axis=0)
    pca_values -= varying_mean
    rank = min(int(args.rank), min(pca_values.shape) - 1)
    print(f"randomized SVD: {pca_values.shape}, rank={rank}", flush=True)
    pca_started = time.perf_counter()
    _, singular_values, components = randomized_svd(
        pca_values,
        n_components=rank,
        n_iter=5,
        random_state=args.seed,
    )
    pca_seconds = time.perf_counter() - pca_started
    del pca_values

    flux_mean = np.zeros(fluxes.shape[1], dtype=np.float32)
    # Constant columns use their exact value; varying columns use the PCA mean.
    flux_mean[:] = np.asarray(fluxes[pca_indices[0]], dtype=np.float32)
    flux_mean[varying_flux_indices] = varying_mean
    flux_basis = np.zeros((rank, fluxes.shape[1]), dtype=np.float32)
    flux_basis[:, varying_flux_indices] = components

    print("projecting exact labels into the mechanistic flux basis", flush=True)
    training_y_varying = np.asarray(
        fluxes[training_indices][:, varying_flux_indices], dtype=np.float32
    )
    validation_y_varying = np.asarray(
        fluxes[validation_indices][:, varying_flux_indices], dtype=np.float32
    )
    training_z = (training_y_varying - varying_mean) @ components.T
    validation_z = (validation_y_varying - varying_mean) @ components.T
    latent_mean = training_z.mean(axis=0).astype(np.float32)
    latent_scale = np.maximum(training_z.std(axis=0), 1e-6).astype(np.float32)
    training_z = ((training_z - latent_mean) / latent_scale).astype(np.float32)
    validation_z = ((validation_z - latent_mean) / latent_scale).astype(np.float32)

    relevant_global = _relevant_indices(list(metadata["reaction_ids"]))
    varying_position = {int(index): pos for pos, index in enumerate(varying_flux_indices)}
    relevant_global = np.asarray(
        [index for index in relevant_global if int(index) in varying_position],
        dtype=np.int64,
    )
    relevant_varying = np.asarray(
        [varying_position[int(index)] for index in relevant_global], dtype=np.int64
    )
    training_relevant = training_y_varying[:, relevant_varying]
    validation_relevant = validation_y_varying[:, relevant_varying]
    relevant_scale = np.maximum(
        np.std(training_relevant, axis=0),
        np.maximum(np.abs(training_relevant).mean(axis=0), 1.0) * 0.05,
    ).astype(np.float32)

    layout = _layout_offsets(metadata)
    lower_start, _ = layout["flux_lower_bounds"]
    upper_start, _ = layout["flux_upper_bounds"]
    objective_start, _ = layout["objective_coefficients"]
    training_lower = np.asarray(
        contexts[training_indices][:, lower_start + varying_flux_indices], dtype=np.float32
    )
    training_upper = np.asarray(
        contexts[training_indices][:, upper_start + varying_flux_indices], dtype=np.float32
    )
    validation_lower = np.asarray(
        contexts[validation_indices][:, lower_start + varying_flux_indices], dtype=np.float32
    )
    validation_upper = np.asarray(
        contexts[validation_indices][:, upper_start + varying_flux_indices], dtype=np.float32
    )
    objective_columns = np.flatnonzero(
        np.any(
            np.asarray(
                contexts[:, objective_start : objective_start + fluxes.shape[1]],
                dtype=np.float32,
            )
            != 0.0,
            axis=0,
        )
    )
    objective_varying = np.asarray(
        [varying_position[int(index)] for index in objective_columns if int(index) in varying_position],
        dtype=np.int64,
    )
    training_objective = np.asarray(
        contexts[training_indices][:, objective_start + objective_columns], dtype=np.float32
    )
    validation_objective = np.asarray(
        contexts[validation_indices][:, objective_start + objective_columns], dtype=np.float32
    )
    objective_column_positions = np.asarray(
        [varying_position[int(index)] for index in objective_columns], dtype=np.int64
    )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    network = make_cooperative_network(
        len(feature_indices), rank, tuple(args.hidden_dims)
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    basis_varying_t = torch.as_tensor(components, device=device)
    mean_varying_t = torch.as_tensor(varying_mean, device=device)
    latent_mean_t = torch.as_tensor(latent_mean, device=device)
    latent_scale_t = torch.as_tensor(latent_scale, device=device)
    relevant_varying_t = torch.as_tensor(relevant_varying, device=device)
    relevant_scale_t = torch.as_tensor(relevant_scale, device=device)
    objective_positions_t = torch.as_tensor(objective_column_positions, device=device)

    best_state = None
    best_score = float("inf")
    training_started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for epoch in range(args.epochs):
        network.train()
        permutation = torch.randperm(len(training_indices), generator=generator)
        total_loss = 0.0
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start : start + args.batch_size].numpy()
            x = torch.as_tensor(training_x[batch], device=device)
            z_target = torch.as_tensor(training_z[batch], device=device)
            lower = torch.as_tensor(training_lower[batch], device=device)
            upper = torch.as_tensor(training_upper[batch], device=device)
            target_relevant = torch.as_tensor(training_relevant[batch], device=device)
            objective = torch.as_tensor(training_objective[batch], device=device)

            z_normalized = network(x)
            z = z_normalized * latent_scale_t + latent_mean_t
            predicted_varying = mean_varying_t + z @ basis_varying_t
            predicted_relevant = predicted_varying.index_select(1, relevant_varying_t)
            latent_loss = F.smooth_l1_loss(z_normalized, z_target)
            relevant_loss = torch.mean(
                ((predicted_relevant - target_relevant) / relevant_scale_t).square()
            )
            bound_loss = (
                F.relu(lower - predicted_varying).square().mean()
                + F.relu(predicted_varying - upper).square().mean()
            )
            predicted_objective = torch.sum(
                objective * predicted_varying.index_select(1, objective_positions_t), dim=1
            )
            exact_objective = torch.sum(
                objective
                * torch.as_tensor(
                    training_y_varying[batch][:, objective_column_positions], device=device
                ),
                dim=1,
            )
            objective_scale = torch.clamp(torch.abs(exact_objective), min=1.0)
            objective_loss = torch.mean(
                ((predicted_objective - exact_objective) / objective_scale).square()
            )
            loss = latent_loss + 8.0 * relevant_loss + 2.0 * objective_loss + 0.1 * bound_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            network.eval()
            with torch.inference_mode():
                validation_prediction_parts = []
                for start in range(0, len(validation_x), 2048):
                    x = torch.as_tensor(validation_x[start : start + 2048], device=device)
                    z = network(x) * latent_scale_t + latent_mean_t
                    prediction = mean_varying_t + z @ basis_varying_t
                    validation_prediction_parts.append(prediction.cpu().numpy())
                validation_prediction = np.concatenate(validation_prediction_parts)
            relevant_rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            validation_prediction[:, relevant_varying]
                            - validation_relevant
                        )
                        ** 2
                    )
                )
            )
            if relevant_rmse < best_score:
                best_score = relevant_rmse
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
            print(
                f"epoch {epoch + 1:03d}/{args.epochs}: "
                f"loss={total_loss / len(training_indices):.6g}, "
                f"relevant_rmse={relevant_rmse:.6g}",
                flush=True,
            )

    assert best_state is not None
    network.load_state_dict(best_state)
    network.eval()
    with torch.inference_mode():
        predictions = []
        for start in range(0, len(validation_x), 2048):
            x = torch.as_tensor(validation_x[start : start + 2048], device=device)
            z = network(x) * latent_scale_t + latent_mean_t
            predictions.append((mean_varying_t + z @ basis_varying_t).cpu().numpy())
    validation_prediction = np.concatenate(predictions)
    bound_violation = np.maximum(
        validation_lower - validation_prediction,
        validation_prediction - validation_upper,
    ).max(axis=1)
    predicted_objective = np.sum(
        validation_objective
        * validation_prediction[:, objective_column_positions],
        axis=1,
    )
    exact_objective = np.sum(
        validation_objective
        * validation_y_varying[:, objective_column_positions],
        axis=1,
    )
    pca_reconstruction = (
        (validation_y_varying - varying_mean) @ components.T
    ) @ components + varying_mean
    metrics = {
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "selected_feature_count": int(len(feature_indices)),
        "varying_flux_count": int(len(varying_flux_indices)),
        "latent_rank": int(rank),
        "pca_seconds": float(pca_seconds),
        "training_seconds": float(time.perf_counter() - training_started),
        "validation_relevant_flux_rmse": float(best_score),
        "validation_objective_mae": float(np.mean(np.abs(predicted_objective - exact_objective))),
        "validation_bound_feasible_fraction": float(np.mean(bound_violation <= 1e-4)),
        "validation_max_bound_violation": float(np.max(np.maximum(bound_violation, 0.0))),
        "validation_pca_relevant_flux_rmse": float(
            np.sqrt(
                np.mean(
                    (pca_reconstruction[:, relevant_varying] - validation_relevant) ** 2
                )
            )
        ),
    }
    artifact_metadata = dict(metadata)
    artifact_metadata.update(
        {
            "format_version": ARTIFACT_VERSION,
            "formulation": FORMULATION,
            "decoder": "affine_exact_flux_subspace",
            "conservation": "hard_block_stoichiometric_nullspace_from_exact_fluxes",
            "context_dimension": int(contexts.shape[1]),
            "flux_dimension": int(fluxes.shape[1]),
            "hidden_dims": [int(value) for value in args.hidden_dims],
            "latent_rank": int(rank),
            "ood_tolerance": float(args.ood_tolerance),
            "source_dataset": str(args.dataset),
            "metrics": metrics,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": artifact_metadata,
            "feature_indices": feature_indices.astype(np.int64),
            "feature_mean": feature_mean.astype(np.float32),
            "feature_scale": feature_scale.astype(np.float32),
            "feature_min": feature_min.astype(np.float32),
            "feature_max": feature_max.astype(np.float32),
            "latent_mean": latent_mean.astype(np.float32),
            "latent_scale": latent_scale.astype(np.float32),
            "flux_mean": flux_mean,
            "flux_basis": flux_basis,
            "state_dict": best_state,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
