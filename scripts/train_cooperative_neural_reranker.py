#!/usr/bin/env python3
"""Train a neural phenotype head that reranks exact feasible flux states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cooperative_neural_surrogate import (
    ARTIFACT_VERSION,
    RERANK_FORMULATION,
    make_cooperative_network,
)


def _chunk_range(array, chunk: int = 1024):
    minimum = np.full(array.shape[1], np.inf, dtype=np.float32)
    maximum = np.full(array.shape[1], -np.inf, dtype=np.float32)
    for start in range(0, len(array), chunk):
        values = np.asarray(array[start : start + chunk], dtype=np.float32)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
    return minimum, maximum


def _load_dataset(dataset: Path):
    """Load either the production NPY directory or a compact NPZ smoke set."""

    if dataset.is_dir():
        contexts = np.load(dataset / "contexts.npy", mmap_mode="r")
        fluxes = np.load(dataset / "fluxes.npy", mmap_mode="r")
        metadata = json.loads(
            (dataset / "metadata.json").read_text(encoding="utf-8")
        )
        return contexts, fluxes, metadata
    if dataset.suffix.lower() != ".npz":
        raise ValueError("dataset must be an NPY directory or .npz archive")
    archive = np.load(dataset)
    metadata_path = dataset.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return archive["contexts"], archive["fluxes"], metadata


def decision_intervals(features, feature_indices, context_template, metadata, decision_indices):
    """Recover live decision bounds from compact context without dense copies."""
    starts, cursor = {}, 0
    for name, width in metadata["context_layout"]:
        starts[name] = cursor
        cursor += int(width)
    selected = {int(global_index): local for local, global_index in enumerate(feature_indices)}
    outputs = []
    for kind in ("flux_lower_bounds", "flux_upper_bounds"):
        positions = starts[kind] + np.asarray(decision_indices)
        values = np.broadcast_to(context_template[positions], (len(features), len(positions))).copy()
        for local, global_index in enumerate(positions):
            if int(global_index) in selected:
                values[:, local] = features[:, selected[int(global_index)]]
        outputs.append(values.astype(np.float32))
    if np.any(outputs[0] > outputs[1]):
        raise ValueError("reconstructed live decision intervals are inconsistent")
    return outputs


def interval_violation(prediction, lower, upper):
    """Squared scaled hinge, differentiable also for fixed/closed reactions."""
    return torch.relu(lower-prediction).square() + torch.relu(prediction-upper).square()


def check_warm_start(payload, feature_indices, decision_indices, metadata, hidden_dims):
    for key in ("species", "reaction_ids", "model_fingerprints", "cooperative_optimize_live_objectives"):
        if payload["metadata"].get(key) != metadata.get(key):
            raise ValueError(f"warm-start {key} mismatch")
    for key, expected in (("feature_indices", feature_indices), ("decision_indices", decision_indices)):
        if not np.array_equal(payload[key], expected):
            raise ValueError(f"warm-start {key} mismatch")
    if list(payload["metadata"]["hidden_dims"]) != list(hidden_dims):
        raise ValueError("warm-start network shape mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "models" / "cooperative_surrogate" / "production_32768",
    )
    parser.add_argument("--base-dictionary", type=Path, required=True)
    parser.add_argument(
        "--augmentation-dictionary",
        type=Path,
        default=None,
        help=(
            "dictionary derived from --base-dictionary; rows appended after the "
            "dataset are also used as supervised neural training examples"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rerank-pool", type=int, default=128)
    parser.add_argument("--decision-strength", type=float, default=4.0)
    parser.add_argument("--critical-weight", type=float, default=8.0)
    parser.add_argument("--pha-weight", type=float, default=24.0)
    parser.add_argument("--augmentation-weight", type=float, default=1.0)
    parser.add_argument("--bounds-weight", type=float, default=0.0)
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="continue an existing head while retaining its exact normalization")
    args = parser.parse_args()
    if (not np.isfinite(args.augmentation_weight) or args.augmentation_weight <= 0
        or not np.isfinite(args.bounds_weight) or args.bounds_weight < 0):
        raise ValueError("invalid augmentation/bounds loss weight")

    dataset = args.dataset
    contexts, fluxes, metadata = _load_dataset(dataset)
    context_min, context_max = _chunk_range(contexts)
    flux_min, flux_max = _chunk_range(fluxes)
    feature_indices = np.flatnonzero(context_max - context_min > 1e-8)
    reaction_ids = list(metadata["reaction_ids"])
    decision_indices = np.asarray(
        [
            index
            for index, reaction_id in enumerate(reaction_ids)
            if flux_max[index] - flux_min[index] > 1e-7
            and (
                reaction_id.startswith("EX_")
                or any(
                    token in reaction_id.lower()
                    for token in ("growth", "biomass")
                )
            )
        ],
        dtype=np.int64,
    )
    critical = np.asarray(
        [
            any(token in reaction_ids[index].lower() for token in ("growth", "biomass"))
            or (
                reaction_ids[index].startswith("EX_")
                and any(
                    token in reaction_ids[index].lower()
                    for token in ("pha", "phb", "phv")
                )
            )
            for index in decision_indices
        ],
        dtype=bool,
    )
    pha = np.asarray(
        [
            reaction_ids[index].startswith("EX_")
            and any(
                token in reaction_ids[index].lower()
                for token in ("pha", "phb", "phv")
            )
            for index in decision_indices
        ],
        dtype=bool,
    )
    if (
        any("ns21" in str(species).lower() for species in metadata.get("species", []))
        and not np.any(pha)
    ):
        raise ValueError(
            "NS21 dataset contains no variable PHB/PHV decision flux; "
            "recollect with live-objective optimization and nitrogen-phase sampling"
        )

    supervised_x = np.asarray(contexts[:, feature_indices], dtype=np.float32)
    warm = None
    if args.warm_start is not None:
        warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        check_warm_start(warm, feature_indices, decision_indices, metadata, args.hidden_dims)
    supervised_y = np.asarray(fluxes[:, decision_indices], dtype=np.float32)
    augmentation_count = 0
    if args.augmentation_dictionary is not None:
        augmentation = torch.load(
            args.augmentation_dictionary, map_location="cpu", weights_only=False
        )
        for key in ("species", "reaction_ids", "model_fingerprints", "cooperative_optimize_live_objectives"):
            if augmentation["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"augmentation {key} differs from the dataset")
        augmentation_feature_indices = np.asarray(
            augmentation["feature_indices"], dtype=np.int64
        )
        if not np.array_equal(augmentation_feature_indices, feature_indices):
            raise ValueError(
                "augmentation dictionary feature layout differs from the dataset"
            )
        normalized_contexts = np.asarray(
            augmentation["normalized_contexts"], dtype=np.float32
        )
        augmentation_fluxes = np.asarray(augmentation["fluxes"], dtype=np.float32)
        base_count = len(contexts)
        if len(normalized_contexts) <= base_count:
            raise ValueError("augmentation dictionary contains no appended rows")
        dictionary_mean = np.asarray(augmentation["feature_mean"], dtype=np.float32)
        dictionary_scale = np.asarray(augmentation["feature_scale"], dtype=np.float32)
        dictionary_weight = np.asarray(
            augmentation.get(
                "feature_weight", np.ones(len(feature_indices), dtype=np.float32)
            ),
            dtype=np.float32,
        )
        appended_x = (
            normalized_contexts[base_count:]
            / np.sqrt(dictionary_weight)[None, :]
            * dictionary_scale[None, :]
            + dictionary_mean[None, :]
        ).astype(np.float32)
        appended_y = augmentation_fluxes[base_count:, decision_indices]
        supervised_x = np.concatenate((supervised_x, appended_x), axis=0)
        supervised_y = np.concatenate((supervised_y, appended_y), axis=0)
        augmentation_count = len(appended_x)

    lower, upper = decision_intervals(supervised_x, feature_indices, np.asarray(contexts[0]),
                                     metadata, decision_indices)
    sample_weight = np.ones(len(supervised_x), dtype=np.float32)
    if augmentation_count:
        sample_weight[-augmentation_count:] = args.augmentation_weight

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(supervised_x))
    if len(order) < 2:
        raise ValueError("at least two samples are required for training")
    validation_count = min(
        len(order) - 1,
        max(1, int(round(0.2 * len(order)))),
    )
    validation_indices = np.sort(order[:validation_count])
    training_indices = np.sort(order[validation_count:])
    training_x = supervised_x[training_indices]
    validation_x = supervised_x[validation_indices]
    feature_mean = training_x.mean(axis=0)
    feature_scale = np.maximum.reduce(
        (
            training_x.std(axis=0),
            np.ptp(training_x, axis=0) / 4.0,
            np.full(len(feature_indices), 1e-6, dtype=np.float32),
        )
    ).astype(np.float32)
    if warm is not None:
        feature_mean = np.asarray(warm["feature_mean"], dtype=np.float32)
        feature_scale = np.asarray(warm["feature_scale"], dtype=np.float32)
    training_x = ((training_x - feature_mean) / feature_scale).astype(np.float32)
    validation_x = ((validation_x - feature_mean) / feature_scale).astype(np.float32)
    training_y = supervised_y[training_indices]
    validation_y = supervised_y[validation_indices]
    decision_mean = training_y.mean(axis=0)
    decision_scale = np.maximum.reduce(
        (
            training_y.std(axis=0),
            np.ptp(training_y, axis=0) / 20.0,
            np.full(len(decision_indices), 1e-5, dtype=np.float32),
        )
    ).astype(np.float32)
    if warm is not None:
        decision_mean = np.asarray(warm["decision_mean"], dtype=np.float32)
        decision_scale = np.asarray(warm["decision_scale"], dtype=np.float32)
    training_y = ((training_y - decision_mean) / decision_scale).astype(np.float32)
    validation_y = ((validation_y - decision_mean) / decision_scale).astype(np.float32)
    lower = ((lower-decision_mean)/decision_scale).astype(np.float32)
    upper = ((upper-decision_mean)/decision_scale).astype(np.float32)
    decision_weight = np.ones(len(decision_indices), dtype=np.float32)
    decision_weight[critical] = float(args.critical_weight)
    decision_weight[pha] = float(args.pha_weight)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    network = make_cooperative_network(
        len(feature_indices), len(decision_indices), tuple(args.hidden_dims)
    ).to(device)
    if warm is not None:
        network.load_state_dict(warm["state_dict"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    weight_t = torch.as_tensor(decision_weight, device=device)
    best_state = None
    best_score = float("inf")
    initial_score = None
    if warm is not None:
        network.eval()
        with torch.inference_mode():
            initial = network(torch.as_tensor(validation_x, device=device)).cpu().numpy()
        mask = critical if critical.any() else np.ones(len(decision_indices), dtype=bool)
        initial_score = float(np.sqrt(np.mean((initial[:, mask]-validation_y[:, mask])**2)))
        best_score = initial_score
        best_state = {key:value.detach().cpu().clone() for key, value in network.state_dict().items()}
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        network.train()
        permutation = torch.randperm(len(training_indices), generator=generator)
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start : start + args.batch_size].numpy()
            x = torch.as_tensor(training_x[batch], device=device)
            target = torch.as_tensor(training_y[batch], device=device)
            prediction = network(x)
            per_output = F.smooth_l1_loss(prediction, target, reduction="none")
            if args.bounds_weight > 0:
                live_lower = torch.as_tensor(lower[training_indices[batch]], device=device)
                live_upper = torch.as_tensor(upper[training_indices[batch]], device=device)
                per_output = per_output + args.bounds_weight*interval_violation(prediction, live_lower, live_upper)
            row_weight = torch.as_tensor(sample_weight[training_indices[batch]], device=device)
            loss = ((per_output*weight_t).mean(dim=1)*row_weight).sum()/row_weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            network.eval()
            with torch.inference_mode():
                predicted = network(torch.as_tensor(validation_x, device=device)).cpu().numpy()
            score_mask = critical if np.any(critical) else np.ones(
                len(decision_indices), dtype=bool
            )
            critical_rmse = float(
                np.sqrt(
                    np.mean(
                        (predicted[:, score_mask] - validation_y[:, score_mask]) ** 2
                    )
                )
            )
            if critical_rmse < best_score:
                best_score = critical_rmse
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            print(f"epoch {epoch + 1:03d}/{args.epochs}: critical_scaled_rmse={critical_rmse:.6g}", flush=True)

    assert best_state is not None
    network.load_state_dict(best_state)
    network.eval().to(device)
    with torch.inference_mode():
        predicted_scaled = network(torch.as_tensor(validation_x, device=device)).cpu().numpy()
    predicted = predicted_scaled * decision_scale + decision_mean
    exact = validation_y * decision_scale + decision_mean
    metrics = {
        "validation_split": "random_samples_not_independent_trajectories",
        "augmentation_weight":args.augmentation_weight,
        "bounds_weight":args.bounds_weight,
        "warm_start":str(args.warm_start) if args.warm_start else None,
        "initial_validation_critical_scaled_rmse":initial_score,
        "selected_validation_critical_scaled_rmse":best_score,
        "validation_scaled_bound_violation_rmse":float(np.sqrt(np.mean(
            np.maximum(lower[validation_indices]-predicted_scaled, 0)**2 +
            np.maximum(predicted_scaled-upper[validation_indices], 0)**2))),
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "augmentation_samples": int(augmentation_count),
        "selected_feature_count": int(len(feature_indices)),
        "decision_flux_count": int(len(decision_indices)),
        "critical_flux_count": int(np.sum(critical)),
        "pha_flux_count": int(np.sum(pha)),
        "validation_decision_rmse": float(np.sqrt(np.mean((predicted - exact) ** 2))),
        "validation_critical_rmse": float(
            np.sqrt(np.mean((predicted[:, critical] - exact[:, critical]) ** 2))
        ) if np.any(critical) else None,
        "validation_pha_rmse": float(np.sqrt(np.mean((predicted[:, pha] - exact[:, pha]) ** 2))) if np.any(pha) else 0.0,
        "training_seconds": float(time.perf_counter() - started),
    }
    try:
        relative_dictionary = args.base_dictionary.resolve().relative_to(args.output.parent.resolve())
        dictionary_value = str(relative_dictionary)
    except ValueError:
        dictionary_value = str(args.base_dictionary.resolve())
    artifact_metadata = dict(metadata)
    artifact_metadata.update(
        {
            "format_version": ARTIFACT_VERSION,
            "formulation": RERANK_FORMULATION,
            "base_dictionary": dictionary_value,
            "context_dimension": int(contexts.shape[1]),
            "hidden_dims": [int(value) for value in args.hidden_dims],
            "rerank_pool": int(args.rerank_pool),
            "decision_strength": float(args.decision_strength),
            "critical_weight": float(args.critical_weight),
            "pha_weight": float(args.pha_weight),
            "metrics": metrics,
            "training_recipe": {"dataset":str(args.dataset),
                "augmentation_dictionary":str(args.augmentation_dictionary),
                "seed":args.seed, "epochs":args.epochs,
                "warm_start":str(args.warm_start) if args.warm_start else None,
                "learning_rate":args.learning_rate,
                "augmentation_weight":args.augmentation_weight, "bounds_weight":args.bounds_weight},
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": artifact_metadata,
            "feature_indices": feature_indices.astype(np.int64),
            "feature_mean": feature_mean.astype(np.float32),
            "feature_scale": feature_scale.astype(np.float32),
            "decision_indices": decision_indices,
            "decision_mean": decision_mean.astype(np.float32),
            "decision_scale": decision_scale,
            "decision_weight": decision_weight,
            "state_dict": best_state,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
