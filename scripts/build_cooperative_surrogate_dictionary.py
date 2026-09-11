#!/usr/bin/env python3
"""Build a GPU-resident aligned flux dictionary from exact cooperative labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _layout_offsets(metadata: dict) -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, width in metadata["context_layout"]:
        width = int(width)
        offsets[str(name)] = (cursor, cursor + width)
        cursor += width
    return offsets


def decision_aware_feature_weights(
    contexts: np.ndarray,
    fluxes: np.ndarray,
    metadata: dict,
    feature_indices: np.ndarray,
    common_growth: np.ndarray | None,
    aggregate_objective: np.ndarray | None,
    strength: float = 3.0,
    critical_floor: float = 4.0,
    maximum: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """Weight state variables that determine metabolic regime and LP decisions.

    The continuous component uses feature/decision correlations.  A conservative
    floor is then applied to known limitation variables and to the bounds and
    objective coefficients of PHA, growth and rubber-cleavage reactions.  This
    prevents a rare metabolic switch from being diluted by dozens of unrelated
    varying bounds in an ordinary Euclidean distance.
    """

    selected = np.asarray(contexts[:, feature_indices], dtype=np.float64)
    selected -= selected.mean(axis=0, keepdims=True)
    selected_scale = selected.std(axis=0)
    selected_scale[selected_scale < 1e-12] = 1.0
    selected /= selected_scale

    reaction_ids = list(metadata["reaction_ids"])
    decision_reactions = [
        index
        for index, reaction_id in enumerate(reaction_ids)
        if any(
            token in reaction_id.lower()
            for token in (
                "ex_pha", "ex_phv", "phb_syn", "phv_syn", "r_lcp",
                "r_rox", "biomass", "growth",
            )
        )
    ]
    targets: list[np.ndarray] = []
    if common_growth is not None:
        targets.append(np.asarray(common_growth, dtype=np.float64))
    if aggregate_objective is not None:
        targets.append(np.asarray(aggregate_objective, dtype=np.float64))
    for index in decision_reactions:
        column = np.asarray(fluxes[:, index], dtype=np.float64)
        if np.ptp(column) > 1e-10:
            targets.append(column)

    association = np.zeros(len(feature_indices), dtype=np.float64)
    for target in targets:
        centered = target - target.mean()
        scale = centered.std()
        if scale < 1e-12:
            continue
        correlation = np.abs(selected.T @ (centered / scale)) / max(1, len(target) - 1)
        association = np.maximum(association, np.clip(correlation, 0.0, 1.0))
    weights = 1.0 + float(strength) * association

    offsets = _layout_offsets(metadata)
    critical_global_indices: set[int] = set()
    supply_start, _ = offsets["shared_supply_mmol_l_h"]
    for local_index, metabolite_id in enumerate(metadata["shared_metabolite_ids"]):
        if any(
            token in metabolite_id.lower()
            for token in ("nh4", "o2", "glc", "malt", "mann", "ptrc", "rubber")
        ):
            critical_global_indices.add(supply_start + local_index)
    for layout_name in ("flux_lower_bounds", "flux_upper_bounds", "objective_coefficients"):
        start, _ = offsets[layout_name]
        critical_global_indices.update(start + index for index in decision_reactions)
    feature_position = {int(index): pos for pos, index in enumerate(feature_indices)}
    emphasized = 0
    for global_index in critical_global_indices:
        position = feature_position.get(global_index)
        if position is not None:
            weights[position] = max(weights[position], float(critical_floor))
            emphasized += 1
    weights = np.clip(weights, 1.0, float(maximum)).astype(np.float32)
    summary = {
        "method": "decision_correlation_plus_metabolic_regime_floor",
        "strength": float(strength),
        "critical_floor": float(critical_floor),
        "maximum": float(maximum),
        "decision_target_count": len(targets),
        "decision_reaction_count": len(decision_reactions),
        "critical_feature_count": emphasized,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }
    return weights, summary


def nearest_neighbour_threshold(
    normalized: np.ndarray,
    seed: int,
    sample_limit: int = 2048,
    reference_candidates: int = 8192,
    quantile: float = 0.99,
    margin: float = 1.25,
) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(
        len(normalized), size=min(sample_limit, len(normalized)), replace=False
    )
    queries = normalized[query_indices]
    nearest = []
    # Keep the OOD coverage scale approximately constant as the dictionary is
    # densified. Using the first neighbour for every N makes the threshold
    # shrink merely because more candidates were added. At 4x the pilot size,
    # the fourth neighbour represents the same effective 8,192-state density.
    neighbour_rank = max(1, int(np.ceil(len(normalized) / reference_candidates)))
    chunk = 128
    for start in range(0, len(queries), chunk):
        query = queries[start : start + chunk]
        distance = (
            np.sum(query * query, axis=1, keepdims=True)
            + np.sum(normalized * normalized, axis=1)[None, :]
            - 2.0 * query @ normalized.T
        )
        for row, own_index in enumerate(query_indices[start : start + chunk]):
            distance[row, own_index] = np.inf
        ranked = np.partition(distance, neighbour_rank - 1, axis=1)[
            :, neighbour_rank - 1
        ]
        nearest.extend(np.sqrt(np.maximum(0.0, ranked)))
    # A modest margin prevents numerical rejection at the training boundary;
    # exact guards and audit/fallback remain mandatory at runtime.
    return float(np.quantile(nearest, quantile) * margin), neighbour_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variance-epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--threshold-margin", type=float, default=1.25)
    parser.add_argument(
        "--decision-aware",
        action="store_true",
        help="weight PHA/nutrient/rubber regime variables and decision-sensitive features",
    )
    parser.add_argument("--decision-weight-strength", type=float, default=3.0)
    parser.add_argument("--critical-weight-floor", type=float, default=4.0)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="build a reproducible random subset, e.g. 8192 from production data",
    )
    args = parser.parse_args()
    if not 0.0 < args.threshold_quantile <= 1.0:
        parser.error("--threshold-quantile must be in (0, 1]")
    if args.threshold_margin <= 0.0:
        parser.error("--threshold-margin must be positive")
    metadata_path = args.metadata or (
        args.dataset / "metadata.json"
        if args.dataset.is_dir()
        else args.dataset.with_suffix(".json")
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.dataset.is_dir():
        contexts = np.load(args.dataset / "contexts.npy", mmap_mode="r")
        fluxes = np.load(args.dataset / "fluxes.npy", mmap_mode="r")
        common_growth = (
            np.load(args.dataset / "common_growth.npy", mmap_mode="r")
            if (args.dataset / "common_growth.npy").exists()
            else None
        )
        aggregate_objective = (
            np.load(args.dataset / "aggregate_objective.npy", mmap_mode="r")
            if (args.dataset / "aggregate_objective.npy").exists()
            else None
        )
    else:
        archive = np.load(args.dataset)
        contexts = np.asarray(archive["contexts"], dtype=np.float32)
        fluxes = np.asarray(archive["fluxes"], dtype=np.float32)
        common_growth = archive.get("common_growth")
        aggregate_objective = archive.get("aggregate_objective")
    source_count = len(contexts)
    subset_indices = None
    if args.max_candidates is not None and args.max_candidates < source_count:
        if args.max_candidates < 8:
            raise ValueError("--max-candidates must be at least eight")
        rng = np.random.default_rng(args.seed)
        subset_indices = np.sort(
            rng.choice(source_count, size=args.max_candidates, replace=False)
        )
        contexts = contexts[subset_indices]
        fluxes = fluxes[subset_indices]
        if common_growth is not None:
            common_growth = common_growth[subset_indices]
        if aggregate_objective is not None:
            aggregate_objective = aggregate_objective[subset_indices]
    if len(contexts) < 8 or len(contexts) != len(fluxes):
        raise ValueError("at least eight aligned context/flux rows are required")
    feature_range = np.ptp(contexts, axis=0)
    feature_indices = np.flatnonzero(feature_range > args.variance_epsilon)
    if not len(feature_indices):
        raise ValueError("no varying context features were found")
    selected = contexts[:, feature_indices]
    feature_mean = selected.mean(axis=0)
    # Rare on/off bounds have a tiny standard deviation even though their
    # observed range is large. Pure z-scoring lets one legitimate switch
    # dominate the whole community distance. A four-sigma range floor keeps
    # every observed feature span finite while retaining variance weighting.
    feature_scale = np.maximum.reduce(
        (
            selected.std(axis=0),
            np.ptp(selected, axis=0) / 4.0,
            np.full(selected.shape[1], 1e-6, dtype=np.float32),
        )
    )
    feature_weight = np.ones(len(feature_indices), dtype=np.float32)
    weight_summary = {"method": "uniform", "weight_min": 1.0, "weight_max": 1.0}
    if args.decision_aware:
        feature_weight, weight_summary = decision_aware_feature_weights(
            contexts,
            fluxes,
            metadata,
            feature_indices,
            common_growth,
            aggregate_objective,
            strength=args.decision_weight_strength,
            critical_floor=args.critical_weight_floor,
        )
    normalized = (
        (selected - feature_mean) / feature_scale * np.sqrt(feature_weight)
    ).astype(np.float32)
    threshold, threshold_neighbour_rank = nearest_neighbour_threshold(
        normalized,
        args.seed,
        quantile=args.threshold_quantile,
        margin=args.threshold_margin,
    )
    artifact_metadata = dict(metadata)
    artifact_metadata.update(
        {
            "format_version": 1,
            "formulation": "exact_cooperative_shared_medium",
            "candidate_count": int(len(fluxes)),
            "selected_feature_count": int(len(feature_indices)),
            "distance_metric": "normalized_euclidean",
            "normalization": "max(std, observed_range/4, 1e-6)",
            "feature_weighting": weight_summary,
            "distance_threshold": threshold,
            "threshold_neighbour_rank": threshold_neighbour_rank,
            "threshold_quantile": args.threshold_quantile,
            "threshold_margin": args.threshold_margin,
            "source_dataset": str(args.dataset),
            "source_candidate_count": int(source_count),
            "subset_seed": args.seed if subset_indices is not None else None,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": artifact_metadata,
            "feature_indices": feature_indices.astype(np.int64),
            "feature_mean": feature_mean.astype(np.float32),
            "feature_scale": feature_scale.astype(np.float32),
            "feature_weight": feature_weight,
            "normalized_contexts": normalized,
            "fluxes": fluxes,
        },
        args.output,
    )
    summary = {
        "artifact": str(args.output),
        "consortium_profile": metadata.get("consortium_profile"),
        "species": metadata.get("species"),
        "model_fingerprints": metadata.get("model_fingerprints"),
        "candidate_count": int(len(fluxes)),
        "context_dimension": int(contexts.shape[1]),
        "selected_feature_count": int(len(feature_indices)),
        "flux_dimension": int(fluxes.shape[1]),
        "distance_threshold": threshold,
        "threshold_neighbour_rank": threshold_neighbour_rank,
        "artifact_mib": args.output.stat().st_size / 2**20,
    }
    # Keep this distinct from the NPZ dataset sidecar (same stem is common in
    # smoke tests); otherwise building ``sample.pt`` from ``sample.npz``
    # silently destroys the dataset metadata required by neural training.
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
