#!/usr/bin/env python3
"""Add exact, boundary-focused trajectories to a cooperative GPU dictionary."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cooperative_surrogate import CooperativeFluxDictionary
from src.fba_surrogate import model_fingerprint
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.utils import get_initial_params, load_sbml_models, select_consortium_models
from scripts.benchmark_cooperative_surrogate_e2e import (
    CONSORTIUM_PROFILES, load_consortium_profile, make_environment,
)

MODEL_DIR = PROJECT_ROOT / "models" / "sbml" / "final_consortium"


def _models(consortium_profile: str = "legacy3"):
    return load_consortium_profile(consortium_profile)


def collect_exact_trajectory(
    seed: int, steps: int, action_distribution: str = "beta",
    consortium_profile: str = "legacy3", initial_nh4: float | None = None,
) -> dict:
    """Collect exact labels along an unseen control trajectory."""

    env = make_environment(
        None, steps, consortium_profile=consortium_profile, initial_nh4=initial_nh4,
    )
    simulator = env.simulator
    simulator.cooperative_capture_training_snapshot = True
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    contexts: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for _ in range(steps):
        if action_distribution == "uniform_validation":
            # Match the independent equal-action qualification protocol while
            # using calibration seeds that are never reused for qualification.
            action = rng.uniform(0.05, 0.95, size=5).astype(np.float32)
        else:
            # Include extremes because limitation switches occur near pump and
            # oxygen bounds, while retaining central actions for ordinary states.
            action = rng.beta(0.65, 0.65, size=5).astype(np.float32)
            if rng.random() < 0.45:
                action[3] = rng.uniform(0.0, 0.22)
            action[4] = rng.uniform(0.0, 1.0)
        if simulator._cooperative_solver is not None:
            simulator._cooperative_solver.last_training_snapshot = None
        _, _, terminated, truncated, _ = env.step(action)
        snapshot = simulator._cooperative_solver.last_training_snapshot
        if snapshot is not None:
            contexts.append(np.asarray(snapshot["context"], dtype=np.float32))
            fluxes.append(np.asarray(snapshot["fluxes"], dtype=np.float32))
            actions.append(action)
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "contexts": np.stack(contexts),
        "fluxes": np.stack(fluxes),
        "actions": np.stack(actions),
    }


def _layout_offsets(metadata: dict) -> dict[str, tuple[int, int]]:
    offsets = {}
    cursor = 0
    for name, width in metadata["context_layout"]:
        width = int(width)
        offsets[name] = (cursor, cursor + width)
        cursor += width
    return offsets


def select_active_rows(
    dictionary: CooperativeFluxDictionary,
    contexts: np.ndarray,
    metadata: dict,
    maximum: int,
) -> tuple[np.ndarray, dict]:
    """Prioritize high-distance and PHA/NH4 limitation-boundary states."""

    ranked = dictionary.rank(contexts, top_k=1)
    distances = ranked.distances[:, 0].astype(np.float64)
    offsets = _layout_offsets(metadata)
    reaction_ids = list(metadata["reaction_ids"])
    objective_start, _ = offsets["objective_coefficients"]
    pha_indices = [
        index for index, reaction_id in enumerate(reaction_ids)
        if reaction_id.lower() in {"ex_pha_c", "ex_phv_c", "phb_syn", "phv_syn"}
    ]
    pha_active = np.zeros(len(contexts), dtype=bool)
    for index in pha_indices:
        pha_active |= np.abs(contexts[:, objective_start + index]) > 1e-7

    supply_start, supply_stop = offsets["shared_supply_mmol_l_h"]
    shared_ids = list(metadata["shared_metabolite_ids"])
    nh4_local = next(
        (index for index, metabolite_id in enumerate(shared_ids) if "nh4" in metabolite_id.lower()),
        None,
    )
    if nh4_local is None:
        low_nh4 = np.zeros(len(contexts), dtype=bool)
    else:
        nh4_supply = contexts[:, supply_start + nh4_local]
        positive = nh4_supply[nh4_supply > 0]
        boundary = float(np.quantile(positive, 0.25)) if len(positive) else 0.0
        low_nh4 = nh4_supply <= boundary

    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    distance_score = (distances - median) / max(1e-6, 1.4826 * mad)
    score = distance_score + 3.0 * pha_active + 1.5 * low_nh4
    selected = np.argsort(score)[::-1][: min(maximum, len(contexts))]
    selected.sort()
    report = {
        "available_rows": int(len(contexts)),
        "selected_rows": int(len(selected)),
        "pha_active_available": int(pha_active.sum()),
        "pha_active_selected": int(pha_active[selected].sum()),
        "low_nh4_available": int(low_nh4.sum()),
        "low_nh4_selected": int(low_nh4[selected].sum()),
        "distance_available_min": float(distances.min()),
        "distance_available_median": float(np.median(distances)),
        "distance_available_max": float(distances.max()),
        "distance_selected_median": float(np.median(distances[selected])),
    }
    return selected, report


def append_to_artifact(
    source: Path,
    output: Path,
    contexts: np.ndarray,
    fluxes: np.ndarray,
    selection_report: dict,
    seeds: list[int],
) -> dict:
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    expected_width = sum(int(width) for _, width in payload["metadata"]["context_layout"])
    if contexts.ndim != 2 or contexts.shape[1] != expected_width:
        raise ValueError("augmentation context layout does not match dictionary")
    if fluxes.shape != (len(contexts), len(payload["metadata"]["reaction_ids"])):
        raise ValueError("augmentation flux layout does not match dictionary")
    if not np.isfinite(contexts).all() or not np.isfinite(fluxes).all():
        raise ValueError("augmentation contains non-finite data")
    indices = np.asarray(payload["feature_indices"], dtype=np.int64)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    scale = np.asarray(payload["feature_scale"], dtype=np.float32)
    weight = np.asarray(
        payload.get("feature_weight", np.ones(len(indices), dtype=np.float32)),
        dtype=np.float32,
    )
    normalized = (
        (contexts[:, indices] - mean) / scale * np.sqrt(weight)
    ).astype(np.float32)
    old_contexts = np.asarray(payload["normalized_contexts"], dtype=np.float32)
    old_fluxes = np.asarray(payload["fluxes"], dtype=np.float32)
    metadata = dict(payload["metadata"])
    metadata["candidate_count"] = int(len(old_fluxes) + len(fluxes))
    metadata["active_learning"] = {
        "method": "distance_plus_PHA_NH4_boundary_query",
        "base_artifact": str(source),
        "calibration_seeds": seeds,
        **selection_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "metadata": metadata,
            "feature_weight": weight,
            "normalized_contexts": np.concatenate((old_contexts, normalized), axis=0),
            "fluxes": np.concatenate((old_fluxes, fluxes.astype(np.float32)), axis=0),
        },
        output,
    )
    return {
        "artifact": str(output),
        "base_candidates": int(len(old_fluxes)),
        "added_candidates": int(len(fluxes)),
        "total_candidates": int(len(old_fluxes) + len(fluxes)),
        "artifact_mib": output.stat().st_size / 2**20,
        "selection": selection_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-additions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--consortium", choices=CONSORTIUM_PROFILES, default="legacy3")
    parser.add_argument("--initial-nh4", type=float, default=None)
    parser.add_argument(
        "--action-distribution",
        choices=("beta", "uniform_validation"),
        default="beta",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    import torch
    source_metadata = torch.load(args.base_artifact, map_location="cpu", weights_only=False)["metadata"]
    selected_models = _models(args.consortium)
    if source_metadata["species"] != list(selected_models):
        raise ValueError("augmentation consortium does not match dictionary species")
    if source_metadata.get("model_fingerprints") not in (
        None, {name: model_fingerprint(model) for name, model in selected_models.items()}
    ):
        raise ValueError("augmentation model fingerprints differ from dictionary")
    if source_metadata.get("cooperative_optimize_live_objectives") is False:
        raise ValueError("cannot append live-objective labels to a maintenance dictionary")
    seeds = [args.seed + 1009 * index for index in range(args.trajectories)]
    workers = max(1, min(args.workers, len(seeds)))
    if workers == 1:
        trajectories = [
            collect_exact_trajectory(seed, args.steps, args.action_distribution,
                                     args.consortium, args.initial_nh4)
            for seed in seeds
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            trajectories = list(
                executor.map(
                    collect_exact_trajectory,
                    seeds,
                    [args.steps] * len(seeds),
                    [args.action_distribution] * len(seeds),
                    [args.consortium] * len(seeds),
                    [args.initial_nh4] * len(seeds),
                )
            )
    contexts = np.concatenate([item["contexts"] for item in trajectories])
    fluxes = np.concatenate([item["fluxes"] for item in trajectories])
    dictionary = CooperativeFluxDictionary(args.base_artifact, device="cuda")
    if dictionary.metadata["species"] != list(_models(args.consortium)):
        raise ValueError("augmentation consortium does not match dictionary species")
    selected, selection_report = select_active_rows(
        dictionary, contexts, dictionary.metadata, args.max_additions
    )
    report = append_to_artifact(
        args.base_artifact,
        args.output,
        contexts[selected],
        fluxes[selected],
        selection_report,
        seeds,
    )
    report["action_distribution"] = args.action_distribution
    report["consortium_profile"] = args.consortium
    report["initial_nh4_mM"] = args.initial_nh4
    report_path = args.report or args.output.with_suffix(".active.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
