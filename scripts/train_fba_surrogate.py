#!/usr/bin/env python3
"""Collect exact FBA labels and train stoichiometry-constrained GPU models.

This is an offline command.  The resulting artifacts are used only for RL
rollouts; final scientific evaluation should continue to use ``highs``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as functional
from cobra.util.array import create_stoichiometric_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dfba_simulator import dFBASimulator
from src.fba_surrogate import (
    compute_nullspace_basis,
    extract_lp_features,
    make_network,
    safe_species_filename,
    save_surrogate_artifact,
)
from src.rl_environment import ConsortiumEnv
from src.utils import get_initial_params, load_sbml_models, select_consortium_models


def load_models(sbml_dir: Path):
    return select_consortium_models(load_sbml_models(sbml_dir))


def collect_exact_dataset(
    models,
    samples: int,
    episode_steps: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    initial_biomass, initial_metabolites = get_initial_params(models)
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        dt=0.2,
        solver_backend="highs",
        fba_mode="separate",
    )
    env = ConsortiumEnv(simulator=simulator, max_time=episode_steps * 0.2)
    rng = np.random.default_rng(seed)
    rows = {name: {"features": [], "fluxes": []} for name in models}
    observation, _ = env.reset(seed=seed)
    del observation
    completed = 0
    while completed < samples:
        # Beta draws cover both interior controls and the important near-zero/
        # near-saturation regions more often than a uniform distribution.
        action = rng.beta(0.7, 0.7, size=env.action_space.shape).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        for name, model in models.items():
            solution = simulator.last_fba_solutions.get(name)
            if solution is None or solution.status != "optimal":
                continue
            rows[name]["features"].append(extract_lp_features(model))
            rows[name]["fluxes"].append(
                solution.fluxes.reindex(
                    [reaction.id for reaction in model.reactions]
                ).to_numpy(dtype=np.float32)
            )
        completed += 1
        if terminated or truncated or completed % episode_steps == 0:
            env.reset()
        if completed % 25 == 0 or completed == samples:
            print(f"exact labels: {completed}/{samples} environment steps", flush=True)
    return {
        name: {
            key: np.asarray(values, dtype=np.float32)
            for key, values in species_rows.items()
        }
        for name, species_rows in rows.items()
    }


def collect_exact_dataset_worker(
    sbml_dir: str,
    samples: int,
    episode_steps: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Load private COBRA models and collect one independent CPU shard."""

    return collect_exact_dataset(
        load_models(Path(sbml_dir)), samples, episode_steps, seed
    )


def collect_exact_dataset_parallel(
    sbml_dir: Path,
    samples: int,
    episode_steps: int,
    seed: int,
    workers: int,
) -> dict[str, dict[str, np.ndarray]]:
    workers = max(1, min(int(workers), int(samples)))
    if workers == 1:
        return collect_exact_dataset_worker(
            str(sbml_dir), samples, episode_steps, seed
        )
    counts = [samples // workers] * workers
    for index in range(samples % workers):
        counts[index] += 1
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        shards = list(
            executor.map(
                collect_exact_dataset_worker,
                [str(sbml_dir)] * workers,
                counts,
                [episode_steps] * workers,
                [seed + 1009 * index for index in range(workers)],
            )
        )
    species_names = shards[0].keys()
    return {
        species: {
            key: np.concatenate([shard[species][key] for shard in shards], axis=0)
            for key in ("features", "fluxes")
        }
        for species in species_names
    }


def relevant_reaction_indices(model) -> np.ndarray:
    ids = {reaction.id for reaction in model.exchanges}
    ids.update(
        reaction_id
        for reaction_id in (
            "R_Growth", "Growth", "R_BIOMASS_LLA", "BIOMASS_LLA",
            "R_LCP", "R_PHA_SYN", "R_PHB_SYN",
        )
        if reaction_id in model.reactions
    )
    return np.asarray(
        [index for index, reaction in enumerate(model.reactions) if reaction.id in ids],
        dtype=np.int64,
    )


def train_species(
    species_name: str,
    model,
    dataset: dict[str, np.ndarray],
    output_path: Path,
    epochs: int,
    hidden_dims: tuple[int, ...],
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    decoder_mode: str,
) -> dict[str, float]:
    features = dataset["features"]
    target_fluxes = dataset["fluxes"]
    if len(features) < 8:
        raise RuntimeError(f"{species_name}: at least 8 feasible labels are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    validation_count = max(1, int(round(0.2 * len(order))))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]

    if decoder_mode == "nullspace":
        print(f"{species_name}: computing null(S) for {len(model.reactions)} reactions")
        basis = compute_nullspace_basis(model)
        target_latent = target_fluxes @ basis
    else:
        dictionary_indices = (
            np.arange(len(target_fluxes))
            if decoder_mode == "feasible_dictionary_optimizer"
            else training_indices
        )
        if decoder_mode == "feasible_dictionary_optimizer":
            # Exact LPs can revisit the same vertex.  Duplicate candidates add
            # GPU load without adding coverage, so remove them before building
            # the runtime library.
            rounded = np.round(target_fluxes[dictionary_indices], decimals=6)
            _, unique_positions = np.unique(
                rounded, axis=0, return_index=True
            )
            dictionary_indices = dictionary_indices[np.sort(unique_positions)]
        print(
            f"{species_name}: building feasible dictionary from "
            f"{len(dictionary_indices)} exact flux vectors"
        )
        # The optimizer is an explicit feasible-solution library, not a fitted
        # predictor.  Validation rows are therefore useful runtime candidates
        # and must not be discarded as if they were training examples.
        basis = target_fluxes[dictionary_indices].T.copy()
        target_latent = None

    x_mean = features[training_indices].mean(axis=0)
    x_scale = features[training_indices].std(axis=0)
    x_scale = np.maximum(x_scale, 1e-6)
    if target_latent is not None:
        z_mean = target_latent[training_indices].mean(axis=0)
        z_scale = target_latent[training_indices].std(axis=0)
        z_scale = np.maximum(z_scale, 1e-6)
    else:
        z_mean = np.zeros(basis.shape[1], dtype=np.float32)
        z_scale = np.ones(basis.shape[1], dtype=np.float32)

    network = make_network(basis, features.shape[1], hidden_dims, decoder_mode)
    with torch.no_grad():
        network.x_mean.copy_(torch.from_numpy(x_mean))
        network.x_scale.copy_(torch.from_numpy(x_scale))
        network.z_mean.copy_(torch.from_numpy(z_mean))
        network.z_scale.copy_(torch.from_numpy(z_scale))
        network.x_min.copy_(torch.from_numpy(features[training_indices].min(axis=0)))
        network.x_max.copy_(torch.from_numpy(features[training_indices].max(axis=0)))
    network.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-5)
    relevant = torch.as_tensor(relevant_reaction_indices(model), device=device)
    n_reactions = len(model.reactions)
    training_features = torch.as_tensor(features[training_indices], device=device)
    training_latent = (
        torch.as_tensor(target_latent[training_indices], device=device)
        if target_latent is not None
        else None
    )
    training_fluxes = torch.as_tensor(target_fluxes[training_indices], device=device)

    best_state = (
        {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        if decoder_mode == "feasible_dictionary_optimizer"
        else None
    )
    best_validation = float("inf")
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    effective_epochs = (
        0 if decoder_mode == "feasible_dictionary_optimizer" else epochs
    )
    for epoch in range(effective_epochs):
        network.train()
        permutation = torch.randperm(len(training_indices), generator=generator)
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size].to(device)
            x = training_features[indices]
            flux_target = training_fluxes[indices]
            if decoder_mode == "nullspace":
                assert training_latent is not None
                z_target = training_latent[indices]
                normalized_target = (z_target - network.z_mean) / network.z_scale
                normalized_prediction = network.encoder(
                    (x - network.x_mean) / network.x_scale
                )
                flux_prediction = (
                    normalized_prediction * network.z_scale + network.z_mean
                ) @ network.basis.T
                latent_loss = functional.smooth_l1_loss(
                    normalized_prediction, normalized_target
                )
            else:
                flux_prediction = network(x)
                latent_loss = torch.zeros((), device=device)
            flux_scale = torch.clamp(torch.abs(flux_target[:, relevant]), min=1.0)
            relevant_loss = torch.mean(
                ((flux_prediction[:, relevant] - flux_target[:, relevant]) / flux_scale) ** 2
            )
            lower = x[:, :n_reactions]
            upper = x[:, n_reactions : 2 * n_reactions]
            bound_loss = (
                functional.relu(lower - flux_prediction).square().mean()
                + functional.relu(flux_prediction - upper).square().mean()
            )
            objective = x[:, 2 * n_reactions : 3 * n_reactions]
            predicted_objective = torch.sum(objective * flux_prediction, dim=1)
            exact_objective = torch.sum(objective * flux_target, dim=1)
            objective_scale = torch.clamp(torch.abs(exact_objective), min=1.0)
            objective_loss = torch.mean(
                ((predicted_objective - exact_objective) / objective_scale) ** 2
            )
            # Internal flux vectors can be non-unique at the same FBA optimum.
            # Give the scientifically consumed exchange/growth fluxes and the
            # objective priority, while the latent target regularizes the
            # remaining feasible solution selected by the exact solver.
            full_flux_scale = torch.clamp(torch.abs(flux_target), min=1.0)
            full_flux_loss = torch.mean(
                ((flux_prediction - flux_target) / full_flux_scale) ** 2
            )
            loss = (
                0.05 * latent_loss
                + 5.0 * relevant_loss
                + 5.0 * objective_loss
                + 0.5 * bound_loss
                + 0.02 * full_flux_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()

        if epoch % 5 == 0 or epoch + 1 == epochs:
            network.eval()
            with torch.inference_mode():
                validation_x = torch.as_tensor(features[validation_indices], device=device)
                validation_target = torch.as_tensor(
                    target_fluxes[validation_indices], device=device
                )
                validation_prediction = network(validation_x)
                score = torch.sqrt(
                    torch.mean(
                        (validation_prediction[:, relevant] - validation_target[:, relevant]) ** 2
                    )
                ).item()
            if score < best_validation:
                best_validation = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
    assert best_state is not None
    network.load_state_dict(best_state)
    network.eval()
    with torch.inference_mode():
        validation_x = torch.as_tensor(features[validation_indices], device=device)
        predicted = network(validation_x).cpu().numpy().astype(np.float64)
        if decoder_mode == "feasible_dictionary_optimizer":
            best_validation = torch.sqrt(
                torch.mean(
                    (
                        torch.as_tensor(predicted, device=device)[:, relevant]
                        - torch.as_tensor(
                            target_fluxes[validation_indices], device=device
                        )[:, relevant]
                    )
                    ** 2
                )
            ).item()
    exact = target_fluxes[validation_indices].astype(np.float64)
    objectives = features[validation_indices, 2 * n_reactions : 3 * n_reactions]
    predicted_objective = np.sum(objectives * predicted, axis=1)
    exact_objective = np.sum(objectives * exact, axis=1)
    residual_matrix = np.asarray(
        create_stoichiometric_matrix(model, array_type="dense", dtype=np.float64)
    )
    metrics = {
        "validation_relevant_flux_rmse": float(best_validation),
        "validation_objective_mae": float(
            np.mean(np.abs(predicted_objective - exact_objective))
        ),
        "validation_objective_max_abs_error": float(
            np.max(np.abs(predicted_objective - exact_objective))
        ),
        "validation_max_mass_balance_residual": float(
            np.max(np.abs(predicted @ residual_matrix.T))
        ),
        "training_samples": float(len(training_indices)),
        "validation_samples": float(len(validation_indices)),
        "latent_dimension": float(basis.shape[1]),
        "training_seconds": float(time.perf_counter() - started),
    }
    save_surrogate_artifact(
        output_path,
        model,
        network,
        basis,
        hidden_dims,
        metrics=metrics,
        decoder_mode=decoder_mode,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sbml-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "sbml" / "final_consortium",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "fba_surrogate")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--collection-workers",
        type=int,
        default=1,
        help="CPU processes used to collect independent exact-HiGHS shards",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=[
            "feasible_dictionary_optimizer",
            "feasible_dictionary",
            "nullspace",
        ],
        default="feasible_dictionary_optimizer",
        help="feasible_dictionary avoids learning non-unique internal flux coordinates",
    )
    parser.add_argument("--dataset", type=Path, default=None,
                        help="reuse an .npz dataset instead of collecting labels")
    parser.add_argument(
        "--allow-legacy-separate-fba",
        action="store_true",
        help=(
            "explicitly allow the legacy per-species dataset; it is not valid "
            "for the cooperative shared-medium RL environment"
        ),
    )
    args = parser.parse_args()
    if not args.allow_legacy_separate_fba:
        parser.error(
            "this command trains the legacy separate-FBA surrogate and cannot "
            "be used for cooperative RL; pass --allow-legacy-separate-fba only "
            "for a deliberate legacy experiment"
        )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    models = load_models(args.sbml_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset is None:
        dataset = collect_exact_dataset_parallel(
            args.sbml_dir,
            args.samples,
            args.episode_steps,
            args.seed,
            args.collection_workers,
        )
        dataset_path = args.output_dir / "exact_training_data.npz"
        flattened = {}
        for species, rows in dataset.items():
            prefix = safe_species_filename(species)
            flattened[f"{prefix}__features"] = rows["features"]
            flattened[f"{prefix}__fluxes"] = rows["fluxes"]
        np.savez_compressed(dataset_path, **flattened)
    else:
        archive = np.load(args.dataset)
        dataset = {}
        for species in models:
            prefix = safe_species_filename(species)
            dataset[species] = {
                "features": archive[f"{prefix}__features"],
                "fluxes": archive[f"{prefix}__fluxes"],
            }

    manifest = {"artifact_version": 1, "species": {}, "metrics": {}}
    for offset, (species, model) in enumerate(models.items()):
        filename = f"{safe_species_filename(species)}.pt"
        metrics = train_species(
            species,
            model,
            dataset[species],
            args.output_dir / filename,
            args.epochs,
            tuple(args.hidden_dims),
            args.batch_size,
            args.learning_rate,
            device,
            args.seed + offset,
            args.decoder_mode,
        )
        manifest["species"][species] = filename
        manifest["metrics"][species] = metrics
        print(species, json.dumps(metrics, ensure_ascii=False, indent=2))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
