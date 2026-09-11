#!/usr/bin/env python3
"""Collect aligned exact cooperative-FBA states for a GPU dictionary."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import sys

import numpy as np
from cobra.io import read_sbml_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dfba_simulator import dFBASimulator
from src.fba_surrogate import model_fingerprint
from src.rl_environment import ConsortiumEnv
from src.utils import (
    get_initial_params,
    load_sbml_models,
    select_consortium_models,
    select_or16_ns21_models,
)

MODEL_DIR = PROJECT_ROOT / "models" / "sbml" / "final_consortium"
TIER_SAMPLES = {"smoke": 1024, "pilot": 8192, "production": 32768}
HELPER_MODEL = (
    PROJECT_ROOT / "models" / "sbml" / "helper_candidates"
    / "Propionibacterium_freudenreichii_shermanii_curated.xml"
)
CONSORTIUM_PROFILES = ("legacy3", "or16-ns21", "pf-helper3")


def _models(consortium_profile: str = "legacy3"):
    loaded = load_sbml_models(MODEL_DIR)
    if consortium_profile == "legacy3":
        return select_consortium_models(loaded)
    selected = select_or16_ns21_models(loaded)
    if consortium_profile == "pf-helper3":
        helper = read_sbml_model(str(HELPER_MODEL))
        selected["Propionibacterium_freudenreichii_shermanii"] = helper
    return selected


def _stratified_action(rng: np.random.Generator) -> np.ndarray:
    action = rng.beta(0.7, 0.7, size=5).astype(np.float32)
    region = int(rng.integers(0, 10))
    if region < 4:
        # Product phase: retain nitrogen limitation so the NS21 PHB/PHV
        # objective is actually represented in the exact teacher set.
        action[3] = rng.uniform(0.00, 0.05)
    elif region < 8:
        action[3] = rng.uniform(0.15, 0.60)  # 0.003--0.012 g/L Defined-10
    elif region < 9:
        action[3] = rng.uniform(0.00, 0.15)
    else:
        action[3] = rng.uniform(0.60, 1.00)
    action[4] = rng.uniform(0.0, 1.0)  # full oxygen-transfer range
    return action


def _perturb_reset_state(env: ConsortiumEnv, rng: np.random.Generator) -> None:
    for species in env.simulator.state.species.values():
        species.biomass *= float(rng.uniform(0.8, 1.2))
    limitation_tokens = ("glc", "ile", "pydam")
    for metabolite in list(env.simulator.state.metabolites):
        lowered = metabolite.lower()
        if any(token in lowered for token in limitation_tokens):
            env.simulator.state.metabolites[metabolite] *= float(
                rng.uniform(0.25, 1.50)
            )
    # The default starter contains 2 mM NH4 and short sampled episodes rarely
    # reach the <0.1 mM storage phase.  Stratify reset nitrogen explicitly so
    # both growth and PHB/PHV objectives are covered by the dictionary.
    nitrogen_region = float(rng.random())
    if nitrogen_region < 0.55:
        nh4 = rng.uniform(0.0, 0.06)
    elif nitrogen_region < 0.80:
        nh4 = rng.uniform(0.06, 0.30)
    else:
        nh4 = rng.uniform(0.30, 3.00)
    env.simulator.state.metabolites["nh4_e"] = float(nh4)


def _sample_reset_horizon(
    rng: np.random.Generator, episode_steps: int
) -> int:
    """Mix full trajectories with short phase-focused trajectories."""

    episode_steps = max(1, int(episode_steps))
    if episode_steps > 4 and rng.random() < 0.60:
        return int(rng.integers(4, min(24, episode_steps) + 1))
    return episode_steps


def collect_shard(
    samples: int,
    seed: int,
    episode_steps: int,
    consortium_profile: str = "legacy3",
) -> dict:
    models = _models(consortium_profile)
    biomass, metabolites = get_initial_params(models)
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=metabolites,
        dt=0.2,
        solver_backend="highs",
        fba_mode="cooperative",
        # Stage 1 protects coexistence; stage 2 must still optimize the live
        # NS21 storage objective in nitrogen limitation.  Without this flag,
        # every PHB/PHV teacher flux is identically zero.
        cooperative_optimize_live_objectives=True,
        cooperative_parsimony=True,
        cooperative_highs_method="highs-ds",
        cooperative_capture_training_snapshot=True,
        ph_control_target=6.5,
    )
    env = ConsortiumEnv(
        simulator=simulator,
        max_time=episode_steps * 0.2,
        max_common_feed_early=0.02,
        max_common_feed_late=0.01,
        max_specific_feed_per_step=0.1,
    )
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    _perturb_reset_state(env, rng)
    contexts = None
    fluxes = None
    actions = np.empty((samples, 5), dtype=np.float32)
    common_growth = np.empty(samples, dtype=np.float32)
    aggregate_objective = np.empty(samples, dtype=np.float32)
    completed = 0
    reset_horizon = _sample_reset_horizon(rng, episode_steps)
    while completed < samples:
        action = _stratified_action(rng)
        if simulator.state.metabolites.get("nh4_e", 0.0) < 0.1:
            # Do not immediately erase deliberately sampled production-phase
            # states with a nitrogen-rich common feed.  Specific feeds and
            # oxygen still vary, so these are not duplicate states.
            action[3] = rng.uniform(0.0, 0.02)
        if simulator._cooperative_solver is not None:
            simulator._cooperative_solver.last_training_snapshot = None
        _, _, terminated, truncated, _ = env.step(action)
        solver = simulator._cooperative_solver
        snapshot = None if solver is None else solver.last_training_snapshot
        if snapshot is not None:
            context_row = np.asarray(snapshot["context"], dtype=np.float32)
            flux_row = np.asarray(snapshot["fluxes"], dtype=np.float32)
            if contexts is None:
                contexts = np.empty((samples, len(context_row)), dtype=np.float32)
                fluxes = np.empty((samples, len(flux_row)), dtype=np.float32)
            contexts[completed] = context_row
            fluxes[completed] = flux_row
            actions[completed] = action
            common_growth[completed] = float(snapshot["common_growth"])
            aggregate_objective[completed] = float(snapshot["aggregate_objective"])
            completed += 1
        if terminated or truncated or env.current_step >= reset_horizon:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            _perturb_reset_state(env, rng)
            reset_horizon = _sample_reset_horizon(rng, episode_steps)
    assert simulator._cooperative_solver is not None
    assert contexts is not None and fluxes is not None
    solver = simulator._cooperative_solver
    reaction_ids = []
    for name in solver.species_names:
        reaction_ids.extend(reaction.id for reaction in models[name].reactions)
    metadata = {
        "format_version": 1,
        "formulation": "exact_cooperative_shared_medium",
        "species": solver.species_names,
        "species_offsets": {
            name: list(solver._offsets[name]) for name in solver.species_names
        },
        "reaction_ids": reaction_ids,
        "shared_metabolite_ids": sorted(solver._exchange_terms),
        "context_layout": [
            ["biomass_g_l", len(solver.species_names)],
            ["shared_supply_mmol_l_h", len(solver._exchange_terms)],
            ["flux_lower_bounds", solver.n_fluxes],
            ["flux_upper_bounds", solver.n_fluxes],
            ["objective_coefficients", solver.n_fluxes],
        ],
        "dt_hours": 0.2,
        "cooperative_optimize_live_objectives": True,
        "pha_trigger_nh4_mmol_l": 0.1,
        "sampling_design": {
            "nitrogen_regions": {
                "product_0_to_0.06_mmol_l": 0.55,
                "transition_0.06_to_0.30_mmol_l": 0.25,
                "growth_0.30_to_3.00_mmol_l": 0.20,
            },
            "short_trajectory_probability": 0.60,
            "short_trajectory_steps": [4, min(24, int(episode_steps))],
            "maximum_trajectory_steps": int(episode_steps),
        },
        "common_feed_max_early_g_l": 0.02,
        "common_feed_max_late_g_l": 0.01,
        "seed": seed,
        "consortium_profile": consortium_profile,
        "model_fingerprints": {
            name: model_fingerprint(models[name]) for name in solver.species_names
        },
    }
    return {
        "contexts": contexts,
        "fluxes": fluxes,
        "actions": actions,
        "common_growth": common_growth,
        "aggregate_objective": aggregate_objective,
        "metadata": metadata,
    }


def collect_shard_to_directory(
    samples: int,
    seed: int,
    episode_steps: int,
    directory: str,
    consortium_profile: str = "legacy3",
) -> str:
    """Write one shard locally so large arrays never cross a process pipe."""

    shard = collect_shard(samples, seed, episode_steps, consortium_profile)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for key in (
        "contexts", "fluxes", "actions", "common_growth", "aggregate_objective"
    ):
        np.save(target / f"{key}.npy", shard[key], allow_pickle=False)
    (target / "metadata.json").write_text(
        json.dumps(shard["metadata"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(target)


def merge_shard_directories(
    shard_directories: list[str], output: Path, metadata: dict
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    keys = (
        "contexts", "fluxes", "actions", "common_growth", "aggregate_objective"
    )
    for key in keys:
        sources = [
            np.load(Path(directory) / f"{key}.npy", mmap_mode="r")
            for directory in shard_directories
        ]
        shape = (sum(len(source) for source in sources), *sources[0].shape[1:])
        merged = np.lib.format.open_memmap(
            output / f"{key}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        cursor = 0
        for source in sources:
            merged[cursor : cursor + len(source)] = source
            cursor += len(source)
        merged.flush()
        del merged
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=sorted(TIER_SAMPLES), default="smoke")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--consortium",
        choices=CONSORTIUM_PROFILES,
        default="legacy3",
        help="legacy WCFS1 trio, current OR16+NS21 pair, or P. freudenreichii helper trio",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "npz", "npy-directory"),
        default="auto",
        help="auto uses a memory-mapped directory for datasets >=4096 rows",
    )
    args = parser.parse_args()
    samples = int(args.samples or TIER_SAMPLES[args.tier])
    workers = max(1, min(int(args.workers), samples))
    counts = [samples // workers] * workers
    for index in range(samples % workers):
        counts[index] += 1
    directory_format = args.dataset_format == "npy-directory" or (
        args.dataset_format == "auto" and samples >= 4096
    )
    output = args.output or (
        PROJECT_ROOT
        / "models"
        / "cooperative_surrogate"
        / (f"{args.tier}_{samples}" if directory_format else f"{args.tier}_{samples}.npz")
    )
    if directory_format:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}_shards_", dir=output.parent
        ) as temporary:
            shard_paths = [str(Path(temporary) / f"shard_{i:02d}") for i in range(workers)]
            if workers == 1:
                written = [
                    collect_shard_to_directory(
                        samples, args.seed, args.episode_steps, shard_paths[0],
                        args.consortium,
                    )
                ]
            else:
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=mp.get_context("spawn")
                ) as executor:
                    written = list(
                        executor.map(
                            collect_shard_to_directory,
                            counts,
                            [args.seed + 1009 * i for i in range(workers)],
                            [args.episode_steps] * workers,
                            shard_paths,
                            [args.consortium] * workers,
                        )
                    )
            metadata = json.loads(
                (Path(written[0]) / "metadata.json").read_text(encoding="utf-8")
            )
            metadata.update(
                {
                    "tier": args.tier,
                    "samples": samples,
                    "workers": workers,
                    "worker_seeds": [args.seed + 1009 * i for i in range(workers)],
                    "context_dimension": int(
                        np.load(Path(written[0]) / "contexts.npy", mmap_mode="r").shape[1]
                    ),
                    "flux_dimension": int(
                        np.load(Path(written[0]) / "fluxes.npy", mmap_mode="r").shape[1]
                    ),
                    "storage": "npy_directory_float32",
                }
            )
            merge_shard_directories(written, output, metadata)
    else:
        if workers == 1:
            shards = [
                collect_shard(
                    samples, args.seed, args.episode_steps, args.consortium
                )
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn")
            ) as executor:
                shards = list(
                    executor.map(
                        collect_shard,
                        counts,
                        [args.seed + 1009 * i for i in range(workers)],
                        [args.episode_steps] * workers,
                        [args.consortium] * workers,
                    )
                )
        keys = (
            "contexts", "fluxes", "actions", "common_growth", "aggregate_objective"
        )
        merged = {
            key: np.concatenate([shard[key] for shard in shards]) for key in keys
        }
        metadata = dict(shards[0]["metadata"])
        metadata.update(
            {
                "tier": args.tier,
                "samples": samples,
                "workers": workers,
                "worker_seeds": [args.seed + 1009 * i for i in range(workers)],
                "context_dimension": int(merged["contexts"].shape[1]),
                "flux_dimension": int(merged["fluxes"].shape[1]),
                "storage": "compressed_npz_float32",
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **merged)
        output.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                key: metadata[key]
                for key in (
                    "tier", "samples", "workers", "context_dimension",
                    "flux_dimension", "storage"
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
