#!/usr/bin/env python3
"""Train WCFS1-corrected PPO in stages until a predeclared effect gate passes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dfba_simulator import dFBASimulator  # noqa: E402
from src.cooperative_gpu_service import start_cooperative_gpu_qp_service  # noqa: E402
from src.rl_environment import ConsortiumEnv, SymmetricPPOActionWrapper  # noqa: E402
from src.utils import get_initial_params, load_sbml_models, select_consortium_models  # noqa: E402


MODEL_DIR = ROOT / "models/sbml/final_consortium"
DT_H = 0.2
HORIZON_H = 24.0
MAX_DEFINED_FEED_SCALE = 0.02
FIXED25 = np.full(5, 0.25, dtype=np.float32)
LOW_AIR = np.asarray([0.0, 0.0, 0.0, 0.40, 1.0], dtype=np.float32)
BUDGET_MAX_AIR = np.asarray([0.0, 0.0, 0.0, 0.25, 1.0], dtype=np.float32)


def short_name(name: str) -> str:
    if "OR16" in name:
        return "OR16"
    if "NS21" in name:
        return "NS21"
    return "WCFS1"


def build_raw_env(
    scenario_seed: int,
    cooperative_surrogate_artifact: str | None = None,
    cooperative_surrogate_exact_interval: int = 128,
    cooperative_surrogate_distance_threshold: float | None = None,
    cooperative_gpu_qp_service=None,
) -> ConsortiumEnv:
    rng = np.random.default_rng(scenario_seed)
    models = select_consortium_models(load_sbml_models(MODEL_DIR))
    biomass, medium = get_initial_params(models)
    biomass = {
        key: value * float(rng.uniform(0.95, 1.05)) for key, value in biomass.items()
    }
    medium = {
        key: value * float(rng.uniform(0.98, 1.02)) if value > 0 else value
        for key, value in medium.items()
    }
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=medium,
        initial_rubber=100.0,
        volume=1.0,
        dt=DT_H,
        solver_backend="highs",
        fba_mode="cooperative",
        ph_control_target=6.5,
        cooperative_highs_method="highs-ds",
        cooperative_surrogate_artifact=(
            None
            if cooperative_gpu_qp_service is not None
            else cooperative_surrogate_artifact
        ),
        cooperative_surrogate_device="cuda",
        cooperative_surrogate_top_k=16,
        cooperative_surrogate_exact_interval=cooperative_surrogate_exact_interval,
        # This artifact is intentionally restricted to approximate training
        # rollouts; every evaluation trajectory in this script remains exact.
        cooperative_surrogate_require_qualified=False,
        cooperative_surrogate_distance_threshold=(
            cooperative_surrogate_distance_threshold
        ),
        cooperative_gpu_qp_projection=cooperative_gpu_qp_service is not None,
        cooperative_gpu_qp_only=cooperative_gpu_qp_service is not None,
        cooperative_gpu_qp_candidates=128,
        cooperative_gpu_qp_service=cooperative_gpu_qp_service,
    )
    return ConsortiumEnv(
        simulator=simulator,
        max_time=HORIZON_H,
        # The repaired-model screen found the useful Defined-10 scale near
        # 0.004--0.008. Mapping action 0--1 to 0--0.02 keeps that region broad
        # enough for PPO instead of compressing it into the bottom 0.4%.
        max_common_feed_early=MAX_DEFINED_FEED_SCALE,
        max_common_feed_late=MAX_DEFINED_FEED_SCALE,
    )


def make_train_env(
    rank: int,
    base_seed: int,
    cooperative_surrogate_artifact: str | None,
    cooperative_surrogate_exact_interval: int,
    cooperative_surrogate_distance_threshold: float | None,
    cooperative_gpu_qp_service=None,
):
    def _init():
        return Monitor(
            SymmetricPPOActionWrapper(
                build_raw_env(
                    base_seed + rank,
                    cooperative_surrogate_artifact=cooperative_surrogate_artifact,
                    cooperative_surrogate_exact_interval=cooperative_surrogate_exact_interval,
                    cooperative_surrogate_distance_threshold=(
                        cooperative_surrogate_distance_threshold
                    ),
                    cooperative_gpu_qp_service=cooperative_gpu_qp_service,
                )
            )
        )

    return _init


def policy_action(policy: str, time_h: float) -> np.ndarray:
    if policy == "no_control":
        return np.zeros(5, dtype=np.float32)
    if policy == "fixed25":
        return FIXED25.copy()
    if policy == "low_feed_max_air":
        return LOW_AIR.copy()
    if policy == "budget_max_air":
        return BUDGET_MAX_AIR.copy()
    if policy == "two_phase":
        action = LOW_AIR.copy()
        if time_h >= 12.0:
            action[3] = 0.0
        return action
    raise ValueError(policy)


def rollout(
    scenario_seed: int,
    policy: str,
    model: PPO | None = None,
    normalizer: VecNormalize | None = None,
    record: bool = False,
    env_override: ConsortiumEnv | None = None,
) -> tuple[dict, list[dict]]:
    env = env_override if env_override is not None else build_raw_env(scenario_seed)
    obs, _ = env.reset(seed=scenario_seed)
    initial = {
        short_name(name): float(state.biomass)
        for name, state in env.simulator.state.species.items()
    }
    raw_return = 0.0
    specific_feed_mmol_l = 0.0
    oxygen_effort_h = 0.0
    rows: list[dict] = []
    terminated = truncated = False
    info: dict = {}
    actions: list[np.ndarray] = []
    while not (terminated or truncated):
        if policy == "ppo":
            if model is None or normalizer is None:
                raise ValueError("PPO rollout requires model and normalizer")
            normalized = normalizer.normalize_obs(obs.reshape(1, -1).copy())
            action, _ = model.predict(normalized, deterministic=True)
            policy_space_action = np.asarray(action[0], dtype=np.float32)
            action = SymmetricPPOActionWrapper.to_physical(policy_space_action)
        else:
            action = policy_action(policy, env.simulator.state.time)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        raw_return += float(reward)
        specific_feed_mmol_l += float(np.sum(action[:3]) * 0.1)
        oxygen_effort_h += float(action[4] * DT_H)
        if record:
            rows.append(
                {
                    "time_h": float(env.simulator.state.time),
                    "action_or16": float(action[0]),
                    "action_ns21": float(action[1]),
                    "action_wcfs1": float(action[2]),
                    "action_defined_feed": float(action[3]),
                    "action_kla": float(action[4]),
                    "rubber_remaining_g_l": float(info["rubber_remaining"]),
                    "pha_mmol": float(info["total_pha"]),
                    "ph": float(info["ph"]),
                }
            )
    final = {
        "OR16": float(info["biomass_or16"]),
        "NS21": float(info["biomass_ns21"]),
        "WCFS1": float(info["biomass_lp"]),
    }
    ratios = {key: final[key] / initial[key] for key in final}
    diagnostics = env.simulator.get_solver_diagnostics()
    metrics = {
        "scenario_seed": scenario_seed,
        "policy": policy,
        "raw_return": raw_return,
        "rubber_degraded_g_l": 100.0 - float(info["rubber_remaining"]),
        "pha_mmol": float(info["total_pha"]),
        "defined_feed_g_l": float(env.simulator.cumulative_defined_feed_g_l),
        "specific_feed_mmol_l": specific_feed_mmol_l,
        "oxygen_effort_h": oxygen_effort_h,
        "final_ph": float(info["ph"]),
        "minimum_final_initial_ratio": min(ratios.values()),
        "final_initial_ratio": ratios,
        "solver_success_rate": float(diagnostics["solve_success_rate"]),
        "terminated_early": bool(terminated and env.simulator.state.time < HORIZON_H),
        "mean_action": np.mean(np.asarray(actions), axis=0).tolist(),
    }
    return metrics, rows


def baseline_worker(task: tuple[int, str]) -> tuple[str, dict]:
    scenario_seed, policy = task
    metrics, _ = rollout(scenario_seed, policy)
    return policy, metrics


def scenario_worker(task: tuple[int, str, str, bool]) -> tuple[list[dict], list[dict]]:
    """Evaluate all comparators for one held-out scenario in a spawned process."""

    scenario_seed, model_path, vecnormalize_path, record = task
    model = PPO.load(model_path, device="cpu")
    ppo_env = build_raw_env(scenario_seed)
    dummy = DummyVecEnv([lambda: ppo_env])
    normalizer = VecNormalize.load(vecnormalize_path, dummy)
    normalizer.training = False
    normalizer.norm_reward = False
    results = []
    ppo_metrics, trajectory = rollout(
        scenario_seed,
        "ppo",
        model,
        normalizer,
        record=record,
        env_override=ppo_env,
    )
    results.append(ppo_metrics)
    for policy in ("fixed25", "low_feed_max_air", "two_phase", "no_control"):
        metrics, _ = rollout(scenario_seed, policy)
        results.append(metrics)
    normalizer.close()
    return results, trajectory


def summarize(rows: list[dict], policy: str) -> dict:
    keys = (
        "raw_return",
        "rubber_degraded_g_l",
        "pha_mmol",
        "defined_feed_g_l",
        "specific_feed_mmol_l",
        "oxygen_effort_h",
        "minimum_final_initial_ratio",
        "solver_success_rate",
    )
    summary = {"policy": policy, "n": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"sd_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary


def effect_gate(ppo_rows: list[dict], fixed_rows: list[dict]) -> dict:
    paired = np.asarray(
        [ppo["raw_return"] - fixed["raw_return"] for ppo, fixed in zip(ppo_rows, fixed_rows)],
        dtype=float,
    )
    mean_diff = float(np.mean(paired))
    if len(paired) > 1 and np.std(paired, ddof=1) > 0:
        se = stats.sem(paired)
        ci = stats.t.interval(0.95, len(paired) - 1, loc=mean_diff, scale=se)
        ci_low, ci_high = map(float, ci)
    else:
        ci_low = ci_high = mean_diff
    ppo = summarize(ppo_rows, "ppo")
    fixed = summarize(fixed_rows, "fixed25")
    checks = {
        "all_constraints": all(
            row["minimum_final_initial_ratio"] >= 0.90
            and 6.3 <= row["final_ph"] <= 6.7
            and row["solver_success_rate"] >= 0.995
            and not row["terminated_early"]
            for row in ppo_rows
        ),
        "return_95ci_above_fixed25": ci_low > 0.0,
        "pha_noninferior": ppo["mean_pha_mmol"] >= fixed["mean_pha_mmol"],
        "rubber_noninferior": ppo["mean_rubber_degraded_g_l"]
        >= fixed["mean_rubber_degraded_g_l"],
        "defined_feed_not_higher": ppo["mean_defined_feed_g_l"]
        <= fixed["mean_defined_feed_g_l"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "paired_return_difference": paired.tolist(),
        "mean_return_difference": mean_diff,
        "return_difference_95ci": [ci_low, ci_high],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def plot_results(payload: dict, trajectory: list[dict], output_dir: Path) -> None:
    mpl.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    blue, orange, green, grey = "#0072B2", "#E69F00", "#009E73", "#777777"
    paired = np.asarray(
        payload["effect_gate"]["paired_return_difference"], dtype=float
    )
    ci_low, ci_high = payload["effect_gate"]["return_difference_95ci"]
    scenario_index = np.arange(1, len(paired) + 1)
    axes[0, 0].scatter(scenario_index, paired, color=blue, s=28, zorder=3)
    axes[0, 0].axhspan(ci_low, ci_high, color=blue, alpha=0.14, linewidth=0)
    axes[0, 0].axhline(
        payload["effect_gate"]["mean_return_difference"],
        color=blue,
        linewidth=1.2,
    )
    axes[0, 0].axhline(0.0, color=grey, linestyle="--", linewidth=0.9)
    axes[0, 0].set_xticks(scenario_index)
    axes[0, 0].set_xlabel("Held-out scenario")
    axes[0, 0].set_ylabel("Paired return difference")
    axes[0, 0].set_title("a", loc="left", fontweight="bold")

    policies = ["ppo", "fixed25", "low_feed_max_air", "two_phase", "no_control"]
    summaries = payload["summaries"]
    x = np.arange(len(policies))
    means = [summaries[key]["mean_raw_return"] for key in policies]
    errors = [summaries[key]["sd_raw_return"] for key in policies]
    axes[0, 1].bar(x, means, yerr=errors, color=[green, grey, blue, orange, "#BBBBBB"], capsize=3)
    axes[0, 1].set_xticks(x, ["PPO", "Fixed-25", "Low-feed", "Two-phase", "None"], rotation=25, ha="right")
    axes[0, 1].set_ylabel("Raw episode return")
    axes[0, 1].set_title("b", loc="left", fontweight="bold")

    metrics = ("rubber_degraded_g_l", "pha_mmol", "defined_feed_g_l")
    labels = ("Rubber degraded", "PHA", "Defined feed")
    width = 0.24
    fixed = summaries["fixed25"]
    for index, policy in enumerate(("ppo", "fixed25", "low_feed_max_air")):
        values = [
            summaries[policy][f"mean_{metric}"] / max(fixed[f"mean_{metric}"], 1e-12)
            for metric in metrics
        ]
        axes[1, 0].bar(
            np.arange(3) + (index - 1) * width,
            values,
            width,
            color=(green, grey, blue)[index],
            label=("PPO", "Fixed-25", "Low-feed rule")[index],
        )
    axes[1, 0].axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xticks(np.arange(3), labels)
    axes[1, 0].set_ylabel("Ratio to Fixed-25")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_title("c", loc="left", fontweight="bold")

    time_h = [row["time_h"] for row in trajectory]
    axes[1, 1].plot(time_h, [row["action_defined_feed"] for row in trajectory], color=orange, label="Defined feed")
    axes[1, 1].plot(time_h, [row["action_kla"] for row in trajectory], color=blue, label="kLa")
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel("Normalized action")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title("d", loc="left", fontweight="bold")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Figure_rl_effect_wcfs1.{suffix}", dpi=500)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rl_wcfs1_effect")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--stage-steps", type=int, default=512)
    parser.add_argument("--max-timesteps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--eval-scenarios", type=int, default=5)
    parser.add_argument(
        "--training-artifact",
        type=Path,
        default=ROOT
        / "models/cooperative_surrogate/cooperative_neural_reranker_34448_dagger2_rebased.pt",
        help="neural-reranked GPU QP artifact used only for training rollouts",
    )
    parser.add_argument("--training-exact-interval", type=int, default=0)
    parser.add_argument("--gpu-batch-window-ms", type=float, default=2.0)
    parser.add_argument("--gpu-max-batch-size", type=int, default=64)
    parser.add_argument(
        "--disable-gpu-qp-service",
        action="store_true",
        help="load the training artifact in each worker instead of one shared GPU service",
    )
    parser.add_argument(
        "--training-distance-threshold",
        type=float,
        default=float("inf"),
        help="training-only OOD threshold; physical guards remain mandatory",
    )
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-vecnormalize", type=Path, default=None)
    parser.add_argument("--resume-timesteps", type=int, default=0)
    parser.add_argument(
        "--disable-training-surrogate",
        action="store_true",
        help="use exact cooperative HiGHS for every training transition",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nominal_seed = args.seed + 1000
    baseline_policies = (
        "no_control",
        "fixed25",
        "low_feed_max_air",
        "budget_max_air",
        "two_phase",
    )
    with mp.get_context("spawn").Pool(len(baseline_policies)) as pool:
        nominal_baselines = dict(
            pool.map(baseline_worker, [(nominal_seed, policy) for policy in baseline_policies])
        )

    artifact = (
        None
        if args.disable_training_surrogate
        else (str(args.training_artifact) if args.training_artifact else None)
    )
    gpu_manager = None
    gpu_service = None
    if artifact is not None and not args.disable_gpu_qp_service:
        service_models = select_consortium_models(load_sbml_models(MODEL_DIR))
        service_original_bounds = {
            name: {
                reaction.id: tuple(reaction.bounds)
                for reaction in model.exchanges
            }
            for name, model in service_models.items()
        }
        gpu_manager, gpu_service = start_cooperative_gpu_qp_service(
            service_models,
            service_original_bounds,
            artifact,
            candidates=128,
            batch_window_ms=args.gpu_batch_window_ms,
            max_batch_size=args.gpu_max_batch_size,
        )
    raw_train_env = SubprocVecEnv(
        [
            make_train_env(
                rank,
                args.seed,
                artifact,
                args.training_exact_interval,
                args.training_distance_threshold,
                gpu_service,
            )
            for rank in range(args.n_envs)
        ],
        start_method="fork",
    )
    if args.resume_vecnormalize is not None:
        train_env = VecNormalize.load(args.resume_vecnormalize, raw_train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(
            raw_train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
    if args.resume_model is not None:
        model = PPO.load(args.resume_model, env=train_env, device="cuda")
        if args.resume_timesteps and model.num_timesteps != args.resume_timesteps:
            raise ValueError(
                "resume checkpoint timesteps do not match --resume-timesteps: "
                f"model={model.num_timesteps}, argument={args.resume_timesteps}"
            )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs={
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "log_std_init": -1.2,
            },
            device="cuda",
            seed=args.seed,
            verbose=1,
        )
        # Start in a screened feasible region, then let PPO refine it. This
        # avoids spending most of the exact-FBA budget rediscovering the known
        # low-feed/high-aeration rule.
        initial_policy_action = 2.0 * LOW_AIR - 1.0
        with torch.no_grad():
            model.policy.action_net.bias.copy_(
                torch.as_tensor(
                    initial_policy_action,
                    dtype=model.policy.action_net.bias.dtype,
                    device=model.policy.action_net.bias.device,
                )
            )

    checkpoint_evaluations = []
    total = int(args.resume_timesteps or model.num_timesteps)
    initial, _ = rollout(nominal_seed, "ppo", model, train_env)
    checkpoint_evaluations.append({"timesteps": total, **initial})

    nominal_pass = False
    started = time.perf_counter()
    while total < args.max_timesteps and not nominal_pass:
        increment = min(args.stage_steps, args.max_timesteps - total)
        model.learn(total_timesteps=increment, reset_num_timesteps=False, progress_bar=False)
        # On-policy PPO always collects a complete n_envs * n_steps rollout,
        # so the realised transition count can exceed a small requested
        # increment.  Persist the framework's authoritative counter.
        total = int(model.num_timesteps)
        stage_path = args.output_dir / f"ppo_wcfs1_{total}_steps"
        model.save(stage_path)
        train_env.save(args.output_dir / f"vecnormalize_{total}_steps.pkl")
        metrics, _ = rollout(nominal_seed, "ppo", model, train_env)
        checkpoint_evaluations.append({"timesteps": total, **metrics})
        fixed = nominal_baselines["fixed25"]
        nominal_pass = bool(
            metrics["raw_return"] > fixed["raw_return"]
            and metrics["pha_mmol"] >= fixed["pha_mmol"]
            and metrics["rubber_degraded_g_l"] >= fixed["rubber_degraded_g_l"]
            and metrics["defined_feed_g_l"] <= fixed["defined_feed_g_l"]
            and metrics["minimum_final_initial_ratio"] >= 0.90
            and metrics["solver_success_rate"] >= 0.995
        )
        print(json.dumps({"timesteps": total, "nominal_effect": nominal_pass, **metrics}, ensure_ascii=False), flush=True)

    final_model = args.output_dir / "ppo_wcfs1_effect_final"
    model.save(final_model)
    final_vecnormalize = args.output_dir / "vecnormalize_final.pkl"
    train_env.save(final_vecnormalize)
    training_solver_diagnostics = train_env.env_method("get_solver_diagnostics")
    gpu_service_diagnostics = (
        None if gpu_service is None else gpu_service.diagnostics()
    )
    # Release training workers before spawning held-out evaluation workers.
    # Keeping both pools alive oversubscribes CPU cores and lengthens exact
    # multi-scenario validation without adding information.
    train_env.close()
    if gpu_service is not None:
        gpu_service.close()
    if gpu_manager is not None:
        gpu_manager.shutdown()

    scenario_seeds = [args.seed + 2000 + index for index in range(args.eval_scenarios)]
    all_rows: list[dict] = []
    grouped: dict[str, list[dict]] = {key: [] for key in ("ppo", "fixed25", "low_feed_max_air", "two_phase", "no_control")}
    nominal_trajectory: list[dict] = []
    tasks = [
        (seed, str(final_model) + ".zip", str(final_vecnormalize), index == 0)
        for index, seed in enumerate(scenario_seeds)
    ]
    with mp.get_context("spawn").Pool(min(args.eval_scenarios, 5)) as pool:
        scenario_results = pool.map(scenario_worker, tasks)
    for result_rows, trajectory in scenario_results:
        for metrics in result_rows:
            grouped[metrics["policy"]].append(metrics)
            all_rows.append(metrics)
        if trajectory:
            nominal_trajectory = trajectory
    summaries = {policy: summarize(rows, policy) for policy, rows in grouped.items()}
    gate = effect_gate(grouped["ppo"], grouped["fixed25"])
    low_rule_gap = summaries["ppo"]["mean_raw_return"] - summaries["low_feed_max_air"]["mean_raw_return"]
    payload = {
        "schema_version": 1,
        "model_dir": str(MODEL_DIR),
        "training": {
            "timesteps": total,
            "n_envs": args.n_envs,
            "stage_steps": args.stage_steps,
            "seed": args.seed,
            "device": str(model.device),
            "elapsed_seconds": time.perf_counter() - started,
            "surrogate_artifact": artifact,
            "exact_training": bool(args.disable_training_surrogate),
            "surrogate_exact_interval": args.training_exact_interval,
            "surrogate_distance_threshold": args.training_distance_threshold,
            "gpu_qp_service": gpu_service_diagnostics,
            "resumed_from_timesteps": int(args.resume_timesteps),
            "policy_action_space": "symmetric [-1, 1] mapped to physical [0, 1]",
            "common_feed_safety_cap": 0.25,
            "initialization": "low_feed_max_air action-head bias; log_std_init=-1.2",
            "solver_diagnostics": training_solver_diagnostics,
        },
        "effect_definition": "All hard constraints; paired raw-return 95% CI above Fixed-25; PHA and rubber noninferior; Defined-10 mass no higher.",
        "nominal_baselines": nominal_baselines,
        "checkpoint_evaluations": checkpoint_evaluations,
        "summaries": summaries,
        "effect_gate": gate,
        "mean_return_difference_vs_low_feed_rule": low_rule_gap,
        "interpretation": (
            "RL effect demonstrated against Fixed-25."
            if gate["passed"]
            else "Predeclared RL effect gate not reached within the maximum timesteps."
        ),
    }
    (args.output_dir / "rl_effect_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "evaluation_scenarios.csv", all_rows)
    write_csv(args.output_dir / "ppo_action_trajectory.csv", nominal_trajectory)
    plot_results(payload, nominal_trajectory, args.output_dir)
    print(json.dumps({"effect_gate": gate, "timesteps": total, "output": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
