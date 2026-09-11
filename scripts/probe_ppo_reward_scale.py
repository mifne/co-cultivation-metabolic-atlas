"""Small CPU-only fixed-action sensitivity probe, not PPO training or speedup.

The terminal-PHA 1% quantity is an illustrative reward scale only. It is NOT a
trajectory-return error bound: timing, clipping, and other reward terms matter.
Numerical GEM simulation is imported only by main(); summary tests need NumPy
only. Run this serial diagnostic without a concurrent performance experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_FILES = (
    "scripts/probe_ppo_reward_scale.py", "scripts/benchmark_gpu_lexicographic.py",
    "scripts/benchmark_cooperative_surrogate_e2e.py", "src/rl_environment.py",
    "src/dfba_simulator.py", "src/community_solver.py",
)


def fixed_actions(steps=8):
    """Three predeclared controls, differing only in common-feed action index3."""
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 120:
        raise ValueError("steps must be an integer in [1, 120]")
    baseline = np.asarray([.25, .25, .25, .25, .5], dtype=np.float32)
    cases = {}
    for name, amount in (("baseline", .25), ("lower_common_feed", .20), ("higher_common_feed", .30)):
        action = baseline.copy()
        action[3] = amount
        cases[name] = np.repeat(action[None, :], steps, axis=0)
    return cases


def _number(value, name):
    if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
        raise ValueError(name + " must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def summarize_case(result, actions, *, gamma=.99):
    """Summarize a complete baseline run without silently accepting partial runs.

Stage counters are copied as stage/API counters only. Internal HiGHS retries
and solver-run counts are not available from this benchmark's return payload.
"""
    gamma = _number(gamma, "gamma")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 5 or len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("Finite [steps, 5] actions required")
    if np.any(actions < 0) or np.any(actions > 1):
        raise ValueError("Physical control fractions must lie in [0,1]")
    rows = result["trajectory"]
    if result["steps"] != len(actions) or len(rows) != len(actions):
        raise ValueError("Incomplete trajectory is not a matched-horizon comparison")
    if result["cpu_lp_stage_calls"] != 3 * len(actions) or result["gpu_lp_stage_calls"] != 0:
        raise ValueError("Expected three CPU LP stage calls per step and zero GPU stages")
    if not result.get("model_fingerprints"):
        raise ValueError("Missing model fingerprints")
    units = result["telemetry_units"]
    if units.get("pha") != "g/L" or units.get("pha_repeat_mmol_l") != "mmol/L":
        raise ValueError("PHA amount/mass units do not match benchmark schema")
    rewards = []
    for index, row in enumerate(rows):
        if row["step"] != index or not row["accepted"] or not row["status"].endswith("stage=parsimonious_exchange"):
            raise ValueError("Nonoptimal or incomplete lexicographic stage status")
        if row["terminated"] or row["truncated"]:
            raise ValueError("Unexpected episode boundary in the fixed-length short probe")
        actual = np.asarray(row["action"])
        if actual.shape != (5,) or not np.array_equal(actual, actions[index].astype(float)):
            raise ValueError("Executed actions differ from the predeclared controls")
        rewards.append(_number(row["reward_raw"], "reward_raw"))
    rewards = np.asarray(rewards, dtype=np.float64)
    raw_return = float(rewards.sum())
    discounted = float(np.dot(np.power(gamma, np.arange(len(rewards))), rewards))
    repeat = _number(rows[-1]["pha_repeat_mmol_l"], "terminal PHA repeat")
    mass = _number(rows[-1]["pha"], "terminal PHA mass")
    if repeat < 0 or mass < 0 or not math.isfinite(raw_return) or not math.isfinite(discounted):
        raise ValueError("Invalid endpoint or return")
    illustrative = .01 * repeat * 500.
    if not math.isfinite(illustrative):
        raise ValueError("Reward-scale overflow")
    absolute_reward_sum = float(np.abs(rewards).sum())
    if not math.isfinite(absolute_reward_sum):
        raise ValueError("Absolute reward sum overflow")
    return dict(steps=len(rows), simulated_hours=len(rows) * .2, gamma=gamma,
        raw_return=raw_return, discounted_return=discounted,
        first_step_reward=float(rewards[0]), rewards_after_first_step_sum=float(rewards[1:].sum()),
        first_step_fraction_of_absolute_reward_sum=(abs(float(rewards[0])) / absolute_reward_sum
                                                    if absolute_reward_sum else None),
        endpoint_pha_repeat_mmol_l=repeat, endpoint_pha_mass_g_l=mass,
        illustrative_1pct_terminal_pha_repeat_reward_units=illustrative,
        illustrative_scope="0.01 * terminal_PHA_mmol_per_L * 500; not a return-error bound",
        cpu_lp_stage_calls_recorded=result["cpu_lp_stage_calls"],
        gpu_lp_stage_calls_recorded=result["gpu_lp_stage_calls"],
        internal_cpu_optimizer_runs=None,
        counter_scope="benchmark LP stage/API counters, not internal HiGHS optimizer runs",
        model_fingerprints=dict(result["model_fingerprints"]))


def pairwise_margins(cases):
    """Compare all control pairs and keep zero-margin ratios undefined."""
    names = list(cases)
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = cases[left], cases[right]
            if a["steps"] != b["steps"] or a["gamma"] != b["gamma"] or a["model_fingerprints"] != b["model_fingerprints"]:
                raise ValueError("Policy comparisons require equal horizon, gamma, and models")
            margin = b["raw_return"] - a["raw_return"]
            discounted_margin = b["discounted_return"] - a["discounted_return"]
            scales = {name: cases[name]["illustrative_1pct_terminal_pha_repeat_reward_units"]
                      for name in (left, right)}
            if not all(math.isfinite(value) for value in (margin, discounted_margin, *scales.values())):
                raise ValueError("Pairwise reward comparison overflow")
            ratios = {name: value / abs(margin) if margin else None for name, value in scales.items()}
            if any(value is not None and not math.isfinite(value) for value in ratios.values()):
                raise ValueError("Illustrative scale ratio overflow")
            rows.append(dict(left=left, right=right, right_minus_left_raw_return=margin,
                absolute_raw_return_margin=abs(margin),
                right_minus_left_discounted_return=discounted_margin,
                illustrative_scale_over_absolute_raw_margin=ratios,
                scope="signal-scale illustration; not policy significance or error-bound qualification"))
    return rows


def _source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202609062)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actions = fixed_actions(args.steps)
    gamma = _number(args.gamma, "gamma")
    if not 0 <= gamma <= 1 or args.seed < 0:
        parser.error("gamma must lie in [0,1] and seed must be nonnegative")
    if args.output.exists():
        raise FileExistsError("Reward-scale probes never overwrite results")
    # Imports are deliberately lazy; summary/unit tests never initialize a GPU,
    # load GEMs, or run a CPU optimizer. Backend=None stays on CPU HiGHS.
    from scripts.benchmark_gpu_lexicographic import run
    import src.community_solver as community

    original_linprog = community.linprog
    observed = []

    def strict_linprog(*positional, **keywords):
        solution = original_linprog(*positional, **keywords)
        good = bool(solution.success and solution.x is not None and np.isfinite(solution.x).all()
                    and np.isfinite(solution.fun))
        observed.append(dict(success=good, status=int(solution.status),
                             message=str(solution.message)))
        if not good:
            raise RuntimeError("CPU LP stage failed; refusing a partial-stage sensitivity comparison")
        return solution

    before = _source_hashes()
    report = dict(schema_version=1, status="in_progress",
        role="CPU_fixed_control_reward_scale_not_PPO_training_or_speedup",
        consortium="pf-helper3", dt_hours=.2, requested_steps=args.steps, seed=args.seed,
        gamma=gamma, backend="CPU SciPy HiGHS dual simplex, original benchmark settings",
        initial_nh4_mmol_l=.05, source_sha256=before, cases={},
        units=dict(raw_return="raw model reward units", discounted_return="raw model reward units",
                   pha_repeat="mmol/L", pha_mass="g/L", action="dimensionless physical fraction"),
        actual_gpu_compute_requested=False, learned_policy_used=False,
        interpretation="Short-horizon action sensitivity, not biological validation, PPO efficacy, or performance comparison")
    failure = None
    started = time.perf_counter()
    try:
        # Narrow process-local observer: same solver and arguments, but reject
        # any intermediate failure that the CPU wrapper might otherwise mask.
        with patch.object(community, "linprog", strict_linprog):
            for name, array in actions.items():
                print(json.dumps(dict(status="starting_cpu_policy", policy=name,
                                      seed=args.seed, steps=args.steps)), flush=True)
                offset = len(observed)
                result = run(array, args.seed, backend=None)
                summary = summarize_case(result, array, gamma=gamma)
                stage_results = observed[offset:]
                if len(stage_results) != 3 * args.steps or not all(row["success"] for row in stage_results):
                    raise RuntimeError("Did not observe all three successful CPU API stage results")
                report["cases"][name] = dict(actions=array.tolist(), summary=summary,
                    scipy_stage_result_observations=stage_results, run=result)
        after = _source_hashes()
        report["source_sha256_after"] = after
        if before != after:
            raise RuntimeError("Implementation changed during sensitivity measurement")
        summaries = {name: data["summary"] for name, data in report["cases"].items()}
        report["policy_pair_return_margins"] = pairwise_margins(summaries)
        report["status"] = "completed_short_horizon_reward_scale_only"
    except Exception as exc:
        report["status"] = "failed_no_complete_policy_comparison"
        report["error"] = repr(exc)
        failure = exc
    report["diagnostic_total_wall_seconds"] = time.perf_counter() - started
    report["observed_cpu_api_stage_results"] = observed
    report["cpu_counter_caveat"] = "API/LP stage observations do not enumerate internal optimizer runs or retries"
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps(dict(output=str(args.output), status=report["status"])), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
