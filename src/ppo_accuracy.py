"""Offline, CPU-only precision screens, not PPO qualification or LP solvers.

LP snapshot files do not contain environment checkpoints, rewards, or policy
values. Passing the LP screen below therefore cannot establish PPO accuracy.
The historic complementarity metric omits equality-residual/dual products and
is NOT a rigorous objective-regret bound. Its max(1, |objective|) denominator
also means that it is not a percentage for objectives smaller than one.

Trajectory comparisons require explicit units and matching layouts. No default
percentage is declared sufficient for PPO. Value-aware validation is motivated
by Farahmand et al. (2017), https://proceedings.mlr.press/v54/farahmand17a.html;
GAE follows Schulman et al., https://arxiv.org/abs/1506.02438.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


OBJECTIVE_RTOLS = (1e-5, 1e-4, 1e-3, 1e-2)
TRAJECTORY_UNITS = {
    "pha_repeat": "mmol/L", "pha_mass": "g/L", "phv_fraction": "mol/mol",
    "biomass": "g/L", "ph": "pH",
}
SCREEN_SCOPE = "screen_only_not_rl_qualification_not_rigorous_regret"


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def screen_lp_metric(metric, *, stage, objective_rtol, objective_atol=1e-9):
    """Re-score one original-LP metric without relaxing physical feasibility.

Malformed/nonfinite metrics are rejected rather than scored as zero. Failure is
reported per metric so that an otherwise readable audit can retain bad records.
"""
    if stage not in {"maxmin", "aggregate", "exchange"}:
        raise ValueError("Unknown lexicographic LP stage")
    rtol = _finite_number(objective_rtol, "objective_rtol")
    atol = _finite_number(objective_atol, "objective_atol")
    if rtol < 0 or atol < 0:
        raise ValueError("Objective tolerances must be nonnegative")
    try:
        primal, dual, gap, objective = [
            _finite_number(metric[name], name) for name in
            ("primal_residual", "dual_violation", "relative_kkt_gap", "objective")
        ]
        if min(primal, dual, gap) < 0:
            raise ValueError("Residuals and complementarity must be nonnegative")
        absolute_gap = gap * max(1.0, abs(objective))
        budget = atol + rtol * abs(objective)
        if not math.isfinite(absolute_gap) or not math.isfinite(budget):
            raise ValueError("Derived metric overflow")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return dict(stage=stage, valid=False, passed=False, reason=str(exc),
                    scope=SCREEN_SCOPE)
    hard_pass = primal <= 1e-5 and dual <= 1e-7
    return dict(stage=stage, valid=True, passed=hard_pass and absolute_gap <= budget,
                hard_constraints_passed=hard_pass, objective_screen_passed=absolute_gap <= budget,
                primal_residual=primal, dual_violation=dual, objective=objective,
                reported_relative_kkt_gap=gap, absolute_complementarity=absolute_gap,
                objective_budget=budget, objective_rtol=rtol, objective_atol=atol,
                scope=SCREEN_SCOPE)


def screen_ipm_record(record, *, objective_rtols=OBJECTIVE_RTOLS, objective_atol=1e-9):
    """Screen final metrics and strictly positive-iteration checkpoints only.

Historic initial-checkpoint lists were aliased to final metrics. All iteration
zero records are conservatively excluded, without claiming every one is corrupt.
Environment counts refer to unique final batch positions, not checkpoint count.
"""
    stage = record.get("stage")
    if stage not in {"maxmin", "aggregate", "exchange"}:
        raise ValueError("Missing/invalid stage")
    ipm = record["ipm"]
    final = ipm["metrics"]
    if not isinstance(final, list) or not final or record.get("batch") != len(final):
        raise ValueError("Final metric batch does not match input metadata")
    iterations = ipm["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("Invalid final iteration")
    objective_rtols = tuple(objective_rtols)
    if not objective_rtols:
        raise ValueError("At least one objective screening tolerance is required")
    observations = [("final", iterations, final)]
    excluded = 0
    for checkpoint in ipm.get("checkpoints", []):
        iteration = checkpoint["iteration"]
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Invalid checkpoint iteration")
        if iteration == 0:
            excluded += 1
            continue
        if iteration > ipm["iterations"] or len(checkpoint["metrics"]) != len(final):
            raise ValueError("Checkpoint shape/iteration mismatch")
        observations.append(("checkpoint", iteration, checkpoint["metrics"]))
    grids = []
    for rtol in objective_rtols:
        samples = []
        ever = set()
        hard_ever = set()
        for source, iteration, rows in observations:
            for environment, metric in enumerate(rows):
                screened = screen_lp_metric(metric, stage=stage,
                    objective_rtol=rtol, objective_atol=objective_atol)
                samples.append(dict(source=source, iteration=iteration,
                                    environment_id=environment, **screened))
                if screened.get("hard_constraints_passed"):
                    hard_ever.add(environment)
                if screened["passed"]:
                    ever.add(environment)
        grids.append(dict(objective_rtol=float(rtol), objective_atol=float(objective_atol),
            final_passed=sum(row["passed"] for row in samples if row["source"] == "final"),
            environments_ever_passed=len(ever), environments_ever_hard_feasible=len(hard_ever),
            samples=samples))
    return dict(scope=SCREEN_SCOPE, stage=stage, physical_step=record.get("step"),
        batch=len(final), status=ipm.get("status"),
        initial_checkpoints_excluded=excluded,
        initial_exclusion_reason="historical_initial_checkpoint_alias_corruption_risk",
        corrupted_initial_excluded=bool(excluded), original_primal_tolerance=1e-5,
        original_dual_tolerance=1e-7, grids=grids,
        inference="LP inputs alone cannot establish transition, reward, or PPO accuracy")


def _array(value, name, *, shape=None, boolean=False):
    raw = np.asarray(value)
    if boolean:
        if raw.dtype.kind != "b":
            raise ValueError(f"{name} must be boolean, not coerced numeric flags")
        array = raw
    else:
        if raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be numeric")
        array = raw.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} != {shape}")
    return array


def generalized_advantage(rewards, values, next_values, terminated, truncated,
                          *, gamma=.99, gae_lambda=.95):
    """GAE for time-by-environment arrays with explicit per-transition values.

For a time-limit truncation, next_values MUST be the value of the terminal
observation, not the auto-reset observation. Termination prevents bootstrap;
both termination and truncation prevent recursion into another episode.
"""
    gamma = _finite_number(gamma, "gamma")
    gae_lambda = _finite_number(gae_lambda, "gae_lambda")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    rewards = _array(rewards, "rewards")
    if rewards.ndim != 2 or min(rewards.shape) < 1:
        raise ValueError("Expected nonempty [time, environment] rewards")
    shape = rewards.shape
    values = _array(values, "values", shape=shape)
    next_values = _array(next_values, "next_values", shape=shape)
    terminated = _array(terminated, "terminated", shape=shape, boolean=True)
    truncated = _array(truncated, "truncated", shape=shape, boolean=True)
    delta = rewards + gamma * (~terminated) * next_values - values
    advantage = np.empty_like(rewards)
    tail = np.zeros(shape[1])
    for step in range(shape[0] - 1, -1, -1):
        tail = delta[step] + gamma * gae_lambda * (~(terminated[step] | truncated[step])) * tail
        advantage[step] = tail
    if not np.isfinite(advantage).all():
        raise ValueError("GAE overflow")
    return advantage


def compare_rankings(reference, candidate, *, tie_margin=0.0):
    """Paired action/condition returns per seed; close pairs are unresolved.

Rows contain seed, action_id and return. tie_margin is an explicit absolute
return difference, not a universal percentage or a policy-quality guarantee.
"""
    margin = _finite_number(tie_margin, "tie_margin")
    if margin < 0:
        raise ValueError("tie_margin must be nonnegative")
    def indexed(rows):
        index = {}
        for row in rows:
            key = (row["seed"], row["action_id"])
            if not isinstance(key[0], int) or isinstance(key[0], bool) or not isinstance(key[1], str):
                raise ValueError("Ranking identifiers require integer seed and string action_id")
            if key in index:
                raise ValueError("Duplicate ranking condition")
            index[key] = _finite_number(row["return"], "return")
        if not index:
            raise ValueError("Ranking rows cannot be empty")
        return index
    ref, cand = indexed(reference), indexed(candidate)
    if ref.keys() != cand.keys():
        raise ValueError("Ranking condition identifiers do not match")
    groups = defaultdict(list)
    for seed, action in ref:
        groups[seed].append(action)
    reports = []
    for seed, actions in sorted(groups.items()):
        comparable = reversed_pairs = unresolved = 0
        for i, a in enumerate(sorted(actions)):
            for b in sorted(actions)[i + 1:]:
                dr = ref[seed, a] - ref[seed, b]
                dc = cand[seed, a] - cand[seed, b]
                if not math.isfinite(dr) or not math.isfinite(dc):
                    raise ValueError("Ranking return difference overflow")
                if abs(dr) <= margin or abs(dc) <= margin:
                    unresolved += 1
                else:
                    comparable += 1
                    reversed_pairs += int((dr > 0) != (dc > 0))
        reports.append(dict(seed=seed, comparable_pairs=comparable,
            reversed_pairs=reversed_pairs, unresolved_pairs=unresolved,
            agreement_fraction=(comparable - reversed_pairs) / comparable if comparable else None))
    return dict(tie_margin=margin, per_seed=reports)


def compare_trajectories(reference, candidate, *, gamma=.99, gae_lambda=.95,
                         absolute_budgets=None, ranking_tie_margin=0.0):
    """Compare matched [T,B] trajectories (biomass [T,B,species]).

Required: rewards, terminated, truncated, units, species_ids and all fields in
TRAJECTORY_UNITS. Values and next_values are optional, but then both sides must
declare the same nonempty value_function_id (including frozen normalization).
    Optional events maps names (e.g. nitrogen_limited) to boolean [T,B] flags.
    Results are diagnostics, never an assertion that a PPO policy is qualified.
"""
    gamma = _finite_number(gamma, "gamma")
    gae_lambda = _finite_number(gae_lambda, "gae_lambda")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    r = _array(reference["rewards"], "reference rewards")
    c = _array(candidate["rewards"], "candidate rewards", shape=r.shape)
    if r.ndim != 2 or min(r.shape) < 1:
        raise ValueError("Expected nonempty [time, environment] rewards")
    shape = r.shape
    species = reference["species_ids"]
    if not isinstance(species, list) or not species or len(set(species)) != len(species) or not all(isinstance(s, str) for s in species):
        raise ValueError("Explicit unique species_ids required")
    if candidate["species_ids"] != species:
        raise ValueError("Species layout mismatch")
    for data in (reference, candidate):
        if any(data["units"].get(key) != unit for key, unit in TRAJECTORY_UNITS.items()):
            raise ValueError("Missing or incompatible trajectory units")
    errors = {}
    for key in TRAJECTORY_UNITS:
        expected = shape + (len(species),) if key == "biomass" else shape
        a = _array(reference[key], "reference " + key, shape=expected)
        b = _array(candidate[key], "candidate " + key, shape=expected)
        for array in (a, b):
            if np.any(array < 0) or (key == "phv_fraction" and np.any(array > 1)) or (key == "ph" and np.any(array > 14)):
                raise ValueError("Invalid physical trajectory domain: " + key)
        errors[key] = float(np.max(np.abs(a-b)))
    masks = {}
    for key in ("terminated", "truncated"):
        masks[key] = (_array(reference[key], "reference " + key, shape=shape, boolean=True),
                      _array(candidate[key], "candidate " + key, shape=shape, boolean=True))
        errors[key + "_mismatches"] = int(np.count_nonzero(masks[key][0] != masks[key][1]))
    if "events" in reference or "events" in candidate:
        if not all(isinstance(data.get("events"), dict) for data in (reference, candidate)):
            raise ValueError("Paired event dictionaries required")
        if reference["events"].keys() != candidate["events"].keys():
            raise ValueError("Event names do not match")
        for key in reference["events"]:
            if not isinstance(key, str) or not key:
                raise ValueError("Event names must be nonempty strings")
            a = _array(reference["events"][key], "reference event " + key, shape=shape, boolean=True)
            b = _array(candidate["events"][key], "candidate event " + key, shape=shape, boolean=True)
            errors["event:" + key] = int(np.count_nonzero(a != b))
    errors["reward"] = float(np.max(np.abs(r-c)))
    # Segment returns: fixed T matched sequences, discount starts at segment zero.
    # If callers concatenate episodes these are not individual episode returns.
    discount = np.power(gamma, np.arange(shape[0]))[:, None]
    rr, cr = r.sum(axis=0), c.sum(axis=0)
    rd, cd = (discount*r).sum(axis=0), (discount*c).sum(axis=0)
    errors["raw_return"] = float(np.max(np.abs(rr-cr)))
    errors["discounted_return"] = float(np.max(np.abs(rd-cd)))
    report = dict(scope="matched_trajectory_diagnostic_not_ppo_qualification",
        time_steps=shape[0], environments=shape[1], units=dict(TRAJECTORY_UNITS),
        gamma=gamma, gae_lambda=gae_lambda, max_absolute_errors=errors,
        reference_raw_segment_return=rr.tolist(), candidate_raw_segment_return=cr.tolist(),
        reference_discounted_segment_return=rd.tolist(), candidate_discounted_segment_return=cd.tolist())
    value_keys = ("values", "next_values", "value_function_id")
    if any(key in data for data in (reference, candidate) for key in value_keys):
        if not all(key in data for data in (reference, candidate) for key in value_keys):
            raise ValueError("Both trajectories require values, next_values and value_function_id")
        identity = reference["value_function_id"]
        if not isinstance(identity, str) or not identity or identity != candidate["value_function_id"]:
            raise ValueError("Use the same fixed value function and normalization")
        advantages = [generalized_advantage(data["rewards"], data["values"], data["next_values"],
            data["terminated"], data["truncated"], gamma=gamma, gae_lambda=gae_lambda)
            for data in (reference, candidate)]
        ar, ac = advantages
        errors["gae"] = float(np.max(np.abs(ar-ac)))
        errors["gae_sign_mismatches"] = int(np.count_nonzero(np.sign(ar) != np.sign(ac)))
        scale = float(np.std(ar))
        rmse = float(np.sqrt(np.mean((ar-ac)**2)))
        if not math.isfinite(scale) or not math.isfinite(rmse):
            raise ValueError("GAE comparison statistic overflow")
        normalized_rmse = rmse / scale if scale else None
        if normalized_rmse is not None and not math.isfinite(normalized_rmse):
            raise ValueError("GAE normalized statistic overflow")
        report["gae"] = dict(value_function_id=identity, reference_std=scale,
            rmse=rmse, normalized_rmse=normalized_rmse,
            zero_reference_variance=scale == 0,
            exact_zero_advantages_included_in_sign_mismatches=True)
    if "rankings" in reference or "rankings" in candidate:
        if not all("rankings" in data for data in (reference, candidate)):
            raise ValueError("Paired rankings required")
        report["rankings"] = compare_rankings(reference["rankings"], candidate["rankings"],
                                              tie_margin=ranking_tie_margin)
    budgets = {} if absolute_budgets is None else dict(absolute_budgets)
    if set(budgets) - set(errors):
        raise ValueError("Unknown or unavailable absolute error budget")
    checks = {}
    for key, value in budgets.items():
        budget = _finite_number(value, "budget " + key)
        if budget < 0:
            raise ValueError("Budgets must be nonnegative")
        checks[key] = dict(budget=budget, passed=errors[key] <= budget)
    if not all(math.isfinite(v) for v in errors.values()) or not all(np.isfinite(v).all() for v in (rr, cr, rd, cd)):
        raise ValueError("Comparison overflow")
    report["explicit_budget_checks"] = checks
    report["all_explicit_budgets_passed"] = all(row["passed"] for row in checks.values()) if checks else None
    return report
