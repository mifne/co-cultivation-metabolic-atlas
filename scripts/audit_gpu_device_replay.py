"""Independently recompute unchanged gates for the device-pivot fourth-LP replay.

Accepting 120 outer steps is insufficient: all 480 actual GPU LPs must pass.
This audits one development/regression trajectory, never five-seed qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = "minimize_exchange_L1_excluding_water_proton_hydroxide_on_primary_optimal_face"


def audit_history(history, steps):
    failures = []
    actual_calls = attempts = 0
    max_residual = 0.
    if len(history) != steps*3:
        failures.append("outer_stage_count")
    for i, record in enumerate(history):
        parsimony = i % 3 == 2
        if record.get("actual_gpu_lp_calls") != (2 if parsimony else 1):
            failures.append(f"actual_call_count:{i}")
        if ("tie_break" in record) != parsimony:
            failures.append(f"tie_stage_position:{i}")
        stages = [record]+([record["tie_break"]] if "tie_break" in record else [])
        actual_calls += len(stages)
        for j, stage in enumerate(stages):
            prefix = f"stage:{i}:{j}"
            gpu = stage.get("gpu", {})
            if stage.get("success") is not True or gpu.get("success") is not True or gpu.get("status") != "optimal":
                failures.append(prefix+":not_optimal")
            if gpu.get("method") != "gpu_device_bounded_simplex" or gpu.get("cpu_lp_calls") != 0:
                failures.append(prefix+":not_gpu_only")
            if stage.get("cpu_optimization_allowed") is not False:
                failures.append(prefix+":cpu_allowed")
            counts = stage.get("cpu_iteration_counts", {})
            if set(counts) != {"simplex_iteration_count", "ipm_iteration_count", "crossover_iteration_count"} or any(
                    not isinstance(v, int) or v not in {-1, 0} for v in counts.values()):
                failures.append(prefix+":cpu_iterations")
            for value in (stage.get("max_original_residual", np.inf), gpu.get("max_original_residual", np.inf)):
                if not np.isfinite(value) or not 0 <= value <= 1e-5:
                    failures.append(prefix+":residual")
                max_residual = max(max_residual, value)
            count = gpu.get("gpu_lp_attempts", 0)
            if not isinstance(count, int) or count < 1:
                failures.append(prefix+":missing_attempt_count")
            else:
                attempts += count
        if parsimony:
            if record.get("selection_policy") != POLICY:
                failures.append(f"selection_policy:{i}")
            optimum = record.get("primary_optimum", np.nan)
            selected = record.get("selected_primary_value", np.nan)
            allowance = record.get("primary_face_allowance", np.nan)
            expected = 1e-10*max(1., abs(optimum))
            if (not np.all(np.isfinite([optimum, selected, allowance])) or
                abs(allowance-expected) > 1e-15 or selected > optimum+allowance+1e-8):
                failures.append(f"primary_objective_retention:{i}")
    if actual_calls != 4*steps:
        failures.append("actual_total_count")
    return dict(passed=not failures, failures=failures, outer_stages=len(history),
        actual_gpu_lp_calls=actual_calls, gpu_lp_attempts=attempts, max_original_residual=max_residual)


def audit(replay, reference, *, check_sources=True):
    failures = []
    meta = replay["metadata"]
    if meta.get("backend") != "device-exchange-tie" or meta.get("stop_after") != 120:
        failures.append("configuration")
    if replay.get("status") != "completed":
        failures.append("not_completed")
    matches = [r for r in reference["runs"] if r["seed"] == meta["seed"]]
    if len(matches) != 1:
        raise ValueError("Reference seed must match exactly once")
    cpu = matches[0]["exact"]
    rows = replay["rows"]
    trace = cpu["trajectory"]
    if len(rows) != 120 or len(trace) != 120:
        failures.append("trajectory_length")
    if meta["model_fingerprints"] != cpu["model_fingerprints"]:
        failures.append("model_fingerprint")
    hashes = dict(meta["implementation_sha256"], **meta.get("diagnostic_source_sha256", {}))
    if meta["implementation_sha256"] != reference["implementation_sha256"]:
        failures.append("reference_source_hashes")
    if "src/gpu_exchange_tie_break.py" not in hashes:
        failures.append("missing_tie_source_hash")
    for required in ("src/gpu_device_bounded_simplex.py", "src/gpu_simplex_pivot_batch.py",
                     "scripts/run_gpu_device_regression.py"):
        if required not in hashes:
            failures.append(f"missing_device_source:{required}")
    if meta.get("basis_cache_mode") == "separate_original_objectives":
        if "src/gpu_stage_cache_backend.py" not in hashes:
            failures.append("missing_stage_cache_source")
        for i, stage in enumerate(replay.get("backend_history", [])):
            expected = ("maxmin", "aggregate", "exchange_primary")[i % 3]
            if stage.get("gpu", {}).get("warm_cache_namespace") != expected:
                failures.append(f"incorrect_cache_namespace:{i}")
            if i % 3 == 2 and stage.get("tie_break", {}).get("gpu", {}).get("warm_cache_namespace") != "exchange_tie":
                failures.append(f"incorrect_tie_cache_namespace:{i}")
    if check_sources:
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                failures.append(f"source_changed:{name}")
    trajectory_errors = []
    for i, (a, b) in enumerate(zip(trace, rows)):
        after = b["after"]
        if a["step"] != i or b["step"] != i+1:
            failures.append(f"step_sequence:{i}")
        if not a.get("accepted") or any("stage=parsimonious_exchange" not in s
                for s in (a.get("status", ""), b.get("status", ""))):
            failures.append(f"incomplete_stage:{i}")
        if set(a["biomass"]) != set(after["biomass"]):
            raise ValueError("Species mismatch")
        values = [a["pha"], a["phv_fraction"], *a["biomass"].values(),
            after["pha"], after["phv_fraction"], *after["biomass"].values(), *after["metabolites"].values()]
        if not np.all(np.isfinite(values)):
            failures.append(f"nonfinite:{i}")
        trajectory_errors.append(dict(step=i+1,
            pha_relative=abs(a["pha"]-after["pha"])/max(abs(a["pha"]), 1e-9),
            biomass_g_l=max(abs(a["biomass"][k]-after["biomass"][k]) for k in a["biomass"]),
            phv_fraction=abs(a["phv_fraction"]-after["phv_fraction"])))
    endpoint = trajectory_errors[-1] if trajectory_errors else {}
    for name in ("pha_relative", "biomass_g_l", "phv_fraction"):
        if not 0 <= endpoint.get(name, np.inf) <= .01:
            failures.append("endpoint:"+name)
    lp_audit = audit_history(replay.get("backend_history", []), 120)
    failures.extend(lp_audit["failures"])
    return dict(scope="single regression audit, not independent five-seed qualification",
        passed=not failures, failures=failures, seed=meta["seed"], endpoint_errors=endpoint,
        lp_audit=lp_audit, trajectory_errors=trajectory_errors,
        cpu_reference_changed=False, source_hashes=hashes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(json.loads(args.replay.read_text()), json.loads(args.reference.read_text()))
    result.update(replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory_errors"}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
