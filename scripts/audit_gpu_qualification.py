"""Independent, read-only recomputation of frozen GPU rollout qualification.

Does not import an optimizer or change the numerical implementation. Endpoint
errors are recomputed from the actual trajectories, not trusted pass flags.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = list(range(20286001, 20286006))


def audit(manifest_path, root=ROOT):
    manifest = json.loads(Path(manifest_path).read_text())
    problems, rows = [], []
    if manifest.get("seeds") != EXPECTED_SEEDS:
        problems.append("Unexpected validation seeds")
    expected_gates = dict(pha_relative=.01, biomass_g_l=.01, phv_mole_fraction=.01,
        steps=120, cpu_lp_calls=0, original_lp_feasibility_residual=1e-5)
    if manifest.get("gates") != expected_gates:
        problems.append("Predefined qualification gates changed")
    items = manifest.get("runs", [])
    if sorted(r.get("seed", -1) for r in items) != EXPECTED_SEEDS:
        problems.append("All five unique completed reports are required")
    for group in ("implementation_sha256", "supporting_source_sha256"):
        if not manifest.get(group): problems.append(f"Missing {group}")
        for name, digest in manifest.get(group, {}).items():
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                problems.append(f"Current source differs: {name}")
    expected_models = None
    for item in sorted(items, key=lambda r: r.get("seed", -1)):
        seed = item.get("seed")
        failures = []
        try:
            raw = (root / item["report"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item.get("report_sha256"):
                failures.append("Report hash mismatch")
            data = json.loads(raw)
            for key in ("steps", "method", "host_presolve", "rank_reduce", "objective_scale",
                        "quadratic_regularization", "reaction_support", "tolerance", "time_limit"):
                if data["config"][key] != manifest["config"][key]:
                    failures.append(f"Frozen solver configuration differs: {key}")
            if data["implementation_sha256"] != manifest["implementation_sha256"]:
                failures.append("Frozen implementation mismatch")
            if len(data["runs"]) != 1: raise ValueError("Exactly one rollout pair required")
            run = data["runs"][0]
            if run["seed"] != seed: failures.append("Seed mismatch")
            cpu, gpu = run["exact"], run["gpu"]
            if cpu["model_fingerprints"] != gpu["model_fingerprints"]:
                failures.append("CPU/GPU GEM mismatch")
            if expected_models is None: expected_models = cpu["model_fingerprints"]
            if cpu["model_fingerprints"] != expected_models:
                failures.append("Different GEM across validation seeds")
            for label, result in (("CPU", cpu), ("GPU", gpu)):
                trace = result["trajectory"]
                if result["steps"] != 120 or len(trace) != 120:
                    raise ValueError(f"Incomplete {label} trajectory")
                if [t["step"] for t in trace] != list(range(120)):
                    failures.append(f"Invalid {label} trajectory step indices")
                if not all(t["accepted"] and "stage=parsimonious_exchange" in t["status"] for t in trace):
                    failures.append(f"Incomplete {label} three-stage solve")
                if not all(math.isfinite(v) for t in trace for v in
                    (t["pha"], t["phv_fraction"], t["nh4"], *t["biomass"].values())):
                    failures.append(f"Nonfinite {label} state")
                if not math.isfinite(result["seconds"]) or result["seconds"] <= 0:
                    failures.append(f"Invalid {label} timing")
            if cpu["cpu_lp_stage_calls"] != 360 or cpu["gpu_lp_stage_calls"] != 0:
                failures.append("CPU reference call count mismatch")
            if gpu["gpu_lp_stage_calls"] != 360 or gpu["cpu_lp_stage_calls"] != 0:
                failures.append("GPU online LP call count mismatch")
            history = gpu["backend_history"]
            if len(history) != 360: failures.append("Missing GPU stage audit records")
            residuals = [h["max_original_residual"] for h in history]
            if not all(math.isfinite(x) and 0 <= x <= 1e-5 for x in residuals):
                failures.append("Original full LP feasibility gate failed")
            if not all(h["success"] and h["gpu"]["success"] and
                       h["gpu"]["status"] == "optimal" for h in history):
                failures.append("A logical GPU LP stage failed")
            if not all(h["cpu_optimization_allowed"] is False and
                       h["gpu"]["cpu_lp_calls"] == 0 and
                       all(v <= 0 for v in h["cpu_iteration_counts"].values()) for h in history):
                failures.append("CPU optimizer activity or permission detected")
            a, b = cpu["trajectory"][-1], gpu["trajectory"][-1]
            if set(a["biomass"]) != set(b["biomass"]) or len(a["biomass"]) != 3:
                raise ValueError("Three matching species required")
            errors = dict(pha_relative=abs(a["pha"] - b["pha"]) / max(abs(a["pha"]), 1e-9),
                biomass_max=max(abs(a["biomass"][k] - b["biomass"][k]) for k in a["biomass"]),
                phv_fraction=abs(a["phv_fraction"] - b["phv_fraction"]))
            if not all(math.isfinite(v) and 0 <= v <= .01 for v in errors.values()):
                failures.append("Joint endpoint accuracy gate failed")
            for key, value in errors.items():
                if not math.isclose(value, run["errors"][key], rel_tol=1e-12, abs_tol=1e-15):
                    failures.append(f"Stored endpoint error differs: {key}")
            if run.get("error_scope") != "full_horizon_endpoint":
                failures.append("Not a full-horizon endpoint comparison")
            stage_totals = []
            for i, name in enumerate(("maxmin_growth", "aggregate_objective", "parsimonious_exchange")):
                stages = history[i::3]
                stage_totals.append(dict(stage=name, count=len(stages),
                    total_seconds=sum(h["total_seconds"] for h in stages),
                    host_presolve_seconds=sum(h["host_presolve_seconds"] for h in stages),
                    hybrid_solver_seconds=sum(h["gpu"]["total_seconds"] for h in stages),
                    hybrid_crash_setup_seconds=sum(h["gpu"].get("crash_setup_seconds_total",
                        h["gpu"]["host_qr_crash_seconds"]) for h in stages),
                    phase_one_iterations=sum(h["gpu"]["phase_iterations"][0] for h in stages),
                    phase_two_iterations=sum(h["gpu"]["phase_iterations"][1] for h in stages)))
            rows.append(dict(seed=seed, passed=not failures, problems=failures, errors=errors,
                max_full_lp_residual=max(residuals),
                gpu_logical_lp_stages=len(history),
                gpu_lp_attempts=sum(h["gpu"].get("gpu_lp_attempts", 1) for h in history),
                gpu_primal_repairs=sum(h["gpu"].get("gpu_primal_repairs", 0) for h in history),
                cpu_reference_seconds=cpu["seconds"],gpu_rollout_seconds=gpu["seconds"],
                stage_totals=stage_totals))
        except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
            rows.append(dict(seed=seed, passed=False, problems=failures+[str(exc)]))
    passed = not problems and len(rows) == 5 and all(r["passed"] for r in rows)
    if passed != (manifest.get("status") == "five_seed_accuracy_pass"):
        problems.append("Recomputed status does not agree with manifest")
        passed = False
    return dict(passed=passed, problems=problems, runs=rows,
        scope="Numerical agreement with CPU model; not biological accuracy or speed qualification",
        timing_scope="Solver and crash times include host control/transfers; not pure GPU kernel times",
        cpu_iteration_note="Nonpositive HiGHS counters can be unavailable (-1); CPU run is explicitly disabled")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.qualification)
    if args.output: args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
