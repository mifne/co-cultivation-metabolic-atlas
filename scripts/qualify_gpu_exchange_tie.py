"""Frozen sequential five-seed evaluation, only after an audited 120-step regression.

The reference stays the original three-stage CPU HiGHS dual-simplex model.
This runner never adjusts objectives, weights, tolerances or seeds from test results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_gpu_exchange_tie_replay import audit
from scripts.benchmark_cooperative_surrogate_e2e import make_environment, PHB_REPEAT_G_PER_MMOL, PHV_REPEAT_G_PER_MMOL
from scripts.qualify_gpu_simplex import environment_info, SUPPORTING_SOURCES
from src.community_solver import CooperativeCommunityFbaSolver
from src.fba_surrogate import model_fingerprint
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend

SEEDS = list(range(20286101, 20286106))
STEPS = 120


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run_side(actions, seed, gpu, checkpoint):
    env = make_environment(None, STEPS, consortium_profile="pf-helper3", initial_nh4=.05)
    env.reset(seed=seed)
    sim = env.simulator
    solver = CooperativeCommunityFbaSolver(sim.models,
        original_exchange_bounds=sim.original_bounds, maximum_coexistence_growth=.005,
        optimize_live_objectives=True, parsimonious_exchange=True, highs_method="highs-ds")
    backend = GpuExchangeTieBreakBackend(solver, tolerance=1e-8, time_limit=600) if gpu else None
    solver.linear_program_backend = backend
    sim._cooperative_solver = solver
    rows = []
    result = dict(model_fingerprints={k: model_fingerprint(v) for k, v in sim.models.items()},
        trajectory=rows, backend_history=None if backend is None else backend.history)
    started = time.perf_counter()
    try:
        for step, action in enumerate(actions):
            _, _, terminated, truncated, _ = env.step(action)
            phb = sum(v.phb_accumulated for v in sim.state.species.values())
            phv = sum(v.phv_accumulated for v in sim.state.species.values())
            accepted = "stage=parsimonious_exchange" in solver.stats.status
            rows.append(dict(step=step, accepted=accepted, status=solver.stats.status,
                biomass={k: float(v.biomass) for k, v in sim.state.species.items()},
                pha=phb*PHB_REPEAT_G_PER_MMOL+phv*PHV_REPEAT_G_PER_MMOL,
                phv_fraction=phv/(phb+phv) if phb+phv > 1e-12 else 0.,
                nh4=float(sim.state.metabolites.get("nh4_e", 0.)),
                metabolites={k: float(v) for k, v in sim.state.metabolites.items()}))
            result.update(seconds=time.perf_counter()-started, steps=len(rows),
                cpu_lp_stage_calls=solver.cpu_lp_stage_calls, gpu_lp_stage_calls=solver.gpu_lp_stage_calls)
            if (step+1) % 5 == 0 or not accepted:
                checkpoint(result)
                print(f"{seed} {'GPU-4LP' if gpu else 'CPU-3LP'} {step+1}/{STEPS}", flush=True)
            if terminated or truncated or not accepted:
                break
    except Exception as exc:
        result["execution_error"] = repr(exc)
    result.update(seconds=time.perf_counter()-started, steps=len(rows),
        cpu_lp_stage_calls=solver.cpu_lp_stage_calls, gpu_lp_stage_calls=solver.gpu_lp_stage_calls)
    checkpoint(result)
    return result


def audit_pair(seed, exact, gpu, hashes):
    # Adapt recorded data, not computed values, to the separately tested auditor.
    replay = dict(status="completed" if gpu["steps"] == STEPS and "execution_error" not in gpu else "incomplete",
        metadata=dict(seed=seed, backend="tableau-exchange-tie", stop_after=STEPS,
            model_fingerprints=gpu["model_fingerprints"], implementation_sha256=hashes,
            diagnostic_source_sha256={}),
        rows=[dict(step=r["step"]+1, status=r["status"], after=dict(
            pha=r["pha"], phv_fraction=r["phv_fraction"], biomass=r["biomass"], metabolites=r["metabolites"]))
            for r in gpu["trajectory"]], backend_history=gpu["backend_history"])
    reference = dict(implementation_sha256=hashes, runs=[dict(seed=seed, exact=exact)])
    result = audit(replay, reference)
    for key, expected in (("cpu_lp_stage_calls", 0), ("gpu_lp_stage_calls", 360)):
        if gpu.get(key) != expected:
            result["failures"].append(key)
    if exact.get("cpu_lp_stage_calls") != 360 or exact.get("gpu_lp_stage_calls") != 0:
        result["failures"].append("cpu_reference_stage_counts")
    if "execution_error" in exact or "execution_error" in gpu:
        result["failures"].append("execution_error")
    result["passed"] = not result["failures"]
    result["scope"] = "one prespecified independent condition"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    development = json.loads(args.development.read_text())
    reference = json.loads(args.reference.read_text())
    development_audit = audit(development, reference)
    if not development_audit["passed"]:
        raise RuntimeError("Development regression has not passed: "+repr(development_audit["failures"]))
    if development_audit["seed"] in SEEDS:
        raise ValueError("Development cannot use an independent test seed")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"Set {variable}=1 before launching")
    source_names = set(development_audit["source_hashes"]) | set(SUPPORTING_SOURCES) | {
        "scripts/qualify_gpu_exchange_tie.py", "scripts/audit_gpu_exchange_tie_replay.py",
        "scripts/qualify_gpu_simplex.py"}
    def hashes_now():
        return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(source_names)}
    hashes = hashes_now()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running_frozen_test", seeds=SEEDS,
        started_at_utc=datetime.now(timezone.utc).isoformat(), source_hashes=hashes,
        development=str(args.development), development_sha256=hashlib.sha256(args.development.read_bytes()).hexdigest(),
        development_reference=str(args.reference), development_audit=development_audit,
        environment=environment_info(), runs=[], concurrency=1,
        reference_policy="unchanged three-stage CPU HiGHS dual simplex",
        gpu_policy="three original stages plus exchange tie LP; host presolve/QR/postsolve",
        config=dict(steps=120, dt_h=.2, tolerance=1e-8, time_limit_per_gpu_lp_s=600,
            primary_face_relative_allowance=1e-10, expected_actual_gpu_lps_per_seed=480),
        gates=dict(pha_relative=.01, biomass_g_l=.01, phv_fraction=.01,
            original_lp_residual=1e-5, cpu_lp_calls_in_gpu_path=0),
        timing_note="Sequential accuracy verification; not a separately controlled speed benchmark")
    output = args.output_dir/"qualification.json"
    save(output, manifest)
    expected_models = development["metadata"]["model_fingerprints"]
    def abort(seed, report_path, report, reason):
        report.update(status="execution_error", error=reason)
        save(report_path, report)
        manifest["runs"].append(dict(seed=seed, passed=False, failures=[reason],
            endpoint_errors={}, report=str(report_path),
            report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest()))
        manifest["status"] = "failed_early"
    for seed in SEEDS:
        if hashes_now() != hashes:
            manifest["status"] = "source_changed"
            break
        actions = np.random.default_rng(seed).uniform(.05, .95, (120, 5)).astype(np.float32)
        report_path = args.output_dir/f"seed_{seed}.json"
        report = dict(seed=seed, status="running_cpu", source_hashes=hashes)
        def cpu_checkpoint(result):
            report["exact"] = result
            save(report_path, report)
        try:
            exact = run_side(actions, seed, False, cpu_checkpoint)
        except Exception as exc:
            abort(seed, report_path, report, "CPU initialization: "+repr(exc))
            break
        if exact["steps"] != 120 or not all(r["accepted"] for r in exact["trajectory"]) or "execution_error" in exact:
            abort(seed, report_path, report, "CPU reference did not complete all stages")
            break
        report["status"] = "running_gpu"
        def gpu_checkpoint(result):
            report["gpu"] = result
            save(report_path, report)
        try:
            gpu = run_side(actions, seed, True, gpu_checkpoint)
            checked = audit_pair(seed, exact, gpu, hashes)
        except Exception as exc:
            abort(seed, report_path, report, "GPU execution/audit: "+repr(exc))
            break
        if hashes_now() != hashes:
            checked["failures"].append("frozen_sources_changed")
        if exact["model_fingerprints"] != expected_models or gpu["model_fingerprints"] != expected_models:
            checked["failures"].append("frozen_models_changed")
        checked["passed"] = not checked["failures"]
        report.update(status="passed" if checked["passed"] else "failed", audit=checked)
        save(report_path, report)
        manifest["runs"].append(dict(seed=seed, passed=checked["passed"],
            endpoint_errors=checked["endpoint_errors"], failures=checked["failures"], report=str(report_path),
            report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest()))
        save(output, manifest)
        print(json.dumps(manifest["runs"][-1]), flush=True)
        if not checked["passed"]:
            # Preregistered fail-fast, not omission from a claimed five-seed pass.
            manifest["status"] = "failed_early"
            break
    if len(manifest["runs"]) == 5 and all(r["passed"] for r in manifest["runs"]):
        manifest["status"] = "five_seed_accuracy_pass"
    manifest["unstarted_seeds"] = [s for s in SEEDS if s not in {r["seed"] for r in manifest["runs"]}]
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(output, manifest)
    return 0 if manifest["status"] == "five_seed_accuracy_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
