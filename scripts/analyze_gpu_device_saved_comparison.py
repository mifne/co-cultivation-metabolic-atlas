"""Audit full original LPs and compare timings from an isolated saved-state run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="Optional earlier reference/device saved-state benchmark")
    args = p.parse_args()
    report = json.loads(args.input.read_text())
    reports = [report]
    if args.reference:
        reports.append(json.loads(args.reference.read_text()))
    capture = ROOT/"results/pf_seed20286001_cpu_diagnostic"
    meta = json.loads((capture/"metadata.json").read_text())
    replay = json.loads((capture/"replay.json").read_text())
    failures, checks, comparisons = [], [], []
    for source in reports:
        if source["status"] != "completed" or not source["records"]:
            failures.append("incomplete_or_empty_run")
        for name, digest in source["source_hashes"].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                failures.append("source_changed:"+name)
    indexed = {}
    for record in [record for source in reports for record in source["records"]]:
        backend, step, stage = (record[k] for k in ("backend", "step", "stage"))
        key = (step, stage)
        if (backend, key) in indexed:
            failures.append(f"duplicate:{backend}:{key}")
        indexed[backend, key] = record
        lp_path = capture/f"step_{step:03d}_stage_{stage}.npz"
        if hashlib.sha256(lp_path.read_bytes()).hexdigest() != record["lp_sha256"]:
            failures.append(f"saved_lp_changed:{backend}:{key}")
        if not np.isfinite(record["seconds"]) or record["seconds"] <= 0:
            failures.append(f"invalid_timing:{backend}:{key}")
        d = np.load(lp_path)
        a = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
        x = np.asarray(record.get("values", []))
        check = dict(backend=backend, step=step, stage=stage, seconds=record["seconds"])
        if not record["success"] or x.shape != d["c"].shape or not np.isfinite(x).all():
            failures.append(f"invalid_values:{backend}:{key}")
            continue
        neq = int(d["neq"])
        activity = a@x
        residual = max(np.max(np.abs(activity[:neq]-d["rhs"][:neq]), initial=0),
            np.max(np.maximum(activity[neq:]-d["rhs"][neq:], 0), initial=0),
            np.max(np.maximum(d["lower"]-x, 0), initial=0),
            np.max(np.maximum(x-d["upper"], 0), initial=0))
        error = abs(float(d["c"]@x)-float(d["cpu_objective"]))
        check.update(recomputed_full_residual=float(residual), objective_absolute_error=error)
        if not residual <= 1e-5 or not error <= 1e-7*max(1., abs(float(d["cpu_objective"]))):
            failures.append(f"full_lp_gate:{backend}:{key}")
        stages = [record["history"]]
        if stage == 3:
            if "tie_break" not in stages[0]:
                failures.append(f"missing_tie:{backend}:{key}")
            else:
                stages += [stages[0]["tie_break"]]
                h = record["history"]
                if h["selected_primary_value"] > h["primary_optimum"]+h["primary_face_allowance"]+1e-8:
                    failures.append(f"primary_objective_loss:{backend}:{key}")
        for h in stages:
            gpu = h.get("gpu", {})
            expected_method = "gpu_bounded_simplex" if backend == "reference" else "gpu_device_bounded_simplex"
            if (not h.get("success") or not gpu.get("success") or gpu.get("cpu_lp_calls") != 0 or
                    h.get("cpu_optimization_allowed") is not False or gpu.get("status") != "optimal" or
                    gpu.get("method") != expected_method):
                failures.append(f"not_gpu_optimal:{backend}:{key}")
            counts = h.get("cpu_iteration_counts", {})
            if set(counts) != {"simplex_iteration_count", "ipm_iteration_count", "crossover_iteration_count"} or any(
                    not isinstance(v, int) or v not in (-1, 0) for v in counts.values()):
                failures.append(f"cpu_iteration_counts:{backend}:{key}")
            if not 0 <= h.get("max_original_residual", np.inf) <= 1e-5:
                failures.append(f"inner_full_residual:{backend}:{key}")
        check.update(gpu_lp_calls=len(stages), iterations=sum(h["gpu"]["iterations"] for h in stages),
            phase1_iterations=sum(h["gpu"]["phase_iterations"][0] for h in stages),
            refactorizations=sum(h["gpu"].get("refactorizations", 0) for h in stages),
            device_pivot_batches=sum(h["gpu"].get("device_pivot_batches", 0) for h in stages))
        checks.append(check)
    for (backend, key), b in indexed.items():
        if backend == "reference" or ("reference", key) not in indexed:
            continue
        a = indexed["reference", key]
        step, stage = key
        item = dict(backend=backend, step=step, stage=stage, reference_gpu_seconds=a["seconds"],
            candidate_gpu_seconds=b["seconds"], reference_gpu_over_candidate_gpu=a["seconds"]/b["seconds"])
        if a.get("values") and b.get("values") and stage == 3:
            delta = np.asarray(b["values"])-np.asarray(a["values"])
            biomass = replay["rows"][step-1]["before"]["biomass"]
            exchanges = {met: sum(biomass[species]*stoich*delta[index]
                for species, index, stoich, reaction in terms) for met, terms in meta["exchange_terms"].items()}
            item["max_net_exchange_rate_difference_mM_h"] = max(map(abs, exchanges.values()), default=0.)
            item["exchange_rate_differences_mM_h"] = exchanges
            item["max_species_growth_rate_difference_h_1"] = max(
                abs(delta[index]*coefficient) for index, coefficient in meta["growth_terms"].values())
        comparisons.append(item)
    out = dict(passed=not failures, failures=failures, scope="saved-state LP diagnostic; n=1 per case, not end-to-end speedup",
        source=str(args.input), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        optional_reference=str(args.reference) if args.reference else None,
        optional_reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=checks, comparisons=comparisons,
        limitations=["Original CPU timing not measured here", "Same-state outputs do not certify 24h trajectories",
                     "Fixed backend order, warmed libraries, no confidence interval",
                     "Stage 3 includes the fourth tie-selection LP"])
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in {"checks", "comparisons"}}), flush=True)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
