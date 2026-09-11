"""Prove the new runner preserves the old CPU trajectory before any new test seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.qualify_gpu_exchange_tie import run_side, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference.read_text())
    seed = reference["runs"][0]["seed"]
    if seed != 20286001:
        raise ValueError("This regression may only consume the already tested development seed")
    for name, digest in reference["implementation_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Original numerical source changed: {name}")
    actions = np.random.default_rng(seed).uniform(.05, .95, (120, 5)).astype(np.float32)
    result = run_side(actions, seed, False, lambda r: None)
    cpu = reference["runs"][0]["exact"]
    differences = []
    for a, b in zip(cpu["trajectory"], result["trajectory"]):
        differences.extend([abs(a["pha"]-b["pha"]), abs(a["phv_fraction"]-b["phv_fraction"]),
            abs(a["nh4"]-b["nh4"]), *[abs(a["biomass"][k]-b["biomass"][k]) for k in a["biomass"]]])
    passed = (result["steps"] == 120 and all(r["accepted"] for r in result["trajectory"])
        and result["cpu_lp_stage_calls"] == 360 and result["gpu_lp_stage_calls"] == 0
        and result["model_fingerprints"] == cpu["model_fingerprints"]
        and max(differences, default=np.inf) <= 1e-12)
    report = dict(scope="CPU runner regression, not GPU validation", passed=passed,
        max_abs_difference_mixed_recorded_units=max(differences, default=None),
        reference=str(args.reference), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256((ROOT/"scripts/qualify_gpu_exchange_tie.py").read_bytes()).hexdigest(), result=result)
    save(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "result"}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
