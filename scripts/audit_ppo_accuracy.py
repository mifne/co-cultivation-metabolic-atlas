"""Offline screens only; never loads LP reference vectors or runs an optimizer."""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ppo_accuracy import OBJECTIVE_RTOLS, screen_ipm_record, compare_trajectories

DEFAULT_INPUTS = [ROOT / "results" / name for name in (
    "pf_ipm_forest_fgmres32_maxmin4_20260906.json",
    "pf_ipm_forest_fgmres32_aggregate4_20260906.json",
    "pf_ipm_forest_fgmres32_exchange4_20260906.json",
    "pf_ipm_forest_fgmres32_exchange_step2_4_20260906.json",
)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", help="IPM JSON filenames/globs; omitted uses four existing final probes")
    parser.add_argument("--objective-rtols", nargs="+", type=float, default=OBJECTIVE_RTOLS)
    parser.add_argument("--objective-atol", type=float, default=1e-9)
    parser.add_argument("--reference-trajectory", type=Path)
    parser.add_argument("--candidate-trajectory", type=Path)
    parser.add_argument("--absolute-budgets", type=Path, help="JSON mapping error name to absolute budget")
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--ranking-tie-margin", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Accuracy audits never overwrite results")
    if bool(args.reference_trajectory) != bool(args.candidate_trajectory):
        parser.error("Both trajectory files are required")
    if args.absolute_budgets and not args.reference_trajectory:
        parser.error("Trajectory budgets require paired trajectories")
    paths = DEFAULT_INPUTS if args.inputs is None else []
    for pattern in args.inputs or []:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        paths.extend(Path(p) for p in matches)
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if not paths and not args.reference_trajectory:
        parser.error("At least one IPM record or paired trajectories required")
    report = dict(schema_version=1, role="offline_precision_screen_not_training_or_qualification",
        cpu_lp_calls=0, gpu_calls=0, reference_vectors_loaded=False,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT / "src/ppo_accuracy.py", Path(__file__).resolve())}, lp_records=[])
    for path in paths:
        data = path.read_bytes()
        screen = screen_ipm_record(json.loads(data), objective_rtols=args.objective_rtols,
                                   objective_atol=args.objective_atol)
        report["lp_records"].append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), **screen))
    if args.reference_trajectory:
        inputs = [p.read_bytes() for p in (args.reference_trajectory, args.candidate_trajectory)]
        budgets = json.loads(args.absolute_budgets.read_text()) if args.absolute_budgets else None
        report["trajectory"] = compare_trajectories(*(json.loads(data) for data in inputs),
            gamma=args.gamma, gae_lambda=args.gae_lambda, absolute_budgets=budgets,
            ranking_tie_margin=args.ranking_tie_margin)
        report["trajectory_input_sha256"] = [hashlib.sha256(data).hexdigest() for data in inputs]
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps(dict(output=str(args.output), lp_records=len(paths),
        summary=[dict(stage=r["stage"], step=r["physical_step"], batch=r["batch"],
                      ever_passed=[g["environments_ever_passed"] for g in r["grids"]],
                      initial_excluded=r["initial_checkpoints_excluded"])
                 for r in report["lp_records"]]), indent=2))


if __name__ == "__main__":
    main()
