"""Read-only progress diagnostics; never labels a partial log qualified."""
import argparse
import ast
import json
from pathlib import Path
import statistics


def summarize(path):
    records=[]; last_progress=None
    for line in path.read_text(errors="replace").splitlines():
        if " step " in line: last_progress=line
        if line.startswith("{'method': 'gpu_"):
            try: record=ast.literal_eval(line)
            except (SyntaxError,ValueError): continue
            records.append(record)
    successful=[r for r in records if r.get("success")]
    phase=[r["phase_iterations"] for r in records if "phase_iterations" in r]
    iterations=sum(sum(p) for p in phase)
    return dict(log=str(path),last_progress=last_progress,completed_logical_lp_stages=len(records),
        successful_logical_lp_stages=len(successful),unrecovered_lp_stages=len(records)-len(successful),
        gpu_lp_attempts=sum(r.get("gpu_lp_attempts",1) for r in records),
        gpu_primal_repairs=sum(r.get("gpu_primal_repairs",0) for r in records),
        maximum_original_residual=max((r["max_original_residual"] for r in records),default=None),
        phase_one_iteration_fraction=sum(p[0] for p in phase)/iterations if iterations else None,
        recorded_lp_seconds=sum(r["total_seconds"] for r in records),
        median_lp_seconds=statistics.median(r["total_seconds"] for r in records) if records else None,
        qualification="not_determined_from_log; use complete JSON gates")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("logs",type=Path,nargs="+")
    args=p.parse_args()
    print(json.dumps([summarize(path) for path in args.logs],indent=2))


if __name__=="__main__": main()
