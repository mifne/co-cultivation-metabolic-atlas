"""Read an Nsight SQLite export without confusing API time with GPU kernel time."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    db = sqlite3.connect(f"file:{args.database.as_posix()}?mode=ro", uri=True)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    api = [dict(name=name, recorded_calls=count, recorded_api_seconds=duration/1e9)
        for name, count, duration in db.execute("""
        SELECT s.value, COUNT(*), SUM(r.end-r.start)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r JOIN StringIds AS s ON s.id=r.nameId
        GROUP BY r.nameId ORDER BY SUM(r.end-r.start) DESC""")]
    diagnostics = [r[0] for r in db.execute("SELECT text FROM DIAGNOSTIC_EVENT")]
    warnings = [message for message in diagnostics if any(s in message.lower()
        for s in ("not supported", "not all", "cannot be traced"))]
    kernel_available = "CUPTI_ACTIVITY_KIND_KERNEL" in tables and db.execute(
        "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] > 0
    probe = json.loads(args.probe.read_text())
    iterations = sum(r["gpu"]["iterations"]+r.get("tie_break", {}).get("gpu", {}).get("iterations", 0)
        for r in probe["history"])
    result = dict(scope="saved-state performance diagnostic; not a full-run utilization or speedup result",
        status="partial_api_trace_only" if not kernel_available else "kernel_trace_present",
        database=str(args.database), database_sha256=hashlib.sha256(args.database.read_bytes()).hexdigest(),
        probe=str(args.probe), profiled_wall_seconds=probe["elapsed_seconds"],
        numerical_solve_success=probe["success"], gpu_lp_calls=probe["actual_gpu_lp_calls"],
        simplex_iterations=iterations, cuda_api_records=api, gpu_kernel_trace_available=bool(kernel_available),
        diagnostics=diagnostics, warnings=warnings,
        gpu_active_fraction=None,
        interpretation="API durations can include waits for prior GPU work and cannot be equated to physical transfer time. Missing GPU activity must not be interpreted as zero GPU computation. Recorded events may be incomplete; counts are observed records, not a guaranteed complete census.")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
