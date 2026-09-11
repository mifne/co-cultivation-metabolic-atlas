"""Profile one saved exchange LP plus its GPU tie-break, no CPU optimizer.

Launch under nsys with --capture-range=cudaProfilerApi. Do not run concurrently
with accuracy qualification. A profiled cold saved state is not a throughput benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gpu_exchange_tie_break import GpuExchangeTieBreakBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lp", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads(args.metadata.read_text())
    for name, digest in meta["implementation_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed: {name}")
    d = np.load(args.lp)
    matrix = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    neq = int(d["neq"])
    layout = SimpleNamespace(n_fluxes=len(meta["reaction_ids"]), _exchange_terms=meta["exchange_terms"])
    backend = GpuExchangeTieBreakBackend(layout, tolerance=1e-8, time_limit=600)
    cp = backend.engine.cp
    # Warm CUDA/BLAS/solver libraries before capture, but no LP and no saved
    # basis. This avoids attributing one-off library loading to steady LP work.
    warm_matrix = cp.eye(320, dtype=cp.float64)
    warm_solution = cp.linalg.solve(warm_matrix, warm_matrix)
    (warm_solution@warm_solution).sum().get()
    del warm_solution, warm_matrix
    cp.cuda.runtime.deviceSynchronize()
    cp.cuda.profiler.start()
    cp.cuda.nvtx.RangePush("whole_saved_exchange_lp_probe")
    started = time.perf_counter()
    try:
        result = backend.solve(d["c"], A_eq=matrix[:neq], b_eq=d["rhs"][:neq],
            A_ub=matrix[neq:], b_ub=d["rhs"][neq:], bounds=list(zip(d["lower"], d["upper"])))
        cp.cuda.runtime.deviceSynchronize()
    finally:
        elapsed = time.perf_counter()-started
        cp.cuda.nvtx.RangePop()
        cp.cuda.profiler.stop()
    report = dict(scope="one cold-basis saved-state exchange LP plus tie LP, warmed CUDA libraries; profiled diagnostic, not throughput",
        success=result.success, elapsed_seconds=elapsed, history=backend.history,
        actual_gpu_lp_calls=backend.actual_gpu_lp_calls,
        lp=str(args.lp), lp_sha256=hashlib.sha256(args.lp.read_bytes()).hexdigest(),
        metadata=str(args.metadata), source_hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in [*meta["implementation_sha256"], "src/gpu_exchange_tie_break.py", "scripts/profile_saved_exchange_lp.py"]})
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
