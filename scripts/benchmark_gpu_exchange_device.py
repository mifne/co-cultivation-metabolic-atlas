"""Sequential saved-state comparison; not dFBA trajectory qualification.

Reference CPU outputs and LPs were saved previously; no CPU optimizer is
called during GPU timing. Includes the unchanged fourth exchange tie LP.
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
from src.gpu_lexicographic_lp import PresolvedCuOptBackend
from src.gpu_device_bounded_simplex import GpuDeviceBoundedSimplex


def save(path, report):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 17, 115])
    parser.add_argument("--backends", nargs="+", choices=["reference", "device"], default=["reference", "device"])
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    capture = ROOT/"results/pf_seed20286001_cpu_diagnostic"
    meta = json.loads((capture/"metadata.json").read_text())
    for path, digest in meta["implementation_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Reference source changed: {path}")
    files = ["src/gpu_bounded_simplex.py", "src/gpu_device_bounded_simplex.py",
             "src/gpu_simplex_pivot_batch.py", "src/gpu_lexicographic_lp.py",
             "src/gpu_exchange_tie_break.py", "scripts/benchmark_gpu_exchange_device.py"]
    report = dict(status="running", scope="saved-state LPs; NOT full-trajectory accuracy or throughput",
        records=[], source_hashes={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})
    save(args.output, report)
    import cupy as cp
    warm = cp.eye(320, dtype=cp.float64)
    (cp.linalg.solve(warm, warm) @ warm).sum().get()
    for backend_name in args.backends:
        inner = PresolvedCuOptBackend(method="tableau", tolerance=1e-8, time_limit=600)
        if backend_name == "device":
            inner.engine = GpuDeviceBoundedSimplex(tolerance=1e-8, time_limit=600)
            inner.validation_engine = inner.engine
        layout = SimpleNamespace(n_fluxes=len(meta["reaction_ids"]), _exchange_terms=meta["exchange_terms"])
        backend = GpuExchangeTieBreakBackend(layout, inner=inner)
        for step in args.steps:
            for stage in args.stages:
                lp = capture/f"step_{step:03d}_stage_{stage}.npz"
                d = np.load(lp)
                a = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
                neq = int(d["neq"])
                cp.cuda.get_current_stream().synchronize()
                started = time.perf_counter()
                result = backend.solve(d["c"], A_eq=a[:neq], b_eq=d["rhs"][:neq],
                    A_ub=a[neq:], b_ub=d["rhs"][neq:], bounds=list(zip(d["lower"], d["upper"])))
                cp.cuda.get_current_stream().synchronize()
                seconds = time.perf_counter()-started
                # Inspect saved schema rather than assume the solver's chosen
                # internal flux vector is unique. Report objective/residuals.
                cpu_objective = float(d["cpu_objective"]) if "cpu_objective" in d else None
                row = dict(backend=backend_name, step=step, stage=stage, success=result.success,
                    seconds=seconds, objective=result.fun, saved_cpu_objective=cpu_objective,
                    lp_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(), history=backend.history[-1])
                if result.success:
                    row["values"] = result.x.tolist()
                report["records"].append(row)
                save(args.output, report)
                print(json.dumps({k: v for k, v in row.items() if k not in {"history", "values"}}), flush=True)
                if not result.success:
                    report["status"] = "failed"
                    save(args.output, report)
                    return 1
    report["status"] = "completed"
    save(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
