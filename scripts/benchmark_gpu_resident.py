"""Measure GPU-resident sparse dFBA kernels and project a target GPU.

This is deliberately a kernel benchmark, not an LP-solver benchmark.  The
stoichiometric CSR matrices and state vectors remain on CUDA for the timed
section.  It gives an empirical lower-level reference on the current GPU and
uses the target card's published memory bandwidth for a conservative scaling
estimate.  Exact LP speed must still be measured with cuOpt on the target.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import load_sbml_models, select_consortium_models
from cobra.util.array import create_stoichiometric_matrix


def _csr_to_cuda(matrix):
    csr = matrix.tocsr()
    crow = torch.as_tensor(csr.indptr, dtype=torch.int64, device="cuda")
    col = torch.as_tensor(csr.indices, dtype=torch.int64, device="cuda")
    values = torch.as_tensor(csr.data, dtype=torch.float32, device="cuda")
    return torch.sparse_csr_tensor(
        crow, col, values, size=csr.shape, device="cuda", dtype=torch.float32
    )


def _time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def run(
    iterations: int,
    target_bandwidth_gbps: float,
    target_fp32_tflops: float,
    current_bandwidth_gbps: float,
    current_fp32_tflops: float,
    baseline_step_seconds: float,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")

    models = select_consortium_models(load_sbml_models(Path("models/sbml/final_consortium")))
    matrices = {}
    matrix_stats = {}
    for name, model in models.items():
        matrix = create_stoichiometric_matrix(model, array_type="lil", dtype=np.float32).tocsr()
        matrices[name] = _csr_to_cuda(matrix)
        matrix_stats[name] = {
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "nnz": int(matrix.nnz),
        }

    # A GPU-resident state vector mimics the repeated S @ v and elementwise
    # state/bound updates in dFBA without timing Python or PCIe transfers.
    vectors = {
        name: torch.randn(matrix.shape[1], device="cuda", dtype=torch.float32)
        for name, matrix in matrices.items()
    }
    outputs = {}
    per_model = {}
    for name, matrix in matrices.items():
        v = vectors[name]
        output = torch.empty(matrix.shape[0], device="cuda", dtype=torch.float32)

        def sparse_step():
            output.copy_(torch.sparse.mm(matrix, v.unsqueeze(1)).squeeze(1))

        seconds = _time_cuda(sparse_step, warmup=20, iterations=iterations)
        outputs[name] = output
        bytes_read = matrix_stats[name]["nnz"] * (4 + 4) + matrix_stats[name]["columns"] * 4
        per_model[name] = {
            **matrix_stats[name],
            "seconds_per_sparse_matvec": seconds,
            "effective_gbps": bytes_read / seconds / 1e9,
            "matvecs_per_second": 1.0 / seconds,
        }

    # Device-to-device copy bandwidth is a useful empirical ceiling for the
    # bound/state update portion.  It is not a cuOpt solve result.
    copy_n = 64 * 1024 * 1024
    src = torch.empty(copy_n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    copy_seconds = _time_cuda(lambda: dst.copy_(src), warmup=10, iterations=50)
    copy_bandwidth = (copy_n * 4) / copy_seconds / 1e9
    # Use published architectural ceilings for the forecast, rather than the
    # small copy-kernel measurement above.  Sparse LP kernels rarely saturate
    # either ceiling, so the lower of bandwidth and FP32 ratios is the
    # deliberately conservative target-GPU factor.
    bandwidth_ratio = target_bandwidth_gbps / current_bandwidth_gbps
    fp32_ratio = target_fp32_tflops / current_fp32_tflops
    target_ratio = min(bandwidth_ratio, fp32_ratio)

    # Amdahl projection for an end-to-end simulator.  `target_ratio` is only
    # the GPU-resident kernel scaling factor; it does not assume the entire
    # Python environment becomes GPU code.
    projection = []
    for lp_fraction in (0.50, 0.75, 0.90, 0.95):
        end_to_end_speedup = 1.0 / ((1.0 - lp_fraction) + lp_fraction / target_ratio)
        projected_step_seconds = baseline_step_seconds / end_to_end_speedup
        projection.append({
            "lp_fraction": lp_fraction,
            "end_to_end_speedup": end_to_end_speedup,
            "projected_seconds_per_step": projected_step_seconds,
            "projected_steps_per_second": 1.0 / projected_step_seconds,
        })

    return {
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "iterations": iterations,
        "gpu_resident": True,
        "copy_bandwidth_gbps": copy_bandwidth,
        "target_bandwidth_gbps": target_bandwidth_gbps,
        "current_bandwidth_reference_gbps": current_bandwidth_gbps,
        "bandwidth_ratio": bandwidth_ratio,
        "current_fp32_reference_tflops": current_fp32_tflops,
        "target_fp32_tflops": target_fp32_tflops,
        "fp32_ratio": fp32_ratio,
        "conservative_target_kernel_ratio": target_ratio,
        "baseline_step_seconds": baseline_step_seconds,
        "per_model": per_model,
        "amdahl_projection": projection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--target-bandwidth-gbps", type=float, default=432.0)
    parser.add_argument("--target-fp32-tflops", type=float, default=24.0)
    parser.add_argument("--current-bandwidth-gbps", type=float, default=256.0)
    parser.add_argument("--current-fp32-tflops", type=float, default=14.6)
    parser.add_argument("--baseline-step-seconds", type=float, default=0.9865)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(
        args.iterations,
        args.target_bandwidth_gbps,
        args.target_fp32_tflops,
        args.current_bandwidth_gbps,
        args.current_fp32_tflops,
        args.baseline_step_seconds,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
