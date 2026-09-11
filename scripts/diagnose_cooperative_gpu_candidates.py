#!/usr/bin/env python3
"""Diagnostic GPU-only ablation against an existing equal-action CPU benchmark.

Never emits a qualification manifest. CPU timings may be from a different
machine load; only numerical endpoints are reused for tuning. Final validation
must use independent seeds and freshly timed serial CPU references.
"""
import argparse
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cooperative_surrogate_e2e import run_case


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None,
                   help="required when reusing a multi-seed validation report")
    p.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 4.0])
    p.add_argument("--candidates", type=int, default=512)
    p.add_argument("--rerank-pool", type=int, default=2048)
    p.add_argument("--match-pha", action="store_true")
    p.add_argument("--blend-candidates", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=400)
    p.add_argument("--multioutput-strength", type=float, default=0.0)
    p.add_argument("--independent-species", action="store_true")
    p.add_argument("--multioutput-strengths", type=float, nargs="+", default=None,
                   help="diagnostic sensitivity sweep; overrides --multioutput-strength")
    args = p.parse_args()
    reference = json.loads(args.reference.read_text())
    steps = reference["steps"]
    if "exact" in reference:
        seed = reference["seed"]
        if args.seed is not None and seed != args.seed:
            raise ValueError("requested seed differs from exact reference")
        exact = reference["exact"]
    else:
        matches = [row for row in reference["runs"] if row["seed"] == args.seed]
        if len(matches) != 1:
            raise ValueError("choose one seed present in the validation reference")
        row, seed = matches[0], args.seed
        exact = {"pha_g_l": row["exact_pha_g_l"],
                 "biomass_g_l": row["exact_biomass_g_l"],
                 "phv_mol_fraction": row["exact_phv_mol_fraction"]}
    profile = reference["consortium_profile"]
    initial_nh4 = reference["initial_nh4_mmol_l"]
    actions = np.random.default_rng(seed).uniform(0.05, 0.95, (steps, 5)).astype(np.float32)
    report = {
        "status": "diagnostic_only", "artifact": str(args.artifact),
        "reference": str(args.reference), "seed": seed, "steps": steps,
        "consortium_profile": profile, "initial_nh4_mmol_l": initial_nh4,
        "note": "CPU time not reused; not a matched-load speed benchmark",
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for multi_strength, strength in product(
        args.multioutput_strengths or [args.multioutput_strength], args.strengths
    ):
        gpu = run_case(
            args.artifact, actions, consortium_profile=profile, initial_nh4=initial_nh4,
            gpu_qp_projection=True, gpu_qp_only=True, gpu_qp_candidates=args.candidates,
            gpu_qp_rerank_pool=args.rerank_pool, gpu_qp_decision_strength=strength,
            gpu_qp_match_pha=args.match_pha,
            gpu_qp_blend_candidates=args.blend_candidates,
            gpu_qp_max_iterations=args.max_iterations,
            gpu_qp_multioutput_strength=multi_strength,
            gpu_qp_independent_species=args.independent_species,
        )
        stats = gpu["solver"]["cooperative"]
        row = {
            "implementation_sha256": gpu.get("implementation_sha256"),
            "strength": strength, "candidates": args.candidates,
            "rerank_pool": args.rerank_pool, "match_pha": args.match_pha,
            "blend_candidates": args.blend_candidates,
            "max_iterations": args.max_iterations,
            "multioutput_strength": multi_strength,
            "independent_species":args.independent_species,
            "seconds": gpu["elapsed_seconds"], "completed_steps": gpu["steps"],
            "pha_relative_error": abs(gpu["pha_g_l"] - exact["pha_g_l"]) / max(exact["pha_g_l"], 1e-9),
            "phv_mol_fraction_error": abs(gpu["phv_mol_fraction"] - exact["phv_mol_fraction"])
                if gpu["phv_mol_fraction"] is not None and exact["phv_mol_fraction"] is not None else None,
            "biomass_max_error_g_l": max(abs(gpu["biomass_g_l"][k] - v)
                for k, v in exact["biomass_g_l"].items()),
            "gpu_accepts": stats["gpu_qp_accepts"], "gpu_failures": stats["gpu_qp_failures"],
            "cpu_lp_stage_calls": stats["cpu_lp_stage_calls"],
            "gpu_target_attempts": stats["gpu_qp_target_attempts"],
            "gpu_target_successes": stats["gpu_qp_target_successes"],
            "gpu_multioutput_attempts": stats["gpu_multioutput_attempts"],
            "gpu_multioutput_improvements": stats["gpu_multioutput_improvements"],
            "gpu_pha_g_l": gpu["pha_g_l"], "exact_pha_g_l": exact["pha_g_l"],
            "gpu_biomass_g_l": gpu["biomass_g_l"], "exact_biomass_g_l": exact["biomass_g_l"],
            "gpu_phv_mol_fraction": gpu["phv_mol_fraction"],
            "exact_phv_mol_fraction": exact["phv_mol_fraction"],
            "gpu_peak_mib": gpu["gpu_peak_memory_allocated_mib"],
        }
        report["runs"].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
