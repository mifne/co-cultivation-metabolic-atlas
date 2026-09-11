#!/usr/bin/env python3
"""Replay held-out states with neural vs offline oracle targets, never train."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment
from src.gpu_multioutput_qp import MultiOutputCooperativeQpProjector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-steps", type=int, nargs="+", default=[0, 5, 30, 60, 90, 119])
    p.add_argument("--candidates", type=int, default=2048)
    p.add_argument("--strength", type=float, default=100)
    p.add_argument("--max-iterations", type=int, default=2000)
    p.add_argument("--independent-species", action="store_true")
    p.add_argument("--affine", action="store_true")
    args = p.parse_args()
    data = np.load(args.queries)
    env = make_environment(args.artifact, 1, consortium_profile="pf-helper3",
        initial_nh4=0.05, gpu_qp_projection=True, gpu_qp_only=True,
        gpu_qp_candidates=args.candidates, gpu_qp_rerank_pool=4096,
        gpu_qp_decision_strength=0, gpu_qp_max_iterations=args.max_iterations)
    env.reset(seed=20260901)
    env.step(np.full(5, .5, dtype=np.float32))  # initialize the GPU-only solver
    solver = env.simulator._cooperative_solver
    head = solver._surrogate_dictionary
    metadata = json.loads(str(data["metadata_json"]))
    for key in ("reaction_ids", "model_fingerprints"):
        if metadata[key] != head.metadata[key]:
            raise ValueError(f"audit {key} mismatch")
    ids = head.decision_indices.cpu().numpy()
    growth = [int(index) for index, _ in solver._growth_terms.values()]
    pha = [i for i, name in enumerate(metadata["reaction_ids"]) if name in {"EX_pha_c", "EX_phv_c"}]
    projector = MultiOutputCooperativeQpProjector(solver._gpu_qp_projector, args.strength,
        independent_species=args.independent_species, signed_weights=args.affine,
        stoichiometry=solver.a_eq_flux if args.affine else None)
    s, m, n = len(solver.species_names), len(solver._gpu_qp_projector.shared_metabolite_ids), solver.n_fluxes
    rows = []
    flux_outputs, steps, modes = [], [], []
    for step in args.sample_steps:
        where = np.flatnonzero(data["steps"] == step)
        if len(where) != 1:
            raise ValueError(f"step {step} absent")
        i = int(where[0])
        context, exact = data["contexts"][i], data["exact_fluxes"][i]
        candidates = head.rank_device(context, top_k=args.candidates)
        for mode, target in (("neural", candidates.predicted_decision_fluxes),
                             ("offline_exact_oracle", exact[ids][None])):
            result = projector.project(candidates.fluxes, context[s+m:s+m+n],
                context[s+m+n:s+m+2*n], context[:s], context[s:s+m],
                reference_weights=candidates.reference_weights, decision_indices=ids,
                decision_targets=target, decision_scales=head.decision_scale,
                decision_weights=head.decision_weight)
            flux = result.fluxes[0]
            mass_residual = np.abs(solver.a_eq_flux @ flux).max()
            row = {"step":step, "mode":mode, "feasible":bool(result.feasible[0]),
                "improved":bool(result.multioutput_improved[0]),
                "weighted_loss_before":float(result.multioutput_loss_before[0]),
                "weighted_loss_after":float(result.multioutput_loss_after[0]),
                "feasible_step_fraction":float(result.multioutput_step_fraction[0]),
                "max_mass_balance_residual":float(mass_residual),
                "weights_l1":float(np.abs(result.weights[0]).sum()),
                "seconds":result.inference_seconds,
                "growth_max_error":float(np.abs(flux[growth]-exact[growth]).max()),
                "pha_fluxes":flux[pha].tolist(), "exact_pha_fluxes":exact[pha].tolist(),
                "decision_scaled_rmse":float(np.sqrt(np.mean(((flux[ids]-exact[ids])/
                    head.decision_scale.cpu().numpy())**2))),
                "decision_scaled_rmse_baseline":float(np.sqrt(np.mean(((data["gpu_fluxes"][i, ids]-exact[ids])/
                    head.decision_scale.cpu().numpy())**2))),
                "growth_fluxes":flux[growth].tolist(), "exact_growth_fluxes":exact[growth].tolist(),
            }
            rows.append(row)
            flux_outputs.append(flux)
            steps.append(step)
            modes.append(mode)
            print(json.dumps(row), flush=True)
    assert solver.cpu_lp_stage_calls == 0
    report = {"status":"diagnostic_only_oracle_never_online", "artifact":str(args.artifact),
        "queries":str(args.queries), "strength":args.strength, "candidates":args.candidates,
        "max_iterations":args.max_iterations,
        "independent_species":args.independent_species,
        "affine":args.affine,
        "implementation_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("src/gpu_multioutput_qp.py", "src/gpu_batch_qp.py", __file__.replace(str(ROOT)+"/", ""))},
        "rows":rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output.with_suffix(".npz"), fluxes=flux_outputs, steps=steps, modes=modes)


if __name__ == "__main__":
    main()
