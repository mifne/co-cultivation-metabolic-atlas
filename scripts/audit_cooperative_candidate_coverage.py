#!/usr/bin/env python3
"""Audit the finite-anchor feasibility/accuracy ceiling at a fixed initial query.

The CPU answer is used ONLY for diagnosis, never as an online GPU prediction.
Checks every dictionary anchor, including anchors omitted by nearest-neighbour
retrieval. Convex mixtures of individually infeasible anchors are not excluded
by this audit, so a poor feasible-anchor range is not a proof of hull infeasibility.
"""
import argparse
import json
from pathlib import Path
import sys
from itertools import product

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--consortium", default="pf-helper3")
    p.add_argument("--initial-nh4", type=float, default=0.05)
    args = p.parse_args()
    action = np.random.default_rng(args.seed).uniform(0.05, 0.95, 5).astype(np.float32)
    snapshots, solvers = [], []
    for artifact in (None, args.artifact):
        env = make_environment(artifact, 1, consortium_profile=args.consortium,
            initial_nh4=args.initial_nh4, gpu_qp_projection=artifact is not None,
            gpu_qp_only=artifact is not None, gpu_qp_candidates=512,
            gpu_qp_rerank_pool=2048, gpu_qp_decision_strength=0.0)
        env.simulator.cooperative_capture_training_snapshot = True
        env.reset(seed=args.seed)
        env.step(action)
        solvers.append(env.simulator._cooperative_solver)
        snapshots.append(solvers[-1].last_training_snapshot)
    exact, gpu = snapshots
    np.testing.assert_array_equal(exact["context"], gpu["context"])
    solver = solvers[-1]
    reranker, projector = solver._surrogate_dictionary, solver._gpu_qp_projector
    dictionary = reranker.dictionary
    n, species = solver.n_fluxes, len(solver.species_names)
    m = len(projector.shared_metabolite_ids)
    context = exact["context"]
    device = dictionary.device
    biomass = torch.as_tensor(context[:species], device=device)[None]
    supply = torch.as_tensor(context[species:species+m], device=device)[None]
    lower = torch.as_tensor(context[species+m:species+m+n], device=device)
    upper = torch.as_tensor(context[species+m+n:species+m+2*n], device=device)
    reaction_ids = reranker.metadata["reaction_ids"]
    pha_indices = [i for i, name in enumerate(reaction_ids) if name in ("EX_pha_c", "EX_phv_c")]
    mass_weights = torch.tensor([0.10012 if reaction_ids[i] == "EX_phv_c" else 0.08609
                                 for i in pha_indices], device=device)
    exact_mass_rate = float(np.dot(exact["fluxes"][pha_indices], mass_weights.cpu().numpy()))
    query_rank = reranker.rank_device(context, top_k=512)
    top_indices = set(query_rank.indices[0].cpu().tolist())
    feasible_ids, feasible_rates = [], []
    for start in range(0, dictionary.candidate_count, 256):
        flux = dictionary.fluxes[start:start+256]
        violation = torch.maximum(torch.relu(lower - flux), torch.relu(flux - upper)).amax(dim=1)
        shared = projector._shared_candidate_activity(flux[None], biomass)[0]
        shared_violation = torch.relu(shared-supply).amax(dim=1)
        feasible = (violation <= projector.bound_tolerance) & (shared_violation <= projector.shared_tolerance)
        ids = torch.nonzero(feasible).flatten()
        rates = (flux[ids][:, pha_indices] * mass_weights).sum(dim=1)
        feasible_ids.extend((ids + start).cpu().tolist())
        feasible_rates.extend(rates.cpu().tolist())
    rate = np.asarray(feasible_rates)
    predicted = query_rank.predicted_decision_fluxes[0]
    decision_indices = reranker.decision_indices.cpu().tolist()
    prediction_mass = float(sum(predicted[decision_indices.index(i)] * w
        for i, w in zip(pha_indices, mass_weights)))
    # Offline oracle separates insufficient PDHG convergence from insufficient
    # candidate coverage. It is never used by the runtime predictor.
    from scipy.optimize import linprog
    candidates = query_rank.fluxes[0].cpu().numpy().astype(np.float64)
    composed = []
    for choices in product(range(4), repeat=len(solver._offsets)):
        if len(set(choices)) == 1:
            continue
        composite = np.zeros(n)
        for choice, (start, end) in zip(choices, solver._offsets.values()):
            composite[start:end] = candidates[choice, start:end]
        composed.append(composite)
    candidates = np.concatenate((candidates, np.asarray(composed)))
    shared_matrix = projector._shared_candidate_activity(
        torch.as_tensor(candidates, dtype=torch.float32, device=device)[None], biomass)[0].cpu().numpy().astype(np.float64)
    lb, ub, rhs = lower.cpu().numpy(), upper.cpu().numpy(), supply[0].cpu().numpy()
    active_lower = candidates.min(axis=0) < lb
    active_upper = candidates.max(axis=0) > ub
    active_shared = shared_matrix.max(axis=0) > rhs
    matrix = np.vstack((-candidates[:, active_lower].T, candidates[:, active_upper].T,
                        shared_matrix[:, active_shared].T))
    # Use exactly the runtime's physical feasibility tolerances, explicitly
    # recorded, to compare the oracle fairly to the float32 GPU projection.
    right = np.r_[-lb[active_lower]+projector.bound_tolerance,
                  ub[active_upper]+projector.bound_tolerance,
                  rhs[active_shared]+projector.shared_tolerance]
    objective = candidates[:, pha_indices] @ mass_weights.cpu().numpy()
    oracle = linprog(-objective, A_ub=matrix, b_ub=right,
                     A_eq=np.ones((1, len(candidates))), b_eq=[1.0], bounds=(0, None), method="highs-ds")
    ray_bounds = {}
    if gpu["fluxes"] is not None:
        base = torch.as_tensor(gpu["fluxes"], dtype=torch.float32, device=device)
        base_shared = projector._shared_candidate_activity(base[None, None], biomass)[0, 0]
        for species_index, (name, (start, end)) in enumerate(solver._offsets.items()):
            direction = torch.zeros_like(base)
            direction[start:end] = base[start:end]
            slack_ratio = torch.where(direction > 1e-9, (upper-base)/direction,
                torch.where(direction < -1e-9, (lower-base)/direction, torch.inf))
            shared_direction = projector._shared_candidate_activity(direction[None, None], biomass)[0, 0]
            shared_ratio = torch.where(shared_direction > 1e-9,
                (supply[0]-base_shared)/shared_direction, torch.inf)
            delta = torch.minimum(slack_ratio.min(), shared_ratio.min()).clamp_min(0.0)
            ray_bounds[name] = float(1.0 + delta)
    report = {
        "artifact": str(args.artifact), "seed": args.seed, "query": "first step, identical CPU/GPU context",
        "candidate_count": dictionary.candidate_count,
        "individually_feasible_anchors": len(feasible_ids),
        "feasible_anchors_in_retrieved_512": len(set(feasible_ids) & top_indices),
        "exact_PHA_mass_rate": exact_mass_rate,
        "neural_predicted_PHA_mass_rate": prediction_mass,
        "offline_retrieved_hull_oracle_success": bool(oracle.success),
        "offline_retrieved_hull_PHA_mass_rate_max": float(-oracle.fun) if oracle.success else None,
        "oracle_bound_tolerance": projector.bound_tolerance,
        "oracle_shared_tolerance": projector.shared_tolerance,
        "maximum_species_ray_scale_from_selected_anchor": ray_bounds,
        "gpu_PHA_mass_rate": float(np.dot(gpu["fluxes"][pha_indices], mass_weights.cpu().numpy()))
            if gpu["fluxes"] is not None else None,
        "feasible_PHA_mass_rate_min": float(rate.min()) if len(rate) else None,
        "feasible_PHA_mass_rate_max": float(rate.max()) if len(rate) else None,
        "best_single_anchor_PHA_relative_error": float(np.min(np.abs(rate-exact_mass_rate))/max(abs(exact_mass_rate), 1e-9))
            if len(rate) else None,
        "note": "mass rate is sum of species-specific PHB/PHV repeat-unit mass fluxes, before biomass integration; no online CPU label used",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
