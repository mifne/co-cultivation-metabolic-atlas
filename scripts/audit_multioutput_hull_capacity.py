#!/usr/bin/env python3
"""OFFLINE ONLY: distinguish a finite candidate-hull limit from QP convergence.

This CPU LP is a diagnostic oracle, not a runtime fallback or training source.
It minimizes worst normalized growth/PHB/PHV deviation at identical states.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment


def independent_operators(raw, base, context, lower, upper, supply, indices):
    """One simplex per species; preserve its Sv=0 and couple shared resources."""
    count, s, m = len(raw), len(base.species_offsets), len(supply)
    blocks, rhs_blocks, shared_blocks = [], [], []
    decision = np.zeros((len(indices), s*count))
    terms = zip(base.term_metabolite.cpu().numpy(), base.term_species.cpu().numpy(),
                base.term_reaction.cpu().numpy(), base.term_coefficient.cpu().numpy())
    terms = list(terms)
    for species, (start, end) in enumerate(base.species_offsets):
        local = raw[:, start:end]
        hi = local.max(axis=0) > upper[start:end]+base.bound_tolerance
        lo = local.min(axis=0) < lower[start:end]-base.bound_tolerance
        blocks.append(np.r_[local[:, hi].T, -local[:, lo].T])
        rhs_blocks.append(np.r_[upper[start:end][hi]+base.bound_tolerance,
                               -lower[start:end][lo]+base.bound_tolerance])
        shared = np.zeros((m, count))
        for metabolite, owner, reaction, coefficient in terms:
            if owner == species:
                shared[metabolite] += context[species]*coefficient*raw[:, reaction]
        shared_blocks.append(shared)
        for j, reaction in enumerate(indices):
            if start <= reaction < end:
                decision[j, species*count:(species+1)*count] = raw[:, reaction]
    active_shared = sum(block.max(axis=1) for block in shared_blocks) > supply+base.shared_tolerance
    matrix = sparse.vstack((sparse.block_diag(blocks, format="csr"),
        sparse.csr_matrix(np.concatenate(shared_blocks, axis=1)[active_shared])), format="csr")
    rhs = np.r_[np.concatenate(rhs_blocks), supply[active_shared]+base.shared_tolerance]
    equality = sparse.block_diag([np.ones((1, count))]*s, format="csr")
    return matrix, rhs, decision, equality


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-steps", type=int, nargs="+", default=[0, 30, 60, 90, 119])
    p.add_argument("--candidates", type=int, default=2048)
    p.add_argument("--independent-species", action="store_true")
    args = p.parse_args()
    data = np.load(args.queries)
    env = make_environment(args.artifact, 1, consortium_profile="pf-helper3",
        initial_nh4=.05, gpu_qp_projection=True, gpu_qp_only=True,
        gpu_qp_candidates=args.candidates, gpu_qp_rerank_pool=4096,
        gpu_qp_decision_strength=0, gpu_qp_max_iterations=2000)
    env.reset(seed=20260901)
    env.step(np.full(5, .5, dtype=np.float32))
    solver = env.simulator._cooperative_solver
    head, base = solver._surrogate_dictionary, solver._gpu_qp_projector
    metadata = json.loads(str(data["metadata_json"]))
    for key in ("reaction_ids", "model_fingerprints"):
        if metadata[key] != head.metadata[key]:
            raise ValueError(f"query {key} mismatch")
    indices = [int(i) for i, _ in solver._growth_terms.values()] + [
        i for i, name in enumerate(metadata["reaction_ids"]) if name in {"EX_pha_c", "EX_phv_c"}]
    s, m, n = len(solver.species_names), len(base.shared_metabolite_ids), solver.n_fluxes
    rows = []
    for step in args.sample_steps:
        i = int(np.flatnonzero(data["steps"] == step)[0])
        context, exact = data["contexts"][i], data["exact_fluxes"][i]
        raw = head.rank_device(context, top_k=args.candidates).fluxes[0].cpu().numpy()
        composed = []
        for choices in product(range(min(base.block_composition_candidates, len(raw))), repeat=s):
            if len(set(choices)) == 1:
                continue
            row = np.zeros(n, dtype=np.float32)
            for choice, (start, end) in zip(choices, base.species_offsets):
                row[start:end] = raw[choice, start:end]
            composed.append(row)
        flux = np.concatenate((raw, composed)).astype(np.float64)
        shared = base._shared_candidate_activity(
            torch.as_tensor(flux[None], device=base.device),
            torch.as_tensor(context[:s][None], device=base.device, dtype=torch.float64))[0].cpu().numpy()
        lower, upper = context[s+m:s+m+n].astype(float), context[s+m+n:s+m+2*n].astype(float)
        supply = context[s:s+m].astype(float)
        hi = flux.max(axis=0) > upper+base.bound_tolerance
        lo = flux.min(axis=0) < lower-base.bound_tolerance
        sh = shared.max(axis=0) > supply+base.shared_tolerance
        matrix = np.concatenate((flux[:, hi].T, -flux[:, lo].T, shared[:, sh].T))
        rhs = np.r_[upper[hi]+base.bound_tolerance, -lower[lo]+base.bound_tolerance,
                    supply[sh]+base.shared_tolerance]
        # One unit is 1% of the exact rate, with an absolute 1e-4 floor.
        # These are rate diagnostics, not the terminal-trajectory acceptance rule.
        scale = np.maximum(np.abs(exact[indices])*.01, 1e-4)
        target_matrix = flux[:, indices].T
        equality = np.ones((1, len(flux)))
        if args.independent_species:
            matrix, rhs, target_matrix, equality = independent_operators(
                raw.astype(float), base, context, lower, upper, supply, indices)
        variable_count = target_matrix.shape[1]
        a_ub = sparse.vstack((sparse.hstack((sparse.csr_matrix(matrix), sparse.csr_matrix((len(rhs), 1)))),
            sparse.csr_matrix(np.c_[target_matrix, -scale]),
            sparse.csr_matrix(np.c_[-target_matrix, -scale])), format="csr")
        b_ub = np.r_[rhs, exact[indices], -exact[indices]]
        objective = np.r_[np.zeros(variable_count), 1.0]
        started = time.perf_counter()
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub,
            A_eq=sparse.hstack((sparse.csr_matrix(equality), sparse.csr_matrix((equality.shape[0], 1)))),
            b_eq=np.ones(equality.shape[0]),
            bounds=(0, None), method="highs-ds")
        row = {"step":step, "success":bool(result.success), "message":result.message,
               "offline_cpu_seconds":time.perf_counter()-started, "total_candidates":len(flux),
               "independent_species":args.independent_species, "mixing_variables":variable_count,
               "reaction_indices":indices, "exact_rates":exact[indices].tolist(),
               "normalization_scale":scale.tolist()}
        if result.success:
            row.update({"best_worst_scaled_rate_error":float(result.fun),
                        "optimal_rates":(target_matrix @ result.x[:-1]).tolist(),
                        "max_inequality_violation":float(np.maximum(a_ub @ result.x-b_ub, 0).max())})
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"status":"offline_diagnostic_not_online_or_training",
            "artifact":str(args.artifact), "queries":str(args.queries), "candidates":args.candidates,
            "independent_species":args.independent_species,
            "rows":rows}, indent=2))
    assert solver.cpu_lp_stage_calls == 0


if __name__ == "__main__":
    main()
