#!/usr/bin/env python3
"""Offline CPU relabeling audit of GPU-visited states; never online fallback."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment
from scripts.augment_cooperative_surrogate_dagger import label_exact_context_chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    env = make_environment(args.artifact, args.steps, consortium_profile="pf-helper3",
        initial_nh4=0.05, gpu_qp_projection=True, gpu_qp_only=True,
        gpu_qp_candidates=2048, gpu_qp_rerank_pool=4096,
        gpu_qp_decision_strength=0, gpu_qp_match_pha=True, gpu_qp_max_iterations=2000)
    env.simulator.cooperative_capture_training_snapshot = True
    env.reset(seed=args.seed)
    actions = np.random.default_rng(args.seed).uniform(0.05, 0.95, (args.steps, 5)).astype(np.float32)
    selected = sorted(set(range(min(6, args.steps))) | set(range(0, args.steps, args.stride)) | {args.steps-1})
    contexts, gpu_flux, visited, records = [], [], [], []
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ("src/community_solver.py", "src/gpu_batch_qp.py", "src/cooperative_neural_surrogate.py")}
    for step, action in enumerate(actions):
        nh4_before = env.simulator.state.metabolites.get("nh4_e", 0.0)
        _, _, terminated, truncated, _ = env.step(action)
        solver = env.simulator._cooperative_solver
        if step in selected:
            snapshot = solver.last_training_snapshot
            if snapshot is None:
                raise RuntimeError("missing current query snapshot")
            contexts.append(snapshot["context"])
            gpu_flux.append(np.full(solver.n_fluxes, np.nan) if snapshot["fluxes"] is None else snapshot["fluxes"])
            visited.append(step)
            records.append({"step": step, "nh4_before_feed_mM": nh4_before,
                            "gpu_accepted": snapshot["fluxes"] is not None})
        if (step+1) % 24 == 0:
            print(f"GPU rollout {step+1}/{args.steps}", flush=True)
        if terminated or truncated:
            break
    assert solver.cpu_lp_stage_calls == 0
    contexts, gpu_flux = np.asarray(contexts), np.asarray(gpu_flux)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".queries.npz"), contexts=contexts,
                        gpu_fluxes=gpu_flux, steps=np.asarray(visited))
    chunks = [v for v in np.array_split(contexts, max(1, args.workers)) if len(v)]
    if len(chunks) == 1:
        exact = label_exact_context_chunk(chunks[0], "pf-helper3")
    else:
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp.get_context("spawn")) as executor:
            exact = np.concatenate(list(executor.map(label_exact_context_chunk,
                chunks, ["pf-helper3"]*len(chunks))))
    head = solver._surrogate_dictionary
    neural = head.rank_device(contexts, top_k=1).predicted_decision_fluxes.cpu().numpy()
    indices = head.decision_indices.cpu().numpy()
    scales = head.decision_scale.cpu().numpy()
    metadata = head.metadata
    names = metadata["reaction_ids"]
    owners = {}
    for name, (start, end) in solver._offsets.items():
        owners.update({i: name for i in range(start, end)})
    reaction_errors = []
    for local, index in enumerate(indices):
        reaction_errors.append({
            "species": owners[int(index)], "reaction": names[index],
            "global_index": int(index), "exact_mean_abs": float(np.mean(np.abs(exact[:, index]))),
            "neural_rmse": float(np.sqrt(np.mean((neural[:, local]-exact[:, index])**2))),
            "gpu_rmse": float(np.sqrt(np.nanmean((gpu_flux[:, index]-exact[:, index])**2))),
            "neural_scaled_rmse": float(np.sqrt(np.mean((neural[:, local]-exact[:, index])**2))/scales[local]),
        })
    medium_errors = []
    for metabolite, terms in solver._exchange_terms.items():
        cpu_rate, gpu_rate = np.zeros(len(contexts)), np.zeros(len(contexts))
        for name, index, coefficient, _ in terms:
            weight = contexts[:, solver.species_names.index(name)] * coefficient
            cpu_rate += weight * exact[:, index]
            gpu_rate += weight * gpu_flux[:, index]
        medium_errors.append({"metabolite": metabolite,
            "max_one_step_concentration_error_mM": float(0.2*np.nanmax(np.abs(gpu_rate-cpu_rate))),
            "net_uptake_rmse_mmol_l_h": float(np.sqrt(np.nanmean((gpu_rate-cpu_rate)**2)))})
        if metabolite in {"nh4_e", "o2_e", "glc__D_e", "lac__L_e", "C30_oligo_e", "ODTD_e"}:
            for row, c, g in zip(records, cpu_rate, gpu_rate):
                row[metabolite] = {"cpu_net_uptake": float(c), "gpu_net_uptake": float(g)}
    for name, (index, coefficient) in solver._growth_terms.items():
        for row, c, g in zip(records, exact[:, index], gpu_flux[:, index]):
            row.setdefault("growth_per_h", {})[name] = {"cpu":float(c*coefficient), "gpu":float(g*coefficient)}
    for index, reaction in enumerate(names):
        if reaction in {"EX_pha_c", "EX_phv_c"}:
            local = list(indices).index(index)
            for row, c, g, predicted in zip(records, exact[:, index], gpu_flux[:, index], neural[:, local]):
                row[reaction] = {"cpu":float(c), "gpu":float(g), "neural":float(predicted)}
    report = {"status":"diagnostic_only_not_training_data", "artifact":str(args.artifact),
        "seed":args.seed, "rollout_steps":solver.solve_calls, "sampled_states":len(contexts),
        "online_cpu_lp_stage_calls":solver.cpu_lp_stage_calls, "gpu_accepts":solver.gpu_qp_accepts,
        "implementation_sha256":source_hashes,
        "reaction_errors":sorted(reaction_errors, key=lambda r:r["neural_scaled_rmse"], reverse=True),
        "medium_errors":sorted(medium_errors, key=lambda r:r["max_one_step_concentration_error_mM"], reverse=True),
        "samples":records}
    np.savez_compressed(args.output.with_suffix(".npz"), contexts=contexts, gpu_fluxes=gpu_flux,
        exact_fluxes=exact, neural_decisions=neural, decision_indices=indices,
        steps=np.asarray(visited), metadata_json=np.asarray(json.dumps(metadata)))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"top_medium_errors":report["medium_errors"][:10],
                      "top_neural_errors":report["reaction_errors"][:10]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
