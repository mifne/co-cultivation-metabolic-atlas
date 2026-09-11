"""Map an observed saved-LP difference to species and adjacent reactions.

This is flux attribution, not proof of a causal metabolic pathway or validation
of reaction biology. It never changes bounds, reactions, genes or reference data.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cooperative_surrogate_e2e import make_environment
from src.fba_surrogate import model_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta = json.loads(args.metadata.read_text())
    cpu_problem, gpu_problem = np.load(args.cpu), np.load(args.gpu)
    x = cpu_problem["cpu_x"]
    y = gpu_problem["gpu_x"]
    if x.shape != y.shape:
        raise ValueError("Different column layouts")
    exchanges = []
    for name in ("val__L_e", "leu__L_e", "for_e", "h2_e"):
        for species, index, stoich, reaction in meta["exchange_terms"][name]:
            exchanges.append(dict(metabolite=name, species=species, reaction=reaction,
                cpu=float(x[index]), gpu=float(y[index]), delta=float(y[index]-x[index])))
    env = make_environment(None, 120, consortium_profile="pf-helper3", initial_nh4=.05)
    env.reset(seed=meta["seed"])
    if {k: model_fingerprint(m) for k, m in env.simulator.models.items()} != meta["model_fingerprints"]:
        raise ValueError("GEM changed")
    lookup = {(s, r): i for i, (s, r) in enumerate(zip(meta["reaction_species"], meta["reaction_ids"]))}
    adjacent = []
    for species, model in env.simulator.models.items():
        chosen = set()
        for name in ("val__L_c", "leu__L_c", "val__L_p", "leu__L_p", "val__L_e", "leu__L_e"):
            if name in model.metabolites:
                chosen.update(model.metabolites.get_by_id(name).reactions)
        for reaction in sorted(chosen, key=lambda r: r.id):
            i = lookup[(species, reaction.id)]
            if abs(y[i]-x[i]) > 1e-6:
                displayed = reaction.copy()
                displayed.bounds = (float(cpu_problem["lower"][i]), float(cpu_problem["upper"][i]))
                adjacent.append(dict(species=species, reaction=reaction.id, name=reaction.name,
                    cpu=float(x[i]), gpu=float(y[i]), delta=float(y[i]-x[i]),
                    equation_at_saved_cpu_bounds=displayed.reaction,
                    saved_cpu_bounds=[float(cpu_problem["lower"][i]), float(cpu_problem["upper"][i])],
                    saved_gpu_bounds=[float(gpu_problem["lower"][i]), float(gpu_problem["upper"][i])],
                    gene_reaction_rule=reaction.gene_reaction_rule))
    report = dict(scope="saved step-16 flux attribution; not metabolic pathway proof",
        units="mmol gDW^-1 h^-1, raw reaction sign; positive canonical exchange denotes secretion",
        cpu=str(args.cpu), gpu=str(args.gpu), metadata=str(args.metadata),
        shared_exchange_differences=exchanges, adjacent_reaction_differences=adjacent)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
