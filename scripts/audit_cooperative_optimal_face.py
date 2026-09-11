#!/usr/bin/env python3
"""Offline optimal-face audit. Never supplies labels to online GPU execution."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.augment_cooperative_surrogate_dagger import label_exact_context_chunk
from scripts.benchmark_cooperative_surrogate_e2e import load_consortium_profile
from src.community_solver import CooperativeCommunityFbaSolver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT/"results/pf_multi_same_state_baseline.npz")
    p.add_argument("--steps", type=int, nargs="+", default=[0, 60, 90, 119])
    p.add_argument("--output", type=Path, default=ROOT/"results/pf_optimal_face_audit.json")
    args = p.parse_args()
    data = np.load(args.input)
    metadata = json.loads(str(data["metadata_json"]))
    layout = CooperativeCommunityFbaSolver(load_consortium_profile("pf-helper3"))
    indices = sorted(set([term[0] for term in layout._growth_terms.values()] +
        [i for i, name in enumerate(metadata["reaction_ids"]) if name in {"EX_pha_c", "EX_phv_c"}]))
    records = []
    for step in args.steps:
        context = data["contexts"][np.flatnonzero(data["steps"] == step)[0]:][:1]
        calls = []
        def record(c, **kw):
            result = linprog(c, **kw)
            calls.append((np.asarray(c), kw, result))
            return result
        with patch("src.community_solver.linprog", record):
            label_exact_context_chunk(context, "pf-helper3")
        c, kw, result = calls[-1]
        if not result.success or len(calls) != 3:
            raise RuntimeError("expected three successful CPU reference stages")
        tolerance = 1e-8*max(1.0, abs(float(result.fun)))
        face = dict(kw, A_ub=vstack((kw["A_ub"], csr_matrix(c[None])), format="csr"),
                    b_ub=np.r_[kw["b_ub"], result.fun+tolerance])
        ipm = linprog(c, **dict(kw, method="highs-ipm"))
        ranges = []
        for index in indices:
            objective = np.zeros(len(c)); objective[index] = 1
            lo = linprog(objective, **face)
            hi = linprog(-objective, **face)
            ranges.append({"reaction":metadata["reaction_ids"][index], "index":index,
                "cpu_ds":float(result.x[index]),
                "cpu_ipm":float(ipm.x[index]) if ipm.success else None,
                "min":float(lo.x[index]) if lo.success else None,
                "max":float(hi.x[index]) if hi.success else None})
        transfer_ranges = []
        biomass = context[0, :len(layout.species_names)]
        for metabolite in ("nh4_e", "o2_e", "glc__D_e", "lac__L_e", "C30_oligo_e", "ODTD_e"):
            vector = np.zeros(len(c))
            for species, index, coefficient, _ in layout._exchange_terms.get(metabolite, ()):
                vector[index] += biomass[layout.species_names.index(species)]*coefficient
            lo = linprog(vector, **face); hi = linprog(-vector, **face)
            transfer_ranges.append({"metabolite":metabolite,
                "cpu_ds":float(vector @ result.x),
                "min":float(vector @ lo.x) if lo.success else None,
                "max":float(vector @ hi.x) if hi.success else None})
        row = {"step":step, "parsimony_optimum":float(result.fun),
               "face_absolute_tolerance":tolerance, "ranges":ranges,
               "net_transfer_ranges_mmol_l_h":transfer_ranges}
        records.append(row)
        args.output.write_text(json.dumps({"status":"offline_diagnostic_not_training",
            "input":str(args.input), "records":records}, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
