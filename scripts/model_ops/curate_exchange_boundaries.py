#!/usr/bin/env python3
"""Remove compulsory exchange fluxes from production GEM defaults.

Measured or fitted exchange ranges must be applied as scenario constraints,
not stored as mandatory uptake/secretion in the reusable base GEM.  This
curation only widens a bound enough to admit zero and otherwise preserves the
recorded direction and magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cobra

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "sbml" / "final_consortium"


def normalize_model(model: cobra.Model):
    changes = []
    for reaction in model.exchanges:
        old = tuple(map(float, reaction.bounds))
        new = (min(old[0], 0.0), max(old[1], 0.0))
        if new != old:
            reaction.bounds = new
            changes.append({"reaction": reaction.id, "old": old, "new": new})
    # This synthetic glycolipoprotein reaction is a hypothesis, not a measured
    # non-growth-associated maintenance demand. Requiring its secretion makes
    # the LP model infeasible in media that lack the pseudo-precursors.
    if "R_GLYCOLIPOPROTEIN_SYN_SEC" in model.reactions:
        reaction = model.reactions.get_by_id("R_GLYCOLIPOPROTEIN_SYN_SEC")
        old = tuple(map(float, reaction.bounds))
        if old[0] > 0.0:
            reaction.lower_bound = 0.0
            changes.append(
                {
                    "reaction": reaction.id,
                    "old": old,
                    "new": tuple(map(float, reaction.bounds)),
                }
            )
    return changes


def curate_directory(model_dir: Path = MODEL_DIR):
    report = {}
    for path in sorted(model_dir.glob("*.xml")):
        model = cobra.io.read_sbml_model(path)
        changes = normalize_model(model)
        if changes:
            cobra.io.write_sbml_model(model, path)
        report[path.name] = changes
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(curate_directory(args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
