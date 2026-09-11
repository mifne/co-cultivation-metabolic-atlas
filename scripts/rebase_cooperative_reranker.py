#!/usr/bin/env python3
"""Retain a neural predictor while replacing its exact-flux candidate dictionary.

Always writes a new, unqualified artifact. A successful old validation is never
inherited, because changed anchors can change rollout dynamics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def rebase(source: Path, dictionary: Path, output: Path) -> dict:
    if output.resolve() in {source.resolve(), dictionary.resolve()}:
        raise ValueError("rebase must not overwrite either input")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    anchors = torch.load(dictionary, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    for key in ("species", "reaction_ids", "context_layout", "model_fingerprints",
                "cooperative_optimize_live_objectives"):
        if metadata.get(key) != anchors["metadata"].get(key):
            raise ValueError(f"rebase dictionary mismatch: {key}")
    if not np.array_equal(payload["feature_indices"], anchors["feature_indices"]):
        raise ValueError("rebase feature indices mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.update({
        "base_dictionary": str(dictionary.resolve()),
        "rebased_from": str(source.resolve()),
        "rebase_candidate_count": len(anchors["fluxes"]),
    })
    torch.save({**payload, "metadata": metadata}, output)
    report = {
        "status": "experimental_not_qualified", "artifact": str(output),
        "base_dictionary": str(dictionary), "rebased_from": str(source),
        "candidate_count": len(anchors["fluxes"]),
        "reason": "new candidate dictionary requires independent rollout validation",
    }
    output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dictionary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(rebase(args.source, args.dictionary, args.output), indent=2))
