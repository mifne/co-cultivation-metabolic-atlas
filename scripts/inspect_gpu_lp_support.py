"""Inspect a training dictionary's active reaction support, not held-out states."""
from pathlib import Path
import json
import torch
import numpy as np

root = Path(__file__).resolve().parents[1]
path = root/"models/cooperative_surrogate/pf_phbv_boundary_34344_20260903.pt"
data = torch.load(path, map_location="cpu", weights_only=False)
print("keys", list(data), flush=True)
for key, value in data.items():
    if hasattr(value, "shape"):
        print(key, value.shape, flush=True)
fluxes = np.asarray(data["fluxes"])
support = np.any(np.abs(fluxes) > 1e-8, axis=0)
out = root/"models/cooperative_surrogate/pf_training_reaction_support_20260903.npz"
np.savez_compressed(out, support=support, metadata_json=np.asarray(json.dumps(data["metadata"])))
print("active reactions", int(support.sum()), "of", len(support), flush=True)
