"""Inspect a saved failure's violated reaction bounds, without optimization."""
import json
from pathlib import Path
import argparse
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("snapshot",type=Path)
args=p.parse_args()
d=np.load(args.snapshot)
meta=json.loads(str(d["metadata_json"]))
x=d["failed_values"]
error=np.maximum(d["lower"]-x,x-d["upper"])
for i in np.argsort(error)[-10:][::-1]:
    print(json.dumps(dict(index=int(i),column=meta["column_ids"][int(i)],
        value=float(x[i]),lower=float(d["lower"][i]),upper=float(d["upper"][i]),violation=float(error[i]))))
