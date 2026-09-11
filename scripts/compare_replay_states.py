"""Trace all extracellular pools in two recorded trajectories, read-only."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cpu",type=Path,required=True)
    p.add_argument("--gpu",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cpu=json.loads(args.cpu.read_text()); gpu=json.loads(args.gpu.read_text())
    records=[]
    for a,b in zip(cpu["rows"],gpu["rows"]):
        if a["step"]!=b["step"]: raise ValueError("Step mismatch")
        pools=[]
        for name in sorted(set(a["after"]["metabolites"])|set(b["after"]["metabolites"])):
            x=a["after"]["metabolites"].get(name,0.)
            y=b["after"]["metabolites"].get(name,0.)
            pools.append(dict(metabolite=name,cpu=x,gpu=y,absolute_difference=abs(x-y)))
        pools.sort(key=lambda x:x["absolute_difference"],reverse=True)
        records.append(dict(step=a["step"],largest_pool_differences=pools[:15],
            sulfate_cpu=a["after"]["metabolites"].get("so4_e",0.),
            sulfate_gpu=b["after"]["metabolites"].get("so4_e",0.),
            original_gpu_replay_pha_difference=b.get("original_gpu_pha_difference"),
            original_gpu_replay_biomass_difference=b.get("original_gpu_biomass_difference")))
    result=dict(scope="post-test offline diagnostic",cpu=str(args.cpu),gpu=str(args.gpu),records=records)
    args.output.write_text(json.dumps(result,indent=2))
    for row in records:
        print(json.dumps(dict(step=row["step"],largest=row["largest_pool_differences"][:3],
            replay_biomass_error=row["original_gpu_replay_biomass_difference"])),flush=True)


if __name__=="__main__": main()
