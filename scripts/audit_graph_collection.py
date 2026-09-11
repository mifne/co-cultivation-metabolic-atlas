"""Read-only complete-shard checksum audit before resuming after host shutdown."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.graph_training_collection import GraphTrainingCollection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    started=time.perf_counter();data=GraphTrainingCollection(a.collection)
    entries=[(e['folder'],r) for episodes in data.episodes.values() for e in episodes
             for stage in e['entries'].values() for r in stage]
    def verify(item):
        folder,e=item;path=(folder/e['filename']).resolve()
        if not path.is_relative_to(folder):raise ValueError('Payload path escaped shard')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError(f'Payload checksum mismatch: {path}')
        return len(raw)
    checked=0;size=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for size_one in pool.map(verify,entries):
            checked+=1;size+=size_one
            if checked%10000==0:print(f'Checked {checked}/{len(entries)} LP payloads',flush=True)
    report=dict(status='passed',scope='Manifest/role/seed and every completed LP payload checksum; not new solver certification',
        collection=str(a.collection),catalog_sha256=data.sha256,lp_payloads=checked,bytes=size,
        trajectories={k:len(v) for k,v in data.episodes.items()},
        incomplete_shards=data.incomplete,seconds=time.perf_counter()-started)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
