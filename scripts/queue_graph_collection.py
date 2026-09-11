"""Dependency-gated local collection job; waits for the catalog writer lock.

Uses OS blocking locks, not polling. A failed dependency never launches training
collection. Queue state is separate from the actively written capture catalog.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.graph_training_collection import GraphTrainingCollection


def verify_dependency(path,selection_count):
    data=GraphTrainingCollection(path)
    job=data.catalog.get('requested_job',{})
    if job.get('status')!='completed' or job.get('split')!='selection' or job.get('count')!=selection_count:
        raise RuntimeError('Selection dependency has not completed successfully; do not start next job')
    data.select('selection',selection_count)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--selection-count',type=int,default=16)
    p.add_argument('--train-count',type=int,default=512)
    args=p.parse_args()
    if args.train_count<1 or args.train_count%8 or args.selection_count<1:p.error('Positive sizes, train in whole shards of 8')
    args.queue.mkdir(parents=True,exist_ok=False)
    record=dict(status='waiting_for_selection',pid=os.getpid(),selection_count=args.selection_count,
        target_train_count=args.train_count,scope='teacher_collection_only',training_queued=False,
        created_unix=time.time(),collection=str(args.collection.resolve()))
    def save():
        tmp=args.queue/'job.json.tmp';tmp.write_text(json.dumps(record,indent=2,allow_nan=False));tmp.replace(args.queue/'job.json')
    save();print(json.dumps(record),flush=True)
    try:
        # This process sleeps in the kernel until the active collector exits.
        with (args.collection/'catalog.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            data=verify_dependency(args.collection,args.selection_count)
            if data.catalog['configuration']['shard_size']!=8:raise ValueError('Expected 8-trajectory shards')
            record['dependency_catalog_sha256']=data.sha256
            (args.queue/'dependency_catalog.json').write_bytes(data.raw)
        command=[sys.executable,str(ROOT/'scripts/collect_graph_learning_curve.py'),
            '--output',str(args.collection),'--split','train','--count',str(args.train_count),
            '--steps',str(data.catalog['configuration']['steps']),'--shard-size','8',
            '--workers','1','--scipy-teacher']
        record.update(status='running',started_unix=time.time(),command=command,
            coordinator_sha256=hashlib.sha256((ROOT/'scripts/collect_graph_learning_curve.py').read_bytes()).hexdigest());save()
        with (args.queue/'collector.log').open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record['returncode']=result.returncode
        if result.returncode:raise RuntimeError('Collection failed; inspect collector.log and catalog, no blind restart')
        final=GraphTrainingCollection(args.collection)
        final.select('train',args.train_count)
        record.update(status='completed',finished_unix=time.time(),final_catalog_sha256=final.sha256);save()
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),finished_unix=time.time());save()
        raise


if __name__=='__main__':main()
