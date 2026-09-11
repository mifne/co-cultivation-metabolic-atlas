"""Resumable, immutable shard collection. No scientific upper cap at 512."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import fcntl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph_training_collection import teacher_source_contract, require_matching_teacher_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','selection','test'],default='train')
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--shard-size',type=int,default=8)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--retry-failed',action='store_true',help='Preserve failed attempt and retry IDENTICAL seeds in a new directory')
    p.add_argument('--cold-teacher',action='store_true')
    p.add_argument('--scipy-teacher',action='store_true')
    args=p.parse_args()
    if min(args.count,args.shard_size,args.workers)<1 or args.count%args.shard_size or not 1<=args.steps<=120:
        p.error('Positive whole shards and 1..120 steps required')
    args.output.mkdir(parents=True,exist_ok=True)
    # One writer per collection, released by the OS on normal exit/crash.
    lock_handle=(args.output/'catalog.lock').open('a')
    try:fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another collector owns this catalog')
    index_path=args.output/'catalog.json'
    config=dict(schema='graph_trajectory_collection_v1',steps=args.steps,shard_size=args.shard_size,
        design_profile='coverage_v1',seed_bases=dict(train=71000000,selection=72000000,test=73000000))
    strategy = ('scipy_cold_serial' if args.scipy_teacher else
                'cold_rebuild_each_lp_batch' if args.cold_teacher else 'persistent_reoptimization')
    contract = teacher_source_contract(ROOT, frozen_inputs=True,
        design_profile=config['design_profile'], teacher_strategy=strategy)
    if index_path.exists():
        index=json.loads(index_path.read_text())
        if index['configuration']!=config:raise ValueError('Existing immutable collection configuration differs')
        # Check before changing requested_job, retry records or any shard.
        # Legacy collections lacked the factory/reset code hashes and cannot
        # safely acquire new labels under a retrospectively invented contract.
        require_matching_teacher_contract(index.get('teacher_contract'), contract)
    else:index=dict(configuration=config,shards=[],planned_training_sizes=[32,64,128,256,512,1024],
        teacher_contract=contract,
        upper_limit_is_scientifically_established=False,
        evaluation_rule='512 requires a larger-size probe, e.g. 1024; plateau is not inferred from a configured cap')
    index['requested_job']=dict(scope='teacher_data_collection_only',split=args.split,count=args.count,
        steps=args.steps,status='running',training_queued=False,
        coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        index['requested_job']['batches']=[dict(start=first,end=first+args.shard_size,
            status=next((s['status'] for s in index['shards'] if s['split']==args.split and s['start']==first),'pending'))
            for first in range(0,args.count,args.shard_size)]
        tmp=index_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(index,indent=2));tmp.replace(index_path)
    save()
    role=dict(train='training_reference',selection='model_selection_reference',test='development_diagnostic_not_training')[args.split]
    for first in range(0,args.count,args.shard_size):
        relative=f'{args.split}_{first:06d}_{first+args.shard_size:06d}'
        known=next((s for s in index['shards'] if s['split']==args.split and s['start']==first),None)
        destination=args.output/(known['path'] if known else relative)
        if known and known['status']=='completed':
            raw=(destination/'manifest.json').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=known['manifest_sha256']:raise ValueError('Completed shard changed')
            require_matching_teacher_contract(contract, json.loads(raw).get('teacher_contract'))
            continue
        require_matching_teacher_contract(contract, teacher_source_contract(ROOT,
            frozen_inputs=True, design_profile=config['design_profile'], teacher_strategy=strategy))
        if destination.exists():
            if not args.retry_failed or not known or known['status'] not in ('failed','running'):
                raise RuntimeError(f'Incomplete shard preserved for investigation: {destination}')
            # Catalog lock ensures no active writer; never overwrite failed outputs.
            index.setdefault('failed_attempts',[]).append(dict(known))
            attempt=1
            while (args.output/f'{relative}_retry{attempt}').exists():attempt+=1
            relative=f'{relative}_retry{attempt}';destination=args.output/relative
        new_seeds=set(range(config['seed_bases'][args.split]+first,config['seed_bases'][args.split]+first+args.shard_size))
        if any(new_seeds.intersection(s.get('seeds',[])) for s in index['shards'] if s is not known):
            raise ValueError('Seed ranges overlap existing shards/splits')
        row=dict(path=relative,split=args.split,start=first,count=args.shard_size,status='running',workers=args.workers,
            teacher_strategy='scipy_cold_serial' if args.scipy_teacher else 'cold_rebuild_each_lp_batch' if args.cold_teacher else 'persistent_reoptimization')
        if known:index['shards'].remove(known)
        index['shards'].append(row);save();tick=time.perf_counter()
        command=[sys.executable,str(ROOT/'scripts/capture_dfba_lp_trace.py'),'--output',str(destination),
            '--steps',str(args.steps),'--environments',str(args.shard_size),'--workers',str(args.workers),
            '--seed',str(config['seed_bases'][args.split]+first),'--role',role,'--frozen-inputs',
            '--design-profile','coverage_v1','--design-offset',str(first)]
        if args.cold_teacher:command.append('--cold-teacher')
        if args.scipy_teacher:command.append('--scipy-teacher')
        with (args.output/(relative+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row['seconds']=time.perf_counter()-tick
        raw=(destination/'manifest.json').read_bytes() if (destination/'manifest.json').exists() else None
        manifest=json.loads(raw) if raw else {}
        row['status']='completed' if result.returncode==0 and manifest.get('status')=='completed' else 'failed'
        if raw:row['manifest_sha256']=hashlib.sha256(raw).hexdigest()
        if row['status']=='completed':
            try:
                require_matching_teacher_contract(contract, manifest.get('teacher_contract'))
            except ValueError:
                row['status']='failed';index['requested_job']['status']='failed';save();raise
            if index.get('model_fingerprints',manifest['model_fingerprints'])!=manifest['model_fingerprints']:
                row['status']='failed';save();raise ValueError('GEM changed between shards')
            index['model_fingerprints']=manifest['model_fingerprints']
            row.update(seeds=manifest['seeds'],lp_count=len(manifest['entries']))
        if row['status']!='completed':index['requested_job']['status']='failed'
        save();print(json.dumps(row),flush=True)
        if row['status']!='completed':raise RuntimeError(f'Failed shard retained, no replacement-seed selection: {relative}')
    index['requested_job']['status']='completed';save()
    print('Requested collection complete',flush=True)


if __name__=='__main__':main()
