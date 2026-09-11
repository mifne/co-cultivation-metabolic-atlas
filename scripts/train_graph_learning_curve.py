"""Bounded-memory GNN/GRU learning-curve training, no fixed 16/512 upper cap."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.graph_training_collection import GraphTrainingCollection
from src.graph_temporal_lp import LPGraphBatch
from src.supervised_graph_lp import SupervisedGraphLP,supervised_loss,checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-count',type=int,required=True);p.add_argument('--selection-count',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--epochs',type=int,default=160)
    p.add_argument('--max-updates',type=int,default=0,help='Separate compute-matched ablation; zero means epoch budget')
    p.add_argument('--chunk',type=int,default=8);p.add_argument('--hidden',type=int,default=16)
    p.add_argument('--seed',type=int,default=20294907)
    p.add_argument('--stage',choices=['maxmin','aggregate','exchange'],default='maxmin')
    p.add_argument('--arms',nargs='+',choices=['gnn','gnn_gru'],default=['gnn','gnn_gru'])
    args=p.parse_args()
    if min(args.train_count,args.selection_count,args.batch_size,args.epochs,args.chunk,args.hidden)<1 or args.max_updates<0:
        p.error('Positive dimensions and nonnegative update budget required')
    if args.train_count%args.batch_size or args.selection_count%args.batch_size:p.error('Whole batches required')
    data=GraphTrainingCollection(args.collection)
    train=data.select('train',args.train_count);dev=data.select('selection',args.selection_count)
    args.output.mkdir(parents=True,exist_ok=False)
    record=dict(status='preparing',scope=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        models=[],stage=args.stage,trace_sha256=data.sha256,model_fingerprints=data.catalog['model_fingerprints'],
        train_seeds=[e['seed'] for e in train],development_seeds=[e['seed'] for e in dev],
        training_problem_hashes=[r['problem_sha256'] for e in train for r in e['entries'][args.stage]],
        development_problem_hashes=[r['problem_sha256'] for e in dev for r in e['entries'][args.stage]],
        input_staging='stream at most a few trajectory batches; not the entire dataset in VRAM',
        untouched_test_labels_loaded=False)
    def save():
        tmp=args.output/'manifest.json.tmp';tmp.write_text(json.dumps(record,indent=2,allow_nan=False));tmp.replace(args.output/'manifest.json')
    save();started=time.perf_counter()
    files=['scripts/train_graph_learning_curve.py','src/graph_training_collection.py','src/graph_temporal_lp.py','src/supervised_graph_lp.py','src/lp_trace.py']
    record['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        dest=args.output/'sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/f).read_bytes())
    (args.output/'collection_snapshot.json').write_bytes(data.raw)
    from scripts.benchmark_basis_bank_rollout import environment
    from src.fba_surrogate import model_fingerprint
    env,_=environment(train[0]['seed']);current={k:model_fingerprint(v) for k,v in env.simulator.models.items()};env.close()
    if current!=data.catalog['model_fingerprints']:raise ValueError('Current GEM differs from collection')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    stats=data.statistics(args.train_count,stage=args.stage,batch_size=args.batch_size)
    model_identity=json.dumps(current,sort_keys=True)
    steps=data.catalog['configuration']['steps']
    record.update(training_samples=args.train_count*steps,development_samples=args.selection_count*steps,
        gpu=torch.cuda.get_device_name(0),status='training')
    staging_seconds=0.
    def batches(split,count,order=None):
        nonlocal staging_seconds
        for trajectories in data.batches(split,count,args.batch_size,stage=args.stage,order=order):
            t0=time.perf_counter()
            graphs=[LPGraphBatch.from_problems([t[s][0] for t in trajectories],stage=args.stage,model_identity=model_identity,device='cuda') for s in range(steps)]
            x=torch.tensor(np.stack([[v[1] for v in t] for t in trajectories]),device='cuda')
            y=torch.tensor(np.stack([[v[2] for v in t] for t in trajectories]),device='cuda')
            torch.cuda.synchronize();staging_seconds+=time.perf_counter()-t0
            yield graphs,x,y
            del graphs,x,y
    first_batch=next(batches('train',args.train_count))
    identity=first_batch[0][0].identity;del first_batch
    record['graph_identity']=identity;save()
    for arm in args.arms:
        torch.manual_seed(args.seed)
        model=SupervisedGraphLP(identity,*stats,hidden=args.hidden,temporal=arm=='gnn_gru').cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        best=float('inf');history=[];updates=0;tick=time.perf_counter();staging_start=staging_seconds
        for epoch in range(1,args.epochs+1):
            model.train();train_total=0.;train_items=0
            order=np.random.default_rng(args.seed+epoch).permutation(np.arange(0,args.train_count,args.batch_size)).tolist()
            for graphs,x,y in batches('train',args.train_count,order):
                state=None
                for first in range(0,steps,args.chunk):
                    optimizer.zero_grad(set_to_none=True);loss=0.;last=min(first+args.chunk,steps)
                    for s in range(first,last):
                        proposal=model(graphs[s],state)
                        if model.temporal:state=proposal.state
                        part,_=supervised_loss(model,graphs[s],proposal,x[:,s],y[:,s],physics_mode='original_worst')
                        loss+=part/(last-first)
                    if not bool(loss.isfinite()):raise RuntimeError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
                    train_total+=float(loss.detach())*(last-first);train_items+=last-first
                    if state is not None:state=state.detach()
                    if args.max_updates and updates>=args.max_updates:break
                if args.max_updates and updates>=args.max_updates:break
            stop=bool(args.max_updates and updates>=args.max_updates)
            if epoch==1 or epoch%10==0 or epoch==args.epochs or stop:
                model.eval();total=0.;items=0
                with torch.no_grad():
                    for graphs,x,y in batches('selection',args.selection_count):
                        state=None
                        for s in range(steps):
                            proposal=model(graphs[s],state)
                            if model.temporal:state=proposal.state
                            loss,_=supervised_loss(model,graphs[s],proposal,x[:,s],y[:,s],physics_mode='original_worst')
                            total+=float(loss);items+=1
                val=total/items
                if val<best:best=val;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
                row=dict(epoch=epoch,optimizer_updates=updates,train_loss=train_total/train_items,development_loss=val)
                history.append(row);print(json.dumps(dict(arm=arm,train_count=args.train_count,**row)),flush=True)
            if stop:break
        model.load_state_dict(best_state)
        meta={k:record[k] for k in ('stage','trace_sha256','model_fingerprints','train_seeds','development_seeds','training_problem_hashes','development_problem_hashes','source_sha256')}
        meta.update(best_epoch=best_epoch,development_loss=best,artifact_role='learning_curve_not_speed_qualified')
        file=args.output/(arm+'.pt');torch.save(checkpoint(model,meta),file)
        record['models'].append(dict(name=arm,training_seconds=time.perf_counter()-tick,optimizer_updates=updates,
            graph_staging_seconds=staging_seconds-staging_start,best_epoch=best_epoch,development_loss=best,history=history,
            checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),parameters=sum(v.numel() for v in model.parameters())))
        save();del model,optimizer,best_state
    record.update(status='completed',total_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated());save()


if __name__=='__main__':main()
