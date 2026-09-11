"""Supervised full-space GNN versus GNN+GRU; trajectory-separated CUDA pilot."""
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
from src.temporal_lp_data import load_trajectories
from src.graph_temporal_lp import LPGraphBatch
from src.supervised_graph_lp import SupervisedGraphLP,supervised_loss,checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,default=ROOT/'results/pf_gru_training16x60_20260905')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--chunk',type=int,default=8)
    p.add_argument('--hidden',type=int,default=16)
    p.add_argument('--seed',type=int,default=20294907)
    p.add_argument('--normalization',choices=['std','rms'],default='std')
    p.add_argument('--physics-mode',choices=['scaled_mean','original_worst'],default='scaled_mean')
    args=p.parse_args()
    if min(args.epochs,args.steps,args.chunk,args.hidden)<1:p.error('Positive budgets required')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='loading',scope=__doc__,stage='maxmin',configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},models=[])
    def save():
        (args.output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    started=time.perf_counter();save()
    manifest,trajs,sha=load_trajectories(args.trace,'maxmin',required_role='training_reference')
    if len(trajs)!=16 or any(len(t)<args.steps for t in trajs):raise ValueError('16 distinct complete training trajectories required')
    trajs=[t[:args.steps] for t in trajs]
    # Recompute model identity without stepping or optimizing the environment.
    from scripts.benchmark_basis_bank_rollout import environment
    from src.fba_surrogate import model_fingerprint
    env,_=environment(manifest['seeds'][0])
    fingerprints={k:model_fingerprint(v) for k,v in env.simulator.models.items()};env.close()
    if fingerprints!=manifest['model_fingerprints']:raise ValueError('Current GEM fingerprints mismatch')
    identity=json.dumps(fingerprints,sort_keys=True)
    if not torch.cuda.is_available():raise RuntimeError('Actual NVIDIA CUDA required')
    torch.set_num_threads(1)
    train_ids=list(range(12));dev_ids=list(range(12,16))
    x=np.stack([[v[1] for v in t] for t in trajs]);y=np.stack([[v[2] for v in t] for t in trajs])
    stats=[]
    for a in (x[train_ids],y[train_ids]):
        scale=np.maximum(a.std((0,1)),1e-3) if args.normalization=='std' else np.maximum(np.sqrt((a*a).mean((0,1))),1.)
        stats.extend((a.mean((0,1)),scale))
    groups=[list(range(i,i+4)) for i in (0,4,8,12)]
    graphs=[]
    for ids in groups:
        graphs.append([LPGraphBatch.from_problems([trajs[i][t][0] for i in ids],stage='maxmin',
            model_identity=identity,device='cuda') for t in range(args.steps)])
    graph_ids={g.identity for group in graphs for g in group}
    if len(graph_ids)!=1:raise ValueError('Graph support differs; explicit remapping needed')
    targets=[(torch.tensor(x[ids],device='cuda'),torch.tensor(y[ids],device='cuda')) for ids in groups]
    train_hash={h for group in graphs[:3] for g in group for h in g.problem_hashes}
    dev_hash={h for g in graphs[3] for h in g.problem_hashes}
    if train_hash&dev_hash:raise ValueError('Identical train/development LP inputs would invalidate split')
    paths=['scripts/train_supervised_graph_lp.py','src/supervised_graph_lp.py','src/graph_temporal_lp.py','src/lp_trace.py']
    report.update(trace_sha256=sha,model_fingerprints=fingerprints,graph_identity=next(iter(graph_ids)),
        train_seeds=[manifest['seeds'][i] for i in train_ids],development_seeds=[manifest['seeds'][i] for i in dev_ids],
        training_problem_hashes=sorted(train_hash),development_problem_hashes=sorted(dev_hash),
        training_samples=12*args.steps,development_samples=4*args.steps,
        parameter_fit_scope='train environments only; development chooses checkpoint',
        previous_teacher_solution_input=False,validation='causal hidden-state sequence, saved LP inputs, not closed-loop dFBA',
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in paths},
        gpu=torch.cuda.get_device_name(0),preprocessing_seconds=time.perf_counter()-started,status='training')
    for f in paths:
        dst=args.output/'sources'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/f).read_bytes())
    save()
    for temporal in (False,True):
        name='gnn_gru' if temporal else 'gnn'
        torch.manual_seed(args.seed)
        model=SupervisedGraphLP(next(iter(graph_ids)),*stats,hidden=args.hidden,temporal=temporal).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        best=float('inf');best_state=None;history=[];tick=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            model.train();train_total=0.
            for group in np.random.default_rng(args.seed+epoch).permutation(3):
                state=None
                for first in range(0,args.steps,args.chunk):
                    optimizer.zero_grad(set_to_none=True);loss=0.
                    last=min(first+args.chunk,args.steps)
                    for t in range(first,last):
                        proposal=model(graphs[group][t],state)
                        if temporal:state=proposal.state
                        part,_=supervised_loss(model,graphs[group][t],proposal,targets[group][0][:,t],targets[group][1][:,t],physics_mode=args.physics_mode)
                        loss=loss+part/(last-first)
                    if not bool(loss.isfinite()):raise RuntimeError('Nonfinite supervised loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step();train_total+=float(loss.detach())*(last-first)/(3*args.steps)
                    if state is not None:state=state.detach()
            if epoch==1 or epoch%10==0 or epoch==args.epochs:
                model.eval();state=None;val=0.;component={}
                with torch.no_grad():
                    for t in range(args.steps):
                        proposal=model(graphs[3][t],state)
                        if temporal:state=proposal.state
                        loss,parts=supervised_loss(model,graphs[3][t],proposal,targets[3][0][:,t],targets[3][1][:,t],physics_mode=args.physics_mode)
                        val+=float(loss)/args.steps
                        for k,v in parts.items():component[k]=component.get(k,0.)+float(v)/args.steps
                if val<best:best=val;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
                row=dict(epoch=epoch,train_loss=train_total,development_loss=val,components=component)
                history.append(row);print(json.dumps(dict(model=name,**row)),flush=True)
        torch.cuda.synchronize();training_seconds=time.perf_counter()-tick
        model.load_state_dict(best_state)
        meta={k:report[k] for k in ('trace_sha256','model_fingerprints','train_seeds','development_seeds',
            'training_problem_hashes','development_problem_hashes','source_sha256')}
        meta.update(best_epoch=best_epoch,development_loss=best,stage='maxmin',
            artifact_role='supervised_development_pilot_not_speed_qualified')
        path=args.output/(name+'.pt');torch.save(checkpoint(model,meta),path)
        report['models'].append(dict(name=name,parameters=sum(v.numel() for v in model.parameters()),
            training_seconds=training_seconds,best_epoch=best_epoch,development_loss=best,history=history,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        save();del model,optimizer,best_state
    report.update(status='completed',total_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated());save()


if __name__=='__main__':main()
