"""Train a joint router on disjoint early/long-horizon offline datasets.

Use only features present in every dataset (explicit intersection, no
fabricated missing values). Preserve each source's trajectory-held-out split.
This trains proposal labels, not a new flux acceptance rule.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np


def align_datasets(datasets):
    common=datasets[0]['indices']
    for data in datasets[1:]:common=np.intersect1d(common,data['indices'])
    if not len(common):raise ValueError('No shared features')
    xs=[];ys=[];train=[];validation=[];offset=0
    for data in datasets:
        lookup={int(index):j for j,index in enumerate(data['indices'])}
        xs.append(data['x'][:,[lookup[int(index)] for index in common]])
        ys.append(data['y']);train.extend(data['train']+offset)
        validation.extend(data['validation']+offset);offset+=len(data['y'])
    return common,np.concatenate(xs),np.concatenate(ys),np.array(train),np.array(validation)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=600)
    p.add_argument('--seed',type=int,default=20287301)
    args=p.parse_args()
    if args.epochs<1:raise ValueError('Positive epochs required')
    manifests=[json.loads((path/'manifest.json').read_text()) for path in args.inputs]
    if any(m['status']!='completed' for m in manifests):raise ValueError('Incomplete source dataset')
    seeds=[s for m in manifests for s in m['actual_training_seeds']]
    if len(seeds)!=len(set(seeds)):raise ValueError('Repeated trajectory seeds across datasets')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,args.output/'training_source.py')
    import torch
    torch.manual_seed(args.seed);started=time.perf_counter()
    report=dict(status='training',seed=args.seed,actual_training_seeds=seeds,stages=[],
        sources=[dict(path=str(path),manifest=m) for path,m in zip(args.inputs,manifests)],
        source_cpu_lp_calls=sum(m['actual_training_cpu_lp_calls'] for m in manifests),
        warning='Joint basis routing; feature intersection; source trajectory splits preserved; full LP certificates still required')
    for stage in ('maxmin','aggregate','exchange'):
        data=[];meta=[];hashes={}
        for path in args.inputs:
            file=path/(stage+'_training.npz')
            hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            with np.load(file,allow_pickle=False) as d:data.append(dict(d))
            with np.load(path/(stage+'.npz'),allow_pickle=False) as d:meta.append(json.loads(str(d['metadata'])))
        if len({(m['bank_identity'],m['feature_width']) for m in meta})!=1:raise ValueError('Incompatible dictionaries')
        indices,x,y,train,test=align_datasets(data)
        k=len(meta[0]['training_class_counts'])
        if len(set(train)&set(test)):raise ValueError('Training/validation overlap')
        mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-4)
        tx=torch.as_tensor((x-mean)/scale,device='cuda');ty=torch.as_tensor(y,device='cuda',dtype=torch.long)
        network=torch.nn.Sequential(torch.nn.Linear(len(indices),64),torch.nn.Tanh(),torch.nn.Linear(64,k)).cuda()
        optimizer=torch.optim.Adam(network.parameters(),lr=.01,weight_decay=1e-4)
        for epoch in range(args.epochs):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(network(tx[train]),ty[train]);loss.backward();optimizer.step()
        with torch.no_grad():accuracy=(network(tx[test]).argmax(1)==ty[test]).float().mean().item()
        metadata=dict(stage=stage,bank_identity=meta[0]['bank_identity'],feature_width=meta[0]['feature_width'],
            feature_count=len(indices),feature_selection='intersection of training-only feature selections',
            training_examples=len(train),validation_examples=len(test),validation_top1_accuracy=accuracy,
            parameters=sum(v.numel() for v in network.parameters()),training_class_counts=np.bincount(y[train],minlength=k).tolist(),
            validation_class_counts=np.bincount(y[test],minlength=k).tolist(),source_dataset_sha256=hashes)
        file=args.output/(stage+'.npz')
        np.savez(file,indices=indices,mean=mean,scale=scale,w1=network[0].weight.detach().cpu().numpy().T,
            b1=network[0].bias.detach().cpu().numpy(),w2=network[2].weight.detach().cpu().numpy().T,
            b2=network[2].bias.detach().cpu().numpy(),metadata=json.dumps(metadata))
        np.savez_compressed(args.output/(stage+'_training.npz'),indices=indices,x=x,y=y,train=train,validation=test)
        report['stages'].append(dict(metadata,filename=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        print(f'{stage}: {len(y)} examples, {len(indices)} common features, validation={accuracy:.3f}',flush=True)
    report.update(status='completed',total_seconds=time.perf_counter()-started)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
