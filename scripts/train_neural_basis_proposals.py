"""Offline LP routing pilot with synthetic or independent CPU dFBA data."""
import argparse,hashlib,json,sys,time,os,shutil,copy
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compiled_basis_artifact import load_anchor
from src.gpu_basis_bank import GpuBasisBank
from src.gpu_neural_basis_proposal import features,bank_identity


def training_feature_indices(x, train):
    """Select varying, nonduplicate inputs using training trajectories only."""
    training=x[train]
    variable=np.flatnonzero(np.ptp(training,axis=0)>1e-5)
    if not len(variable):return np.array([0],dtype=np.int64)
    _,unique=np.unique(training[:,variable].T,axis=0,return_index=True)
    return variable[unique]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--seed',type=int,default=20286601)
    p.add_argument('--trajectories',type=int,default=0)
    p.add_argument('--trajectory-steps',type=int,default=2)
    p.add_argument('--trajectory-seed',type=int,default=20286701)
    p.add_argument('--cache-inputs',action='store_true',help='Persist normalized GPU inputs for offline relabeling')
    args=p.parse_args()
    if args.samples<64 or args.epochs<1 or args.trajectories<0 or not 1<=args.trajectory_steps<=120:
        raise ValueError('Invalid training size')
    if args.trajectories==1:raise ValueError('Need separate training and validation trajectories')
    args.output.mkdir(parents=True,exist_ok=False)
    import cupy as cp,torch
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    groups=defaultdict(list);provenance=[]
    for directory in (ROOT/'results/pf_basis_train20286311_4_compiled',ROOT/'results/pf_aggregate_extension_20260904'):
        manifest=json.loads((directory/'manifest.json').read_text());provenance.append(manifest)
        for entry in manifest['entries']:
            if entry['key'][0]=='exchange_tie':continue
            path=(directory/entry['filename']).resolve()
            if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Invalid offline source')
            groups[tuple(entry['key'])].append(load_anchor(path))
    report=dict(status='training',seed=args.seed,provenance=provenance,stages=[],
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(cupy=cp.__version__,torch=torch.__version__,CUPY_ACCELERATORS=os.environ.get('CUPY_ACCELERATORS','default')),
        warning='Synthetic interpolation of offline LP parameters; not independent biological validation')
    source_dir=args.output/'sources';source_dir.mkdir()
    for filename in [Path(__file__),ROOT/'src/gpu_neural_basis_proposal.py',ROOT/'src/gpu_basis_bank.py',ROOT/'src/gpu_certified_basis.py']:
        shutil.copyfile(filename,source_dir/filename.name)
    started=time.perf_counter()
    def save():
        (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    save()
    primary_rows=next(key[1] for key in groups if key[0]=='maxmin')
    actual=defaultdict(list)
    if args.trajectories:
        from scipy.optimize import linprog
        from scripts.benchmark_basis_bank_rollout import environment
        from src.gpu_compiled_community_backend import lp_arrays,stage_key
        from src.gpu_certified_basis import CommunityCoordinates
        from src.fba_surrogate import model_fingerprint
        training_started=time.perf_counter();sample,layout=environment(args.trajectory_seed)
        fingerprints={k:model_fingerprint(v) for k,v in sample.simulator.models.items()}
        if fingerprints!=provenance[0]['identity']['model_fingerprints']:raise ValueError('Training GEMs changed')
        meta=dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
            reaction_ids=[r.id for m in sample.simulator.models.values() for r in m.reactions],
            reaction_species=[k for k,m in sample.simulator.models.items() for r in m.reactions])
        first=next(iter(groups.values()))[0]['lp'];coords=CommunityCoordinates(meta,first.a,first.neq)
        lp_calls=0
        def collect(c,**kwargs):
            nonlocal lp_calls
            result=linprog(c,**kwargs);lp_calls+=1
            if not result.success:raise RuntimeError('Offline training CPU LP failed')
            arrays=lp_arrays(c,**kwargs);key=stage_key(arrays[4],arrays[0],arrays[-1],layout.n_fluxes)
            actual[key].append(coords.normalize(*arrays))
            return result
        training_seeds=list(range(args.trajectory_seed,args.trajectory_seed+args.trajectories))
        for seed in training_seeds:
            env=copy.deepcopy(sample);env.reset(seed=seed)
            env.simulator._cooperative_solver._linprog_options.update(threads=1,parallel=False)
            actions=np.random.default_rng(seed).uniform(.05,.95,(120,5)).astype(np.float32)
            with patch('src.community_solver.linprog',collect):
                for step,action in enumerate(actions[:args.trajectory_steps],1):
                    env.step(action)
                    if not env.simulator._cooperative_solver.stats.status.startswith('optimal;'):
                        raise RuntimeError('Offline training rollout failed')
                    if step%20==0:print(f'actual training seed={seed} step={step}/{args.trajectory_steps}; CPU LP calls={lp_calls}',flush=True)
            print(f'actual training trajectory {seed}; CPU LP calls={lp_calls}',flush=True)
            del env
        report.update(actual_training_seeds=training_seeds,actual_training_steps=args.trajectory_steps,
            actual_training_cpu_lp_calls=lp_calls,actual_training_seconds=time.perf_counter()-training_started,
            actual_cpu_lp_options=dict(threads=1,parallel=False),
            warning='Independent CPU dFBA trajectories; labels choose a bank basis, not proof of successful repair')
        save()
    for key,anchors in groups.items():
        stage_started=time.perf_counter();neq=key[-1];rows=list(range(neq,neq+3))
        if key[0]=='exchange':rows.append(primary_rows)
        bank=GpuBasisBank(anchors,rows);base=bank.prepare_host([a['lp'] for a in anchors])
        anchor_features=features(base).get()
        variable=np.flatnonzero(np.ptp(anchor_features,axis=0)>1e-5)
        if len(variable):
            _,unique=np.unique(anchor_features[:,variable].T,axis=0,return_index=True)
            indices=variable[unique]
        else:indices=np.array([0])
        xs=[];ys=[];accepted=0
        count=len(actual[key]) if args.trajectories else args.samples
        for start in range(0,count,32):
            size=min(32,count-start)
            if args.trajectories:data=bank.prepare_host(actual[key][start:start+size])
            else:
                left=cp.asarray(rng.integers(len(anchors),size=size));right=cp.asarray(rng.integers(len(anchors),size=size))
                weights=cp.asarray(rng.uniform(0,.25,size=size))
                data={}
                for name,array in base.items():
                    w=weights.reshape((size,)+(1,)*(array.ndim-1));a=array[left];b=array[right]
                    finite=cp.isfinite(a)&cp.isfinite(b)
                    mixed=(1-w)*cp.where(finite,a,0.)+w*cp.where(finite,b,0.)
                    data[name]=cp.where(finite,mixed,a)
            if args.cache_inputs and args.trajectories:
                cache_dir=args.output/'inputs'/key[0];cache_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(cache_dir/f'{start:06d}.npz',**{k:v.get() for k,v in data.items()})
            teacher=bank.evaluate_device(**data);valid=teacher['accepted'].get()
            accepted+=int(valid.sum())
            if args.trajectories:
                labels=cp.where(teacher['accepted'],teacher['candidate_index'],teacher['best_candidate_index']).get()
                # Anchor-only variation misses features that start changing
                # later in a trajectory. Select from TRAINING states below.
                xs.append(features(data).get());ys.append(labels)
            else:
                xs.append(features(data).get()[valid][:,indices]);ys.append(teacher['candidate_index'].get()[valid])
        x=np.concatenate(xs);y=np.concatenate(ys)
        if len(y)<32:raise RuntimeError(f'Insufficient certified synthetic examples: {key}: {len(y)}')
        if args.trajectories:
            # Split entire trajectories, not adjacent states from one seed.
            shuffled=rng.permutation(args.trajectories);split=max(1,int(.8*args.trajectories))
            train=np.array([i for t in shuffled[:split] for i in range(t*args.trajectory_steps,(t+1)*args.trajectory_steps)])
            test=np.array([i for t in shuffled[split:] for i in range(t*args.trajectory_steps,(t+1)*args.trajectory_steps)])
        else:
            order=rng.permutation(len(y));split=max(1,int(.8*len(y)));train=order[:split];test=order[split:]
        if args.trajectories:
            indices=training_feature_indices(x,train)
            x=x[:,indices]
        np.savez_compressed(args.output/(key[0]+'_training.npz'),x=x,y=y,train=train,
            validation=test,indices=indices)
        mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-4)
        tx=torch.as_tensor((x-mean)/scale,device='cuda');ty=torch.as_tensor(y,device='cuda',dtype=torch.long)
        network=torch.nn.Sequential(torch.nn.Linear(len(indices),64),torch.nn.Tanh(),torch.nn.Linear(64,len(anchors))).cuda()
        optimizer=torch.optim.Adam(network.parameters(),lr=.01,weight_decay=1e-4)
        for epoch in range(args.epochs):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(network(tx[train]),ty[train]);loss.backward();optimizer.step()
        with torch.no_grad():
            accuracy=(network(tx[test]).argmax(1)==ty[test]).float().mean().item()
        filename=key[0]+'.npz'
        metadata=dict(stage=key[0],bank_identity=bank_identity(bank),feature_width=anchor_features.shape[1],
            training_kind='actual independent CPU trajectories; candidate or best-rejected warm-basis labels' if args.trajectories else 'synthetic offline LP interpolation; certified labels only',training_seed=args.seed,
            certified_examples=accepted,training_examples=len(train),validation_examples=len(test),
            validation_top1_accuracy=accuracy,parameters=sum(v.numel() for v in network.parameters()),
            feature_count=len(indices),feature_selection='training trajectories only' if args.trajectories else 'offline anchors',
            training_class_counts=np.bincount(y[train],minlength=len(anchors)).tolist(),
            validation_class_counts=np.bincount(y[test],minlength=len(anchors)).tolist(),
            validation_majority_class_accuracy=float(np.mean(y[test]==np.bincount(y[train]).argmax())))
        np.savez(args.output/filename,indices=indices,mean=mean,scale=scale,
            w1=network[0].weight.detach().cpu().numpy().T,b1=network[0].bias.detach().cpu().numpy(),
            w2=network[2].weight.detach().cpu().numpy().T,b2=network[2].bias.detach().cpu().numpy(),metadata=json.dumps(metadata))
        report['stages'].append(dict(metadata,filename=filename,seconds=time.perf_counter()-stage_started,
            sha256=hashlib.sha256((args.output/filename).read_bytes()).hexdigest()))
        save();print(f'{key[0]}: {accepted}/{count} certified, routing validation top1={accuracy:.3f}',flush=True)
        del bank,base,tx,ty,network,optimizer
        cp.get_default_memory_pool().free_all_blocks();torch.cuda.empty_cache()
    report.update(status='completed',total_seconds=time.perf_counter()-started);save()


if __name__=='__main__':main()
