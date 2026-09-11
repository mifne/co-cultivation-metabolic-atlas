"""GPU GRU/MLP warm-start pilot on explicitly captured training references.

Supervised latent loss is not proof of LP feasibility or online speedup.
Development metrics use teacher-forced previous labels and are marked as such.
Independent causal correction is measured by benchmark_temporal_lp.py.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.temporal_lp_data import load_trajectories, fit_codec, trajectory_arrays


def train_models(codec, arrays, train_ids, dev_ids, output, *, epochs=160, hidden_dim=64,
                 learning_rate=1e-3, seed=1, device='cuda', previous_dropout=.35):
    import torch
    from src.temporal_lp_model import TemporalLPConfig, TemporalLPModel, temporal_checkpoint
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This pilot requires actual CUDA training')
    features, previous, labels, _ = arrays
    features, previous, labels = [torch.as_tensor(a, device=device, dtype=torch.float32)
                                   for a in (features,previous,labels)]
    tr = torch.tensor(train_ids, device=device)
    dv = torch.tensor(dev_ids, device=device)
    results = []
    for kind in ('mlp','gru'):
        torch.manual_seed(seed)
        model = TemporalLPModel(TemporalLPConfig(kind=kind, feature_dim=codec.feature_dim,
            latent_dim=codec.rank, hidden_dim=hidden_dim)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss, best, history = float('inf'), None, []
        torch.cuda.synchronize(); started = time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train()
            x, prev = features[tr].clone(), previous[tr].clone()
            # Real online failures remove the preceding certified state.
            # Random dropout teaches this explicitly; it does not mark an
            # unverified model prediction as a certified previous solution.
            drop = torch.rand(prev.shape[:2], device=device) < previous_dropout
            prev[drop] = 0.; x[:,:,-1][drop] = 0.
            predicted, _ = model(x,prev)
            loss = torch.mean((predicted-labels[tr])**2)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                estimate, _ = model(features[dv],previous[dv])
                dev_loss = float(torch.mean((estimate-labels[dv])**2))
                # Also expose the harder no-certified-previous trajectory.
                no_prev = features[dv].clone(); no_prev[:,:,-1] = 0.
                cold, _ = model(no_prev, torch.zeros_like(previous[dv]))
                missing_loss = float(torch.mean((cold-labels[dv])**2))
            if dev_loss < best_loss:
                best_loss = dev_loss
                best = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                history.append(dict(epoch=epoch, train_mse=float(loss.detach()),
                    development_teacher_forced_mse=dev_loss,
                    development_no_previous_mse=missing_loss))
                print(f'{kind} epoch {epoch}: train={float(loss.detach()):.5g}, '
                      f'dev teacher={dev_loss:.5g}, no previous={missing_loss:.5g}',flush=True)
        torch.cuda.synchronize(); elapsed = time.perf_counter()-started
        model.load_state_dict(best); model.eval()
        path = output / f'{kind}.pt'
        meta = dict(codec.metadata, codec_sha256=hashlib.sha256((output/'codec.npz').read_bytes()).hexdigest(),
            training_mode='supervised latent prediction with certified-previous dropout',
            selection_metric='development teacher-forced latent MSE; not online accuracy',
            best_epoch=best_epoch)
        torch.save(temporal_checkpoint(model, metadata=meta),path)
        results.append(dict(kind=kind, parameters=sum(p.numel() for p in model.parameters()),
            training_seconds=elapsed, best_epoch=best_epoch,
            development_teacher_forced_mse=best_loss, history=history,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return results, dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0), device='cuda')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage', choices=['maxmin','aggregate','exchange'],default='exchange')
    parser.add_argument('--train-environments',type=int,default=12)
    parser.add_argument('--rank',type=int,default=32)
    parser.add_argument('--features',type=int,default=128)
    parser.add_argument('--epochs',type=int,default=160)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20294701)
    args = parser.parse_args()
    if min(args.rank,args.features,args.epochs,args.hidden) < 1:
        raise ValueError('Positive training dimensions/budget required')
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(status='initializing', configuration={k:str(v) if isinstance(v,Path) else v
                  for k,v in vars(args).items()},
        scope='Offline supervised development pilot; no end-to-end GPU speedup qualification')
    def save():
        temp = args.output/'manifest.json.tmp'
        temp.write_text(json.dumps(report,indent=2,allow_nan=False))
        temp.replace(args.output/'manifest.json')
    started = time.perf_counter()
    try:
        save()
        source_files = ['src/temporal_lp_model.py','src/temporal_lp_data.py',
            'src/lp_trace.py','scripts/train_temporal_lp.py']
        report['source_hashes'] = {}
        for name in source_files:
            dst = args.output/'sources'/name; dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dst)
            report['source_hashes'][name] = hashlib.sha256(dst.read_bytes()).hexdigest()
        manifest, trajectories, sha = load_trajectories(args.trace,args.stage,required_role='training_reference')
        count = len(trajectories)
        if not 2 <= args.train_environments < count:
            raise ValueError('Separate train and development trajectories required')
        # Explicit chronological environment IDs; all preprocessing fits train.
        train_ids = list(range(args.train_environments))
        dev_ids = list(range(args.train_environments,count))
        meta = dict(stage=args.stage, trace_sha256=sha, model_fingerprints=manifest['model_fingerprints'],
            train_seeds=[manifest['seeds'][i] for i in train_ids],
            development_seeds=[manifest['seeds'][i] for i in dev_ids],
            trace_role=manifest['role'])
        codec = fit_codec(trajectories,train_ids,rank=args.rank,max_features=args.features,
                          seed=args.seed,metadata=meta)
        codec.save(args.output/'codec.npz')
        arrays = trajectory_arrays(trajectories,codec)
        reconstruction = codec.decode(arrays[2])
        report.update(meta, codec_sha256=hashlib.sha256((args.output/'codec.npz').read_bytes()).hexdigest(),
            feature_dim=codec.feature_dim, rank=codec.rank,
            samples=dict(training=len(train_ids)*len(trajectories[0]),
                         development=len(dev_ids)*len(trajectories[0])),
            reconstruction_normalized_mse={key:float(np.mean(((reconstruction[ids]-arrays[3][ids])/
                codec.arrays['target_scale'])**2)) for key,ids in [('train',train_ids),('development',dev_ids)]},
            teacher_warning='Development model selection uses previous reference labels, '
                'not proof of causal online speed or acceptance')
        training_hashes = {e['problem_sha256'] for e in manifest['entries']
                           if e['stage']==args.stage and e['environment_id'] in train_ids}
        dev_entries = [e for e in manifest['entries'] if e['stage']==args.stage and e['environment_id'] in dev_ids]
        report['development_inputs_identical_to_training'] = sum(e['problem_sha256'] in training_hashes for e in dev_entries)
        report['preprocessing_seconds'] = time.perf_counter()-started
        report['status'] = 'training'; save()
        report['models'], report['runtime'] = train_models(codec, arrays,train_ids,dev_ids,args.output,
            epochs=args.epochs,hidden_dim=args.hidden,seed=args.seed)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['offline_total_seconds'] = time.perf_counter()-started
        save()


if __name__ == '__main__':
    main()
