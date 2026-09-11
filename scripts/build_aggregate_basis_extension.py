"""Paid offline stage-2 coverage expansion; never train on benchmark queries."""
import argparse,copy,hashlib,json,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_basis_bank_rollout import environment
from src.gpu_certified_basis import CommunityCoordinates,compile_basis,factor_compiled_basis
from src.gpu_compiled_community_backend import lp_arrays,stage_key
from src.compiled_basis_artifact import save_anchor
from src.fba_surrogate import model_fingerprint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20286501,20286502,20286503,20286504])
    p.add_argument('--steps',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.steps<=120 or len(set(args.seeds))!=len(args.seeds):raise ValueError('Invalid training design')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();sample,layout=environment(args.seeds[0])
    fingerprints={k:model_fingerprint(v) for k,v in sample.simulator.models.items()}
    meta=dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for m in sample.simulator.models.values() for r in m.reactions],
        reaction_species=[k for k,m in sample.simulator.models.items() for r in m.reactions])
    report=dict(status='training',identity=dict(train_seeds=args.seeds,train_steps=args.steps,
        model_fingerprints=fingerprints),entries=[],cpu_lp_calls=0,
        scope='Additional offline training only; three original objectives unchanged')
    for filename in ['src/gpu_certified_basis.py','src/compiled_basis_artifact.py',
                     'src/community_solver.py','src/dfba_simulator.py',str(Path(__file__).relative_to(ROOT))]:
        destination=args.output/'sources'/filename;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/filename,destination)
    report['identity']['compiler_source_sha256']=hashlib.sha256((ROOT/'src/gpu_certified_basis.py').read_bytes()).hexdigest()
    report['identity']['artifact_source_sha256']=hashlib.sha256((ROOT/'src/compiled_basis_artifact.py').read_bytes()).hexdigest()
    def save():
        temp=args.output/'manifest.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(args.output/'manifest.json')
    save();anchors={};coordinates=None
    for seed in args.seeds:
        env=copy.deepcopy(sample);env.reset(seed=seed)
        def collect(c,**kwargs):
            nonlocal coordinates
            report['cpu_lp_calls']+=1
            arrays=lp_arrays(c,**kwargs)
            key=stage_key(arrays[4],arrays[0],arrays[-1],layout.n_fluxes)
            if key[0]!='aggregate':return linprog(c,**kwargs)
            if coordinates is None:coordinates=CommunityCoordinates(meta,arrays[0],arrays[-1])
            anchor=compile_basis(*arrays,coordinates=coordinates,factorize=False)
            signature=hashlib.sha256(b''.join(anchor[k].tobytes() for k in ('basic','active','kind','row_kind'))).hexdigest()
            anchors.setdefault(signature,(key,anchor,seed,step))
            return SimpleNamespace(success=True,x=anchor['cpu_anchor_values'],fun=anchor['cpu_anchor_objective'],message='offline training')
        actions=np.random.default_rng(seed).uniform(.05,.95,(120,5)).astype(np.float32)
        with patch('src.community_solver.linprog',collect):
            for step,action in enumerate(actions[:args.steps],1):
                env.step(action)
                status=env.simulator._cooperative_solver.stats.status
                if not status.startswith('optimal;'):raise RuntimeError(status)
                print(f'train seed={seed} step={step}; unique aggregate bases={len(anchors)}',flush=True)
    report.update(status='factorization',training_and_environment_seconds=time.perf_counter()-started)
    save();factor_started=time.perf_counter()
    for index,(key,anchor,seed,step) in enumerate(anchors.values()):
        anchor=factor_compiled_basis(anchor);filename=f'aggregate_{index:04d}.npz';path=args.output/filename
        save_anchor(path,anchor)
        report['entries'].append(dict(key=list(key),filename=filename,seed=seed,step=step,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        save();print(f'factorized {index+1}/{len(anchors)}',flush=True)
    report.update(status='completed',factorization_seconds=time.perf_counter()-factor_started,
                  total_offline_seconds=time.perf_counter()-started)
    save()


if __name__=='__main__':main()
