"""Offline compact basis maps from declared full-horizon training inputs."""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix,diags
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.gpu_certified_basis import NormalizedLP,compile_basis
from src.compiled_basis_artifact import load_anchor
from src.gpu_compact_basis import project_basis
from scripts.benchmark_basis_bank_rollout import environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'results/pf_neural_basis_actual4x120_20260904')
    p.add_argument('--steps',type=int,nargs='+',default=[1,2,3,4,8,16,32,64,96,120])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    training=json.loads((args.data/'manifest.json').read_text())
    if training['status']!='completed':raise ValueError('Incomplete source')
    sample,layout=environment(training['actual_training_seeds'][0])
    tie_c=np.zeros(layout.n_fluxes+1+sum(map(len,layout._exchange_terms.values())))
    tie_c[layout.n_fluxes+1:]=[0. if met in {'h2o_e','h_e','oh1_e'} else 1. for met,terms in layout._exchange_terms.items() for _ in terms]
    started=time.perf_counter();report=dict(status='building',source_data=str(args.data),source_manifest=training,
        model_fingerprints=training['provenance'][0]['identity']['model_fingerprints'],stages=[],cpu_lp_calls=0,
        train_seeds=sorted(set(training['actual_training_seeds']+training.get('parent_training_seeds',[])+
            [20286311,20287001,20287002,20287003,20287004,20287005,20287006,20287007,20287008])))
    for filename in [Path(__file__),ROOT/'src/gpu_compact_basis.py',ROOT/'src/gpu_certified_basis.py']:
        shutil.copyfile(filename,args.output/filename.name)
    def save(): (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    save();base=ROOT/'results/pf_basis_train20286311_4_compiled'
    base_manifest=json.loads((base/'manifest.json').read_text())
    for stage in ('maxmin','aggregate','exchange'):
        entries=[e for e in base_manifest['entries'] if e['key'][0]==stage]
        original=load_anchor(base/entries[0]['filename']);root=original['lp'];key=entries[0]['key']
        variable_rows=list(range(root.neq,root.neq+3))
        if stage=='exchange':variable_rows.append(root.a.shape[0]-2*(root.a.shape[1]-layout.n_fluxes-1)-1)
        chunks=[]
        for file in sorted((args.data/'inputs'/stage).glob('*.npz')):
            with np.load(file,allow_pickle=False) as d:chunks.append(dict(d))
        data={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]};del chunks
        stage_dir=args.output/stage;stage_dir.mkdir()
        np.savez(stage_dir/'root.npz',a_data=root.a.data,a_indices=root.a.indices,a_indptr=root.a.indptr,
            a_shape=root.a.shape,neq=root.neq,variable_rows=variable_rows)
        with np.load(args.data/(stage+'.npz'),allow_pickle=False) as d:
            feature_indices=d['indices'];feature_scale=d['scale']
        anchors=[(load_anchor(base/e['filename']),dict(source='existing_base',source_filename=e['filename'])) for e in entries]
        if stage=='aggregate':
            directory=ROOT/'results/pf_aggregate_reserve32_20260904'
            for e in json.loads((directory/'manifest.json').read_text())['entries']:
                if e['step']==2:anchors.append((load_anchor(directory/e['filename']),dict(source='existing_reserve',seed=e['seed'],step=e['step'])))
        for trajectory,seed in enumerate(training['actual_training_seeds']):
            for step in args.steps:
                index=trajectory*training['actual_training_steps']+step-1
                a=root.a.tolil(copy=True);a[variable_rows]=root.a[variable_rows]+csr_matrix(data['delta'][index]);a=a.tocsr()
                q=NormalizedLP(a,*(data[k][index] for k in ('rhs','lower','upper','c')),root.neq,data['col_scale'][index],data['row_scale'][index])
                original_a=(diags(1/q.row_scale)@a@diags(q.col_scale)).tocsr()
                anchor=compile_basis(original_a,q.rhs/q.row_scale,q.lower/q.col_scale,q.upper/q.col_scale,
                    q.c*q.col_scale,q.neq,coordinates=SimpleNamespace(normalize=lambda *a:q),factorize=False)
                report['cpu_lp_calls']+=1;anchors.append((anchor,dict(source='new_offline',seed=seed,step=step)))
        stage_report=dict(stage=stage,key=key,entries=[],projected_bytes=0);report['stages'].append(stage_report)
        signatures=set();centers=[]
        for anchor,source in anchors:
            signature=hashlib.sha256(b''.join(anchor[k].tobytes() for k in ('basic','active','kind','row_kind'))).hexdigest()
            if signature in signatures:continue
            signatures.add(signature);q=anchor['lp'];offset=(q.a-root.a)[variable_rows].toarray()[None]
            projected=project_basis(anchor,data,offset,variable_rows,[tie_c] if stage=='exchange' else [])
            filename=f'basis_{len(centers):04d}.npz';file=stage_dir/filename
            np.savez(file,**projected)
            vectors=dict(delta=offset,rhs=q.rhs[None],lower=q.lower[None],upper=q.upper[None],c=q.c[None],col_scale=q.col_scale[None],row_scale=q.row_scale[None])
            joined=np.concatenate([vectors[k].reshape(1,-1) for k in ('rhs','lower','upper','c','delta','col_scale','row_scale')],axis=1)
            joined=np.nan_to_num(joined,nan=0.,posinf=1e13,neginf=-1e13)
            centers.append((np.sign(joined)*np.log1p(np.abs(joined))).astype(np.float32)[0,feature_indices])
            stage_report['entries'].append(dict(filename=filename,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),**source))
            stage_report['projected_bytes']+=sum(v.nbytes for v in projected.values())
            print(f'{stage}: compact basis {len(centers)}/{len(anchors)}; bytes={stage_report["projected_bytes"]}',flush=True)
            save()
        np.savez(stage_dir/'router.npz',centers=np.stack(centers),indices=feature_indices,scale=feature_scale)
        stage_report['root_sha256']=hashlib.sha256((stage_dir/'root.npz').read_bytes()).hexdigest()
        stage_report['router_sha256']=hashlib.sha256((stage_dir/'router.npz').read_bytes()).hexdigest()
        del anchors,data
    report.update(status='completed',total_seconds=time.perf_counter()-started);save()


if __name__=='__main__':main()
