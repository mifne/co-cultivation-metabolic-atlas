"""Offline inverse operators for compact-bank GPU repair; no new LP samples.

Only the already declared training bases are factored. These are linear
operators, not additional fitted solutions or validation-derived anchors.
"""
import argparse,hashlib,json,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def build(job):
    directory,stage,index,entry,target=job
    folder=Path(directory)/stage;target=Path(target);started=time.perf_counter()
    def checked(name,digest=None):
        file=folder/name
        if digest and hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise ValueError('Changed source artifact')
        with np.load(file,allow_pickle=False) as d:return dict(d)
    root=checked('root.npz');data=checked(entry['filename'],entry['sha256'])
    a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape'])).tolil()
    a[root['variable_rows']]=a[root['variable_rows']]+csr_matrix(data['offset'][0]);a=a.tocsr()
    b=a[data['active']][:,data['basic']].tocsc();factor=splu(b);nb=b.shape[0]
    blocks=[]
    for start in range(0,nb,128):
        stop=min(start+128,nb);unit=np.zeros((nb,stop-start));unit[np.arange(start,stop),np.arange(stop-start)]=1.
        blocks.append(csr_matrix(factor.solve(unit)))
    from scipy.sparse import hstack
    inverse=hstack(blocks,format='csr');del blocks
    filename=f'{stage}_{index:04d}.npz';file=target/filename
    if file.exists():raise FileExistsError(file)
    np.savez(file,inverse_data=inverse.data,inverse_indices=inverse.indices,inverse_indptr=inverse.indptr,inverse_shape=inverse.shape)
    return dict(stage=stage,index=index,filename=filename,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        source_sha256=entry['sha256'],seconds=time.perf_counter()-started,bytes=file.stat().st_size)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.bank/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Incomplete source')
    report=dict(status='building',source_bank=str(args.bank),source_manifest_sha256=hashlib.sha256((args.bank/'manifest.json').read_bytes()).hexdigest(),
        entries=[],offline_new_cpu_lp_calls=0,scope='Offline factorization of existing training bases only')
    def save():(args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    save();started=time.perf_counter()
    jobs=[(str(args.bank),s['stage'],i,e,str(args.output)) for s in manifest['stages'] for i,e in enumerate(s['entries'])]
    with ProcessPoolExecutor(args.workers) as pool:
        for entry in pool.map(build,jobs):
            report['entries'].append(entry);save();print(f'{entry["stage"]} {entry["index"]}: {entry["seconds"]:.1f} s; {entry["bytes"]/1e6:.1f} MB',flush=True)
    report.update(status='completed',wall_seconds=time.perf_counter()-started);save()


if __name__=='__main__':main()
