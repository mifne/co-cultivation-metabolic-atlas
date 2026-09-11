"""Read-only structural presolve audit of declared training LP families.

Observed fixed bounds are NOT assumed to hold on unseen trajectories. Any
future reduced runtime must guard them and certify the full original LP.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_compact_gpu import checked_npz


def audit(root,data):
    a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
    neq=int(root['neq']);n=a.shape[1]
    fixed=np.ones(n,dtype=bool);rows=0
    for path in sorted(data.glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            fixed&=((d['lower']==0)&(d['upper']==0)).all(axis=0);rows+=len(d['lower'])
    if not rows:raise ValueError('Missing training inputs')
    original=fixed.copy();rounds=[];dropped=np.zeros(neq,dtype=bool)
    eq=a[:neq].tocsr();eq.eliminate_zeros()
    # These input families have structural homogeneous mass-balance rows.
    # Check, rather than infer, that the cached equality RHS is zero.
    for path in sorted(data.glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            if np.any(d['rhs'][:,:neq]!=0):raise ValueError('Nonzero equality RHS')
    while True:
        remaining=eq[:,~fixed].tocsr();remaining.eliminate_zeros()
        count=np.diff(remaining.indptr);single=np.flatnonzero((count==1)&~dropped)
        dropped|=count==0
        if not len(single):break
        indices=np.flatnonzero(~fixed)
        variables=np.unique(indices[remaining.indices[remaining.indptr[single]]])
        fixed[variables]=True;dropped[single]=True
        rounds.append(dict(singleton_rows=len(single),forced_zero_variables=len(variables)))
    reduced=a[~np.r_[dropped,np.zeros(a.shape[0]-neq,dtype=bool)]][:,~fixed]
    return dict(samples=rows,original_rows=a.shape[0],original_columns=n,original_nnz=a.nnz,
        observed_zero_fixed_bounds=int(original.sum()),singleton_forced_zero_variables=int((fixed&~original).sum()),
        removable_equalities=int(dropped.sum()),remaining_rows=reduced.shape[0],remaining_columns=reduced.shape[1],
        remaining_nnz=reduced.nnz,rounds=rounds,
        warning='Candidate presolve only. Bounds fixed in observed training data require runtime guards; no runtime model has been changed.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads((args.bank/'manifest.json').read_text())
    report=dict(bank=str(args.bank),data=str(args.data),stages={})
    for stage in manifest['stages']:
        root=checked_npz(args.bank/stage['stage'],'root.npz',stage['root_sha256'])
        report['stages'][stage['stage']]=audit(root,args.data/'inputs'/stage['stage'])
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
