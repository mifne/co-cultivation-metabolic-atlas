"""Read-only diagnosis of saved held-out LPs; never produces training data."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.gpu_certified_basis import NormalizedLP
from src.compiled_basis_artifact import load_anchor
from src.gpu_revised_basis import GpuRevisedBasis


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--row',type=int,default=0)
    p.add_argument('--anchor',type=Path,default=ROOT/'results/pf_basis_train20286311_4_compiled/anchor_0004.npz')
    p.add_argument('--pivots',type=int,default=64)
    p.add_argument('--dual-edge',choices=['dantzig','devex'],default='dantzig')
    p.add_argument('--sweep',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.input,allow_pickle=False) as d:
        i=str(args.row)+'_'
        a=csr_matrix((d[i+'a_data'],d[i+'a_indices'],d[i+'a_indptr']),shape=tuple(d[i+'a_shape']))
        q=NormalizedLP(a,*(d[i+k] for k in ('rhs','lower','upper','c')),int(d[i+'neq']),d[i+'col_scale'],d[i+'row_scale'])
    if args.sweep:
        paths=[]
        for directory in (ROOT/'results/pf_basis_train20286311_4_compiled',ROOT/'results/pf_aggregate_extension_20260904'):
            manifest=json.loads((directory/'manifest.json').read_text())
            paths.extend(directory/e['filename'] for e in manifest['entries'] if e['key'][0]=='aggregate')
    else:paths=[args.anchor]
    reports=[]
    for path in paths:
        report=probe(q,path,args)
        reports.append(report)
        args.output.write_text(json.dumps(reports if args.sweep else report,indent=2))


def probe(q,path,args):
    anchor=load_anchor(path)
    solver=GpuRevisedBasis(anchor,range(q.neq,q.neq+3),max_pivots=args.pivots,capture_safe=True,dual_edge=args.dual_edge)
    started=time.perf_counter()
    out=solver.solve_device(**solver.prepare_host([q]),diagnostics=True)
    result={k:v.get().tolist() for k,v in out.items() if k in ('accepted','pivots','primal_residual',
        'dual_violation','relative_kkt_gap','trace','objective_trace')}
    result.update(seconds=time.perf_counter()-started,scope='Held-out failure diagnosis, no training',
        input=str(args.input),anchor=str(path),row=args.row,dual_edge=args.dual_edge)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trace','objective_trace')}),flush=True)
    return result


if __name__=='__main__':main()
