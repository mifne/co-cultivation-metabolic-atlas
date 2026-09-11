"""Read-only candidate audit on a recorded failed LP, never training data."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_compact_gpu import checked_npz
from src.gpu_compact_basis import CompactBank


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--stage',default='aggregate');p.add_argument('--row',type=int,default=5)
    p.add_argument('--operators',type=Path);p.add_argument('--repair-top',type=int,default=0)
    p.add_argument('--refinement',type=int,nargs='+',default=[0,1]);p.add_argument('--pivots',type=int,default=128)
    p.add_argument('--compact-updates',action='store_true')
    p.add_argument('--reuse-small-factor',action='store_true')
    p.add_argument('--skip-unused-dual',action='store_true')
    p.add_argument('--adaptive-refinement-tolerance',type=float,default=0.)
    p.add_argument('--candidate-indices',type=int,nargs='+')
    args=p.parse_args()
    import cupy as cp
    manifest=json.loads((args.bank/'manifest.json').read_text());stage=next(s for s in manifest['stages'] if s['stage']==args.stage)
    folder=args.bank/args.stage;root=checked_npz(folder,'root.npz',stage['root_sha256']);router=checked_npz(folder,'router.npz',stage['router_sha256'])
    a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
    entries=[checked_npz(folder,e['filename'],e['sha256']) for e in stage['entries']]
    bank=CompactBank(dict(a=a,neq=int(root['neq'])),root['variable_rows'],entries,router['centers'],router['indices'],router['scale'])
    with np.load(args.inputs,allow_pickle=False) as d:inputs={k:cp.asarray(d[k][args.row:args.row+1]) for k in d.files}
    rows=[]
    for i,e in enumerate(bank.evaluators):
        out=e.evaluate(inputs);x=out['raw_values'];rs=inputs['row_scale'];cs=inputs['col_scale']
        bounds=cp.maximum(inputs['lower']-x,x-inputs['upper'])/cs
        inequalities=((out['activity']-inputs['rhs'])/rs)[:,bank.neq:]
        rows.append(dict(index=i,primal=float(out['primal_residual'].get()[0]),dual=float(out['basis_dual_violation'].get()[0]),
            bound_count=int(cp.sum(bounds>1e-5).get()),row_count=int(cp.sum(inequalities>1e-5).get()),
            l1=float((cp.maximum(bounds,0.).sum()+cp.maximum(inequalities,0.).sum()).get())))
    ranked=sorted(rows,key=lambda r:(r['dual']>1e-8,r['bound_count']+r['row_count'],r['l1']))
    print(json.dumps(ranked[:20],indent=1),flush=True)
    bank.candidate_ranking='count';bank.full_batch_candidates=True
    screened=bank.evaluate_device(inputs)
    print('screened',screened['best_candidate_index'].get().tolist(),'rank',bank.rank().get().tolist(),
        'scores',bank.last_candidate_scores.get().tolist(),flush=True)
    if args.repair_top:
        from src.gpu_revised_basis import GpuRevisedBasis
        from unittest.mock import patch
        import highspy,gc
        operators=json.loads((args.operators/'manifest.json').read_text())
        for entry in operators['entries']:
            if entry['stage']==args.stage:
                bank.evaluators[entry['index']].repair_path=(args.operators/entry['filename'],entry['sha256'])
        def forbidden(*a,**kw):raise AssertionError('Online CPU LP forbidden')
        selected=ranked[:args.repair_top] if args.candidate_indices is None else [next(r for r in rows if r['index']==i) for i in args.candidate_indices]
        for row in selected:
            for refinement in args.refinement:
                index=row['index'];adapter=bank.evaluators[index].repair_adapter()
                solver=GpuRevisedBasis(adapter.anchor,adapter.variable_rows,max_pivots=args.pivots,
                    capture_safe=True,adapter=adapter,dual_edge='devex',pivot_refinement=refinement,compact_updates=args.compact_updates,
                    reuse_small_factor=args.reuse_small_factor,skip_unused_dual=args.skip_unused_dual,
                    adaptive_refinement_tolerance=args.adaptive_refinement_tolerance)
                query=dict(inputs,delta=inputs['delta']-bank.evaluators[index].d['offset'])
                cp.cuda.get_current_stream().synchronize();started=time.perf_counter()
                with patch.object(highspy.Highs,'run',forbidden):result=solver.run_device(**query)
                cp.cuda.get_current_stream().synchronize()
                print(json.dumps(dict(index=index,refinement=refinement,seconds=time.perf_counter()-started,
                    **{k:v.get().tolist() for k,v in result.items() if k in ('accepted','primal_residual','dual_violation','relative_kkt_gap','pivots','unique_update_columns')})),flush=True)
                solver.clear_graph_cache();del result,solver,adapter;gc.collect();cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':main()
