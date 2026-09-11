"""Recorded-LP development probe; no teacher fitting or online CPU LP."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_compact_gpu import checked_npz
from src.gpu_compact_basis import CompactBank
from src.gpu_revised_basis import GpuRevisedBasis
from src.gpu_restricted_basis import GpuRestrictedBasis


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--operators',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',default='exchange');p.add_argument('--row',type=int,default=3);p.add_argument('--index',type=int,default=2)
    p.add_argument('--columns',type=int,nargs='+',default=[16,32,64,128]);p.add_argument('--pivots',type=int,default=512)
    p.add_argument('--adaptive',action='store_true')
    p.add_argument('--continue-basis',action='store_true',help='Keep each lifted basis and reprice a new small direction pool')
    p.add_argument('--memory-audit',action='store_true',help='Diagnostic GC/memory measurements; not fair timing evidence')
    p.add_argument('--rank-bucket',action='store_true')
    p.add_argument('--polish',type=int,default=0,help='Full GPU pivots after each primal-feasible rejected reduced result')
    args=p.parse_args()
    if args.adaptive and args.continue_basis:raise ValueError('Choose prefix enrichment or continued column generation')
    if args.output.exists():raise FileExistsError(args.output)
    import cupy as cp
    manifest=json.loads((args.bank/'manifest.json').read_text());stage=next(s for s in manifest['stages'] if s['stage']==args.stage)
    folder=args.bank/args.stage;root=checked_npz(folder,'root.npz',stage['root_sha256']);router=checked_npz(folder,'router.npz',stage['router_sha256'])
    a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
    entry=stage['entries'][args.index];arrays=checked_npz(folder,entry['filename'],entry['sha256'])
    bank=CompactBank(dict(a=a,neq=int(root['neq'])),root['variable_rows'],[arrays],router['centers'][:1],router['indices'],router['scale'])
    repair=json.loads((args.operators/'manifest.json').read_text())
    entry=next(e for e in repair['entries'] if e['stage']==args.stage and e['index']==args.index)
    bank.evaluators[0].repair_path=(args.operators/entry['filename'],entry['sha256'])
    adapter=bank.evaluators[0].repair_adapter()
    base=GpuRevisedBasis(adapter.anchor,adapter.variable_rows,max_pivots=384,adapter=adapter,
        capture_safe=True,compact_updates=True,reuse_small_factor=True,pivot_refinement=1,
        skip_unused_dual=True,adaptive_refinement_tolerance=1e-12)
    solver=GpuRestrictedBasis(base,max_pivots=args.pivots,rank_bucket=args.rank_bucket)
    with np.load(args.inputs,allow_pickle=False) as d:inputs={k:cp.asarray(d[k][args.row:args.row+1]) for k in d.files}
    inputs['delta']-=bank.evaluators[0].d['offset']
    report=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope='Recorded development input only; not independent end-to-end performance evidence',runs=[])
    def forbidden(*a,**kw):raise AssertionError('Online CPU optimizer prohibited')
    with patch('highspy.Highs.run',forbidden),patch('scipy.optimize.linprog',forbidden):
        selected=None;last=None
        for count in args.columns:
            if args.adaptive and last is not None:
                more=min(count,len(solver.nonbasic0))-selected.shape[1]
                if more<=0:continue
                new=solver.nonbasic0[cp.argsort(-last['expansion_scores'],axis=1)[:,:more]]
                selected=cp.concatenate((selected,new),axis=1)
            cp.cuda.get_current_stream().synchronize();started=time.perf_counter()
            result=solver.run_device(**inputs,columns=count,
                selected_columns=None if args.continue_basis else selected,
                warm_start=last['warm_state'] if args.continue_basis and last is not None else None,
                reuse_lifted_state=args.rank_bucket and args.continue_basis and last is not None)
            cp.cuda.get_current_stream().synchronize()
            row=dict(columns=count,seconds=time.perf_counter()-started,
                **{k:v.get().tolist() for k,v in result.items() if k in ('accepted','primal_residual','dual_violation','relative_kkt_gap','restricted_pivots','restricted_failed','restricted_lifted_dual_violation')})
            if args.polish and not bool(result['accepted'].all()) and bool((result['primal_residual']<=1e-5).all()):
                cp.cuda.get_current_stream().synchronize();started=time.perf_counter()
                polished=base.run_device(**inputs,warm_start=result['warm_state'],pivot_budget=args.polish)
                cp.cuda.get_current_stream().synchronize()
                row['polish']=dict(seconds=time.perf_counter()-started,
                    **{k:v.get().tolist() for k,v in polished.items() if k in ('accepted','primal_residual','dual_violation','relative_kkt_gap','pivots')})
            report['runs'].append(row);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
            selected=result['selected_columns'].copy();last=result
            if args.memory_audit:
                import gc
                def memory():
                    from cuda.bindings import runtime as rt
                    objects=gc.get_objects();calls=[o for o in objects if type(o).__name__=='GpuReplayCall']
                    free,total=cp.cuda.runtime.memGetInfo()
                    return dict(device_used=total-free,default_reserved=cp.get_default_memory_pool().total_bytes(),
                        graph_reserved=int(rt.cudaDeviceGetGraphMemAttribute(0,rt.cudaGraphMemAttributeType.cudaGraphMemAttrReservedMemCurrent)[1]),
                        replay_objects=len(calls),private_reserved=sum(o.pool.total_bytes() for o in calls),
                        private_used=sum(o.pool.used_bytes() for o in calls))
                before=memory();collected=gc.collect();after=memory()
                row['memory_audit']=dict(before=before,collected=collected,after=after)
                args.output.write_text(json.dumps(report,indent=2));print(json.dumps(row['memory_audit']),flush=True)
            if args.continue_basis and bool(result['accepted'].all()):break
    solver.clear_graph_cache();base.clear_graph_cache()


if __name__=='__main__':main()
