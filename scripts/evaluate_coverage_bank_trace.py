"""Independent trace coverage only; CPU reference solutions never rank proposals."""
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compact_training_data import checked_npz,sha256_file
from src.compact_bank_coverage import measure_compact_bank_coverage,save_compact_coverage_npz,summarize_routing_coverage
from src.coverage_router import coverage_metrics
from src.coverage_router_binding import load_bound_coverage_router
from src.gpu_compact_basis import CompactBank
from src.gpu_certified_basis import CommunityCoordinates
from src.temporal_lp_data import load_trajectories


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--bank',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--router',type=Path);p.add_argument('--router-sha256');p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    trace,trajectories,trace_sha=load_trajectories(args.trace,'maxmin',required_role='development_diagnostic_not_training')
    # Reference x/y are deliberately discarded before proposal generation.
    problems=[row[0] for trajectory in trajectories for row in trajectory];del trajectories
    from scripts.benchmark_basis_bank_rollout import environment
    from src.fba_surrogate import model_fingerprint
    sample,layout=environment(trace['seeds'][0])
    fingerprints={name:model_fingerprint(model) for name,model in sample.simulator.models.items()}
    if fingerprints!=trace['model_fingerprints']:raise ValueError('Current GEMs do not match trace')
    metadata=dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for model in sample.simulator.models.values() for r in model.reactions],
        reaction_species=[name for name,model in sample.simulator.models.items() for r in model.reactions])
    coordinates=CommunityCoordinates(metadata,problems[0][0],problems[0][-1])
    normalized=[coordinates.normalize(*problem) for problem in problems]
    report=dict(status='evaluating',trace_sha256=trace_sha,seeds=trace['seeds'],results=[],
        scope='Held-out-input strict GPU certificate coverage, not closed-loop or speedup. No references in proposals.')
    for bank_index,directory in enumerate(args.bank):
        manifest=json.loads((directory/'manifest.json').read_text());digest=sha256_file(directory/'manifest.json')
        if manifest['status']!='completed' or fingerprints!=manifest['model_fingerprints']:raise ValueError('Bank provenance mismatch')
        if set(trace['seeds'])&set(manifest['train_seeds']):raise ValueError('Trace/bank training overlap')
        stage=next(s for s in manifest['stages'] if s['stage']=='maxmin')
        root=checked_npz(directory/'maxmin','root.npz',stage['root_sha256'])
        nearest=checked_npz(directory/'maxmin','router.npz',stage['router_sha256'])
        a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
        entries=[checked_npz(directory/'maxmin',e['filename'],e['sha256']) for e in stage['entries']]
        bank=CompactBank(dict(a=a,neq=int(root['neq'])),root['variable_rows'],entries,
            nearest['centers'],nearest['indices'],nearest['scale'],capture=False,selected_features=True)
        features=[]
        def batches():
            for i in range(0,len(normalized),args.batch_size):
                inputs=bank.prepare_host(normalized[i:i+args.batch_size])
                features.append(bank.proposal_features(inputs).get())
                yield inputs
        started=time.perf_counter()
        result=measure_compact_bank_coverage(bank,batches(),routing_topk=(1,2,4))
        save_compact_coverage_npz(args.output/f'bank_{bank_index}_coverage.npz',result,
            dict(trace_sha256=trace_sha,bank_sha256=digest,seeds=trace['seeds'],role='evaluation_only'))
        row=dict(bank=str(directory),bank_sha256=digest,nearest=summarize_routing_coverage(result),
            diagnostic_seconds=time.perf_counter()-started)
        if args.router and bank_index==len(args.bank)-1:
            router,binding=load_bound_coverage_router(args.router,args.router_sha256,bank_manifest_sha256=digest,
                stage_manifest=stage,bank=bank,model_fingerprints=fingerprints,xp=bank.cp)
            rank=router.rank(bank.cp.asarray(np.concatenate(features)),k=4).get()
            row.update(learned=coverage_metrics(result.coverage,rank,top_k=(1,2,4)),router_binding=binding)
        report['results'].append(row)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(row,indent=2),flush=True)
        del bank,entries;gc.collect()
    report['status']='completed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
