"""Input-only development microprofile, never a closed-loop speed claim."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.compact_training_data import checked_npz,sha256_file
from src.temporal_lp_data import load_trajectories
from src.gpu_certified_basis import CommunityCoordinates
from src.gpu_compact_basis import CompactBank
from src.gpu_heterogeneous_compact import HeterogeneousCompactBank
from src.coverage_router_binding import load_bound_coverage_router
from src.gpu_hybrid_lp import rank_compact_candidates
from src.lp_trace import problem_request
from src.cpu_repeated_lp import _problem
from src.gpu_compiled_community_backend import lp_arrays


def timed(call):
    start=time.perf_counter();result=call()
    return result,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--router',type=Path,required=True);p.add_argument('--router-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--repeats',type=int,default=12);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.batch_size,args.repeats)<1:raise ValueError('Positive sizes required')
    import cupy as cp
    trace,trajectories,trace_sha=load_trajectories(args.trace,'maxmin',required_role='development_diagnostic_not_training')
    # Drop reference solutions before ANY feature/rank/kernel calls.
    problems=[row[0] for trajectory in trajectories for row in trajectory][:args.batch_size]
    del trajectories
    if len(problems)!=args.batch_size:raise ValueError('Trace too short')
    from scripts.benchmark_basis_bank_rollout import environment
    from src.fba_surrogate import model_fingerprint
    sample,layout=environment(trace['seeds'][0])
    fingerprints={name:model_fingerprint(m) for name,m in sample.simulator.models.items()}
    if fingerprints!=trace['model_fingerprints']:raise ValueError('GEM mismatch')
    coordinates=CommunityCoordinates(dict(growth_terms=layout._growth_terms,exchange_terms=layout._exchange_terms,
        reaction_ids=[r.id for m in sample.simulator.models.values() for r in m.reactions],
        reaction_species=[name for name,m in sample.simulator.models.items() for r in m.reactions]),
        problems[0][0],problems[0][-1])
    manifest=json.loads((args.bank/'manifest.json').read_text());digest=sha256_file(args.bank/'manifest.json')
    if manifest['status']!='completed' or fingerprints!=manifest['model_fingerprints']:raise ValueError('Bank mismatch')
    if set(trace['seeds'])&set(manifest['train_seeds']):raise ValueError('Train/evaluation overlap')
    stage=next(s for s in manifest['stages'] if s['stage']=='maxmin')
    root=checked_npz(args.bank/'maxmin','root.npz',stage['root_sha256'])
    nearest=checked_npz(args.bank/'maxmin','router.npz',stage['router_sha256'])
    a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
    entries=[checked_npz(args.bank/'maxmin',e['filename'],e['sha256']) for e in stage['entries']]
    bank=CompactBank(dict(a=a,neq=int(root['neq'])),root['variable_rows'],entries,
        nearest['centers'],nearest['indices'],nearest['scale'],capture=True,selected_features=True,
        candidate_ranking='count')
    router,binding=load_bound_coverage_router(args.router,args.router_sha256,
        bank_manifest_sha256=digest,stage_manifest=stage,bank=bank,model_fingerprints=fingerprints,xp=cp)
    requests=[problem_request(problem,stage='maxmin') for problem in problems]
    # Live community assembly uses lists of pairs; restored traces use ndarray.
    # Keep both timings explicit instead of extrapolating ndarray unpack cost
    # to the live simulator.
    list_requests=[(c,dict(kw,bounds=[tuple(pair) for pair in kw['bounds']])) for c,kw in requests]
    report=dict(status='profiling',scope='Input-only maxmin development microbenchmark. No online CPU solve, physical state update, training, or end-to-end speed claim.',
        trace_sha256=trace_sha,bank_sha256=digest,router_binding=binding,batch_size=args.batch_size,
        input_rows='first batch, trajectory-major then step; not independent concurrent environments',
        source_hashes={name:sha256_file(ROOT/name) for name in ('scripts/profile_compact_dataflow.py',
            'src/gpu_heterogeneous_compact.py','src/gpu_compact_basis.py','src/gpu_certified_basis.py',
            'src/gpu_hybrid_lp.py','src/cpu_repeated_lp.py','src/gpu_compiled_community_backend.py','src/lp_bounds.py')},
        host_phases=[],gpu_modes=[])
    for i in range(args.repeats+1):
        _,cpu_parse=timed(lambda:[_problem(c,kw) for c,kw in requests])
        arrays,gpu_parse=timed(lambda:[lp_arrays(c,**kw) for c,kw in requests])
        _,cpu_parse_list=timed(lambda:[_problem(c,kw) for c,kw in list_requests])
        _,gpu_parse_list=timed(lambda:[lp_arrays(c,**kw) for c,kw in list_requests])
        normalized,normalize=timed(lambda:[coordinates.normalize(*v) for v in arrays])
        inputs,stack_upload=timed(lambda:bank.prepare_host(normalized))
        cp.cuda.get_current_stream().synchronize()
        (order,_),rank=timed(lambda:rank_compact_candidates(cp,bank,inputs,router))
        if i:report['host_phases'].append(dict(cpu_parse=cpu_parse,gpu_parse=gpu_parse,
            cpu_parse_list=cpu_parse_list,gpu_parse_list=gpu_parse_list,
            normalize=normalize,stack_upload_host=stack_upload,rank_with_download=rank))
    for k in (1,4):
        expected=None
        for certificate_only in (False,True):
            engine=HeterogeneousCompactBank(bank,certificate_only=certificate_only)
            ranks=cp.asarray(order[:,:k])
            def evaluate():
                result=engine.evaluate_replay(inputs,ranks)
                cp.cuda.get_current_stream().synchronize()
                return result
            result,cold=timed(evaluate)
            rows=[]
            for _ in range(args.repeats):
                start,end=cp.cuda.Event(),cp.cuda.Event()
                start.record();before=time.perf_counter()
                result=engine.evaluate_replay(inputs,ranks)
                end.record();end.synchronize()
                rows.append(dict(host_wall_seconds=time.perf_counter()-before,
                    device_seconds=cp.cuda.get_elapsed_time(start,end)/1000.))
            host={key:result[key].get() for key in ('accepted','candidate_index','values','objective',
                'primal_residual','dual_violation','relative_kkt_gap')}
            if expected is None:expected=host
            else:
                np.testing.assert_array_equal(host['accepted'],expected['accepted'])
                good=host['accepted']
                for name in host:
                    np.testing.assert_array_equal(host[name][good],expected[name][good])
            report['gpu_modes'].append(dict(k=k,certificate_only=certificate_only,cold_seconds=cold,
                graph_compilation_seconds=engine.graph_compilation_seconds,accepted=int(host['accepted'].sum()),
                measurements=rows,accepted_output_bitwise_equal=True))
            engine.close()
    report['status']='completed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(host_medians={key:float(np.median([r[key] for r in report['host_phases']]))
        for key in report['host_phases'][0]},gpu_modes=[dict(k=r['k'],certificate_only=r['certificate_only'],
        accepted=r['accepted'],host_median=float(np.median([v['host_wall_seconds'] for v in r['measurements']])),
        device_median=float(np.median([v['device_seconds'] for v in r['measurements']]))) for r in report['gpu_modes']]),indent=2),flush=True)


if __name__=='__main__':main()
