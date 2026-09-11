"""Development-only diagnosis: wrong basis versus unstable compact expansion.

Re-factor the SAME saved combinatorial basis at the current normalized matrix
using CPU sparse LU, without LP optimization, pivot changes, CPU reference x/y,
or simulation updates. A success may support improving the numerical expansion;
a failure does not prove that no nearby basis or accurate solution exists.
This is not an online GPU fallback or a performance comparison.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import (
    select_entries, paired_certificate, original_candidates, _json_safe, LIMITS)
from src.compact_training_data import checked_child, checked_npz, sha256_file


def refactor_same_basis(original, normalized, arrays, *, refinement_steps=2):
    """Interpret exactly the compact map's basic/active/kind at current A."""
    if isinstance(refinement_steps,bool) or not isinstance(refinement_steps,int) or refinement_steps<0:
        raise ValueError('Nonnegative integer refinement steps required')
    m,n=normalized.a.shape
    basic,active,kind=(np.asarray(arrays[k]) for k in ('basic','active','kind'))
    for ids,limit in ((basic,n),(active,m)):
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids)
                or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=limit)):
            raise ValueError('Invalid basis index vector')
    if (len(basic)!=len(active) or kind.shape!=(n,) or kind.dtype.kind not in 'iu'
            or not np.isin(kind,[-1,0,1,2]).all()
            or not np.array_equal(np.sort(basic),np.flatnonzero(kind==2))):
        raise ValueError('Invalid compact combinatorial basis')
    if original[0].shape!=(m,n) or original[-1]!=normalized.neq:
        raise ValueError('Original and normalized LP structures disagree')
    record=dict(status='running',success=False,certificates=[],lp_optimizer_calls=0,
                basis_changes=0,reference_vectors_read=False)
    started=time.perf_counter()
    x=np.where(kind==-1,normalized.lower,np.where(kind==1,normalized.upper,0.)).astype(float)
    if not np.isfinite(x).all():
        return dict(record,status='nonfinite_nonbasic_bound',total_seconds=time.perf_counter()-started)
    a=normalized.a; b=a[active][:,basic].tocsc()
    before=time.perf_counter()
    try:
        factor=splu(b)
    except (RuntimeError,ValueError) as error:
        return dict(record,status='singular_or_invalid_basis',error=str(error),
                    total_seconds=time.perf_counter()-started)
    record['factor_seconds']=time.perf_counter()-before
    x[basic]=factor.solve(np.asarray(normalized.rhs[active]-a[active]@x))
    y=np.zeros(m);y[active]=factor.solve(normalized.c[basic],trans='T')
    for iteration in range(refinement_steps+1):
        original_x=x/normalized.col_scale;original_y=y*normalized.row_scale
        certificate=paired_certificate(original,original_x,original_y)
        record['certificates'].append(dict(refinement=iteration,**certificate))
        if certificate['certificate_passed']:
            record['success']=True
            break
        if iteration<refinement_steps:
            x[basic]+=factor.solve(np.asarray(normalized.rhs[active]-a[active]@x))
            y[active]+=factor.solve(np.asarray(normalized.c[basic]-a[:,basic].T@y),trans='T')
    record.update(status='certified' if record['success'] else 'original_certificate_failed',
                  maximum_absolute_original_x=float(np.max(np.abs(original_x))),
                  total_seconds=time.perf_counter()-started)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--trace-sha256',required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--bank-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--environment-id',type=int,default=0)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    trace=json.loads(checked_child(args.trace,'manifest.json',args.trace_sha256).read_text())
    manifest=json.loads(checked_child(args.bank,'manifest.json',args.bank_sha256).read_text())
    if (manifest.get('status')!='completed'
            or manifest['model_fingerprints']!=trace['model_fingerprints']
            or set(manifest.get('train_seeds',[]))&set(trace['seeds'])):
        raise ValueError('Completed matching bank and disjoint diagnostic seeds required')
    selected={stage:select_entries(trace,stage,args.environment_id,args.steps)
              for stage in ('aggregate','exchange')}
    first=_load_problem_without_reference(args.trace,
        select_entries(trace,'maxmin',args.environment_id,[args.steps[0]])[0])
    report=dict(status='running',scope=__doc__,training_performed=False,reference_vectors_read=False,
        cpu_lp_optimizer_calls=0,speed_comparison_valid=False,stages=[],
        trace_sha256=args.trace_sha256,bank_sha256=args.bank_sha256,certificate_limits=LIMITS,
        source_hashes={name:sha256_file(ROOT/name) for name in (
            'scripts/probe_compact_basis_refactor.py','scripts/probe_downstream_gpu_coverage.py',
            'scripts/analyze_coverage_temporal_routing.py','src/gpu_compact_basis.py',
            'src/gpu_heterogeneous_compact.py','src/gpu_certified_basis.py','src/cpu_repeated_lp.py')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False))
    save()
    try:
        with ExitStack() as cleanup:
            import cupy as cp
            from scripts.benchmark_basis_bank_rollout import environment
            from src.fba_surrogate import model_fingerprint
            from src.gpu_certified_basis import CommunityCoordinates
            from src.gpu_compact_basis import CompactBank
            from src.gpu_heterogeneous_compact import HeterogeneousCompactBank
            sample,layout=environment(trace['seeds'][args.environment_id]);cleanup.callback(sample.close)
            if {name:model_fingerprint(model) for name,model in sample.simulator.models.items()}!=trace['model_fingerprints']:
                raise ValueError('Runtime GEMs differ from the diagnostic trace')
            coordinates=CommunityCoordinates(dict(growth_terms=layout._growth_terms,
                exchange_terms=layout._exchange_terms,
                reaction_ids=[r.id for model in sample.simulator.models.values() for r in model.reactions],
                reaction_species=[name for name,model in sample.simulator.models.items() for r in model.reactions]),
                first[0],first[-1])
            for name,entries in selected.items():
                stage=next(s for s in manifest['stages'] if s['stage']==name);folder=args.bank/name
                root=checked_npz(folder,'root.npz',stage['root_sha256'])
                nearest=checked_npz(folder,'router.npz',stage['router_sha256'])
                maps=[checked_npz(folder,e['filename'],e['sha256']) for e in stage['entries']]
                a=csr_matrix((root['a_data'],root['a_indices'],root['a_indptr']),shape=tuple(root['a_shape']))
                bank=CompactBank(dict(a=a,neq=int(root['neq'])),root['variable_rows'],maps,
                    nearest['centers'],nearest['indices'],nearest['scale'],capture=True)
                engine=HeterogeneousCompactBank(bank,certificate_only=False)
                cleanup.callback(bank.math.close);cleanup.callback(bank.clear_graph_cache);cleanup.callback(engine.close)
                out_stage=dict(stage=name,candidates=len(maps),rows=[]);report['stages'].append(out_stage)
                for entry in entries:
                    original=_load_problem_without_reference(args.trace,entry)
                    if stage['key']!=[name,*original[0].shape,original[-1]]:raise ValueError('LP stage mismatch')
                    normalized=coordinates.normalize(*original);inputs=bank.prepare_host([normalized])
                    row=dict(step=entry['step'],environment_id=entry['environment_id'],
                        input_file_sha256=entry['sha256'],problem_sha256=entry['problem_sha256'],candidates=[])
                    out_stage['rows'].append(row)
                    for offset in range(0,len(maps),8):
                        ids=list(range(offset,min(offset+8,len(maps))))
                        raw,_,_,_=engine._candidates(inputs,cp.asarray(ids,dtype=cp.int32)[None])
                        x,y=original_candidates(raw['raw_values'].get(),raw['raw_y'].get(),
                            normalized.col_scale,normalized.row_scale)
                        family=raw['input_family_valid'].get();gpu=raw['accepted'].get()
                        for i,index in enumerate(ids):
                            direct=refactor_same_basis(original,normalized,maps[index])
                            row['candidates'].append(dict(candidate_id=index,family_valid=bool(family[i]),
                                gpu_certificate_passed=bool(gpu[i]),
                                compact_original_pair=paired_certificate(original,x[i],y[i]),refactor=direct))
                    row.update(gpu_certified=sum(r['gpu_certificate_passed'] for r in row['candidates']),
                        refactor_certified=sum(r['refactor']['success'] for r in row['candidates']),
                        refactor_rescued=sum(r['refactor']['success'] and not r['gpu_certificate_passed']
                            for r in row['candidates']))
                    save();print(json.dumps(dict(stage=name,step=entry['step'],gpu=row['gpu_certified'],
                        refactor=row['refactor_certified'],rescued=row['refactor_rescued'])),flush=True)
                engine.close();bank.clear_graph_cache()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    report['status']='completed';save()


if __name__=='__main__':main()
