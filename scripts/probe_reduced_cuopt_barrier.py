"""Input-only reduced LP -> native cuOpt barrier -> original-pair audit.

This is a bounded development probe, NOT a PPO rollout or speedup claim.
ForestGpuBatchedIPM is borrowed ONLY to prepare/recover structural maps. Its
unused cuDSS symbolic analysis is deliberately included in setup/lifecycle
costs; its Python IPM solve/factor/triangular solve paths are never called.
cuOpt's Python solution API returns host vectors, so this is not a claim of
fully device-resident end-to-end execution. No stored/model x/y are loaded.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import select_entries,paired_certificate
from src.gpu_block_lp import GpuBlockLP,assemble_blocks,certify_blocks_device
from src.lp_direct_dual_audit import audit_direct_dual
from src.lp_trace import problem_hash


SOURCES=('scripts/probe_reduced_cuopt_barrier.py',
    'scripts/analyze_coverage_temporal_routing.py','scripts/probe_downstream_gpu_coverage.py',
    'src/gpu_block_lp.py','src/gpu_forest_ipm.py','src/gpu_zero_face_ipm.py',
    'src/gpu_condensed_ipm.py','src/gpu_batched_ipm.py','src/gpu_sparse_factor.py',
    'src/lp_equality_reduction.py','src/lp_zero_face.py','src/lp_exact_equalities.py',
    'src/lp_bounded_near_equality.py','src/lp_direct_dual_audit.py',
    'src/gpu_pdhg_corrector.py','src/cpu_repeated_lp.py','src/lp_bounds.py','src/lp_trace.py')


def json_safe(value):
    """JSON does not admit IEEE nonfinite numbers; preserve failure explicitly."""
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else ('Infinity' if value>0 else '-Infinity')
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unsupported diagnostic JSON type: {type(value).__name__}')


def write_exclusive(path,record):
    serialized=json.dumps(json_safe(record),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(path).open('x',encoding='utf-8') as handle:handle.write(serialized+'\n')


def load_inputs(trace,stage,step,batch):
    """Only the existing SHA-verified INPUT loader may open LP archives."""
    if batch not in (4,32) or type(step) is not int or step<1:
        raise ValueError('A positive step and batch 4 or 32 are required')
    trace=Path(trace).resolve()
    payload=(trace/'manifest.json').read_bytes()
    manifest=json.loads(payload)
    if batch>len(manifest.get('seeds',[])):
        raise ValueError('Batch requires distinct recorded environments; duplicated padding is forbidden')
    entries=[select_entries(manifest,stage,env,[step])[0] for env in range(batch)]
    problems=[_load_problem_without_reference(trace,entry) for entry in entries]
    return problems,dict(input_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        entries=entries,input_problem_sha256=[problem_hash(p) for p in problems],
        current_reference_vectors_loaded=False,model_vectors_loaded=False)


def load_near_proposal(path):
    """Load only proposed row IDs/exact rational weights, never model vectors."""
    if path is None:return None,None
    from src.lp_bounded_near_equality import NearEqualityCandidate
    path=Path(path).resolve()
    payload=path.read_bytes()
    relation=json.loads(payload)['exact_stored_data_relation']
    candidate=NearEqualityCandidate(row=relation['target_zero_face_row'],
        basis_rows=tuple(row['zero_face_row'] for row in relation['support']),
        coefficients=tuple(Fraction(row['coefficient']['numerator'],row['coefficient']['denominator'])
                           for row in relation['support']))
    return candidate,dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),
        scope='Untrusted row/weight proposal only; each current LP/bounds is re-verified; NOT exact equivalence')


def native_barrier(backend,problems):
    """Use GpuBlockLP's explicit native settings, retaining both raw outputs.

    Its existing _solve_problems wrapper intentionally discards uncertified
    x and all y. This probe uses the same DataModel/Solve interface directly
    so original-space postsolve/auditing is possible for a stopped solve too.
    """
    cp=backend.cp
    started=time.perf_counter()
    packed=assemble_blocks(problems)
    a,row_lower,rhs,lo,hi,c=packed
    model=backend.lp.DataModel()
    model.set_csr_constraint_matrix(a.data,a.indices.astype(np.int32),a.indptr.astype(np.int32))
    model.set_constraint_lower_bounds(row_lower)
    model.set_constraint_upper_bounds(rhs)
    model.set_variable_lower_bounds(lo)
    model.set_variable_upper_bounds(hi)
    model.set_objective_coefficients(c)
    model.set_maximize(False)
    setup_seconds=time.perf_counter()-started
    before=time.perf_counter()
    solution=backend.lp.Solve(model,backend.settings)
    cp.cuda.runtime.deviceSynchronize()
    solve_seconds=time.perf_counter()-before
    status=solution.get_termination_status().name.lower()
    solved_by=solution.get_solved_by().name.lower()
    gpu_confirmed=solved_by=='barrier'
    before=time.perf_counter()
    x=np.asarray(solution.get_primal_solution(),dtype=np.float64)
    y=np.asarray(solution.get_dual_solution(),dtype=np.float64)
    host_return_seconds=time.perf_counter()-before
    shape_ok=x.shape==c.shape and y.shape==rhs.shape
    record=dict(method='barrier',status=status,solved_by=solved_by,
        gpu_backend_confirmed=gpu_confirmed,cpu_lp_calls=0 if gpu_confirmed else None,
        cpu_lp_calls_scope='No Python CPU optimizer; method Barrier+crossover off+presolve 0, confirmed by get_solved_by',
        configuration=dict(method='Barrier',presolve=0,crossover=False,num_cpu_threads=1,
            time_limit_seconds=20.,optimality_tolerance=1e-9,relative_primal_tolerance=0.,
            augmented=1,barrier_iterative_refinement=1,cudss_deterministic=True),
        variables=a.shape[1],constraints=a.shape[0],nonzeros=a.nnz,
        model_and_block_setup_seconds=setup_seconds,solve_seconds=solve_seconds,
        host_return_seconds=host_return_seconds,cuopt_solve_seconds=float(solution.get_solve_time()),
        cuopt_stats=solution.get_lp_stats(),valid_shape=shape_ok,
        scope='Native GPU LP iterations; host setup/control and cuOpt host-return vectors remain')
    return packed,x,y,record


def direct_gate_from_audit(audit,*,xp):
    """Additional direct gap at the existing primal-objective scaling, 1e-7."""
    relative=xp.abs(audit['signed_gap'])/xp.maximum(1.,xp.abs(audit['primal_objective']))
    passed=(audit['finite_inputs']&audit['finite_arithmetic']&xp.isfinite(relative)
            &(relative<=1e-7)&(audit['dual_sign_violation']<=1e-7)
            &(audit['dual_bound_domain_violation']<=1e-7))
    return relative,passed


def audit_pairs(problems,x,y,*,xp):
    rows=[]
    for problem,cx,cy in zip(problems,x,y):
        values=audit_direct_dual(problem,cx,cy,xp=xp)
        relative,passed=direct_gate_from_audit(values,xp=xp)
        values.update(relative_direct_gap_primal_scale=relative,direct_dual_gate_passed=passed)
        rows.append({key:(value.get()[0].item() if hasattr(value,'get') else value[0].item())
                     for key,value in values.items()})
    return rows


def run_probe(problems,near_candidate,record):
    import cupy as cp
    from src.gpu_forest_ipm import ForestGpuBatchedIPM
    device=cp.cuda.runtime.getDevice()
    name=cp.cuda.runtime.getDeviceProperties(device)['name']
    record['device']=dict(id=device,name=name.decode() if isinstance(name,bytes) else str(name))
    lifecycle=time.perf_counter()
    with ForestGpuBatchedIPM(problems,exact_equalities=True,
            bounded_near_equality=near_candidate,allow_box_dual=False) as maps:
        record['map_setup_seconds_including_unused_cudss_analysis']=time.perf_counter()-lifecycle
        record['mapping']=dict(full_variables=maps.full_n,full_rows=maps.full_m,
            forest_variables=maps.original_n,forest_rows=maps.original_m,
            reduced_variables=maps.n,reduced_rows=maps.m,
            reduced_problem_sha256=[problem_hash(p) for p in maps.problems],
            exact_equality_plans=[p.summary for p in maps.equality_plans],
            bounded_near_plans=[p.summary for p in maps.near_equality_plans],
            forest_setup_timing=maps.forest_setup_timing,
            unused_cudss_analysis_count=maps.factor.analysis_count,
            unused_cudss_version=maps.factor.version)
        before=time.perf_counter()
        backend=GpuBlockLP(method='barrier',time_limit=20.,tolerance=1e-9,presolve=0)
        record['backend_settings_setup_seconds']=time.perf_counter()-before
        packed,host_x,host_y,native=native_barrier(backend,maps.problems)
        record['native']=native
        record['cpu_lp_calls']=native['cpu_lp_calls']
        record['accepted']=[False]*len(problems)
        if native['valid_shape']:
            before=time.perf_counter()
            reduced_x=cp.asarray(host_x).reshape(len(problems),maps.n)
            reduced_y=cp.asarray(host_y).reshape(len(problems),maps.m)
            cp.cuda.get_current_stream().synchronize()
            record['native_host_to_device_seconds']=time.perf_counter()-before
            before=time.perf_counter()
            forest_x,forest_y=maps.lift_device(reduced_x,reduced_y)
            full_x,full_y=maps.expand_forest_device(forest_x,forest_y)
            cp.cuda.get_current_stream().synchronize()
            record['gpu_two_stage_postsolve_seconds']=time.perf_counter()-before
            before=time.perf_counter()
            record['reduced_gpu_certificates']=certify_blocks_device(maps.problems,packed,
                reduced_x.ravel(),reduced_y.ravel(),cp=cp,require_direct_dual=True)
            record['original_gpu_certificates']=certify_blocks_device(maps.full_problems,
                maps._full_assembled,full_x.ravel(),full_y.ravel(),cp=cp,require_direct_dual=True)
            record['original_gpu_direct_audit']=audit_pairs(maps.full_problems,full_x,full_y,xp=cp)
            record['gpu_original_audit_seconds_including_summary']=time.perf_counter()-before
            before=time.perf_counter()
            original_x,original_y=full_x.get(),full_y.get()
            record['original_pair_d2h_seconds']=time.perf_counter()-before
            before=time.perf_counter()
            record['independent_host_original_certificates']=[paired_certificate(p,x,y)
                for p,x,y in zip(problems,original_x,original_y)]
            record['independent_host_direct_audit']=audit_pairs(problems,original_x,original_y,xp=np)
            record['independent_host_audit_seconds']=time.perf_counter()-before
            record['accepted']=[bool(native['gpu_backend_confirmed'] and native['status']=='optimal'
                and reduced['certificate_passed'] and gpu['certificate_passed']
                and direct['direct_dual_gate_passed'] and host['certificate_passed']
                and host_direct['direct_dual_gate_passed'])
                for reduced,gpu,direct,host,host_direct in zip(record['reduced_gpu_certificates'],
                    record['original_gpu_certificates'],record['original_gpu_direct_audit'],
                    record['independent_host_original_certificates'],record['independent_host_direct_audit'])]
        record['unused_ipm_numeric_factor_count']=maps.factor.factor_count
        record['unused_ipm_triangular_solve_count']=maps.factor.solve_count
        if maps.factor.factor_count or maps.factor.solve_count:
            raise RuntimeError('Unexpected Python IPM numerical work in native barrier probe')
    cp.cuda.get_current_stream().synchronize()
    record['solver_lifecycle_seconds_including_unused_cudss_setup_and_close']=time.perf_counter()-lifecycle
    record['all_accepted']=all(record['accepted'])
    record['completed']=True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--stage',choices=['maxmin','aggregate','exchange'],default='exchange')
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--batch',type=int,choices=[4,32],default=4)
    parser.add_argument('--near-proposal',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('Probe outputs are exclusive; no overwrite')
    started=time.perf_counter()
    problems,inputs=load_inputs(args.trace,args.stage,args.step,args.batch)
    candidate,near=load_near_proposal(args.near_proposal)
    record=dict(mode='reduced_native_cuopt_barrier',role='development_diagnostic_not_training_or_rollout',
        stage=args.stage,step=args.step,batch=args.batch,**inputs,near_proposal=near,
        input_load_seconds=time.perf_counter()-started,cpu_lp_calls=None,completed=False,
        time_limit_scope='20 seconds inside cuOpt only; setup/audits/close are additional wall time',
        performance_scope='Cold diagnostic including unused cuDSS symbolic setup; no CPU/PPO speedup claim')
    record['sources']={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCES}
    record['source_snapshots']={path:(ROOT/path).read_text() for path in SOURCES}
    print(json.dumps(dict(status='starting',stage=args.stage,step=args.step,batch=args.batch,
                           near_proposal=candidate is not None)),flush=True)
    try:
        run_probe(problems,candidate,record)
    except Exception as error:
        record.update(error=dict(type=type(error).__name__,message=str(error)),completed=False)
    record['diagnostic_wall_seconds_including_input_setup_solve_audits']=time.perf_counter()-started
    write_exclusive(args.output,record)
    print(json.dumps(json_safe(dict(completed=record['completed'],accepted=record.get('accepted'),
        native=record.get('native'),error=record.get('error'),output=str(args.output))),allow_nan=False),flush=True)
    return 0 if record['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
