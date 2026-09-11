"""Bounded development diagnostics; not training, rollout, or PPO timing.

Reads only SHA-verified LP INPUTS. CPU reference x/y in the NPZ files are never
loaded. Graph mode measures RANDOM untrained inference, not model accuracy.
IPM mode has no CPU optimizer/fallback. Output is new and never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_coverage_temporal_routing import _load_problem_without_reference
from scripts.probe_downstream_gpu_coverage import select_entries, paired_certificate


def json_safe_diagnostics(value):
    """Preserve numerical failure diagnostics as explicit strings, not success/zero.

    Serialize before creating the output so an unexpected type cannot leave a
    half-written JSON. JSON has no IEEE NaN/infinity numeric literals.
    """
    if isinstance(value,dict):
        return {key:json_safe_diagnostics(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [json_safe_diagnostics(item) for item in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else ('Infinity' if value>0 else '-Infinity')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['graph', 'ipm', 'factor'], required=True)
    parser.add_argument('--trace', type=Path, default=ROOT/'results/pf_coverage_holdout4x120_20260905')
    parser.add_argument('--stage', choices=['maxmin', 'aggregate', 'exchange'], default='exchange')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--batch', type=int, choices=[1, 2, 4, 8, 16, 32], default=4)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--check-interval',type=int,default=1)
    parser.add_argument('--matrix-type', choices=['symmetric','general'], default='symmetric')
    parser.add_argument('--regularization',type=float,default=1e-9)
    parser.add_argument('--lp-scale',action='store_true')
    parser.add_argument('--condense',action='store_true')
    parser.add_argument('--zero-face',action='store_true')
    parser.add_argument('--forest',action='store_true')
    parser.add_argument('--second-forest',action='store_true')
    parser.add_argument('--fix-singleton-equalities',action='store_true')
    parser.add_argument('--original-newton',action='store_true')
    parser.add_argument('--refinements',type=int,default=2)
    parser.add_argument('--krylov',type=int,default=0)
    parser.add_argument('--krylov-coordinates',choices=['full','condensed'],default='full')
    parser.add_argument('--krylov-microkernels',choices=['none','mgs','all'],default='none')
    parser.add_argument('--device-checked-solves',action='store_true')
    parser.add_argument('--factor-refinements',type=int,choices=[0,1,2],default=2)
    parser.add_argument('--equality-row-scaling',action='store_true')
    parser.add_argument('--krylov-defer-lane-checks',action='store_true')
    parser.add_argument('--reuse-gmres-workspace',action='store_true')
    parser.add_argument('--globalized',action='store_true')
    parser.add_argument('--forcing-eta',type=float,default=0.05)
    parser.add_argument('--regularization-retries',type=int,default=0)
    parser.add_argument('--regularization-schedule',choices=['fixed','barrier'],default='fixed')
    parser.add_argument('--allow-box-dual',action='store_true')
    parser.add_argument('--exact-equalities',action='store_true')
    parser.add_argument('--bounded-near-equality',type=Path,
        help='Input-only relation diagnostic JSON; propose a bounded working relaxation, not an equivalent LP')
    parser.add_argument('--predictor-corrector',action='store_true')
    parser.add_argument('--predictor-affine-fraction',type=float,default=1.)
    parser.add_argument('--ipm-initialization',choices=['legacy','balanced'],default='legacy')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-python',action='store_true')
    parser.add_argument('--capture-failure',type=Path)
    args = parser.parse_args()
    if args.lp_scale and (args.condense or args.zero_face or args.forest):raise ValueError('Compare scaling and condensation separately first')
    if args.mode=='factor' and (args.condense or args.zero_face or args.forest):raise ValueError('Use full Newton matrix for the manufactured factor diagnostic')
    if args.allow_box_dual and (args.mode!='ipm' or not args.forest):
        raise ValueError('Analytic box dual is currently an explicit full-original forest IPM option')
    if args.exact_equalities and (args.mode!='ipm' or not (args.forest or args.zero_face)):
        raise ValueError('Exact equalities require zero-face or forest IPM')
    if args.second_forest and (args.mode!='ipm' or not (args.forest or args.zero_face)):
        raise ValueError('Second forest requires zero-face or forest IPM')
    if args.fix_singleton_equalities and (args.mode!='ipm' or not (args.forest or args.zero_face)):
        raise ValueError('Singleton equality propagation requires zero-face or forest IPM')
    if args.output.exists():
        raise FileExistsError('Diagnostics never overwrite existing results')
    if args.capture_failure and args.capture_failure.exists():
        raise FileExistsError('Failure snapshots never overwrite existing data')
    if args.mode!='ipm' and (args.profile_python or args.capture_failure):
        raise ValueError('Solver profiling and failure capture require IPM mode')
    near_candidate=None
    if args.bounded_near_equality:
        if args.mode!='ipm' or not args.forest:
            raise ValueError('Bounded near-equality diagnostic currently requires full-original forest IPM')
        from fractions import Fraction
        from src.lp_bounded_near_equality import NearEqualityCandidate
        relation=json.loads(args.bounded_near_equality.read_text())['exact_stored_data_relation']
        near_candidate=NearEqualityCandidate(row=relation['target_zero_face_row'],
            basis_rows=tuple(r['zero_face_row'] for r in relation['support']),
            coefficients=tuple(Fraction(r['coefficient']['numerator'],r['coefficient']['denominator'])
                               for r in relation['support']))
    manifest_path = args.trace/'manifest.json'
    load_start=time.perf_counter()
    manifest = json.loads(manifest_path.read_text())
    if args.batch>len(manifest['seeds']):
        raise ValueError('Batch must use distinct recorded environments; no duplicated LP padding')
    entries = [select_entries(manifest, args.stage, i, [args.step])[0] for i in range(args.batch)]
    problems = [_load_problem_without_reference(args.trace, e) for e in entries]
    input_load_seconds=time.perf_counter()-load_start
    m, n = problems[0][0].shape
    record = dict(mode=args.mode, role='development_diagnostic_not_training_or_rollout',
        stage=args.stage, step=args.step, batch=args.batch, rows=m, columns=n,
        configuration=dict(regularization=args.regularization,matrix_type=args.matrix_type,
            iterations=args.iterations,lp_scale=args.lp_scale,condense=args.condense,
            zero_face=args.zero_face,forest=args.forest,original_newton=args.original_newton,
            refinements=args.refinements,krylov=args.krylov,krylov_coordinates=args.krylov_coordinates,
            globalized=args.globalized,forcing_eta=args.forcing_eta,
            regularization_retries=args.regularization_retries,allow_box_dual=args.allow_box_dual,
            exact_equalities=args.exact_equalities,predictor_corrector=args.predictor_corrector,
            predictor_affine_fraction=args.predictor_affine_fraction,
            ipm_initialization=args.ipm_initialization),
        input_load_seconds=input_load_seconds,check_interval=args.check_interval,
        input_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        entries=entries, current_reference_vectors_loaded=False, cpu_lp_calls=0,
        sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('src/gpu_sparse_factor.py', 'src/gpu_batched_ipm.py', 'src/gpu_condensed_ipm.py',
             'src/lp_power_scaling.py','src/lp_zero_face.py','src/gpu_zero_face_ipm.py',
             'src/gpu_newton_krylov.py','src/gpu_forest_ipm.py','src/lp_equality_reduction.py',
             'src/graph_temporal_lp.py',
             'scripts/probe_graph_ipm.py')})
    record['source_snapshots']={p:(ROOT/p).read_text() for p in record['sources']}
    record['configuration']['regularization_schedule']=args.regularization_schedule
    record['configuration']['profile_python']=args.profile_python
    record['configuration']['krylov_microkernels']=args.krylov_microkernels
    record['configuration']['device_checked_solves']=args.device_checked_solves
    record['configuration']['factor_refinements']=args.factor_refinements
    record['configuration']['equality_row_scaling']=args.equality_row_scaling
    record['configuration']['krylov_defer_lane_checks']=args.krylov_defer_lane_checks
    record['configuration']['reuse_gmres_workspace']=args.reuse_gmres_workspace
    record['configuration']['second_forest']=args.second_forest
    record['configuration']['fix_singleton_equalities']=args.fix_singleton_equalities
    if args.second_forest:
        path='src/gpu_forest_map.py'
        record['sources'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        record['source_snapshots'][path]=(ROOT/path).read_text()
    if args.equality_row_scaling:
        path='src/gpu_ipm_equality_scaling.py'
        record['sources'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        record['source_snapshots'][path]=(ROOT/path).read_text()
    if args.bounded_near_equality:
        record['bounded_near_equality_proposal']=dict(path=str(args.bounded_near_equality),
            sha256=hashlib.sha256(args.bounded_near_equality.read_bytes()).hexdigest(),
            scope='Only proposed row IDs/weights loaded; each current input is exactly re-verified')
        for path in ('src/lp_bounded_near_equality.py','src/gpu_block_lp.py'):
            record['sources'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            record['source_snapshots'][path]=(ROOT/path).read_text()
    if args.krylov_microkernels!='none':
        path='src/gpu_krylov_microkernels.py'
        record['sources'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        record['source_snapshots'][path]=(ROOT/path).read_text()
    if args.globalized:
        for path in ('src/gpu_globalized_ipm.py','src/gpu_ipm_initialization.py'):
            record['sources'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            record['source_snapshots'][path]=(ROOT/path).read_text()
    if args.exact_equalities:
        path='src/lp_exact_equalities.py'
        record['sources'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        record['source_snapshots'][path]=(ROOT/path).read_text()
    if args.globalized and args.krylov_coordinates=='condensed':
        path='src/gpu_globalized_condensed.py'
        record['sources'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        record['source_snapshots'][path]=(ROOT/path).read_text()
    print(json.dumps(dict(status='starting', mode=args.mode, stage=args.stage,
                         step=args.step, batch=args.batch, rows=m, columns=n)), flush=True)
    if args.mode == 'graph':
        import torch
        from src.graph_temporal_lp import GraphTemporalLP, LPGraphBatch
        torch.manual_seed(20260905)
        torch.set_num_threads(1)
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        graph = LPGraphBatch.from_problems(problems, stage=args.stage,
            model_identity='development-trace:'+record['input_manifest_sha256'], device='cuda')
        torch.cuda.synchronize()
        record['graph_setup_seconds'] = time.perf_counter()-before
        record['union_edges'] = len(graph.row)
        record['device'] = torch.cuda.get_device_name()
        record['model_trained'] = False
        record['accuracy_claimed'] = False
        record['models'] = []
        for temporal in (False, True):
            model = GraphTemporalLP(hidden=32, rounds=2, temporal=temporal).cuda().eval()
            with torch.inference_mode():
                for _ in range(3):
                    proposal = model(graph)
                # Device events + synchronized wall both include complete forward.
                times, gpu_times = [], []
                for _ in range(10):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before = time.perf_counter()
                    start.record()
                    proposal = model(graph, proposal.state if temporal else None)
                    end.record()
                    end.synchronize()
                    times.append(time.perf_counter()-before)
                    gpu_times.append(start.elapsed_time(end)/1000.)
            record['models'].append(dict(temporal=temporal, parameters=sum(p.numel() for p in model.parameters()),
                synchronized_wall_seconds=times, cuda_event_seconds=gpu_times,
                all_outputs_finite=bool(proposal.x.isfinite().all() & proposal.y.isfinite().all()),
                x_shape=list(proposal.x.shape), y_shape=list(proposal.y.shape)))
            del proposal, model
        record['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    else:
        import cupy as cp
        from src.gpu_batched_ipm import GpuBatchedIPM
        if args.lp_scale:
            from src.lp_power_scaling import ScaledGpuBatchedIPM as GpuBatchedIPM
        if args.condense:
            from src.gpu_condensed_ipm import GpuCondensedBatchedIPM as GpuBatchedIPM
        if args.zero_face:
            from src.gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM as GpuBatchedIPM
        if args.forest:
            from src.gpu_forest_ipm import ForestGpuBatchedIPM as GpuBatchedIPM
        record['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        lifecycle_start=time.perf_counter()
        with GpuBatchedIPM(problems,matrix_type=args.matrix_type,regularization=args.regularization,
                          original_newton_target=args.original_newton,newton_refinements=args.refinements,
                          newton_krylov_iterations=args.krylov,krylov_coordinates=args.krylov_coordinates,
                          globalized=args.globalized,forcing_eta=args.forcing_eta,
                          regularization_retries=args.regularization_retries,
                          regularization_schedule=args.regularization_schedule,
                          device_checked_solves=args.device_checked_solves,
                          factor_refinements=args.factor_refinements,
                          equality_row_scaling=args.equality_row_scaling,
                          krylov_defer_lane_checks=args.krylov_defer_lane_checks,
                          reuse_gmres_workspace=args.reuse_gmres_workspace,
                          krylov_microkernels=args.krylov_microkernels,
                          predictor_corrector=args.predictor_corrector,
                          predictor_affine_fraction=args.predictor_affine_fraction,
                          ipm_initialization=args.ipm_initialization,
                          **({'allow_box_dual':True} if args.allow_box_dual else {}),
                          **({'bounded_near_equality':near_candidate} if near_candidate is not None else {}),
                          **({'second_forest':True} if args.second_forest else {}),
                          **({'fix_singleton_equalities':True} if args.fix_singleton_equalities else {}),
                          **({'exact_equalities':True} if args.exact_equalities else {})) as solver:
            record['constructor_wall_seconds']=time.perf_counter()-lifecycle_start
            print(json.dumps(dict(status='setup_complete', setup_seconds=solver.setup_seconds,
                kkt_dimension=solver.size, kkt_nnz=solver.factor.nnz)), flush=True)
            record.update(setup_seconds=solver.setup_seconds, kkt_dimension=solver.size,
                          kkt_nnz=solver.factor.nnz, cudss_version=solver.factor.version)
            if args.mode == 'factor':
                # Regularized INITIAL Newton matrix, not an LP solution.
                # Known manufactured vector gives a residual check without a CPU solve.
                from cupyx.scipy.sparse import csr_matrix as device_csr
                cp.random.seed(20260905)
                expected = cp.random.standard_normal((args.batch, solver.size, 1), dtype=cp.float64)
                rhs = cp.stack([device_csr((solver.values[i], solver.factor.indices, solver.factor.indptr),
                    shape=(solver.size, solver.size))@expected[i] for i in range(args.batch)])
                times = []
                for _ in range(3):
                    before = time.perf_counter()
                    solver.factor.factor(solver.values)
                    solution = solver.factor.solve(rhs)
                    cp.cuda.get_current_stream().synchronize()
                    times.append(time.perf_counter()-before)
                residuals = []
                for i in range(args.batch):
                    a = device_csr((solver.values[i], solver.factor.indices, solver.factor.indptr),
                                  shape=(solver.size, solver.size))
                    residuals.append(float(cp.max(cp.abs(a@solution[i]-rhs[i]))))
                record.update(factor_plus_solve_seconds=times, absolute_residual_max_by_env=residuals,
                    solution_error_max_by_env=cp.max(cp.abs(solution-expected), axis=(1, 2)).get().tolist(),
                    analysis_count=solver.factor.analysis_count, factor_count=solver.factor.factor_count,
                    interpretation='One regularized Newton matrix; not LP convergence or CPU speedup')
            else:
                profiler=None
                if args.profile_python:
                    import cProfile
                    profiler=cProfile.Profile()
                    profiler.enable()
                try:
                    result = solver.solve(iterations=args.iterations, check_interval=args.check_interval,
                        capture_failure=args.capture_failure is not None)
                finally:
                    if profiler is not None:
                        profiler.disable()
                if profiler is not None:
                    import pstats
                    stats=pstats.Stats(profiler)
                    rows=[dict(file=f[0],line=f[1],name=f[2],primitive_calls=v[0],calls=v[1],
                        own_seconds=v[2],cumulative_seconds=v[3]) for f,v in stats.stats.items()]
                    record['python_profile']=dict(scope='solve API only; Python wall includes CUDA waits, not GPU kernel time; instrumentation overhead included',
                        total_calls=stats.total_calls,primitive_calls=stats.prim_calls,total_seconds=stats.total_tt,
                        top_own=sorted(rows,key=lambda r:r['own_seconds'],reverse=True)[:100],
                        project_functions=[r for r in rows if '/co-cultivation/' in r['file']])
                transfer_start=time.perf_counter()
                x, y = result.pop('x').get(), result.pop('y').get()
                record['original_pair_d2h_seconds']=time.perf_counter()-transfer_start
                result['accepted'] = result['accepted'].tolist()
                result['accepted_iteration'] = result['accepted_iteration'].tolist()
                result.pop('reduced_x',None);result.pop('reduced_y',None)
                result.pop('forest_x',None);result.pop('forest_y',None)
                record['ipm'] = result
                audit_start=time.perf_counter()
                record['independent_host_original_certificates'] = [paired_certificate(p, xx, yy)
                    for p, xx, yy in zip(problems, x, y)]
                from src.lp_direct_dual_audit import audit_direct_dual
                record['independent_host_direct_dual_audits']=[
                    {key:value.tolist() for key,value in audit_direct_dual(p,xx,yy,xp=np).items()}
                    for p,xx,yy in zip(problems,x,y)]
                record['independent_host_audit_seconds']=time.perf_counter()-audit_start
                if args.capture_failure:
                    stamp=time.perf_counter()
                    payload={'returned_original_x':x,'returned_original_y':y}
                    if solver.failure_snapshot is not None:
                        payload.update({'failure_'+key:value.get() for key,value in solver.failure_snapshot.items()})
                    args.capture_failure.parent.mkdir(parents=True,exist_ok=True)
                    with args.capture_failure.open('xb') as stream:
                        np.savez(stream,**payload)
                    record['failure_capture']=dict(path=str(args.capture_failure),
                        sha256=hashlib.sha256(args.capture_failure.read_bytes()).hexdigest(),
                        first_failure_present=solver.failure_snapshot is not None,
                        arrays={key:list(value.shape) for key,value in payload.items()},
                        save_seconds=time.perf_counter()-stamp,
                        scope='current GPU iterates only; input trace reference vectors never loaded; snapshot persistence outside solve timing')
                print(json.dumps(dict(status=result['status'], accepted=result['accepted'],
                    solve_seconds=result['total_seconds'], metrics=result['metrics'])), flush=True)
        record['solver_lifecycle_wall_seconds']=time.perf_counter()-lifecycle_start
        record['lifecycle_scope']='constructor, solve, original-pair D2H, independent host audit and close; excludes imports/input loading/output serialization; frozen LP inputs, not dFBA'
        record['cupy_pool_used_bytes_after_close'] = cp.get_default_memory_pool().used_bytes()
    record['completed'] = True
    record['nonfinite_serialization']='IEEE nonfinite diagnostic values encoded as NaN/Infinity/-Infinity strings; never acceptance'
    serialized=json.dumps(json_safe_diagnostics(record),indent=2,allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(serialized)
    print('Saved '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
