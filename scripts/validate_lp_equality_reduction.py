"""Offline algebra/postsolve diagnostic, NOT a GPU speed/accuracy benchmark.

CPU reference solutions are used only here to test coordinate equivalence.
The reduced LP is also solved with persistent CPU HiGHS, then independently
certified in ORIGINAL coordinates. No neural model is fit on this trace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpu_repeated_lp import RepeatedCpuLP, _certificate
from src.lp_trace import load_trace_lp, problem_request
from src.lp_equality_reduction import HomogeneousEqualityReduction
from scripts.benchmark_compact_gpu import json_finite_values


def certificate(problem, x, y):
    solution = SimpleNamespace(col_value=x, row_dual=y, col_dual=problem[4]-problem[0].T@y,
                               value_valid=True, dual_valid=True)
    return _certificate(*problem, solution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 2, 41])
    parser.add_argument('--environments', type=int, default=4)
    parser.add_argument('--stages', nargs='+', choices=['maxmin','aggregate','exchange'],
                        default=['maxmin','aggregate','exchange'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.environments < 1 or not args.steps or min(args.steps) < 1 or len(set(args.steps)) != len(args.steps):
        raise ValueError('Positive environment count and unique positive steps required')
    report = dict(status='initializing', scope=__doc__, results=[],
                  configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(json_finite_values(report), indent=2, allow_nan=False))
        temporary.replace(args.output)
    try:
        save()
        manifest_path = args.trace/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] != 'completed' or len(manifest['seeds']) < args.environments:
            raise ValueError('Completed trace with requested environments required')
        report.update(trace_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      trace_role=manifest['role'],seeds=manifest['seeds'][:args.environments],
                      model_fingerprints=manifest['model_fingerprints'])
        entries = {(e['stage'],e['step'],e['environment_id']):e for e in manifest['entries']}
        if len(entries) != len(manifest['entries']):
            raise ValueError('Duplicate trace labels')
        source_dir = args.output.with_suffix('.sources')
        source_dir.mkdir(exist_ok=False)
        report['source_hashes'] = {}
        for f in ['scripts/validate_lp_equality_reduction.py','src/lp_equality_reduction.py',
                  'src/cpu_repeated_lp.py','src/lp_trace.py']:
            target = source_dir/f
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/f, target)
            report['source_hashes'][f] = hashlib.sha256(target.read_bytes()).hexdigest()
        report['status']='running'
        for stage in args.stages:
            plan = None
            backend = RepeatedCpuLP(min(4,args.environments))
            stage_result = dict(stage=stage, steps=[])
            try:
                for step in sorted(args.steps):
                    items = [load_trace_lp(args.trace, entries[(stage,step,env)])
                             for env in range(args.environments)]
                    if plan is None:
                        before = time.perf_counter()
                        plan = HomogeneousEqualityReduction.from_problem(items[0][0])
                        stage_result.update(plan_build_seconds=time.perf_counter()-before,
                            equality_fingerprint=plan.equality_fingerprint,
                            original_shape=list(items[0][0][0].shape),
                            eliminated_equalities=len(plan.eliminated_rows),
                            decoder_shape=list(plan.T.shape),decoder_nonzeros=plan.T.nnz,
                            dual_lift_nonzeros=plan.dual_lift_map.nnz)
                    reductions = [plan.reduce(p) for p,_,_ in items]
                    roundtrip = []
                    for reduced,(p,x,y) in zip(reductions,items):
                        z = plan.compress_primal(x)
                        yr = y[plan.kept_rows]
                        rp = reduced.problem
                        lifted_y = reduced.lift_dual(yr,rp[4]-rp[0].T@yr)
                        lifted_x = plan.expand_primal(z)
                        roundtrip.append(dict(certificate=certificate(p,lifted_x,lifted_y),
                            primal_roundtrip_error_max=float(np.max(np.abs(lifted_x-x),initial=0)),
                            objective_roundtrip_error=float(abs(p[4]@(lifted_x-x)))))
                    before = time.perf_counter()
                    solved = backend.solve_batch([problem_request(r.problem,stage=stage) for r in reductions])
                    seconds = time.perf_counter()-before
                    rows = []
                    for env,(reduced,(p,x,_),solution) in enumerate(zip(reductions,items,solved)):
                        row = dict(environment_id=env, reduced_cpu_accepted=solution.success,
                                   reference_roundtrip=roundtrip[env],reduced_shape=list(reduced.problem[0].shape),
                                   reduced_nonzeros=reduced.problem[0].nnz,
                                   cpu_diagnostics=solution.diagnostics)
                        if solution.success:
                            rp = reduced.problem
                            m,n = rp[0].shape
                            raw = backend.models[(env,stage,m,n,rp[-1])]['solver'].getSolution()
                            yr = np.asarray(raw.row_dual)
                            y = reduced.lift_dual(yr,rp[4]-rp[0].T@yr)
                            restored_x = plan.expand_primal(solution.x)
                            row.update(original_certificate=certificate(p,restored_x,y),
                                original_objective_abs_error=float(abs(p[4]@(restored_x-x))),
                                original_primal_difference_max=float(np.max(np.abs(restored_x-x),initial=0)))
                        rows.append(row)
                    stage_result['steps'].append(dict(step=step,diagnostic_cpu_solve_seconds=seconds,rows=rows))
                    passed = sum(r.get('original_certificate',{}).get('certificate_passed',False) for r in rows)
                    print(f'{stage} step {step}: restored CPU certificate {passed}/{len(rows)}',flush=True)
            finally:
                backend.close()
            report['results'].append(stage_result)
            save()
        rows = [r for stage in report['results'] for step in stage['steps'] for r in step['rows']]
        report.update(status='completed',requested=len(rows),
            reference_roundtrip_accepted=sum(r['reference_roundtrip']['certificate']['certificate_passed'] for r in rows),
            reduced_cpu_restored_accepted=sum(r.get('original_certificate',{}).get('certificate_passed',False) for r in rows),
            gpu_speed_claim=False,closed_loop_claim=False)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
