"""Cold same-time LP microbenchmark, NOT a dFBA trajectory speed comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpu_repeated_lp import RepeatedCpuLP
from src.gpu_block_lp import GpuBlockLP
from src.lp_trace import load_trace_lp, problem_request


def finite_json(value):
    if isinstance(value, dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def error_details(error, *, phase):
    return dict(error_phase=phase, error_type=type(error).__name__, error=str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 41])
    parser.add_argument('--stages', nargs='+', choices=['maxmin','aggregate','exchange'],
                        default=['maxmin','aggregate','exchange'])
    parser.add_argument('--method', choices=['pdlp','barrier'], default='pdlp')
    parser.add_argument('--time-limit', type=float, default=5.)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--presolve', choices=[0,2], type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--box-dual-certificate', action='store_true')
    parser.add_argument('--box-face', action='store_true', help='Single-objective-variable optimal-box-face feasibility proposal')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = ['src/gpu_block_lp.py','src/lp_trace.py','src/cpu_repeated_lp.py',
               str(Path(__file__).relative_to(ROOT))]
    report = dict(status='initializing', scope='Cold same-time independent LP replay; '
        'no environment updates, no temporal warm start, not an end-to-end speed benchmark',
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        timing_scope=dict(kind='cold_model_development_diagnostic',
            gpu_library_setup_in_ratio=False,
            gpu_backend_and_cuda_context_reused_across_rows=True,
            cpu_backend_recreated_for_each_row=True,
            cpu_worker_threads_start_lazily_inside_cpu_seconds=True,
            execution_order='cpu-first',
            inference='Descriptive timings only; not a persistent-CPU or trajectory speedup claim'),
        results=[])
    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False))
        temp.replace(args.output)
    primary_error = None
    phase = 'initial_report'
    try:
        save()
        phase = 'trace_manifest'
        manifest_path = args.trace/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] != 'completed' or manifest['role'] != 'development_diagnostic_not_training':
            raise ValueError('A completed development trace is required')
        if not args.steps or min(args.steps) < 1 or max(args.steps) > min(manifest['completed_steps']):
            raise ValueError('Trace does not contain requested steps')
        report.update(trace_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            model_fingerprints=manifest['model_fingerprints'], seeds=manifest['seeds'])
        phase = 'source_snapshot'
        source_dir = args.output.with_suffix('.sources')
        source_dir.mkdir(parents=True, exist_ok=False)
        for f in sources:
            dest = source_dir/f; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/f, dest)
        report['source_hashes'] = {
            f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
        phase = 'gpu_initialization'
        before = time.perf_counter()
        gpu = GpuBlockLP(method=args.method, time_limit=args.time_limit,
                     tolerance=args.tolerance, presolve=args.presolve,
                     box_dual_certificate=args.box_dual_certificate, box_face=args.box_face)
        gpu.cp.cuda.runtime.deviceSynchronize()
        report['gpu_library_setup_seconds'] = time.perf_counter()-before
        report['status'] = 'running'; save()
        for step in args.steps:
            for stage in args.stages:
                phase = f'replay_step_{step}_{stage}'
                entries = sorted([e for e in manifest['entries'] if e['step'] == step and e['stage'] == stage],
                                 key=lambda e:e['environment_id'])
                if [e['environment_id'] for e in entries] != list(range(len(manifest['seeds']))):
                    raise ValueError('Incomplete or duplicated environment cohort')
                loaded = [load_trace_lp(args.trace, e) for e in entries]
                problems = [item[0] for item in loaded]
                # Reference x/y are only used for scoring after both solvers.
                requests = [problem_request(p, stage=stage) for p in problems]
                cpu = None
                cpu_error = None
                try:
                    cpu = RepeatedCpuLP(args.workers)
                    before = time.perf_counter()
                    cpu_results = cpu.solve_batch(requests)
                    cpu_seconds = time.perf_counter()-before
                except BaseException as error:
                    cpu_error = error
                    raise
                finally:
                    if cpu is not None:
                        try:
                            cpu.close()
                        except BaseException as close_error:
                            close_phase = f'cpu_close_step_{step}_{stage}'
                            report.setdefault('cleanup_errors', []).append(dict(
                                phase=close_phase,
                                error_type=type(close_error).__name__, error=str(close_error)))
                            if cpu_error is None:
                                phase = close_phase
                                raise
                gpu_results = gpu.solve_batch(requests)
                row = dict(step=step, stage=stage, batch=len(entries),
                    execution_order='cpu-first', input_hashes=[e['problem_sha256'] for e in entries],
                    cpu_seconds=cpu_seconds, cpu_rows=[r.diagnostics for r in cpu_results],
                    gpu=gpu.history[-1], all_accepted=all(r.success for r in cpu_results+gpu_results))
                row['reference_objective_errors'] = [dict(
                    cpu=abs(cr.fun-float(p[4]@x)) if cr.success else None,
                    gpu=abs(gr.fun-float(p[4]@x)) if gr.success else None)
                    for (p,x,_),cr,gr in zip(loaded,cpu_results,gpu_results)]
                if row['all_accepted']:
                    row['cold_lp_cpu_over_gpu_ratio'] = cpu_seconds/row['gpu']['total_seconds']
                report['results'].append(row); save()
                print(f'{stage} step {step}: CPU {cpu_seconds:.3f}s, GPU '
                      f'{row["gpu"]["total_seconds"]:.3f}s, '
                      f'GPU certified {sum(r.success for r in gpu_results)}/{len(entries)}, '
                      f'{row["gpu"]["status"]}', flush=True)
        report['status'] = 'completed'
    except BaseException as error:
        primary_error = error
        report.update(status='failed', **error_details(error, phase=phase))
        raise
    finally:
        try:
            save()
        except BaseException as save_error:
            # A reporting failure must not replace the actual initialization,
            # solve, or close failure already propagating to the caller.
            if primary_error is None:
                raise
            print(f'Failed to save final benchmark status: {save_error}', file=sys.stderr)


if __name__ == '__main__':
    main()
