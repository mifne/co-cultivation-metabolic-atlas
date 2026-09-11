"""Diagnostic host profile with LP service time excluded, not a speed benchmark."""
import argparse
import copy
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greenlet import getcurrent
from scripts.benchmark_basis_bank_rollout import environment, snapshot
from scripts.benchmark_compact_gpu import matched_actions, json_finite_values
from scripts.microbatch_comparison_support import drive_microbatch
from src.cpu_repeated_lp import RepeatedCpuLP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--environments', type=int, default=4)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20291001)
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--reference-host-source', type=Path,
        help='Archived source folder: use its objective and uptake methods for diagnostic equivalence')
    args = parser.parse_args()
    if min(args.environments, args.steps, args.cpu_workers) < 1 or args.steps > 120:
        raise ValueError('Invalid diagnostic size')
    profile_path = args.output.with_suffix('.pstats')
    if args.output.exists() or profile_path.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference_hashes = {}
    if args.reference_host_source:
        from src.community_solver import CooperativeCommunityFbaSolver
        from src.dfba_simulator import dFBASimulator
        for filename, target_class, source_class, method, static in (
            ('community_solver.py', CooperativeCommunityFbaSolver, 'CooperativeCommunityFbaSolver', '_objective_vector', True),
            ('community_solver.py', CooperativeCommunityFbaSolver, 'CooperativeCommunityFbaSolver', '_bounds', False),
            ('dfba_simulator.py', dFBASimulator, 'dFBASimulator', 'set_uptake_constraints', False),
            ('dfba_simulator.py', dFBASimulator, 'dFBASimulator', '_enforce_polymer_boundary_separation', False)):
            path = args.reference_host_source / filename
            spec = importlib.util.spec_from_file_location('src._profile_reference_' + filename[:-3], path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            implementation = getattr(getattr(module, source_class), method)
            setattr(target_class, method, staticmethod(implementation) if static else implementation)
            reference_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    started = time.perf_counter()
    sample, layout = environment(args.seed)
    environments = []
    for seed in range(args.seed, args.seed + args.environments):
        env = copy.deepcopy(sample)
        env.reset(seed=seed)
        model = env.simulator._cooperative_solver
        model._linprog_options.update(threads=1, parallel=False)
        environments.append((env, model))
    service = RepeatedCpuLP(args.cpu_workers, n_fluxes=layout.n_fluxes)
    setup_seconds = time.perf_counter() - started
    profiler = cProfile.Profile()
    original = service.solve_batch
    service_seconds = []
    request_hashes = []

    def solve(requests):
        profiler.disable()
        before = time.perf_counter()
        try:
            from src.cpu_repeated_lp import _problem
            digests = []
            for request in requests:
                a, rhs, lower, upper, c, neq = _problem(*request)
                digest = hashlib.sha256()
                for value in (np.asarray(a.shape), a.data, a.indices, a.indptr,
                              rhs, lower, upper, c, np.asarray([neq])):
                    digest.update(str((value.dtype.str, value.shape)).encode())
                    digest.update(value.tobytes())
                digests.append(digest.hexdigest())
            request_hashes.append(digests)
            return original(requests)
        finally:
            service_seconds.append(time.perf_counter() - before)
            profiler.enable()

    service.solve_batch = solve
    progress = [0] * args.environments
    parent = getcurrent()
    started = time.perf_counter()
    failure = None
    endpoints = []
    try:
        with patch('src.community_solver.linprog', lambda c, **kw: parent.switch((c, kw))):
            profiler.enable()
            endpoints = drive_microbatch(environments,
                matched_actions(list(range(args.seed, args.seed + args.environments)), args.steps),
                service, snapshot, progress)
    except Exception as error:
        failure = repr(error)
    finally:
        profiler.disable()
        elapsed = time.perf_counter() - started
        service.close()
    profiler.dump_stats(str(profile_path))
    stats = pstats.Stats(profiler)
    functions = [dict(file=file, line=line, function=name, primitive_calls=cc,
        calls=nc, self_seconds=tt, cumulative_seconds=ct)
        for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items()]
    functions.sort(key=lambda row: row['self_seconds'], reverse=True)
    report = dict(status='completed' if failure is None else 'failed', failure=failure,
        scope='Diagnostic cProfile on host; disabled inside LP service; profiling changes runtime',
        configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        setup_seconds=setup_seconds, profiled_wall_seconds=elapsed,
        lp_service_seconds=sum(service_seconds), profiled_host_wall_seconds=elapsed-sum(service_seconds),
        completed_steps=progress, endpoints=endpoints, cpu_history=service.history,
        request_hashes=request_hashes, reference_method_source_hashes=reference_hashes,
        functions=functions, source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('src/dfba_simulator.py','src/community_solver.py','src/rl_environment.py',
                         'src/cpu_repeated_lp.py','scripts/profile_dfba_host.py')})
    args.output.write_text(json.dumps(json_finite_values(report), indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('functions','cpu_history','endpoints','request_hashes')}, indent=2))
    for row in functions[:25]:
        print(f"{row['self_seconds']:8.3f} self  {row['cumulative_seconds']:8.3f} cumulative  {row['calls']:8d} calls  {row['file']}:{row['line']} {row['function']}")
    if failure:
        raise RuntimeError(failure)


if __name__ == '__main__':
    main()
