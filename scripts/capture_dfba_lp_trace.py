"""Capture certified CPU LPs through mid-trajectory for diagnostic replay.

The measured capture time includes compression and I/O; it is NOT a CPU
performance baseline. Full preroll LPs are retained so replay can warm start
causally, without using current reference solutions to initialize a solver.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
from greenlet import getcurrent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_basis_bank_rollout import environment, snapshot
from scripts.benchmark_compact_gpu import matched_actions, json_finite_values
from scripts.microbatch_comparison_support import drive_microbatch
from src.cpu_repeated_lp import RepeatedCpuLP, _problem
from src.fba_surrogate import model_fingerprint
from src.lp_trace import write_trace_lp, load_trace_lp, problem_hash
from src.graph_training_collection import (
    teacher_source_contract, require_matching_teacher_contract,
)


class CaptureBackend:
    def __init__(self, cpu, directory, manifest, save, *, cold_teacher=False):
        self.cpu, self.directory, self.manifest, self.save = cpu, directory, manifest, save
        self.cold_teacher=cold_teacher
        self.batch = 0

    def solve_batch(self, requests):
        # Snapshot mutable arrays before any solver or environment resumes.
        problems = [_problem(c, kw) for c, kw in requests]
        if self.cold_teacher:self.cpu.clear_models()
        results = self.cpu.solve_batch(requests)
        step, stage_index = self.batch // 3 + 1, self.batch % 3
        expected = ('maxmin', 'aggregate', 'exchange')[stage_index]
        for env_id, (problem, result) in enumerate(zip(problems, results)):
            if not result.success or result.diagnostics['stage'] != expected:
                raise RuntimeError('Uncertified or unexpected stage in trace')
            a, _, _, _, _, neq = problem
            key = (env_id, expected, *a.shape, neq)
            solution = self.cpu.models[key]['solver'].getSolution()
            path = self.directory / f'lp_{step:03d}_{stage_index}_{env_id:03d}.npz'
            entry = write_trace_lp(path, problem,
                reference_x=solution.col_value, reference_y=solution.row_dual)
            # Prove the on-disk roundtrip before adding it to the manifest.
            restored, x, y = load_trace_lp(self.directory, entry)
            if problem_hash(restored) != problem_hash(problem) or not np.array_equal(x, result.x):
                raise RuntimeError('Lossless trace roundtrip failed')
            entry.update(step=step, stage=expected, environment_id=env_id,
                         reference_diagnostics=result.diagnostics)
            self.manifest['entries'].append(entry)
        self.batch += 1
        if stage_index == 2 and step % 10 == 0:
            self.save()
            print(f'Captured {step} steps x {len(requests)} environments', flush=True)
        return results


def error_details(error, *, phase):
    return dict(error_phase=phase, error_type=type(error).__name__, error=str(error))


def describe_teacher_environment(env):
    sim = env.simulator
    observed = dict(simulator=f'{type(sim).__module__}.{type(sim).__name__}',
        fba_mode=sim.fba_mode, controller_dt_h=float(sim.dt),
        physical_profile=getattr(env, 'physical_profile', None),
        max_internal_dt=getattr(sim, 'max_internal_dt', None),
        numerics_version=getattr(sim, 'NUMERICS_VERSION', 'legacy'),
        solver_backend=sim.solver_backend)
    if (observed['simulator'] != 'src.dfba_simulator.dFBASimulator'
            or observed['fba_mode'] != 'cooperative' or observed['controller_dt_h'] != .2):
        raise ValueError('Teacher factory changed dynamics; this capture path is qualified only for legacy cooperative inputs')
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--environments', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20293401)
    parser.add_argument('--role', choices=['development_diagnostic_not_training',
        'training_reference','model_selection_reference'], default='development_diagnostic_not_training',
        help='Assign purpose at capture time; existing diagnostic traces stay excluded from training')
    parser.add_argument('--frozen-inputs', action='store_true')
    parser.add_argument('--design-profile',choices=['legacy','coverage_v1'],default='legacy')
    parser.add_argument('--design-offset',type=int,default=0)
    parser.add_argument('--cold-teacher',action='store_true',help='Rebuild every offline teacher LP; avoid mutable solver update path')
    parser.add_argument('--scipy-teacher',action='store_true',help='Offline serial cold SciPy solver; bypass persistent highspy backend')
    args = parser.parse_args()
    if not 1 <= args.steps <= 120 or min(args.environments, args.workers) < 1 or args.design_offset<0:
        raise ValueError('Invalid trace size')
    args.output.mkdir(parents=True, exist_ok=False)
    teacher_strategy = ('scipy_cold_serial' if args.scipy_teacher else
                        'cold_rebuild_each_lp_batch' if args.cold_teacher else 'persistent_reoptimization')
    def current_contract():
        return teacher_source_contract(ROOT, frozen_inputs=args.frozen_inputs,
            design_profile=args.design_profile, teacher_strategy=teacher_strategy)
    manifest = dict(status='initializing', role=args.role,
        scope='Lossless LP inputs, not an environment checkpoint or end-to-end speed benchmark',
        configuration=vars(args) | {'output':str(args.output)},
        entries=[], seeds=list(range(args.seed, args.seed+args.environments)))
    def save():
        temp = args.output/'manifest.json.tmp'
        temp.write_text(json.dumps(json_finite_values(manifest), indent=2, allow_nan=False))
        temp.replace(args.output/'manifest.json')
    lifecycle_started = time.perf_counter()
    capture_started = None
    cpu = None
    primary_error = None
    cleanup_error = None
    phase = 'initial_report'
    try:
        save()
        phase = 'source_snapshot'
        manifest['teacher_contract'] = current_contract()
        manifest['source_hashes'] = manifest['teacher_contract']['source_hashes']
        for f in (*manifest['source_hashes'], *manifest['teacher_contract']['model_source_hashes']):
            target = args.output/'sources'/f
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/f, target)
        phase = 'environment_setup'
        started = time.perf_counter()
        sample, layout = environment(args.seed)
        manifest['observed_environment'] = describe_teacher_environment(sample)
        manifest['model_fingerprints'] = {
            k:model_fingerprint(v) for k,v in sample.simulator.models.items()}
        envs = []
        designs=[];designed_actions=[]
        for seed in manifest['seeds']:
            env = copy.deepcopy(sample)
            env.reset(seed=seed)
            if args.design_profile=='coverage_v1':
                from src.trajectory_design import trajectory_design,apply_design
                design,action=trajectory_design(seed,args.design_offset+len(envs),args.steps)
                apply_design(env,design);designs.append(design);designed_actions.append(action)
            if args.frozen_inputs:
                env.simulator.enable_frozen_community_inputs()
            env.simulator._cooperative_solver._linprog_options.update(
                threads=1, parallel=False)
            envs.append((env, env.simulator._cooperative_solver))
        actions = designed_actions if designs else matched_actions(manifest['seeds'], args.steps)
        manifest['trajectory_designs']=designs
        manifest['initial_states']=[snapshot(env) for env,_ in envs]
        manifest['actions'] = [a.tolist() for a in actions]
        manifest['environment_setup_seconds'] = time.perf_counter()-started
        phase = 'cpu_initialization'
        if args.scipy_teacher:
            from src.offline_scipy_teacher import OfflineScipyTeacher
            cpu=OfflineScipyTeacher(n_fluxes=layout.n_fluxes)
        else:cpu = RepeatedCpuLP(args.workers, n_fluxes=layout.n_fluxes)
        manifest['teacher_strategy']=teacher_strategy
        capture = CaptureBackend(cpu, args.output, manifest, save,cold_teacher=args.cold_teacher)
        parent = getcurrent()
        capture_started = time.perf_counter()
        manifest['status'] = 'capturing'; save()
        phase = 'trajectory_capture'
        with patch('src.community_solver.linprog', lambda c, **kw:parent.switch((c, kw))):
            endpoints = drive_microbatch(envs, actions, capture, snapshot)
        if len(manifest['entries']) != 3*args.environments*args.steps:
            raise RuntimeError('Incomplete trace')
        phase = 'source_contract_verification'
        require_matching_teacher_contract(manifest['teacher_contract'], current_contract())
        manifest.update(status='completed', endpoints=endpoints,
                        completed_steps=[args.steps]*args.environments)
    except BaseException as error:
        primary_error = error
        manifest.update(status='failed', **error_details(error, phase=phase))
        raise
    finally:
        if capture_started is not None:
            manifest['capture_seconds_including_io'] = time.perf_counter()-capture_started
        manifest['lifecycle_seconds'] = time.perf_counter()-lifecycle_started
        if cpu is not None:
            try:
                cpu.close()
            except BaseException as error:
                cleanup_error = error
                manifest.setdefault('cleanup_errors', []).append(dict(
                    phase='cpu_close', error_type=type(error).__name__, error=str(error)))
                if primary_error is None:
                    manifest.update(status='failed', **error_details(error, phase='cpu_close'))
        try:
            save()
        except BaseException as save_error:
            # Preserve an already-active startup/capture/cleanup exception.
            if primary_error is None and cleanup_error is None:
                raise
            print(f'Failed to save final trace status: {save_error}', file=sys.stderr)
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


if __name__ == '__main__':
    main()
