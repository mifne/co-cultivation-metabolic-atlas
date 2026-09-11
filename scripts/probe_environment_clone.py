"""Check independent, resettable benchmark environment cloning."""
import sys,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.benchmark_basis_bank_rollout import environment,snapshot
from src.fba_surrogate import model_fingerprint
source,layout=environment(20286411)
before={k:model_fingerprint(m) for k,m in source.simulator.models.items()}
started=time.perf_counter();clone=copy.deepcopy(source);clone.reset(seed=20286411)
print('clone_seconds',time.perf_counter()-started,flush=True)
assert {k:model_fingerprint(m) for k,m in clone.simulator.models.items()}==before
assert all(clone.simulator.models[k] is not source.simulator.models[k] for k in before)
assert snapshot(clone)==snapshot(source)
action=np.random.default_rng(20286411).uniform(.05,.95,(120,5)).astype(np.float32)[0]
source.step(action);clone.step(action)
assert snapshot(clone)==snapshot(source)
assert clone.simulator._cooperative_solver.stats.status.startswith('optimal;')
print('independent clone matched complete CPU step',flush=True)
