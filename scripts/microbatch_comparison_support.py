"""Benchmark-only CPU comparator and cooperative host scheduling.

Only SciPy LP arrays enter worker threads; COBRA and environment state remain
on the calling thread. This does not alter production solver defaults.
"""
from concurrent.futures import ThreadPoolExecutor
import time
from greenlet import greenlet
from scipy.optimize import linprog


class ParallelCpuLP:
    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.history = []

    def solve_batch(self, requests):
        started = time.perf_counter()
        def solve(request):
            objective, kwargs = request
            kwargs = dict(kwargs)
            kwargs['options'] = dict(kwargs.get('options', {}), threads=1, parallel=False)
            return linprog(objective, **kwargs)
        results = list(self.pool.map(solve, requests))
        self.history.append(dict(batch=len(requests), cpu_lp_calls=len(requests),
                                 seconds=time.perf_counter()-started))
        return results

    def close(self):
        self.pool.shutdown(wait=True)


def drive_microbatch(environments, actions, service, snapshot, progress=None):
    """Run independent environments through ordered LP barriers on one thread."""
    def worker(i):
        env, model = environments[i]
        for action in actions[i]:
            env.step(action)
            if not model.stats.status.startswith('optimal;'):
                raise RuntimeError(model.stats.status)
            if progress is not None:progress[i]+=1
        return snapshot(env)
    workers = [greenlet(lambda i=i: worker(i)) for i in range(len(environments))]
    requests = [worker.switch() for worker in workers]
    while not all(worker.dead for worker in workers):
        if any(worker.dead for worker in workers):
            raise RuntimeError('Unequal LP stage counts')
        results = service.solve_batch(requests)
        if len(results) != len(workers) or not all(result.success for result in results):
            raise RuntimeError('An LP was rejected; no fallback')
        requests = [worker.switch(result) for worker, result in zip(workers, results)]
    return requests
