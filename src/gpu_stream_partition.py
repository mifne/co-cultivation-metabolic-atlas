"""Benchmark-only, same-host-thread cooperative CUDA stream partitioning.

Each workspace owns separate cuDSS handle/data/buffers and a fixed stream.
All constructors/analysis MUST finish serially before entry. No native API is
called from competing Python threads, and no context guard is bypassed.
Numerical helpers run unchanged before recording an event and yielding.
This scheduling experiment does not change any LP or numerical acceptance.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import threading
from types import MethodType

from greenlet import GreenletExit, getcurrent, greenlet


def partition_indices(batch_size, partitions):
    """Deterministic round-robin, exact coverage without duplicated padding."""
    if (type(batch_size) is not int or batch_size < 1 or type(partitions) is not int
            or partitions < 1 or partitions > batch_size or batch_size % partitions):
        raise ValueError('Positive integer batch evenly divisible by partitions required')
    return tuple(tuple(range(i, batch_size, partitions)) for i in range(partitions))


@contextmanager
def cooperative_hooks(solver, yield_callback, *, mode='factor-and-solve'):
    """Temporarily wrap only this instance; restore its exact lookup state."""
    if mode not in ('factor-only', 'factor-and-solve'):
        raise ValueError('Select factor-only or factor-and-solve')
    names = ['_factor_newton'] + (['_factor_solve'] if mode == 'factor-and-solve' else [])
    saved = []
    try:
        for name in names:
            existed = name in solver.__dict__
            old_instance_value = solver.__dict__.get(name)
            original = getattr(solver, name)
            if not callable(original):
                raise ValueError('A callable original numerical helper is required')
            def wrapped(self, *args, _original=original, _name=name, **kwargs):
                value = _original(*args, **kwargs)
                yield_callback(_name)
                return value
            saved.append((name, existed, old_instance_value))
            setattr(solver, name, MethodType(wrapped, solver))
        yield
    finally:
        for name, existed, old_value in reversed(saved):
            if existed:
                setattr(solver, name, old_value)
            else:
                delattr(solver, name)


@dataclass(frozen=True)
class _Pending:
    shard: int
    phase: str


def solve_cooperatively(workspaces, streams, *, solve_kwargs=None,
                        yield_mode='factor-and-solve', event_factory=None):
    """Run prebuilt workspaces on independent streams, using one owner thread.

    Returns ``(results_in_shard_order, scheduling_diagnostics)``. Event queries
    decide readiness, not numerical acceptance. At most one reusable event is
    pending per shard. When no shard is ready, wait for one event, never the
    whole CUDA device. Each resume restores the shard stream because CuPy's
    current stream is thread-local, not greenlet-local.

    Workspaces remain caller-owned. Success and exceptions both drain streams;
    exceptions also unwind suspended greenlets and restore helper methods.
    The caller MUST subsequently close every workspace in its original stream.
    Individual solver timing fields include time yielded to other shards and
    cannot be summed or interpreted as exclusive GPU phase/kernel time.

    ``event_factory`` is injectable for CPU scheduler tests. Production uses
    CuPy ``Event(disable_timing=True)`` with record(stream), done, synchronize().
    """
    workspaces, streams = tuple(workspaces), tuple(streams)
    if (not workspaces or len(workspaces) != len(streams)
            or len({id(w) for w in workspaces}) != len(workspaces)
            or len({id(w.factor) for w in workspaces}) != len(workspaces)
            or len({s.ptr for s in streams}) != len(streams)):
        raise ValueError('Distinct workspaces, factors and matching independent streams required')
    if yield_mode not in ('factor-only', 'factor-and-solve'):
        raise ValueError('Select factor-only or factor-and-solve')
    kwargs = {} if solve_kwargs is None else dict(solve_kwargs)
    if event_factory is None:
        import cupy as cp
        event_factory = lambda: cp.cuda.Event(disable_timing=True)
    owner_thread, parent = threading.get_ident(), getcurrent()
    for solver, stream in zip(workspaces, streams):
        if solver.factor.thread != owner_thread or solver.factor.stream.ptr != stream.ptr:
            raise ValueError('Workspaces must be created on this owner thread and assigned stream')
        with stream:
            solver.factor._context()
    events = []
    for stream in streams:
        with stream:
            events.append(event_factory())
    pending = {}
    results = [None] * len(workspaces)
    counts = [{'_factor_newton': 0, '_factor_solve': 0} for _ in workspaces]
    diagnostics = dict(execution_mode='same_owner_thread_greenlet_event_scheduling',
        owner_thread_id=owner_thread, stream_pointers=[int(s.ptr) for s in streams],
        partitions=len(workspaces), yield_mode=yield_mode, resumes=0,
        event_queries=0, event_waits=0, max_pending_shards=0, yield_counts=counts,
        cuDSS_host_calls_concurrently_from_multiple_threads=False,
        pending_overlap_is_not_proof_of_kernel_concurrency=True,
        per_solver_timings_include_yield_time=True)
    workers = []

    def yielded(shard, phase):
        if threading.get_ident() != owner_thread or getcurrent() is not workers[shard]:
            raise RuntimeError('Cooperative helper invoked outside its owner greenlet/thread')
        workspaces[shard].factor._context()
        if shard in pending:
            raise RuntimeError('Cannot overwrite an event while it is pending')
        events[shard].record(streams[shard])
        counts[shard][phase] += 1
        parent.switch(_Pending(shard, phase))
        # A resumed callback must still see the exact owner context.
        workspaces[shard].factor._context()

    def resume(shard):
        if threading.get_ident() != owner_thread:
            raise RuntimeError('Scheduler cannot migrate to a different host thread')
        with streams[shard]:
            workspaces[shard].factor._context()
            diagnostics['resumes'] += 1
            value = workers[shard].switch()
        if workers[shard].dead:
            results[shard] = value
        elif isinstance(value, _Pending) and value.shard == shard:
            pending[shard] = value
            diagnostics['max_pending_shards'] = max(
                diagnostics['max_pending_shards'], len(pending))
        else:
            raise RuntimeError('Unexpected yield from numerical workspace')

    error = None
    with ExitStack() as stack:
        for shard, solver in enumerate(workspaces):
            stack.enter_context(cooperative_hooks(solver,
                lambda phase, shard=shard: yielded(shard, phase), mode=yield_mode))
            workers.append(greenlet(lambda solver=solver: solver.solve(**kwargs), parent=parent))
        try:
            for shard in range(len(workspaces)):
                resume(shard)
            while pending:
                ready = []
                for shard in tuple(pending):
                    diagnostics['event_queries'] += 1
                    if events[shard].done:
                        ready.append(shard)
                if not ready:
                    shard = next(iter(pending))
                    events[shard].synchronize()
                    diagnostics['event_waits'] += 1
                    ready = [shard]
                for shard in ready:
                    pending.pop(shard)
                    resume(shard)
            if not all(worker.dead for worker in workers):
                raise RuntimeError('Live numerical workspace missing its pending event')
        except BaseException as caught:
            error = caught
        finally:
            cleanup_errors = []
            # Do not unwind suspended frames while their local device arrays
            # can still be referenced by enqueued kernels/cuDSS work.
            for stream in streams:
                try:
                    stream.synchronize()
                except BaseException as caught:
                    cleanup_errors.append(caught)
            cancelled = False
            for shard, worker in enumerate(workers):
                if not worker.dead:
                    cancelled = True
                    try:
                        with streams[shard]:
                            worker.throw(GreenletExit)
                    except BaseException as caught:
                        cleanup_errors.append(caught)
            if cancelled:
                # A workspace's finally blocks may enqueue cleanup operations.
                for stream in streams:
                    try:
                        stream.synchronize()
                    except BaseException as caught:
                        cleanup_errors.append(caught)
            if error is not None:
                for caught in cleanup_errors:
                    error.add_note(f'Cooperative stream cleanup also failed: {caught}')
            elif cleanup_errors:
                error = BaseExceptionGroup('Cooperative stream cleanup failed', cleanup_errors)
    if error is not None:
        raise error
    return results, diagnostics
