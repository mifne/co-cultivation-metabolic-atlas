"""Streaming, original-certificate coverage measurement for compact LP banks.

The runtime bank remains unchanged.  This offline utility evaluates a small
number of existing compact candidates at a time, retains only boolean coverage
on the host, and never uses CPU LP labels or an optimizer.  A coverage bit is
true only when the candidate's existing full original-LP certificate and all
of its finite/family guards pass.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np


FIELDS = ('rhs', 'lower', 'upper', 'c', 'delta', 'col_scale', 'row_scale')
CERTIFICATE_THRESHOLDS = dict(
    primal_residual=1e-5,
    dual_violation=1e-7,
    relative_kkt_gap=1e-7,
)


@dataclass(frozen=True)
class CompactCoverageResult:
    coverage: np.ndarray
    candidate_indices: np.ndarray
    routing_order: np.ndarray
    routing_topk: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    candidate_chunk_size: int
    candidate_failures: tuple[dict, ...]
    timing: dict

    @property
    def complete(self):
        return not self.candidate_failures

    @property
    def row_count(self):
        return int(self.coverage.shape[0])


def _host(value):
    return np.asarray(value.get() if hasattr(value, 'get') else value)


def _synchronize(cp):
    cuda = getattr(cp, 'cuda', None)
    if cuda is not None:
        cuda.get_current_stream().synchronize()


def _integer_indices(values, size, name, *, nonempty=True):
    raw = tuple(values)
    if ((nonempty and not raw)
            or any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, (int, np.integer)) for value in raw)):
        qualifier = 'nonempty ' if nonempty else ''
        raise ValueError(f'{name} must be a {qualifier}integer sequence')
    result = tuple(int(value) for value in raw)
    if len(result) != len(set(result)) or any(value < 0 or value >= size for value in result):
        raise ValueError(f'{name} must be unique and in range')
    return result


def _device_inputs(bank, supplied):
    if not isinstance(supplied, dict) or set(supplied) != set(FIELDS):
        raise ValueError('Each input batch must contain the exact compact LP fields')
    cp = bank.cp
    arrays = {name:cp.asarray(supplied[name], dtype=cp.float64) for name in FIELDS}
    m, n = bank.host_a.shape
    variable_count = len(bank.variable_rows)
    lower = arrays['lower']
    if lower.ndim != 2 or not lower.shape[0]:
        raise ValueError('Each input batch must contain at least one LP row')
    batch = lower.shape[0]
    expected = dict(
        rhs=(batch, m), lower=(batch, n), upper=(batch, n), c=(batch, n),
        delta=(batch, variable_count, n), col_scale=(batch, n),
        row_scale=(batch, m),
    )
    if any(arrays[name].shape != shape for name, shape in expected.items()):
        raise ValueError('Compact LP input batch shape does not match the bank')
    return arrays, batch


def _certified_mask(cp, result, batch, columns, rows):
    vector_shapes = dict(
        accepted=(batch,), input_family_valid=(batch,),
        primal_residual=(batch,), dual_violation=(batch,),
        relative_kkt_gap=(batch,), objective=(batch,),
        raw_values=(batch, columns), raw_reduced=(batch, columns),
        raw_y=(batch, rows), activity=(batch, rows),
    )
    if not isinstance(result, dict):
        raise ValueError('Candidate result is not a mapping')
    missing = sorted(set(vector_shapes)-set(result))
    if missing:
        raise ValueError('Candidate result lacks certificate fields: ' + ', '.join(missing))
    if any(result[name].shape != shape for name, shape in vector_shapes.items()):
        raise ValueError('Candidate certificate field has the wrong shape')
    # Do not coerce integer/float flags: values such as ``-1`` would otherwise
    # become truthy and could turn malformed evaluator output into coverage.
    # The per-candidate caller records this as a failed, incomplete measurement.
    if (result['accepted'].dtype.kind != 'b'
            or result['input_family_valid'].dtype.kind != 'b'):
        raise ValueError('Candidate acceptance and input-family flags must be boolean')
    accepted = result['accepted']
    family = result['input_family_valid']
    finite = cp.isfinite(result['objective'])
    for name in ('primal_residual', 'dual_violation', 'relative_kkt_gap'):
        finite &= cp.isfinite(result[name])
    for name in ('raw_values', 'raw_reduced', 'raw_y', 'activity'):
        finite &= cp.isfinite(result[name]).all(axis=1)
    metrics_nonnegative = ((result['primal_residual'] >= 0.)
            & (result['dual_violation'] >= 0.)
            & (result['relative_kkt_gap'] >= 0.))
    gate = (metrics_nonnegative
            & (result['primal_residual'] <= CERTIFICATE_THRESHOLDS['primal_residual'])
            & (result['dual_violation'] <= CERTIFICATE_THRESHOLDS['dual_violation'])
            & (result['relative_kkt_gap'] <= CERTIFICATE_THRESHOLDS['relative_kkt_gap']))
    # ``accepted`` is authoritative because CompactEvaluator also checks the
    # unmodified LP and input family.  Repeating the visible guards makes a
    # future missing/nonfinite output fail closed rather than become coverage.
    return accepted & family & finite & gate


def measure_compact_bank_coverage(bank, input_batches, *, candidate_indices=None,
                                  candidate_chunk_size=1, routing_topk=(1, 4)):
    """Measure candidate-by-training-row coverage with bounded device outputs.

    ``input_batches`` is consumed once and should normally yield B=32/64 input
    dictionaries from the independent training cache.  The routing result is
    stateless nearest-centroid order; it intentionally does not simulate the
    runtime's previous-certified-candidate promotion.
    """
    entries = len(bank.evaluators)
    if not entries:
        raise ValueError('At least one compact candidate is required')
    candidates = (_integer_indices(range(entries), entries, 'candidate_indices')
                  if candidate_indices is None else
                  _integer_indices(candidate_indices, entries, 'candidate_indices'))
    if (isinstance(candidate_chunk_size, bool)
            or not isinstance(candidate_chunk_size, (int, np.integer))
            or candidate_chunk_size < 1):
        raise ValueError('candidate_chunk_size must be a positive integer')
    candidate_chunk_size = int(candidate_chunk_size)
    topk = _integer_indices(routing_topk, entries+1, 'routing_topk')
    if any(value < 1 or value > entries for value in topk):
        raise ValueError('routing_topk values must be between one and the bank size')

    cp = bank.cp
    coverage_batches = []
    routing_batches = []
    batch_sizes = []
    failures = []
    timing = dict(input_transfer_seconds=0., routing_seconds=0.,
                  candidate_evaluation_seconds=0., coverage_download_seconds=0.,
                  total_seconds=0.)
    started_total = time.perf_counter()
    maximum_k = max(topk)
    columns = bank.host_a.shape[1]
    rows = bank.host_a.shape[0]

    for batch_index, supplied in enumerate(input_batches):
        started = time.perf_counter()
        inputs, batch = _device_inputs(bank, supplied)
        _synchronize(cp)
        timing['input_transfer_seconds'] += time.perf_counter()-started
        batch_sizes.append(batch)

        started = time.perf_counter()
        features = bank.proposal_features(inputs)
        distance = cp.sum(((features[:, None]-bank.centers[None])/bank.feature_scale)**2, axis=2)
        order = cp.argsort(distance, axis=1)[:, :maximum_k]
        routing_batches.append(_host(order).astype(np.int32, copy=False))
        timing['routing_seconds'] += time.perf_counter()-started

        host_coverage = np.zeros((batch, len(candidates)), dtype=bool)
        for start in range(0, len(candidates), candidate_chunk_size):
            chunk = candidates[start:start+candidate_chunk_size]
            masks = []
            evaluation_started = time.perf_counter()
            for position, candidate in enumerate(chunk, start):
                try:
                    result = bank.evaluators[candidate].solve_device(**inputs)
                    masks.append(_certified_mask(cp, result, batch, columns, rows))
                except Exception as error:
                    masks.append(cp.zeros(batch, dtype=cp.bool_))
                    failures.append(dict(batch_index=batch_index,
                        candidate_index=int(candidate), error_type=type(error).__name__,
                        error=str(error)))
            stacked = cp.stack(masks, axis=1)
            _synchronize(cp)
            timing['candidate_evaluation_seconds'] += time.perf_counter()-evaluation_started
            download_started = time.perf_counter()
            host_coverage[:, start:start+len(chunk)] = _host(stacked)
            timing['coverage_download_seconds'] += time.perf_counter()-download_started
        coverage_batches.append(host_coverage)

    if not coverage_batches:
        raise ValueError('At least one compact LP input batch is required')
    coverage = np.concatenate(coverage_batches, axis=0)
    routing = np.concatenate(routing_batches, axis=0)
    timing['total_seconds'] = time.perf_counter()-started_total
    return CompactCoverageResult(
        coverage=coverage,
        candidate_indices=np.asarray(candidates, dtype=np.int32),
        routing_order=routing,
        routing_topk=topk,
        batch_sizes=tuple(batch_sizes),
        candidate_chunk_size=candidate_chunk_size,
        candidate_failures=tuple(failures),
        timing=timing,
    )


def summarize_routing_coverage(result):
    """Separate dictionary coverage misses from nearest-router misses."""
    if not result.complete:
        raise ValueError('Incomplete candidate measurements cannot define routing coverage')
    coverage = np.asarray(result.coverage, dtype=bool)
    order = np.asarray(result.routing_order)
    lookup = {int(candidate): column for column, candidate in enumerate(result.candidate_indices)}
    oracle = coverage.any(axis=1)
    summary = dict(
        rows=len(coverage),
        oracle_covered_rows=int(oracle.sum()),
        structural_uncovered_rows=int((~oracle).sum()),
        topk={},
        scope='Stateless nearest-centroid routing over strict original-certificate coverage; no temporal promotion.',
    )
    row_ids = np.arange(len(coverage))[:, None]
    for k in result.routing_topk:
        ids = order[:, :k]
        missing = sorted(set(int(value) for value in ids.ravel())-set(lookup))
        if missing:
            raise ValueError(f'Routing candidates were not measured: {missing}')
        columns = np.vectorize(lookup.__getitem__, otypes=[np.int64])(ids)
        accepted = coverage[row_ids, columns].any(axis=1)
        summary['topk'][str(k)] = dict(
            accepted_rows=int(accepted.sum()),
            acceptance_rate=float(accepted.mean()),
            ranking_miss_rows=int((oracle & ~accepted).sum()),
        )
    return summary


def save_compact_coverage_npz(path, result, metadata):
    """Atomically save a non-pickle coverage artifact with explicit scope."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    failures_json = json.dumps(result.candidate_failures, sort_keys=True, allow_nan=False)
    summary = None
    if result.complete:
        try:
            summary = summarize_routing_coverage(result)
        except ValueError:
            summary = None
    payload = dict(
        coverage=np.asarray(result.coverage, dtype=bool),
        candidate_indices=np.asarray(result.candidate_indices, dtype=np.int32),
        routing_order=np.asarray(result.routing_order, dtype=np.int32),
        routing_topk=np.asarray(result.routing_topk, dtype=np.int32),
        batch_sizes=np.asarray(result.batch_sizes, dtype=np.int32),
        candidate_chunk_size=np.asarray(result.candidate_chunk_size, dtype=np.int32),
        complete=np.asarray(result.complete),
        schema_version=np.asarray(1, dtype=np.int32),
        certificate_thresholds=np.asarray(json.dumps(CERTIFICATE_THRESHOLDS, sort_keys=True)),
        timing=np.asarray(json.dumps(result.timing, sort_keys=True, allow_nan=False)),
        failures=np.asarray(failures_json),
        metadata=np.asarray(metadata_json),
        routing_summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
        scope=np.asarray('Offline streamed compact-bank coverage; True means unchanged original-LP certificate passed.'),
    )
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
