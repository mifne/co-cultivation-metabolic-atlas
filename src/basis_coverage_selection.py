"""Training-only inventory and deterministic selection for LP basis banks.

This module contains no optimizer and no GPU code.  It is the small, audited
selection layer intended to sit between an *offline* exact-basis collector and
a separately built compact GPU bank.  In particular, a ``True`` entry in a
coverage matrix must already mean that the reconstructed candidate passed the
unchanged original-LP certificate; scores or distances are not coverage.
"""

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np


_HIGHS_BASIC = 1


@dataclass(frozen=True)
class TrainingScope:
    seeds: tuple[int, ...]
    steps: int
    expected_rows: int
    model_fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class CoverageSelection:
    selected: tuple[int, ...]
    mandatory_count: int
    marginal_weighted_gains: tuple[float, ...]
    covered_rows: tuple[bool, ...]
    covered_count: int
    total_rows: int
    selected_bytes: int | None

    @property
    def coverage_rate(self):
        return self.covered_count / self.total_rows


def _status_matrix(values, name):
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f'{name} must be a nonempty state-by-status matrix')
    if values.dtype.kind not in 'iu' or np.any((values < 0) | (values > 4)):
        raise ValueError(f'{name} must contain HiGHS status integers 0 through 4')
    return values.astype(np.uint8, copy=False)


def basis_signature(col_status, row_status):
    """Return a stable signature of one complete HiGHS combinatorial basis."""
    columns = _status_matrix(np.asarray(col_status)[None], 'col_status')[0]
    rows = _status_matrix(np.asarray(row_status)[None], 'row_status')[0]
    if np.count_nonzero(columns == _HIGHS_BASIC) + np.count_nonzero(rows == _HIGHS_BASIC) != len(rows):
        raise ValueError('A basis must contain exactly one basic variable per LP row')
    digest = hashlib.sha256(b'co-cultivation-highs-basis-v1\0')
    digest.update(np.asarray([len(columns), len(rows)], dtype='<u8').tobytes())
    digest.update(np.ascontiguousarray(columns).tobytes())
    digest.update(np.ascontiguousarray(rows).tobytes())
    return digest.hexdigest()


def inventory_basis_statuses(col_statuses, row_statuses, trajectory_ids, steps):
    """Deduplicate exact offline bases and describe within-trajectory changes.

    Input order must be trajectory-major and strictly increasing in step within
    each trajectory.  This prevents an accidental shuffled/held-out table from
    being reported as a temporal training inventory.
    """
    columns = _status_matrix(col_statuses, 'col_statuses')
    rows = _status_matrix(row_statuses, 'row_statuses')
    if columns.shape[0] != rows.shape[0]:
        raise ValueError('Column and row status tables must contain the same states')
    n_states = columns.shape[0]
    trajectory_ids = np.asarray(trajectory_ids)
    steps = np.asarray(steps)
    if trajectory_ids.ndim != 1 or len(trajectory_ids) != n_states:
        raise ValueError('One trajectory ID is required per state')
    if steps.shape != (n_states,) or steps.dtype.kind not in 'iu' or np.any(steps < 1):
        raise ValueError('One positive integer step is required per state')
    trajectory_keys = []
    for value in trajectory_ids:
        value = value.item() if isinstance(value, np.generic) else value
        try:
            hash(value)
        except TypeError:
            raise ValueError('Trajectory IDs must be hashable scalar values') from None
        trajectory_keys.append(value)
    if len(set(zip(trajectory_keys, (int(step) for step in steps)))) != n_states:
        raise ValueError('Trajectory/step pairs must be unique')
    closed = set()
    current = None
    have_current = False
    for trajectory in trajectory_keys:
        if not have_current or trajectory != current:
            if have_current:
                closed.add(current)
            if trajectory in closed:
                raise ValueError('Each trajectory must occupy one contiguous block')
            current = trajectory
            have_current = True
    for state in range(n_states):
        if (np.count_nonzero(columns[state] == _HIGHS_BASIC)
                + np.count_nonzero(rows[state] == _HIGHS_BASIC) != rows.shape[1]):
            raise ValueError(f'State {state} does not contain a complete HiGHS basis')

    signatures = []
    signature_to_id = {}
    state_basis_ids = []
    counts = []
    first_occurrence = []
    for state in range(n_states):
        signature = basis_signature(columns[state], rows[state])
        basis_id = signature_to_id.get(signature)
        if basis_id is None:
            basis_id = len(signatures)
            signature_to_id[signature] = basis_id
            signatures.append(signature)
            counts.append(0)
            first_occurrence.append(state)
        counts[basis_id] += 1
        state_basis_ids.append(basis_id)

    previous = {}
    adjacent_hamming = []
    basis_changes = 0
    for state, (key, step) in enumerate(zip(trajectory_keys, steps)):
        if key in previous:
            old_step, old_state = previous[key]
            if int(step) <= old_step:
                raise ValueError('Steps must increase within every trajectory')
            distance = (np.count_nonzero(columns[state] != columns[old_state])
                        + np.count_nonzero(rows[state] != rows[old_state]))
            adjacent_hamming.append(int(distance))
            basis_changes += int(state_basis_ids[state] != state_basis_ids[old_state])
        previous[key] = (int(step), state)

    return dict(
        states=n_states,
        unique_basis_count=len(signatures),
        signatures=signatures,
        state_basis_ids=state_basis_ids,
        counts=counts,
        first_occurrence=first_occurrence,
        adjacent_comparisons=len(adjacent_hamming),
        basis_change_count=basis_changes,
        adjacent_status_hamming=adjacent_hamming,
    )


def validate_training_scope(manifest, expected_model_fingerprints, *, forbidden_seeds=(), required_steps=None):
    """Validate that a cached source is the declared independent training set."""
    if not isinstance(manifest, dict) or manifest.get('status') != 'completed':
        raise ValueError('Training manifest must be complete')
    seeds = manifest.get('actual_training_seeds')
    steps = manifest.get('actual_training_steps')
    if (not isinstance(seeds, list) or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('Training seeds must be a nonempty unique integer list')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('Training step count must be a positive integer')
    if required_steps is not None and steps != required_steps:
        raise ValueError('Training horizon does not match the requested horizon')
    forbidden = set(forbidden_seeds)
    overlap = sorted(set(seeds) & forbidden)
    if overlap:
        raise ValueError(f'Forbidden diagnostic/holdout seeds in training source: {overlap}')
    provenance = manifest.get('provenance')
    if not isinstance(provenance, list) or not provenance:
        raise ValueError('Training manifest lacks GEM fingerprint provenance') from None
    expected = dict(expected_model_fingerprints)
    for index, source in enumerate(provenance):
        try:
            fingerprints = source['identity']['model_fingerprints']
        except (KeyError, TypeError):
            raise ValueError(f'Provenance entry {index} lacks GEM fingerprints') from None
        if not isinstance(fingerprints, dict) or dict(fingerprints) != expected:
            raise ValueError(f'Provenance entry {index} GEM fingerprints do not match the requested models')
    return TrainingScope(tuple(seeds), steps, len(seeds)*steps, expected)


def greedy_set_cover(coverage, max_candidates, *, mandatory=(), candidate_bytes=None,
                     max_bytes=None, state_weights=None):
    """Select a deterministic, memory-bounded candidate cover.

    ``coverage[state, candidate]`` must be boolean original-certificate results.
    Mandatory candidates are retained in the supplied order.  Without a byte
    budget, candidates maximize uncovered weighted states; with a byte budget,
    gain per byte is primary and absolute gain is the tie-break.  Candidate
    index is the final deterministic tie-break.
    """
    coverage = np.asarray(coverage)
    if (coverage.ndim != 2 or not coverage.shape[0] or not coverage.shape[1]
            or coverage.dtype.kind != 'b'):
        raise ValueError('Coverage must be a nonempty boolean state-by-candidate matrix')
    states, candidates = coverage.shape
    if (isinstance(max_candidates, bool)
            or not isinstance(max_candidates, (int, np.integer))
            or not 1 <= int(max_candidates) <= candidates):
        raise ValueError('max_candidates must be between one and the candidate count')
    max_candidates = int(max_candidates)
    mandatory_values = tuple(mandatory)
    if any(isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer)) for index in mandatory_values):
        raise ValueError('Mandatory candidate indices must be integers, not coerced values')
    mandatory = tuple(int(index) for index in mandatory_values)
    if len(mandatory) != len(set(mandatory)) or any(index < 0 or index >= candidates for index in mandatory):
        raise ValueError('Mandatory candidate indices must be unique and in range')
    if len(mandatory) > max_candidates:
        raise ValueError('Mandatory candidates exceed the candidate-count budget')

    if state_weights is None:
        weights = np.ones(states, dtype=np.float64)
    else:
        weights = np.asarray(state_weights, dtype=np.float64)
        if weights.shape != (states,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('State weights must be finite, nonnegative and match states')
        if not np.any(weights > 0):
            raise ValueError('At least one state weight must be positive')

    costs = None
    if candidate_bytes is not None:
        raw = np.asarray(candidate_bytes)
        if raw.shape != (candidates,) or raw.dtype.kind not in 'iu' or np.any(raw <= 0):
            raise ValueError('Candidate byte costs must be positive integer values')
        # Python integers preserve uint64 values above INT64_MAX.  Converting
        # those to np.int64 would wrap negative and could bypass max_bytes.
        costs = tuple(int(value) for value in raw)
    if max_bytes is not None:
        if costs is None:
            raise ValueError('A byte budget requires candidate byte costs')
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, (int, np.integer)) or max_bytes < 1:
            raise ValueError('max_bytes must be a positive integer')
        max_bytes = int(max_bytes)
        if sum(costs[index] for index in mandatory) > max_bytes:
            raise ValueError('Mandatory candidates exceed the byte budget')

    selected = []
    marginal = []
    covered = np.zeros(states, dtype=bool)
    used_bytes = 0

    def append(index):
        nonlocal covered, used_bytes
        newly = ~covered & coverage[:, index]
        marginal.append(float(weights[newly].sum()))
        covered |= coverage[:, index]
        selected.append(index)
        if costs is not None:
            used_bytes += int(costs[index])

    for index in mandatory:
        append(index)

    while len(selected) < max_candidates:
        choices = []
        selected_set = set(selected)
        for index in range(candidates):
            if index in selected_set:
                continue
            if max_bytes is not None and used_bytes + int(costs[index]) > max_bytes:
                continue
            gain = float(weights[~covered & coverage[:, index]].sum())
            if gain <= 0:
                continue
            efficiency = gain / int(costs[index]) if max_bytes is not None else gain
            choices.append((-efficiency, -gain, index))
        if not choices:
            break
        append(min(choices)[2])

    return CoverageSelection(
        selected=tuple(selected),
        mandatory_count=len(mandatory),
        marginal_weighted_gains=tuple(marginal),
        covered_rows=tuple(bool(value) for value in covered),
        covered_count=int(covered.sum()),
        total_rows=states,
        selected_bytes=None if costs is None else used_bytes,
    )
