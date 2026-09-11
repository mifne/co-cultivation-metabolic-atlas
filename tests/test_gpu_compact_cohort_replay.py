"""One graph contains the entire cohort, certificates, merge, and uncertainty."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tests.test_gpu_compact_cohort import _real_bank
from tests.test_gpu_revised_basis import problem


def _snapshot(result):
    return {key: value.get().copy() if hasattr(value, 'get') else value for key, value in result.items()}


def _assert_matches(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        measured = actual[key].get() if hasattr(actual[key], 'get') else actual[key]
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(measured, value, atol=1e-10, rtol=1e-10, equal_nan=True, err_msg=key)
        else:
            assert measured == value


def test_cohort_replay_updates_all_input_fields_and_metadata_without_cpu(monkeypatch):
    import highspy
    import scipy.optimize
    bank, p, q = _real_bank(True)
    cp = bank.cp
    bank.configure_observables(csr_matrix([[1., 0.], [0., 1.]]), [1., 2.])

    def forbidden(*args, **kwargs):
        raise AssertionError('Online CPU LP forbidden')

    monkeypatch.setattr(highspy.Highs, 'run', forbidden)
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    first = bank.prepare_host([p, q])
    changed = bank.prepare_host([
        problem(rhs=(6., 3.), upper=(2., 10.)),
        problem(rhs=(0.5, 3.), lower=(0., 0.2)),
    ])
    changed['col_scale'][:] = cp.asarray([[2., 4.], [3., 2.]])
    changed['row_scale'][:] = cp.asarray([[1., 3.], [2., 1.]])
    changed_cost = bank.prepare_host([problem(cost=(2., 1.)), problem(cost=(-1., -2.))])
    unknown = bank.prepare_host([p, q])
    unknown['delta'][0, 0, 1] = 0.5  # Direction absent from the projected family.
    unknown['rhs'][1, 0] = cp.nan
    try:
        for inputs in (first, changed, changed_cost, unknown, first):
            expected = _snapshot(bank.evaluate_cohort(inputs, [2, 0, 1], _eager=True))
            scores = bank.last_candidate_scores.get().copy()
            dual = bank.last_candidate_dual.get().copy()
            metrics = [(index, ids.copy(), _snapshot(values)) for index, ids, values in bank.last_candidates]
            result = bank.evaluate_cohort_replay(inputs, [2, 0, 1])
            _assert_matches(result, expected)
            np.testing.assert_allclose(bank.last_candidate_scores.get(), scores, equal_nan=True)
            np.testing.assert_array_equal(bank.last_candidate_dual.get(), dual)
            for (index, ids, values), (expected_index, expected_ids, expected_values) in zip(bank.last_candidates, metrics):
                assert index == expected_index
                np.testing.assert_array_equal(ids, expected_ids)
                _assert_matches(values, expected_values)
            assert len(bank.cohort_graph_cache) == 1
            assert not bank.graph_cache  # No nested per-candidate captures.
        assert bank.cohort_graph_compilation_seconds > 0.0
        assert bank.graph_compilation_seconds == bank.cohort_graph_compilation_seconds
    finally:
        bank.clear_graph_cache()


def test_cohort_replay_cache_switch_eviction_and_close_preserve_results():
    bank, p, q = _real_bank(True)
    inputs = bank.prepare_host([p, q])
    try:
        retained = bank.evaluate_cohort_replay(inputs, [0, 1])
        expected = _snapshot(retained)
        first_call = next(iter(bank.cohort_graph_cache.values()))
        before = first_call.pool.total_bytes()
        bank.evaluate_cohort_replay(inputs, [2])
        # Return to a cached graph and restore its metadata, rather than that
        # of the most recently compiled graph with a different candidate set.
        again = bank.evaluate_cohort_replay(inputs, [0, 1])
        _assert_matches(again, expected)
        assert [item[0] for item in bank.last_candidates] == [0, 1]
        assert bank.last_candidate_scores is first_call.result['scores']
        bank.evaluate_cohort_replay(inputs, [1, 0, 2])
        assert len(bank.cohort_graph_cache) == 2
        # Evict the oldest remaining graph, then explicitly close everything.
        bank.evaluate_cohort_replay(inputs, [1])
        assert first_call.graph_exec is None and first_call.raw_graph is None
        assert first_call.inputs is None and first_call.result is None
        assert first_call.pool.total_bytes() < before
        _assert_matches(retained, expected)
        calls = list(bank.cohort_graph_cache.values())
        bank.clear_graph_cache()
        assert not bank.cohort_graph_cache and not bank.graph_cache
        assert all(call.graph_exec is None and call.solver is None for call in calls)
        bank.clear_graph_cache()  # Idempotent destruction.
    finally:
        bank.clear_graph_cache()


def test_observable_reconfiguration_invalidates_captured_projection():
    bank, p, q = _real_bank(False)
    inputs = bank.prepare_host([p, q])
    try:
        bank.configure_observables(csr_matrix([[1., 0.]]), [1.])
        initial = bank.evaluate_cohort_replay(inputs, [0, 1])
        spread = initial['observable_dispersion'].get().copy()
        old = next(iter(bank.cohort_graph_cache.values()))
        bank.configure_observables(csr_matrix([[1., 0.]]), [2.])
        assert not bank.cohort_graph_cache and old.graph_exec is None
        updated = bank.evaluate_cohort_replay(inputs, [0, 1])
        np.testing.assert_allclose(updated['observable_dispersion'].get(), spread / 2.)
        assert bank.math is not None  # Explicit replay lazily enables capture math.
    finally:
        bank.clear_graph_cache()


@pytest.mark.parametrize('indices', [[0, 0], [-1], [3], [[0]], [0.5]])
def test_cohort_replay_rejects_invalid_candidate_sets_before_capture(indices):
    bank, p, q = _real_bank(True)
    try:
        with pytest.raises(ValueError):
            bank.evaluate_cohort_replay(bank.prepare_host([p, q]), indices)
        assert not bank.cohort_graph_cache
    finally:
        bank.clear_graph_cache()
