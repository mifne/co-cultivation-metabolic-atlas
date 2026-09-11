"""Toy-only correctness tests for the unqualified heterogeneous GPU prototype."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.gpu_basis_bank import GpuBasisBank
from src.gpu_certified_basis import compile_basis
from src.gpu_compact_basis import CompactBank, project_basis
from src.gpu_heterogeneous_compact import HeterogeneousCompactBank
from tests.test_gpu_revised_basis import problem, cpu


def _fixture(variable_rows=(0,), extra_anchor=False):
    cp = pytest.importorskip('cupy')
    p, q = problem(), problem(rhs=(0.1, 3.))
    training = [p, q, problem(coef=1.2), problem(rhs=(0.2, 3.), coef=0.7),
        problem(cost=(-1., -2.), upper=(2., 10.))]
    anchor_problems = [p, q] + ([problem(upper=(1., 10.))] if extra_anchor else [])
    anchors = [compile_basis(row.a, row.rhs, row.lower, row.upper, row.c, row.neq) for row in anchor_problems]
    full = GpuBasisBank([anchors[0]], list(variable_rows))
    data = {key: value.get() for key, value in full.prepare_host(training).items()}
    entries = [project_basis(anchor, data, np.zeros((1, len(variable_rows), 2)), list(variable_rows)) for anchor in anchors]
    bank = CompactBank(dict(a=p.a, neq=0), list(variable_rows), entries, np.zeros((len(entries), 1)),
        np.array([0]), np.ones(1), capture=True, full_batch_candidates=True)
    assert len(entries[0]['basic']) != len(entries[1]['basic'])
    bank.configure_observables(csr_matrix([[1., 0.], [0., 2.]]), [1., 2.])
    return cp, bank, HeterogeneousCompactBank(bank), training


def _reference_slots(bank, inputs, order):
    results = {}
    for row in range(len(order)):
        for slot, index in enumerate(order[row]):
            if 0 <= index < len(bank.evaluators):
                result = bank.evaluators[index].solve_device(**{key: value[row:row+1] for key, value in inputs.items()})
                results[row, slot] = {key: value.get()[0].copy() for key, value in result.items()}
    return results


@pytest.mark.parametrize('device_order', [False, True])
def test_heterogeneous_varying_basis_dimensions_and_woodbury_match_original(device_order, monkeypatch):
    import highspy
    import scipy.optimize
    cp, bank, service, training = _fixture()
    queries = [training[0], training[1], training[2], training[3]]
    expected_objective = [cpu(query).fun for query in queries]
    inputs = bank.prepare_host(queries)
    host_order = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)

    def forbidden(*args, **kwargs):
        raise AssertionError('Online CPU LP forbidden')

    monkeypatch.setattr(highspy.Highs, 'run', forbidden)
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    try:
        references = _reference_slots(bank, inputs, host_order)
        order = cp.asarray(host_order) if device_order else host_order
        result = service.evaluate(inputs, order)
        metrics = result['candidate_metrics']
        for (row, slot), reference in references.items():
            for key in ('accepted', 'input_family_valid', 'primal_residual', 'dual_violation', 'relative_kkt_gap', 'objective'):
                np.testing.assert_allclose(metrics[key].get()[row, slot], reference[key], atol=1e-9, rtol=1e-9, equal_nan=True)
        assert result['accepted'].get().all()
        np.testing.assert_allclose(result['objective'].get(), expected_objective, atol=1e-7)
        assert result['candidate_index'].get().tolist() == [0, 1, 0, 1]
        assert result['candidate_evaluations'] == 8 and result['cpu_lp_calls'] == 0
        # Flat per-candidate weights are independent of the number of queries;
        # there is no B*K*H*R projection-weight tensor.
        assert all(values[0].ndim == 1 for values in service.weights.values())
        expected_bytes = sum(e.d[name].nbytes for e in bank.evaluators for name in ('p_rhs', 'p_bound', 'u', 'p_cost', 'p_delta'))
        assert service.setup_weight_bytes == expected_bytes
        assert not bank.graph_cache and not bank.cohort_graph_cache
    finally:
        service.close(); bank.clear_graph_cache()


def test_heterogeneous_invalid_ids_duplicates_and_observable_dispersion():
    cp, bank, service, training = _fixture()
    inputs = bank.prepare_host([training[0], training[1], training[2]])
    host_order = np.array([[0, 1, 0], [-1, 0, 1], [99, 1, 1]])
    try:
        references = _reference_slots(bank, inputs, host_order)
        result = service.evaluate(inputs, cp.asarray(host_order))
        metrics = result['candidate_metrics']
        assert not metrics['accepted'].get()[1, 0]
        assert not metrics['accepted'].get()[2, 0]
        expected_count, expected_spread = [], []
        for row, indices in enumerate(host_order):
            seen, points = set(), []
            for slot, index in enumerate(indices):
                if index in seen or (row, slot) not in references:
                    continue
                seen.add(index)
                reference = references[row, slot]
                physical = reference['raw_values']/inputs['col_scale'].get()[row]
                if reference['input_family_valid'] and np.isfinite(physical).all():
                    points.append(np.array([physical[0], 2.*physical[1]]))
            expected_count.append(len(points))
            expected_spread.append(np.max(np.std(points, axis=0, ddof=1)/[1., 2.]) if len(points) >= 2 else np.inf)
        np.testing.assert_array_equal(result['candidate_count'].get(), expected_count)
        np.testing.assert_allclose(result['observable_dispersion'].get(), expected_spread, atol=1e-9)
        assert result['candidate_index'].get()[0] == 0  # Duplicate/later solution cannot overwrite the first.
        assert result['candidate_count'].get().tolist() == [2, 2, 1]
    finally:
        service.close(); bank.clear_graph_cache()


def test_heterogeneous_unseen_direction_invalid_scaling_and_infeasibility_fail_closed():
    cp, bank, service, training = _fixture()
    inputs = bank.prepare_host([problem(rhs=(-1., 3.)), training[0], training[1], training[0]])
    inputs['row_scale'][1, 0] = 0.
    inputs['rhs'][2, 0] = cp.nan
    # The training varied matrix column 0 only. Column 1 is an unsupported
    # basic-direction change for candidate 0, so its family guard must reject.
    inputs['delta'][3, 0, 1] = 0.5
    try:
        references = _reference_slots(bank, inputs, np.zeros((4, 1), dtype=int))
        result = service.evaluate(inputs, np.zeros((4, 1), dtype=int))
        assert not result['accepted'].get().any()
        assert np.isnan(result['values'].get()).all()
        for row in range(4):
            assert result['input_family_valid'].get()[row] == references[row, 0]['input_family_valid']
        assert result['best_candidate_index'].get()[1:3].tolist() == [-1, -1]
    finally:
        service.close(); bank.clear_graph_cache()


@pytest.mark.parametrize('ranking', ['residual', 'count', 'count_only'])
def test_heterogeneous_repair_ranking_matches_single_candidate_metrics(ranking):
    cp, bank, service, training = _fixture()
    bank.candidate_ranking = ranking
    inputs = bank.prepare_host([problem(cost=(2., 1.)), problem(upper=(1., 10.))])
    order = np.array([[0, 1], [1, 0]])
    try:
        references = _reference_slots(bank, inputs, order)
        result = service.evaluate(inputs, order)
        for row in range(len(order)):
            accepted = [slot for slot in range(2) if references[row, slot]['accepted']]
            if accepted:
                assert result['candidate_index'].get()[row] == order[row, accepted[0]]
                continue
            choices = []
            for slot in range(2):
                reference = references[row, slot]
                if not reference['input_family_valid']:
                    continue
                score = max(reference['primal_residual']/1e-5, reference['dual_violation']/1e-7, reference['relative_kkt_gap']/1e-7)
                if ranking != 'residual':
                    l1 = reference['primal_violation_l1']
                    score = reference['primal_violation_count']+l1/(1.+l1)
                if np.isfinite(score):
                    dual = reference['basis_dual_violation'] <= 1e-8
                    choices.append(((0 if ranking == 'count_only' or dual else 1, score, slot), order[row, slot]))
            expected = min(choices)[1] if choices else -1
            assert result['best_candidate_index'].get()[row] == expected
    finally:
        service.close(); bank.clear_graph_cache()


def test_heterogeneous_rejects_mismatched_shapes_and_close_is_terminal():
    cp, bank, service, training = _fixture()
    inputs = bank.prepare_host([training[0]])
    try:
        with pytest.raises(ValueError, match='integer'):
            service.evaluate(inputs, [[0.5]])
        with pytest.raises(ValueError, match='shape'):
            service.evaluate(inputs, [[0], [1]])
        with pytest.raises(ValueError, match='nonempty'):
            service.evaluate(inputs, np.empty((1, 0), dtype=int))
        service.close(); service.close()
        with pytest.raises(RuntimeError, match='closed'):
            service.evaluate(inputs, [[0]])
    finally:
        service.close(); bank.clear_graph_cache()


def _snapshot(value):
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value.get().copy() if hasattr(value, 'get') else value


def _assert_snapshot(value, expected):
    if isinstance(expected, dict):
        assert set(value) == set(expected)
        for key in expected:
            _assert_snapshot(value[key], expected[key])
    elif isinstance(expected, np.ndarray):
        np.testing.assert_allclose(value.get(), expected, atol=1e-9, rtol=1e-9, equal_nan=True)
    else:
        assert value == expected


def test_heterogeneous_replay_dynamic_candidate_ids_inputs_and_unknowns_without_cpu(monkeypatch):
    import highspy
    import scipy.optimize
    cp, bank, service, training = _fixture()
    first = bank.prepare_host(training[:2])
    changed = bank.prepare_host([training[3], training[2]])
    changed['col_scale'][:] = cp.asarray([[2., 3.], [4., 1.]])
    changed['row_scale'][:] = cp.asarray([[1., 2.], [3., 1.]])
    unknown = bank.prepare_host([problem(rhs=(-1., 3.)), training[0]])
    unknown['c'][1, 0] = cp.nan
    pairs = [
        (first, [[0, 1], [1, 0]]),
        (first, [[1, 1], [0, 0]]),
        (changed, [[0, 1], [1, 0]]),
        (unknown, [[1, 0], [-1, 99]]),
        (first, [[0, 1], [1, 0]]),
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError('Online CPU LP forbidden')

    monkeypatch.setattr(highspy.Highs, 'run', forbidden)
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    try:
        for inputs, ids in pairs:
            order = cp.asarray(ids, dtype=cp.int32)
            expected = _snapshot(service.evaluate(inputs, order))
            output = service.evaluate_replay(inputs, order)
            _assert_snapshot(output, expected)
            assert len(service.graph_cache) == 1
            assert service.last_candidate_scores is output['last_candidate_scores']
            assert service.last_candidates is output['candidate_metrics']
        assert service.graph_compilation_seconds > 0.
        assert not bank.graph_cache and not bank.cohort_graph_cache
    finally:
        service.close(); bank.clear_graph_cache()


def test_heterogeneous_replay_cache_release_and_changed_observables():
    cp, bank, service, training = _fixture()
    inputs = bank.prepare_host(training[:2])
    try:
        retained = service.evaluate_replay(inputs, cp.asarray([[0, 1], [1, 0]]))
        snapshot = _snapshot(retained)
        old = next(iter(service.graph_cache.values()))
        before = old.pool.total_bytes()
        # A different K requires another graph; it must not change old output.
        service.evaluate_replay(inputs, cp.asarray([[0], [1]]))
        _assert_snapshot(retained, snapshot)
        bank.configure_observables(csr_matrix([[1., 0.], [0., 2.]]), [2., 4.])
        updated = service.evaluate_replay(inputs, cp.asarray([[0, 1], [1, 0]]))
        np.testing.assert_allclose(updated['observable_dispersion'].get(), snapshot['observable_dispersion']/2., atol=1e-9)
        assert len(service.graph_cache) == 2
        assert old.graph_exec is None and old.result is None
        assert old.pool.total_bytes() < before
        _assert_snapshot(retained, snapshot)
        calls = list(service.graph_cache.values())
        service.close()
        assert not service.graph_cache
        assert all(call.graph_exec is None and call.solver is None for call in calls)
        _assert_snapshot(retained, snapshot)
    finally:
        service.close(); bank.clear_graph_cache()


def test_heterogeneous_mixed_low_rank_padding_and_upper_nonbasic_bound():
    cp, bank, service, training = _fixture(variable_rows=(0, 1), extra_anchor=True)
    queries = [training[0], problem(upper=(1., 10.)), training[1]]
    inputs = bank.prepare_host(queries)
    order = np.array([[0, 1, 2], [2, 1, 0], [1, 0, 2]])
    try:
        assert service.rank == 2
        assert sorted(set(service.counts['update_positions'].get().tolist())) == [1, 2]
        references = _reference_slots(bank, inputs, order)
        result = service.evaluate_replay(inputs, cp.asarray(order))
        for (row, slot), reference in references.items():
            for key in ('accepted', 'input_family_valid', 'objective', 'primal_residual', 'dual_violation', 'relative_kkt_gap'):
                np.testing.assert_allclose(result['candidate_metrics'][key].get()[row, slot], reference[key], atol=1e-9, rtol=1e-9)
        assert result['accepted'].get().all()
        assert result['candidate_index'].get().tolist() == [0, 2, 1]
        np.testing.assert_allclose(result['objective'].get(), [cpu(query).fun for query in queries], atol=1e-8)
    finally:
        service.close(); bank.clear_graph_cache()
