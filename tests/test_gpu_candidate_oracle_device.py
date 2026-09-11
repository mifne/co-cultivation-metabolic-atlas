"""Real tiny CUDA oracle checks, separate from host-only dispatch tests."""
import numpy as np
import pytest

from src.gpu_hybrid_lp import candidate_oracle_snapshot
from tests.test_gpu_heterogeneous_compact import _fixture, _snapshot, _assert_snapshot
from tests.test_gpu_revised_basis import cpu


@pytest.mark.parametrize('replay', [False, True])
def test_device_oracle_finds_outside_k_bases_without_mutating_normal_results_or_metadata(monkeypatch, replay):
    import highspy
    import scipy.optimize
    cp, bank, engine, training = _fixture()
    queries = training[:4]
    expected_objectives = [cpu(query).fun for query in queries]
    inputs = bank.prepare_host(queries)
    # Candidate 0 fits the two wide-resource LPs; the narrow-resource LPs
    # require candidate 1, which is deliberately omitted from normal K=1.
    order = np.zeros((4, 1), dtype=np.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError('The diagnostic oracle must not optimize on CPU')

    monkeypatch.setattr(highspy.Highs, 'run', forbidden)
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    try:
        normal = (engine.evaluate_replay if replay else engine.evaluate)(inputs, cp.asarray(order))
        normal_accepted = normal['accepted'].get()
        assert normal_accepted.tolist() == [True, False, True, False]
        normal_snapshot, input_snapshot = _snapshot(normal), _snapshot(inputs)
        references = {name:getattr(engine, name) for name in
            ('last_candidates', 'last_candidate_scores', 'last_candidate_dual')}
        graph_keys = tuple(engine.graph_cache)
        graph_objects = tuple(engine.graph_cache.values())

        oracle = candidate_oracle_snapshot(engine, inputs, order, len(bank.evaluators), normal_accepted)
        assert oracle['status'] == 'completed', oracle
        assert oracle['normal_k_accepted'] == [True, False, True, False]
        assert oracle['snapshot']['selection']['accepted'] == [True]*4
        assert oracle['snapshot']['selection']['candidate_index'] == [0, 1, 0, 1]
        assert oracle['additional_accepted_environment_ids'] == [1, 3]
        assert oracle['outside_k_rescued_environment_ids'] == [1, 3]
        assert oracle['additional_accepted_count'] == oracle['outside_k_rescued_count'] == 2
        for row, candidate in enumerate(oracle['snapshot']['selection']['candidate_index']):
            assert oracle['snapshot']['accepted'][row][candidate]
            assert oracle['snapshot']['input_family_valid'][row][candidate]
            np.testing.assert_allclose(oracle['snapshot']['objective'][row][candidate],
                                       expected_objectives[row], atol=1e-7)
            assert all(values[row][candidate] for values in oracle['snapshot']['scalar_gate_pass'].values())

        _assert_snapshot(normal, normal_snapshot)
        _assert_snapshot(inputs, input_snapshot)
        for name, reference in references.items():
            assert getattr(engine, name) is reference
        assert tuple(engine.graph_cache) == graph_keys
        assert tuple(engine.graph_cache.values()) == graph_objects
        assert not bank.graph_cache and not bank.cohort_graph_cache
        # A subsequent normal call must still produce the exact K=1 result,
        # rather than adopting the successful counterfactual K=2 fluxes.
        repeated = (engine.evaluate_replay if replay else engine.evaluate)(inputs, cp.asarray(order))
        _assert_snapshot(repeated, normal_snapshot)
    finally:
        engine.close(); bank.clear_graph_cache()
