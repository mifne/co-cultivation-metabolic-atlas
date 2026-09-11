from types import SimpleNamespace

import numpy as np
import pytest

from src.gpu_hybrid_lp import rank_compact_candidates


class Bank:
    def __init__(self):
        self.evaluators = [object(), object(), object()]
        self.centers = np.array([[0., 0.], [5., 5.], [10., 10.]])
        self.feature_scale = np.ones(2)
        self.feature_indices = np.array([0, 1])

    def proposal_features(self, inputs):
        return inputs['encoded']


class Router:
    candidate_count = 3
    input_dim = 2

    def __init__(self, order):
        self.order = np.asarray(order)
        self.seen = None

    def rank(self, features, k=None):
        self.seen = np.array(features, copy=True)
        assert k == 3
        return self.order


def test_opt_in_router_replaces_only_full_candidate_order():
    bank = Bank()
    inputs = {'encoded':np.array([[.1, .2], [9., 9.]])}
    nearest, source = rank_compact_candidates(np, bank, inputs)
    assert source == 'nearest_centroid'
    assert nearest.tolist() == [[0, 1, 2], [2, 1, 0]]
    router = Router([[2, 0, 1], [1, 0, 2]])
    learned, source = rank_compact_candidates(np, bank, inputs, router)
    assert source == 'learned_coverage'
    assert learned.tolist() == [[2, 0, 1], [1, 0, 2]]
    np.testing.assert_array_equal(router.seen, inputs['encoded'])


@pytest.mark.parametrize('order', [
    [[0, 1, 1], [0, 1, 2]],
    [[0, 1], [1, 0]],
    [[0., 1., 2.], [0., 1., 2.]],
    [[0, 1, 3], [0, 1, 2]],
])
def test_malformed_router_order_fails_instead_of_changing_candidate_budget(order):
    with pytest.raises(ValueError, match='full integer permutation'):
        rank_compact_candidates(np, Bank(),
            {'encoded':np.zeros((2, 2))}, Router(order))


@pytest.mark.parametrize('router_args', [
    ['--coverage-router', 'router.npz'],
    ['--coverage-router-sha256', '0'*64],
])
def test_cli_requires_router_path_and_sha_as_one_pair(tmp_path, monkeypatch, router_args):
    import sys
    from scripts.benchmark_compact_gpu import main
    output = tmp_path/'existing.json'; output.touch()
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--bank', str(tmp_path/'bank'),
        '--output', str(output), '--hybrid', '--hybrid-rounds', '0',
        '--tie-policy', 'original3', '--cpu-backend', 'dictionary', *router_args])
    with pytest.raises(ValueError, match='path and SHA256'):
        main()


def test_cli_router_requires_matched_dictionary_cpu_comparator(tmp_path, monkeypatch):
    import sys
    from scripts.benchmark_compact_gpu import main
    output = tmp_path/'existing.json'; output.touch()
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--bank', str(tmp_path/'bank'),
        '--output', str(output), '--hybrid', '--hybrid-rounds', '0',
        '--tie-policy', 'original3', '--coverage-router', 'router.npz',
        '--coverage-router-sha256', '0'*64])
    with pytest.raises(ValueError, match='dictionary CPU comparator'):
        main()
