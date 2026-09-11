import numpy as np
import pytest
import torch
from scripts.train_cooperative_neural_reranker import decision_intervals, interval_violation
from scripts.train_cooperative_neural_reranker import check_warm_start


def test_compact_context_recovers_variable_and_constant_bounds():
    metadata = {"context_layout":[["biomass_g_l", 1], ["flux_lower_bounds", 2], ["flux_upper_bounds", 2]]}
    lower, upper = decision_intervals(np.asarray([[.2, 3], [.4, 0]]), [0, 4],
        np.asarray([.1, 0, 0, 1, 100]), metadata, [0, 1])
    np.testing.assert_allclose(lower, [[0, 0], [0, 0]])
    np.testing.assert_allclose(upper, [[1, 3], [1, 0]])


def test_closed_flux_has_a_restoring_gradient_not_only_posthoc_clipping():
    prediction = torch.tensor([2., -1., .3], requires_grad=True)
    loss = interval_violation(prediction, torch.tensor([0., 0., 0.]), torch.tensor([0., 1., 1.])).sum()
    loss.backward()
    np.testing.assert_allclose(prediction.grad.numpy(), [4, -2, 0])
    assert loss.item() == pytest.approx(5)


def test_warm_start_requires_identical_gene_model_feature_and_output_layout():
    metadata = {"species":["toy"], "reaction_ids":["growth"], "model_fingerprints":{"toy":"abc"},
                "cooperative_optimize_live_objectives":True, "hidden_dims":[4]}
    payload = {"metadata":metadata, "feature_indices":[2], "decision_indices":[0]}
    check_warm_start(payload, [2], [0], metadata, [4])
    with pytest.raises(ValueError, match="feature_indices"):
        check_warm_start(payload, [3], [0], metadata, [4])
    with pytest.raises(ValueError, match="model_fingerprints"):
        check_warm_start(payload, [2], [0], {**metadata, "model_fingerprints":{"toy":"new"}}, [4])
