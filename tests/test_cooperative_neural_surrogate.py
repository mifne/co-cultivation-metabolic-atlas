from pathlib import Path

import numpy as np
import pytest

from src.cooperative_neural_surrogate import (
    CooperativeNeuralDecisionReranker,
    CooperativeNeuralMechanisticSurrogate,
    FORMULATION,
    RERANK_FORMULATION,
    make_cooperative_network,
)
from src.gpu_batch_qp import BatchedCooperativeQpProjector


def _zero_state(network):
    import torch

    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    return {key: value.detach().cpu() for key, value in network.state_dict().items()}


def test_affine_neural_mechanistic_decoder_and_ood_guard(tmp_path: Path):
    import torch

    network = make_cooperative_network(1, 1, (4,))
    artifact = tmp_path / "amn.pt"
    torch.save(
        {
            "metadata": {
                "format_version": 1,
                "formulation": FORMULATION,
                "context_dimension": 2,
                "hidden_dims": [4],
                "ood_tolerance": 0.25,
            },
            "feature_indices": np.asarray([0], dtype=np.int64),
            "feature_mean": np.asarray([0.0], dtype=np.float32),
            "feature_scale": np.asarray([1.0], dtype=np.float32),
            "feature_min": np.asarray([-1.0], dtype=np.float32),
            "feature_max": np.asarray([1.0], dtype=np.float32),
            "latent_mean": np.asarray([0.0], dtype=np.float32),
            "latent_scale": np.asarray([1.0], dtype=np.float32),
            "flux_mean": np.asarray([1.0, 2.0], dtype=np.float32),
            "flux_basis": np.asarray([[1.0, -1.0]], dtype=np.float32),
            "state_dict": _zero_state(network),
        },
        artifact,
    )
    surrogate = CooperativeNeuralMechanisticSurrogate(artifact, device="cpu")
    accepted, rejected = surrogate.predict(
        np.asarray([[0.0, 99.0], [2.0, 99.0]], dtype=np.float32)
    )
    assert accepted.accepted_domain
    assert accepted.fluxes == pytest.approx([1.0, 2.0])
    assert not rejected.accepted_domain
    assert rejected.reason == "out_of_distribution"


def test_neural_reranker_returns_only_exact_dictionary_fluxes(tmp_path: Path):
    import torch

    dictionary = tmp_path / "dictionary.pt"
    base_metadata = {
        "format_version": 1,
        "formulation": "exact_cooperative_shared_medium",
        "context_dimension": 2,
        "species": ["a"],
        "reaction_ids": ["Growth", "EX_x_e"],
        "distance_threshold": 10.0,
    }
    exact_fluxes = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    torch.save(
        {
            "metadata": base_metadata,
            "feature_indices": np.asarray([0], dtype=np.int64),
            "feature_mean": np.asarray([0.0], dtype=np.float32),
            "feature_scale": np.asarray([1.0], dtype=np.float32),
            "feature_weight": np.asarray([1.0], dtype=np.float32),
            "normalized_contexts": np.asarray([[0.0], [1.0]], dtype=np.float32),
            "fluxes": exact_fluxes,
        },
        dictionary,
    )
    network = make_cooperative_network(1, 1, (4,))
    artifact = tmp_path / "reranker.pt"
    torch.save(
        {
            "metadata": {
                **base_metadata,
                "formulation": RERANK_FORMULATION,
                "base_dictionary": dictionary.name,
                "hidden_dims": [4],
                "rerank_pool": 2,
                "decision_strength": 4.0,
            },
            "feature_indices": np.asarray([0], dtype=np.int64),
            "feature_mean": np.asarray([0.0], dtype=np.float32),
            "feature_scale": np.asarray([1.0], dtype=np.float32),
            "decision_indices": np.asarray([0], dtype=np.int64),
            "decision_mean": np.asarray([0.0], dtype=np.float32),
            "decision_scale": np.asarray([1.0], dtype=np.float32),
            "decision_weight": np.asarray([1.0], dtype=np.float32),
            "state_dict": _zero_state(network),
        },
        artifact,
    )
    reranker = CooperativeNeuralDecisionReranker(artifact, device="cpu")
    ranked = reranker.rank(np.asarray([0.2, 0.0], dtype=np.float32), top_k=2)
    returned = ranked.fluxes[0]
    assert all(any(np.allclose(row, exact) for exact in exact_fluxes) for row in returned)


def test_batched_qp_repairs_bounds_and_shared_supply_with_convex_weights():
    projector = BatchedCooperativeQpProjector(
        species_names=["a"],
        shared_metabolite_ids=["x_e"],
        shared_terms=[(0, 0, 1, 1.0)],
        device="cpu",
        maximum_iterations=1000,
        normalized_tolerance=1e-6,
        bound_tolerance=2e-3,
        shared_tolerance=2e-3,
    )
    candidates = np.asarray(
        [
            [[2.0, 2.0], [0.0, 0.0]],
            [[1.5, 1.5], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    result = projector.project(
        candidates,
        lower_bounds=np.zeros((2, 2), dtype=np.float32),
        upper_bounds=np.ones((2, 2), dtype=np.float32),
        biomass_g_l=np.ones((2, 1), dtype=np.float32),
        shared_supply=np.ones((2, 1), dtype=np.float32),
    )
    assert np.all(result.feasible)
    assert np.allclose(result.weights.sum(axis=1), 1.0, atol=1e-6)
    assert np.max(result.fluxes) <= 1.002
    assert np.min(result.fluxes) >= -1e-7


def test_batched_qp_masks_infeasible_reference_weights_and_blends_feasible_rows():
    projector = BatchedCooperativeQpProjector(
        species_names=["a"],
        shared_metabolite_ids=["x_e"],
        shared_terms=[(0, 0, 1, 1.0)],
        device="cpu",
    )
    candidates = np.asarray(
        [[[0.2, 0.2], [0.8, 0.8], [2.0, 2.0]]], dtype=np.float32
    )
    result = projector.project(
        candidates,
        lower_bounds=np.zeros((1, 2), dtype=np.float32),
        upper_bounds=np.ones((1, 2), dtype=np.float32),
        biomass_g_l=np.ones((1, 1), dtype=np.float32),
        shared_supply=np.ones((1, 1), dtype=np.float32),
        reference_weights=np.asarray([[0.25, 0.25, 0.5]], dtype=np.float32),
    )
    assert result.feasible[0]
    assert result.weights[0] == pytest.approx([0.5, 0.5, 0.0])
    assert result.fluxes[0] == pytest.approx([0.5, 0.5])
