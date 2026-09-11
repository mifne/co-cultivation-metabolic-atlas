import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.cooperative_surrogate import CooperativeFluxDictionary
from src.community_solver import (
    inverse_distance_flux_interpolation,
    surrogate_validation_status,
)


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_cooperative_surrogate_dictionary.py"
    spec = importlib.util.spec_from_file_location("cooperative_dictionary_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_decision_aware_weights_emphasize_limitation_and_pha_features():
    builder = _load_builder_module()
    rng = np.random.default_rng(4)
    metadata = {
        "context_layout": [
            ["biomass_g_l", 3],
            ["shared_supply_mmol_l_h", 2],
            ["flux_lower_bounds", 4],
            ["flux_upper_bounds", 4],
            ["objective_coefficients", 4],
        ],
        "shared_metabolite_ids": ["nh4_e", "o2_e"],
        "reaction_ids": ["R_Growth", "EX_pha_c", "R_ROXA", "ordinary"],
    }
    contexts = rng.normal(size=(64, 17)).astype(np.float32)
    fluxes = rng.normal(size=(64, 4)).astype(np.float32)
    features = np.arange(contexts.shape[1], dtype=np.int64)
    weights, summary = builder.decision_aware_feature_weights(
        contexts,
        fluxes,
        metadata,
        features,
        common_growth=fluxes[:, 0],
        aggregate_objective=fluxes[:, 1],
    )
    assert weights.shape == (17,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 1.0)
    assert weights[3] >= 4.0  # NH4 shared-supply feature
    assert weights[4] >= 4.0  # oxygen shared-supply feature
    assert summary["critical_feature_count"] > 0


def test_runtime_applies_artifact_feature_weights(tmp_path):
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "weighted.pt"
    torch.save(
        {
            "metadata": {
                "format_version": 1,
                "formulation": "exact_cooperative_shared_medium",
                "context_dimension": 2,
                "distance_threshold": 20.0,
            },
            "feature_indices": np.asarray([0, 1], dtype=np.int64),
            "feature_mean": np.zeros(2, dtype=np.float32),
            "feature_scale": np.ones(2, dtype=np.float32),
            "feature_weight": np.asarray([16.0, 1.0], dtype=np.float32),
            # Stored contexts already include sqrt(feature_weight).
            "normalized_contexts": np.asarray([[0.0, 1.0], [4.0, 0.0]], dtype=np.float32),
            "fluxes": np.asarray([[10.0], [20.0]], dtype=np.float32),
        },
        artifact,
    )
    dictionary = CooperativeFluxDictionary(artifact, device="cpu")
    ranked = dictionary.rank(np.asarray([0.45, 0.10], dtype=np.float32), top_k=1)
    assert ranked.indices[0, 0] == 0
    assert ranked.fluxes[0, 0, 0] == pytest.approx(10.0)


def test_legacy_artifact_defaults_to_uniform_weights(tmp_path):
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "legacy.pt"
    torch.save(
        {
            "metadata": {
                "format_version": 1,
                "formulation": "exact_cooperative_shared_medium",
                "context_dimension": 1,
                "distance_threshold": 10.0,
            },
            "feature_indices": np.asarray([0], dtype=np.int64),
            "feature_mean": np.zeros(1, dtype=np.float32),
            "feature_scale": np.ones(1, dtype=np.float32),
            "normalized_contexts": np.asarray([[0.0], [1.0]], dtype=np.float32),
            "fluxes": np.asarray([[1.0], [2.0]], dtype=np.float32),
        },
        artifact,
    )
    dictionary = CooperativeFluxDictionary(artifact, device="cpu")
    ranked = dictionary.rank(np.asarray([0.8], dtype=np.float32), top_k=1)
    assert ranked.indices[0, 0] == 1


def test_inverse_distance_interpolation_is_convex_and_prefers_near_candidate():
    result = inverse_distance_flux_interpolation(
        [
            (1.0, np.asarray([0.0, 2.0])),
            (2.0, np.asarray([4.0, 0.0])),
        ],
        power=2.0,
    )
    assert result == pytest.approx([0.8, 1.6])
    assert np.all(result >= 0.0)


def test_surrogate_requires_explicit_qualified_manifest(tmp_path):
    missing = tmp_path / "missing.json"
    assert surrogate_validation_status(missing) == (
        False,
        "validation_manifest_missing",
    )
    rejected = tmp_path / "rejected.json"
    rejected.write_text('{"status":"experimental_not_qualified"}', encoding="utf-8")
    assert surrogate_validation_status(rejected) == (
        False,
        "validation_status:experimental_not_qualified",
    )
    accepted = tmp_path / "accepted.json"
    accepted.write_text('{"status":"qualified"}', encoding="utf-8")
    assert surrogate_validation_status(accepted) == (True, "qualified")
