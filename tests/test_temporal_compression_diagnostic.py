"""Oracle diagnostics distinguish metric error from original-LP acceptance."""
import copy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.diagnose_temporal_compression import diagnose_samples, error_metrics
from src.temporal_lp_data import TemporalLPCodec


def fixture(rank=3):
    problem = (csr_matrix([[1., 1.]]), np.array([1.]), np.array([0., 0.]),
               np.array([1., 1.]), np.array([1., 1.]), 1)
    x, y = np.array([.3, .7]), np.array([1.])
    arrays = dict(components=np.eye(3)[:rank], target_mean=np.zeros(3), target_scale=np.ones(3),
                  feature_indices=np.array([], dtype=np.int64), matrix_keys=np.array([], dtype=np.int64),
                  feature_mean=np.array([]), feature_scale=np.array([]))
    codec = TemporalLPCodec(arrays, dict(dimensions=[2, 1, 1], stage="exchange"))
    return codec, [(dict(step=1, environment_id=0), (problem, x, y))]


def test_full_rank_oracle_reconstruction_preserves_certificate_and_inputs():
    codec, samples = fixture()
    original = copy.deepcopy(samples)
    report = diagnose_samples(codec, samples)
    assert report["summary"]["reference_accepted"] == 1
    assert report["summary"]["reconstruction_accepted"] == 1
    assert report["compression"]["scaled_discarded_energy_fraction"] == 0
    assert "Oracle" in report["scope"]
    assert report["compression"]["training_singular_values"] is None
    for a, b in zip(samples[0][1][1:], original[0][1][1:]):
        np.testing.assert_array_equal(a, b)


def test_rank_loss_fails_original_certificate_without_relaxing_thresholds():
    codec, samples = fixture(rank=1)
    report = diagnose_samples(codec, samples)
    row = report["rows"][0]
    assert report["summary"]["reference_accepted"] == 1
    assert report["summary"]["reconstruction_accepted"] == 0
    assert row["reconstruction_certificate"]["primal_residual"] == pytest.approx(.7)
    assert row["reconstruction_certificate"]["relative_kkt_gap"] > 1e-7
    assert row["reference_x_reconstructed_y_certificate"]["primal_residual"] == 0
    assert row["reference_x_reconstructed_y_certificate"]["relative_kkt_gap"] == pytest.approx(1.)
    assert row["primal_error"]["error_abs_max"] == pytest.approx(.7)
    assert row["dual_error"]["error_abs_max"] == pytest.approx(1.)
    assert report["certificate_thresholds"] == dict(primal=1e-5, dual=1e-7, relative_kkt_gap=1e-7)


def test_oracle_projection_does_not_call_neural_codec_encode_or_decode(monkeypatch):
    codec, samples = fixture()
    def forbidden(*args):
        raise AssertionError("Diagnostic must use saved affine subspace independently of latent normalization")
    monkeypatch.setattr(codec, "encode", forbidden)
    monkeypatch.setattr(codec, "decode", forbidden)
    assert diagnose_samples(codec, samples)["summary"]["reconstruction_accepted"] == 1


@pytest.mark.parametrize("invalid", ["reference", "dimensions", "orthogonality", "empty"])
def test_invalid_diagnostic_inputs_rejected(invalid):
    codec, samples = fixture()
    if invalid == "reference":
        samples[0][1][1][0] = 2.
    elif invalid == "dimensions":
        codec.metadata["dimensions"][2] = 0
    elif invalid == "orthogonality":
        codec.arrays["components"][0, 0] = 2.
    else:
        samples = []
    with pytest.raises(ValueError):
        diagnose_samples(codec, samples)


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf])])
def test_nonfinite_metric_inputs_rejected(bad):
    with pytest.raises(ValueError, match="Finite"):
        error_metrics(np.array([1.]), bad, np.array([1.]))


def test_subset_spectra_are_evaluation_only_and_dimensionally_explicit():
    codec, samples = fixture(rank=1)
    report = diagnose_samples(codec, samples)
    compression = report["compression"]
    assert "selected evaluation" in compression["evaluation_spectrum_scope"]
    assert compression["reference_subset_spectrum"]["shape"] == [1, 3]
    assert compression["reference_primal_subset_spectrum"]["shape"] == [1, 2]
    assert compression["reference_dual_subset_spectrum"]["shape"] == [1, 1]
    assert compression["residual_primal_subset_spectrum"]["squared_singular_energy"] == pytest.approx(.7**2)
    assert compression["residual_dual_subset_spectrum"]["squared_singular_energy"] == pytest.approx(1.)
    expected_energy = .3**2+.7**2+1.
    assert compression["reference_subset_spectrum"]["squared_singular_energy"] == pytest.approx(expected_energy)
    assert compression["scaled_discarded_squared_energy"] == pytest.approx(.7**2+1.)
