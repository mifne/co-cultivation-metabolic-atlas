"""CPU toy tests for multi-positive routing and leakage-free preprocessing."""

import builtins
import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from src.coverage_router import CoverageRouter, coverage_metrics, fit_coverage_router, trajectory_split


def _data():
    value = np.tile(np.array([-2., -1., 0., 1., 2.]), 3)
    features = np.stack([value, value**2, np.ones(15)*7.], axis=1).astype(np.float32)
    coverage = np.stack([value < -.5, value > .5, value < -1.5], axis=1)
    return features, coverage, np.repeat(np.arange(3), 5)


def _provenance():
    return dict(bank_sha256="1"*64, coverage_sha256="2"*64, feature_schema_sha256="3"*64,
                model_fingerprints={"OR16":"4"*64, "NS21":"5"*64},
                candidate_ids=["basis-a", "basis-b", "basis-c"],
                bank_validation_scope="Router split only; bank contains validation trajectory sources")


def _deployment_router():
    """Small deterministic artifact; unlike _fit this never imports Torch."""
    metadata = dict(format="multilabel_original_lp_coverage_router", version=1,
        architecture=dict(input_dim=3, candidate_count=3, hidden_dim=2,
                          selected_feature_count=2, activation="tanh"),
        provenance=_provenance())
    arrays = dict(
        indices=np.array([0, 2], dtype=np.int64),
        mean=np.array([0., .25], dtype=np.float32),
        scale=np.array([1., 2.], dtype=np.float32),
        w1=np.array([[1., -.5], [.25, 1.]], dtype=np.float32),
        b1=np.array([.1, -.2], dtype=np.float32),
        w2=np.array([[1., 0., -1.], [.5, -1., .25]], dtype=np.float32),
        b2=np.array([0., .2, -.1], dtype=np.float32),
    )
    return CoverageRouter(arrays, metadata)


def _fit(**overrides):
    x, coverage, ids = _data()
    args = dict(provenance=_provenance(), train_indices=np.arange(10),
                validation_indices=np.arange(10, 15), epochs=60, learning_rate=.08, seed=91)
    args.update(overrides)
    return fit_coverage_router(x, coverage, ids, **args)


@pytest.mark.parametrize("hidden_dim", [0, 8])
def test_linear_and_mlp_learn_multiple_valid_candidates_without_fake_miss_labels(hidden_dim):
    router, report = _fit(hidden_dim=hidden_dim)
    assert report["training_loss_final"] < report["training_loss_initial"]
    assert report["training_metrics"]["full_bank_covered_rows"] == 8
    assert report["metadata"]["training"]["all_false_training_rows"] == 2
    assert report["metadata"]["training"]["positive_counts"] == [4, 4, 2]
    assert report["validation_metrics"]["capture_at_k"]["1"]["conditional_capture_rate"] >= .75
    assert report["validation_metrics"]["full_bank_coverage"] == .8
    assert "not external-seed validation" in report["validation_warning"]
    assert router.rank(_data()[0], k=2).shape == (15, 2)
    assert router.arrays["w1"].dtype == np.float32
    assert report["numpy_torch_max_abs_logit_difference"] < 2e-5


def test_bce_receives_all_true_labels_and_zero_positive_rows_unchanged(monkeypatch):
    import torch
    original = torch.nn.functional.binary_cross_entropy_with_logits
    targets = []
    def capture(input, target, **kwargs):
        targets.append(target.detach().cpu().numpy().copy())
        return original(input, target, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", capture)
    _fit(epochs=1)
    np.testing.assert_array_equal(targets[0], _data()[1][:10].astype(np.float32))
    assert targets[0][0].tolist() == [1., 0., 1.]
    assert targets[0][2].tolist() == [0., 0., 0.]


def test_metrics_distinguish_router_loss_from_bank_misses_and_multiple_positives():
    coverage = np.array([[True, True, False], [False, False, True], [False, False, False]])
    rank = np.array([[1, 0, 2], [0, 2, 1], [1, 2, 0]])
    metrics = coverage_metrics(coverage, rank, top_k=(1, 2, 3))
    assert metrics["full_bank_coverage"] == pytest.approx(2/3)
    assert metrics["uncovered_rows"] == 1
    assert metrics["capture_at_k"]["1"]["capture_rate"] == pytest.approx(1/3)
    assert metrics["capture_at_k"]["1"]["conditional_capture_rate"] == .5
    assert metrics["capture_at_k"]["1"]["full_bank_coverage_gap"] == pytest.approx(1/3)
    assert metrics["capture_at_k"]["2"]["conditional_capture_rate"] == 1.
    assert metrics["capture_at_k"]["2"]["capture_rate"] == pytest.approx(2/3)


def test_all_uncovered_evaluation_has_null_conditional_capture_not_fake_success():
    metrics = coverage_metrics(np.zeros((2, 3), dtype=bool), np.tile(np.arange(3), (2, 1)))
    assert metrics["full_bank_covered_rows"] == 0
    assert metrics["capture_at_k"]["1"]["capture_rate"] == 0.
    assert metrics["capture_at_k"]["1"]["conditional_capture_rate"] is None
    json.dumps(metrics, allow_nan=False)


def test_trajectory_partition_is_deterministic_and_never_splits_a_seed():
    ids = [1, "b", 1, "c", "b", "d", "c", "d"]
    train, validation = trajectory_split(ids, seed=15)
    repeated = trajectory_split(ids, seed=15)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(validation, repeated[1])
    assert not {ids[i] for i in train} & {ids[i] for i in validation}
    assert sorted(np.r_[train, validation]) == list(range(len(ids)))


def test_validation_values_do_not_influence_feature_selection_scaling_or_weights():
    first, _ = _fit(epochs=4)
    x, coverage, ids = _data()
    x[10:, 0] += 100.
    x[10:, 2] = np.arange(5)*1000.  # previously constant TRAINING feature
    second, report = fit_coverage_router(x, coverage, ids, provenance=_provenance(),
        train_indices=np.arange(10), validation_indices=np.arange(10, 15),
        epochs=4, learning_rate=.08, seed=91)
    assert second.arrays["indices"].tolist() == [0, 1]
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
    assert report["metadata"]["training"]["positive_weights"] == [1.5, 1.5, 4.]


def test_positive_weight_cap_is_training_only_and_recorded():
    router, report = _fit(epochs=1, positive_weight_cap=2.)
    assert router.metadata["training"]["positive_weights"] == [1.5, 1.5, 2.]
    assert report["metadata"]["training"]["positive_weight_cap"] == 2.


@pytest.mark.parametrize("hidden_dim", [0, 8])
def test_explicit_all_training_refit_has_no_validation_and_uses_every_training_row(hidden_dim, tmp_path):
    x, coverage, ids = _data()
    # A formerly constant feature varies only in the third training trajectory.
    x[10:, 2] += np.arange(5)
    router, report = fit_coverage_router(x, coverage, ids, provenance=_provenance(),
        refit_all=True, epochs=3, hidden_dim=hidden_dim, seed=91)
    training = router.metadata["training"]
    assert training["refit_all"] is True
    assert training["train_rows"] == list(range(15))
    assert training["validation_rows"] == []
    assert training["train_trajectory_ids"] == [0, 1, 2]
    assert training["validation_trajectory_ids"] == []
    assert training["positive_counts"] == [6, 6, 3]
    assert training["all_false_training_rows"] == 3
    assert router.arrays["indices"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(router.arrays["mean"], x.astype(np.float64).mean(axis=0))
    assert report["status"] == "trained_all_training_refit"
    assert report["validation_metrics"] is None
    assert report["training_metrics"]["rows"] == 15
    assert "not independent evaluation" in report["validation_warning"]
    assert report["validation_warning"] == router.metadata["validation_warning"]
    json.dumps(report, allow_nan=False)
    saved = router.save(tmp_path/"refit.npz")
    restored = CoverageRouter.load(saved["path"], expected_sha256=saved["sha256"],
        expected_provenance=_provenance())
    assert restored.metadata["training"]["refit_all"] is True
    np.testing.assert_array_equal(restored.rank(x), router.rank(x))


def test_refit_all_is_explicit_and_can_fit_one_training_trajectory():
    x, coverage, _ = _data()
    ids = ["single-training-trajectory"]*len(x)
    with pytest.raises(ValueError, match="At least two"):
        fit_coverage_router(x, coverage, ids, provenance=_provenance(), epochs=1)
    _, report = fit_coverage_router(x, coverage, ids, provenance=_provenance(),
        refit_all=True, epochs=1)
    assert report["validation_metrics"] is None


@pytest.mark.parametrize("bad", ["partition", "non_boolean", "coverage", "provenance", "top_k", "all_false"])
def test_refit_all_preserves_input_guards_and_rejects_ambiguous_partition(bad):
    x, coverage, ids = _data()
    args = dict(provenance=_provenance(), refit_all=True, epochs=1)
    if bad == "partition":
        args.update(train_indices=np.arange(15), validation_indices=np.empty(0, dtype=np.int64))
    elif bad == "non_boolean":
        args["refit_all"] = 1
    elif bad == "coverage":
        coverage = coverage.astype(np.float32)
    elif bad == "provenance":
        args["provenance"] = {}
    elif bad == "top_k":
        args["top_k"] = [4]
    else:
        coverage[:] = False
    with pytest.raises(ValueError):
        fit_coverage_router(x, coverage, ids, **args)


def test_npz_roundtrip_checks_sha_provenance_and_requires_no_torch_import(tmp_path, monkeypatch):
    router, _ = _fit(epochs=2, hidden_dim=4)
    path = tmp_path/"router.npz"
    saved = router.save(path)
    assert saved["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Deployment imported Torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    loaded = CoverageRouter.load(path, expected_sha256=saved["sha256"], expected_provenance=_provenance())
    np.testing.assert_array_equal(loaded.rank(_data()[0]), router.rank(_data()[0]))
    np.testing.assert_array_equal(loaded.to_device(np).logits(_data()[0]), router.logits(_data()[0]))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        CoverageRouter.load(path, expected_sha256="0"*64, expected_provenance=_provenance())
    changed = _provenance()
    changed["candidate_ids"].reverse()
    with pytest.raises(ValueError, match="candidate order mismatch"):
        CoverageRouter.load(path, expected_sha256=saved["sha256"], expected_provenance=changed)
    with pytest.raises(FileExistsError):
        router.save(path)


def test_module_import_does_not_load_torch():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import src.coverage_router; assert 'torch' not in sys.modules"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutation", [
    lambda metadata: metadata["architecture"].update(hidden_dim=5),
    lambda metadata: metadata["architecture"].update(selected_feature_count=999),
    lambda metadata: metadata["architecture"].update(activation="relu"),
    lambda metadata: metadata["provenance"].update(bank_sha256="wrong"),
    lambda metadata: metadata["provenance"].update(candidate_ids=["duplicate"]*3),
])
def test_malformed_architecture_or_provenance_is_rejected(mutation):
    router, _ = _fit(epochs=1)
    metadata = copy.deepcopy(router.metadata)
    mutation(metadata)
    with pytest.raises(ValueError):
        CoverageRouter(router.arrays, metadata)


def test_invalid_numeric_artifact_and_inference_are_rejected():
    router, _ = _fit(epochs=1)
    arrays = {name: value.copy() for name, value in router.arrays.items()}
    arrays["w1"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        CoverageRouter(arrays, router.metadata)
    with pytest.raises(ValueError, match="finite"):
        router.rank(np.full((1, 3), np.nan))
    with pytest.raises(ValueError, match="wrong shape"):
        router.rank(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="exceeds"):
        router.rank(np.zeros((1, 3)), k=4)


def test_rank_with_validity_matches_strict_numpy_rank_for_finite_rows():
    router = _deployment_router()
    features = np.array([[0., 1., 2.], [-2., 0., .5], [3., -4., -1.]],
                        dtype=np.float32)
    order, valid = router.rank_with_validity(features, k=2)
    np.testing.assert_array_equal(order, router.rank(features, k=2))
    np.testing.assert_array_equal(valid, np.ones(len(features), dtype=bool))
    assert order.dtype.kind in "iu" and valid.dtype == np.bool_


def test_rank_with_validity_masks_bad_rows_but_returns_safe_candidate_ids():
    router = _deployment_router()
    features = np.array([[0., 1., 2.], [1., np.nan, 2.],
                         [np.inf, 0., 1.]], dtype=np.float32)
    order, valid = router.rank_with_validity(features)
    np.testing.assert_array_equal(valid, [True, False, False])
    assert order.shape == (3, 3)
    for row in order:
        np.testing.assert_array_equal(np.sort(row), np.arange(3))
    # The existing public API remains batch-wide fail-fast.
    with pytest.raises(ValueError, match="finite"):
        router.rank(features)


def test_rank_with_validity_detects_finite_fp32_standardization_overflow():
    base = _deployment_router()
    arrays = {name:value.copy() for name, value in base.arrays.items()}
    arrays["scale"][0] = np.float32(1e-30)
    router = CoverageRouter(arrays, base.metadata)
    features = np.array([[np.finfo(np.float32).max, 0., 0.],
                         [1., 0., 1.]], dtype=np.float32)
    order, valid = router.rank_with_validity(features, k=2)
    np.testing.assert_array_equal(valid, [False, True])
    assert np.all((order >= 0) & (order < router.candidate_count))
    assert len(np.unique(order[0])) == 2


def test_rank_with_validity_cupy_matches_numpy_and_masks_bad_rows_without_host_decision():
    cp = pytest.importorskip("cupy")
    router = _deployment_router()
    features = np.array([[0., 1., 2.], [-2., 0., .5],
                         [1., np.nan, 2.]], dtype=np.float32)
    expected_order, expected_valid = router.rank_with_validity(features, k=2)
    order, valid = router.to_device(cp).rank_with_validity(cp.asarray(features), k=2)
    assert isinstance(order, cp.ndarray) and isinstance(valid, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(order), expected_order)
    np.testing.assert_array_equal(cp.asnumpy(valid), expected_valid)

    arrays = {name:value.copy() for name, value in router.arrays.items()}
    arrays["scale"][0] = np.float32(1e-30)
    overflow = CoverageRouter(arrays, router.metadata).to_device(cp)
    values = cp.asarray([[np.finfo(np.float32).max, 0., 0.]], dtype=cp.float32)
    overflow_order, overflow_valid = overflow.rank_with_validity(values, k=2)
    assert cp.asnumpy(overflow_valid).tolist() == [False]
    result = cp.asnumpy(overflow_order)
    assert result.shape == (1, 2) and len(np.unique(result[0])) == 2
    assert np.all((result >= 0) & (result < 3))


@pytest.mark.parametrize("bad", ["leakage", "all_false", "float_labels", "nan_features", "missing_split"])
def test_invalid_training_dataset_or_partition_fails_before_training(bad):
    x, coverage, ids = _data()
    args = dict(provenance=_provenance(), train_indices=np.arange(10), validation_indices=np.arange(10, 15), epochs=1)
    if bad == "leakage":
        args.update(train_indices=np.arange(9), validation_indices=np.arange(9, 15))
    elif bad == "all_false":
        coverage[:] = False
    elif bad == "float_labels":
        coverage = coverage.astype(np.float32)
    elif bad == "nan_features":
        x[0, 0] = np.nan
    else:
        args.pop("validation_indices")
    with pytest.raises(ValueError):
        fit_coverage_router(x, coverage, ids, **args)


@pytest.mark.parametrize("rank", [np.array([[0, 0, 1]]), np.array([[0, 1, 3]]), np.array([[0., 1., 2.]])])
def test_malformed_candidate_rankings_rejected(rank):
    with pytest.raises(ValueError):
        coverage_metrics(np.array([[True, False, False]]), rank)
