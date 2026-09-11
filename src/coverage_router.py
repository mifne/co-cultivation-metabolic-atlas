"""Multi-label basis routing, with NumPy/CuPy inference and offline Torch fit.

Coverage labels are boolean ORIGINAL-LP certificate outcomes for every
state/candidate pair. A miss row is all negative, never a fabricated best
candidate. Scores only order proposals: they cannot certify or accept an LP.

The candidate bank may contain bases from validation trajectories. A router
trajectory split alone therefore does NOT establish bank-independent holdout
accuracy, biological validity, or speedup. External-seed final tests remain
necessary. Torch is imported only inside ``fit_coverage_router``; deployment
loads pickle-free NPZ arrays and uses NumPy or a caller-supplied CuPy module.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np


FORMAT = "multilabel_original_lp_coverage_router"
VERSION = 1
VALIDATION_WARNING = (
    "Router trajectory split only; candidate-bank independence is not established. "
    "The bank may include validation-trajectory bases. This is internal routing "
    "diagnosis, not external-seed validation or proof of accepted-LP speedup.")
REFIT_ALL_WARNING = (
    "All-training refit after architecture and fixed epochs have been selected; "
    "all supplied training rows are fitted, with no validation partition. "
    "Training metrics are resubstitution diagnostics, not independent evaluation "
    "or proof of accepted-LP speedup. External-seed evaluation remains necessary.")
_PROVENANCE_KEYS = ("bank_sha256", "coverage_sha256", "feature_schema_sha256",
                    "model_fingerprints", "candidate_ids")


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _provenance(value, candidate_count):
    if not isinstance(value, dict) or any(key not in value for key in _PROVENANCE_KEYS):
        raise ValueError("Complete bank/coverage/feature/model/candidate provenance is required")
    # JSON round trip rejects arbitrary pickle-bearing Python metadata.
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("Provenance must be finite JSON metadata") from error
    for key in _PROVENANCE_KEYS[:3]:
        _sha(result[key], key)
    models = result["model_fingerprints"]
    if not isinstance(models, dict) or not models or any(not isinstance(key, str) or not key for key in models):
        raise ValueError("model_fingerprints must be a nonempty named mapping")
    for name, digest in models.items():
        _sha(digest, f"model_fingerprints.{name}")
    ids = result["candidate_ids"]
    if (not isinstance(ids, list) or len(ids) != candidate_count
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != candidate_count):
        raise ValueError("candidate_ids must identify every candidate uniquely and in order")
    return result


def _trajectory_ids(values, count):
    ids = list(values)
    if len(ids) != count:
        raise ValueError("One trajectory ID is required per state")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, str)) for value in ids):
        raise ValueError("Trajectory IDs must be integers or strings")
    return [int(value) if isinstance(value, np.integer) else value for value in ids]


def trajectory_split(trajectory_ids, *, validation_fraction=.25, seed=0):
    """Return train/dev row indices without splitting any trajectory."""
    ids = _trajectory_ids(trajectory_ids, len(trajectory_ids))
    seed = _positive_integer(seed, "seed", minimum=0)
    if not np.isfinite(validation_fraction) or not 0. < validation_fraction < 1.:
        raise ValueError("validation_fraction must be strictly between zero and one")
    unique = list(dict.fromkeys(ids))
    if len(unique) < 2:
        raise ValueError("At least two independent trajectory IDs are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    validation_count = min(len(unique)-1, max(1, int(np.ceil(len(unique)*validation_fraction))))
    validation_ids = {unique[index] for index in shuffled[:validation_count]}
    train = np.array([index for index, value in enumerate(ids) if value not in validation_ids], dtype=np.int64)
    validation = np.array([index for index, value in enumerate(ids) if value in validation_ids], dtype=np.int64)
    return train, validation


def _partition(train, validation, ids):
    count = len(ids)
    checked = []
    for name, indices in (("train_indices", train), ("validation_indices", validation)):
        value = np.asarray(indices)
        if value.ndim != 1 or value.dtype.kind not in "iu" or not len(value):
            raise ValueError(f"{name} must be a nonempty integer vector")
        if np.any(value < 0) or np.any(value >= count) or len(np.unique(value)) != len(value):
            raise ValueError(f"{name} contains invalid or duplicate row indices")
        checked.append(value.astype(np.int64))
    train, validation = checked
    if not np.array_equal(np.sort(np.r_[train, validation]), np.arange(count)):
        raise ValueError("Train/validation indices must form a disjoint complete partition")
    if {ids[i] for i in train} & {ids[i] for i in validation}:
        raise ValueError("A trajectory cannot appear in both train and validation")
    return train, validation


def _coverage(value, rows=None):
    coverage = np.asarray(value)
    if (coverage.ndim != 2 or coverage.dtype != np.bool_ or min(coverage.shape) < 1
            or rows is not None and coverage.shape[0] != rows):
        raise ValueError("coverage must be nonempty boolean [state, candidate]")
    return coverage


def coverage_metrics(coverage, ranking, *, top_k=(1,)):
    """Measure captured states, not arbitrary single-label classification."""
    coverage = _coverage(coverage)
    ranking = np.asarray(ranking)
    rows, candidates = coverage.shape
    if (ranking.ndim != 2 or ranking.shape[0] != rows or ranking.dtype.kind not in "iu"
            or ranking.shape[1] < 1 or np.any(ranking < 0) or np.any(ranking >= candidates)):
        raise ValueError("ranking must contain valid integer candidate indices per state")
    if any(len(np.unique(row)) != len(row) for row in ranking):
        raise ValueError("Candidate rankings cannot contain duplicate indices")
    ks = sorted({_positive_integer(k, "top_k") for k in top_k})
    if not ks or ks[-1] > ranking.shape[1]:
        raise ValueError("top_k exceeds the supplied ranking width")
    coverable = coverage.any(axis=1)
    covered = int(coverable.sum())
    metrics = dict(rows=rows, candidates=candidates, full_bank_covered_rows=covered,
                   full_bank_coverage=covered/rows, uncovered_rows=rows-covered, capture_at_k={})
    for k in ks:
        hit = coverage[np.arange(rows)[:, None], ranking[:, :k]].any(axis=1)
        count = int(hit.sum())
        metrics["capture_at_k"][str(k)] = dict(captured_rows=count, capture_rate=count/rows,
            conditional_capture_rate=count/covered if covered else None,
            full_bank_coverage_gap=(covered-count)/rows,
            coverable_missed_rows=covered-count)
    return metrics


class CoverageRouter:
    """Validated FP32 affine/tanh routing, with no Torch deployment dependency."""

    def __init__(self, arrays, metadata):
        try:
            self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Router metadata must be finite JSON") from error
        if self.metadata.get("format") != FORMAT or self.metadata.get("version") != VERSION:
            raise ValueError("Unsupported router artifact format/version")
        architecture = self.metadata.get("architecture", {})
        self.input_dim = _positive_integer(architecture.get("input_dim"), "input_dim")
        self.candidate_count = _positive_integer(architecture.get("candidate_count"), "candidate_count")
        self.hidden_dim = _positive_integer(architecture.get("hidden_dim"), "hidden_dim", minimum=0)
        self.provenance = _provenance(self.metadata.get("provenance"), self.candidate_count)
        required = {"indices", "mean", "scale", "w1", "b1"} | ({"w2", "b2"} if self.hidden_dim else set())
        if not isinstance(arrays, dict) or set(arrays) != required:
            raise ValueError("Router weight names disagree with architecture")
        self.arrays = {name: np.array(value, copy=True) for name, value in arrays.items()}
        indices = self.arrays["indices"]
        if (indices.dtype != np.int64 or indices.ndim != 1 or not len(indices)
                or np.any(indices < 0) or np.any(indices >= self.input_dim)
                or np.any(np.diff(indices) <= 0)):
            raise ValueError("Feature indices must be sorted unique int64 within the input width")
        width = len(indices)
        if _positive_integer(architecture.get("selected_feature_count"), "selected_feature_count") != width:
            raise ValueError("Selected feature count disagrees with architecture")
        if architecture.get("activation") != ("tanh" if self.hidden_dim else "linear"):
            raise ValueError("Activation disagrees with router architecture")
        middle = self.hidden_dim or self.candidate_count
        shapes = dict(mean=(width,), scale=(width,), w1=(width, middle), b1=(middle,))
        if self.hidden_dim:
            shapes.update(w2=(self.hidden_dim, self.candidate_count), b2=(self.candidate_count,))
        for name, shape in shapes.items():
            value = self.arrays[name]
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid FP32 shape or finite values: {name}")
        if np.any(self.arrays["scale"] <= 0):
            raise ValueError("Feature scales must be positive")
        for value in self.arrays.values():
            value.setflags(write=False)

    def logits(self, features):
        return self.to_device(np).logits(features)

    def rank(self, features, k=None):
        return self.to_device(np).rank(features, k=k)

    def rank_with_validity(self, features, k=None):
        """Return a safe NumPy ranking and a per-row numerical-validity mask.

        This is the non-raising, row-wise counterpart used by asynchronous
        device pipelines.  It does not certify an LP: callers must AND the
        returned mask with the unchanged original-LP certificate.  The
        existing :meth:`logits` and :meth:`rank` APIs deliberately retain
        their batch-wide fail-fast behaviour.
        """
        return self.to_device(np).rank_with_validity(features, k=k)

    def to_device(self, xp):
        """Upload weights once with ``xp=cupy``; ``xp=numpy`` stays on CPU."""
        return _ArrayRouter(self, xp)

    def save(self, path):
        """Write a new pickle-free NPZ; return its external integrity digest."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if path.suffix != ".npz":
            raise ValueError("Router artifact requires an explicit .npz suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays,
            metadata=np.array(json.dumps(self.metadata, sort_keys=True, allow_nan=False)))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    @classmethod
    def load(cls, path, *, expected_sha256, expected_provenance):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != _sha(expected_sha256, "expected_sha256"):
            raise ValueError("Router artifact SHA256 mismatch")
        with np.load(path, allow_pickle=False) as data:
            if "metadata" not in data or data["metadata"].shape != () or data["metadata"].dtype.kind != "U":
                raise ValueError("Router requires scalar JSON metadata")
            metadata = json.loads(str(data["metadata"]))
            arrays = {name: data[name] for name in data.files if name != "metadata"}
        router = cls(arrays, metadata)
        expected = _provenance(expected_provenance, router.candidate_count)
        if any(router.provenance[key] != expected[key] for key in _PROVENANCE_KEYS):
            raise ValueError("Router bank, coverage, feature schema, models or candidate order mismatch")
        return router


class _ArrayRouter:
    def __init__(self, router, xp):
        self.xp = xp
        self.input_dim, self.candidate_count = router.input_dim, router.candidate_count
        self.hidden_dim = router.hidden_dim
        self.arrays = {name: xp.asarray(value) for name, value in router.arrays.items()}

    def logits(self, features):
        xp, a = self.xp, self.arrays
        x = xp.asarray(features, dtype=xp.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Router features have the wrong shape")
        if not bool(xp.isfinite(x).all()):
            raise ValueError("Router features must be finite")
        x = (x[:, a["indices"]]-a["mean"])/a["scale"]
        hidden = x @ a["w1"] + a["b1"]
        scores = xp.tanh(hidden) @ a["w2"] + a["b2"] if self.hidden_dim else hidden
        if not bool(xp.isfinite(scores).all()):
            raise ValueError("Router produced non-finite scores")
        return scores

    def rank(self, features, k=None):
        k = self.candidate_count if k is None else _positive_integer(k, "k")
        if k > self.candidate_count:
            raise ValueError("k exceeds candidate count")
        scores = self.logits(features)
        # CuPy's default sort is stable; NumPy requests the same tie policy.
        order = (np.argsort(-scores, axis=1, kind="stable") if self.xp is np
                 else self.xp.argsort(-scores, axis=1))
        return order[:, :k]

    def rank_with_validity(self, features, k=None):
        """Rank without a device-to-host synchronization.

        Nonfinite raw features, standardized features, hidden activations, or
        logits invalidate only their own row.  Invalid intermediates are
        replaced before subsequent arithmetic so the returned order remains a
        safe, in-range permutation suitable for GPU indexing.  ``row_valid``
        stays on ``xp``; it is not a confidence score and can only *reject* a
        later original-LP certificate.
        """
        k = self.candidate_count if k is None else _positive_integer(k, "k")
        if k > self.candidate_count:
            raise ValueError("k exceeds candidate count")
        xp, a = self.xp, self.arrays
        x = xp.asarray(features, dtype=xp.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Router features have the wrong shape")

        # Preserve the strict API's all-input-feature finite requirement, even
        # though only a fixed subset enters the learned affine map.
        finite = xp.isfinite(x)
        row_valid = finite.all(axis=1)
        safe_x = xp.where(finite, x, xp.float32(0.0))
        # NumPy reports host RuntimeWarnings for deliberately probed overflow;
        # CuPy has no errstate API and device arithmetic does not require it.
        numerical_errors = (np.errstate(over="ignore", invalid="ignore", divide="ignore")
                            if xp is np else nullcontext())
        with numerical_errors:
            normalized = (safe_x[:, a["indices"]] - a["mean"]) / a["scale"]
            finite = xp.isfinite(normalized)
            row_valid = row_valid & finite.all(axis=1)
            normalized = xp.where(finite, normalized, xp.float32(0.0))

            hidden = normalized @ a["w1"] + a["b1"]
            finite = xp.isfinite(hidden)
            row_valid = row_valid & finite.all(axis=1)
            hidden = xp.where(finite, hidden, xp.float32(0.0))

            if self.hidden_dim:
                scores = xp.tanh(hidden) @ a["w2"] + a["b2"]
            else:
                scores = hidden
        finite = xp.isfinite(scores)
        row_valid = row_valid & finite.all(axis=1)
        safe_scores = xp.where(finite, scores, xp.float32(0.0))
        order = (np.argsort(-safe_scores, axis=1, kind="stable") if xp is np
                 else xp.argsort(-safe_scores, axis=1))
        return order[:, :k], row_valid


def fit_coverage_router(features, coverage, trajectory_ids, *, provenance,
                        train_indices=None, validation_indices=None, hidden_dim=0,
                        epochs=200, learning_rate=.01, weight_decay=1e-4, seed=0,
                        device="cpu", top_k=None, positive_weight_cap=20., scale_floor=1e-4,
                        refit_all=False):
    """Fit linear/small-MLP BCE on a fully labeled candidate coverage matrix.

    Every candidate in an all-false row receives a negative label. Training
    positive weights use only training negative/positive counts, clipped to
    [1, cap]. Validation is reported after fixed epochs, never optimized on.
    Caller must evaluate every candidate to distinguish false from unknown.

    ``refit_all=True`` explicitly fits every supplied TRAINING row, with no
    validation metrics. Use only after selecting architecture and fixed epochs
    by separate internal diagnostics. Do not mix independent evaluation data
    into this input. Explicit partition arguments are disallowed in this mode.
    """
    started = time.perf_counter()
    x = np.asarray(features)
    if x.ndim != 2 or min(x.shape) < 1 or x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError("features must be a finite, nonempty real [state, feature] matrix")
    x = x.astype(np.float64)
    coverage = _coverage(coverage, len(x))
    ids = _trajectory_ids(trajectory_ids, len(x))
    provenance = _provenance(provenance, coverage.shape[1])
    epochs = _positive_integer(epochs, "epochs")
    seed = _positive_integer(seed, "seed", minimum=0)
    hidden_dim = _positive_integer(hidden_dim, "hidden_dim", minimum=0)
    for name, value, allow_zero in (("learning_rate", learning_rate, False), ("weight_decay", weight_decay, True),
                                    ("positive_weight_cap", positive_weight_cap, False), ("scale_floor", scale_floor, False)):
        if isinstance(value, bool) or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ValueError(f"{name} has an invalid value")
    if positive_weight_cap < 1:
        raise ValueError("positive_weight_cap must be at least one")
    if not isinstance(refit_all, (bool, np.bool_)):
        raise ValueError("refit_all must be a boolean")
    refit_all = bool(refit_all)
    if refit_all:
        if train_indices is not None or validation_indices is not None:
            raise ValueError("refit_all cannot be combined with explicit partition indices")
        train = np.arange(len(x), dtype=np.int64)
        validation = np.empty(0, dtype=np.int64)
    else:
        if (train_indices is None) != (validation_indices is None):
            raise ValueError("Supply both train_indices and validation_indices, or neither")
        if train_indices is None:
            train_indices, validation_indices = trajectory_split(ids, seed=seed)
        train, validation = _partition(train_indices, validation_indices, ids)
    if not coverage[train].any():
        raise ValueError("No certified positive training coverage; a useful router cannot be learned")
    # No validation extrema, duplicates or moments enter feature selection.
    indices = np.flatnonzero(np.ptp(x[train], axis=0) > 1e-6).astype(np.int64)
    if not len(indices):
        indices = np.array([0], dtype=np.int64)  # bias-only information remains
    mean = x[train][:, indices].mean(axis=0).astype(np.float32)
    scale = np.maximum(x[train][:, indices].std(axis=0), scale_floor).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Training feature scaling is non-finite in FP32")
    normalized = (x[:, indices].astype(np.float32)-mean)/scale
    if not np.isfinite(normalized).all():
        raise ValueError("Features cannot be represented by finite standardized FP32 values")
    positives = coverage[train].sum(axis=0)
    negatives = len(train)-positives
    weights = np.clip(negatives/np.maximum(positives, 1), 1., positive_weight_cap).astype(np.float32)
    ks = [k for k in (1, 2, 4) if k <= coverage.shape[1]] if top_k is None else list(top_k)
    # Validate evaluation K before paying for training.
    evaluation_rows = train if refit_all else validation
    coverage_metrics(coverage[evaluation_rows],
        np.tile(np.arange(coverage.shape[1]), (len(evaluation_rows), 1)), top_k=ks)

    import torch  # offline-only; never imported by deployment load/rank
    target_device = torch.device(device)
    cuda_devices = [] if target_device.type != "cuda" else [torch.cuda.current_device() if target_device.index is None else target_device.index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        layers = ([torch.nn.Linear(len(indices), hidden_dim), torch.nn.Tanh(),
                   torch.nn.Linear(hidden_dim, coverage.shape[1])] if hidden_dim
                  else [torch.nn.Linear(len(indices), coverage.shape[1])])
        network = torch.nn.Sequential(*layers).to(target_device)
        tx = torch.as_tensor(normalized[train], device=target_device)
        ty = torch.as_tensor(coverage[train].astype(np.float32), device=target_device)
        pos_weight = torch.as_tensor(weights, device=target_device)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_history = []
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(network(tx), ty, pos_weight=pos_weight)
            if not bool(torch.isfinite(loss).item()):
                raise ValueError("Non-finite coverage training loss")
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu()))
        network.eval()
        arrays = dict(indices=indices, mean=mean, scale=scale,
                      w1=network[0].weight.detach().cpu().numpy().T.copy(),
                      b1=network[0].bias.detach().cpu().numpy().copy())
        if hidden_dim:
            arrays.update(w2=network[2].weight.detach().cpu().numpy().T.copy(),
                          b2=network[2].bias.detach().cpu().numpy().copy())
        with torch.no_grad():
            torch_logits = network(torch.as_tensor(normalized, device=target_device)).cpu().numpy()

    data_hash = hashlib.sha256()
    for name, value in (("features", x), ("coverage", coverage), ("train", train), ("validation", validation)):
        array = np.ascontiguousarray(value)
        data_hash.update(json.dumps([name, list(array.shape), array.dtype.str]).encode())
        data_hash.update(array.tobytes())
    data_hash.update(json.dumps(ids, separators=(",", ":")).encode())
    metadata = dict(format=FORMAT, version=VERSION,
        architecture=dict(input_dim=x.shape[1], selected_feature_count=len(indices),
                          candidate_count=coverage.shape[1], hidden_dim=hidden_dim, activation="tanh" if hidden_dim else "linear"),
        provenance=provenance, input_data_and_split_sha256=data_hash.hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training=dict(seed=seed, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            refit_all=refit_all,
            device=str(target_device), torch_version=str(torch.__version__),
            train_rows=train.tolist(), validation_rows=validation.tolist(),
            train_trajectory_ids=list(dict.fromkeys(ids[index] for index in train)),
            validation_trajectory_ids=list(dict.fromkeys(ids[index] for index in validation)),
            feature_scaling="Training states only; input selected log-sign schema comes from caller",
            scale_floor=scale_floor, positive_weight_cap=positive_weight_cap,
            positive_counts=positives.tolist(), negative_counts=negatives.tolist(),
            positive_weights=weights.tolist(),
            loss="Independent per-candidate weighted BCE, averaged over all training state/candidate pairs",
            miss_row_policy="All-false rows are all negative, no best-rejected label",
            covered_training_rows=int(coverage[train].any(axis=1).sum()),
            all_false_training_rows=int((~coverage[train].any(axis=1)).sum())),
        validation_warning=REFIT_ALL_WARNING if refit_all else VALIDATION_WARNING,
        acceptance_scope="Ranking only; every selected candidate still requires an ORIGINAL-LP certificate")
    router = CoverageRouter(arrays, metadata)
    numpy_logits = router.logits(x)
    if not np.allclose(numpy_logits, torch_logits, atol=2e-5, rtol=2e-5):
        raise ValueError("Exported NumPy router differs from Torch inference")
    report = dict(status="trained_all_training_refit" if refit_all else "trained_internal_router", metadata=metadata,
        training_loss_initial=loss_history[0], training_loss_final=loss_history[-1],
        loss_history=loss_history, training_metrics=coverage_metrics(coverage[train], router.rank(x[train]), top_k=ks),
        validation_metrics=(None if refit_all else
            coverage_metrics(coverage[validation], router.rank(x[validation]), top_k=ks)),
        numpy_torch_max_abs_logit_difference=float(np.max(np.abs(numpy_logits-torch_logits))),
        validation_warning=metadata["validation_warning"], parameter_count=sum(arrays[name].size for name in arrays if name.startswith(("w", "b"))),
        training_seconds=time.perf_counter()-started)
    return router, report
