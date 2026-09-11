"""Causal latent proposals for repeated LPs; never an LP acceptance decision.

The GRU and pointwise MLP consume identical current features and *previous
certified* latent solutions.  Teacher-forced loss alone does not demonstrate
closed-loop accuracy or acceleration: decoded proposals still need the original
LP certificate and, where necessary, an independently measured correction.

Input/output finiteness checks deliberately synchronize CUDA tensors.  Include
these checks, latent decoding, certification and correction in online timings;
this module does not claim a synchronization-free, fully GPU-resident solver.
"""
from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalLPConfig:
    """A reproducible architecture; ``kind`` changes only the temporal core."""

    kind: str
    feature_dim: int
    latent_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    residual: bool = True

    def __post_init__(self):
        if self.kind not in ("gru", "mlp"):
            raise ValueError("kind must be 'gru' or 'mlp'")
        for name in ("feature_dim", "latent_dim", "hidden_dim", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.residual, bool):
            raise ValueError("residual must be a boolean")

    @classmethod
    def from_dict(cls, architecture):
        if not isinstance(architecture, dict):
            raise ValueError("architecture must be a dictionary")
        expected = {field.name for field in fields(cls)}
        if set(architecture) != expected:
            raise ValueError("architecture fields do not match TemporalLPConfig")
        return cls(**architecture)


def _finite(tensor, label):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains non-finite values")


class TemporalLPModel(nn.Module):
    """Predict a current latent, without observing the current reference solution.

    ``forward(features[B,T,F], previous_latent[B,T,L], hidden)`` returns
    ``(prediction[B,T,L], next_hidden)``.  Hidden has shape ``[layers,B,H]`` for
    the unidirectional GRU and is always ``None`` for the MLP.  Features and the
    previous certified latent must use the model's floating dtype and device.

    With ``residual=True``, the head predicts an additive update.  Its zero
    initialization makes the initial predictor exactly previous-latent reuse,
    so that the learned correction has a meaningful untrained baseline.
    """

    def __init__(self, config: TemporalLPConfig):
        super().__init__()
        if not isinstance(config, TemporalLPConfig):
            raise TypeError("config must be TemporalLPConfig")
        self.config = config
        input_dim = config.feature_dim + config.latent_dim
        if config.kind == "gru":
            self.core = nn.GRU(input_dim, config.hidden_dim,
                               num_layers=config.num_layers, batch_first=True,
                               bidirectional=False, dropout=0.0)
        else:
            layers = []
            for layer in range(config.num_layers):
                layers.extend((nn.Linear(input_dim if layer == 0 else config.hidden_dim,
                                         config.hidden_dim), nn.Tanh()))
            self.core = nn.Sequential(*layers)
        self.head = nn.Linear(config.hidden_dim, config.latent_dim)
        if config.residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_hidden(self, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.config.kind == "mlp":
            return None
        parameter = next(self.parameters())
        return parameter.new_zeros(self.config.num_layers, int(batch_size),
                                   self.config.hidden_dim)

    def _validate_inputs(self, features, previous_latent, hidden):
        parameter = next(self.parameters())
        config = self.config
        for label, tensor, dimension in (("features", features, config.feature_dim),
                                         ("previous_latent", previous_latent, config.latent_dim)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise ValueError(f"{label} must be a [batch, time, dimension] tensor")
            if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] != dimension:
                raise ValueError(f"{label} has invalid dimensions")
            if tensor.dtype != parameter.dtype or tensor.device != parameter.device:
                raise ValueError(f"{label} dtype/device must match model parameters")
            _finite(tensor, label)
        if features.shape[:2] != previous_latent.shape[:2]:
            raise ValueError("features and previous_latent must share batch/time dimensions")
        if hidden is not None:
            if config.kind != "gru":
                raise ValueError("MLP does not accept recurrent hidden state")
            expected = (config.num_layers, features.shape[0], config.hidden_dim)
            if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != expected:
                raise ValueError(f"hidden must have shape {expected}")
            if hidden.dtype != parameter.dtype or hidden.device != parameter.device:
                raise ValueError("hidden dtype/device must match model parameters")
            _finite(hidden, "hidden")

    def forward(self, features, previous_latent, hidden=None):
        self._validate_inputs(features, previous_latent, hidden)
        combined = torch.cat((features, previous_latent), dim=-1)
        if self.config.kind == "gru":
            representation, next_hidden = self.core(combined, hidden)
        else:
            representation, next_hidden = self.core(combined), None
        prediction = self.head(representation)
        if self.config.residual:
            prediction = prediction + previous_latent
        _finite(prediction, "prediction")
        if next_hidden is not None:
            _finite(next_hidden, "next_hidden")
        return prediction, next_hidden


def _environment_ids(environment_ids):
    """Restrict IDs to stable, serialization-friendly scalar values."""
    if isinstance(environment_ids, (str, bytes)):
        raise ValueError("environment_ids must be a sequence, not a string")
    try:
        values = list(environment_ids)
    except TypeError as error:
        raise ValueError("environment_ids must be an iterable") from error
    canonical = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, Integral)):
            raise ValueError("environment IDs must be strings or integers, not booleans")
        canonical.append(int(value) if isinstance(value, Integral) else value)
    if len(set(canonical)) != len(canonical):
        raise ValueError("environment_ids contains duplicate IDs")
    return canonical


class TemporalLPSession:
    """Inference-only, environment-keyed recurrent state.

    Use a separate session/model for each LP stage and reset an environment at
    every episode boundary.  Batch ordering may change freely: state follows
    IDs, never positions.  The caller supplies the previous *certified* latent;
    predicted latents are not silently fed back as if they were accepted.

    ``propose`` returns a prediction and candidate hidden state without any
    mutation.  Pass that hidden state to ``commit`` only after downstream
    processing returns normally; simply discard it on an exception.  A normal
    LP rejection may still advance observed-input history, if the caller's
    replay contract advances that physical step.  It must not enter the
    caller's certified-solution cache.

    Calls are serialized.  Commit with the same IDs in the same order used
    for the proposal, before resetting or advancing any of those environments;
    this API does not schedule concurrent or stale proposals.  A committed
    call advances each listed environment once.  Omitted environments remain
    untouched. ``predict(..., commit=False)`` remains a non-mutating convenience
    method for comparisons that do not need to retain the candidate hidden.
    """

    def __init__(self, model: TemporalLPModel):
        if not isinstance(model, TemporalLPModel):
            raise TypeError("model must be TemporalLPModel")
        if model.training:
            raise ValueError("call model.eval() before creating an inference session")
        self.model = model
        self._hidden = {}

    @property
    def environment_ids(self):
        return tuple(self._hidden)

    def reset(self, environment_ids=None):
        if environment_ids is None:
            self._hidden.clear()
        else:
            for environment_id in _environment_ids(environment_ids):
                self._hidden.pop(environment_id, None)

    def hidden_for(self, environment_ids):
        """Return an independent snapshot in the requested order, zeros if new."""
        ids = _environment_ids(environment_ids)
        if not ids:
            raise ValueError("environment_ids must not be empty")
        hidden = self.model.initial_hidden(len(ids))
        if hidden is None:
            return None
        for index, environment_id in enumerate(ids):
            state = self._hidden.get(environment_id)
            if state is not None:
                if state.dtype != hidden.dtype or state.device != hidden.device:
                    raise ValueError("model dtype/device changed; reset the session before reuse")
                hidden[:, index, :] = state
        return hidden

    @torch.inference_mode()
    def propose(self, environment_ids, features, previous_latent):
        """Return ``(prediction[B,L], next_hidden[layers,B,H] | None)``.

        No cache changes, including on failure.  The returned hidden tensor
        may be discarded if a GPU corrector raises; retrying then starts from
        exactly the previous committed input history.
        """
        if self.model.training:
            raise ValueError("inference session requires model.eval()")
        ids = _environment_ids(environment_ids)
        if not ids:
            raise ValueError("environment_ids must not be empty")
        for label, tensor in (("features", features), ("previous_latent", previous_latent)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                raise ValueError(f"session {label} must be a [batch, dimension] tensor")
            if tensor.shape[0] != len(ids):
                raise ValueError(f"session {label} batch must match environment IDs")
        hidden = self.hidden_for(ids)
        prediction, next_hidden = self.model(features.unsqueeze(1),
                                             previous_latent.unsqueeze(1), hidden)
        return prediction[:, 0, :], next_hidden

    @torch.inference_mode()
    def commit(self, environment_ids, next_hidden):
        """Validate all candidate state, then atomically own its environment rows.

        IDs must correspond to the candidate hidden's batch order.  GRU state
        must be finite and match the model's shape, dtype and device.  MLP
        proposals use ``None`` and commit no recurrent state.
        """
        if self.model.training:
            raise ValueError("inference session requires model.eval()")
        ids = _environment_ids(environment_ids)
        if not ids:
            raise ValueError("environment_ids must not be empty")
        if self.model.config.kind == "mlp":
            if next_hidden is not None:
                raise ValueError("MLP commit requires next_hidden=None")
            return
        config = self.model.config
        expected = (config.num_layers, len(ids), config.hidden_dim)
        if not isinstance(next_hidden, torch.Tensor) or tuple(next_hidden.shape) != expected:
            raise ValueError(f"next_hidden must have shape {expected}")
        parameter = next(self.model.parameters())
        if next_hidden.dtype != parameter.dtype or next_hidden.device != parameter.device:
            raise ValueError("next_hidden dtype/device must match model parameters")
        _finite(next_hidden, "next_hidden")
        # Build the entire owned update first: malformed/failed proposals must
        # not partially advance a batch, and no views retain a larger batch.
        updates = {environment_id: next_hidden[:, index, :].clone()
                   for index, environment_id in enumerate(ids)}
        self._hidden.update(updates)

    @torch.inference_mode()
    def predict(self, environment_ids, features, previous_latent, *, commit=True):
        """Backward-compatible immediate proposal, optionally committed."""
        if not isinstance(commit, bool):
            raise ValueError("commit must be a boolean")
        # A generator must not be consumed independently by propose/commit.
        ids = _environment_ids(environment_ids)
        prediction, next_hidden = self.propose(ids, features, previous_latent)
        if commit:
            self.commit(ids, next_hidden)
        return prediction


def _json_metadata(value, path="metadata"):
    """Copy only JSON-like finite metadata, not arbitrary pickle objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite metadata")
        return value
    if isinstance(value, list):
        return [_json_metadata(item, f"{path}[]") for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_metadata(item, f"{path}.{key}") for key, item in value.items()}
    raise ValueError(f"{path} must contain only JSON-like values")


def temporal_checkpoint(model: TemporalLPModel, metadata=None):
    """Return a weights-only-loadable payload; the caller owns file I/O.

    Training split, feature schema, PCA identity and data provenance belong in
    metadata and must be checked by the dataset/solver integration, not inferred
    from matching neural dimensions.  This payload is not a complete LP solver.
    """
    if not isinstance(model, TemporalLPModel):
        raise TypeError("model must be TemporalLPModel")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for name, value in weights.items():
        _finite(value, f"state_dict.{name}")
    return {"format": "temporal_lp_model", "version": 1,
            "architecture": asdict(model.config), "state_dict": weights,
            "metadata": _json_metadata(metadata)}


def model_from_checkpoint(payload, *, expected_config=None, device="cpu"):
    """Strictly restore an architecture and finite weights, in evaluation mode.

    Load files with ``torch.load(path, map_location='cpu', weights_only=True)``
    before calling this function.  Unexpected architecture fields, weight keys,
    shapes, or mixed dtypes are rejected rather than guessed or silently cast.
    """
    expected_keys = {"format", "version", "architecture", "state_dict", "metadata"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("invalid temporal checkpoint fields")
    if payload["format"] != "temporal_lp_model" or type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported temporal checkpoint format/version")
    config = TemporalLPConfig.from_dict(payload["architecture"])
    if expected_config is not None:
        if not isinstance(expected_config, TemporalLPConfig) or config != expected_config:
            raise ValueError("checkpoint architecture does not match expected_config")
    if not isinstance(payload["metadata"], dict):
        raise ValueError("metadata must be a dictionary")
    _json_metadata(payload["metadata"])
    weights = payload["state_dict"]
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("state_dict must be a non-empty tensor mapping")
    model = TemporalLPModel(config)
    expected_weights = model.state_dict()
    if set(weights) != set(expected_weights):
        raise ValueError("state_dict keys do not match checkpoint architecture")
    dtypes = set()
    for name, expected in expected_weights.items():
        value = weights[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"state_dict.{name} must be a dense tensor")
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(f"state_dict.{name} shape does not match architecture")
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"state_dict.{name} must have a supported floating dtype")
        _finite(value, f"state_dict.{name}")
        dtypes.add(value.dtype)
    if len(dtypes) != 1:
        raise ValueError("state_dict contains mixed parameter dtypes")
    model.to(device=device, dtype=next(iter(dtypes)))
    model.load_state_dict(weights, strict=True)
    return model.eval()
