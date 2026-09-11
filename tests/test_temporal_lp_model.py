"""Causality, state ownership and reproducible weights-only model artifacts."""
import copy
import io

import pytest
import torch

from src.temporal_lp_model import (TemporalLPConfig, TemporalLPModel,
                                   TemporalLPSession, model_from_checkpoint,
                                   temporal_checkpoint)


def model(kind="gru", *, residual=False, layers=2):
    torch.manual_seed(716)
    return TemporalLPModel(TemporalLPConfig(kind, 5, 3, 7, layers, residual))


def inputs(batch=3, steps=6):
    generator = torch.Generator().manual_seed(717)
    return (torch.randn(batch, steps, 5, generator=generator),
            torch.randn(batch, steps, 3, generator=generator))


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_forward_shapes_and_gradients(kind):
    network = model(kind)
    features, previous = inputs()
    predicted, hidden = network(features, previous)
    assert predicted.shape == (3, 6, 3)
    if kind == "gru":
        assert hidden.shape == (2, 3, 7)
    else:
        assert hidden is None
    predicted.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in network.parameters())
    assert network.parameter_count == sum(item.numel() for item in network.parameters())


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_future_changes_cannot_modify_past_outputs(kind):
    network = model(kind).eval()
    features, previous = inputs()
    expected, _ = network(features, previous)
    changed_features, changed_previous = features.clone(), previous.clone()
    changed_features[:, 3:] += 1000
    changed_previous[:, 3:] -= 1000
    actual, _ = network(changed_features, changed_previous)
    torch.testing.assert_close(actual[:, :3], expected[:, :3], rtol=0, atol=0)


def test_mlp_is_time_pointwise_and_gru_is_history_sensitive():
    features, previous = inputs()
    changed = features.clone()
    changed[:, 0] += 10
    for kind in ("mlp", "gru"):
        network = model(kind).eval()
        expected, _ = network(features, previous)
        actual, _ = network(changed, previous)
        if kind == "mlp":
            torch.testing.assert_close(actual[:, 1:], expected[:, 1:], rtol=0, atol=0)
        else:
            assert not torch.allclose(actual[:, 1:], expected[:, 1:])


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_untrained_residual_head_is_previous_solution_baseline(kind):
    network = model(kind, residual=True)
    features, previous = inputs()
    actual, _ = network(features, previous)
    torch.testing.assert_close(actual, previous, rtol=0, atol=0)


def test_gru_chunking_matches_complete_causal_sequence():
    network = model().eval()
    features, previous = inputs()
    complete, final_hidden = network(features, previous)
    first, hidden = network(features[:, :2], previous[:, :2])
    second, hidden = network(features[:, 2:], previous[:, 2:], hidden)
    torch.testing.assert_close(torch.cat((first, second), dim=1), complete)
    torch.testing.assert_close(hidden, final_hidden)


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_session_reordering_matches_unbatched_environment_histories(kind):
    network = model(kind).eval()
    batched = TemporalLPSession(network)
    independent = [TemporalLPSession(network) for _ in range(3)]
    features, previous = inputs()
    for step in range(6):
        order = [2, 0, 1] if step % 2 else [1, 2, 0]
        actual = batched.predict(order, features[order, step], previous[order, step])
        for row, environment in enumerate(order):
            expected = independent[environment].predict(
                [environment], features[environment:environment+1, step],
                previous[environment:environment+1, step])
            torch.testing.assert_close(actual[row], expected[0], rtol=1e-5, atol=1e-7)


def test_session_selective_reset_and_new_ids_do_not_mix_environments():
    network = model().eval()
    session = TemporalLPSession(network)
    features, previous = inputs()
    session.predict([11, "second", 33], features[:, 0], previous[:, 0])
    before = session.hidden_for([11, "second", 33])
    session.reset(["second", "absent"])
    after = session.hidden_for([11, "second", 33, "new"])
    torch.testing.assert_close(after[:, 0], before[:, 0])
    torch.testing.assert_close(after[:, 2], before[:, 2])
    assert torch.count_nonzero(after[:, 1]) == 0
    assert torch.count_nonzero(after[:, 3]) == 0
    actual = session.predict(["second"], features[1:2, 1], previous[1:2, 1])
    expected = TemporalLPSession(network).predict(["second"], features[1:2, 1], previous[1:2, 1])
    torch.testing.assert_close(actual, expected)
    session.reset()
    assert session.environment_ids == ()


def test_snapshot_ownership_and_noncommitting_proposals():
    session = TemporalLPSession(model().eval())
    features, previous = inputs()
    session.predict([1, 2, 3], features[:, 0], previous[:, 0])
    before = session.hidden_for([1, 2, 3])
    snapshot = session.hidden_for([1, 2, 3])
    snapshot.fill_(100)
    torch.testing.assert_close(session.hidden_for([1, 2, 3]), before)
    trial = session.predict([1, 2, 3], features[:, 1], previous[:, 1], commit=False)
    torch.testing.assert_close(session.hidden_for([1, 2, 3]), before)
    committed = session.predict([1, 2, 3], features[:, 1], previous[:, 1])
    torch.testing.assert_close(trial, committed)
    assert not torch.equal(session.hidden_for([1, 2, 3]), before)


@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_propose_then_commit_matches_predict_without_early_mutation(kind):
    network = model(kind).eval()
    transaction = TemporalLPSession(network)
    immediate = TemporalLPSession(network)
    features, previous = inputs()
    for step in range(3):
        order = [2, 0, 1] if step % 2 else [0, 1, 2]
        before = transaction.hidden_for(order)
        prediction, hidden = transaction.propose(iter(order), features[order, step], previous[order, step])
        if before is not None:
            torch.testing.assert_close(transaction.hidden_for(order), before)
        expected = immediate.predict(iter(order), features[order, step], previous[order, step])
        torch.testing.assert_close(prediction, expected)
        transaction.commit(iter(order), hidden)
        if hidden is not None:
            torch.testing.assert_close(transaction.hidden_for(order), immediate.hidden_for(order))
        else:
            assert transaction.environment_ids == ()


def test_downstream_exception_discards_proposal_and_retry_preserves_history():
    session = TemporalLPSession(model().eval())
    features, previous = inputs()
    ids = [11, 29, 53]
    session.predict(ids, features[:, 0], previous[:, 0])
    before = session.hidden_for(ids)
    proposal, candidate_hidden = session.propose(ids, features[:, 1], previous[:, 1])
    with pytest.raises(RuntimeError, match="downstream GPU failure"):
        # A service commits only after this downstream operation returns.
        raise RuntimeError("downstream GPU failure")
    torch.testing.assert_close(session.hidden_for(ids), before)
    retry, retry_hidden = session.propose(ids, features[:, 1], previous[:, 1])
    torch.testing.assert_close(retry, proposal, rtol=0, atol=0)
    torch.testing.assert_close(retry_hidden, candidate_hidden, rtol=0, atol=0)
    session.commit(ids, retry_hidden)
    assert not torch.equal(session.hidden_for(ids), before)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "none", "duplicate_ids", "boolean_ids"])
def test_invalid_commit_is_rejected_before_any_environment_changes(bad):
    session = TemporalLPSession(model().eval())
    features, previous = inputs()
    ids = [11, 29, 53]
    session.predict(ids, features[:, 0], previous[:, 0])
    before = session.hidden_for(ids)
    _, candidate = session.propose(ids, features[:, 1], previous[:, 1])
    # Clone outside inference_mode to allow deliberate invalid input mutation.
    candidate = candidate.clone()
    commit_ids = ids
    if bad == "shape":
        candidate = candidate[:, :2]
    elif bad == "dtype":
        candidate = candidate.double()
    elif bad == "nan":
        candidate[0, 1, 0] = float("nan")
    elif bad == "none":
        candidate = None
    elif bad == "duplicate_ids":
        commit_ids = [11, 11, 53]
    else:
        commit_ids = [True, 29, 53]
    with pytest.raises(ValueError):
        session.commit(commit_ids, candidate)
    torch.testing.assert_close(session.hidden_for(ids), before)


def test_commit_owns_state_without_aliasing_caller_candidate():
    session = TemporalLPSession(model().eval())
    features, previous = inputs()
    _, proposed = session.propose([1, 2, 3], features[:, 0], previous[:, 0])
    candidate = proposed.clone()
    session.commit([1, 2, 3], candidate)
    expected = session.hidden_for([1, 2, 3])
    candidate.fill_(0)
    torch.testing.assert_close(session.hidden_for([1, 2, 3]), expected)


def test_commit_rejects_mlp_hidden_or_train_mode():
    session = TemporalLPSession(model("mlp").eval())
    with pytest.raises(ValueError, match="MLP"):
        session.commit([1], torch.zeros(2, 1, 7))
    session.model.train()
    with pytest.raises(ValueError, match="eval"):
        session.commit([1], None)


@pytest.mark.parametrize("field", ["features", "previous_latent", "hidden"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_forward_input_rejected(field, invalid):
    network = model()
    features, previous = inputs()
    hidden = network.initial_hidden(3)
    arguments = {"features": features, "previous_latent": previous, "hidden": hidden}
    arguments[field].view(-1)[0] = invalid
    with pytest.raises(ValueError, match="non-finite"):
        network(**arguments)


def test_failed_session_call_does_not_advance_hidden():
    session = TemporalLPSession(model().eval())
    features, previous = inputs()
    session.predict([1, 2, 3], features[:, 0], previous[:, 0])
    before = session.hidden_for([1, 2, 3])
    features[1, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        session.predict([1, 2, 3], features[:, 1], previous[:, 1])
    torch.testing.assert_close(session.hidden_for([1, 2, 3]), before)


@pytest.mark.parametrize("changes", [dict(kind="rnn"), dict(feature_dim=0),
    dict(latent_dim=True), dict(hidden_dim=-1), dict(num_layers=1.2), dict(residual=1)])
def test_invalid_configuration_rejected(changes):
    arguments = dict(kind="gru", feature_dim=5, latent_dim=3)
    arguments.update(changes)
    with pytest.raises(ValueError):
        TemporalLPConfig(**arguments)


def test_forward_dimensions_dtype_and_mlp_hidden_rejected():
    features, previous = inputs()
    network = model()
    bad_arguments = [(features[:, 0], previous, None),
                     (features[:, :, :4], previous, None),
                     (features, previous[:2], None),
                     (features[:, :0], previous[:, :0], None),
                     (features.double(), previous, None),
                     (features, previous, torch.zeros(1, 3, 7))]
    for args in bad_arguments:
        with pytest.raises(ValueError):
            network(*args)
    with pytest.raises(ValueError, match="MLP"):
        model("mlp")(features, previous, torch.zeros(2, 3, 7))


@pytest.mark.parametrize("ids", [[1, 1, 2], [True, 2, 3], [None, 2, 3], "abc", []])
def test_invalid_session_ids_rejected(ids):
    session = TemporalLPSession(model().eval())
    features, previous = inputs()
    with pytest.raises(ValueError):
        session.predict(ids, features[:, 0], previous[:, 0])


def test_session_requires_eval_and_reset_after_dtype_change():
    network = model()
    with pytest.raises(ValueError, match="eval"):
        TemporalLPSession(network)
    session = TemporalLPSession(network.eval())
    features, previous = inputs()
    session.predict([1, 2, 3], features[:, 0], previous[:, 0])
    network.double()
    with pytest.raises(ValueError, match="reset"):
        session.predict([1, 2, 3], features[:, 1].double(), previous[:, 1].double())
    session.reset()
    actual = session.predict([1, 2, 3], features[:, 1].double(), previous[:, 1].double())
    assert actual.dtype == torch.float64
    network.train()
    with pytest.raises(ValueError, match="eval"):
        session.predict([1, 2, 3], features[:, 1].double(), previous[:, 1].double())


@pytest.mark.parametrize("kind", ["gru", "mlp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weights_only_artifact_roundtrip(kind, dtype):
    network = model(kind).to(dtype=dtype).eval()
    features, previous = [value.to(dtype=dtype) for value in inputs()]
    expected, expected_hidden = network(features, previous)
    payload = temporal_checkpoint(network, {"training_only": True, "seeds": [1, 2],
                                             "feature_schema": "fixture_v1"})
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    restored_payload = torch.load(buffer, weights_only=True, map_location="cpu")
    restored = model_from_checkpoint(restored_payload, expected_config=network.config)
    actual, actual_hidden = restored(features, previous)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if expected_hidden is not None:
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    assert not restored.training
    assert next(restored.parameters()).dtype == dtype


@pytest.mark.parametrize("mutation", ["extra_architecture", "wrong_kind", "wrong_dim",
    "missing_weight", "weight_shape", "weight_nan", "weight_dtype", "mixed_dtype",
    "unknown_format", "wrong_version", "metadata_nan", "metadata_object", "extra_field"])
def test_artifact_architecture_weights_and_metadata_validation(mutation):
    payload = temporal_checkpoint(model())
    key = next(iter(payload["state_dict"]))
    if mutation == "extra_architecture":
        payload["architecture"]["bidirectional"] = True
    elif mutation == "wrong_kind":
        payload["architecture"]["kind"] = "mlp"
    elif mutation == "wrong_dim":
        payload["architecture"]["latent_dim"] = 4
    elif mutation == "missing_weight":
        del payload["state_dict"][key]
    elif mutation == "weight_shape":
        payload["state_dict"][key] = torch.zeros(2)
    elif mutation == "weight_nan":
        payload["state_dict"][key].view(-1)[0] = float("nan")
    elif mutation == "weight_dtype":
        payload["state_dict"][key] = payload["state_dict"][key].long()
    elif mutation == "mixed_dtype":
        payload["state_dict"][key] = payload["state_dict"][key].double()
    elif mutation == "unknown_format":
        payload["format"] = "unknown"
    elif mutation == "wrong_version":
        payload["version"] = True
    elif mutation == "metadata_nan":
        payload["metadata"]["loss"] = float("nan")
    elif mutation == "metadata_object":
        payload["metadata"]["unsafe"] = object()
    else:
        payload["unexpected"] = 1
    with pytest.raises(ValueError):
        model_from_checkpoint(payload)


def test_artifact_expected_configuration_and_owned_snapshot():
    network = model()
    payload = temporal_checkpoint(network, {"seeds": [1]})
    key = next(iter(payload["state_dict"]))
    original = payload["state_dict"][key].clone()
    with torch.no_grad():
        next(network.parameters()).add_(1)
    torch.testing.assert_close(payload["state_dict"][key], original)
    with pytest.raises(ValueError, match="expected_config"):
        model_from_checkpoint(payload, expected_config=model("mlp").config)
    bad = copy.deepcopy(payload)
    bad["architecture"].pop("residual")
    with pytest.raises(ValueError, match="architecture fields"):
        model_from_checkpoint(bad)
