from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cobra
import numpy as np
import torch

from src.dfba_simulator import dFBASimulator
from src.fba_surrogate import (
    FbaSurrogateBackend,
    compute_nullspace_basis,
    extract_lp_features,
    make_network,
    save_surrogate_artifact,
)
from src.fba_surrogate_service import start_surrogate_service


def make_balanced_model():
    model = cobra.Model("balanced")
    substrate = cobra.Metabolite("a_c", compartment="c")
    source = cobra.Reaction("SOURCE_A")
    source.add_metabolites({substrate: 1.0})
    source.bounds = (0.0, 10.0)
    growth = cobra.Reaction("R_Growth")
    growth.add_metabolites({substrate: -1.0})
    growth.bounds = (0.0, 10.0)
    model.add_reactions([source, growth])
    model.objective = growth
    return model


def write_constant_artifact(tmp_path: Path, model, flux=2.0):
    basis = compute_nullspace_basis(model)
    network = make_network(basis, 3 * len(model.reactions) + 1, (8,))
    features = extract_lp_features(model)
    target = np.full(len(model.reactions), flux, dtype=np.float32)
    latent = target @ basis
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.x_mean.copy_(torch.from_numpy(features))
        network.x_scale.fill_(1.0)
        network.z_mean.copy_(torch.from_numpy(latent))
        network.z_scale.fill_(1.0)
    path = tmp_path / "balanced.pt"
    save_surrogate_artifact(path, model, network, basis, (8,))
    (tmp_path / "manifest.json").write_text(
        '{"artifact_version": 1, "species": {"balanced": "balanced.pt"}}'
    )
    return path


def test_nullspace_surrogate_conserves_mass_and_predicts_objective(tmp_path):
    model = make_balanced_model()
    write_constant_artifact(tmp_path, model)
    backend = FbaSurrogateBackend(
        {"balanced": model}, tmp_path, device="cpu", ood_threshold=100.0
    )
    prediction = backend.predict("balanced", model)
    assert prediction.accepted
    assert prediction.max_mass_balance_residual < 1e-6
    assert prediction.solution.objective_value == pytest.approx(2.0, abs=1e-6)
    assert prediction.solution.fluxes["SOURCE_A"] == pytest.approx(2.0, abs=1e-6)


def test_runtime_guard_rejects_bound_violation(tmp_path):
    model = make_balanced_model()
    write_constant_artifact(tmp_path, model)
    backend = FbaSurrogateBackend(
        {"balanced": model}, tmp_path, device="cpu", ood_threshold=100.0
    )
    model.reactions.SOURCE_A.upper_bound = 1.0
    prediction = backend.predict("balanced", model)
    assert not prediction.accepted
    assert prediction.reason == "bound_violation"


def test_dictionary_optimizer_uses_physical_guards_not_training_distance(tmp_path):
    model = make_balanced_model()
    candidates = np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    network = make_network(
        candidates,
        3 * len(model.reactions) + 1,
        (4,),
        decoder_mode="feasible_dictionary_optimizer",
    )
    path = tmp_path / "balanced.pt"
    save_surrogate_artifact(
        path,
        model,
        network,
        candidates,
        (4,),
        decoder_mode="feasible_dictionary_optimizer",
    )
    (tmp_path / "manifest.json").write_text(
        '{"artifact_version": 1, "species": {"balanced": "balanced.pt"}}'
    )
    backend = FbaSurrogateBackend(
        {"balanced": model}, tmp_path, device="cpu", ood_threshold=0.01
    )
    prediction = backend.predict("balanced", model)
    assert prediction.ood_score > 0.01
    assert prediction.accepted
    assert prediction.solution.objective_value == pytest.approx(2.0, abs=1e-6)


def test_simulator_falls_back_to_exact_highs_without_artifact(tmp_path):
    model = make_balanced_model()
    simulator = dFBASimulator(
        models={"balanced": model},
        initial_biomass={"balanced": 0.1},
        initial_metabolites={},
        solver_backend="surrogate",
        surrogate_dir=str(tmp_path),
        surrogate_device="cpu",
    )
    solution = simulator.solve_fba("balanced")
    assert solution is not None
    assert solution.status == "optimal"
    assert simulator.cpu_fallback_solves == 1
    assert simulator.surrogate_rejection_reasons == {"artifact_unavailable": 1}


def test_manager_batches_concurrent_environment_requests(tmp_path):
    model = make_balanced_model()
    write_constant_artifact(tmp_path, model)
    manager, service = start_surrogate_service(
        {"balanced": model},
        str(tmp_path),
        device="cpu",
        batch_window_ms=30.0,
        max_batch_size=8,
    )
    try:
        features = extract_lp_features(model)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(lambda _: service.predict("balanced", features), range(4))
            )
        assert all(result["accepted"] for result in results)
        assert service.diagnostics()["max_observed_batch"] >= 2
    finally:
        service.close()
        manager.shutdown()


def test_manager_predict_many_batches_environment_requests(tmp_path):
    model = make_balanced_model()
    write_constant_artifact(tmp_path, model)
    manager, service = start_surrogate_service(
        {"balanced": model},
        str(tmp_path),
        device="cpu",
        batch_window_ms=30.0,
        max_batch_size=8,
    )
    try:
        features = extract_lp_features(model)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: service.predict_many({"balanced": features}), range(4)
                )
            )
        assert all(result["balanced"]["accepted"] for result in results)
        diagnostics = service.diagnostics()
        assert diagnostics["max_observed_batch"] >= 2
        assert diagnostics["species"]["balanced"]["requests"] == 4
    finally:
        service.close()
        manager.shutdown()


import pytest
