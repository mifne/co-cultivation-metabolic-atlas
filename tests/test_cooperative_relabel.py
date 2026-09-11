import numpy as np
import pytest

from scripts import augment_cooperative_surrogate_dagger as dagger
from src.community_solver import CooperativeCommunityFbaSolver


def toy_models():
    cobra = pytest.importorskip("cobra")
    model = cobra.Model("toy")
    substrate = cobra.Metabolite("substrate_c", compartment="c")
    uptake = cobra.Reaction("uptake", lower_bound=0, upper_bound=10)
    uptake.add_metabolites({substrate: 1})
    growth = cobra.Reaction("BIOMASS", lower_bound=0, upper_bound=100)
    growth.add_metabolites({substrate: -1})
    product = cobra.Reaction("EX_phv_c", lower_bound=0, upper_bound=100)
    product.add_metabolites({substrate: -1})
    model.add_reactions([uptake, growth, product])
    model.objective = growth
    return {"toy": model}


def test_relabel_restores_live_product_objective_and_selected_profile(monkeypatch):
    models = toy_models()
    solver = CooperativeCommunityFbaSolver(
        models, maximum_coexistence_growth=0.005, optimize_live_objectives=True,
        capture_training_snapshot=True, highs_method="highs-ds",
    )
    models["toy"].objective = models["toy"].reactions.EX_phv_c
    solver.solve(models, {"toy": 0.1}, {}, dt=0.2)
    snapshot = solver.last_training_snapshot
    assert snapshot["fluxes"][2] > 9.0
    profiles = []

    def load(profile):
        profiles.append(profile)
        return toy_models()  # deliberately starts with biomass objective

    monkeypatch.setattr(dagger, "load_consortium_profile", load)
    labels = dagger.label_exact_context_chunk(snapshot["context"][None], "pf-helper3")
    assert profiles == ["pf-helper3"]
    np.testing.assert_allclose(labels[0], snapshot["fluxes"], atol=1e-5)


def test_gpu_only_without_a_gpu_solver_never_falls_back_to_cpu():
    models = toy_models()
    solver = CooperativeCommunityFbaSolver(models, gpu_qp_only=True)
    with pytest.raises(RuntimeError, match="refusing CPU LP fallback"):
        solver.solve(models, {"toy": 0.1}, {}, dt=0.2)
    assert solver.cpu_cooperative_solve_calls == 0
    assert solver.cpu_lp_stage_calls == 0


@pytest.mark.parametrize("settings", [
    {"gpu_qp_projection":False},
    {"gpu_qp_projection":True, "gpu_qp_service":object()},
])
def test_multioutput_does_not_silently_use_an_unsupported_path(settings):
    with pytest.raises(ValueError, match="local GPU QP path"):
        CooperativeCommunityFbaSolver(toy_models(), gpu_qp_multioutput_strength=100, **settings)


def test_scalar_qualification_does_not_qualify_multioutput(tmp_path):
    manifest = tmp_path / "model.validation.json"
    manifest.write_text('{"status": "qualified", "gpu_qp_multioutput_strength": 0}')
    with pytest.warns(RuntimeWarning, match="validation_multioutput_configuration_mismatch"):
        with pytest.raises(ValueError, match="requires a neural-reranked"):
            CooperativeCommunityFbaSolver(toy_models(), gpu_qp_projection=True,
                gpu_qp_only=True, gpu_qp_multioutput_strength=100,
                surrogate_artifact=str(tmp_path / "absent.pt"),
                surrogate_validation_manifest=str(manifest))
