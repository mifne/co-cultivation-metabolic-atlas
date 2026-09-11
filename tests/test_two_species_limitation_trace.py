from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from src.coexistence_audit import SharedMediumCommunityLP
from src.limitation_trace import (
    candidate_group,
    classify_polymer_bottleneck,
    rank_present_nutrient_dropouts,
    rank_single_addition_rescues,
)
from src.utils import select_or16_ns21_models


def _consumer() -> Model:
    model = Model("consumer")
    ammonium = Metabolite("nh4_e", compartment="e")
    exchange = Reaction("EX_nh4_e", lower_bound=-10.0, upper_bound=1000.0)
    exchange.add_metabolites({ammonium: -1.0})
    growth = Reaction("Growth", lower_bound=0.0, upper_bound=10.0)
    growth.add_metabolites({ammonium: -1.0})
    model.add_reactions([exchange, growth])
    model.objective = growth
    return model


def test_two_species_selector_fails_closed_and_preserves_order(monkeypatch) -> None:
    monkeypatch.setattr("src.utils.validate_canonical_gene_ids", lambda *_: None)
    or16, ns21 = Model("or16"), Model("ns21")
    selected = select_or16_ns21_models(
        {"unrelated": Model("x"), "Rhizobacter_gummiphilus_NS21": ns21, "Actinoplanes_sp_OR16_lcp": or16}
    )
    assert list(selected) == ["Actinoplanes_sp_OR16_lcp", "Rhizobacter_gummiphilus_NS21"]
    with pytest.raises(ValueError):
        select_or16_ns21_models({"Actinoplanes_sp_OR16_lcp": or16})


def test_dropout_and_rescue_are_causal_perturbations() -> None:
    solver = SharedMediumCommunityLP(
        {"consumer": _consumer()},
        {"nh4_e": 1.0},
        {"consumer": 1.0},
        dt=1.0,
    )
    baseline = solver.solve_common_growth()
    dropouts = rank_present_nutrient_dropouts(
        solver, baseline, {"nh4_e": 1.0}, ["nh4_e"]
    )
    assert dropouts[0]["model_essential_at_tested_medium"]
    empty_solver = SharedMediumCommunityLP(
        {"consumer": _consumer()}, {}, {"consumer": 1.0}, dt=1.0
    )
    rescues = rank_single_addition_rescues(
        empty_solver, 0.0, ["nh4_e"], maximum_results=None
    )
    assert rescues[0]["rescued_growth_per_h"] == pytest.approx(0.1)
    assert rescues[0]["absolute_gain_per_h"] == pytest.approx(0.1)


def test_candidate_and_polymer_classifications_are_explicit() -> None:
    assert candidate_group("M_M_ribflv_e") == "vitamin"
    result = classify_polymer_bottleneck(
        {"bulk_substrate_scale": 0.5},
        rubber_g_l=10.0,
        oxygen_mmol_l=0.1,
        glucose_mmol_l=0.0,
    )
    assert result["code"] == "allocated_oxygen_limited"
