from __future__ import annotations

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction

from src.coexistence_audit import (
    SharedMediumCommunityLP,
    apply_ideal_ph_control,
    canonical_metabolite_id,
)


def test_ideal_ph_control_reports_base_and_sets_target_ph() -> None:
    from types import SimpleNamespace

    simulator = SimpleNamespace(
        pKa=7.21,
        buffer_total=100.0,
        buffer_base=1.0,
        buffer_acid=99.0,
        state=SimpleNamespace(metabolites={"h_e": 1.0}),
    )
    addition = apply_ideal_ph_control(simulator, target_ph=7.0)
    assert addition > 0.0
    assert simulator.buffer_base + simulator.buffer_acid == pytest.approx(100.0)
    assert -np.log10(simulator.state.metabolites["h_e"] / 1000.0) == pytest.approx(7.0)


def _producer_model() -> Model:
    model = Model("producer")
    glucose = Metabolite("glc__D_e", compartment="e")
    acetate = Metabolite("ac_e", compartment="e")
    ex_glucose = Reaction("EX_glc__D_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_glucose.add_metabolites({glucose: -1.0})
    ex_acetate = Reaction("EX_ac_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_acetate.add_metabolites({acetate: -1.0})
    growth = Reaction("A_Growth", lower_bound=0.0, upper_bound=10.0)
    growth.add_metabolites({glucose: -1.0, acetate: 1.0})
    model.add_reactions([ex_glucose, ex_acetate, growth])
    model.objective = growth
    return model


def _consumer_model() -> Model:
    model = Model("consumer")
    acetate = Metabolite("ac_e", compartment="e")
    ex_acetate = Reaction("EX_ac_e", lower_bound=-10.0, upper_bound=1000.0)
    ex_acetate.add_metabolites({acetate: -1.0})
    growth = Reaction("B_Growth", lower_bound=0.0, upper_bound=10.0)
    growth.add_metabolites({acetate: -1.0})
    model.add_reactions([ex_acetate, growth])
    model.objective = growth
    return model


def test_canonical_metabolite_id_removes_repeated_sbml_prefixes() -> None:
    assert canonical_metabolite_id("M_M_glc__D_e") == "glc__D_e"


def test_shared_medium_lp_detects_obligate_cross_feeding() -> None:
    producer = _producer_model()
    consumer = _consumer_model()
    community = SharedMediumCommunityLP(
        {"producer": producer, "consumer": consumer},
        medium_concentrations={"glc__D_e": 10.0},
        biomass_g_l={"producer": 1.0, "consumer": 1.0},
        dt=1.0,
    ).solve()
    consumer_alone = SharedMediumCommunityLP(
        {"consumer": consumer},
        medium_concentrations={"glc__D_e": 10.0},
        biomass_g_l={"consumer": 1.0},
        dt=1.0,
    ).solve()

    assert community.feasible
    assert community.common_growth_per_h == pytest.approx(10.0, abs=1e-7)
    assert not consumer_alone.feasible
    assert any(
        edge.metabolite == "ac_e"
        and edge.producer == "producer"
        and edge.consumer == "consumer"
        for edge in community.cross_feeding
    )


def test_shared_medium_supply_is_not_duplicated_across_species() -> None:
    model_a = _consumer_model()
    model_b = _consumer_model()
    result = SharedMediumCommunityLP(
        {"a": model_a, "b": model_b},
        medium_concentrations={"ac_e": 2.0},
        biomass_g_l={"a": 1.0, "b": 1.0},
        dt=1.0,
    ).solve()
    assert result.feasible
    assert result.common_growth_per_h == pytest.approx(1.0, abs=1e-7)


def test_lightweight_common_growth_matches_diagnostic_solver() -> None:
    producer = _producer_model()
    consumer = _consumer_model()
    solver = SharedMediumCommunityLP(
        {"producer": producer, "consumer": consumer},
        medium_concentrations={"glc__D_e": 10.0},
        biomass_g_l={"producer": 1.0, "consumer": 1.0},
        dt=1.0,
    )
    assert solver.solve_common_growth() == pytest.approx(
        solver.solve().common_growth_per_h, abs=1e-7
    )


def test_zero_growth_optimum_reports_required_supply() -> None:
    consumer = _consumer_model()
    result = SharedMediumCommunityLP(
        {"consumer": consumer},
        medium_concentrations={},
        biomass_g_l={"consumer": 1.0},
        dt=1.0,
        growth_threshold=0.01,
    ).solve()

    assert not result.feasible
    assert result.common_growth_per_h == pytest.approx(0.0, abs=1e-9)
    assert len(result.required_additional_supply) == 1
    required = result.required_additional_supply[0]
    assert required["metabolite"] == "ac_e"
    assert required["additional_supply_mmol_l_h"] == pytest.approx(0.01)
    assert required["existing_supply_mmol_l_h"] == pytest.approx(0.0)
