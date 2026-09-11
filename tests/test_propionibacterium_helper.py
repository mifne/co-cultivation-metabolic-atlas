from scripts.analysis.screen_propionibacterium_helper import (
    build_scenarios,
    prepare_helper_model,
    static_helper_validation,
)


def test_pfreudenreichii_adapter_preserves_network_and_constrains_phenotype():
    model = prepare_helper_model(write_artifacts=False)
    assert len(model.reactions) == 1351
    assert len(model.genes) == 719
    assert model.reactions.get_by_id("EX_lac__L_e").lower_bound == -10.0
    assert model.reactions.get_by_id("EX_glc__D_e").bounds == (-10.0, 0.0)
    assert model.reactions.get_by_id("EX_o2_pfreud_e").lower_bound == 0.0
    assert model.metabolites.get_by_id("lac__L_e").compartment == "e"


def test_lactate_phenotype_and_required_controls_are_present():
    validation = static_helper_validation()
    assert validation["growth_h-1"] > 0.0
    assert validation["propionate_at_growth_optimum"] > 0.0
    assert validation["acetate_at_growth_optimum"] > 0.0
    names = {scenario.name for scenario in build_scenarios(50.0)}
    assert "two_lactate_0.50" in names
    assert "two_ppa_ac_mix_0.50" in names
    assert "two_helper_effluent_replay_0.50" in names
    assert "three_lactate_b0.03_r0.50" in names
