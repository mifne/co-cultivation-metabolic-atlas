from scripts.analysis.screen_nonproducer_helper import (
    build_scenarios,
    candidate_rows,
    prepare_helper_model,
)


def test_selected_helper_has_gene_supported_starch_route_and_no_named_pha_pathway():
    model = prepare_helper_model(write_artifacts=False)

    assert candidate_rows()[0]["candidate"] == "Bacillus subtilis 168"
    assert model.reactions.get_by_id("EX_starch_e").lower_bound == -10.0
    assert "BSU03040" in model.reactions.get_by_id("AAMYL_1").gene_reaction_rule
    assert not any(
        term in f"{reaction.id} {reaction.name or ''}".lower()
        for reaction in model.reactions
        for term in ("polyhydroxyalkanoate", "polyhydroxybutyrate")
    )


def test_each_helper_feed_rate_has_rate_matched_direct_controls():
    scenarios = build_scenarios(kla=50.0)
    keys = {
        (scenario.helper, scenario.feed_metabolite, scenario.feed_rate_mmol_l_h)
        for scenario in scenarios
    }

    for rate in (0.10, 0.25, 0.50):
        assert (True, "starch_e", rate) in keys
        assert (False, "glc__D_e", rate) in keys
        assert (False, "dextrin_e", rate) in keys

