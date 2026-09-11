from scripts.analysis.audit_three_pump_feed_design import merge_caps


def test_merge_caps_preserves_initial_medium_and_sums_feed_rates() -> None:
    caps = merge_caps(
        {"mnl_e": 0.2},
        dt=0.2,
        additions=[{"mnl_e": 0.5}, {"mnl_e": 0.25, "glu__L_e": 0.1}],
    )
    assert caps["mnl_e"] == 1.75
    assert caps["glu__L_e"] == 0.1
