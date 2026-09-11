from __future__ import annotations

import pytest

from scripts.analysis.screen_defined_feed_fine import feed_mass_g_l


def test_feed_mass_converts_mmol_rates_to_24h_grams_per_litre() -> None:
    feed = {
        "glu__L_e": 0.0005009829998016358,
        "ile__L_e": 0.00021542268991470335,
        "pydam_e": 1.0019659996032715e-05,
    }
    assert feed_mass_g_l(feed, 24.0) == pytest.approx(0.0024876463153982167)


def test_feed_mass_ignores_zero_rate_components() -> None:
    assert feed_mass_g_l({"glu__L_e": 0.0}, 24.0) == 0.0
