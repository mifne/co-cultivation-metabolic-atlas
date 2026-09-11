from pathlib import Path

import pytest
from cobra.io import read_sbml_model


SOURCE = Path(
    "models/sbml/reference/Lactiplantibacillus_plantarum_WCFS1_Koduru2022.xml"
)
PRODUCTION = Path("models/sbml/final_consortium/Lactobacillus_plantarum.xml")


def _fully_open_growth(model) -> float:
    local = model.copy()
    local.objective = local.reactions.get_by_id("biomass")
    for exchange in local.exchanges:
        exchange.lower_bound = min(float(exchange.lower_bound), -20.0)
        exchange.upper_bound = max(float(exchange.upper_bound), 1000.0)
    return float(local.slim_optimize(error_value=0.0) or 0.0)


def test_deposited_2022_model_has_reproduced_biomass_block():
    model = read_sbml_model(str(SOURCE))
    biomass = model.reactions.get_by_id("biomass")
    assert "peptido_LPL_c" in {metabolite.id for metabolite in biomass.metabolites}
    assert "UGMDDS2" not in model.reactions
    assert _fully_open_growth(model) == pytest.approx(0.0, abs=1e-9)


def test_repaired_2022_model_uses_wcfs1_d_lactate_peptidoglycan():
    model = read_sbml_model(str(PRODUCTION))
    biomass = model.reactions.get_by_id("biomass")
    coefficients = {
        metabolite.id: float(coefficient)
        for metabolite, coefficient in biomass.metabolites.items()
    }

    assert model.id == "WCFS1_Koduru2022_repaired"
    assert coefficients["PGlac2_c"] == pytest.approx(-0.1462)
    assert coefficients["CPS_LPL2_c"] == pytest.approx(-0.1294)
    assert coefficients["nad_c"] == pytest.approx(-0.002)
    assert "peptido_LPL_c" not in coefficients


def test_repaired_2022_model_restores_gene_supported_ugmdds2_without_bypass():
    source = read_sbml_model(str(SOURCE))
    repaired = read_sbml_model(str(PRODUCTION))
    reaction = repaired.reactions.get_by_id("UGMDDS2")

    assert reaction.gene_reaction_rule == "lp_0518"
    assert reaction.lower_bound == pytest.approx(0.0)
    assert {metabolite.id: coefficient for metabolite, coefficient in reaction.metabolites.items()} == {
        "adp_c": 1.0,
        "alalac_c": -1.0,
        "atp_c": -1.0,
        "h_c": 1.0,
        "pi_c": 1.0,
        "ugmd_c": -1.0,
        "ugmdalac_c": 1.0,
    }
    assert len(repaired.exchanges) == len(source.exchanges)
    assert len(repaired.sinks) == len(source.sinks)
    assert len(repaired.demands) == len(source.demands)
    assert len(repaired.reactions) == len(source.reactions) + 1
    assert _fully_open_growth(repaired) > 1e-6
