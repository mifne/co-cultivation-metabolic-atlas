from pathlib import Path

import cobra
import pytest

from src.dfba_simulator import dFBASimulator
from src.utils import load_sbml_models, select_consortium_models


MODEL_DIR = Path("models/sbml/final_consortium")


def _models():
    return select_consortium_models(load_sbml_models(MODEL_DIR))


def test_ns21_rox_gene_mapping_and_direct_roxa_route():
    model = cobra.io.read_sbml_model(MODEL_DIR / "Rhizobacter_gummiphilus_NS21.xml")
    assert model.reactions.R_ROXB.gene_reaction_rule == "A4W93_01825"
    assert model.reactions.R_ROXA.gene_reaction_rule == "A4W93_07150"
    assert model.reactions.R_ROXA_BULK.gene_reaction_rule == "A4W93_07150"
    for reaction_id in ("R_ROXB", "R_ROXA", "R_ROXA_BULK", "R_ODTD_cat"):
        reaction = model.reactions.get_by_id(reaction_id)
        assert reaction.check_mass_balance() == {}
    assert model.reactions.EX_rubber_bulk_e.bounds == (0.0, 0.0)
    for obsolete in ("LATA1", "LATA2", "EX_oligomer_e", "EX_rubber_fragment_e"):
        assert obsolete not in model.reactions


def test_extracellular_polymer_step_conserves_carbon_and_uses_oxygen():
    models = _models()
    simulator = dFBASimulator(
        models=models,
        initial_biomass={name: 0.2 for name in models},
        initial_metabolites={
            "o2_e": 10.0,
            "glc__D_e": 0.0,
            "C30_oligo_e": 0.0,
            "odtd_e": 0.0,
        },
        initial_rubber=10.0,
        dt=0.5,
    )
    oxygen_before = simulator.state.metabolites["o2_e"]
    simulator.degrade_rubber({})

    assert simulator.state.rubber_concentration < 10.0
    assert simulator.state.metabolites["o2_e"] < oxygen_before
    assert simulator.state.metabolites["C30_oligo_e"] > 0.0
    assert simulator.state.metabolites["odtd_e"] > 0.0
    assert simulator.last_polymer_fluxes[
        "carbon_c5_equivalent_error_mmol_l"
    ] == pytest.approx(0.0, abs=1e-10)


def test_bulk_rubber_never_enters_fba_and_rate_override_is_used():
    models = _models()
    simulator = dFBASimulator(
        models=models,
        initial_biomass={name: 0.1 for name in models},
        initial_metabolites={"o2_e": 10.0},
        initial_rubber=10.0,
        dt=1.0,
    )
    for model in simulator.models.values():
        if "EX_rubber_bulk_e" in model.reactions:
            assert model.reactions.EX_rubber_bulk_e.bounds == (0.0, 0.0)

    simulator.degrade_rubber(
        {
            "lcp_c5": 0.0,
            "roxb_c5": 0.0,
            "roxa_direct_c5": 0.0,
            "roxa_oligo_c30": 0.0,
        }
    )
    assert simulator.state.rubber_concentration == pytest.approx(10.0)
    assert simulator.last_polymer_fluxes["rubber_degraded_g_l_step"] == 0.0
