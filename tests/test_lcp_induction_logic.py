import cobra
import pytest

from src.dfba_simulator import dFBASimulator


MODEL_PATH = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"


def _simulator(glucose: float, rubber: float = 10.0):
    model = cobra.io.read_sbml_model(MODEL_PATH)
    return dFBASimulator(
        models={"Actinoplanes_sp_OR16": model},
        initial_biomass={"Actinoplanes_sp_OR16": 0.1},
        initial_metabolites={"o2_e": 10.0, "glc__D_e": glucose},
        initial_rubber=rubber,
        dt=1.0,
    )


def test_lcp_is_external_and_glucose_repressed():
    no_glucose = _simulator(0.0)
    high_glucose = _simulator(2.0)

    no_glucose.degrade_rubber({})
    high_glucose.degrade_rubber({})

    assert no_glucose.models["Actinoplanes_sp_OR16"].reactions.R_LCP.bounds == (0.0, 0.0)
    assert no_glucose.last_polymer_fluxes["lcp_c5_mmol_l_step"] > 0.0
    assert (
        high_glucose.last_polymer_fluxes["lcp_c5_mmol_l_step"]
        < no_glucose.last_polymer_fluxes["lcp_c5_mmol_l_step"]
    )


def test_lcp_does_not_create_product_without_rubber():
    simulator = _simulator(0.0, rubber=0.0)
    simulator.degrade_rubber({})
    assert simulator.state.rubber_concentration == 0.0
    assert simulator.state.metabolites["C30_oligo_e"] == 0.0
    assert simulator.last_polymer_fluxes["lcp_c5_mmol_l_step"] == pytest.approx(0.0)
