"""Regression tests for defects found in the cultivation accounting audit."""
import copy
import csv
import json
from dataclasses import asdict

import cobra
import numpy as np
import pytest

import src.audited_dfba as audited_module
from src.audited_dfba import AuditedDFBASimulator


def toy_model(*, storage=False, growth_id="Growth", acid_per_growth=0.0):
    """One uptake route per nutrient, fixed growth ceiling, optional storage.

    Numerical units are chosen so one unit of growth consumes two carbon
    units and emits one CO2; the remaining unit is residual cell biomass.
    Storage consumes one carbon unit per repeat. This is an accounting fixture,
    not a biochemical reconstruction.
    """
    model = cobra.Model("accounting_toy")
    pools = {name: cobra.Metabolite(name, compartment="e" if name.endswith("_e") else "c")
             for name in ["glc__D_e", "nh4_e", "o2_e", "co2_e", "h_e", "c_c", "n_c", "o_c", "pha_c"]}
    for name in ["glc__D_e", "nh4_e", "o2_e", "co2_e", "h_e"]:
        exchange = cobra.Reaction("EX_" + name)
        exchange.add_metabolites({pools[name]: -1.0})
        exchange.bounds = (-20.0, 1000.0) if name in {"glc__D_e", "nh4_e", "o2_e"} else (0.0, 1000.0)
        model.add_reactions([exchange])
    for outside, inside in [("glc__D_e", "c_c"), ("nh4_e", "n_c"), ("o2_e", "o_c")]:
        transport = cobra.Reaction("T_" + outside)
        transport.add_metabolites({pools[outside]: -1.0, pools[inside]: 1.0})
        model.add_reactions([transport])
    growth = cobra.Reaction(growth_id)
    growth.add_metabolites({pools["c_c"]: -2.0, pools["n_c"]: -1.0,
                           pools["o_c"]: -1.0, pools["co2_e"]: 1.0})
    if acid_per_growth:
        growth.add_metabolites({pools["h_e"]: acid_per_growth})
    growth.bounds = (0.0, 0.5)
    model.add_reactions([growth])
    if storage:
        synthesis = cobra.Reaction("STORE")
        synthesis.add_metabolites({pools["c_c"]: -1.0, pools["o_c"]: -1.0,
                                  pools["pha_c"]: 1.0})
        sink = cobra.Reaction("EX_pha_c")
        sink.add_metabolites({pools["pha_c"]: -1.0})
        model.add_reactions([synthesis, sink])
    model.objective = growth
    return model


def make_sim(*, species=("A",), biomass=0.1, dt=0.1, internal=0.025,
             nh4=20.0, storage=False, **kwargs):
    populations = {name: biomass for name in species}
    medium = {"glc__D_e": 200.0, "nh4_e": nh4, "o2_e": 0.25, "co2_e": 0.0,
              "lac__L_e": 0.0}
    return AuditedDFBASimulator(
        models={name: toy_model(storage=storage) for name in species},
        initial_biomass=populations, initial_metabolites=medium,
        initial_rubber=0.0, dt=dt, max_internal_dt=internal,
        solver_backend="highs", ph_control_target=7.0, **kwargs)


def test_species_order_does_not_change_biomass_or_shared_medium():
    simulations = []
    for order in [("A", "B"), ("B", "A")]:
        sim = make_sim(species=order, biomass=4.0, dt=0.1, internal=0.1,
                       carrying_capacity=10.0)
        sim.step({}, {}, dynamic_kla=50.0)
        simulations.append(sim)
        assert sim.state.species["A"].biomass == pytest.approx(sim.state.species["B"].biomass, abs=1e-12)
    for name in ["A", "B"]:
        assert simulations[0].state.species[name].biomass == pytest.approx(
            simulations[1].state.species[name].biomass, abs=1e-12)
    assert simulations[0].state.metabolites == pytest.approx(simulations[1].state.metabolites, abs=1e-12)


@pytest.mark.parametrize("density_policy", ["flux_consistent", "none"])
def test_biomass_growth_and_exchange_integration_use_the_same_density_factor(density_policy):
    sim = make_sim(species=("A", "B"), biomass=4.0, dt=0.1, internal=0.1,
                   carrying_capacity=10.0, density_policy=density_policy)
    before_carbon = sim.state.metabolites["glc__D_e"]
    before_biomass = sum(state.biomass for state in sim.state.species.values())
    sim.step({}, {}, dynamic_kla=50.0)
    growth = sum(state.biomass for state in sim.state.species.values()) - before_biomass
    consumed = before_carbon - sim.state.metabolites["glc__D_e"]
    assert growth > 0.0
    assert consumed == pytest.approx(2.0 * growth, abs=1e-11)
    assert sim.state.metabolites["co2_e"] == pytest.approx(growth, abs=1e-11)


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_full_or_disabled_storage_is_constrained_before_substrate_is_consumed(fraction):
    sim = make_sim(species=("NS21",), nh4=0.0, storage=True,
                   density_policy="none", max_pha_fraction_g_gdcw=fraction)
    state = sim.state.species["NS21"]
    maximum_mass = state.biomass * fraction / (1.0 - fraction)
    state.phb_accumulated = maximum_mass / sim.PHB_REPEAT_G_PER_MMOL
    state.pha_accumulated = state.phb_accumulated
    before = (sim.state.metabolites["glc__D_e"], state.phb_accumulated, state.biomass)
    sim.step({}, {}, dynamic_kla=50.0)
    assert state.biomass == pytest.approx(before[2], abs=1e-12)
    assert state.phb_accumulated == pytest.approx(before[1], abs=1e-12)
    assert sim.state.metabolites["glc__D_e"] == pytest.approx(before[0], abs=1e-10)
    assert abs(sim.last_fba_solutions["NS21"].fluxes["EX_pha_c"]) < 1e-9


def test_tiny_inventory_bound_is_not_rounded_outward_in_actual_step():
    sim = make_sim(nh4=1.5e-9, dt=0.025, internal=0.025, density_policy="none")
    sim.step({}, {}, dynamic_kla=50.0)
    lower = sim.models["A"].reactions.get_by_id("EX_nh4_e").lower_bound
    exact_inventory_limit = 1.5e-9 / (0.1 * 0.025)
    assert lower >= -exact_inventory_limit * (1.0 + 1e-12)
    assert sim.state.metabolites["nh4_e"] >= 0.0


def test_storage_capacity_crossing_retains_every_consumed_carbon_unit():
    sim = make_sim(species=("NS21",), storage=True, nh4=0.05,
                   dt=1.0, internal=1.0, density_policy="none",
                   max_pha_fraction_g_gdcw=0.005)
    initial_carbon = sim.state.metabolites["glc__D_e"]
    initial_x = sim.state.species["NS21"].biomass
    sim.step({}, {}, dynamic_kla=50.0)
    state = sim.state.species["NS21"]
    polymer_mass = state.phb_accumulated * sim.PHB_REPEAT_G_PER_MMOL
    ceiling = state.biomass * 0.005 / (1.0 - 0.005)
    assert polymer_mass > 0.0
    assert polymer_mass == pytest.approx(ceiling, abs=1e-10)
    consumed = initial_carbon - sim.state.metabolites["glc__D_e"]
    assert consumed == pytest.approx(2.0 * (state.biomass - initial_x) + state.phb_accumulated, abs=1e-10)


def test_cumulative_co2_agrees_with_medium_and_growth_over_multiple_steps():
    sim = make_sim(dt=0.2, internal=0.025)
    initial_x = sim.state.species["A"].biomass
    for _ in range(3):
        sim.step({}, {}, dynamic_kla=50.0)
    assert sim.cumulative_co2_emission == pytest.approx(sim.state.metabolites["co2_e"], abs=1e-11)
    assert sim.cumulative_co2_emission == pytest.approx(sim.state.species["A"].biomass - initial_x, abs=1e-11)


@pytest.mark.parametrize("kla", [0.0, 50.0])
def test_initial_zero_do_uses_transfer_during_same_interval_without_inventing_supply(kla):
    sim = make_sim(dt=0.025, internal=0.025, density_policy="none")
    sim.state.metabolites["o2_e"] = 0.0
    sim.step({}, {}, dynamic_kla=kla)
    growth = sim.state.species["A"].biomass - 0.1
    if kla == 0.0:
        assert growth == pytest.approx(0.0, abs=1e-12)
        assert sim.oxygen_audit["transferred"] == 0.0
    else:
        assert growth > 1e-6
        assert sim.oxygen_audit["transferred"] > 0.0
    audit = sim.oxygen_audit
    assert audit["transferred"] - audit["cellular_consumed"] - audit["polymer_consumed"] == pytest.approx(
        sim.state.metabolites["o2_e"], abs=1e-12)
    assert sim.accounting_audit["max_oxygen_kinetic_residual"] <= 1e-6
    assert sim.accounting_audit["oxygen_trial_solves"] >= sim.current_step


def test_ph_stat_preserves_acid_equivalents_exceeding_initial_buffer_base():
    model = toy_model(acid_per_growth=1000.0)
    sim = AuditedDFBASimulator(
        models={"A": model}, initial_biomass={"A": 0.1},
        initial_metabolites={"glc__D_e": 200.0, "nh4_e": 20.0, "o2_e": 0.25},
        initial_rubber=0.0, dt=1.0, max_internal_dt=1.0,
        solver_backend="highs", ph_control_target=7.0, density_policy="none")
    initial_buffer_base = sim.buffer_base
    sim.step({}, {}, dynamic_kla=50.0)
    generated_acid = (sim.state.species["A"].biomass - 0.1) * 1000.0
    assert generated_acid > initial_buffer_base
    assert sim.cumulative_base_added_mmol_l == pytest.approx(generated_acid, abs=1e-9)
    assert sim.cumulative_acid_added_mmol_l == pytest.approx(0.0, abs=1e-9)
    assert sim.buffer_base == pytest.approx(initial_buffer_base, abs=1e-9)
    assert sim.state.metabolites["h_e"] == pytest.approx(1e-4, abs=1e-12)


def test_controller_call_reports_entire_defined_feed_not_final_substep():
    sim = make_sim(dt=0.1, internal=0.025)
    sim.step({}, {"coexistence_feed_rate": 2.0}, dynamic_kla=50.0,
             feed_rates_mmol_l_h={"lac__L_e": 0.3})
    first_total = sim.cumulative_defined_feed_g_l
    assert first_total > 0.0
    assert sim.last_defined_feed_g_l == pytest.approx(first_total, abs=1e-14)
    assert sim.cumulative_continuous_feed["lac__L_e"] == pytest.approx(0.03, abs=1e-14)
    sim.step({}, {"coexistence_feed_rate": 1.0}, dynamic_kla=50.0)
    assert sim.last_defined_feed_g_l == pytest.approx(sim.cumulative_defined_feed_g_l - first_total, abs=1e-14)
    assert sim.last_defined_feed_g_l == pytest.approx(first_total / 2.0, abs=1e-14)


def test_telemetry_records_post_integration_endpoint_time(tmp_path):
    path = tmp_path / "trajectory.csv"
    sim = make_sim(dt=0.1, internal=0.025, data_log_path=str(path))
    sim.step({}, {}, dynamic_kla=50.0)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    times = [float(row["time"]) for row in rows]
    assert times[0] > 0.0
    assert times == sorted(times)
    assert times[-1] == pytest.approx(sim.state.time, abs=1e-12)
    assert float(rows[-1]["biomass"]) == pytest.approx(sim.state.species["A"].biomass, abs=1e-12)


def test_reset_restores_initial_state_and_all_public_accounting():
    sim = make_sim(dt=0.1, internal=0.025)
    initial_state = copy.deepcopy(asdict(sim.state))
    initial_oxygen = copy.deepcopy(sim.oxygen_audit)
    initial_accounting = copy.deepcopy(sim.accounting_audit)
    sim.step({}, {"coexistence_feed_rate": 1.0}, dynamic_kla=50.0,
             feed_rates_mmol_l_h={"lac__L_e": 0.3})
    assert sim.state.time > 0.0
    sim.reset()
    assert asdict(sim.state) == initial_state
    assert sim.oxygen_audit == initial_oxygen
    assert sim.accounting_audit == initial_accounting
    assert sim.current_step == sim.controller_steps == sim.solve_attempts == sim.solve_successes == 0
    assert sim.cumulative_continuous_feed == {}
    assert sim.last_defined_feed_g_l == sim.cumulative_defined_feed_g_l == 0.0
    assert sim.cumulative_co2_emission == 0.0
    assert sim.cumulative_base_added_mmol_l == sim.cumulative_acid_added_mmol_l == 0.0


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
def test_invalid_feed_is_rejected_before_any_state_mutation(rate):
    sim = make_sim()
    initial_state = copy.deepcopy(asdict(sim.state))
    with pytest.raises(ValueError):
        sim.step({}, {}, dynamic_kla=50.0, feed_rates_mmol_l_h={"nh4_e": rate})
    assert asdict(sim.state) == initial_state
    assert sim.current_step == sim.controller_steps == 0
    # Invalid caller input does not poison a simulation which has not advanced.
    sim.step({}, {}, dynamic_kla=50.0)
    assert sim.state.time > 0.0


@pytest.mark.parametrize("pool", ["pha_c", "phb_c", "phv_c", "rubber_e",
                                  "rubber_bulk_e", "M_rubber_bulk_e"])
def test_storage_and_insoluble_bulk_cannot_use_dissolved_feed_interface(pool):
    sim = make_sim(species=("NS21",), storage=True)
    # Even a declared key must not bypass a different physical interface.
    sim.state.metabolites[pool] = 0.0
    initial_state = copy.deepcopy(asdict(sim.state))
    with pytest.raises(ValueError):
        sim.step({}, {}, dynamic_kla=50.0, feed_rates_mmol_l_h={pool: 0.1})
    assert asdict(sim.state) == initial_state


@pytest.mark.parametrize("overrides", [
    {"initial_biomass": {"A": -0.1}}, {"initial_biomass": {"A": float("nan")}},
    {"initial_metabolites": {"glc__D_e": -1.0}},
    {"initial_metabolites": {"glc__D_e": float("inf")}},
    {"carrying_capacity": 0.0}, {"carrying_capacity": -1.0},
    {"carrying_capacity": float("inf")}, {"initial_rubber": -1.0},
])
def test_initial_nonphysical_values_are_rejected(overrides):
    inputs = dict(models={"A": toy_model()}, initial_biomass={"A": 0.1},
                  initial_metabolites={"glc__D_e": 20.0, "nh4_e": 20.0, "o2_e": 0.25},
                  initial_rubber=0.0, solver_backend="highs")
    inputs.update(overrides)
    with pytest.raises(ValueError):
        AuditedDFBASimulator(**inputs)


def test_solver_failure_invalidates_until_reset_without_artificial_death(monkeypatch):
    sim = make_sim()
    initial_biomass = sim.state.species["A"].biomass
    with monkeypatch.context() as patch:
        patch.setattr(sim, "solve_fba", lambda name: None)
        with pytest.raises(RuntimeError):
            sim.step({}, {}, dynamic_kla=50.0)
    assert sim.state.species["A"].biomass == initial_biomass
    with pytest.raises(RuntimeError):
        sim.step({}, {}, dynamic_kla=50.0)
    sim.reset()
    sim.step({}, {}, dynamic_kla=50.0)
    assert sim.state.species["A"].biomass > initial_biomass


def test_generic_objective_cannot_silently_be_used_as_growth():
    model = toy_model(growth_id="arbitrary_objective")
    inputs = dict(models={"A": model}, initial_biomass={"A": 0.1},
                  initial_metabolites={"glc__D_e": 20.0, "nh4_e": 20.0, "o2_e": 0.25},
                  initial_rubber=0.0, solver_backend="highs")
    with pytest.raises(ValueError, match="[Gg]rowth"):
        AuditedDFBASimulator(**inputs)
    # Explicit identification makes the same reaction unambiguous.
    sim = AuditedDFBASimulator(**inputs, growth_reactions={"A": "arbitrary_objective"})
    sim.step({}, {}, dynamic_kla=50.0)
    assert sim.state.species["A"].biomass > 0.1


@pytest.mark.parametrize("growth_id", ["Growth", "R_Growth", "biomass_c0"])
def test_documented_growth_reaction_identifiers_are_recognized(growth_id):
    sim = AuditedDFBASimulator(
        models={"A": toy_model(growth_id=growth_id)}, initial_biomass={"A": 0.1},
        initial_metabolites={"glc__D_e": 20.0, "nh4_e": 20.0, "o2_e": 0.25},
        initial_rubber=0.0, dt=0.025, max_internal_dt=0.025, solver_backend="highs")
    sim.step({}, {}, dynamic_kla=50.0)
    assert sim.state.species["A"].biomass > 0.1


@pytest.mark.parametrize("metabolite_id", ["co2_e", "M_co2_e"])
def test_duplicate_physical_or_canonical_exchange_pool_is_rejected(metabolite_id):
    model = toy_model()
    metabolite = (model.metabolites.get_by_id(metabolite_id) if metabolite_id in model.metabolites
                  else cobra.Metabolite(metabolite_id, compartment="e"))
    duplicate = cobra.Reaction("EX_co2_duplicate")
    duplicate.add_metabolites({metabolite: -1.0})
    duplicate.bounds = (0.0, 0.0)
    model.add_reactions([duplicate])
    with pytest.raises(ValueError, match="[Cc]anonical|[Ee]xchange|[Dd]uplicate"):
        AuditedDFBASimulator(
            models={"A": model}, initial_biomass={"A": 0.1},
            initial_metabolites={"glc__D_e": 20.0, "nh4_e": 20.0, "o2_e": 0.25},
            initial_rubber=0.0, solver_backend="highs")


@pytest.mark.parametrize("change", ["stoichiometry", "objective", "gpr", "formula",
                                    "charge", "compartment", "reaction", "constraint"])
def test_model_mutation_is_rejected_before_feed_or_culture_state_changes(change):
    sim = make_sim()
    model = sim.models["A"]
    if change == "stoichiometry":
        model.reactions.get_by_id("Growth").add_metabolites({model.metabolites.get_by_id("c_c"): -1.0})
    elif change == "objective":
        model.objective = model.reactions.get_by_id("EX_co2_e")
    elif change == "gpr":
        model.reactions.get_by_id("Growth").gene_reaction_rule = "test_gene"
    elif change == "formula":
        model.metabolites.get_by_id("c_c").formula = "C"
    elif change == "charge":
        model.metabolites.get_by_id("c_c").charge = 1
    elif change == "compartment":
        model.metabolites.get_by_id("c_c").compartment = "changed"
    elif change == "reaction":
        extra = cobra.Reaction("NEW_DRAIN")
        extra.add_metabolites({model.metabolites.get_by_id("c_c"): -1.0})
        model.add_reactions([extra])
    else:
        constraint = model.problem.Constraint(model.reactions.get_by_id("Growth").flux_expression,
                                              ub=0.1, name="additional_growth_constraint")
        model.add_cons_vars([constraint])
    state = copy.deepcopy(asdict(sim.state))
    with pytest.raises(RuntimeError, match="[Ss]tructure|objective|GEM|reference"):
        sim.step({}, {}, dynamic_kla=50.0, feed_rates_mmol_l_h={"lac__L_e": 0.1})
    assert asdict(sim.state) == state
    assert sim.cumulative_continuous_feed == {}
    assert sim.solve_attempts == sim.current_step == 0


@pytest.mark.parametrize("density_policy", ["none", "flux_consistent"])
@pytest.mark.parametrize("ph", [7.0, 9.0])
@pytest.mark.parametrize("required_bounds", [(0.1, 1.0), (-1.0, -0.1)])
def test_required_internal_flux_is_rejected_for_every_density_and_ph_policy(density_policy, ph, required_bounds):
    model = toy_model()
    maintenance = cobra.Reaction("MAINTENANCE")
    maintenance.add_metabolites({model.metabolites.get_by_id("c_c"): -1.0})
    maintenance.bounds = required_bounds
    model.add_reactions([maintenance])
    with pytest.raises(ValueError, match="[Bb]ound|[Rr]equired|[Ss]caling|zero"):
        AuditedDFBASimulator(
            models={"A": model}, initial_biomass={"A": 0.1},
            initial_metabolites={"glc__D_e": 20.0, "nh4_e": 20.0, "o2_e": 0.25},
            initial_rubber=0.0, solver_backend="highs", density_policy=density_policy,
            ph_control_target=ph)


@pytest.mark.parametrize("density_policy", ["none", "flux_consistent"])
def test_required_internal_bound_added_after_initialization_is_rejected_before_feed(density_policy):
    sim = make_sim(density_policy=density_policy)
    sim.models["A"].reactions.get_by_id("Growth").lower_bound = 0.1
    state = copy.deepcopy(asdict(sim.state))
    with pytest.raises((ValueError, RuntimeError), match="[Bb]ound|[Rr]equired|[Ss]caling|zero"):
        sim.step({}, {}, dynamic_kla=50.0, feed_rates_mmol_l_h={"lac__L_e": 0.1})
    assert asdict(sim.state) == state
    assert sim.cumulative_continuous_feed == {}
    assert sim.solve_attempts == sim.current_step == 0


@pytest.mark.parametrize("storage", [False, True])
def test_exchange_parsimony_retains_primary_optimum_and_removes_unnecessary_exchange(storage):
    name = "NS21" if storage else "A"
    model = toy_model(storage=storage)
    waste = cobra.Reaction("WASTE_CARBON")
    waste.add_metabolites({model.metabolites.get_by_id("c_c"): -1.0,
                          model.metabolites.get_by_id("o_c"): -1.0,
                          model.metabolites.get_by_id("co2_e"): 1.0})
    model.add_reactions([waste])
    sim = AuditedDFBASimulator(
        models={name: model}, initial_biomass={name: 0.1},
        initial_metabolites={"glc__D_e": 200.0, "nh4_e": 0.0 if storage else 20.0,
                             "o2_e": 0.25}, initial_rubber=0.0,
        dt=0.025, max_internal_dt=0.025, solver_backend="highs",
        density_policy="none", flux_selection="primary_only")
    # Compare the two LP objectives at exactly the same bounds and state;
    # full coupled runs could legitimately change DO and hence the bounds.
    sim._integration_biomass = {name: 0.1}
    sim._cell_scale = 1.0
    for reaction in ["EX_glc__D_e", "EX_o2_e"]:
        sim.models[name].reactions.get_by_id(reaction).lower_bound = -20.0
    sim.models[name].reactions.get_by_id("EX_nh4_e").lower_bound = 0.0 if storage else -20.0
    objective = {"EX_pha_c": sim.PHB_REPEAT_G_PER_MMOL} if storage else {"Growth": 1.0}
    primary = sim._solve_lp(name, objective, storage=storage)
    sim.flux_selection = "parsimonious_exchange"
    selected = sim._solve_lp(name, objective, storage=storage)
    assert primary is not None and selected is not None
    assert selected.objective_value >= primary.objective_value - 1.1e-9
    assert selected.objective_value <= primary.objective_value + 1e-9
    assert selected.fluxes["WASTE_CARBON"] == pytest.approx(0.0, abs=1e-9)
    if storage:
        assert selected.fluxes["EX_pha_c"] == pytest.approx(20.0, abs=2e-8)
        minimum_exchange = 2.0 * selected.fluxes["EX_pha_c"]
    else:
        assert selected.fluxes["Growth"] == pytest.approx(0.5, abs=1.1e-9)
        minimum_exchange = 5.0 * selected.fluxes["Growth"]
    physical = ["EX_glc__D_e", "EX_nh4_e", "EX_o2_e", "EX_co2_e"]
    assert sum(abs(selected.fluxes[reaction]) for reaction in physical) == pytest.approx(minimum_exchange, abs=1e-8)
    assert sim.solve_attempts == sim.solve_successes == 3


def test_polymer_carbon_audit_measures_real_balance_with_existing_soluble_products():
    sim = AuditedDFBASimulator(
        models={"OR16": toy_model(), "NS21": toy_model()},
        initial_biomass={"OR16": 0.5, "NS21": 0.1},
        initial_metabolites={"glc__D_e": 2.0, "nh4_e": 2.0, "o2_e": 0.25,
                             "C30_oligo_e": 0.7, "odtd_e": 0.3},
        initial_rubber=10.0, dt=0.025, max_internal_dt=0.025,
        solver_backend="highs", ph_control_target=7.0)

    def carbon():
        return (sim.state.rubber_concentration * 1000.0 / sim.RUBBER_C5_MOLAR_MASS_G_PER_MOL
                + 6.0 * sim.state.metabolites["C30_oligo_e"]
                + 3.0 * sim.state.metabolites["odtd_e"])

    before = carbon()
    sim.step({}, {}, dynamic_kla=50.0)
    difference = carbon() - before
    assert sim.state.rubber_concentration < 10.0
    assert difference == pytest.approx(0.0, abs=1e-10)
    assert sim.last_polymer_fluxes["carbon_c5_equivalent_error_mmol_l"] == pytest.approx(difference, abs=1e-10)


def test_roundoff_growth_potential_does_not_make_storage_spuriously_infeasible(monkeypatch):
    sim = make_sim(species=("NS21",), storage=True, nh4=0.1, density_policy="none")
    sim._integration_biomass = {"NS21": 0.1}
    sim.models["NS21"].reactions.get_by_id("EX_nh4_e").bounds = (0.0, 1000.0)
    original = sim._solve_lp

    def slightly_positive_potential(name, objective, **kwargs):
        solution = original(name, objective, **kwargs)
        if not kwargs.get("storage", False):
            assert solution is not None
            assert abs(solution.fluxes["Growth"]) < 1e-10
            solution.fluxes["Growth"] = 5.05637894415588e-10
        return solution

    monkeypatch.setattr(sim, "_solve_lp", slightly_positive_potential)
    solution = sim.solve_fba("NS21")
    assert solution is not None and abs(solution.fluxes["Growth"]) < 1e-10
    allocation = sim.nitrogen_allocation["NS21"]
    assert allocation["growth_floor_requested"] == pytest.approx(2.52818947207794e-10)
    assert allocation["growth_floor_applied"] == 0.0
    assert allocation["lp_growth_shortfall_h_inv"] <= 1.1e-9
    sim.reset()
    assert sim.nitrogen_allocation == {}


def test_normal_growth_target_is_preserved_to_declared_precision():
    sim = make_sim(species=("NS21",), storage=True, density_policy="none")
    sim._integration_biomass = {"NS21": 0.1}
    result = sim._solve_lp("NS21", {"EX_pha_c": sim.PHB_REPEAT_G_PER_MMOL}, storage=True, growth_lower=0.2)
    assert result is not None
    allocation = sim.nitrogen_allocation["NS21"]
    assert allocation["growth_floor_applied"] == pytest.approx(0.2 - 1e-9, abs=1e-12)
    assert result.fluxes["Growth"] >= 0.2 - 1.1e-9
    assert allocation["lp_growth_realized_h_inv"] == result.fluxes["Growth"]


def test_growth_shortfall_has_a_stricter_gate_than_general_feasibility(monkeypatch):
    from scipy.sparse import csr_matrix
    sim = make_sim()
    original = audited_module.linprog
    calls = []

    def solver(c, **kwargs):
        calls.append(_snapshot_lp_call(c, kwargs))
        result = original(c, **kwargs)
        if len(calls) == 1:
            result = copy.deepcopy(result)
            result.x[0] -= 5e-9
        return result

    monkeypatch.setattr(audited_module, "linprog", solver)
    result = sim._certified_linprog("A", "growth_floor_test", np.array([0.0]),
        A_eq=csr_matrix((0, 1)), b_eq=np.empty(0), bounds=[(0.2 - 1e-9, 1.0)], growth_floor=(0, 0.2))
    assert result is not None and len(calls) == 2
    _assert_identical_lp(calls[0], calls[1])
    rejected = sim.accounting_audit["rejected_lp_trials"][0]
    assert rejected["lower_bound"] < 1e-7
    assert rejected["growth_floor_shortfall_h_inv"] > 1.1e-9
    assert 0.2 - result.x[0] <= 1.1e-9


@pytest.mark.parametrize("change", ["rhs", "coefficient"])
@pytest.mark.parametrize("when", ["constructor", "after_initialization"])
def test_custom_optlang_balance_equation_is_never_silently_ignored(change, when):
    model = toy_model()
    if when == "after_initialization":
        sim = AuditedDFBASimulator(
            models={"A": model}, initial_biomass={"A": 0.1},
            initial_metabolites={"glc__D_e": 20.0, "nh4_e": 20.0, "o2_e": 0.25,
                                 "lac__L_e": 0.0}, initial_rubber=0.0, solver_backend="highs")
        model = sim.models["A"]
        before = copy.deepcopy(asdict(sim.state))
    constraint = model.constraints["c_c"]
    if change == "rhs":
        constraint.ub = 1.0
        constraint.lb = 1.0
    else:
        constraint.set_linear_coefficients({model.reactions.get_by_id("Growth").forward_variable: -3.0})
    # Reaction stoichiometry is deliberately unchanged; checking reaction IDs
    # or S alone cannot detect this distinct, explicitly edited solver model.
    assert model.reactions.get_by_id("Growth").metabolites[model.metabolites.get_by_id("c_c")] == -2.0
    if when == "constructor":
        with pytest.raises(ValueError, match="[Bb]alance|constraint|adapter"):
            AuditedDFBASimulator(
                models={"A": model}, initial_biomass={"A": 0.1},
                initial_metabolites={"glc__D_e": 20.0, "nh4_e": 20.0, "o2_e": 0.25},
                initial_rubber=0.0, solver_backend="highs")
    else:
        with pytest.raises(RuntimeError, match="[Ss]tructure|objective|GEM|reference"):
            sim.step({}, {}, dynamic_kla=50.0, feed_rates_mmol_l_h={"lac__L_e": 0.1})
        assert asdict(sim.state) == before
        assert sim.cumulative_continuous_feed == {}
        assert sim.solve_attempts == sim.current_step == 0


def _snapshot_lp_call(c, kwargs):
    """Copy the submitted mathematical LP, excluding solver configuration."""
    return {
        "c": np.asarray(c).copy(),
        "A_eq": kwargs["A_eq"].toarray().copy(),
        "b_eq": np.asarray(kwargs["b_eq"]).copy(),
        "A_ub": None if kwargs.get("A_ub") is None else kwargs["A_ub"].toarray().copy(),
        "b_ub": None if kwargs.get("b_ub") is None else np.asarray(kwargs["b_ub"]).copy(),
        "bounds": tuple(tuple(pair) for pair in kwargs["bounds"]),
    }


def _assert_identical_lp(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if key == "bounds":
            assert left[key] == right[key]
        elif left[key] is None:
            assert right[key] is None
        else:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("failure", ["primal", "dual_stationarity", "duality_gap",
                                    "nonfinite_primal", "nonfinite_dual"])
def test_uncertified_solver_result_is_retried_without_changing_the_lp(monkeypatch, failure):
    sim = make_sim(flux_selection="primary_only")
    before = copy.deepcopy(asdict(sim.state))
    original_solver = audited_module.linprog
    calls = []
    growth_index = [r.id for r in sim.models["A"].reactions].index("Growth")

    def solver(c, **kwargs):
        calls.append((_snapshot_lp_call(c, kwargs), kwargs["method"], copy.deepcopy(kwargs["options"])))
        result = original_solver(c, **kwargs)
        if len(calls) == 1:
            result = copy.deepcopy(result)
            if failure == "primal":
                result.x[0] += 1e-3  # A carbon balance row is now violated.
            elif failure == "dual_stationarity":
                result.eqlin.marginals[0] += 1.0
            elif failure == "duality_gap":
                # Opposite bound multiplier changes leave stationarity intact;
                # the nonzero growth upper bound exposes the wrong dual value.
                result.lower.marginals[growth_index] += 1.0
                result.upper.marginals[growth_index] -= 1.0
            elif failure == "nonfinite_primal":
                result.x[0] = np.nan
            else:
                result.lower.marginals[growth_index] = np.nan
        return result

    monkeypatch.setattr(audited_module, "linprog", solver)
    solution = sim._solve_lp("A", {"Growth": 1.0}, select_fluxes=False)
    assert solution is not None
    assert solution.fluxes["Growth"] == pytest.approx(0.5, abs=1e-10)
    assert len(calls) == 2
    _assert_identical_lp(calls[0][0], calls[1][0])
    assert [entry[1] for entry in calls] == ["highs", "highs-ds"]
    assert calls[0][2]["presolve"] is True
    assert calls[1][2]["presolve"] is False
    assert calls[1][2]["primal_feasibility_tolerance"] < calls[0][2]["primal_feasibility_tolerance"]
    assert asdict(sim.state) == before
    audit = sim.accounting_audit
    assert sim.solve_attempts == 2 and sim.solve_successes == 1
    assert audit["lp_requests"] == audit["lp_certified_requests"] == 1
    assert audit["lp_retry_attempts"] == 1
    assert len(audit["rejected_lp_trials"]) == 1
    rejected = audit["rejected_lp_trials"][0]
    assert rejected["species"] == "A" and rejected["stage"] == "growth"
    assert len(rejected["lp_sha256"]) == 64
    if failure == "primal":
        assert rejected["equality"] > 1e-7
    elif failure == "dual_stationarity":
        assert rejected["equality"] <= 1e-7 and rejected["dual_residual"] > 1e-7
    elif failure == "duality_gap":
        assert rejected["dual_residual"] <= 1e-7 and rejected["relative_duality_gap"] > 1e-7
    else:
        assert rejected["nonfinite_primal_or_dual"] is True
    json.dumps(audit, allow_nan=False)


def test_primary_is_certified_before_its_value_is_used_in_secondary_lp(monkeypatch):
    sim = make_sim(flux_selection="parsimonious_exchange")
    original_solver = audited_module.linprog
    count = len(sim.models["A"].reactions)
    growth_index = [r.id for r in sim.models["A"].reactions].index("Growth")
    calls = []

    def solver(c, **kwargs):
        calls.append(_snapshot_lp_call(c, kwargs))
        result = original_solver(c, **kwargs)
        if len(calls) == 1:
            result = copy.deepcopy(result)
            result.x[growth_index] += 0.1  # Spurious better primary objective.
        return result

    monkeypatch.setattr(audited_module, "linprog", solver)
    solution = sim._solve_lp("A", {"Growth": 1.0})
    assert solution is not None
    assert [len(call["c"]) for call in calls[:2]] == [count, count]
    assert len(calls) == 3 and len(calls[2]["c"]) > count
    _assert_identical_lp(calls[0], calls[1])
    # The rejected apparent optimum .6 must never become the retention target.
    assert calls[2]["b_ub"][-1] == pytest.approx(-0.5 + 1e-9, abs=1e-12)
    assert solution.fluxes["Growth"] >= 0.5 - 1.1e-9
    assert sim.accounting_audit["lp_requests"] == sim.accounting_audit["lp_certified_requests"] == 2
    assert sim.solve_attempts == 3 and sim.solve_successes == 2


def test_secondary_primary_loss_has_its_own_strict_retry_gate(monkeypatch):
    sim = make_sim(flux_selection="parsimonious_exchange")
    original_solver = audited_module.linprog
    reaction_count = len(sim.models["A"].reactions)
    secondary_calls = []

    def solver(c, **kwargs):
        result = original_solver(c, **kwargs)
        if len(c) > reaction_count:
            secondary_calls.append(_snapshot_lp_call(c, kwargs))
            if len(secondary_calls) == 1:
                result = copy.deepcopy(result)
                result.x *= 1.0 - 1e-8
        return result

    monkeypatch.setattr(audited_module, "linprog", solver)
    solution = sim._solve_lp("A", {"Growth": 1.0})
    assert solution is not None and len(secondary_calls) == 2
    _assert_identical_lp(secondary_calls[0], secondary_calls[1])
    rejected = sim.accounting_audit["rejected_lp_trials"][0]
    assert rejected["stage"] == "exchange_selection"
    assert rejected["equality"] <= 1e-7 and rejected["inequality"] <= 1e-7
    assert rejected["dual_residual"] <= 1e-7 and rejected["relative_duality_gap"] <= 1e-7
    assert rejected["primary_loss"] > 1.1e-9
    assert solution.fluxes["Growth"] >= 0.5 - 1.1e-9


def test_exhausted_certificate_retries_invalidate_without_integrating_and_reset_clears_audit(monkeypatch):
    sim = make_sim(dt=0.025, internal=0.025, flux_selection="primary_only")
    initial_state = copy.deepcopy(asdict(sim.state))
    initial_audit = copy.deepcopy(sim.accounting_audit)
    original_solver = audited_module.linprog
    calls = []

    def always_uncertified(c, **kwargs):
        calls.append((_snapshot_lp_call(c, kwargs), kwargs["method"]))
        result = copy.deepcopy(original_solver(c, **kwargs))
        assert result.success
        result.x[0] += 1e-3
        return result

    with monkeypatch.context() as patch:
        patch.setattr(audited_module, "linprog", always_uncertified)
        with pytest.raises(RuntimeError, match="LP failed"):
            sim.step({}, {}, dynamic_kla=50.0, feed_rates_mmol_l_h={"lac__L_e": 0.1})
    assert [method for _, method in calls] == ["highs", "highs-ds", "highs-ipm"]
    for lp, _ in calls[1:]:
        _assert_identical_lp(calls[0][0], lp)
    assert sim.state.time == sim.current_step == 0
    assert sim.state.species["A"].biomass == initial_state["species"]["A"]["biomass"]
    assert sim.state.metabolites["lac__L_e"] == pytest.approx(0.1 * 0.025, abs=1e-14)
    assert sim.solve_attempts == 3 and sim.solve_successes == 0
    audit = sim.accounting_audit
    assert audit["lp_requests"] == 1 and audit["lp_certified_requests"] == 0
    assert audit["lp_retry_attempts"] == 2 and audit["failed_steps"] == 1
    assert len(audit["rejected_lp_trials"]) == 3
    assert len({trial["lp_sha256"] for trial in audit["rejected_lp_trials"]}) == 1
    failed_state = copy.deepcopy(asdict(sim.state))
    with pytest.raises(RuntimeError, match="reset"):
        sim.step({}, {}, dynamic_kla=50.0, feed_rates_mmol_l_h={"lac__L_e": 0.1})
    assert asdict(sim.state) == failed_state
    sim.reset()
    assert asdict(sim.state) == initial_state
    assert sim.accounting_audit == initial_audit
    assert sim.solve_attempts == sim.solve_successes == 0
    sim.step({}, {}, dynamic_kla=50.0)
    assert sim.state.species["A"].biomass > initial_state["species"]["A"]["biomass"]


@pytest.mark.parametrize("unbounded_side", ["lower", "upper"])
def test_nonzero_marginal_on_absent_bound_cannot_certify_an_unbounded_lp(monkeypatch, unbounded_side):
    from scipy.optimize import OptimizeResult
    from scipy.sparse import csr_matrix
    sim = make_sim()
    original_solver = audited_module.linprog
    calls = []
    upper = unbounded_side == "upper"
    c = np.array([-1.0 if upper else 1.0])
    bounds = [(0.0, None)] if upper else [(None, 0.0)]

    def solver(cost, **kwargs):
        calls.append(_snapshot_lp_call(cost, kwargs))
        if len(calls) == 1:
            # This has zero primal residual, apparent zero stationarity/gap,
            # and ordinary dual signs. It is nevertheless no certificate:
            # its only marginal refers to a bound which does not exist.
            return OptimizeResult(
                success=True, status=0, x=np.array([0.0]),
                eqlin=OptimizeResult(marginals=np.empty(0)),
                ineqlin=OptimizeResult(marginals=np.empty(0)),
                lower=OptimizeResult(marginals=np.array([0.0 if upper else 1.0])),
                upper=OptimizeResult(marginals=np.array([-1.0 if upper else 0.0])))
        return original_solver(cost, **kwargs)

    monkeypatch.setattr(audited_module, "linprog", solver)
    result = sim._certified_linprog("A", "unbounded_certificate_test", c,
        A_eq=csr_matrix((0, 1)), b_eq=np.empty(0), bounds=bounds)
    assert result is None and len(calls) == 3
    for call in calls[1:]:
        _assert_identical_lp(calls[0], call)
    rejected = sim.accounting_audit["rejected_lp_trials"][0]
    assert rejected["status"] == 0
    assert rejected["equality"] == rejected["relative_duality_gap"] == 0.0
    assert rejected["dual_residual"] == 1.0
    assert sim.solve_successes == sim.accounting_audit["lp_certified_requests"] == 0
    json.dumps(sim.accounting_audit, allow_nan=False)
