"""CPU-only regression oracle for deferred COBRA uptake-bound updates."""

from copy import deepcopy
from types import SimpleNamespace

import cobra
import numpy as np
import pytest

from src.dfba_simulator import dFBASimulator


def _legacy_polymer(sim):
    for model in sim.models.values():
        if "EX_rubber_bulk_e" in model.reactions:
            model.reactions.get_by_id("EX_rubber_bulk_e").bounds = (0.0, 0.0)
        for rid in {"R_LCP", "R_ROXB", "R_ROXA", "R_ROXA_BULK"}:
            if rid in model.reactions:
                model.reactions.get_by_id(rid).bounds = (0.0, 0.0)


def _legacy_uptake(sim, species_name, concentrations):
    """Unmodified sequential mathematics/assignment order before batching."""
    model = sim.models[species_name]
    max_uptake = sim.max_uptake_rate
    for rxn in model.exchanges:
        if rxn.id in {"EX_rubber_bulk_e", "R_EX_rubber_bulk_e"}:
            rxn.bounds = (0.0, 0.0)
            continue
        if "pha_c" in rxn.id or "phb_c" in rxn.id:
            continue
        rxn.lower_bound = sim._safe_lb(0.0, rxn.upper_bound)
    for h_id in ["h2o_e", "h_e", "o2_e"]:
        if h_id in sim.exchange_reactions[species_name]:
            target = model.reactions.get_by_id(sim.exchange_reactions[species_name][h_id])
            bound = -1000.0 if h_id != "h_e" else -0.1
            target.lower_bound = sim._safe_lb(bound, target.upper_bound)
    for met_id, concentration in concentrations.items():
        if met_id in ["h2o_e", "h_e"]:
            continue
        if met_id in {"rubber_e", "rubber_bulk_e", "M_rubber_bulk_e"}:
            continue
        if met_id in sim.exchange_reactions[species_name]:
            rxn_id = sim.exchange_reactions[species_name][met_id]
            try:
                rxn = model.reactions.get_by_id(rxn_id)
                orig_lb, _ = sim.original_bounds[species_name].get(rxn_id, (-1000.0, 1000.0))
                km = 0.01 if met_id in ["glc__D_e", "o2_e", "pi_e", "nh4_e"] else 0.1
                kinetics = max_uptake * concentration / (km + concentration)
                current_biomass = max(1e-6, sim.state.species[species_name].biomass)
                total = sum(max(1e-6, state.biomass) for state in sim.state.species.values())
                physical = concentration / (total * sim.dt)
                effective = min(physical, max_uptake * 10.0)
                limit = min(kinetics, effective)
                combined = max(-limit, orig_lb)
                rxn.lower_bound = sim._safe_lb(min(0.0, combined), rxn.upper_bound)
            except:
                continue
    _legacy_polymer(sim)


def _exchange(rid, met_id=None, bounds=(-7.0, 1000.0)):
    reaction = cobra.Reaction(rid, lower_bound=bounds[0], upper_bound=bounds[1])
    reaction.add_metabolites({cobra.Metabolite(met_id or rid + "_e", compartment="e"): -1.0})
    reaction.annotation["sbo"] = "SBO:0000627"
    return reaction


def _sim():
    sim = dFBASimulator.__new__(dFBASimulator)
    sim.max_uptake_rate = 20.0
    sim.dt = 0.25
    sim.models = {}
    sim.exchange_reactions = {}
    sim.original_bounds = {}
    sim.state = SimpleNamespace(species={})
    for species, biomass in [("a", 0.23), ("b", 0.0), ("c", 1.37)]:
        model = cobra.Model(species)
        model.solver = "glpk"
        mapping = {met: "EX_" + met for met in ["h2o_e", "h_e", "o2_e", "glc__D_e", "nh4_e", "pi_e"]}
        reactions = [_exchange(rid, met) for met, rid in mapping.items()]
        reactions.extend([
            _exchange("EX_absent_e"),
            _exchange("EX_fixed_e", bounds=(0.0, 0.0)),
            _exchange("EX_negative_e", bounds=(-1.0, -0.25)),
            _exchange("EX_phb_c_special", bounds=(-2.0, 99.0)),
            _exchange("EX_pha_c_special", bounds=(-3.0, 98.0)),
            _exchange("EX_rubber_bulk_e", bounds=(-4.0, 12.0)),
            _exchange("R_EX_rubber_bulk_e", bounds=(-6.0, 14.0)),
        ])
        for rid in ["R_LCP", "R_ROXB", "R_ROXA", "R_ROXA_BULK"]:
            reaction = cobra.Reaction(rid, lower_bound=-1.0, upper_bound=5.0)
            reaction.add_metabolites({cobra.Metabolite(rid + "_c", compartment="c"): -1.0})
            reaction.annotation["sbo"] = "SBO:0000632"
            reactions.append(reaction)
        model.add_reactions(reactions)
        mapping.update({
            "glucose_alias": "EX_glc__D_e", "negative_e": "EX_negative_e",
            "fixed_e": "EX_fixed_e", "pha_c": "EX_pha_c_special",
            "rubber_e": "EX_rubber_bulk_e", "rubber_bulk_e": "R_EX_rubber_bulk_e",
            "missing_e": "not_a_reaction",
        })
        sim.models[species] = model
        sim.exchange_reactions[species] = mapping
        sim.original_bounds[species] = {r.id: r.bounds for r in reactions}
        sim.state.species[species] = SimpleNamespace(biomass=biomass)
    return sim


def _bounds_and_optlang(sim):
    return {
        (species, reaction.id): (
            *reaction.bounds, reaction.forward_variable.lb, reaction.forward_variable.ub,
            reaction.reverse_variable.lb, reaction.reverse_variable.ub,
        )
        for species, model in sim.models.items() for reaction in model.reactions
    }


def _compare(sim, medium, species="a"):
    original = deepcopy(sim)
    _legacy_uptake(original, species, medium)
    sim.set_uptake_constraints(species, medium)
    assert _bounds_and_optlang(sim) == _bounds_and_optlang(original)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("reverse", [False, True])
def test_final_bounds_and_solver_variables_match_legacy(seed, reverse):
    sim = _sim()
    rng = np.random.default_rng(seed)
    medium = {met: float(rng.uniform(0.0, 10.0)) for met in sim.exchange_reactions["a"]}
    medium.update({"rubber_bulk_e": 100.0, "M_rubber_bulk_e": 100.0, "unmapped": 10.0})
    if reverse:
        medium = dict(reversed(list(medium.items())))
    _compare(sim, medium)
    # O2 basal allowance is overwritten by the actual dissolved inventory.
    _compare(sim, {"o2_e": 0.0, "glc__D_e": 0.0, "glucose_alias": 0.37})
    assert sim.models["a"].reactions.EX_o2_e.lower_bound == 0.0
    assert sim.models["a"].reactions.EX_absent_e.lower_bound == 0.0


@pytest.mark.parametrize("concentration", [0.0, -0.01, -0.1, -2.0, float("nan"), float("inf")])
def test_unusual_concentrations_preserve_existing_scalar_and_exception_semantics(concentration):
    _compare(_sim(), {"glc__D_e": concentration, "o2_e": concentration, "glucose_alias": concentration})


@pytest.mark.parametrize("dt,biomass", [(0.0, 0.3), (0.13, -1.0), (0.7, 0.0)])
def test_biomass_floor_and_zero_time_exception_semantics(dt, biomass):
    sim = _sim()
    sim.dt = dt
    for state in sim.state.species.values():
        state.biomass = biomass
    _compare(sim, {"glc__D_e": 0.32, "o2_e": 1.24, "nh4_e": 3.02})


def test_external_bound_changes_and_aliases_to_bulk_keep_legacy_behavior():
    sim = _sim()
    _compare(sim, {"glc__D_e": 0.2})
    sim.models["a"].reactions.EX_glc__D_e.bounds = (-0.5, 8.0)
    sim.models["a"].reactions.EX_absent_e.bounds = (-0.3, 12.0)
    # Even unusual mappings retain the original last assignment behavior;
    # this optimization is not a silent change to polymer model semantics.
    sim.exchange_reactions["a"]["h2o_e"] = "R_EX_rubber_bulk_e"
    sim.exchange_reactions["a"]["bulk_alias"] = "EX_rubber_bulk_e"
    _compare(sim, {"glc__D_e": 1.0, "bulk_alias": 0.3})


def test_changed_reactions_annotations_stoichiometry_and_compartments_not_cached():
    sim = _sim()
    _compare(sim, {})
    model = sim.models["a"]
    reaction = model.reactions.EX_absent_e
    reaction.bounds = (-3.0, 4.0)
    reaction.annotation["sbo"] = "SBO:0000632"
    _compare(sim, {})
    assert reaction.lower_bound == -3.0
    reaction.annotation.clear()
    next(iter(reaction.metabolites)).compartment = "c"
    _compare(sim, {})
    assert reaction.lower_bound == -3.0
    reaction.add_metabolites({cobra.Metabolite("extra_c", compartment="c"): 1.0})
    _compare(sim, {})
    model.remove_reactions([reaction])
    model.add_reactions([_exchange("EX_absent_e", bounds=(-12.0, 16.0))])
    model.add_reactions([_exchange("EX_new_e", bounds=(-17.0, 19.0))])
    _compare(sim, {})
    assert model.reactions.EX_absent_e.lower_bound == 0.0
    assert model.reactions.EX_new_e.lower_bound == 0.0


@pytest.mark.parametrize("failure", ["missing_basal", "invalid_reset"])
def test_raised_error_preserves_preceding_assignments(failure):
    sim = _sim()
    if failure == "missing_basal":
        sim.exchange_reactions["a"]["o2_e"] = "missing_basal_reaction"
        error = KeyError
    else:
        sim.models["a"].reactions.EX_negative_e.bounds = (-1.0, -0.1234562)
        error = ValueError
    original = deepcopy(sim)
    with pytest.raises(error):
        _legacy_uptake(original, "a", {})
    with pytest.raises(error):
        sim.set_uptake_constraints("a", {})
    assert _bounds_and_optlang(sim) == _bounds_and_optlang(original)


def test_cobra_context_restores_bounds_and_solver_variables():
    sim = _sim()
    _legacy_polymer(sim)
    before = _bounds_and_optlang(sim)
    with sim.models["a"]:
        _compare(sim, {"glc__D_e": 0.22, "o2_e": 0.013})
    assert _bounds_and_optlang(sim) == before


def test_repeated_unchanged_call_performs_no_optlang_bound_writes(monkeypatch):
    sim = _sim()
    medium = {"glc__D_e": 0.33, "o2_e": 0.25, "nh4_e": 0.2, "glucose_alias": 0.28}
    sim.set_uptake_constraints("a", medium)
    writes = []
    original_update = cobra.Reaction.update_variable_bounds

    def tracked_update(reaction):
        writes.append((reaction.model.id, reaction.id))
        return original_update(reaction)

    monkeypatch.setattr(cobra.Reaction, "update_variable_bounds", tracked_update)
    sim.set_uptake_constraints("a", medium)
    assert writes == []
    # A changed medium commits each changed non-polymer reaction at most once.
    sim.set_uptake_constraints("a", {"glc__D_e": 0.07, "o2_e": 0.03})
    assert writes
    assert len(writes) == len(set(writes))

