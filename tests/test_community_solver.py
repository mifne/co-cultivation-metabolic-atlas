"""Regression tests for the block-diagonal community FBA path."""

from __future__ import annotations

import os
import numpy as np

from cobra import Metabolite, Model, Reaction

from src.community_solver import CooperativeCommunityFbaSolver, JointCommunityFbaSolver
from src.dfba_simulator import dFBASimulator
from src.gpu_assignment import assign_gpu_for_worker
from src.utils import create_mock_models, get_initial_params


def test_joint_solver_splits_one_solution_per_species():
    models = create_mock_models()
    solver = JointCommunityFbaSolver(models, backend="scipy")
    solutions = solver.solve(models)

    assert solver.a_eq.shape[1] == sum(len(model.reactions) for model in models.values())
    assert solver.a_eq.nnz > 0
    assert set(solutions) == set(models)
    for name, solution in solutions.items():
        assert solution is not None
        assert len(solution.fluxes) == len(models[name].reactions)
        assert np.isfinite(solution.objective_value)


def test_dFBA_joint_mode_updates_all_species():
    models = create_mock_models()
    biomass, metabolites = get_initial_params(models)
    simulator = dFBASimulator(
        models=models,
        initial_biomass=biomass,
        initial_metabolites=metabolites,
        solver_backend="glpk",
        fba_mode="joint",
    )
    simulator.step({}, {})

    assert simulator._joint_solver is not None
    assert simulator._joint_solver.stats.status.startswith("Optimization terminated")
    assert all(np.isfinite(state.growth_rate) for state in simulator.state.species.values())


def test_gpu_assignment_tracks_logical_slots(monkeypatch):
    # Restore process visibility after this assignment-only test. Otherwise a
    # later lazy PyTorch initialization can incorrectly see nonexistent GPU 1.
    for key in ('CUDA_VISIBLE_DEVICES', 'DFBA_ASSIGNED_GPU',
                'DFBA_GPU_SLOT', 'DFBA_GPU_SLOTS_PER_DEVICE'):
        monkeypatch.setenv(key, os.environ.get(key, ''))
    assigned = assign_gpu_for_worker(3, [0, 1], slots_per_device=2)
    assert assigned == 1
    assert os.environ["DFBA_GPU_SLOT"] == "1"
    assert os.environ["DFBA_GPU_SLOTS_PER_DEVICE"] == "2"


def _crossfeeding_models():
    producer = Model("producer")
    carbon_e = Metabolite("carbon_e", compartment="e")
    carbon_c = Metabolite("carbon_c", compartment="c")
    factor_c_p = Metabolite("factor_c_p", compartment="c")
    factor_e_p = Metabolite("factor_e", compartment="e")
    ex_carbon = Reaction("EX_carbon_e")
    ex_carbon.add_metabolites({carbon_e: -1})
    ex_carbon.bounds = (-10, 1000)
    carbon_transport = Reaction("carbon_transport")
    carbon_transport.add_metabolites({carbon_e: -1, carbon_c: 1})
    make_factor = Reaction("make_factor")
    make_factor.add_metabolites({carbon_c: -2, factor_c_p: 1})
    factor_export = Reaction("factor_export")
    factor_export.add_metabolites({factor_c_p: -1, factor_e_p: 1})
    ex_factor_p = Reaction("EX_factor_e")
    ex_factor_p.add_metabolites({factor_e_p: -1})
    ex_factor_p.bounds = (0, 1000)
    growth_p = Reaction("Growth")
    growth_p.add_metabolites({carbon_c: -1})
    producer.add_reactions([ex_carbon, carbon_transport, make_factor, factor_export, ex_factor_p, growth_p])
    producer.objective = growth_p

    consumer = Model("consumer")
    factor_e_c = Metabolite("factor_e", compartment="e")
    factor_c_c = Metabolite("factor_c_c", compartment="c")
    ex_factor_c = Reaction("EX_factor_e")
    ex_factor_c.add_metabolites({factor_e_c: -1})
    ex_factor_c.bounds = (-10, 1000)
    factor_import = Reaction("factor_import")
    factor_import.add_metabolites({factor_e_c: -1, factor_c_c: 1})
    growth_c = Reaction("Growth")
    growth_c.add_metabolites({factor_c_c: -1})
    consumer.add_reactions([ex_factor_c, factor_import, growth_c])
    consumer.objective = growth_c
    return {"producer": producer, "consumer": consumer}


def test_cooperative_solver_enables_mass_balanced_same_step_crossfeeding():
    models = _crossfeeding_models()
    original = {
        name: {reaction.id: reaction.bounds for reaction in model.exchanges}
        for name, model in models.items()
    }
    # A separate consumer LP has no environmental factor and cannot grow.
    models["consumer"].reactions.EX_factor_e.lower_bound = 0.0
    assert models["consumer"].slim_optimize() == 0.0

    solver = CooperativeCommunityFbaSolver(models, original)
    solutions = solver.solve(
        models,
        biomass_g_l={"producer": 1.0, "consumer": 1.0},
        medium_mmol_l={"carbon_e": 10.0, "factor_e": 0.0},
        dt=1.0,
    )
    assert solutions["producer"] is not None
    assert solutions["consumer"] is not None
    produced = solutions["producer"].fluxes["EX_factor_e"]
    consumed = -solutions["consumer"].fluxes["EX_factor_e"]
    assert produced > 0.0
    assert consumed > 0.0
    assert consumed <= produced + 1e-8
    assert solutions["consumer"].fluxes["Growth"] > 0.0


def test_cooperative_training_snapshot_aligns_complete_community_flux():
    models = _crossfeeding_models()
    original = {
        name: {reaction.id: reaction.bounds for reaction in model.exchanges}
        for name, model in models.items()
    }
    solver = CooperativeCommunityFbaSolver(
        models, original, capture_training_snapshot=True
    )
    solutions = solver.solve(
        models,
        biomass_g_l={"producer": 1.0, "consumer": 1.0},
        medium_mmol_l={"carbon_e": 10.0, "factor_e": 0.0},
        dt=1.0,
    )
    assert all(solution is not None for solution in solutions.values())
    snapshot = solver.last_training_snapshot
    assert snapshot is not None
    assert snapshot["fluxes"].shape == (solver.n_fluxes,)
    expected_context = (
        len(solver.species_names)
        + len(solver._exchange_terms)
        + 3 * solver.n_fluxes
    )
    assert snapshot["context"].shape == (expected_context,)
    aligned = np.concatenate(
        [
            solutions[name].fluxes.to_numpy(dtype=np.float32)
            for name in solver.species_names
        ]
    )
    np.testing.assert_allclose(snapshot["fluxes"], aligned, atol=1e-6)
