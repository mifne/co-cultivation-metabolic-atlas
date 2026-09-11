"""CPU-only equivalence checks for the opt-in frozen community LP inputs."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import linprog as scipy_linprog

from src.community_solver import CooperativeCommunityFbaSolver
from src.dfba_simulator import dFBASimulator
from src.frozen_community_inputs import FrozenCommunityInputBuilder
from src.rl_environment import ConsortiumEnv
from tests.test_community_solver import _crossfeeding_models
from tests.test_uptake_bound_updates import _sim


def linprog(objective, **kwargs):
    # SciPy/HiGHS has a process-wide scheduler. Use the same one-thread
    # reference as the persistent-LP tests; a default-thread reference first
    # would make later explicit threads=1 references return a solver error.
    return scipy_linprog(objective, **dict(kwargs,
        options=dict(kwargs.get('options', {}), threads=1, parallel=False)))


@pytest.fixture(autouse=True)
def single_thread_reference(monkeypatch):
    monkeypatch.setattr('src.community_solver.linprog', linprog)


def _same_csr(actual, expected):
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


def _objective_coefficients(model):
    from cobra.util.solver import linear_reaction_coefficients

    result = np.zeros(len(model.reactions), dtype=np.float64)
    positions = {reaction: index for index, reaction in enumerate(model.reactions)}
    for reaction, coefficient in linear_reaction_coefficients(model).items():
        result[positions[reaction]] = float(coefficient)
    return result


@pytest.mark.parametrize("reverse", [False, True])
def test_array_bounds_match_all_three_legacy_species_and_alias_order(reverse):
    source = _sim()
    # The mutable-path oracle deliberately contains a dangling mapping. A
    # frozen rollout rejects that at initialization, so remove it for the
    # supported-contract equivalence comparison.
    for mapping in source.exchange_reactions.values():
        mapping.pop("missing_e")
    legacy = deepcopy(source)
    builder = FrozenCommunityInputBuilder(
        source.models, source.exchange_reactions, source.original_bounds
    )
    medium = {
        "glc__D_e": 0.41,
        "o2_e": 0.023,
        "nh4_e": 0.0,
        "pi_e": 3.1,
        "glucose_alias": 0.17,
        "negative_e": 2.0,
        "fixed_e": 5.0,
        "rubber_bulk_e": 100.0,
        "M_rubber_bulk_e": 100.0,
        "unmapped": 9.0,
    }
    if reverse:
        medium = dict(reversed(tuple(medium.items())))
    biomass = {
        species: float(state.biomass)
        for species, state in source.state.species.items()
    }
    step = builder.build_step(
        medium,
        biomass,
        dt=source.dt,
        max_uptake_rate=source.max_uptake_rate,
        phv_requires_rubber_intermediate=True,
        phb_repeat_g_per_mmol=source.PHB_REPEAT_G_PER_MMOL,
        phv_repeat_g_per_mmol=source.PHV_REPEAT_G_PER_MMOL,
    )
    for species in source.models:
        legacy.set_uptake_constraints(species, medium)
        expected = np.asarray(
            [reaction.bounds for reaction in legacy.models[species].reactions],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(step.bounds_by_species[species], expected)
        np.testing.assert_array_equal(
            step.objective_by_species[species],
            _objective_coefficients(legacy.models[species]),
        )
        assert not step.bounds_by_species[species].flags.writeable
        assert not step.objective_by_species[species].flags.writeable


def _phase_builder():
    import cobra

    models = {}
    mappings = {}
    originals = {}
    for species in ("Actinoplanes_sp_OR16_lcp", "Rhizobacter_gummiphilus_NS21"):
        model = cobra.Model(species)
        reactions = []
        mapping = {}
        for metabolite in ("glc__D_e", "nh4_e", "o2_e", "h2o_e", "h_e"):
            external = cobra.Metabolite(metabolite, compartment="e")
            reaction = cobra.Reaction("EX_" + metabolite)
            reaction.add_metabolites({external: -1.0})
            reaction.bounds = (-20.0, 1000.0)
            reaction.annotation["sbo"] = "SBO:0000627"
            reactions.append(reaction)
            mapping[metabolite] = reaction.id
        cell = cobra.Metabolite("cell_" + species, compartment="c")
        growth = cobra.Reaction("R_Growth")
        growth.add_metabolites({cell: -1.0})
        growth.bounds = (0.0, 1000.0)
        reactions.append(growth)
        if "NS21" in species:
            for pool in ("pha_c", "phv_c"):
                reaction = cobra.Reaction("EX_" + pool)
                reaction.add_metabolites(
                    {cobra.Metabolite(pool, compartment="c"): -1.0}
                )
                reaction.bounds = (0.0, 77.0 if pool == "pha_c" else 55.0)
                reactions.append(reaction)
                mapping[pool] = reaction.id
        model.add_reactions(reactions)
        model.objective = growth
        models[species] = model
        mappings[species] = mapping
        originals[species] = {
            reaction.id: reaction.bounds for reaction in model.exchanges
        }
        if "NS21" in species:
            originals[species]["EX_pha_c"] = (0.0, 77.0)
            originals[species]["EX_phv_c"] = (0.0, 55.0)
    return models, mappings, originals


@pytest.mark.parametrize(
    "nh4,intermediate,expect_storage,expect_phv",
    [
        (0.1, 2.0, False, False),
        (np.nextafter(0.1, 0.0), 1e-12, True, False),
        (0.05, np.nextafter(1e-12, np.inf), True, True),
    ],
)
def test_phase_bounds_and_objectives_keep_exact_thresholds(
    nh4, intermediate, expect_storage, expect_phv
):
    models, mappings, originals = _phase_builder()
    builder = FrozenCommunityInputBuilder(models, mappings, originals)
    biomass = {species: 0.3 for species in models}
    step = builder.build_step(
        {"nh4_e": nh4, "C30_oligo_e": intermediate},
        biomass,
        dt=0.2,
        max_uptake_rate=20.0,
        phv_requires_rubber_intermediate=True,
        phb_repeat_g_per_mmol=0.08609,
        phv_repeat_g_per_mmol=0.10012,
    )
    species = "Rhizobacter_gummiphilus_NS21"
    ids = builder.contract.reaction_ids[species]
    pha, phv = ids.index("EX_pha_c"), ids.index("EX_phv_c")
    growth = ids.index("R_Growth")
    objective = step.objective_by_species[species]
    bounds = step.bounds_by_species[species]
    if expect_storage:
        assert objective[pha] == 0.08609
        assert objective[phv] == (0.10012 if expect_phv else 0.0)
        assert objective[growth] == 0.0
        assert bounds[pha, 1] == 77.0
        assert bounds[phv, 1] == (55.0 if expect_phv else 0.0)
    else:
        assert objective[growth] == 1.0
        assert objective[pha] == objective[phv] == 0.0
        assert bounds[pha, 1] == bounds[phv, 1] == 0.0


class _RecordingBackend:
    name = "recording_cpu"
    method = "highs-ds"

    def __init__(self):
        self.calls = []

    def solve(self, objective, **kwargs):
        self.calls.append(
            dict(
                objective=np.asarray(objective).copy(),
                A_ub=kwargs["A_ub"].copy(),
                b_ub=np.asarray(kwargs["b_ub"]).copy(),
                A_eq=kwargs["A_eq"].copy(),
                b_eq=np.asarray(kwargs["b_eq"]).copy(),
                bounds=tuple(kwargs["bounds"]),
            )
        )
        return linprog(objective, **kwargs)


def _small_simulator(models, frozen):
    return dFBASimulator(
        models=models,
        initial_biomass={species: 1.0 for species in models},
        initial_metabolites={"carbon_e": 4.0, "factor_e": 0.0},
        dt=0.2,
        solver_backend="highs",
        fba_mode="cooperative",
        cooperative_optimize_live_objectives=True,
        cooperative_frozen_inputs=frozen,
        ph_control_target=None,
    )


def test_three_stage_original_lp_inputs_match_live_cobra_path():
    live = _small_simulator(_crossfeeding_models(), frozen=False)
    frozen = _small_simulator(_crossfeeding_models(), frozen=True)
    medium = {"carbon_e": 4.0, "factor_e": 0.0}
    biomass = {species: 1.0 for species in live.models}
    for species in live.models:
        live.set_uptake_constraints(species, medium)
    packed = frozen._frozen_community_input_builder.build_step(
        medium,
        biomass,
        dt=frozen.dt,
        max_uptake_rate=frozen.max_uptake_rate,
        phv_requires_rubber_intermediate=frozen.phv_requires_rubber_intermediate,
        phb_repeat_g_per_mmol=frozen.PHB_REPEAT_G_PER_MMOL,
        phv_repeat_g_per_mmol=frozen.PHV_REPEAT_G_PER_MMOL,
    )
    live_backend, frozen_backend = _RecordingBackend(), _RecordingBackend()
    live_solver = CooperativeCommunityFbaSolver(
        live.models,
        live.original_bounds,
        optimize_live_objectives=True,
        linear_program_backend=live_backend,
    )
    frozen_solver = CooperativeCommunityFbaSolver(
        frozen.models,
        frozen.original_bounds,
        optimize_live_objectives=True,
        linear_program_backend=frozen_backend,
    )
    live_result = live_solver.solve(live.models, biomass, medium, live.dt)
    frozen_result = frozen_solver.solve(
        frozen.models, biomass, medium, frozen.dt, frozen_inputs=packed
    )
    assert len(live_backend.calls) == len(frozen_backend.calls) == 3
    for actual, expected in zip(frozen_backend.calls, live_backend.calls):
        np.testing.assert_array_equal(actual["objective"], expected["objective"])
        _same_csr(actual["A_ub"], expected["A_ub"])
        _same_csr(actual["A_eq"], expected["A_eq"])
        np.testing.assert_array_equal(actual["b_ub"], expected["b_ub"])
        np.testing.assert_array_equal(actual["b_eq"], expected["b_eq"])
        assert actual["bounds"] == expected["bounds"]
    for species in live_result:
        np.testing.assert_array_equal(
            frozen_result[species].fluxes.to_numpy(),
            live_result[species].fluxes.to_numpy(),
        )


def test_frozen_step_does_not_mutate_cobra_bounds_or_objectives():
    simulator = _small_simulator(_crossfeeding_models(), frozen=True)
    before = {
        species: (
            tuple(reaction.bounds for reaction in model.reactions),
            _objective_coefficients(model),
            str(model.objective.direction),
        )
        for species, model in simulator.models.items()
    }
    simulator.step({}, {})
    for species, model in simulator.models.items():
        bounds, objective, direction = before[species]
        assert tuple(reaction.bounds for reaction in model.reactions) == bounds
        np.testing.assert_array_equal(_objective_coefficients(model), objective)
        assert str(model.objective.direction) == direction


def test_public_enable_after_deepcopy_reset_primes_solver_before_first_step():
    sample = ConsortiumEnv(
        _small_simulator(_crossfeeding_models(), frozen=False), max_time=4.0
    )
    environment = deepcopy(sample)
    environment.reset(seed=23)

    contract = environment.simulator.enable_frozen_community_inputs()
    builder = environment.simulator._frozen_community_input_builder
    solver = environment.simulator._cooperative_solver
    assert builder is not None
    assert builder.contract is contract
    assert solver is not None
    assert solver._accepted_frozen_contract is contract

    # Reset must keep static upper bounds/objectives equal to the accepted
    # contract while dynamic exchange lowers remain step-generated inputs.
    environment.reset(seed=23)
    for species, model in environment.simulator.models.items():
        np.testing.assert_array_equal(
            [reaction.upper_bound for reaction in model.reactions],
            contract.base_upper[species],
        )
        np.testing.assert_array_equal(
            _objective_coefficients(model), contract.base_objective[species]
        )

    legacy = deepcopy(sample)
    legacy.reset(seed=23)
    biomass = {
        species: float(state.biomass)
        for species, state in environment.simulator.state.species.items()
    }
    packed = builder.build_step(
        environment.simulator.state.metabolites,
        biomass,
        dt=environment.simulator.dt,
        max_uptake_rate=environment.simulator.max_uptake_rate,
        phv_requires_rubber_intermediate=(
            environment.simulator.phv_requires_rubber_intermediate
        ),
        phb_repeat_g_per_mmol=environment.simulator.PHB_REPEAT_G_PER_MMOL,
        phv_repeat_g_per_mmol=environment.simulator.PHV_REPEAT_G_PER_MMOL,
    )
    for species, model in legacy.simulator.models.items():
        legacy.simulator.set_uptake_constraints(
            species, legacy.simulator.state.metabolites
        )
        np.testing.assert_array_equal(
            packed.bounds_by_species[species],
            [reaction.bounds for reaction in model.reactions],
        )


def _state_snapshot(simulator):
    return dict(
        time=float(simulator.state.time),
        rubber=float(simulator.state.rubber_concentration),
        metabolites=dict(simulator.state.metabolites),
        species={
            species: (
                float(state.biomass),
                float(state.growth_rate),
                float(state.phb_accumulated),
                float(state.phv_accumulated),
            )
            for species, state in simulator.state.species.items()
        },
    )


def test_frozen_environment_deepcopy_and_reset_repeat_are_deterministic():
    simulator = _small_simulator(_crossfeeding_models(), frozen=True)
    environment = ConsortiumEnv(simulator, max_time=4.0)
    clone = deepcopy(environment)
    assert (
        clone.simulator._frozen_community_input_builder.contract
        is environment.simulator._frozen_community_input_builder.contract
    )
    action = np.asarray([0.11, 0.22, 0.33, 0.04, 0.55], dtype=np.float32)
    clone.reset(seed=17)
    for _ in range(8):
        clone.step(action)
    first = _state_snapshot(clone.simulator)
    assert clone.simulator._cooperative_solver._accepted_frozen_contract is not None

    # Deep-copy after the solver accepted the contract as well; the solver
    # and builder must retain a shared immutable identity token.
    continued = deepcopy(clone)
    assert (
        continued.simulator._cooperative_solver._accepted_frozen_contract
        is continued.simulator._frozen_community_input_builder.contract
    )
    continued.reset(seed=17)
    for _ in range(8):
        continued.step(action)
    assert _state_snapshot(continued.simulator) == first

    clone.reset(seed=17)
    for _ in range(8):
        clone.step(action)
    assert _state_snapshot(clone.simulator) == first


def test_full_contract_validation_rejects_structure_mapping_and_static_bound_edits():
    models, mappings, originals = _phase_builder()
    builder = FrozenCommunityInputBuilder(models, mappings, originals)
    builder.validate_static_contract(models, mappings, originals)

    changed_mapping = deepcopy(mappings)
    changed_mapping["Actinoplanes_sp_OR16_lcp"]["glc__D_e"] = "R_Growth"
    with pytest.raises(RuntimeError, match="exchange mapping"):
        builder.validate_static_contract(models, changed_mapping, originals)

    species = "Actinoplanes_sp_OR16_lcp"
    reaction = models[species].reactions.get_by_id("R_Growth")
    reaction.upper_bound = 999.0
    with pytest.raises(RuntimeError, match="upper bounds"):
        builder.validate_static_contract(models, mappings, originals)

    reaction.upper_bound = 1000.0
    reaction.add_metabolites(
        {next(iter(models[species].metabolites)): -0.25}
    )
    with pytest.raises(RuntimeError, match="stoichiometry"):
        builder.validate_static_contract(models, mappings, originals)


def test_opt_in_rejects_noncooperative_mode():
    with pytest.raises(ValueError, match="requires fba_mode='cooperative'"):
        dFBASimulator(
            models=_crossfeeding_models(),
            initial_biomass={"producer": 1.0, "consumer": 1.0},
            initial_metabolites={"carbon_e": 1.0},
            fba_mode="joint",
            cooperative_frozen_inputs=True,
        )
