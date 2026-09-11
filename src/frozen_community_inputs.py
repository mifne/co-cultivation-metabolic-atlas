"""Frozen-model array inputs for the cooperative dFBA LP.

The ordinary simulator deliberately keeps COBRA models live: exchange
classification, bounds and objectives may be edited between calls.  That is a
useful interactive contract, but it is unnecessarily expensive for a training
rollout whose GEM structure is fixed.  This module provides an opt-in contract
for that latter case.  It evaluates the *same* scalar uptake rules into NumPy
arrays without writing through COBRA/optlang on every time step.

Only the LP inputs are frozen.  Biomass, extracellular concentrations, feed,
polymer kinetics and phase-dependent PHA objectives remain dynamic.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from cobra.util.solver import linear_reaction_coefficients

from .fba_surrogate import model_fingerprint


_POLYMER_REACTION_IDS = {
    "EX_rubber_bulk_e",
    "R_EX_rubber_bulk_e",
    "R_LCP",
    "R_ROXB",
    "R_ROXA",
    "R_ROXA_BULK",
}
_FAST_KM_METABOLITES = {"glc__D_e", "o2_e", "pi_e", "nh4_e"}
_BASAL_UPTAKE = (("h2o_e", -1000.0), ("h_e", -0.1), ("o2_e", -1000.0))
_BULK_POLYMER_METABOLITES = {"rubber_e", "rubber_bulk_e", "M_rubber_bulk_e"}


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _objective(model: Any) -> tuple[np.ndarray, str]:
    """Pack COBRA's current net-flux objective once in reaction order."""

    coefficients = np.zeros(len(model.reactions), dtype=np.float64)
    positions = {reaction: index for index, reaction in enumerate(model.reactions)}
    for reaction, coefficient in linear_reaction_coefficients(model).items():
        coefficients[positions[reaction]] = float(coefficient)
    direction = str(model.objective.direction)
    if direction not in {"min", "max"}:
        raise ValueError("A frozen community objective must be min or max")
    return coefficients, direction


def _safe_lower_bound(desired: float, upper: float) -> float:
    """Match :meth:`dFBASimulator._safe_lb`, including six-place rounding."""

    value = round(min(desired, upper), 6)
    # Reaction._check_bounds rejects the rounded-above-upper corner.  Keeping
    # this check here makes malformed fixed bounds fail before an LP is built.
    if value > upper:
        raise ValueError("The lower bound cannot exceed the upper bound")
    return value


@dataclass(frozen=True)
class FrozenCommunityContract:
    """Immutable identity and static arrays for one ordered model collection."""

    species_names: tuple[str, ...]
    reaction_ids: Mapping[str, tuple[str, ...]]
    model_fingerprints: Mapping[str, str]
    exchange_reaction_ids: Mapping[str, tuple[str, ...]]
    exchange_indices: Mapping[str, tuple[int, ...]]
    exchange_mapping: Mapping[str, Mapping[str, int]]
    exchange_mapping_ids: Mapping[str, tuple[tuple[str, str], ...]]
    original_exchange_lower: Mapping[str, Mapping[int, float]]
    original_exchange_upper: Mapping[str, Mapping[int, float]]
    original_exchange_bounds: Mapping[str, tuple[tuple[str, float, float], ...]]
    base_lower: Mapping[str, np.ndarray]
    base_upper: Mapping[str, np.ndarray]
    base_objective: Mapping[str, np.ndarray]
    base_objective_direction: Mapping[str, str]
    polymer_indices: Mapping[str, tuple[int, ...]]
    pha_index: Mapping[str, int | None]
    phv_index: Mapping[str, int | None]
    growth_index: Mapping[str, int | None]

    def __deepcopy__(self, memo):
        # Every nested array is read-only and every mapping is a proxy over
        # immutable values. Sharing the contract also lets a deep-copied
        # solver and builder retain one identity token for the cheap hot path.
        memo[id(self)] = self
        return self


@dataclass(frozen=True)
class FrozenCommunityStepInputs:
    """Read-only bounds/objectives for one cooperative dFBA time step."""

    contract: FrozenCommunityContract
    bounds_by_species: Mapping[str, np.ndarray]
    objective_by_species: Mapping[str, np.ndarray]
    objective_direction_by_species: Mapping[str, str]

    def __deepcopy__(self, memo):
        # A step packet is immutable/read-only and is never used as a mutable
        # trajectory accumulator.
        memo[id(self)] = self
        return self


class FrozenCommunityInputBuilder:
    """Evaluate dynamic dFBA constraints under a fixed-GEM contract."""

    def __init__(
        self,
        models: Mapping[str, Any],
        exchange_reactions: Mapping[str, Mapping[str, str]],
        original_bounds: Mapping[str, Mapping[str, tuple[float, float]]],
    ) -> None:
        species_names = tuple(models)
        if not species_names:
            raise ValueError("At least one species is required")

        reaction_ids: dict[str, tuple[str, ...]] = {}
        fingerprints: dict[str, str] = {}
        exchange_ids: dict[str, tuple[str, ...]] = {}
        exchange_indices: dict[str, tuple[int, ...]] = {}
        exchange_mapping: dict[str, Mapping[str, int]] = {}
        exchange_mapping_ids: dict[str, tuple[tuple[str, str], ...]] = {}
        original_lower: dict[str, Mapping[int, float]] = {}
        original_upper: dict[str, Mapping[int, float]] = {}
        original_records: dict[str, tuple[tuple[str, float, float], ...]] = {}
        base_lower: dict[str, np.ndarray] = {}
        base_upper: dict[str, np.ndarray] = {}
        base_objective: dict[str, np.ndarray] = {}
        base_direction: dict[str, str] = {}
        polymer_indices: dict[str, tuple[int, ...]] = {}
        pha_index: dict[str, int | None] = {}
        phv_index: dict[str, int | None] = {}
        growth_index: dict[str, int | None] = {}

        for species in species_names:
            model = models[species]
            reactions = tuple(model.reactions)
            ids = tuple(reaction.id for reaction in reactions)
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate reaction IDs in frozen model {species!r}")
            position = {reaction_id: index for index, reaction_id in enumerate(ids)}
            reaction_ids[species] = ids
            fingerprints[species] = model_fingerprint(model)

            # This is the one intentional full COBRA boundary-classification
            # pass.  Training steps reuse the resulting ordered indices.
            classified = tuple(reaction.id for reaction in model.exchanges)
            exchange_ids[species] = classified
            exchange_indices[species] = tuple(position[reaction_id] for reaction_id in classified)

            supplied_mapping = exchange_reactions.get(species)
            if supplied_mapping is None:
                raise ValueError(f"Missing exchange mapping for {species!r}")
            mapped: dict[str, int] = {}
            mapped_ids: list[tuple[str, str]] = []
            for metabolite_id, reaction_id in supplied_mapping.items():
                if reaction_id not in position:
                    raise ValueError(
                        f"Frozen exchange mapping references missing reaction {reaction_id!r}"
                    )
                mapped[str(metabolite_id)] = position[reaction_id]
                mapped_ids.append((str(metabolite_id), str(reaction_id)))
            exchange_mapping[species] = MappingProxyType(mapped)
            exchange_mapping_ids[species] = tuple(mapped_ids)

            supplied_originals = original_bounds.get(species)
            if supplied_originals is None:
                raise ValueError(f"Missing original exchange bounds for {species!r}")
            original_by_index: dict[int, float] = {}
            original_upper_by_index: dict[int, float] = {}
            records: list[tuple[str, float, float]] = []
            for reaction_id, bounds in supplied_originals.items():
                if reaction_id not in position:
                    raise ValueError(
                        f"Original bounds reference missing reaction {reaction_id!r}"
                    )
                if len(bounds) != 2:
                    raise ValueError("Original bounds must be lower/upper pairs")
                lower, upper = map(float, bounds)
                original_by_index[position[reaction_id]] = lower
                original_upper_by_index[position[reaction_id]] = upper
                records.append((str(reaction_id), lower, upper))
            original_lower[species] = MappingProxyType(original_by_index)
            original_upper[species] = MappingProxyType(original_upper_by_index)
            original_records[species] = tuple(records)

            packed = np.asarray([reaction.bounds for reaction in reactions], dtype=np.float64)
            if packed.shape != (len(reactions), 2) or np.isnan(packed).any():
                raise ValueError("Frozen reaction bounds must be numeric lower/upper pairs")
            if np.any(packed[:, 0] > packed[:, 1]):
                raise ValueError("Frozen reaction lower bounds cannot exceed upper bounds")
            base_lower[species] = _readonly(packed[:, 0].copy())
            base_upper[species] = _readonly(packed[:, 1].copy())
            objective, direction = _objective(model)
            base_objective[species] = _readonly(objective)
            base_direction[species] = direction

            polymer_indices[species] = tuple(
                position[reaction_id]
                for reaction_id in ids
                if reaction_id in _POLYMER_REACTION_IDS
            )
            mapping = mapped
            pha_index[species] = mapping.get("pha_c")
            phv_index[species] = mapping.get("phv_c")
            growth_id = "R_Growth" if "R_Growth" in position else (
                "Growth" if "Growth" in position else None
            )
            growth_index[species] = None if growth_id is None else position[growth_id]
            if ("NS21" in species and pha_index[species] is not None) or "OR16" in species:
                if growth_index[species] is None:
                    raise ValueError(
                        f"Frozen objective switching requires a growth reaction for {species!r}"
                    )

        self.contract = FrozenCommunityContract(
            species_names=species_names,
            reaction_ids=MappingProxyType(reaction_ids),
            model_fingerprints=MappingProxyType(fingerprints),
            exchange_reaction_ids=MappingProxyType(exchange_ids),
            exchange_indices=MappingProxyType(exchange_indices),
            exchange_mapping=MappingProxyType(exchange_mapping),
            exchange_mapping_ids=MappingProxyType(exchange_mapping_ids),
            original_exchange_lower=MappingProxyType(original_lower),
            original_exchange_upper=MappingProxyType(original_upper),
            original_exchange_bounds=MappingProxyType(original_records),
            base_lower=MappingProxyType(base_lower),
            base_upper=MappingProxyType(base_upper),
            base_objective=MappingProxyType(base_objective),
            base_objective_direction=MappingProxyType(base_direction),
            polymer_indices=MappingProxyType(polymer_indices),
            pha_index=MappingProxyType(pha_index),
            phv_index=MappingProxyType(phv_index),
            growth_index=MappingProxyType(growth_index),
        )

    def __deepcopy__(self, memo):
        duplicate = object.__new__(type(self))
        memo[id(self)] = duplicate
        duplicate.contract = self.contract
        return duplicate

    def validate_static_contract(
        self,
        models: Mapping[str, Any],
        exchange_reactions: Mapping[str, Mapping[str, str]],
        original_bounds: Mapping[str, Mapping[str, tuple[float, float]]],
    ) -> None:
        """Fully validate the opt-in frozen contract at a rollout boundary.

        This deliberately performs the expensive model/exchange fingerprint at
        initialization or reset, never on every dFBA time step.
        """

        contract = self.contract
        if tuple(models) != contract.species_names:
            raise RuntimeError("Frozen community species order changed")
        for species in contract.species_names:
            model = models[species]
            ids = tuple(reaction.id for reaction in model.reactions)
            if ids != contract.reaction_ids[species]:
                raise RuntimeError(f"Frozen reaction order changed for {species!r}")
            if model_fingerprint(model) != contract.model_fingerprints[species]:
                raise RuntimeError(f"Frozen stoichiometry changed for {species!r}")
            if tuple(reaction.id for reaction in model.exchanges) != contract.exchange_reaction_ids[species]:
                raise RuntimeError(f"Frozen exchange classification changed for {species!r}")
            mapping = tuple(
                (str(metabolite), str(reaction))
                for metabolite, reaction in exchange_reactions.get(species, {}).items()
            )
            if mapping != contract.exchange_mapping_ids[species]:
                raise RuntimeError(f"Frozen exchange mapping changed for {species!r}")
            originals = tuple(
                (str(reaction), float(bounds[0]), float(bounds[1]))
                for reaction, bounds in original_bounds.get(species, {}).items()
            )
            if originals != contract.original_exchange_bounds[species]:
                raise RuntimeError(f"Frozen original bounds changed for {species!r}")

            packed = np.asarray([reaction.bounds for reaction in model.reactions], dtype=np.float64)
            # Exchange lower bounds are scenario state and are reset on each
            # step. All uppers and non-exchange lowers are part of the frozen
            # model contract.
            if not np.array_equal(packed[:, 1], contract.base_upper[species]):
                raise RuntimeError(f"Frozen static upper bounds changed for {species!r}")
            dynamic_lower = set(contract.exchange_indices[species])
            static_lower = np.asarray(
                [index for index in range(len(ids)) if index not in dynamic_lower],
                dtype=np.intp,
            )
            if not np.array_equal(
                packed[static_lower, 0], contract.base_lower[species][static_lower]
            ):
                raise RuntimeError(f"Frozen static lower bounds changed for {species!r}")
            objective, direction = _objective(model)
            if not np.array_equal(objective, contract.base_objective[species]):
                raise RuntimeError(f"Frozen objective changed for {species!r}")
            if direction != contract.base_objective_direction[species]:
                raise RuntimeError(f"Frozen objective direction changed for {species!r}")

    def build_step(
        self,
        metabolite_concentrations: Mapping[str, float],
        species_biomass: Mapping[str, float],
        *,
        dt: float,
        max_uptake_rate: float,
        phv_requires_rubber_intermediate: bool,
        phb_repeat_g_per_mmol: float,
        phv_repeat_g_per_mmol: float,
    ) -> FrozenCommunityStepInputs:
        """Return the exact dynamic bounds/objectives without COBRA mutation."""

        contract = self.contract
        if tuple(species_biomass) != contract.species_names:
            raise ValueError("Biomass order must match the frozen species order")
        max_uptake = float(max_uptake_rate)
        dt = float(dt)
        bounds_by_species: dict[str, np.ndarray] = {}
        objectives: dict[str, np.ndarray] = {}
        directions: dict[str, str] = {}

        total_biomass: float | None = None
        for species in contract.species_names:
            lower = contract.base_lower[species].copy()
            upper = contract.base_upper[species].copy()
            ids = contract.reaction_ids[species]
            mapping = contract.exchange_mapping[species]

            # Pass 1: reset all currently classified exchange uptake bounds.
            for index in contract.exchange_indices[species]:
                reaction_id = ids[index]
                if reaction_id in {"EX_rubber_bulk_e", "R_EX_rubber_bulk_e"}:
                    lower[index] = upper[index] = 0.0
                    continue
                if "pha_c" in reaction_id or "phb_c" in reaction_id:
                    continue
                lower[index] = _safe_lower_bound(0.0, float(upper[index]))

            # Pass 2: basal water, proton and oxygen availability.
            for metabolite_id, desired in _BASAL_UPTAKE:
                index = mapping.get(metabolite_id)
                if index is not None:
                    lower[index] = _safe_lower_bound(desired, float(upper[index]))

            # Pass 3: medium entries retain insertion order, so aliases to one
            # reaction have the same last-assignment-wins behavior as COBRA.
            for metabolite_id, concentration in metabolite_concentrations.items():
                if metabolite_id in {"h2o_e", "h_e"} or metabolite_id in _BULK_POLYMER_METABOLITES:
                    continue
                index = mapping.get(metabolite_id)
                if index is None:
                    continue
                try:
                    original_lower = contract.original_exchange_lower[species].get(index, -1000.0)
                    km = 0.01 if metabolite_id in _FAST_KM_METABOLITES else 0.1
                    kinetics = max_uptake * concentration / (km + concentration)
                    if total_biomass is None:
                        # Preserve lookup and summation order of the legacy
                        # scalar implementation.
                        max(1e-6, species_biomass[species])
                        total_biomass = sum(
                            max(1e-6, biomass) for biomass in species_biomass.values()
                        )
                    physical = concentration / (total_biomass * dt)
                    effective_physical = min(physical, max_uptake * 10.0)
                    uptake_limit = min(kinetics, effective_physical)
                    combined = max(-uptake_limit, original_lower)
                    lower[index] = _safe_lower_bound(
                        min(0.0, combined), float(upper[index])
                    )
                except Exception:
                    # This matches the existing per-medium-entry fail-soft
                    # path. Basal/reset failures above still fail the step.
                    continue

            for index in contract.polymer_indices[species]:
                lower[index] = upper[index] = 0.0

            objective = contract.base_objective[species].copy()
            direction = contract.base_objective_direction[species]
            pha = contract.pha_index[species]
            phv = contract.phv_index[species]
            growth = contract.growth_index[species]
            if "NS21" in species:
                nitrogen = metabolite_concentrations.get("nh4_e", 0.0)
                intermediate = any(
                    metabolite_concentrations.get(metabolite_id, 0.0) > 1e-12
                    for metabolite_id in ("C30_oligo_e", "odtd_e")
                )
                phv_enabled = phv is not None and (
                    not phv_requires_rubber_intermediate or intermediate
                )
                if nitrogen < 0.1 and pha is not None:
                    upper[pha] = max(
                        0.0,
                        float(contract.original_exchange_upper[species].get(pha, 1000.0)),
                    )
                    objective.fill(0.0)
                    objective[pha] = float(phb_repeat_g_per_mmol)
                    if phv_enabled:
                        upper[phv] = max(
                            0.0,
                            float(contract.original_exchange_upper[species].get(phv, 1000.0)),
                        )
                        objective[phv] = float(phv_repeat_g_per_mmol)
                    elif phv is not None:
                        upper[phv] = 0.0
                    direction = "max"
                else:
                    if pha is not None:
                        upper[pha] = 0.0
                    if phv is not None:
                        upper[phv] = 0.0
                    if growth is not None:
                        objective.fill(0.0)
                        objective[growth] = 1.0
                        direction = "max"
            elif "OR16" in species and growth is not None:
                objective.fill(0.0)
                objective[growth] = 1.0
                direction = "max"

            bounds = np.column_stack((lower, upper))
            bounds_by_species[species] = _readonly(bounds)
            objectives[species] = _readonly(objective)
            directions[species] = direction

        return FrozenCommunityStepInputs(
            contract=contract,
            bounds_by_species=MappingProxyType(bounds_by_species),
            objective_by_species=MappingProxyType(objectives),
            objective_direction_by_species=MappingProxyType(directions),
        )
