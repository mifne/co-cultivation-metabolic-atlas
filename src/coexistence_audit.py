"""Coexistence pre-audit for the three-member dFBA consortium.

The existing joint FBA implementation is block diagonal and only reduces
solver launch overhead.  It does not prove ecological coexistence.  This
module therefore adds an independent, read-only audit with two complementary
tests:

1. a shared-medium max-min LP that maximizes the smallest growth rate across
   all members while allowing mass-balanced cross-feeding; and
2. exact-HiGHS dFBA persistence runs for singletons, pairs, and the triplet.

The max-min formulation is a feasibility screen inspired by balanced-growth
community models.  It is not a stability proof and must be interpreted with
the dynamic persistence result and experimental validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from cobra import Model
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
from scipy.optimize import linprog
from scipy.sparse import block_diag, csr_matrix, hstack, vstack

from .dfba_simulator import dFBASimulator
from .metabolite_ids import canonical_metabolite_id
from .utils import get_initial_params


EXPECTED_MEMBERS = ("OR16", "NS21", "Lactobacillus_plantarum")
DEFAULT_SURVIVAL_THRESHOLD = 0.01
DEFAULT_GROWTH_THRESHOLD = 1e-5


def _single_external_metabolite(reaction) -> Optional[tuple[str, float]]:
    """Return ``(canonical_id, coefficient)`` for a true external boundary."""

    if len(reaction.metabolites) != 1:
        return None
    metabolite, coefficient = next(iter(reaction.metabolites.items()))
    compartment = str(getattr(metabolite, "compartment", "")).lower()
    met_id = canonical_metabolite_id(metabolite.id)
    if compartment not in {"e", "extracellular", "external"} and not met_id.endswith("_e"):
        return None
    if not np.isfinite(float(coefficient)) or abs(float(coefficient)) < 1e-12:
        return None
    return met_id, float(coefficient)


def find_growth_reaction(model: Model):
    """Find the reaction representing the model's biomass objective."""

    coefficients = linear_reaction_coefficients(model)
    if coefficients:
        reaction, coefficient = max(
            coefficients.items(), key=lambda item: abs(float(item[1]))
        )
        if abs(float(coefficient)) > 1e-12:
            return reaction, float(coefficient)
    preferred = ("R_Growth", "Growth", "R_BIOMASS_LLA", "BIOMASS_LLA", "BIOMASS")
    for reaction_id in preferred:
        if reaction_id in model.reactions:
            return model.reactions.get_by_id(reaction_id), 1.0
    candidates = [
        reaction
        for reaction in model.reactions
        if "biomass" in reaction.id.lower() or "growth" in reaction.id.lower()
    ]
    if candidates:
        return candidates[0], 1.0
    raise ValueError(f"No growth/biomass reaction found for model {model.id!r}")


@dataclass
class CrossFeedingEdge:
    metabolite: str
    producer: str
    consumer: str
    producer_rate: float
    consumer_rate: float


@dataclass
class StaticCommunityResult:
    members: list[str]
    status: str
    feasible: bool
    common_growth_per_h: float
    species_growth_per_h: Dict[str, float]
    limiting_metabolites: list[dict[str, float]] = field(default_factory=list)
    exchange_balance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cross_feeding: list[CrossFeedingEdge] = field(default_factory=list)
    required_additional_supply: list[dict[str, float]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cross_feeding"] = [asdict(edge) for edge in self.cross_feeding]
        return result


@dataclass
class DynamicCommunityResult:
    members: list[str]
    scenario: str
    hours: float
    steps: int
    all_survive: bool
    stable_at_horizon: bool
    initial_biomass_g_l: Dict[str, float]
    final_biomass_g_l: Dict[str, float]
    final_fraction_of_initial: Dict[str, float]
    minimum_biomass_g_l: Dict[str, float]
    final_growth_per_h: Dict[str, float]
    extinction_time_h: Dict[str, Optional[float]]
    final_rubber_g_l: float
    final_pha_mmol: float
    final_ph: float
    base_added_mmol_l: float
    depleted_metabolites: list[str]
    solver_diagnostics: Dict[str, Any]
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_three_member_models(models: Mapping[str, Model]) -> None:
    """Fail fast unless each intended consortium member is present exactly once."""

    names = list(models)
    missing = [marker for marker in EXPECTED_MEMBERS if not any(marker in name for name in names)]
    if missing:
        raise ValueError(
            "The coexistence audit requires OR16, NS21, and L. plantarum; "
            f"missing markers: {missing}; loaded={names}"
        )
    if len(models) != 3:
        raise ValueError(f"Expected exactly 3 consortium models, received {len(models)}")


class SharedMediumCommunityLP:
    """Shared-medium max-min community LP with fixed biomass abundances."""

    def __init__(
        self,
        models: Mapping[str, Model],
        medium_concentrations: Mapping[str, float],
        biomass_g_l: Mapping[str, float],
        dt: float = 0.2,
        oxygen_transfer_mmol_l_h: float = 1.0,
        growth_threshold: float = DEFAULT_GROWTH_THRESHOLD,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required")
        # The static formulation is read-only.  Avoid deep-copying full GEMs:
        # COBRApy copies are expensive and unnecessary here, especially when
        # all seven singleton/pair/triplet combinations are audited.
        self.models = dict(models)
        self.members = list(self.models)
        self.medium = {
            canonical_metabolite_id(key): max(0.0, float(value))
            for key, value in medium_concentrations.items()
            if np.isfinite(float(value))
        }
        self.biomass = {name: max(1e-9, float(biomass_g_l[name])) for name in self.members}
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self.oxygen_transfer = max(0.0, float(oxygen_transfer_mmol_l_h))
        self.growth_threshold = max(0.0, float(growth_threshold))
        self._build()

    def _build(self) -> None:
        matrices = []
        self.offsets: Dict[str, tuple[int, int]] = {}
        self.growth_terms: Dict[str, tuple[int, float]] = {}
        self.exchange_terms: Dict[str, list[tuple[str, int, float]]] = {}
        self.bounds: list[tuple[Optional[float], Optional[float]]] = []
        cursor = 0
        for name, model in self.models.items():
            matrix = create_stoichiometric_matrix(
                model, array_type="lil", dtype=np.float64
            ).tocsr()
            matrices.append(matrix)
            start = cursor
            end = start + len(model.reactions)
            self.offsets[name] = (start, end)
            growth_reaction, coefficient = find_growth_reaction(model)
            self.growth_terms[name] = (
                start + model.reactions.index(growth_reaction),
                coefficient,
            )
            exchange_reactions = set(model.exchanges)
            for local_index, reaction in enumerate(model.reactions):
                lower, upper = map(float, reaction.bounds)
                self.bounds.append(
                    (
                        lower if np.isfinite(lower) else None,
                        upper if np.isfinite(upper) else None,
                    )
                )
                # COBRApy may classify a one-metabolite biomass drain as an
                # exchange in small or non-standard models.  It is an internal
                # objective, not a transfer to the shared environment.
                if reaction is growth_reaction or reaction not in exchange_reactions:
                    continue
                external = _single_external_metabolite(reaction)
                if external is None:
                    continue
                metabolite, stoich = external
                self.exchange_terms.setdefault(metabolite, []).append(
                    (name, start + local_index, stoich)
                )
            cursor = end

        self.n_fluxes = cursor
        mass = block_diag(matrices, format="csr")
        self.a_eq = hstack(
            [mass, csr_matrix((mass.shape[0], 1), dtype=np.float64)],
            format="csr",
        )
        self.b_eq = np.zeros(mass.shape[0], dtype=np.float64)
        self.bounds.append((0.0, 10.0))  # common growth variable

    def _supply_cap(self, metabolite: str, overrides: Mapping[str, float]) -> float:
        if metabolite in overrides:
            return max(0.0, float(overrides[metabolite]))
        low = metabolite.lower()
        if low in {"h2o_e", "h_e", "oh1_e"}:
            return 1e6
        if low in {"o2_e", "oxygen_e"}:
            return self.oxygen_transfer
        concentration = self.medium.get(metabolite, 0.0)
        return concentration / self.dt

    def _inequalities(self, overrides: Mapping[str, float]):
        rows = []
        rhs = []
        row_labels: list[tuple[str, str]] = []
        for name in self.members:
            row = np.zeros(self.n_fluxes + 1, dtype=np.float64)
            index, coefficient = self.growth_terms[name]
            row[index] = -coefficient
            row[-1] = 1.0
            rows.append(csr_matrix(row.reshape(1, -1)))
            rhs.append(0.0)
            row_labels.append(("growth", name))
        for metabolite in sorted(self.exchange_terms):
            row = np.zeros(self.n_fluxes + 1, dtype=np.float64)
            for name, index, stoich in self.exchange_terms[metabolite]:
                row[index] += self.biomass[name] * stoich
            if not np.any(np.abs(row) > 0):
                continue
            rows.append(csr_matrix(row.reshape(1, -1)))
            rhs.append(self._supply_cap(metabolite, overrides))
            row_labels.append(("medium", metabolite))
        return vstack(rows, format="csr"), np.asarray(rhs, dtype=np.float64), row_labels

    def diagnose_required_supply(
        self,
        supply_overrides: Optional[Mapping[str, float]] = None,
        minimum_growth: float = 0.0,
        allowed_metabolites: Optional[Iterable[str]] = None,
        supply_weights: Optional[Mapping[str, float]] = None,
    ) -> list[dict[str, float]]:
        """Minimize per-metabolite supply slacks for an infeasible medium.

        A positive result is the additional net environmental supply needed
        by this mathematical model at the requested common growth.  Optional
        allow-list and positive weights support cost-aware alternatives (for
        example, prohibiting expensive glutathione while permitting its amino
        acid precursors). It remains a curation hypothesis, not an
        experimentally recommended concentration.
        """

        overrides = {
            canonical_metabolite_id(key): float(value)
            for key, value in (supply_overrides or {}).items()
        }
        allowed = (
            None
            if allowed_metabolites is None
            else {canonical_metabolite_id(value) for value in allowed_metabolites}
        )
        weights = {
            canonical_metabolite_id(key): max(0.0, float(value))
            for key, value in (supply_weights or {}).items()
        }
        a_ub, b_ub, row_labels = self._inequalities(overrides)
        medium_rows = [index for index, label in enumerate(row_labels) if label[0] == "medium"]
        if not medium_rows:
            return []
        n_slacks = len(medium_rows)
        slack_matrix = np.zeros((a_ub.shape[0], n_slacks), dtype=np.float64)
        for slack_index, row_index in enumerate(medium_rows):
            slack_matrix[row_index, slack_index] = -1.0
        augmented_ub = hstack([a_ub, csr_matrix(slack_matrix)], format="csr")
        augmented_eq = hstack(
            [self.a_eq, csr_matrix((self.a_eq.shape[0], n_slacks), dtype=np.float64)],
            format="csr",
        )
        objective = np.zeros(self.n_fluxes + 1 + n_slacks, dtype=np.float64)
        for slack_index, row_index in enumerate(medium_rows):
            metabolite = row_labels[row_index][1]
            objective[self.n_fluxes + 1 + slack_index] = weights.get(metabolite, 1.0)
        bounds = list(self.bounds)
        requested = max(0.0, float(minimum_growth))
        bounds[-1] = (requested, requested)
        bounds.extend(
            [
                (0.0, None)
                if allowed is None or row_labels[row_index][1] in allowed
                else (0.0, 0.0)
                for row_index in medium_rows
            ]
        )
        result = linprog(
            objective,
            A_ub=augmented_ub,
            b_ub=b_ub,
            A_eq=augmented_eq,
            b_eq=self.b_eq,
            bounds=bounds,
            method="highs",
        )
        if not result.success or result.x is None:
            return []
        slacks = np.asarray(result.x[self.n_fluxes + 1 :], dtype=float)
        required = []
        for slack_index, row_index in enumerate(medium_rows):
            if slacks[slack_index] <= 1e-8:
                continue
            metabolite = row_labels[row_index][1]
            required.append(
                {
                    "metabolite": metabolite,
                    "additional_supply_mmol_l_h": float(slacks[slack_index]),
                    "existing_supply_mmol_l_h": float(b_ub[row_index]),
                }
            )
        required.sort(key=lambda item: item["additional_supply_mmol_l_h"], reverse=True)
        return required

    def solve_common_growth(
        self, supply_overrides: Optional[Mapping[str, float]] = None
    ) -> float:
        """Return only the max-min growth rate for large sensitivity screens.

        ``solve`` additionally constructs a much larger parsimonious exchange
        LP and extracts cross-feeding diagnostics.  Repeating that diagnostic
        formulation hundreds of times is unnecessary for dose/subset screens
        and can exhaust native HiGHS resources.  This lightweight path solves
        only the primary max-min problem and returns zero for failed or
        non-finite solves.
        """

        overrides = {
            canonical_metabolite_id(key): float(value)
            for key, value in (supply_overrides or {}).items()
        }
        a_ub, b_ub, _ = self._inequalities(overrides)
        objective = np.zeros(self.n_fluxes + 1, dtype=np.float64)
        objective[-1] = -1.0
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=self.a_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method="highs",
        )
        if result.success and result.x is not None and np.all(np.isfinite(result.x)):
            return max(0.0, float(result.x[-1]))
        return 0.0

    def solve(
        self,
        supply_overrides: Optional[Mapping[str, float]] = None,
        cooperative_fraction: float = 0.99,
    ) -> StaticCommunityResult:
        overrides = {
            canonical_metabolite_id(key): float(value)
            for key, value in (supply_overrides or {}).items()
        }
        a_ub, b_ub, row_labels = self._inequalities(overrides)
        objective = np.zeros(self.n_fluxes + 1, dtype=np.float64)
        objective[-1] = -1.0
        first = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=self.a_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method="highs",
        )
        if not first.success or first.x is None or not np.all(np.isfinite(first.x)):
            return StaticCommunityResult(
                members=list(self.members),
                status="infeasible",
                feasible=False,
                common_growth_per_h=0.0,
                species_growth_per_h={name: 0.0 for name in self.members},
                required_additional_supply=self.diagnose_required_supply(
                    overrides, minimum_growth=self.growth_threshold
                )[:30],
                message=str(first.message),
            )

        common_growth = max(0.0, float(first.x[-1]))
        fraction = float(np.clip(cooperative_fraction, 0.0, 1.0))
        floor = common_growth * fraction
        # A raw alternate optimum can contain large gratuitous exchange cycles
        # (especially siderophore/iron loops).  Choose a parsimonious exchange
        # solution at 99% of the max-min growth by minimizing the L1 norm of
        # all environmental transfer rates.
        exchange_variables: list[tuple[int, float]] = []
        for terms in self.exchange_terms.values():
            for name, index, stoich in terms:
                exchange_variables.append((index, self.biomass[name] * stoich))
        n_aux = len(exchange_variables)
        second_bounds = list(self.bounds)
        second_bounds[-1] = (floor, floor)
        second_bounds.extend([(0.0, None)] * n_aux)
        second_objective = np.zeros(self.n_fluxes + 1 + n_aux, dtype=np.float64)
        second_objective[self.n_fluxes + 1 :] = 1.0
        augmented_ub = hstack(
            [a_ub, csr_matrix((a_ub.shape[0], n_aux), dtype=np.float64)],
            format="csr",
        )
        abs_rows = []
        abs_rhs = []
        for aux_index, (flux_index, rate_coefficient) in enumerate(exchange_variables):
            positive = np.zeros(self.n_fluxes + 1 + n_aux, dtype=np.float64)
            positive[flux_index] = rate_coefficient
            positive[self.n_fluxes + 1 + aux_index] = -1.0
            negative = np.zeros(self.n_fluxes + 1 + n_aux, dtype=np.float64)
            negative[flux_index] = -rate_coefficient
            negative[self.n_fluxes + 1 + aux_index] = -1.0
            abs_rows.extend((csr_matrix(positive.reshape(1, -1)), csr_matrix(negative.reshape(1, -1))))
            abs_rhs.extend((0.0, 0.0))
        if abs_rows:
            augmented_ub = vstack([augmented_ub, *abs_rows], format="csr")
            augmented_rhs = np.concatenate((b_ub, np.asarray(abs_rhs, dtype=np.float64)))
        else:
            augmented_rhs = b_ub
        augmented_eq = hstack(
            [self.a_eq, csr_matrix((self.a_eq.shape[0], n_aux), dtype=np.float64)],
            format="csr",
        )
        second = linprog(
            second_objective,
            A_ub=augmented_ub,
            b_ub=augmented_rhs,
            A_eq=augmented_eq,
            b_eq=self.b_eq,
            bounds=second_bounds,
            method="highs",
        )
        solution = second if second.success and second.x is not None else first
        values = np.asarray(solution.x[: self.n_fluxes + 1], dtype=np.float64)
        species_growth = {
            name: float(values[index] * coefficient)
            for name, (index, coefficient) in self.growth_terms.items()
        }

        balances: Dict[str, Dict[str, Any]] = {}
        species_exchange: Dict[str, Dict[str, float]] = {name: {} for name in self.members}
        for metabolite, terms in self.exchange_terms.items():
            rates: Dict[str, float] = {}
            for name, index, stoich in terms:
                rates[name] = rates.get(name, 0.0) + float(
                    self.biomass[name] * stoich * values[index]
                )
            species_exchange.update(
                {
                    name: {**species_exchange[name], metabolite: rate}
                    for name, rate in rates.items()
                }
            )
            cap = self._supply_cap(metabolite, overrides)
            net = float(sum(rates.values()))
            balances[metabolite] = {
                "supply_cap_mmol_l_h": cap,
                "net_consumption_mmol_l_h": net,
                "slack_mmol_l_h": cap - net,
                "species_rates_mmol_l_h": rates,
            }

        edges: list[CrossFeedingEdge] = []
        tolerance = 1e-7
        # Proton and water exchange mainly reflects bookkeeping/pH balance and
        # is not sufficiently specific to call a biological cross-feeding edge.
        noninformative_transfers = {"h_e", "h2o_e"}
        for metabolite, balance in balances.items():
            if metabolite in noninformative_transfers:
                continue
            rates = balance["species_rates_mmol_l_h"]
            producers = [(name, rate) for name, rate in rates.items() if rate < -tolerance]
            consumers = [(name, rate) for name, rate in rates.items() if rate > tolerance]
            for producer, produced in producers:
                for consumer, consumed in consumers:
                    if producer == consumer:
                        continue
                    edges.append(
                        CrossFeedingEdge(
                            metabolite=metabolite,
                            producer=producer,
                            consumer=consumer,
                            producer_rate=float(-produced),
                            consumer_rate=float(consumed),
                        )
                    )

        limiting = []
        try:
            marginals = np.asarray(first.ineqlin.marginals, dtype=float)
            residuals = np.asarray(first.ineqlin.residual, dtype=float)
            for label, marginal, residual in zip(row_labels, marginals, residuals):
                if label[0] != "medium" or marginal >= -1e-9:
                    continue
                limiting.append(
                    {
                        "metabolite": label[1],
                        "shadow_benefit": float(-marginal),
                        "slack_mmol_l_h": float(residual),
                    }
                )
            limiting.sort(key=lambda item: item["shadow_benefit"], reverse=True)
        except (AttributeError, TypeError, ValueError):
            limiting = []

        return StaticCommunityResult(
            members=list(self.members),
            status="optimal",
            feasible=common_growth >= self.growth_threshold,
            common_growth_per_h=common_growth,
            species_growth_per_h=species_growth,
            limiting_metabolites=limiting[:12],
            exchange_balance=balances,
            cross_feeding=edges,
            required_additional_supply=(
                self.diagnose_required_supply(
                    overrides, minimum_growth=self.growth_threshold
                )[:30]
                if common_growth < self.growth_threshold
                else []
            ),
            message=str(solution.message),
        )


def all_member_combinations(models: Mapping[str, Model]) -> Iterable[Dict[str, Model]]:
    names = list(models)
    for size in range(1, len(names) + 1):
        for selected in combinations(names, size):
            yield {name: models[name] for name in selected}


def _ph_from_state(state) -> float:
    h_conc = max(1e-12, float(state.metabolites.get("h_e", 1e-4)))
    return float(-np.log10(h_conc / 1000.0))


def apply_ideal_ph_control(simulator, target_ph: float = 7.0) -> float:
    """Apply an ideal base-only pH-stat step and return demand in mmol/L."""

    ratio = 10.0 ** (float(target_ph) - simulator.pKa)
    target_base = simulator.buffer_total * ratio / (1.0 + ratio)
    addition = max(0.0, target_base - simulator.buffer_base)
    simulator.buffer_base += addition
    simulator.buffer_acid = simulator.buffer_total - simulator.buffer_base
    simulator.state.metabolites["h_e"] = 10.0 ** (3.0 - float(target_ph))
    return float(addition)


def _scenario_feed(
    scenario: str, time_h: float, dt: float
) -> tuple[dict[str, float], float, dict[str, float]]:
    if scenario == "no_feed":
        return {}, 50.0, {}
    if scenario == "maintenance_feed":
        # Conservative non-RL schedule used only as an existence probe.
        common = 0.02 if time_h < 48.0 else 0.005
        return {
            "sn_or16": 0.005,
            "sn_ns21": 0.005,
            "sn_lp": 0.005,
            "yeast_extract": common,
        }, 100.0, {}
    if scenario in {"candidate_rescue", "candidate_rescue_ph_control"}:
        # Static phase-I diagnosis identified g3ps_e and ile__L_e as the
        # smallest missing net supplies in the current model.  Feed their
        # diagnostic rates directly into the shared medium.  This is a model
        # rescue experiment, not a proposed biological formulation.
        common = 0.02 if time_h < 48.0 else 0.005
        return {
            "sn_or16": 0.005,
            "sn_ns21": 0.005,
            "sn_lp": 0.005,
            "yeast_extract": common,
        }, 100.0, {"g3ps_e": 0.01 * dt, "ile__L_e": 0.01 * dt}
    raise ValueError(f"Unknown dynamic scenario: {scenario}")


def simulate_dynamic_coexistence(
    models: Mapping[str, Model],
    hours: float,
    scenario: str = "no_feed",
    dt: float = 0.2,
    survival_threshold: float = DEFAULT_SURVIVAL_THRESHOLD,
    record_every_steps: int = 10,
) -> DynamicCommunityResult:
    """Run an exact-HiGHS dFBA persistence test for one community subset."""

    local_models = {name: model.copy() for name, model in models.items()}
    initial_biomass, metabolites = get_initial_params(local_models)
    simulator = dFBASimulator(
        models=local_models,
        initial_biomass=initial_biomass,
        initial_metabolites=metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=dt,
        solver_backend="highs",
        fba_mode="separate",
    )
    steps = max(1, int(np.ceil(float(hours) / dt)))
    minimum = dict(initial_biomass)
    extinction: Dict[str, Optional[float]] = {name: None for name in local_models}
    trajectory: list[dict[str, Any]] = []
    base_added_mmol_l = 0.0
    for step_index in range(steps):
        if scenario == "candidate_rescue_ph_control":
            # Ideal pH-stat existence probe. It converts the acidic buffer form
            # back to base at pH 7 and reports the implied titrant demand.
            base_added_mmol_l += apply_ideal_ph_control(simulator, target_ph=7.0)
        feed, kla, direct_medium = _scenario_feed(scenario, simulator.state.time, dt)
        for metabolite, amount in direct_medium.items():
            simulator.state.metabolites[metabolite] = (
                simulator.state.metabolites.get(metabolite, 0.0) + amount
            )
        state = simulator.step({}, feed, dynamic_kla=kla)
        for name, species in state.species.items():
            minimum[name] = min(minimum[name], float(species.biomass))
            if extinction[name] is None and species.biomass < survival_threshold:
                extinction[name] = float(state.time)
        if step_index % max(1, int(record_every_steps)) == 0 or step_index == steps - 1:
            trajectory.append(
                {
                    "time_h": float(state.time),
                    "biomass_g_l": {
                        name: float(species.biomass)
                        for name, species in state.species.items()
                    },
                    "growth_per_h": {
                        name: float(species.growth_rate)
                        for name, species in state.species.items()
                    },
                    "rubber_g_l": float(state.rubber_concentration),
                    "pha_mmol": float(sum(s.pha_accumulated for s in state.species.values())),
                    "ph": _ph_from_state(state),
                    "nh4_mmol_l": float(state.metabolites.get("nh4_e", 0.0)),
                    "o2_mmol_l": float(state.metabolites.get("o2_e", 0.0)),
                    "glucose_mmol_l": float(state.metabolites.get("glc__D_e", 0.0)),
                }
            )

    state = simulator.state
    final_biomass = {name: float(species.biomass) for name, species in state.species.items()}
    final_growth = {name: float(species.growth_rate) for name, species in state.species.items()}
    final_ph = _ph_from_state(state)
    all_survive = all(value >= survival_threshold for value in final_biomass.values())
    stable_at_horizon = bool(
        all_survive
        and 4.0 <= final_ph <= 9.5
        and all(value >= -1e-3 for value in final_growth.values())
    )
    depleted = sorted(
        key
        for key, initial in metabolites.items()
        if initial > 1e-6 and float(state.metabolites.get(key, 0.0)) <= max(1e-6, initial * 1e-3)
    )
    return DynamicCommunityResult(
        members=list(local_models),
        scenario=scenario,
        hours=float(state.time),
        steps=steps,
        all_survive=all_survive,
        stable_at_horizon=stable_at_horizon,
        initial_biomass_g_l={name: float(value) for name, value in initial_biomass.items()},
        final_biomass_g_l=final_biomass,
        final_fraction_of_initial={
            name: float(final_biomass[name] / max(1e-12, initial_biomass[name]))
            for name in final_biomass
        },
        minimum_biomass_g_l=minimum,
        final_growth_per_h=final_growth,
        extinction_time_h=extinction,
        final_rubber_g_l=float(state.rubber_concentration),
        final_pha_mmol=float(sum(s.pha_accumulated for s in state.species.values())),
        final_ph=final_ph,
        base_added_mmol_l=float(base_added_mmol_l),
        depleted_metabolites=depleted,
        solver_diagnostics=simulator.get_solver_diagnostics(),
        trajectory=trajectory,
    )


def relaxed_medium_requirements(
    model: Model,
    uptake_bound: float = 20.0,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Diagnose whether a non-growing model is medium-limited or structural.

    This deliberately opens all extracellular exchanges as a diagnostic only.
    The returned nutrients are hypotheses for model/medium curation, not a
    biologically valid recipe.
    """

    local = model.copy()
    for reaction in local.exchanges:
        external = _single_external_metabolite(reaction)
        if external is None:
            continue
        _, coefficient = external
        if coefficient < 0:
            reaction.lower_bound = min(reaction.lower_bound, -abs(float(uptake_bound)))
        else:
            reaction.upper_bound = max(reaction.upper_bound, abs(float(uptake_bound)))
    reaction, coefficient = find_growth_reaction(local)
    local.objective = reaction
    local.objective.direction = "max" if coefficient > 0 else "min"
    solution = local.optimize()
    if solution.status != "optimal" or not np.isfinite(solution.objective_value):
        return {"status": str(solution.status), "growth_per_h": 0.0, "required_uptakes": []}
    required = []
    for exchange in local.exchanges:
        external = _single_external_metabolite(exchange)
        if external is None or exchange.id not in solution.fluxes:
            continue
        metabolite, stoich = external
        consumption = float(stoich * solution.fluxes[exchange.id])
        if consumption > tolerance:
            required.append({"metabolite": metabolite, "rate": consumption})
    required.sort(key=lambda item: item["rate"], reverse=True)
    return {
        "status": "optimal",
        "growth_per_h": float(abs(solution.objective_value)),
        "required_uptakes": required[:30],
    }
