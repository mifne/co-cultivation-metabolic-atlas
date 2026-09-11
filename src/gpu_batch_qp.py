"""Device-resident batched QP projection for cooperative FBA candidates.

The QP is solved in the convex hull of exact cooperative-FBA flux vectors.
For one environment with candidate matrix ``V`` (K candidates x N reactions),
the layer solves

    minimise_w  0.5 * ||w - e_0||²
    subject to  w >= 0, sum(w) = 1,
                lower <= V.T w <= upper,
                A_shared V.T w <= supply.

Every candidate obeys the block stoichiometric equality ``S v = 0``.  The
convex combination therefore retains mass balance exactly without putting the
large metabolite equality system into the numerical QP.  A first-order
primal-dual method performs all iterations and residual checks on one GPU and
supports a batch of environments with different bounds, biomass and supply.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from itertools import product
import time
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BatchedQpProjectionResult:
    fluxes: np.ndarray
    weights: np.ndarray
    feasible: np.ndarray
    iterations: int
    inference_seconds: float
    max_bound_violation: np.ndarray
    max_shared_violation: np.ndarray
    max_shared_violation_index: np.ndarray
    max_normalized_violation: np.ndarray
    target_projection_attempted: bool = False
    target_projection_success: np.ndarray | None = None
    multioutput_projection_attempted: bool = False
    multioutput_improved: np.ndarray | None = None
    multioutput_loss_before: np.ndarray | None = None
    multioutput_loss_after: np.ndarray | None = None
    multioutput_step_fraction: np.ndarray | None = None


def _simplex_projection(values):
    """Euclidean projection of each row onto the probability simplex."""

    import torch

    sorted_values, _ = torch.sort(values, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=1) - 1.0
    positions = torch.arange(
        1, values.shape[1] + 1, device=values.device, dtype=values.dtype
    )[None, :]
    support = sorted_values - cumulative / positions > 0.0
    rho = torch.sum(support, dim=1).clamp_min(1) - 1
    theta = cumulative.gather(1, rho[:, None]) / (rho.to(values.dtype) + 1.0)[:, None]
    return torch.clamp(values - theta, min=0.0)


class BatchedCooperativeQpProjector:
    """Projected primal-dual QP solver over exact feasible flux anchors."""

    def __init__(
        self,
        species_names: Sequence[str],
        shared_metabolite_ids: Sequence[str],
        shared_terms: Sequence[tuple[int, int, int, float]],
        species_offsets: Sequence[tuple[int, int]] | None = None,
        growth_reaction_indices: Sequence[int] | None = None,
        growth_coefficients: Sequence[float] | None = None,
        coexistence_fraction: float = 0.99,
        parsimonious_reference: bool = True,
        device: str = "cuda",
        maximum_iterations: int = 400,
        check_interval: int = 20,
        normalized_tolerance: float = 2e-5,
        bound_tolerance: float = 2e-4,
        shared_tolerance: float = 3e-4,
        step_safety: float = 0.90,
        quadratic_strength: float = 1.0,
        block_composition_candidates: int = 4,
    ) -> None:
        import torch

        requested = device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda is unavailable")
        self.species_names = tuple(species_names)
        self.shared_metabolite_ids = tuple(shared_metabolite_ids)
        self.species_offsets = tuple(species_offsets or ())
        self.growth_reaction_indices = tuple(
            int(value) for value in (growth_reaction_indices or ())
        )
        self.growth_coefficients = tuple(
            float(value) for value in (growth_coefficients or ())
        )
        if len(self.growth_reaction_indices) != len(self.growth_coefficients):
            raise ValueError("growth reaction indices and coefficients must align")
        self.coexistence_fraction = float(
            np.clip(coexistence_fraction, 0.0, 1.0)
        )
        self.parsimonious_reference = bool(parsimonious_reference)
        self.maximum_iterations = max(1, int(maximum_iterations))
        self.check_interval = max(1, int(check_interval))
        self.normalized_tolerance = float(normalized_tolerance)
        self.bound_tolerance = float(bound_tolerance)
        self.shared_tolerance = float(shared_tolerance)
        self.step_safety = float(step_safety)
        self.quadratic_strength = float(quadratic_strength)
        self.block_composition_candidates = max(
            1, int(block_composition_candidates)
        )

        if shared_terms:
            metabolite, species, reaction, coefficient = zip(*shared_terms)
        else:
            metabolite, species, reaction, coefficient = (), (), (), ()
        self.term_metabolite = torch.as_tensor(
            metabolite, dtype=torch.int64, device=self.device
        )
        self.term_species = torch.as_tensor(
            species, dtype=torch.int64, device=self.device
        )
        self.term_reaction = torch.as_tensor(
            reaction, dtype=torch.int64, device=self.device
        )
        self.term_coefficient = torch.as_tensor(
            coefficient, dtype=torch.float32, device=self.device
        )

    def _shared_candidate_activity(self, candidates, biomass):
        import torch

        batch, candidate_count, _ = candidates.shape
        metabolite_count = len(self.shared_metabolite_ids)
        output = torch.zeros(
            (batch, candidate_count, metabolite_count),
            dtype=candidates.dtype,
            device=self.device,
        )
        if not self.term_reaction.numel():
            return output
        flux_terms = candidates.index_select(2, self.term_reaction)
        coefficients = (
            biomass.index_select(1, self.term_species)
            * self.term_coefficient[None, :]
        )
        contributions = flux_terms * coefficients[:, None, :]
        indices = self.term_metabolite[None, None, :].expand(
            batch, candidate_count, -1
        )
        output.scatter_add_(2, indices, contributions)
        return output

    def project(
        self,
        candidates,
        lower_bounds,
        upper_bounds,
        biomass_g_l,
        shared_supply,
        reference_weights=None,
        target_reaction_index: int | None = None,
        target_flux=None,
        target_reaction_indices: Sequence[int] | None = None,
        target_reaction_weights: Sequence[float] | None = None,
        enforce_target_in_hull: bool = False,
        target_relative_tolerance: float = 0.0025,
    ) -> BatchedQpProjectionResult:
        """Project a batch of candidate pools without leaving the device.

        ``candidates`` may already be a CUDA tensor from the neural dictionary;
        all other inputs accept NumPy arrays or tensors.  Shapes are ``B,K,N``,
        ``B,N``, ``B,N``, ``B,S`` and ``B,M`` respectively.
        """

        import torch

        started = time.perf_counter()
        target_projection = None
        candidate_tensor = torch.as_tensor(
            candidates, dtype=torch.float32, device=self.device
        )
        if candidate_tensor.ndim == 2:
            candidate_tensor = candidate_tensor[None, :, :]
        batch, candidate_count, reaction_count = candidate_tensor.shape
        original_candidate_count = candidate_count
        if self.species_offsets:
            if len(self.species_offsets) != len(self.species_names):
                raise ValueError("species offset count does not match species names")
            compose_count = min(
                self.block_composition_candidates, candidate_count
            )
            composed = []
            for choices in product(
                range(compose_count), repeat=len(self.species_offsets)
            ):
                if len(set(choices)) == 1:
                    # This whole-community candidate is already present.
                    continue
                anchor = torch.zeros(
                    (batch, reaction_count),
                    dtype=candidate_tensor.dtype,
                    device=self.device,
                )
                for species_index, (start, end) in enumerate(
                    self.species_offsets
                ):
                    anchor[:, start:end] = candidate_tensor[
                        :, choices[species_index], start:end
                    ]
                composed.append(anchor)
            if composed:
                candidate_tensor = torch.cat(
                    (candidate_tensor, torch.stack(composed, dim=1)), dim=1
                )
                candidate_count = candidate_tensor.shape[1]

        supplied_reference = None
        if reference_weights is not None:
            supplied_reference = torch.as_tensor(
                reference_weights, dtype=torch.float32, device=self.device
            )
            if supplied_reference.ndim == 1:
                supplied_reference = supplied_reference[None, :]
            if supplied_reference.shape != (batch, original_candidate_count):
                raise ValueError(
                    "reference weights must align with the uncomposed candidate pool"
                )
            if candidate_count > original_candidate_count:
                supplied_reference = torch.nn.functional.pad(
                    supplied_reference,
                    (0, candidate_count - original_candidate_count),
                )

        def tensor2(values, width):
            result = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            if result.ndim == 1:
                result = result[None, :]
            if result.shape != (batch, width):
                raise ValueError(
                    f"QP input shape {tuple(result.shape)} does not match {(batch, width)}"
                )
            return result

        lower = tensor2(lower_bounds, reaction_count)
        upper = tensor2(upper_bounds, reaction_count)
        biomass = tensor2(biomass_g_l, len(self.species_names))
        shared_rhs = tensor2(shared_supply, len(self.shared_metabolite_ids))
        if torch.any(lower > upper):
            raise ValueError("lower reaction bounds exceed upper bounds")

        finite_lower = torch.isfinite(lower)
        finite_upper = torch.isfinite(upper)
        lower_safe = torch.where(finite_lower, lower, torch.zeros_like(lower))
        upper_safe = torch.where(finite_upper, upper, torch.zeros_like(upper))
        candidate_abs = torch.amax(torch.abs(candidate_tensor), dim=1)
        bound_scale = torch.maximum(candidate_abs, torch.ones_like(candidate_abs))
        bound_scale = torch.maximum(
            bound_scale,
            torch.where(finite_lower, torch.abs(lower_safe), torch.zeros_like(lower)),
        )
        bound_scale = torch.maximum(
            bound_scale,
            torch.where(finite_upper, torch.abs(upper_safe), torch.zeros_like(upper)),
        )

        shared_candidates = self._shared_candidate_activity(candidate_tensor, biomass)
        shared_scale = torch.maximum(
            torch.amax(torch.abs(shared_candidates), dim=1),
            torch.maximum(torch.abs(shared_rhs), torch.ones_like(shared_rhs)),
        )

        candidate_lower_violation = torch.where(
            finite_lower[:, None, :],
            torch.relu(lower_safe[:, None, :] - candidate_tensor),
            torch.zeros_like(candidate_tensor),
        )
        candidate_upper_violation = torch.where(
            finite_upper[:, None, :],
            torch.relu(candidate_tensor - upper_safe[:, None, :]),
            torch.zeros_like(candidate_tensor),
        )
        candidate_bound_max = torch.amax(
            torch.maximum(candidate_lower_violation, candidate_upper_violation),
            dim=2,
        )
        candidate_shared_max = torch.amax(
            torch.relu(shared_candidates - shared_rhs[:, None, :]), dim=2
        )
        candidate_feasible = (
            candidate_bound_max <= self.bound_tolerance
        ) & (candidate_shared_max <= self.shared_tolerance)
        has_feasible_candidate = torch.any(candidate_feasible, dim=1)
        first_feasible = torch.argmax(candidate_feasible.to(torch.int64), dim=1)

        # Reproduce the CPU LP's lexicographic intent inside the candidate
        # hull: first retain the maximum common growth fraction, then choose
        # the feasible anchor with the smallest biomass-weighted absolute
        # environmental transfer.  The latter is the same quantity minimized
        # by the CPU parsimonious-exchange stage (up to candidate discretisation).
        if self.parsimonious_reference and self.growth_reaction_indices:
            growth_indices = torch.as_tensor(
                self.growth_reaction_indices,
                dtype=torch.int64,
                device=self.device,
            )
            growth_coefficients = torch.as_tensor(
                self.growth_coefficients,
                dtype=candidate_tensor.dtype,
                device=self.device,
            )
            candidate_growth = candidate_tensor.index_select(2, growth_indices)
            candidate_growth = candidate_growth * growth_coefficients[None, None, :]
            common_growth = torch.amin(candidate_growth, dim=2)
            negative_infinity = torch.full_like(common_growth, -torch.inf)
            best_common_growth = torch.amax(
                torch.where(candidate_feasible, common_growth, negative_infinity),
                dim=1,
            )
            growth_floor = (
                best_common_growth * self.coexistence_fraction
            )[:, None]
            eligible = candidate_feasible & (common_growth >= growth_floor - 1e-7)
            if self.term_reaction.numel():
                exchange_flux = candidate_tensor.index_select(
                    2, self.term_reaction
                )
                exchange_scale = (
                    biomass.index_select(1, self.term_species)
                    * torch.abs(self.term_coefficient)[None, :]
                )
                transfer_l1 = torch.sum(
                    torch.abs(exchange_flux) * exchange_scale[:, None, :], dim=2
                )
            else:
                transfer_l1 = torch.zeros_like(common_growth)
            infinity = torch.full_like(transfer_l1, torch.inf)
            parsimonious_index = torch.argmin(
                torch.where(eligible, transfer_l1, infinity), dim=1
            )
            first_feasible = torch.where(
                torch.any(eligible, dim=1), parsimonious_index, first_feasible
            )

        weights_reference = torch.zeros(
            (batch, candidate_count), dtype=torch.float32, device=self.device
        )
        reference_indices = torch.where(
            has_feasible_candidate,
            first_feasible,
            torch.zeros_like(first_feasible),
        )
        weights_reference.scatter_(1, reference_indices[:, None], 1.0)
        if supplied_reference is not None:
            supplied_reference = torch.clamp(supplied_reference, min=0.0)
            supplied_reference = torch.where(
                candidate_feasible,
                supplied_reference,
                torch.zeros_like(supplied_reference),
            )
            supplied_sum = supplied_reference.sum(dim=1, keepdim=True)
            normalized_reference = supplied_reference / supplied_sum.clamp_min(1e-12)
            weights_reference = torch.where(
                supplied_sum > 1e-12,
                normalized_reference,
                weights_reference,
            )
            reference_indices = torch.argmax(weights_reference, dim=1)
        if (
            target_reaction_index is not None or target_reaction_indices is not None
        ) and target_flux is not None:
            if target_reaction_indices is None:
                target_reaction_indices = [int(target_reaction_index)]
            indices = [int(value) for value in target_reaction_indices]
            if not indices or any(not 0 <= value < reaction_count for value in indices):
                raise ValueError("target reaction indices are empty or out of range")
            coefficients = (
                [1.0] * len(indices)
                if target_reaction_weights is None
                else list(target_reaction_weights)
            )
            if len(coefficients) != len(indices):
                raise ValueError("target reaction indices and weights must align")
            target = torch.as_tensor(
                target_flux, dtype=torch.float32, device=self.device
            ).reshape(-1)
            if target.shape != (batch,):
                raise ValueError("target flux must provide one value per batch row")
            index_tensor = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
            coefficient_tensor = torch.as_tensor(
                coefficients, dtype=torch.float32, device=self.device
            )
            values = torch.sum(
                candidate_tensor.index_select(2, index_tensor)
                * coefficient_tensor[None, None, :], dim=2
            )
            infinity = torch.full_like(values, torch.inf)
            below_gap = torch.where(
                candidate_feasible & (values <= target[:, None]),
                target[:, None] - values,
                infinity,
            )
            above_gap = torch.where(
                candidate_feasible & (values >= target[:, None]),
                values - target[:, None],
                infinity,
            )
            below_gap_min, below_index = torch.min(below_gap, dim=1)
            above_gap_min, above_index = torch.min(above_gap, dim=1)
            bracketed = torch.isfinite(below_gap_min) & torch.isfinite(above_gap_min)
            closest_gap, closest_index = torch.min(
                torch.where(
                    candidate_feasible,
                    torch.abs(values - target[:, None]),
                    infinity,
                ),
                dim=1,
            )
            del closest_gap
            weights_reference.zero_()
            weights_reference.scatter_(1, closest_index[:, None], 1.0)
            lower_value = values.gather(1, below_index[:, None]).squeeze(1)
            upper_value = values.gather(1, above_index[:, None]).squeeze(1)
            span = upper_value - lower_value
            upper_weight = torch.where(
                bracketed & (torch.abs(span) > 1e-12),
                (target - lower_value) / torch.clamp(span, min=1e-12),
                torch.zeros_like(target),
            ).clamp(0.0, 1.0)
            lower_weight = 1.0 - upper_weight
            bracket_weights = torch.zeros_like(weights_reference)
            bracket_weights.scatter_add_(1, below_index[:, None], lower_weight[:, None])
            bracket_weights.scatter_add_(1, above_index[:, None], upper_weight[:, None])
            weights_reference = torch.where(
                bracketed[:, None], bracket_weights, weights_reference
            )
            # A feasible anchor need not be accurate. If the neural target is
            # outside the range of individually feasible anchors, try mixtures
            # of *all* retrieved anchors, including complementary infeasible
            # ones. A virtual flux column encodes a linear target-band constraint
            # without changing stoichiometry or relaxing physical constraints.
            band = torch.maximum(target.abs() * target_relative_tolerance,
                                 torch.full_like(target, 1e-5))
            reference_target = (weights_reference * values).sum(dim=1)
            target_in_full_range = (target >= values.amin(dim=1)) & (target <= values.amax(dim=1))
            needs_target_projection = ((reference_target-target).abs() > band) & target_in_full_range
            if enforce_target_in_hull and bool(needs_target_projection.any()):
                augmented = torch.cat((candidate_tensor, values[:, :, None]), dim=2)
                # The outer call already composed species blocks. Composing
                # again would lose the virtual column's linear identity.
                inner = copy.copy(self)
                inner.species_offsets = ()
                inner.iterative_method = "admm"
                target_projection = inner.project(
                    augmented,
                    torch.cat((lower, (target-band)[:, None]), dim=1),
                    torch.cat((upper, (target+band)[:, None]), dim=1),
                    biomass, shared_rhs, reference_weights=weights_reference,
                )
                actual_flux = torch.as_tensor(target_projection.fluxes[:, :-1],
                    dtype=torch.float32, device=self.device)
                actual_shared = self._shared_candidate_activity(actual_flux[:, None], biomass)[:, 0]
                actual_bounds = torch.maximum(torch.relu(lower-actual_flux), torch.relu(actual_flux-upper))
                actual_shared_residual = torch.relu(actual_shared-shared_rhs)
                target_projection = replace(
                    target_projection, fluxes=target_projection.fluxes[:, :-1],
                    max_bound_violation=actual_bounds.amax(dim=1).cpu().numpy(),
                    max_shared_violation=actual_shared_residual.amax(dim=1).cpu().numpy(),
                    max_shared_violation_index=actual_shared_residual.argmax(dim=1).cpu().numpy(),
                    max_normalized_violation=torch.maximum(
                        (actual_bounds/bound_scale).amax(dim=1),
                        (actual_shared_residual/shared_scale).amax(dim=1),
                    ).cpu().numpy(),
                )
                if bool(np.all(target_projection.feasible)):
                    return replace(target_projection,
                        inference_seconds=time.perf_counter()-started,
                        target_projection_attempted=True,
                        target_projection_success=target_projection.feasible.copy())

        def finish(result):
            if target_projection is None:
                return result
            accepted = target_projection.feasible
            # If a neural target is unattainable, retain a physically feasible
            # GPU result, never a target-violating LP or an implicit CPU solve.
            updates = {}
            for field in ("fluxes", "weights", "feasible", "max_bound_violation",
                          "max_shared_violation", "max_shared_violation_index",
                          "max_normalized_violation"):
                current = getattr(result, field)
                proposed = getattr(target_projection, field)
                mask = accepted[:, None] if current.ndim == 2 else accepted
                updates[field] = np.where(mask, proposed, current)
            return replace(result, **updates,
                iterations=result.iterations+target_projection.iterations,
                inference_seconds=time.perf_counter()-started,
                target_projection_attempted=True,
                target_projection_success=accepted.copy())
        weights = weights_reference.clone()
        extrapolated = weights.clone()
        dual_lower = torch.zeros_like(lower)
        dual_upper = torch.zeros_like(upper)
        dual_shared = torch.zeros_like(shared_rhs)

        # In the common case an exact dictionary candidate already satisfies
        # the live bounds and shared-medium constraints.  Selecting it on the
        # GPU is the exact solution of the projection QP (zero distance to the
        # chosen reference) and must not pay hundreds of iterative launches.
        if bool(torch.all(has_feasible_candidate)):
            projected = torch.bmm(
                weights[:, None, :], candidate_tensor
            ).squeeze(1)
            # Diagnostics must describe the actual convex blend, not the
            # anchor chosen before target interpolation changed its weights.
            bound_residual = torch.maximum(
                torch.where(finite_lower, torch.relu(lower_safe - projected), 0.0),
                torch.where(finite_upper, torch.relu(projected - upper_safe), 0.0),
            )
            shared_activity = torch.bmm(weights[:, None, :], shared_candidates).squeeze(1)
            shared_residual = torch.relu(shared_activity - shared_rhs)
            selected_bound = bound_residual.amax(dim=1)
            selected_shared, selected_shared_index = shared_residual.max(dim=1)
            selected_normalized = torch.maximum(
                (bound_residual / bound_scale).amax(dim=1),
                (shared_residual / shared_scale).amax(dim=1),
            )
            actual_feasible = (selected_bound <= self.bound_tolerance) & (
                selected_shared <= self.shared_tolerance
            ) & torch.isfinite(projected).all(dim=1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            return finish(BatchedQpProjectionResult(
                fluxes=projected.detach().cpu().numpy().astype(np.float64, copy=False),
                weights=weights.detach().cpu().numpy().astype(np.float64, copy=False),
                feasible=actual_feasible.detach().cpu().numpy(),
                iterations=0,
                inference_seconds=elapsed,
                max_bound_violation=selected_bound.detach().cpu().numpy().astype(float, copy=False),
                max_shared_violation=selected_shared.detach().cpu().numpy().astype(float, copy=False),
                max_shared_violation_index=selected_shared_index.detach().cpu().numpy(),
                max_normalized_violation=selected_normalized.detach().cpu().numpy().astype(float, copy=False),
            ))

        # A convex combination automatically satisfies any linear constraint
        # satisfied by every anchor. Remove those rows from the iterative
        # operator (but not the final physical audit). Otherwise thousands of
        # irrelevant internal-cycle columns dominate the norm/step size.
        active_lower = finite_lower & (candidate_tensor.amin(dim=1) < lower_safe)
        active_upper = finite_upper & (candidate_tensor.amax(dim=1) > upper_safe)
        bound_columns = torch.nonzero((active_lower | active_upper).any(dim=0)).flatten()
        shared_columns = torch.nonzero((shared_candidates.amax(dim=1) > shared_rhs).any(dim=0)).flatten()
        operator_candidates = candidate_tensor.index_select(2, bound_columns)
        operator_shared = shared_candidates.index_select(2, shared_columns)
        operator_lower = lower_safe.index_select(1, bound_columns)
        operator_upper = upper_safe.index_select(1, bound_columns)
        operator_lower_mask = active_lower.index_select(1, bound_columns)
        operator_upper_mask = active_upper.index_select(1, bound_columns)
        operator_bound_scale = bound_scale.index_select(1, bound_columns)
        operator_rhs = shared_rhs.index_select(1, shared_columns)
        operator_shared_scale = shared_scale.index_select(1, shared_columns)
        dual_lower = torch.zeros_like(operator_lower)
        dual_upper = torch.zeros_like(operator_upper)
        dual_shared = torch.zeros_like(operator_rhs)

        def row_max(values):
            return values.amax(dim=1) if values.shape[1] else torch.zeros(batch, device=self.device)

        # Power iteration estimates ||K|| for the normalized bound/shared
        # operator.  This is much less pessimistic than a Frobenius bound and
        # keeps PDHG useful for 6,585-reaction models.
        probe = torch.full_like(weights, 1.0 / np.sqrt(candidate_count))
        operator_norm_squared = torch.ones(batch, device=self.device)
        for _ in range(8):
            flux_probe = torch.bmm(probe[:, None, :], operator_candidates).squeeze(1)
            lower_probe = torch.where(
                operator_lower_mask, -flux_probe / operator_bound_scale, torch.zeros_like(flux_probe)
            )
            upper_probe = torch.where(
                operator_upper_mask, flux_probe / operator_bound_scale, torch.zeros_like(flux_probe)
            )
            shared_probe = torch.bmm(
                probe[:, None, :], operator_shared
            ).squeeze(1) / operator_shared_scale
            adjoint = torch.bmm(
                operator_candidates,
                ((upper_probe - lower_probe) / operator_bound_scale)[:, :, None],
            ).squeeze(2)
            adjoint += torch.bmm(
                operator_shared,
                (shared_probe / operator_shared_scale)[:, :, None],
            ).squeeze(2)
            operator_norm = torch.linalg.vector_norm(adjoint, dim=1).clamp_min(1e-6)
            probe = adjoint / operator_norm[:, None]
            operator_norm_squared = torch.sum(probe * adjoint, dim=1).clamp_min(1e-6)
        operator_norm = torch.sqrt(operator_norm_squared).clamp_min(1.0)
        tau = self.step_safety / operator_norm
        sigma = self.step_safety / operator_norm

        iterations = 0
        iteration_budget = self.maximum_iterations
        if getattr(self, "iterative_method", "pdhg") == "admm":
            weights, iterations = self._admm_weights(
                weights_reference, operator_candidates, operator_shared,
                operator_lower, operator_upper, operator_rhs,
            )
            iteration_budget = 0
        for iteration in range(iteration_budget):
            flux_extrapolated = torch.bmm(
                extrapolated[:, None, :], operator_candidates
            ).squeeze(1)
            shared_extrapolated = torch.bmm(
                extrapolated[:, None, :], operator_shared
            ).squeeze(1)
            dual_lower = torch.relu(
                dual_lower
                + sigma[:, None]
                * torch.where(
                    operator_lower_mask,
                    (operator_lower - flux_extrapolated) / operator_bound_scale,
                    torch.zeros_like(flux_extrapolated),
                )
            )
            dual_upper = torch.relu(
                dual_upper
                + sigma[:, None]
                * torch.where(
                    operator_upper_mask,
                    (flux_extrapolated - operator_upper) / operator_bound_scale,
                    torch.zeros_like(flux_extrapolated),
                )
            )
            dual_shared = torch.relu(
                dual_shared
                + sigma[:, None]
                * ((shared_extrapolated - operator_rhs) / operator_shared_scale)
            )

            gradient = torch.bmm(
                operator_candidates,
                ((dual_upper - dual_lower) / operator_bound_scale)[:, :, None],
            ).squeeze(2)
            gradient += torch.bmm(
                operator_shared,
                (dual_shared / operator_shared_scale)[:, :, None],
            ).squeeze(2)
            previous = weights
            proximal = (
                weights
                - tau[:, None] * gradient
                + tau[:, None] * self.quadratic_strength * weights_reference
            ) / (1.0 + tau[:, None] * self.quadratic_strength)
            weights = _simplex_projection(proximal)
            extrapolated = 2.0 * weights - previous
            iterations = iteration + 1

            if iterations % self.check_interval == 0 or iterations == self.maximum_iterations:
                flux_check = torch.bmm(
                    weights[:, None, :], operator_candidates
                ).squeeze(1)
                shared_check = torch.bmm(
                    weights[:, None, :], operator_shared
                ).squeeze(1)
                lower_violation = torch.where(
                    operator_lower_mask,
                    torch.relu(operator_lower - flux_check) / operator_bound_scale,
                    torch.zeros_like(flux_check),
                )
                upper_violation = torch.where(
                    operator_upper_mask,
                    torch.relu(flux_check - operator_upper) / operator_bound_scale,
                    torch.zeros_like(flux_check),
                )
                shared_violation = torch.relu(shared_check - operator_rhs) / operator_shared_scale
                maximum = torch.maximum(
                    row_max(torch.maximum(lower_violation, upper_violation)),
                    row_max(shared_violation),
                )
                physical_bound = row_max(
                    torch.maximum(
                        lower_violation * operator_bound_scale,
                        upper_violation * operator_bound_scale,
                    ),
                )
                physical_shared = row_max(
                    shared_violation * operator_shared_scale
                )
                if bool(
                    torch.all(
                        (maximum <= self.normalized_tolerance)
                        | (
                            (physical_bound <= self.bound_tolerance)
                            & (physical_shared <= self.shared_tolerance)
                        )
                    )
                ):
                    break

        projected = torch.bmm(weights[:, None, :], candidate_tensor).squeeze(1)
        projected_shared = torch.bmm(
            weights[:, None, :], shared_candidates
        ).squeeze(1)
        physical_lower = torch.where(
            finite_lower,
            torch.relu(lower_safe - projected),
            torch.zeros_like(projected),
        )
        physical_upper = torch.where(
            finite_upper,
            torch.relu(projected - upper_safe),
            torch.zeros_like(projected),
        )
        bound_violation = torch.amax(
            torch.maximum(physical_lower, physical_upper), dim=1
        )
        shared_violation = torch.amax(
            torch.relu(projected_shared - shared_rhs), dim=1
        )
        shared_violation_index = torch.argmax(
            torch.relu(projected_shared - shared_rhs), dim=1
        )
        normalized_bound = torch.amax(
            torch.maximum(
                physical_lower / bound_scale, physical_upper / bound_scale
            ),
            dim=1,
        )
        normalized_shared = torch.amax(
            torch.relu(projected_shared - shared_rhs) / shared_scale, dim=1
        )
        normalized_violation = torch.maximum(normalized_bound, normalized_shared)
        feasible = (bound_violation <= self.bound_tolerance) & (
            shared_violation <= self.shared_tolerance
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        return finish(BatchedQpProjectionResult(
            fluxes=projected.detach().cpu().numpy().astype(np.float64, copy=False),
            weights=weights.detach().cpu().numpy().astype(np.float64, copy=False),
            feasible=feasible.detach().cpu().numpy().astype(bool, copy=False),
            iterations=iterations,
            inference_seconds=elapsed,
            max_bound_violation=bound_violation.detach().cpu().numpy().astype(float, copy=False),
            max_shared_violation=shared_violation.detach().cpu().numpy().astype(float, copy=False),
            max_shared_violation_index=shared_violation_index.detach().cpu().numpy().astype(np.int64, copy=False),
            max_normalized_violation=normalized_violation.detach().cpu().numpy().astype(float, copy=False),
        ))

    def _admm_weights(self, reference, flux_operator, shared_operator, lower, upper, supply,
                      soft_operator=None, simplex_groups=1, allowed_mask=None, signed_weights=False):
        """GPU ADMM for the compact hull QP; no CPU numerical solve.

        Nonnegativity is split off as an identity block. A Woodbury update
        factors the small constraint-space matrix once, instead of factoring
        the complete GEM or a K-by-K candidate system on every iteration.
        Only immutable-hull-redundant rows were removed; final audit is full.
        """
        import torch

        batch, count = reference.shape
        if allowed_mask is not None and allowed_mask.shape != reference.shape:
            raise ValueError("allowed candidate mask shape mismatch")
        if simplex_groups < 1 or count % simplex_groups:
            raise ValueError("candidate count must divide into complete simplex groups")
        simplex_rows = torch.eye(simplex_groups, device=self.device).repeat_interleave(
            count//simplex_groups, dim=1)[None].expand(batch, -1, -1)
        simplex_rhs = torch.full((batch, simplex_groups), 1.0/simplex_groups, device=self.device)
        dtype = torch.float64
        matrix = torch.cat((flux_operator.transpose(1, 2),
                            shared_operator.transpose(1, 2),
                            simplex_rows), dim=1).to(dtype)
        row_lower = torch.cat((lower, torch.full_like(supply, -torch.inf),
                               simplex_rhs), dim=1).to(dtype)
        row_upper = torch.cat((upper, supply,
                               simplex_rhs), dim=1).to(dtype)
        scale = matrix.abs().amax(dim=2).clamp_min(1e-8)
        matrix = matrix / scale[:, :, None]
        row_lower, row_upper = row_lower / scale, row_upper / scale
        tolerance = torch.cat((torch.full_like(lower, self.bound_tolerance),
                               torch.full_like(supply, self.shared_tolerance),
                               torch.full((batch, simplex_groups), 1e-6/simplex_groups, device=self.device)), dim=1).to(dtype)
        rho = 100.0
        alpha = self.quadratic_strength
        beta = alpha + rho
        transposed = matrix.transpose(1, 2)
        # Extra rows are a quadratic objective, NOT physical constraints.
        # soft_operator contains sqrt(weight) * (candidate flux - target)/scale.
        factor_matrix = matrix if soft_operator is None else torch.cat(
            (matrix, soft_operator.to(dtype)/np.sqrt(rho)), dim=1)
        factor_transposed = factor_matrix.transpose(1, 2)
        row_count = factor_matrix.shape[1]
        use_woodbury = row_count < count
        if use_woodbury:
            factor = torch.linalg.cholesky(
                factor_matrix @ factor_transposed + (beta/rho) * torch.eye(row_count, dtype=dtype, device=self.device)[None]
            )
        else:
            factor = torch.linalg.cholesky(
                rho*(factor_transposed @ factor_matrix) + beta*torch.eye(count, dtype=dtype, device=self.device)[None]
            )
        ref = reference.to(dtype)
        x = ref.clone()
        positive = x.clamp_min(0)
        dual_positive = torch.zeros_like(x)
        activity = (matrix @ x[:, :, None]).squeeze(2)
        boxed = torch.maximum(torch.minimum(activity, row_upper), row_lower)
        dual_box = torch.zeros_like(boxed)
        projected = reference
        previous_objective = None
        stable_checks = 0
        for iteration in range(self.maximum_iterations):
            rhs = alpha*ref + rho*(positive-dual_positive +
                (transposed @ (boxed-dual_box)[:, :, None]).squeeze(2))
            if use_woodbury:
                correction = torch.cholesky_solve(factor_matrix @ rhs[:, :, None], factor)
                x = (rhs - (factor_transposed @ correction).squeeze(2)) / beta
            else:
                x = torch.cholesky_solve(rhs[:, :, None], factor).squeeze(2)
            activity = (matrix @ x[:, :, None]).squeeze(2)
            positive = x+dual_positive if signed_weights else (x+dual_positive).clamp_min(0)
            if allowed_mask is not None:
                positive = positive*allowed_mask
            boxed = torch.maximum(torch.minimum(activity+dual_box, row_upper), row_lower)
            dual_positive += x-positive
            dual_box += activity-boxed
            if (iteration+1) % self.check_interval == 0 or iteration+1 == self.maximum_iterations:
                simplex_input = x if allowed_mask is None else torch.where(allowed_mask, x,
                                                                            torch.full_like(x, -1e30))
                grouped = simplex_input.reshape(batch*simplex_groups, -1)
                if signed_weights:
                    projected = (grouped + (1.0/simplex_groups-grouped.sum(dim=1, keepdim=True))
                                 /grouped.shape[1]).reshape(batch, count).to(torch.float32)
                else:
                    projected = (_simplex_projection(grouped*simplex_groups)
                                 /simplex_groups).reshape(batch, count).to(torch.float32)
                check = (matrix @ projected.to(dtype)[:, :, None]).squeeze(2)
                residual = torch.maximum(torch.relu(row_lower-check), torch.relu(check-row_upper)) * scale
                physically_feasible = bool(torch.all(residual <= tolerance))
                if soft_operator is None:
                    if physically_feasible:
                        break
                else:
                    current = projected.to(dtype)
                    error = (soft_operator.to(dtype) @ current[:, :, None]).squeeze(2)
                    objective = 0.5*error.square().sum(dim=1) + 0.5*alpha*(current-ref).square().sum(dim=1)
                    stable = previous_objective is not None and bool(torch.all(
                        (objective-previous_objective).abs() <= 1e-5*(1+objective.abs())))
                    stable_checks = stable_checks+1 if stable else 0
                    previous_objective = objective
                    if physically_feasible and iteration+1 >= 100 and stable_checks >= 4:
                        break
        return projected, iteration+1
