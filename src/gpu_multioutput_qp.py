"""Joint growth/product/exchange matching on the GPU with physical constraints.

The neural target is soft and may be infeasible. Only independently audited
physical fluxes can be returned. This is a surrogate QP, not a certificate of
the full CPU lexicographic LP optimum.
"""
from dataclasses import replace
from itertools import product
import time

import numpy as np


def structural_face_mask(flux, lower, upper, groups=1):
    """Eliminate weights forced to zero on exact one-sided boundary faces.

    E.g. all candidates have v >= 0 and the live bound is v <= 0: a
    positive-v anchor cannot have positive weight in any exact feasible mix.
    Use strict comparisons, not a relaxed numerical boundary or heuristic.
    """
    import torch
    minimum, maximum = flux.amin(dim=1), flux.amax(dim=1)
    upper_face = (minimum >= upper) & (maximum > upper)
    lower_face = (maximum <= lower) & (minimum < lower)
    active = (upper_face | lower_face).any(dim=0)
    allowed = ~(((flux[:, :, active] > upper[:, None, active]) & upper_face[:, None, active]) |
                ((flux[:, :, active] < lower[:, None, active]) & lower_face[:, None, active])).any(dim=2)
    # Empty exact faces can result from rounded anchors. Preserve the existing
    # tolerance-based fallback instead of introducing an empty simplex.
    nonempty = allowed.reshape(len(flux), groups, -1).any(dim=2).all(dim=1)
    return torch.where(nonempty[:, None], allowed, torch.ones_like(allowed))


class MultiOutputCooperativeQpProjector:
    def __init__(self, physical_projector, strength=100.0, independent_species=False,
                 signed_weights=False, stoichiometry=None, mass_tolerance=2e-3):
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("multi-output strength must be positive and finite")
        self.base = physical_projector
        self.strength = float(strength)
        self.independent_species = bool(independent_species)
        self.signed_weights = bool(signed_weights)
        self.mass_tolerance = float(mass_tolerance)
        self.stoichiometry = None
        if self.signed_weights:
            if stoichiometry is None or independent_species:
                raise ValueError("affine projection requires stoichiometry and the joint basis")
            import torch
            matrix = stoichiometry.tocsr()
            self.stoichiometry = torch.sparse_csr_tensor(matrix.indptr, matrix.indices,
                matrix.data, size=matrix.shape, dtype=torch.float64, device=self.base.device,
                check_invariants=True)
        if self.independent_species and not self.base.species_offsets:
            raise ValueError("independent mixing requires species offsets")

    def project(self, candidates, lower_bounds, upper_bounds, biomass_g_l, shared_supply,
                reference_weights=None, decision_indices=None, decision_targets=None,
                decision_scales=None, decision_weights=None, **unused_scalar_target):
        import torch

        started = time.perf_counter()
        base, device = self.base, self.base.device
        flux = torch.as_tensor(candidates, dtype=torch.float32, device=device)
        if flux.ndim == 2:
            flux = flux[None]
        batch, original_count, n = flux.shape
        ids = torch.as_tensor(decision_indices, dtype=torch.int64, device=device)
        if ids.ndim != 1 or not ids.numel() or bool(torch.any((ids < 0) | (ids >= n))):
            raise ValueError("invalid decision reaction indices")
        target = torch.as_tensor(decision_targets, dtype=torch.float32, device=device)
        if target.ndim == 1:
            target = target[None]
        if target.shape != (batch, len(ids)) or not bool(torch.isfinite(target).all()):
            raise ValueError("decision targets must be finite B x D values")
        scales = torch.as_tensor(decision_scales, dtype=torch.float32, device=device)
        weights = torch.ones_like(scales) if decision_weights is None else torch.as_tensor(
            decision_weights, dtype=torch.float32, device=device)
        if scales.shape != ids.shape or weights.shape != ids.shape or not bool(
            torch.all(torch.isfinite(scales) & (scales > 0) & torch.isfinite(weights) & (weights >= 0))
        ):
            raise ValueError("decision scales/weights are invalid")
        baseline = base.project(flux, lower_bounds, upper_bounds, biomass_g_l,
                                shared_supply, reference_weights=reference_weights)
        if self.stoichiometry is not None:
            baseline_mass = torch.sparse.mm(self.stoichiometry,
                torch.as_tensor(baseline.fluxes.T, dtype=torch.float64, device=device)).abs().amax(dim=0)
            baseline = replace(baseline, feasible=baseline.feasible &
                               (baseline_mass <= self.mass_tolerance).cpu().numpy())
        composed = []
        if base.species_offsets:
            for choices in product(range(min(base.block_composition_candidates, original_count)),
                                   repeat=len(base.species_offsets)):
                if len(set(choices)) == 1:
                    continue
                row = torch.zeros((batch, n), device=device)
                for choice, (start, end) in zip(choices, base.species_offsets):
                    row[:, start:end] = flux[:, choice, start:end]
                composed.append(row)
        if composed:
            flux = torch.cat((flux, torch.stack(composed, dim=1)), dim=1)
        if baseline.weights.shape != flux.shape[:2]:
            raise RuntimeError("baseline and multi-output candidate layout disagree")

        def rows(values, width):
            out = torch.as_tensor(values, dtype=torch.float32, device=device)
            if out.ndim == 1:
                out = out[None]
            if out.shape != (batch, width):
                raise ValueError("multi-output physical input shape mismatch")
            return out

        lower, upper = rows(lower_bounds, n), rows(upper_bounds, n)
        biomass = rows(biomass_g_l, len(base.species_names))
        supply = rows(shared_supply, len(base.shared_metabolite_ids))
        # Each reaction belongs to one species. Its achievable min/max is
        # unchanged by replacing the joint simplex with one simplex per species.
        active = ((flux.amin(dim=1) < lower) | (flux.amax(dim=1) > upper)).any(dim=0)
        if self.signed_weights:
            # Convex-hull min/max bounds do NOT certify an affine extrapolation.
            active |= (flux.amax(dim=1) != flux.amin(dim=1)).any(dim=0)
        groups = len(base.species_offsets) if self.independent_species else 1
        if groups > 1:
            joint_count = flux.shape[1]
            independent_flux = torch.zeros((batch, groups*joint_count, n), device=device)
            for group, (start, end) in enumerate(base.species_offsets):
                independent_flux[:, group*joint_count:(group+1)*joint_count, start:end] = groups*flux[:, :, start:end]
            flux = independent_flux
            baseline = replace(baseline, weights=np.tile(baseline.weights, (1, groups))/groups)
        shared = base._shared_candidate_activity(flux, biomass)
        active_shared = (shared.amax(dim=1) > supply).any(dim=0)
        if self.signed_weights:
            active_shared |= (shared.amax(dim=1) != shared.amin(dim=1)).any(dim=0)
        # Clipping a prediction to a live interval does not alter that interval.
        target = target.maximum(lower.index_select(1, ids)).minimum(upper.index_select(1, ids))
        residual = (flux.index_select(2, ids)-target[:, None, :])/scales
        weighted = residual * torch.sqrt(weights*self.strength/len(ids))[None, None, :]
        soft = weighted.transpose(1, 2)
        reference = torch.as_tensor(baseline.weights, dtype=torch.float32, device=device)
        projected_weights, iterations = base._admm_weights(reference, flux[:, :, active],
            shared[:, :, active_shared], lower[:, active], upper[:, active], supply[:, active_shared],
            soft_operator=soft, simplex_groups=groups,
            allowed_mask=structural_face_mask(flux, lower, upper, groups) if groups > 1 else None,
            signed_weights=self.signed_weights)
        # Float64 convex accumulation avoids spurious residuals at exact bounds
        # when large internal cycles cancel. All arithmetic remains on the GPU.
        projected_weights = projected_weights.to(torch.float64)
        if not self.signed_weights:
            projected_weights = projected_weights.clamp_min(0)
        grouped = projected_weights.reshape(batch, groups, -1)
        grouped /= groups*grouped.sum(dim=2, keepdim=True).clamp_min(1e-12)
        projected_weights = grouped.reshape(batch, -1)
        projected = (projected_weights[:, None] @ flux.to(torch.float64)).squeeze(1)
        projected_shared = base._shared_candidate_activity(projected[:, None], biomass.to(torch.float64))[:, 0]
        # An unconverged ADMM iterate can still define a useful feasible
        # direction. Clip the step from the audited baseline analytically to
        # ALL physical intervals; never relax the existing tolerances.
        anchor = torch.as_tensor(baseline.fluxes, dtype=torch.float64, device=device)
        anchor_shared = base._shared_candidate_activity(anchor[:, None], biomass.to(torch.float64))[:, 0]
        delta, delta_shared = projected-anchor, projected_shared-anchor_shared
        def step_limit(slack, direction):
            return torch.where(direction > 0, slack.clamp_min(0)/direction.clamp_min(1e-300),
                               torch.full_like(direction, torch.inf)).amin(dim=1)
        fraction = torch.minimum(step_limit(upper.to(torch.float64)+base.bound_tolerance-anchor, delta),
            step_limit(anchor-lower.to(torch.float64)+base.bound_tolerance, -delta))
        fraction = torch.minimum(fraction,
            step_limit(supply.to(torch.float64)+base.shared_tolerance-anchor_shared, delta_shared)).clamp(0, 1)
        fraction = torch.where(fraction < 1, fraction*0.99, fraction)
        fraction = torch.where(torch.as_tensor(baseline.feasible, device=device), fraction,
                               torch.ones_like(fraction))
        projected_weights = (1-fraction[:, None])*reference.to(torch.float64) + fraction[:, None]*projected_weights
        grouped = projected_weights.reshape(batch, groups, -1)
        grouped /= groups*grouped.sum(dim=2, keepdim=True).clamp_min(1e-12)
        projected_weights = grouped.reshape(batch, -1)
        projected = (projected_weights[:, None] @ flux.to(torch.float64)).squeeze(1)
        projected_shared = base._shared_candidate_activity(projected[:, None], biomass.to(torch.float64))[:, 0]
        bound_residual = torch.maximum(torch.relu(lower-projected), torch.relu(projected-upper))
        shared_residual = torch.relu(projected_shared-supply)
        valid = (bound_residual.amax(dim=1) <= base.bound_tolerance) & (
            shared_residual.amax(dim=1) <= base.shared_tolerance) & torch.isfinite(projected).all(dim=1)
        if self.stoichiometry is not None:
            mass_residual = torch.sparse.mm(self.stoichiometry, projected.T).abs().amax(dim=0)
            valid &= mass_residual <= self.mass_tolerance
        before = (soft.to(torch.float64) @ reference.to(torch.float64)[:, :, None]).squeeze(2).square().sum(dim=1)
        after = (soft.to(torch.float64) @ projected_weights[:, :, None]).squeeze(2).square().sum(dim=1)
        baseline_valid = torch.as_tensor(baseline.feasible, device=device)
        improved = valid & ((after < before-1e-9) | ~baseline_valid)
        choose = improved.cpu().numpy()
        selected = projected.cpu().numpy()
        count_weights = projected_weights.cpu().numpy()
        bound_scale = torch.maximum(flux.abs().amax(dim=1), torch.ones_like(lower))
        shared_scale = torch.maximum(shared.abs().amax(dim=1), torch.ones_like(supply))
        normalized = torch.maximum((bound_residual/bound_scale).amax(dim=1),
                                    (shared_residual/shared_scale).amax(dim=1))
        return replace(baseline,
            fluxes=np.where(choose[:, None], selected, baseline.fluxes),
            weights=np.where(choose[:, None], count_weights, baseline.weights),
            feasible=np.where(choose, True, baseline.feasible),
            max_bound_violation=np.where(choose, bound_residual.amax(dim=1).cpu().numpy(), baseline.max_bound_violation),
            max_shared_violation=np.where(choose, shared_residual.amax(dim=1).cpu().numpy(), baseline.max_shared_violation),
            max_shared_violation_index=np.where(choose, shared_residual.argmax(dim=1).cpu().numpy(), baseline.max_shared_violation_index),
            max_normalized_violation=np.where(choose, normalized.cpu().numpy(), baseline.max_normalized_violation),
            iterations=baseline.iterations+iterations,
            inference_seconds=time.perf_counter()-started,
            multioutput_projection_attempted=True, multioutput_improved=choose,
            multioutput_step_fraction=fraction.cpu().numpy(),
            multioutput_loss_before=before.cpu().numpy(),
            multioutput_loss_after=np.where(choose, after.cpu().numpy(), before.cpu().numpy()))
