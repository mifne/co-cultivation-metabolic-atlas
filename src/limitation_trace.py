"""Causal nutrient-limitation tracing for the OR16--NS21 shared medium.

The diagnostic distinguishes three concepts that are often conflated in FBA:

* a depleted extracellular pool;
* a binding resource with a non-zero shadow benefit; and
* a causal rescue, where a specified external addition improves the selected
  community objective.

Only the last item is described as a model-predicted rescue.  Alternative
carbon substrates and model-specific siderophores are retained as separate
categories so that they are not mistaken for a minimal culture supplement.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .coexistence_audit import SharedMediumCommunityLP
from .metabolite_ids import canonical_metabolite_id


FEED_CANDIDATE_GROUPS: dict[str, tuple[str, ...]] = {
    "gas_transfer": ("o2_e",),
    "macronutrient": ("nh4_e", "pi_e", "so4_e"),
    "mineral": (
        "mg2_e", "ca2_e", "k_e", "fe2_e", "fe3_e", "zn2_e",
        "mn2_e", "cu2_e", "cu_e", "cobalt2_e", "ni2_e", "mobd_e",
    ),
    "vitamin": (
        "nac_e", "ribflv_e", "pnto__R_e", "thm_e", "btn_e",
        "4abz_e", "fol_e", "nicnt_e",
    ),
    "amino_acid": (
        "ala__L_e", "arg__L_e", "asn__L_e", "asp__L_e", "cys__L_e",
        "gln__L_e", "glu__L_e", "gly_e", "his__L_e", "ile__L_e",
        "leu__L_e", "lys__L_e", "met__L_e", "phe__L_e", "pro__L_e",
        "ser__L_e", "thr__L_e", "trp__L_e", "tyr__L_e", "val__L_e",
    ),
    # These are chemically feedable but can replace rubber as carbon.  They
    # are reported separately and are not included in the minimal-supplement
    # rescue set by default.
    "alternative_carbon": (
        "glc__D_e", "ac_e", "succ_e", "pyr_e", "ppa_e", "lac__L_e",
        "glyc_e", "mnl_e", "fru_e", "sbt__D_e",
    ),
    # Reconstruction-specific requirements are useful curation signals but
    # are not automatically promoted to a wet-lab recipe.
    "model_specific": (
        "salchs4fe_e", "istfrnB_e", "tsul_e", "ump_e", "xtsn_e",
        "nmn_e", "ppi_e",
    ),
}

DEFAULT_RESCUE_INCREMENT_MMOL_L_H = {
    "gas_transfer": 1.0,
    "macronutrient": 0.10,
    "mineral": 0.005,
    "vitamin": 0.001,
    "amino_acid": 0.02,
    "alternative_carbon": 0.10,
    "model_specific": 0.005,
}

MINIMAL_SUPPLEMENT_GROUPS = {
    "gas_transfer", "macronutrient", "mineral", "vitamin", "amino_acid"
}


def candidate_group(metabolite: str) -> str:
    metabolite = canonical_metabolite_id(metabolite)
    for group, members in FEED_CANDIDATE_GROUPS.items():
        if metabolite in members:
            return group
    return "non_actionable"


def candidate_ids(
    groups: Iterable[str] = MINIMAL_SUPPLEMENT_GROUPS,
) -> set[str]:
    return {
        metabolite
        for group in groups
        for metabolite in FEED_CANDIDATE_GROUPS.get(group, ())
    }


def _base_supply_cap(solver: SharedMediumCommunityLP, metabolite: str) -> float:
    # Centralize the intentional access to the LP's supply convention.  This
    # keeps finite-difference rescues numerically identical to the base solve.
    return float(solver._supply_cap(canonical_metabolite_id(metabolite), {}))


def rank_single_addition_rescues(
    solver: SharedMediumCommunityLP,
    baseline_growth: float,
    candidates: Iterable[str],
    maximum_results: Optional[int] = 20,
) -> list[dict[str, Any]]:
    """Rank finite-dose, one-at-a-time additions by common-growth gain."""

    rows = []
    available = set(solver.exchange_terms)
    for metabolite in sorted({canonical_metabolite_id(item) for item in candidates}):
        if metabolite not in available:
            continue
        group = candidate_group(metabolite)
        increment = DEFAULT_RESCUE_INCREMENT_MMOL_L_H.get(group)
        if increment is None:
            continue
        base_cap = _base_supply_cap(solver, metabolite)
        rescued_growth = solver.solve_common_growth(
            {metabolite: base_cap + increment}
        )
        gain = max(0.0, float(rescued_growth - baseline_growth))
        rows.append(
            {
                "metabolite": metabolite,
                "group": group,
                "increment_mmol_l_h": float(increment),
                "baseline_growth_per_h": float(baseline_growth),
                "rescued_growth_per_h": float(rescued_growth),
                "absolute_gain_per_h": gain,
                "relative_gain": (
                    gain / baseline_growth if baseline_growth > 1e-12 else None
                ),
            }
        )
    rows.sort(key=lambda row: row["absolute_gain_per_h"], reverse=True)
    return rows if maximum_results is None else rows[: max(0, int(maximum_results))]


def rank_present_nutrient_dropouts(
    solver: SharedMediumCommunityLP,
    baseline_growth: float,
    medium_mmol_l: Mapping[str, float],
    candidates: Iterable[str],
) -> list[dict[str, Any]]:
    """Rank supplied nutrients by the growth loss caused by removing each one."""

    available = set(solver.exchange_terms)
    rows = []
    for metabolite in sorted({canonical_metabolite_id(item) for item in candidates}):
        concentration = float(medium_mmol_l.get(metabolite, 0.0))
        if metabolite not in available or concentration <= 0.0:
            continue
        dropout_growth = solver.solve_common_growth({metabolite: 0.0})
        loss = max(0.0, float(baseline_growth - dropout_growth))
        rows.append(
            {
                "metabolite": metabolite,
                "group": candidate_group(metabolite),
                "initial_concentration_mmol_l": concentration,
                "baseline_growth_per_h": float(baseline_growth),
                "dropout_growth_per_h": float(dropout_growth),
                "absolute_loss_per_h": loss,
                "relative_loss": (
                    loss / baseline_growth if baseline_growth > 1e-12 else None
                ),
                "model_essential_at_tested_medium": bool(
                    baseline_growth > 1e-7 and dropout_growth <= 1e-7
                ),
            }
        )
    rows.sort(key=lambda row: row["absolute_loss_per_h"], reverse=True)
    return rows


def trace_community_limitation(
    models: Mapping[str, Any],
    medium_mmol_l: Mapping[str, float],
    biomass_g_l: Mapping[str, float],
    dt_h: float,
    oxygen_transfer_mmol_l_h: float = 1.0,
    target_gain_fraction: float = 0.10,
    minimum_target_growth_per_h: float = 0.005,
    run_single_addition_scan: bool = False,
) -> dict[str, Any]:
    """Return an interpretable limitation and external-rescue snapshot."""

    solver = SharedMediumCommunityLP(
        models,
        medium_mmol_l,
        biomass_g_l,
        dt=dt_h,
        oxygen_transfer_mmol_l_h=oxygen_transfer_mmol_l_h,
    )
    result = solver.solve()
    baseline = float(result.common_growth_per_h)
    target = max(
        float(minimum_target_growth_per_h),
        baseline * (1.0 + max(0.0, float(target_gain_fraction))),
    )
    allowed = candidate_ids()
    required = solver.diagnose_required_supply(
        minimum_growth=target,
        allowed_metabolites=allowed,
    )
    required_overrides = {
        item["metabolite"]: _base_supply_cap(solver, item["metabolite"])
        + float(item["additional_supply_mmol_l_h"])
        for item in required
    }
    combined_growth = (
        solver.solve_common_growth(required_overrides) if required_overrides else baseline
    )
    limiting = []
    for item in result.limiting_metabolites:
        row = dict(item)
        row["group"] = candidate_group(row["metabolite"])
        # A shadow price is a local opportunity value, not proof of an
        # auxotrophy or an experimentally essential nutrient.
        row["interpretation"] = "marginal_growth_opportunity"
        limiting.append(row)

    depleted = sorted(
        canonical_metabolite_id(key)
        for key, value in medium_mmol_l.items()
        if np.isfinite(float(value)) and float(value) <= 1e-8
    )
    candidates = allowed if run_single_addition_scan else {
        item["metabolite"] for item in required
    }
    single = rank_single_addition_rescues(
        solver, baseline, candidates, maximum_results=20
    )
    return {
        "common_growth_per_h": baseline,
        "species_growth_per_h": result.species_growth_per_h,
        "limiting_constraints": limiting,
        "depleted_pools": depleted,
        "target_growth_per_h": target,
        "required_practical_supply": required,
        "combined_rescue_growth_per_h": float(combined_growth),
        "combined_rescue_reaches_target": bool(combined_growth >= target - 1e-7),
        "single_addition_rescues": single,
        "cross_feeding": [edge.__dict__ for edge in result.cross_feeding],
        "diagnostic_scope": (
            "common-growth LP; additions are model hypotheses, not validated recipes"
        ),
    }


def classify_polymer_bottleneck(
    polymer_fluxes: Mapping[str, float],
    rubber_g_l: float,
    oxygen_mmol_l: float,
    glucose_mmol_l: float,
) -> dict[str, Any]:
    """Classify the extracellular cleavage limitation without overclaiming."""

    scale = float(polymer_fluxes.get("bulk_substrate_scale", 1.0))
    if rubber_g_l <= 1e-8:
        code = "rubber_depleted"
    elif scale < 0.999:
        code = "allocated_oxygen_limited"
    elif glucose_mmol_l > 0.05:
        code = "or16_glucose_repression_possible"
    else:
        code = "calibrated_enzyme_capacity_limited"
    return {
        "code": code,
        "bulk_substrate_scale": scale,
        "dissolved_oxygen_mmol_l": float(oxygen_mmol_l),
        "glucose_mmol_l": float(glucose_mmol_l),
        "note": (
            "Classification follows the implemented kinetic layer; enzyme-rate "
            "constants and oxygen allocation require wet-lab calibration."
        ),
    }
