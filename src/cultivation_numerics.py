"""Conservative inventory helpers for the audited cultivation reference.

These functions do not infer biological competition. Equal specific substrate
quotas among eligible consumers, and proportional oxygen-demand reduction, are
explicit allocation assumptions. Their parameters require experimental checks.
Concentrations/amounts are mmol/L, biomass is g/L, and time is hours.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .resolved_dfba import oxygen_interval


# These pools have separate physical or intracellular accounting interfaces.
_SEPARATE_POOLS = frozenset({
    "h_e", "h2o_e", "o2_e", "pha_c", "phb_c", "phv_c",
    "rubber_e", "rubber_bulk_e", "M_rubber_bulk_e",
})
# Retained provisional Monod constants (mmol/L), not fitted measurements.
_LOW_KM_POOLS = frozenset({"glc__D_e", "pi_e", "nh4_e"})


def _finite(value: float, label: str, *, nonnegative: bool = False,
            positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _sum_finite(values, label: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite numerical range") from error
    return _finite(result, label, nonnegative=True)


def compute_uptake_limits(
    concentrations: Mapping[str, float],
    biomass: Mapping[str, float],
    exchanges: Mapping[str, Mapping[str, str]],
    original_bounds: Mapping[str, Mapping[str, tuple[float, float]]],
    dt: float,
    max_uptake_rate: float,
) -> dict[str, dict[str, float]]:
    """Return nonnegative specific uptake limits in mmol/g/h, without rounding.

    ``exchanges`` must map canonical physical pools to exchange reaction IDs;
    exchange orientation/coefficient validation belongs to the caller. Each
    ordinary mapped pool is returned, including zero for absent medium, missing
    original uptake permission, or zero consumer biomass. Oxygen, water, proton,
    intracellular storage, and insoluble rubber pools are omitted.

    For a pool, only species with positive biomass AND a negative original
    exchange lower bound share its inventory. Each receives the same specific
    inventory quota C/(sum(eligible X)*dt), additionally limited by its original
    bound and a provisional Monod rate. Unused quotas are not reallocated; this
    is a conservative reference assumption rather than measured competition.
    """
    duration = _finite(dt, "dt", positive=True)
    maximum = _finite(max_uptake_rate, "max_uptake_rate", nonnegative=True)
    medium = {met: _finite(value, f"concentration[{met}]", nonnegative=True)
              for met, value in concentrations.items()}
    populations = {name: _finite(value, f"biomass[{name}]", nonnegative=True)
                   for name, value in biomass.items()}
    lower_bounds = {}
    for name, bounds in original_bounds.items():
        lower_bounds[name] = {}
        for reaction, pair in bounds.items():
            if len(pair) != 2:
                raise ValueError(f"original_bounds[{name}][{reaction}] needs two bounds")
            lower = _finite(pair[0], f"lower_bound[{name}][{reaction}]")
            upper = _finite(pair[1], f"upper_bound[{name}][{reaction}]")
            if lower > upper:
                raise ValueError(f"unordered original bounds for {name}/{reaction}")
            lower_bounds[name][reaction] = lower

    eligible: dict[str, list[str]] = {}
    result = {name: {} for name in exchanges}
    for name, mapping in exchanges.items():
        for met, reaction in mapping.items():
            if met in _SEPARATE_POOLS:
                continue
            result[name][met] = 0.0
            lower = lower_bounds.get(name, {}).get(reaction, 0.0)
            if populations.get(name, 0.0) > 0.0 and lower < 0.0:
                eligible.setdefault(met, []).append(name)

    for met, names in eligible.items():
        concentration = medium.get(met, 0.0)
        if concentration == 0.0:
            continue
        total_biomass = _sum_finite((populations[name] for name in names),
                                   f"eligible biomass[{met}]")
        km = 0.01 if met in _LOW_KM_POOLS else 0.1
        kinetic_limit = maximum * (concentration / (km + concentration))
        inventory_limit = concentration / total_biomass / duration
        for name in names:
            lower = lower_bounds[name][exchanges[name][met]]
            result[name][met] = min(kinetic_limit, inventory_limit, -lower)
    return result


def allocate_oxygen(
    c0: float,
    kla: float,
    dt: float,
    saturation: float,
    polymer_demand_rate: float,
    cellular_demand_rates: Mapping[str, float],
) -> dict:
    """Reduce kinetic oxygen demands by one common conservative factor.

    Demand rates are nonnegative volumetric rates (mmol/L/h), supplied by the
    caller from physical kinetics. They remain constant over this interval.
    The exact transfer/sink solution sets a total inventory ceiling. If demand
    exceeds it, all cellular and polymer demands receive the SAME factor;
    there is no fixed polymer share per numerical step. This proportional
    reduction is an explicit assumption, not empirical affinity competition.

    ``free`` and ``mean`` are the no-consumption endpoint/mean from
    :func:`oxygen_interval`; the caller must not confuse ``mean`` with actual
    oxygen under consumption. Final DO is ``free - sum(amounts)/dt*response``.
    """
    initial = _finite(c0, "c0", nonnegative=True)
    transfer_rate = _finite(kla, "kla", nonnegative=True)
    duration = _finite(dt, "dt", positive=True)
    saturated = _finite(saturation, "saturation", positive=True)
    polymer_rate = _finite(polymer_demand_rate, "polymer_demand_rate", nonnegative=True)
    cells = {name: _finite(rate, f"cellular_demand_rates[{name}]", nonnegative=True)
             for name, rate in cellular_demand_rates.items()}
    total_rate = _sum_finite([polymer_rate, *cells.values()], "total oxygen demand rate")
    requested = _finite(total_rate * duration, "oxygen demand amount", nonnegative=True)
    free, response, mean, budget = oxygen_interval(initial, transfer_rate, duration, saturated)
    for label, value in (("free", free), ("response", response),
                         ("mean", mean), ("budget", budget)):
        _finite(value, f"oxygen {label}", nonnegative=True)
    scale = min(1.0, budget / requested) if requested > 0.0 else 1.0
    return {
        "free": free, "response": response, "mean": mean, "budget": budget,
        "polymer_amount": polymer_rate * duration * scale,
        "cellular_amounts": {name: rate * duration * scale for name, rate in cells.items()},
        "scale": scale,
    }


def oxygen_mean_response(kla: float, dt: float) -> float:
    """Mean concentration response to a unit constant oxygen sink (hours).

    For a constant volumetric sink OUR, actual interval mean oxygen equals
    ``free_mean - OUR * oxygen_mean_response(kla, dt)``. The mathematical
    coefficient is ``(dt - (1-exp(-kla*dt))/kla)/(kla*dt)``. A small-argument
    series avoids subtraction cancellation, and the zero-transfer limit is
    exactly dt/2. This coefficient supplies no biological kinetics itself.
    """
    transfer = _finite(kla, "kla", nonnegative=True)
    duration = _finite(dt, "dt", positive=True)
    if transfer == 0.0:
        return duration / 2.0
    x = transfer * duration
    if x < 1e-3:
        # (x + expm1(-x))/x**2, through x**5.
        coefficient = 0.5 + x * (-1.0 / 6.0 + x * (1.0 / 24.0 + x * (
            -1.0 / 120.0 + x * (1.0 / 720.0 - x / 5040.0))))
        return duration * coefficient
    # This order remains finite even when transfer*duration overflows: the
    # large-x response then tends to 1/kLa, rather than an inf/inf quotient.
    return (1.0 + math.expm1(-x) / x) / transfer
