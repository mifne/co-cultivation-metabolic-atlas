import copy
import math

import pytest

from src.cultivation_numerics import (allocate_oxygen, compute_uptake_limits,
                                    oxygen_mean_response)
from src.resolved_dfba import oxygen_interval


def test_uptake_inventory_is_unchanged_by_nonconsumers_and_zero_biomass():
    medium = {"glc__D_e": 0.001}
    populations = {"consumer": 0.1}
    exchanges = {"consumer": {"glc__D_e": "EX_g"}}
    bounds = {"consumer": {"EX_g": (-20.0, 1000.0)}}
    initial = compute_uptake_limits(medium, populations, exchanges, bounds, 1.0, 20.0)
    populations.update(inert=10.0, closed=10.0, absent=0.0)
    exchanges.update(inert={}, closed={"glc__D_e": "EX_closed"},
                     absent={"glc__D_e": "EX_absent"})
    bounds.update(closed={"EX_closed": (0.0, 1000.0)},
                  absent={"EX_absent": (-20.0, 1000.0)})
    actual = compute_uptake_limits(medium, populations, exchanges, bounds, 1.0, 20.0)
    assert actual["consumer"] == initial["consumer"] == {"glc__D_e": 0.01}
    assert actual["closed"]["glc__D_e"] == actual["absent"]["glc__D_e"] == 0.0


def test_tiny_inventory_is_not_rounded_outward_and_shared_total_is_conservative():
    medium = {"nh4_e": 1.5e-9}
    populations = {"a": 0.1, "b": 0.2}
    exchanges = {name: {"nh4_e": "EX_n"} for name in populations}
    bounds = {name: {"EX_n": (-20.0, 1000.0)} for name in populations}
    actual = compute_uptake_limits(medium, populations, exchanges, bounds, 0.025, 20.0)
    consumed = math.fsum(actual[name]["nh4_e"] * mass * 0.025
                         for name, mass in populations.items())
    assert actual["a"]["nh4_e"] == pytest.approx(2e-7, rel=1e-15, abs=0.0)
    assert consumed == pytest.approx(medium["nh4_e"], rel=1e-15, abs=0.0)
    assert consumed <= medium["nh4_e"] * (1.0 + 1e-15)


def test_uptake_preserves_original_permission_monod_and_separate_interfaces():
    exchanges = {"a": {pool: "EX_" + pool for pool in
                       ["glc__D_e", "lac__L_e", "nh4_e", "missing_e", "closed_e",
                        "o2_e", "h_e", "h2o_e", "pha_c", "phv_c", "rubber_e"]}}
    bounds = {"a": {reaction: (-20.0, 1000.0)
                    for reaction in exchanges["a"].values()}}
    bounds["a"]["EX_glc__D_e"] = (-0.02, 1000.0)
    del bounds["a"]["EX_closed_e"]
    medium = {"glc__D_e": 2.0, "lac__L_e": 0.1, "nh4_e": 0.0, "closed_e": 1.0}
    original = copy.deepcopy((medium, exchanges, bounds))
    actual = compute_uptake_limits(medium, {"a": 0.001}, exchanges, bounds, 0.01, 20.0)["a"]
    assert actual == {"glc__D_e": 0.02, "lac__L_e": 10.0, "nh4_e": 0.0,
                      "missing_e": 0.0, "closed_e": 0.0}
    assert (medium, exchanges, bounds) == original


@pytest.mark.parametrize("changes", [
    {"concentrations": {"n": -1.0}}, {"concentrations": {"n": float("nan")}},
    {"biomass": {"a": -0.1}}, {"biomass": {"a": float("inf")}},
    {"dt": 0.0}, {"dt": float("inf")}, {"max_uptake_rate": -1.0},
    {"original_bounds": {"a": {"EX_n": (1.0, 0.0)}}},
    {"original_bounds": {"a": {"EX_n": (-float("inf"), 1000.0)}}},
])
def test_uptake_rejects_invalid_inputs(changes):
    inputs = dict(concentrations={"n": 1.0}, biomass={"a": 0.1},
                  exchanges={"a": {"n": "EX_n"}},
                  original_bounds={"a": {"EX_n": (-20.0, 1000.0)}},
                  dt=0.025, max_uptake_rate=20.0)
    inputs.update(changes)
    with pytest.raises(ValueError):
        compute_uptake_limits(**inputs)


@pytest.mark.parametrize("kla", [0.0, 5.0, 50.0])
def test_oxygen_constrained_demands_share_one_factor_and_close_inventory(kla):
    result = allocate_oxygen(0.1, kla, 0.2, 0.25, 10.0, {"a": 20.0, "b": 30.0})
    amounts = [result["polymer_amount"], *result["cellular_amounts"].values()]
    assert result["scale"] < 1.0
    assert math.fsum(amounts) == pytest.approx(result["budget"], abs=1e-13)
    assert amounts[1] / amounts[0] == pytest.approx(2.0)
    assert amounts[2] / amounts[0] == pytest.approx(3.0)
    final = result["free"] - math.fsum(amounts) / 0.2 * result["response"]
    assert final == pytest.approx(0.0, abs=1e-13)
    if kla == 0.0:
        assert 0.1 - math.fsum(amounts) == pytest.approx(final, abs=1e-13)


@pytest.mark.parametrize("kla", [0.0, 5.0, 50.0])
def test_oxygen_fixed_unconstrained_rates_are_independent_of_partition(kla):
    rate = 0.03
    duration = 2.0
    expected = (0.2 - rate * duration if kla == 0.0 else
                0.25 - rate / kla + (0.2 - 0.25 + rate / kla) * math.exp(-kla * duration))
    for dt in [1.0, 0.2, 0.025]:
        concentration = 0.2
        total_polymer = 0.0
        total_cells = 0.0
        for _ in range(round(duration / dt)):
            result = allocate_oxygen(concentration, kla, dt, 0.25, 0.01, {"a": 0.02})
            assert result["scale"] == 1.0
            amount = result["polymer_amount"] + result["cellular_amounts"]["a"]
            concentration = result["free"] - amount / dt * result["response"]
            total_polymer += result["polymer_amount"]
            total_cells += result["cellular_amounts"]["a"]
        assert concentration == pytest.approx(expected, abs=1e-12)
        assert total_polymer == pytest.approx(0.02, abs=1e-13)
        assert total_cells == pytest.approx(0.04, abs=1e-13)


def test_zero_oxygen_demand_and_zero_supply_are_well_defined():
    empty = allocate_oxygen(0.0, 0.0, 1.0, 0.25, 0.0, {})
    assert empty["scale"] == 1.0
    assert empty["polymer_amount"] == empty["budget"] == 0.0
    starved = allocate_oxygen(0.0, 0.0, 1.0, 0.25, 1.0, {"a": 2.0})
    assert starved["scale"] == starved["polymer_amount"] == starved["cellular_amounts"]["a"] == 0.0


@pytest.mark.parametrize("changes", [
    {"c0": -0.1}, {"c0": float("nan")}, {"kla": -1.0}, {"dt": 0.0},
    {"saturation": 0.0}, {"saturation": float("inf")},
    {"polymer_demand_rate": -1.0}, {"cellular_demand_rates": {"a": float("inf")}},
    {"cellular_demand_rates": {"a": -1.0}},
])
def test_oxygen_rejects_invalid_inputs(changes):
    inputs = dict(c0=0.2, kla=5.0, dt=0.025, saturation=0.25,
                  polymer_demand_rate=0.1, cellular_demand_rates={"a": 0.2})
    inputs.update(changes)
    with pytest.raises(ValueError):
        allocate_oxygen(**inputs)


@pytest.mark.parametrize("kla", [0.0, 1e-9, 5.0, 50.0])
@pytest.mark.parametrize("dt", [0.025, 0.2, 1.0])
def test_mean_oxygen_response_matches_independent_time_integral(kla, dt):
    from scipy.integrate import quad
    initial, saturation, sink = 0.2, 0.25, 0.01
    free_mean = oxygen_interval(initial, kla, dt, saturation)[2]
    predicted_mean = free_mean - sink * oxygen_mean_response(kla, dt)

    def concentration(t):
        if kla == 0.0:
            return initial - sink * t
        relaxation = -math.expm1(-kla * t)
        return initial + (saturation - initial) * relaxation - sink * relaxation / kla

    integrated_mean = quad(concentration, 0.0, dt, epsabs=1e-13)[0] / dt
    assert predicted_mean == pytest.approx(integrated_mean, abs=1e-12)


def test_mean_oxygen_response_has_stable_zero_and_extreme_argument_limits():
    assert oxygen_mean_response(0.0, 0.025) == 0.0125
    assert oxygen_mean_response(1e-20, 0.025) == pytest.approx(0.0125, rel=1e-15)
    assert oxygen_mean_response(1e308, 1e308) == pytest.approx(1e-308, rel=1e-15, abs=0.0)
    result = oxygen_mean_response(1e-309, 1e308)
    assert math.isfinite(result) and 0.0 < result < 5e307


@pytest.mark.parametrize("kla,dt", [(-1.0, 1.0), (float("nan"), 1.0),
                                    (1.0, 0.0), (1.0, float("inf"))])
def test_mean_oxygen_response_rejects_invalid_inputs(kla, dt):
    with pytest.raises(ValueError):
        oxygen_mean_response(kla, dt)
