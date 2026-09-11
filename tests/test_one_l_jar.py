from __future__ import annotations

from types import SimpleNamespace
import copy

import numpy as np
import pytest

from src.one_l_jar import (
    KLaCalibration,
    OneLJarConfig,
    OneLJarProfile,
    PumpStock,
    default_virtual_pump_stocks,
)


class FakeSimulator:
    def __init__(self, base_delta_mM: float = 0.0) -> None:
        self.dt = 1.0
        self.volume = 0.8
        self.state = SimpleNamespace(
            time=0.0,
            metabolites={"x_e": 10.0, "h_e": 10.0 ** (3.0 - 6.5)},
            rubber_concentration=100.0,
            species={
                "a": SimpleNamespace(biomass=1.0, pha_accumulated=2.0),
            },
        )
        self.buffer_total = 100.0
        self.buffer_base = 16.3
        self.buffer_acid = 83.7
        self.pKa = 7.21
        self.ph_control_target = 6.5
        self.cumulative_base_added_mmol_l = 0.0
        self.cumulative_acid_added_mmol_l = 0.0
        self.base_delta_mM = base_delta_mM

    def step(self, *_args, **_kwargs):
        self.cumulative_base_added_mmol_l += self.base_delta_mM
        self.state.time += self.dt
        return self.state


def four_stocks(first: PumpStock) -> dict[str, PumpStock]:
    return {
        first.name: first,
        "p2": PumpStock("p2", {"y_e": 1.0}, 10.0),
        "p3": PumpStock("p3", {"z_e": 1.0}, 10.0),
        "p4": PumpStock("p4", {"w_e": 1.0}, 10.0),
    }


def test_default_virtual_profile_has_four_explicit_stocks() -> None:
    stocks = default_virtual_pump_stocks()
    assert len(stocks) == 4
    assert stocks["p4_defined10_virtual"].composition_mmol_l["glc__D_e"] == 250.0


def test_feed_dilution_sampling_and_volume_close_exactly() -> None:
    simulator = FakeSimulator()
    stock = PumpStock("p1", {"x_e": 100.0}, max_flow_ml_h=10.0)
    jar = OneLJarProfile(
        simulator,
        four_stocks(stock),
        OneLJarConfig(
            initial_volume_l=0.8,
            vessel_capacity_l=1.0,
            minimum_operating_volume_l=0.5,
            sample_events_h_ml=((1.0, 5.0),),
        ),
    )
    jar.step({"p1": 10.0}, rpm=400.0, airflow_l_min=0.4)
    assert jar.volume_l == pytest.approx(0.805)
    assert simulator.state.metabolites["x_e"] == pytest.approx(9.0 / 0.81)
    assert simulator.state.species["a"].biomass == pytest.approx(0.8 / 0.81)
    assert simulator.state.species["a"].pha_accumulated == pytest.approx(1.6 / 0.81)
    assert jar.cumulative_sampled_rubber_g == pytest.approx(
        simulator.state.rubber_concentration * 0.005
    )
    assert jar.volume_balance_error_ml == pytest.approx(0.0, abs=1e-9)


def test_titrant_amount_adds_volume_and_counterion() -> None:
    simulator = FakeSimulator(base_delta_mM=1.0)
    jar = OneLJarProfile(
        simulator,
        four_stocks(PumpStock("p1", {"x_e": 1.0}, 10.0)),
        OneLJarConfig(initial_volume_l=0.8, base_stock_mol_l=2.0),
    )
    jar.step({}, rpm=400.0, airflow_l_min=0.4)
    assert jar.cumulative_base_ml == pytest.approx(0.4)
    assert jar.volume_l == pytest.approx(0.8004)
    assert simulator.state.metabolites["na1_e"] == pytest.approx(0.8 / 0.8004)
    assert jar.volume_balance_error_ml == pytest.approx(0.0, abs=1e-9)


def test_kla_is_bounded_and_increases_with_rpm() -> None:
    calibration = KLaCalibration()
    low = calibration.calculate(200.0, 0.4, 0.75, 100.0)
    high = calibration.calculate(600.0, 0.4, 0.75, 100.0)
    assert 0.0 < low < high <= calibration.max_kla_h


def test_invalid_pump_command_is_rejected() -> None:
    simulator = FakeSimulator()
    jar = OneLJarProfile(
        simulator,
        four_stocks(PumpStock("p1", {"x_e": 1.0}, 1.0)),
        OneLJarConfig(initial_volume_l=0.8),
    )
    with pytest.raises(ValueError, match="outside"):
        jar.step({"p1": 1.1}, rpm=400.0, airflow_l_min=0.4)
    assert jar.command_violations == 1


@pytest.mark.parametrize('error', ['later_pump', 'capacity', 'kla', 'addition', 'sample'])
def test_invalid_operation_preserves_state_and_feed_counters(error):
    simulator = FakeSimulator()
    config = OneLJarConfig(initial_volume_l=.8, vessel_capacity_l=.805,
        sample_events_h_ml=((.5, 400.),) if error == 'sample' else ())
    jar = OneLJarProfile(simulator, four_stocks(PumpStock('p1', {'x_e': 100.}, 10.)), config)
    initial = copy.deepcopy(simulator.state)
    with pytest.raises(ValueError):
        if error == 'addition':
            jar._add_well_mixed_volume(.001, {'x_e': float('nan')})
        else:
            jar.step({'p1': 10. if error == 'capacity' else 1.,
                      'p2': 11. if error == 'later_pump' else 0.},
                rpm=400., airflow_l_min=.4,
                kla_multiplier=float('nan') if error == 'kla' else 1.)
    assert simulator.state == initial
    assert jar.volume_l == .8
    assert jar.cumulative_feed_mmol == {}
    assert jar.cumulative_nutrient_feed_ml == 0.


def test_internal_pump_intervals_and_sample_times_preserve_amounts():
    class RecordingSimulator(FakeSimulator):
        def __init__(self):
            super().__init__()
            self.max_internal_dt = .2
            self.intervals = []

        def step(self, *args, **kwargs):
            self.intervals.append((self.state.time, self.dt))
            return super().step(*args, **kwargs)

    simulator = RecordingSimulator()
    jar = OneLJarProfile(simulator, four_stocks(PumpStock('p1', {'x_e': 100.}, 10.)),
        OneLJarConfig(initial_volume_l=.8, sample_events_h_ml=((0., 5.), (.35, 10.), (1., 5.))))
    jar.step({'p1': 10.}, rpm=400., airflow_l_min=.4)
    assert simulator.dt == 1.
    assert max(dt for _, dt in simulator.intervals) <= .2 + 1e-12
    assert [(r['scheduled_time_h'], r['actual_time_h']) for r in jar.sample_records] == [
        (0., 0.), (.35, .35), (1., 1.)]
    assert jar.cumulative_feed_mmol['x_e'] == pytest.approx(1.)
    assert jar.volume_l == pytest.approx(.79)
    assert jar.volume_balance_error_ml == pytest.approx(0., abs=1e-9)
    assert (simulator.state.rubber_concentration * jar.volume_l
            + jar.cumulative_sampled_rubber_g) == pytest.approx(80.)
    assert (simulator.state.species['a'].biomass * jar.volume_l
            + jar.cumulative_sampled_biomass_g['a']) == pytest.approx(.8)


def test_sample_can_make_room_at_interior_boundary():
    simulator = FakeSimulator()
    jar = OneLJarProfile(simulator, four_stocks(PumpStock('p1', {'x_e': 100.}, 10.)),
        OneLJarConfig(initial_volume_l=.8, vessel_capacity_l=.806,
                      sample_events_h_ml=((.5, 10.),)))
    jar.step({'p1': 10.}, rpm=400., airflow_l_min=.4)
    assert jar.volume_l == pytest.approx(.8)
    assert jar.sample_records[0]['actual_time_h'] == .5


def test_unpredicted_titrant_capacity_failure_cannot_be_resumed():
    simulator = FakeSimulator(base_delta_mM=100.)
    jar = OneLJarProfile(simulator, four_stocks(PumpStock('p1', {'x_e': 100.}, 10.)),
        OneLJarConfig(initial_volume_l=.8, vessel_capacity_l=.81))
    with pytest.raises(ValueError, match='capacity'):
        jar.step({}, rpm=400., airflow_l_min=.4)
    assert jar.failed and simulator.dt == 1.
    with pytest.raises(RuntimeError, match='checkpoint'):
        jar.step({}, rpm=400., airflow_l_min=.4)


@pytest.mark.parametrize('factory', [lambda: KLaCalibration(reference_rpm=0.),
    lambda: KLaCalibration(volume_exponent=float('nan')),
    lambda: OneLJarConfig(rpm_min=500., rpm_max=400.),
    lambda: default_virtual_pump_stocks(float('nan'))])
def test_invalid_engineering_configuration_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()
