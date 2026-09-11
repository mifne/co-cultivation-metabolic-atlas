"""Physical 1 L jar-fermenter wrapper for the existing concentration dFBA.

The legacy simulator evolves concentrations and treats feed as a direct
concentration increment.  This module keeps that validated biology/LP path
unchanged, but performs the reactor bookkeeping outside it: stock flow,
dilution, working-volume change, pH titrant volume, and manual sampling.

All stock compositions are provisional virtual values until measured.  They
are labelled as such and must not be interpreted as a wet-lab recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from .utils import COEXISTENCE_DEFINED_FEED_MMOL_L_H


@dataclass(frozen=True)
class PumpStock:
    """One nutrient stock bottle.

    ``composition_mmol_l`` is numerically equal to mM.  Flow is specified in
    mL h-1 by the controller.
    """

    name: str
    composition_mmol_l: Mapping[str, float]
    max_flow_ml_h: float = 2.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pump stock name must not be empty")
        if not np.isfinite(self.max_flow_ml_h) or self.max_flow_ml_h <= 0.0:
            raise ValueError("max_flow_ml_h must be finite and positive")
        if not self.composition_mmol_l:
            raise ValueError(f"pump stock {self.name!r} has no components")
        for metabolite, concentration in self.composition_mmol_l.items():
            if not metabolite:
                raise ValueError("stock metabolite id must not be empty")
            if not np.isfinite(concentration) or concentration < 0.0:
                raise ValueError(
                    f"invalid stock concentration for {metabolite}: {concentration}"
                )


@dataclass(frozen=True)
class KLaCalibration:
    """Provisional, replaceable engineering map for virtual smoke tests."""

    reference_kla_h: float = 60.0
    reference_rpm: float = 400.0
    reference_vvm: float = 0.5
    reference_volume_l: float = 0.75
    rpm_exponent: float = 0.8
    vvm_exponent: float = 0.5
    volume_exponent: float = -0.2
    rubber_penalty_per_g_l: float = 0.002
    max_kla_h: float = 200.0

    def __post_init__(self) -> None:
        positive = (self.reference_kla_h, self.reference_rpm, self.reference_vvm,
                    self.reference_volume_l, self.max_kla_h)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError('kLa reference values and maximum must be positive and finite')
        if not all(np.isfinite(v) for v in (self.rpm_exponent, self.vvm_exponent,
                                           self.volume_exponent)):
            raise ValueError('kLa exponents must be finite')
        if not np.isfinite(self.rubber_penalty_per_g_l) or self.rubber_penalty_per_g_l < 0:
            raise ValueError('rubber penalty must be finite and nonnegative')

    def calculate(
        self,
        rpm: float,
        airflow_l_min: float,
        volume_l: float,
        rubber_g_l: float,
        multiplier: float = 1.0,
    ) -> float:
        values = (rpm, airflow_l_min, volume_l, rubber_g_l, multiplier)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("kLa inputs must be finite")
        if rpm < 0.0 or airflow_l_min < 0.0 or volume_l <= 0.0 or rubber_g_l < 0:
            raise ValueError("rpm/airflow must be nonnegative and volume positive")
        if multiplier < 0.0:
            raise ValueError("kLa multiplier must be nonnegative")
        if rpm == 0.0 or airflow_l_min == 0.0 or multiplier == 0.0:
            return 0.0
        vvm = airflow_l_min / volume_l
        kla = (
            self.reference_kla_h
            * (rpm / self.reference_rpm) ** self.rpm_exponent
            * (vvm / self.reference_vvm) ** self.vvm_exponent
            * (volume_l / self.reference_volume_l) ** self.volume_exponent
            / (1.0 + self.rubber_penalty_per_g_l * max(0.0, rubber_g_l))
            * multiplier
        )
        return float(np.clip(kla, 0.0, self.max_kla_h))


@dataclass(frozen=True)
class OneLJarConfig:
    initial_volume_l: float = 0.75
    vessel_capacity_l: float = 1.0
    minimum_operating_volume_l: float = 0.5
    base_stock_mol_l: float = 2.0
    acid_stock_mol_l: float = 1.0
    sample_events_h_ml: Sequence[tuple[float, float]] = field(default_factory=tuple)
    rpm_min: float = 0.0
    rpm_max: float = 800.0
    airflow_l_min_min: float = 0.0
    airflow_l_min_max: float = 2.0

    def __post_init__(self) -> None:
        numeric = (
            self.initial_volume_l,
            self.vessel_capacity_l,
            self.minimum_operating_volume_l,
            self.base_stock_mol_l,
            self.acid_stock_mol_l,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("volumes and titrant concentrations must be positive")
        if not (
            self.minimum_operating_volume_l
            <= self.initial_volume_l
            <= self.vessel_capacity_l
        ):
            raise ValueError("initial working volume is outside vessel limits")
        for lower, upper in ((self.rpm_min, self.rpm_max),
                             (self.airflow_l_min_min, self.airflow_l_min_max)):
            if not np.isfinite(lower) or not np.isfinite(upper) or not 0 <= lower <= upper:
                raise ValueError('rpm and airflow limits must be finite, ordered and nonnegative')
        previous_time = -np.inf
        for event_time, sample_ml in self.sample_events_h_ml:
            if not np.isfinite(event_time) or event_time < 0.0:
                raise ValueError("sample time must be finite and nonnegative")
            if event_time < previous_time:
                raise ValueError("sample events must be sorted by time")
            if not np.isfinite(sample_ml) or sample_ml <= 0.0:
                raise ValueError("sample volume must be finite and positive")
            previous_time = event_time


def default_virtual_pump_stocks(
    reference_common_flow_ml_h: float = 2.0,
) -> Dict[str, PumpStock]:
    """Return four explicit virtual stocks matching the current model inputs.

    Pump 4 reproduces the legacy Defined-10 concentration-rate vector at 1 L
    when operated at ``reference_common_flow_ml_h``.  Pumps 1--3 expose the
    current species-support aliases and remain uncalibrated virtual stocks.
    """

    if not np.isfinite(reference_common_flow_ml_h) or reference_common_flow_ml_h <= 0.0:
        raise ValueError("reference_common_flow_ml_h must be positive")
    reference_flow_l_h = reference_common_flow_ml_h / 1000.0
    common_stock = {
        metabolite: rate_mmol_l_h / reference_flow_l_h
        for metabolite, rate_mmol_l_h in COEXISTENCE_DEFINED_FEED_MMOL_L_H.items()
    }
    return {
        "p1_or16_virtual": PumpStock(
            "p1_or16_virtual", {"mlttr_e": 50.0}, max_flow_ml_h=2.0
        ),
        "p2_ns21_virtual": PumpStock(
            "p2_ns21_virtual", {"ptrc_e": 50.0}, max_flow_ml_h=2.0
        ),
        "p3_wcfs1_virtual": PumpStock(
            "p3_wcfs1_virtual", {"mnl_e": 100.0}, max_flow_ml_h=2.0
        ),
        "p4_defined10_virtual": PumpStock(
            "p4_defined10_virtual", common_stock, max_flow_ml_h=2.0
        ),
    }


class OneLJarProfile:
    """Amount-conserving physical wrapper around ``dFBASimulator``."""

    def __init__(
        self,
        simulator,
        pump_stocks: Mapping[str, PumpStock] | None = None,
        config: OneLJarConfig | None = None,
        kla_calibration: KLaCalibration | None = None,
    ) -> None:
        self.simulator = simulator
        self.config = config or OneLJarConfig()
        self.pump_stocks = dict(pump_stocks or default_virtual_pump_stocks())
        if len(self.pump_stocks) != 4:
            raise ValueError("one_l_jar requires exactly four nutrient pumps")
        if set(self.pump_stocks) != {stock.name for stock in self.pump_stocks.values()}:
            raise ValueError("pump mapping keys must match PumpStock.name")
        self.kla_calibration = kla_calibration or KLaCalibration()
        self.volume_l = float(self.config.initial_volume_l)
        self.simulator.volume = self.volume_l
        self.cumulative_nutrient_feed_ml = 0.0
        self.cumulative_base_ml = 0.0
        self.cumulative_acid_ml = 0.0
        self.cumulative_sample_ml = 0.0
        self.cumulative_sampled_rubber_g = 0.0
        self.cumulative_sampled_biomass_g: Dict[str, float] = {}
        self.cumulative_sampled_pha_mmol: Dict[str, float] = {}
        self.cumulative_feed_mmol: Dict[str, float] = {}
        self.command_violations = 0
        self.capacity_violations = 0
        self._next_sample_index = 0
        self.records: list[dict] = []
        self.sample_records: list[dict] = []
        self.failed = False

    def _set_buffer_ph(self) -> None:
        simulator = self.simulator
        if not all(
            hasattr(simulator, name)
            for name in ("buffer_total", "buffer_base", "buffer_acid", "pKa")
        ):
            return
        simulator.buffer_base = float(
            np.clip(simulator.buffer_base, 1e-9, simulator.buffer_total - 1e-9)
        )
        simulator.buffer_acid = simulator.buffer_total - simulator.buffer_base
        ph = simulator.pKa + np.log10(
            simulator.buffer_base / simulator.buffer_acid
        )
        simulator.state.metabolites["h_e"] = 10.0 ** (3.0 - ph)

    def _add_well_mixed_volume(
        self,
        added_volume_l: float,
        additions_mmol: Mapping[str, float] | None = None,
    ) -> None:
        if added_volume_l < 0.0 or not np.isfinite(added_volume_l):
            raise ValueError("added volume must be finite and nonnegative")
        additions = dict(additions_mmol or {})
        # Validate the complete operation before diluting anything.
        for metabolite, amount_mmol in additions.items():
            if not metabolite or not np.isfinite(amount_mmol) or amount_mmol < 0.0:
                raise ValueError(f"invalid addition for {metabolite}: {amount_mmol}")
        if added_volume_l == 0.0 and not additions:
            return
        old_volume = self.volume_l
        new_volume = old_volume + added_volume_l
        if new_volume > self.config.vessel_capacity_l + 1e-12:
            self.capacity_violations += 1
            raise ValueError(
                f"vessel capacity exceeded: {new_volume:.6f} L > "
                f"{self.config.vessel_capacity_l:.6f} L"
            )
        dilution = old_volume / new_volume
        state = self.simulator.state
        for metabolite in list(state.metabolites):
            state.metabolites[metabolite] = max(
                0.0, float(state.metabolites[metabolite]) * dilution
            )
        for metabolite, amount_mmol in additions.items():
            state.metabolites[metabolite] = (
                state.metabolites.get(metabolite, 0.0)
                + float(amount_mmol) / new_volume
            )
        for species in state.species.values():
            species.biomass *= dilution
            species.pha_accumulated *= dilution
            # Older checkpoints and lightweight test doubles expose only the
            # legacy total-PHA field.
            if hasattr(species, "phb_accumulated"):
                species.phb_accumulated *= dilution
            if hasattr(species, "phv_accumulated"):
                species.phv_accumulated *= dilution
        state.rubber_concentration *= dilution
        if hasattr(self.simulator, 'record_volume_addition'):
            self.simulator.record_volume_addition(old_volume, new_volume, additions)
        for name in ("buffer_total", "buffer_base", "buffer_acid"):
            if hasattr(self.simulator, name):
                setattr(self.simulator, name, getattr(self.simulator, name) * dilution)
        self.volume_l = new_volume
        self.simulator.volume = new_volume
        self._set_buffer_ph()

    def _apply_nutrient_feed(self, pump_flows_ml_h: Mapping[str, float]) -> None:
        unknown = set(pump_flows_ml_h) - set(self.pump_stocks)
        if unknown:
            self.command_violations += 1
            raise ValueError(f"unknown pump commands: {sorted(unknown)}")
        additions: Dict[str, float] = {}
        total_feed_ml = 0.0
        for pump_name, stock in self.pump_stocks.items():
            flow = float(pump_flows_ml_h.get(pump_name, 0.0))
            if not np.isfinite(flow) or flow < 0.0 or flow > stock.max_flow_ml_h:
                self.command_violations += 1
                raise ValueError(
                    f"pump {pump_name} flow {flow} is outside "
                    f"[0, {stock.max_flow_ml_h}] mL/h"
                )
            step_volume_l = flow * self.simulator.dt / 1000.0
            total_feed_ml += flow * self.simulator.dt
            for metabolite, concentration_mmol_l in stock.composition_mmol_l.items():
                amount = concentration_mmol_l * step_volume_l
                additions[metabolite] = additions.get(metabolite, 0.0) + amount
        self._add_well_mixed_volume(total_feed_ml / 1000.0, additions)
        for metabolite, amount in additions.items():
            self.cumulative_feed_mmol[metabolite] = (
                self.cumulative_feed_mmol.get(metabolite, 0.0) + amount)
        self.cumulative_nutrient_feed_ml += total_feed_ml

    def _apply_titrant_volume(self, previous_base_mM: float, previous_acid_mM: float) -> None:
        simulator = self.simulator
        base_delta_mM = max(
            0.0, float(simulator.cumulative_base_added_mmol_l) - previous_base_mM
        )
        acid_delta_mM = max(
            0.0, float(simulator.cumulative_acid_added_mmol_l) - previous_acid_mM
        )
        base_mmol = base_delta_mM * self.volume_l
        acid_mmol = acid_delta_mM * self.volume_l
        base_ml = base_mmol / self.config.base_stock_mol_l
        acid_ml = acid_mmol / self.config.acid_stock_mol_l
        additions: Dict[str, float] = {}
        if base_mmol:
            additions["na1_e"] = base_mmol
        if acid_mmol:
            additions["cl_e"] = acid_mmol
        self._add_well_mixed_volume((base_ml + acid_ml) / 1000.0, additions)
        self.cumulative_base_ml += base_ml
        self.cumulative_acid_ml += acid_ml

    def _apply_due_samples(self) -> None:
        events = self.config.sample_events_h_ml
        while self._next_sample_index < len(events):
            event_time, sample_ml = events[self._next_sample_index]
            if self.simulator.state.time + 1e-12 < event_time:
                break
            new_volume = self.volume_l - sample_ml / 1000.0
            if new_volume < self.config.minimum_operating_volume_l - 1e-12:
                self.command_violations += 1
                raise ValueError(
                    f"sample at {event_time} h lowers volume below minimum"
                )
            # A representative well-mixed sample removes amount and volume in
            # equal proportion, so concentrations do not change. Removed
            # material is tracked so endpoint amount balances can distinguish
            # degradation/production from analytical sampling.
            sample_l = sample_ml / 1000.0
            if hasattr(self.simulator, 'record_sample'):
                self.simulator.record_sample(sample_l)
            self.cumulative_sampled_rubber_g += (
                self.simulator.state.rubber_concentration * sample_l
            )
            for species_name, species in self.simulator.state.species.items():
                self.cumulative_sampled_biomass_g[species_name] = (
                    self.cumulative_sampled_biomass_g.get(species_name, 0.0)
                    + species.biomass * sample_l
                )
                self.cumulative_sampled_pha_mmol[species_name] = (
                    self.cumulative_sampled_pha_mmol.get(species_name, 0.0)
                    + species.pha_accumulated * sample_l
                )
            self.volume_l = new_volume
            self.simulator.volume = new_volume
            self.cumulative_sample_ml += sample_ml
            self.sample_records.append(dict(scheduled_time_h=float(event_time),
                actual_time_h=float(self.simulator.state.time), sample_ml=float(sample_ml)))
            self._next_sample_index += 1

    def _preflight(self, pump_flows_ml_h, rpm, airflow_l_min, rates, kla_multiplier):
        """Check known commands and the complete feed/sample volume schedule.

        Titrant demand is only known after solving metabolism; if it exhausts
        capacity, step fails and the wrapper is invalidated, never resumable.
        """
        if self.failed:
            raise RuntimeError('Jar step previously failed during integration; restore a checkpoint before reuse')
        dt = float(self.simulator.dt)
        internal_dt = float(getattr(self.simulator, 'max_internal_dt', dt))
        start = float(self.simulator.state.time)
        if not all(np.isfinite(v) for v in (dt, internal_dt, start)) or min(dt, internal_dt) <= 0 or start < 0:
            raise ValueError('simulation time and integration intervals must be finite and valid')
        unknown = set(pump_flows_ml_h) - set(self.pump_stocks)
        if unknown:
            raise ValueError(f'unknown pump commands: {sorted(unknown)}')
        total_flow = 0.0
        for name, stock in self.pump_stocks.items():
            stock.__post_init__()  # Mapping values may have been mutated externally.
            flow = float(pump_flows_ml_h.get(name, 0.0))
            if not np.isfinite(flow) or not 0 <= flow <= stock.max_flow_ml_h:
                raise ValueError(f'pump {name} flow {flow} is outside [0, {stock.max_flow_ml_h}] mL/h')
            total_flow += flow
        if not self.config.rpm_min <= rpm <= self.config.rpm_max:
            raise ValueError('rpm command is outside configured limits')
        if not self.config.airflow_l_min_min <= airflow_l_min <= self.config.airflow_l_min_max:
            raise ValueError('airflow command is outside configured limits')
        if any(not np.isfinite(v) or v < 0 for v in rates.values()):
            raise ValueError('rubber degradation rates must be finite and nonnegative')
        self.kla_calibration.calculate(rpm, airflow_l_min, self.volume_l,
            self.simulator.state.rubber_concentration, multiplier=kla_multiplier)
        end = start + dt
        if end <= start or dt <= 1e-12:
            raise ValueError('integration interval is below the jar event tolerance')
        volume = self.volume_l
        last = start
        for event_time, sample_ml in self.config.sample_events_h_ml[self._next_sample_index:]:
            if event_time > end + 1e-12:
                break
            if event_time < start - 1e-12:
                raise ValueError('Unprocessed sample event is in the past; cannot sample retrospectively')
            volume += total_flow * max(0., event_time-last) / 1000.
            if volume > self.config.vessel_capacity_l + 1e-12:
                raise ValueError('vessel capacity exceeded before scheduled sample')
            volume -= sample_ml / 1000.
            if volume < self.config.minimum_operating_volume_l - 1e-12:
                raise ValueError(f'sample at {event_time} h lowers volume below minimum')
            last = event_time
        volume += total_flow * (end-last) / 1000.
        if volume > self.config.vessel_capacity_l + 1e-12:
            raise ValueError('vessel capacity exceeded by requested nutrient feed')
        return dt, internal_dt, end

    def step(
        self,
        pump_flows_ml_h: Mapping[str, float],
        rpm: float,
        airflow_l_min: float,
        rubber_degradation_rates: Mapping[str, float] | None = None,
        kla_multiplier: float = 1.0,
    ):
        rates = dict(rubber_degradation_rates or {})
        try:
            control_dt, internal_dt, end = self._preflight(
                pump_flows_ml_h, rpm, airflow_l_min, rates, kla_multiplier)
        except ValueError:
            self.command_violations += 1
            raise
        try:
            self._apply_due_samples()  # Includes a sample explicitly scheduled at t=0.
            while self.simulator.state.time < end - 1e-12:
                now = float(self.simulator.state.time)
                boundary = min(end, now + internal_dt)
                if self._next_sample_index < len(self.config.sample_events_h_ml):
                    boundary = min(boundary, self.config.sample_events_h_ml[self._next_sample_index][0])
                self.simulator.dt = boundary - now
                # First-order pump/dilution integration, at the biological
                # internal interval rather than once per controller action.
                self._apply_nutrient_feed(pump_flows_ml_h)
                kla_h = self.kla_calibration.calculate(rpm, airflow_l_min, self.volume_l,
                    self.simulator.state.rubber_concentration, multiplier=kla_multiplier)
                previous_base = float(self.simulator.cumulative_base_added_mmol_l)
                previous_acid = float(self.simulator.cumulative_acid_added_mmol_l)
                state = self.simulator.step(rates, {}, dynamic_kla=kla_h)
                if abs(float(state.time) - boundary) > 1e-9:
                    raise RuntimeError('Simulator did not advance to the requested jar interval boundary')
                self._apply_titrant_volume(previous_base, previous_acid)
                self._apply_due_samples()
        except BaseException:
            self.failed = True
            raise
        finally:
            self.simulator.dt = control_dt
        record = {
            "time_h": float(state.time),
            "volume_l": self.volume_l,
            "kla_h": kla_h,
            "rpm": float(rpm),
            "airflow_l_min": float(airflow_l_min),
            "nutrient_feed_ml": self.cumulative_nutrient_feed_ml,
            "base_ml": self.cumulative_base_ml,
            "acid_ml": self.cumulative_acid_ml,
            "sample_ml": self.cumulative_sample_ml,
        }
        self.records.append(record)
        return state

    @property
    def expected_volume_l(self) -> float:
        return (
            self.config.initial_volume_l
            + (
                self.cumulative_nutrient_feed_ml
                + self.cumulative_base_ml
                + self.cumulative_acid_ml
                - self.cumulative_sample_ml
            )
            / 1000.0
        )

    @property
    def volume_balance_error_ml(self) -> float:
        return 1000.0 * (self.volume_l - self.expected_volume_l)

    def diagnostics(self) -> dict:
        return {
            "volume_l": self.volume_l,
            "expected_volume_l": self.expected_volume_l,
            "volume_balance_error_ml": self.volume_balance_error_ml,
            "cumulative_nutrient_feed_ml": self.cumulative_nutrient_feed_ml,
            "cumulative_base_ml": self.cumulative_base_ml,
            "cumulative_acid_ml": self.cumulative_acid_ml,
            "cumulative_sample_ml": self.cumulative_sample_ml,
            "cumulative_sampled_rubber_g": self.cumulative_sampled_rubber_g,
            "cumulative_sampled_biomass_g": dict(
                self.cumulative_sampled_biomass_g
            ),
            "cumulative_sampled_pha_mmol": dict(
                self.cumulative_sampled_pha_mmol
            ),
            "cumulative_feed_mmol": dict(self.cumulative_feed_mmol),
            "command_violations": self.command_violations,
            "capacity_violations": self.capacity_violations,
            "failed": self.failed,
            "sample_records": list(self.sample_records),
            "feed_integration": "first_order_at_internal_interval_and_sample_events",
        }
