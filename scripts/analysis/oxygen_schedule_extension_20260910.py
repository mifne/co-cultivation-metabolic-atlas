"""Extend the oxygen-schedule study: finish dt convergence and widen the candidate set.

Three bounded additions to `results/oxygen_pulse_validation_20260910_repaired_v2`:

1. Convergence closure.  In that study the only unmet numerical criterion was the
   3HV mole fraction of `pulse_2_10` (0.0048 and 0.0015 against a 0.001 bound),
   so `pulse_2_10` is refined once more to internal dt 0.00625 h.  The equal-budget
   control `const_k6` had only been run at 0.025 h, so it is refined to 0.0125 h
   and now carries the same refinement as the pulses it is compared against.

2. Equal-budget candidate expansion.  `pulse4_2_10` and `pulse4_10_2` keep the
   same kLa levels (2 and 10 h^-1) and the same total time at each level (12 h
   each, integral 144 h), and change only the switching frequency (four 6 h
   blocks instead of two 12 h blocks) and phase.  This isolates temporal
   structure at fixed oxygen-transfer capacity, which is the question that
   decides whether an oxygen schedule is worth putting in a learned action space.

3. Intermediate-oxygen window.  With this fixed feed the 3-species arm was worse
   at kLa=2 (-51.6%) and kLa=50 (-1.6%), about neutral at kLa=10 (+0.8%) and
   better at kLa=6 (+3.7%).  `const_k4` and `const_k8` test whether that is a
   window or a single-point artifact.  These two are an oxygen-supply scan, NOT
   part of the equal-budget family.

Everything else is held fixed and identical to the referenced studies: horizon,
controller interval, inoculum, full background medium, feed profile, maintenance,
death rates, nitrogen policy, PHA remobilization, pH target and solver.  Pulse
boundaries are exact controller-step boundaries, so no schedule is rounded to the
internal step size.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/oxygen_schedule_extension_20260910"
PULSE_STUDY = ROOT / "results/oxygen_pulse_validation_20260910_repaired_v2"
REFERENCE_DESIGN = ROOT / "results/equal_budget_comparison_20260908/design.json"

OR = "Actinoplanes_sp_OR16_lcp"
NS = "Rhizobacter_gummiphilus_NS21"
PF = "Propionibacterium_freudenreichii_shermanii"

HOURS = 24.0
CONTROLLER_DT = 0.25
SATURATION = 0.25
# kLa integral of the equal-budget family: 6 h^-1 held for 24 h.
EQUAL_BUDGET_INTEGRAL_H = 144.0

SCHEDULES: dict[str, list[list[float]]] = {
    "const_k4": [[0.0, 24.0, 4.0]],
    "const_k6": [[0.0, 24.0, 6.0]],
    "const_k8": [[0.0, 24.0, 8.0]],
    "pulse_2_10": [[0.0, 12.0, 2.0], [12.0, 24.0, 10.0]],
    "pulse_10_2": [[0.0, 12.0, 10.0], [12.0, 24.0, 2.0]],
    "pulse4_2_10": [[0.0, 6.0, 2.0], [6.0, 12.0, 10.0], [12.0, 18.0, 2.0], [18.0, 24.0, 10.0]],
    "pulse4_10_2": [[0.0, 6.0, 10.0], [6.0, 12.0, 2.0], [12.0, 18.0, 10.0], [18.0, 24.0, 2.0]],
}
EQUAL_BUDGET_SCHEDULES = {"const_k6", "pulse_2_10", "pulse_10_2", "pulse4_2_10", "pulse4_10_2"}

# Coarse counterparts that already exist in the referenced pulse study.
EXTERNAL_COARSE = {
    "pulse_2_10_two_dt0.00625": "pulse_2_10_two_dt0.0125",
    "pulse_2_10_three_dt0.00625": "pulse_2_10_three_dt0.0125",
    "const_k6_two_dt0.0125": "const_k6_two_dt0.025",
    "const_k6_three_dt0.0125": "const_k6_three_dt0.025",
}
SHARED_SOURCES = [
    "main.py", "src/audited_dfba.py", "src/resolved_dfba.py", "src/physiology_dfba.py",
    "src/dfba_simulator.py", "src/cultivation_numerics.py", "src/b12_evidence.py",
    "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml",
    "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml",
    "models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml",
]
CONVERGENCE = dict(pha_relative=0.02, biomass_relative=0.02, hv_fraction_absolute=0.001)
LIMITATIONS = [
    "Bounded candidate set: three constant kLa levels and four pulse patterns, not a global optimum or a learned policy.",
    "kLa, oxygen saturation, maintenance, death, PHA remobilization and the Pf phenotype are uncalibrated.",
    "Equal integrated kLa equalizes oxygen-transfer capacity, not actual oxygen uptake or aeration energy cost.",
    "const_k4 and const_k8 change the oxygen budget and are a supply scan, not equal-budget candidates.",
    "Only the fixed early lactate / early NH4 feed profile is used; feed optimization is not repeated here.",
    "Single deterministic trajectory per case: no replicate spread and no statistical test.",
]


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes(paths: list[str]) -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def feed_rates(time_h: float) -> dict[str, float]:
    """Fixed early lactate / early NH4 profile, identical to the referenced studies."""
    return {
        "lac__L_e": 0.5 if time_h < 12.0 - 1e-10 else 0.0,
        "nh4_e": 0.1 if time_h < 4.0 - 1e-10 else 0.0,
    }


def schedule_at(time_h: float, intervals: list[list[float]], hours: float) -> tuple[float, float]:
    for start, end, kla in intervals:
        if time_h >= start - 1e-10 and time_h < end - 1e-10:
            return kla, end
    if abs(time_h - hours) <= 1e-9:
        return intervals[-1][2], hours
    raise ValueError(f"no oxygen schedule interval at t={time_h}")


def integrated_kla(intervals: list[list[float]]) -> float:
    return math.fsum(kla * (end - start) for start, end, kla in intervals)


def make_case(schedule_name: str, arm: str, internal_dt: float, group: str,
              intervals: list[list[float]] | None = None, hours: float = HOURS) -> dict:
    used = [list(row) for row in (intervals or SCHEDULES[schedule_name])]
    if abs(used[0][0]) > 1e-12 or abs(used[-1][1] - hours) > 1e-12:
        raise ValueError(f"schedule {schedule_name} does not span the horizon")
    for left, right in zip(used, used[1:]):
        if abs(left[1] - right[0]) > 1e-12:
            raise ValueError(f"schedule {schedule_name} has a gap or overlap")
    for start, end, _ in used:
        for boundary in (start, end):
            steps = boundary / CONTROLLER_DT
            if abs(steps - round(steps)) > 1e-9:
                raise ValueError(f"schedule boundary {boundary} is not a controller-step boundary")
    return {
        "id": f"{schedule_name}_{arm}_dt{internal_dt:g}",
        "schedule": schedule_name,
        "arm": arm,
        "internal_dt": internal_dt,
        "controller_dt": CONTROLLER_DT,
        "hours": hours,
        "feed_profile": "early_lactate_early_nh4",
        "group": group,
        "intervals": used,
        "integrated_kla_h": integrated_kla(used),
        "equal_budget": schedule_name in EQUAL_BUDGET_SCHEDULES,
    }


def _row(sim, models, capacity: float) -> dict:
    accounting = sim.accounting_audit
    elements = sim.element_accounting()
    species = sim.state.species
    phb = sum(state.phb_accumulated for state in species.values())
    phv = sum(state.phv_accumulated for state in species.values())
    live_pha = phb * sim.PHB_REPEAT_G_PER_MMOL + phv * sim.PHV_REPEAT_G_PER_MMOL
    dead_pha = sum(
        data["phb_mmol_l"] * sim.PHB_REPEAT_G_PER_MMOL
        + data["phv_mmol_l"] * sim.PHV_REPEAT_G_PER_MMOL
        for data in sim.dead_matter.values()
    )
    row = {
        "time_h": float(sim.state.time),
        "pha_live_g_l": float(live_pha),
        "pha_dead_g_l": float(dead_pha),
        "pha_total_g_l": float(live_pha + dead_pha),
        "phb_live_g_l": float(phb * sim.PHB_REPEAT_G_PER_MMOL),
        "phv_live_g_l": float(phv * sim.PHV_REPEAT_G_PER_MMOL),
        "hv_mol_fraction": float(phv / (phb + phv) if phb + phv > 1e-12 else 0.0),
        "rubber_removed_g_l": float(10.0 - sim.state.rubber_concentration),
        "nh4_mmol_l": float(sim.state.metabolites.get("nh4_e", 0.0)),
        "o2_mmol_l": float(sim.state.metabolites.get("o2_e", 0.0)),
        "lactate_mmol_l": float(sim.state.metabolites.get("lac__L_e", 0.0)),
        "propionate_mmol_l": float(sim.state.metabolites.get("ppa_e", 0.0)),
        "lactate_delivered": float(sim.cumulative_delivered_mmol_l.get("lac__L_e", 0.0)),
        "nh4_delivered": float(sim.cumulative_delivered_mmol_l.get("nh4_e", 0.0)),
        "oxygen_transferred": float(sim.oxygen_audit["transferred"]),
        "oxygen_cellular": float(sim.oxygen_audit["cellular_consumed"]),
        "oxygen_polymer": float(sim.oxygen_audit["polymer_consumed"]),
        "oxygen_capacity_integral": float(capacity),
        "C_residual": float(elements["known_medium_closure_residual"]["C"]),
        "N_residual": float(elements["known_medium_closure_residual"]["N"]),
        "oxygen_balance_error": float(sim.oxygen_audit["max_balance_error"]),
        "oxygen_kinetic_error": float(accounting["max_oxygen_kinetic_residual"]),
        "storage_excess": float(accounting["max_storage_excess_g_l"]),
        "lp_residual": float(accounting["max_lp_residual"]),
    }
    for name in models:
        short = "OR16" if name == OR else "NS21" if name == NS else "Pf"
        row[f"{short}_live_g_l"] = float(species[name].biomass)
        row[f"{short}_dead_g_l"] = float(sim.dead_matter[name]["biomass_g_l"])
    return row


def validate_row(row: dict, case: dict) -> None:
    if not all(math.isfinite(value) for value in row.values()):
        raise AssertionError("non-finite trajectory value")
    if any(value < -1e-8 for key, value in row.items()
           if key.endswith("_g_l") or key.endswith("_mmol_l")):
        raise AssertionError("negative concentration or mass")
    if abs(row["C_residual"]) > 1e-6 or abs(row["N_residual"]) > 1e-6:
        raise AssertionError("C/N closure failed")
    if row["oxygen_balance_error"] > 1e-8 or row["oxygen_kinetic_error"] > 1e-6:
        raise AssertionError("oxygen accounting failed")
    if row["storage_excess"] > 1e-7 or row["lp_residual"] > 1e-7:
        raise AssertionError("LP/storage audit failed")
    lactate_budget = 6.0 * min(1.0, case["hours"] / 12.0)
    nh4_budget = 0.4 * min(1.0, case["hours"] / 4.0)
    if row["lactate_delivered"] > lactate_budget + 1e-8 or row["nh4_delivered"] > nh4_budget + 1e-8:
        raise AssertionError("feed budget exceeded")
    if row["oxygen_transferred"] > row["oxygen_capacity_integral"] + 1e-8:
        raise AssertionError("oxygen transfer exceeded integrated kLa capacity")


def run_case(case: dict, out_dir: str) -> dict:
    from main import load_requested_models
    from src.physiology_dfba import PhysiologyDFBASimulator

    out = Path(out_dir)
    dest = out / case["id"]
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sim = None
    rows: list[dict] = []
    try:
        intervals = [list(row) for row in case["intervals"]]
        hours = float(case["hours"])
        models = load_requested_models(None, "pf-helper3")
        if case["arm"] != "three":
            models.pop(PF)
        biomass = ({OR: 0.5, NS: 0.1, PF: 0.03} if case["arm"] == "three"
                   else {OR: 0.525, NS: 0.105})
        config = dict(
            maintenance={name: dict(reaction="rxn00062_c0" if name == PF else "ATPM",
                                    rate_mmol_g_h=0.1) for name in models},
            basal_death_rates={name: 0.01 for name in models},
            starvation_death_rates={name: 0.1 for name in models},
            nitrogen_policy="capacity_ratio", remobilize_pha=True,
        )
        medium = dict(read_json(out / "design.json")["common_initial_medium"])
        sim = PhysiologyDFBASimulator(
            models, biomass, medium, initial_rubber=10.0, dt=CONTROLLER_DT,
            max_internal_dt=case["internal_dt"], ph_control_target=7.0, **config,
        )
        write_json(dest / "settings.json", dict(
            case=case, initial_biomass=biomass, initial_medium=medium, config=config,
            intervals=intervals, oxygen_saturation_mmol_l=sim.oxygen_saturation,
            integrated_kla_h=integrated_kla(intervals),
            gas_transfer_capacity_integral=integrated_kla(intervals) * sim.oxygen_saturation,
            pulse_boundary_policy="split simulator controller step exactly at schedule boundaries",
            biological_parameters_calibrated=False,
        ))

        capacity = 0.0
        rows.append(_row(sim, models, capacity))
        validate_row(rows[-1], case)
        with (dest / "trajectory.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerow(rows[0])
            handle.flush()
            while sim.state.time < hours - 1e-10:
                now = float(sim.state.time)
                kla, schedule_end = schedule_at(now, intervals, hours)
                # Keep the requested controller interval but cut it at a pulse
                # boundary, so the schedule does not depend on internal_dt.
                end = min(hours, now + CONTROLLER_DT, schedule_end)
                sim.dt = end - now
                if sim.dt <= 1e-12:
                    raise AssertionError(f"zero-length pulse interval at t={now}")
                sim.step({}, {}, dynamic_kla=kla, feed_rates_mmol_l_h=feed_rates(now))
                capacity += kla * sim.dt * sim.oxygen_saturation
                row = _row(sim, models, capacity)
                validate_row(row, case)
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                write_json(dest / "progress.json", dict(
                    id=case["id"], status="running", hours=sim.state.time,
                    seconds=time.monotonic() - started, pha_g_l=row["pha_live_g_l"],
                ))
        final = rows[-1]
        expected_lactate = 6.0 * min(1.0, hours / 12.0)
        expected_nh4 = 0.4 * min(1.0, hours / 4.0)
        if (abs(final["lactate_delivered"] - expected_lactate) > 1e-8
                or abs(final["nh4_delivered"] - expected_nh4) > 1e-8):
            raise AssertionError("fixed feed totals do not close")
        expected_capacity = integrated_kla(intervals) * sim.oxygen_saturation
        if abs(final["oxygen_capacity_integral"] - expected_capacity) > 1e-9:
            raise AssertionError("integrated kLa capacity does not match the schedule")
        audit = sim.accounting_audit
        if (audit["failed_steps"] != 0 or audit["max_dual_residual"] > 1e-7
                or audit["max_relative_duality_gap"] > 1e-7):
            raise AssertionError("solver audit failed")
        result = dict(
            case=case, final=final,
            peak_pha_g_l=max(row["pha_live_g_l"] for row in rows),
            seconds=time.monotonic() - started,
            diagnostics=sim.get_solver_diagnostics(),
        )
        write_json(dest / "result.json", result)
        status = dict(id=case["id"], status="complete", hours=sim.state.time,
                      seconds=result["seconds"], pha_g_l=final["pha_live_g_l"])
        write_json(dest / "progress.json", status)
        return status
    except Exception as error:
        failure = dict(
            id=case["id"], status="failed", error=repr(error),
            traceback=traceback.format_exc(), hours=sim.state.time if sim else 0.0,
            seconds=time.monotonic() - started,
            diagnostics=sim.get_solver_diagnostics() if sim else None,
        )
        write_json(dest / "failure.json", failure)
        write_json(dest / "progress.json", {k: v for k, v in failure.items()
                                            if k not in {"traceback", "diagnostics"}})
        return {k: v for k, v in failure.items() if k not in {"traceback", "diagnostics"}}


def _sources() -> list[str]:
    return [
        "scripts/analysis/oxygen_schedule_extension_20260910.py",
        "scripts/analysis/oxygen_pulse_validation_20260910.py",
        *SHARED_SOURCES,
        "results/oxygen_pulse_validation_20260910_repaired_v2/design.json",
        "results/equal_budget_comparison_20260908/design.json",
    ]


def build_design() -> dict:
    cases = [make_case("pulse_2_10", arm, 0.00625, "convergence_closure")
             for arm in ("two", "three")]
    cases.extend(make_case("const_k6", arm, 0.0125, "convergence_closure")
                 for arm in ("two", "three"))
    cases.extend(make_case(schedule, arm, dt, "equal_budget_frequency")
                 for schedule in ("pulse4_2_10", "pulse4_10_2")
                 for arm in ("two", "three")
                 for dt in (0.025, 0.0125))
    cases.extend(make_case(schedule, arm, dt, "oxygen_supply_window")
                 for schedule in ("const_k4", "const_k8")
                 for arm in ("two", "three")
                 for dt in (0.025, 0.0125))
    for case in cases:
        if case["equal_budget"] and abs(case["integrated_kla_h"] - EQUAL_BUDGET_INTEGRAL_H) > 1e-9:
            raise RuntimeError(f"{case['id']} is marked equal-budget but its kLa integral is not 144 h")
    reference = read_json(REFERENCE_DESIGN)
    pulse_design = read_json(PULSE_STUDY / "design.json")
    return dict(
        cases=cases, schedules=SCHEDULES, equal_budget_integral_h=EQUAL_BUDGET_INTEGRAL_H,
        source_sha256=source_hashes(_sources()),
        objective=("finish the internal-step convergence of the oxygen pulse study, add equal-budget "
                   "pulse-frequency candidates, and test whether the 3-species advantage at kLa=6 is "
                   "an intermediate-oxygen window"),
        common=dict(hours=HOURS, controller_dt=CONTROLLER_DT, initial_rubber_g_l=10.0,
                    initial_biomass_g_l=0.63, lactate_mmol_l=6.0, nh4_mmol_l=0.4,
                    oxygen_saturation_mmol_l=SATURATION),
        common_initial_medium=reference["common_initial_medium"],
        convergence_criteria=CONVERGENCE,
        external_coarse_reference=dict(
            study="results/oxygen_pulse_validation_20260910_repaired_v2",
            pairs=EXTERNAL_COARSE,
            shared_source_sha256={path: pulse_design["source_sha256"][path] for path in SHARED_SOURCES},
        ),
        limitations=LIMITATIONS,
    )


def _assert_shared_sources_match(design: dict) -> None:
    """The coarse counterparts live in another study; the simulator must be identical."""
    expected = design["external_coarse_reference"]["shared_source_sha256"]
    actual = source_hashes(list(expected))
    if actual != expected:
        differing = sorted(path for path in expected if expected[path] != actual[path])
        raise RuntimeError(f"simulator or model sources differ from the referenced pulse study: {differing}")


def _expected_cost(case: dict) -> float:
    return case["hours"] / case["internal_dt"] * (1.2 if case["arm"] == "three" else 1.0)


def prepare(out: Path, design: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    design_path = out / "design.json"
    if design_path.exists():
        stored = read_json(design_path)
        if stored != design:
            raise RuntimeError("existing design differs; refusing to mix experiments")
        return stored
    write_json(design_path, design)
    (out / "sources").mkdir(exist_ok=True)
    for path in design["source_sha256"]:
        shutil.copy2(ROOT / path, out / "sources" / path.replace("/", "__"))
    return design


def run_all(workers: int, rerun_interrupted: bool) -> None:
    design = build_design()
    _assert_shared_sources_match(design)
    prepare(OUT, design)
    if source_hashes(list(design["source_sha256"])) != design["source_sha256"]:
        raise RuntimeError("source changed before run")

    pending, statuses, archived = [], [], []
    for case in design["cases"]:
        case_dir = OUT / case["id"]
        progress_path = case_dir / "progress.json"
        if not progress_path.exists():
            if case_dir.exists() and any(case_dir.iterdir()):
                raise RuntimeError(f"case directory without progress requires audit: {case['id']}")
            pending.append(case)
            continue
        status = read_json(progress_path)
        if status.get("status") == "complete":
            statuses.append(status)
            continue
        if not rerun_interrupted:
            raise RuntimeError(f"retained incomplete case requires audit: {case['id']} "
                               f"(status={status.get('status')}); rerun with --rerun-interrupted")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        attic = OUT / f"interrupted_{stamp}"
        attic.mkdir(exist_ok=True)
        shutil.move(str(case_dir), str(attic / case["id"]))
        archived.append(dict(id=case["id"], previous_status=status, archived_to=str(attic.relative_to(OUT))))
        pending.append(case)
    if archived:
        write_json(OUT / "interrupted_archive.json", dict(archived=archived))

    pending.sort(key=_expected_cost, reverse=True)
    print(json.dumps(dict(pending=len(pending), already_complete=len(statuses),
                          archived=len(archived), workers=workers), ensure_ascii=False), flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(run_case, case, str(OUT)) for case in pending]
            for future in as_completed(futures):
                status = future.result()
                statuses.append(status)
                write_json(OUT / "execution_progress.json", dict(
                    results=statuses,
                    completed=sum(s.get("status") == "complete" for s in statuses),
                    total=len(design["cases"])))
                print(json.dumps(status, ensure_ascii=False), flush=True)
    if source_hashes(list(design["source_sha256"])) != design["source_sha256"]:
        raise RuntimeError("source changed during run")
    if any(s.get("status") != "complete" for s in statuses) or len(statuses) != len(design["cases"]):
        raise RuntimeError("some cases failed; retained artifacts were not overwritten")
    write_json(OUT / "execution_verification.json", dict(
        all_completed=True, count=len(statuses),
        source_sha256=source_hashes(list(design["source_sha256"]))))
    finalize(design)


def _convergence_check(case_id: str, coarse: dict, fine: dict) -> dict:
    pha = abs(coarse["pha_live_g_l"] - fine["pha_live_g_l"]) / max(0.001, abs(fine["pha_live_g_l"]))
    biomass = max(
        abs(coarse[key] - fine[key]) / max(1e-6, abs(fine[key]))
        for key in fine
        if key.endswith("_live_g_l") and key not in {"pha_live_g_l", "phb_live_g_l", "phv_live_g_l"}
    )
    hv = abs(coarse["hv_mol_fraction"] - fine["hv_mol_fraction"])
    return dict(id=case_id, pha_error=pha, biomass_error=biomass, hv_fraction_error=hv,
                passed=bool(pha <= CONVERGENCE["pha_relative"]
                            and biomass <= CONVERGENCE["biomass_relative"]
                            and hv <= CONVERGENCE["hv_fraction_absolute"]))


def finalize(design: dict) -> None:
    results = {case["id"]: read_json(OUT / case["id"] / "result.json") for case in design["cases"]}
    external = {case_id: read_json(PULSE_STUDY / case_id / "result.json")
                for case_id in set(EXTERNAL_COARSE.values())}

    checks = []
    for case in design["cases"]:
        fine = results[case["id"]]["final"]
        if case["id"] in EXTERNAL_COARSE:
            coarse = external[EXTERNAL_COARSE[case["id"]]]["final"]
            check = _convergence_check(case["id"], coarse, fine)
            check["coarse"] = EXTERNAL_COARSE[case["id"]]
            check["coarse_study"] = PULSE_STUDY.name
        elif case["internal_dt"] == 0.0125:
            coarse_id = case["id"].replace("dt0.0125", "dt0.025")
            if coarse_id not in results:
                continue
            check = _convergence_check(case["id"], results[coarse_id]["final"], fine)
            check["coarse"] = coarse_id
            check["coarse_study"] = OUT.name
        else:
            continue
        check["group"] = case["group"]
        checks.append(check)

    table = []
    for case in design["cases"]:
        finest = max((c for c in design["cases"]
                      if c["schedule"] == case["schedule"] and c["arm"] == case["arm"]),
                     key=lambda c: 1.0 / c["internal_dt"])
        if finest["id"] != case["id"]:
            continue
        final = results[case["id"]]["final"]
        converged = next((c["passed"] for c in checks if c["id"] == case["id"]), None)
        table.append(dict(schedule=case["schedule"], arm=case["arm"], group=case["group"],
                          internal_dt=case["internal_dt"], integrated_kla_h=case["integrated_kla_h"],
                          equal_budget=case["equal_budget"], converged=converged,
                          pha=final["pha_live_g_l"], pha_total=final["pha_total_g_l"],
                          phv=final["phv_live_g_l"], hv=final["hv_mol_fraction"],
                          rubber=final["rubber_removed_g_l"],
                          o2_transfer=final["oxygen_transferred"],
                          o2_capacity=final["oxygen_capacity_integral"],
                          seconds=results[case["id"]]["seconds"]))

    pairs = []
    for schedule in dict.fromkeys(row["schedule"] for row in table):
        two = next((r for r in table if r["schedule"] == schedule and r["arm"] == "two"), None)
        three = next((r for r in table if r["schedule"] == schedule and r["arm"] == "three"), None)
        if two is None or three is None:
            continue
        pairs.append(dict(
            schedule=schedule, group=two["group"], equal_budget=two["equal_budget"],
            integrated_kla_h=two["integrated_kla_h"],
            pha_two=two["pha"], pha_three=three["pha"],
            pha_delta_pct=100.0 * (three["pha"] / max(1e-12, two["pha"]) - 1.0),
            rubber_delta_pct=100.0 * (three["rubber"] / max(1e-12, two["rubber"]) - 1.0),
            hv_delta_mol_fraction=three["hv"] - two["hv"],
            both_converged=bool(two["converged"]) and bool(three["converged"])))

    summary = dict(
        cases=len(results), convergence=checks,
        convergence_passed=sum(c["passed"] for c in checks), convergence_total=len(checks),
        table=table, paired_species=pairs,
        equal_budget_family=[r for r in table if r["equal_budget"]],
        biological_validation=False, global_optimality_claim=False,
        teacher_collection_started=False, limitations=design["limitations"])
    write_json(OUT / "summary.json", summary)

    with (OUT / "endpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    lines = [
        "# 酸素スケジュール拡張：収束の完成と等予算パルス候補",
        "",
        "先行研究 `oxygen_pulse_validation_20260910_repaired_v2` と同一のシミュレータ・GEM・培地・接種量・",
        "供給プロファイル（早期乳酸6 mmol/L、早期NH4 0.4 mmol/L）・24時間・pH7で、酸素スケジュールと内部刻みのみを変えた。",
        "共有ソースのSHA256が先行研究と一致することを実行前に照合してから、粗い刻みの対応ケースを参照した。",
        "",
        f"数値収束（PHA・菌体量2%、3HVモル分率0.1ポイント）は{summary['convergence_passed']}/{summary['convergence_total']}件。",
        "",
        "## 収束判定",
        "",
        "|ケース|群|粗い対応|PHA誤差|菌体誤差|3HV分率誤差|判定|",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for check in checks:
        lines.append(f"|{check['id']}|{check['group']}|{check['coarse']}|{check['pha_error']:.5f}|"
                     f"{check['biomass_error']:.5f}|{check['hv_fraction_error']:.5f}|"
                     f"{'合格' if check['passed'] else '未達'}|")
    lines.extend([
        "",
        "## 最細刻みでの終点",
        "",
        "|schedule|構成|群|kLa積分 (h)|内部刻み (h)|PHA live (g/L)|3HV mol%|ゴム除去 (g/L)|O2移動 (mmol/L)|積分容量 (mmol/L)|dt収束|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in table:
        flag = "―" if row["converged"] is None else ("合格" if row["converged"] else "未達")
        lines.append(f"|{row['schedule']}|{row['arm']}|{row['group']}|{row['integrated_kla_h']:.1f}|"
                     f"{row['internal_dt']:.5f}|{row['pha']:.6f}|{100*row['hv']:.3f}|"
                     f"{row['rubber']:.6f}|{row['o2_transfer']:.6f}|{row['o2_capacity']:.6f}|{flag}|")
    lines.extend([
        "",
        "## 3種−2種の差",
        "",
        "|schedule|等予算|PHA差 (%)|ゴム除去差 (%)|3HV分率差 (ポイント)|両者dt収束|",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in pairs:
        lines.append(f"|{row['schedule']}|{'はい' if row['equal_budget'] else 'いいえ'}|"
                     f"{row['pha_delta_pct']:+.3f}|{row['rubber_delta_pct']:+.3f}|"
                     f"{100*row['hv_delta_mol_fraction']:+.3f}|"
                     f"{'はい' if row['both_converged'] else 'いいえ'}|")
    lines.extend([
        "",
        "等予算群（const_k6、pulse_2_10、pulse_10_2、pulse4_2_10、pulse4_10_2）はkLa積分144 hで一致するが、",
        "実際の酸素移動量はDO履歴とOURで変わるため、差を曝気量の差とは解釈しない。",
        "const_k4とconst_k8は酸素予算そのものを変える供給スキャンであり、等予算比較には含めない。",
        "dt収束を満たさない指標は優位性の根拠に使わない。",
        "kLa、酸素飽和、維持代謝・死滅、PHA再利用、Pf表現型は未校正であり、生物学的有意差は主張しない。",
        "各条件は決定論的な1軌道であり、反復のばらつきも統計検定もない。教師大量収集とRL学習は開始していない。",
    ])
    (OUT / "REPORT_JA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifacts = ["design.json", "execution_verification.json", "summary.json",
                 "endpoints.csv", "REPORT_JA.md"]
    write_json(OUT / "completion.json", dict(
        completed=len(results), all_completed=True,
        convergence_passed=summary["convergence_passed"],
        convergence_total=summary["convergence_total"],
        source_hashes_match=True, resource_and_lp_audit=True, biological_validation=False,
        artifact_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                         for name in artifacts}))
    print(json.dumps(dict(completed=len(results),
                          convergence_passed=summary["convergence_passed"],
                          convergence_total=summary["convergence_total"]), ensure_ascii=False))


def smoke() -> None:
    """One short case that exercises pulse-boundary splitting and every audit."""
    out = OUT.parent / (OUT.name + "_smoke")
    design = build_design()
    hours = 1.0
    case = make_case("pulse4_2_10", "three", 0.025, "smoke",
                     intervals=[[0.0, 0.5, 2.0], [0.5, 1.0, 10.0]], hours=hours)
    design = dict(design, cases=[case], smoke=True, smoke_hours=hours)
    prepare(out, design)
    status = run_case(case, str(out))
    print(json.dumps(status, ensure_ascii=False))
    if status.get("status") != "complete":
        raise SystemExit(1)
    final = read_json(out / case["id"] / "result.json")["final"]
    expected = integrated_kla(case["intervals"]) * SATURATION
    assert abs(final["time_h"] - hours) < 1e-9, final["time_h"]
    assert abs(final["oxygen_capacity_integral"] - expected) < 1e-9, final["oxygen_capacity_integral"]
    print(json.dumps(dict(smoke="ok", hours=final["time_h"],
                          capacity=final["oxygen_capacity_integral"],
                          pha=final["pha_live_g_l"]), ensure_ascii=False))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--rerun-interrupted", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    design = build_design()
    if args.prepare_only:
        _assert_shared_sources_match(design)
        prepare(OUT, design)
        print(json.dumps(dict(cases=len(design["cases"]),
                              groups=sorted({c["group"] for c in design["cases"]}),
                              workers=args.workers), ensure_ascii=False))
        return
    if args.finalize_only:
        finalize(read_json(OUT / "design.json"))
        return
    run_all(args.workers, args.rerun_interrupted)


if __name__ == "__main__":
    main()
