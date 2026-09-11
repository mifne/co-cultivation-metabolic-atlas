"""Validate piecewise-constant oxygen pulses without time-step alignment artifacts.

This is a bounded, reproducible follow-up to the direct-propionate comparison.
The feed schedule, inoculum, medium, solver and total horizon are fixed.  Only
the oxygen-transfer schedule, species set and internal integration step vary.
Pulse boundaries are explicit simulator-step boundaries, so a pulse is never
rounded to whichever internal time step happened to be selected.
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
OUT = ROOT / "results/oxygen_pulse_validation_20260910_repaired_v2"
REFERENCE_DESIGN = ROOT / "results/equal_budget_comparison_20260908/design.json"

OR = "Actinoplanes_sp_OR16_lcp"
NS = "Rhizobacter_gummiphilus_NS21"
PF = "Propionibacterium_freudenreichii_shermanii"

HOURS = 24.0
CONTROLLER_DT = 0.25
SATURATION = 0.25

# The two half-day pulses have the same integral kLa as constant kLa=6 h^-1.
# Both orderings are tested because timing can matter even at equal transfer
# capacity.  Constant kLa=2 and 10 are the previously observed low/high
# reference regimes; constant kLa=6 is the equal-budget control.
SCHEDULES = {
    "const_k2": [[0.0, HOURS, 2.0]],
    "const_k6": [[0.0, HOURS, 6.0]],
    "const_k10": [[0.0, HOURS, 10.0]],
    "pulse_2_10": [[0.0, 12.0, 2.0], [12.0, HOURS, 10.0]],
    "pulse_10_2": [[0.0, 12.0, 10.0], [12.0, HOURS, 2.0]],
}


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def source_hashes(paths: list[str]) -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def feed_rates(time_h: float) -> dict[str, float]:
    """Fixed early lactate/NH4 profile used for every oxygen comparison."""
    return {
        "lac__L_e": 0.5 if time_h < 12.0 - 1e-10 else 0.0,
        "nh4_e": 0.1 if time_h < 4.0 - 1e-10 else 0.0,
    }


def schedule_at(time_h: float, schedule: tuple[tuple[float, float, float], ...]) -> tuple[float, float]:
    for start, end, kla in schedule:
        if time_h >= start - 1e-10 and time_h < end - 1e-10:
            return kla, end
    if abs(time_h - HOURS) <= 1e-9:
        return schedule[-1][2], HOURS
    raise ValueError(f"no oxygen schedule interval at t={time_h}")


def make_case(schedule_name: str, arm: str, internal_dt: float) -> dict:
    return {
        "id": f"{schedule_name}_{arm}_dt{internal_dt:g}",
        "schedule": schedule_name,
        "arm": arm,
        "internal_dt": internal_dt,
        "controller_dt": CONTROLLER_DT,
        "hours": HOURS,
        "feed_profile": "early_lactate_early_nh4",
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
        "hv_mol_fraction": float(phv / (phb + phv) if phb + phv > 1e-12 else 0.0),
        "rubber_removed_g_l": float(10.0 - sim.state.rubber_concentration),
        "nh4_mmol_l": float(sim.state.metabolites.get("nh4_e", 0.0)),
        "o2_mmol_l": float(sim.state.metabolites.get("o2_e", 0.0)),
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
    if row["lactate_delivered"] > 6.0 + 1e-8 or row["nh4_delivered"] > 0.4 + 1e-8:
        raise AssertionError("feed budget exceeded")
    if row["oxygen_transferred"] > row["oxygen_capacity_integral"] + 1e-8:
        raise AssertionError("oxygen transfer exceeded integrated kLa capacity")


def run_case(case: dict) -> dict:
    from main import load_requested_models
    from src.physiology_dfba import PhysiologyDFBASimulator

    dest = OUT / case["id"]
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sim = None
    rows: list[dict] = []
    try:
        schedule = SCHEDULES[case["schedule"]]
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
        design = read_json(OUT / "design.json")
        # Reuse the exact full background medium from the audited equal-budget
        # comparison, including vitamins, salts, trace metals and initial O2.
        medium = dict(design["common_initial_medium"])
        sim = PhysiologyDFBASimulator(
            models, biomass, medium, initial_rubber=10.0, dt=CONTROLLER_DT,
            max_internal_dt=case["internal_dt"], ph_control_target=7.0, **config,
        )
        write_json(dest / "settings.json", dict(
            case=case, initial_biomass=biomass, initial_medium=medium,
            config=config, schedules=SCHEDULES, oxygen_saturation_mmol_l=sim.oxygen_saturation,
            integrated_kla_h=math.fsum(kla * (end - start) for start, end, kla in schedule),
            gas_transfer_capacity_integral=math.fsum(kla * (end - start) * sim.oxygen_saturation
                                                      for start, end, kla in schedule),
            pulse_boundary_policy="split simulator controller step exactly at schedule boundaries",
            biological_parameters_calibrated=False,
        ))

        capacity = 0.0
        current_schedule_index = 0
        rows.append(_row(sim, models, capacity))
        validate_row(rows[-1], case)
        with (dest / "trajectory.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerow(rows[0])
            handle.flush()
            while sim.state.time < HOURS - 1e-10:
                now = float(sim.state.time)
                kla, schedule_end = schedule_at(now, schedule)
                # Keep the requested controller interval, but split at a pulse
                # boundary.  This makes the schedule independent of internal_dt.
                end = min(HOURS, now + CONTROLLER_DT, schedule_end)
                sim.dt = end - now
                if sim.dt <= 1e-12:
                    raise AssertionError(f"zero-length pulse interval at t={now}")
                rate = feed_rates(now)
                sim.step({}, {}, dynamic_kla=kla, feed_rates_mmol_l_h=rate)
                capacity += kla * sim.dt * sim.oxygen_saturation
                row = _row(sim, models, capacity)
                validate_row(row, case)
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                write_json(dest / "progress.json", dict(
                    id=case["id"], status="running", hours=sim.state.time,
                    seconds=time.monotonic() - started, pha_g_l=row["pha_live_g_l"],
                    schedule_index=current_schedule_index,
                ))
                if sim.state.time >= schedule_end - 1e-10:
                    current_schedule_index += 1
        final = rows[-1]
        if abs(final["lactate_delivered"] - 6.0) > 1e-8 or abs(final["nh4_delivered"] - 0.4) > 1e-8:
            raise AssertionError("fixed feed totals do not close")
        audit = sim.accounting_audit
        if audit["failed_steps"] != 0 or audit["max_dual_residual"] > 1e-7 or audit["max_relative_duality_gap"] > 1e-7:
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


def _default_sources() -> list[str]:
    return [
        "scripts/analysis/oxygen_pulse_validation_20260910.py",
        "main.py", "src/audited_dfba.py", "src/resolved_dfba.py",
        "src/physiology_dfba.py", "src/dfba_simulator.py",
        "src/cultivation_numerics.py", "src/b12_evidence.py",
        "results/equal_budget_comparison_20260908/design.json",
        "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml",
        "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml",
        "models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml",
    ]


def build_design() -> dict:
    paths = _default_sources()
    # kLa=2 and kLa=10 are already available from the full-background-medium
    # comparison. Recompute the new equal-budget kLa=6 control and the two
    # pulse schedules here; only pulse cases need a second internal step size.
    cases = [make_case("const_k6", arm, 0.025) for arm in ("two", "three")]
    cases.extend(make_case(schedule, arm, dt)
                 for schedule in ("pulse_2_10", "pulse_10_2")
                 for arm in ("two", "three")
                 for dt in (0.025, 0.0125))
    reference = json.loads(REFERENCE_DESIGN.read_text(encoding="utf-8"))
    return dict(
        cases=cases, schedules=SCHEDULES, source_sha256=source_hashes(paths),
        objective="same early lactate/NH4 feed and same integrated kLa budget; compare pulse timing and 2/3 species",
        common=dict(hours=HOURS, controller_dt=CONTROLLER_DT, initial_rubber_g_l=10.0,
                    initial_biomass_g_l=0.63, lactate_mmol_l=6.0, nh4_mmol_l=0.4,
                    oxygen_saturation_mmol_l=SATURATION),
        common_initial_medium=reference["common_initial_medium"],
        convergence_criteria=dict(pha_relative=0.02, biomass_relative=0.02,
                                  hv_fraction_absolute=0.001),
        limitations=[
            "Pulse schedule is a bounded candidate set, not a global optimum or RL policy.",
            "kLa, oxygen saturation, maintenance, death, PHA remobilization and Pf phenotype are uncalibrated.",
            "Equal integrated kLa is an oxygen-transfer capacity comparison, not equal actual OUR or energy cost.",
            "Only the fixed early lactate/early NH4 feed profile is used; feed optimization is deferred.",
        ],
    )


def run_all(workers: int) -> None:
    design = build_design()
    OUT.mkdir(parents=True, exist_ok=True)
    design_path = OUT / "design.json"
    if design_path.exists():
        if json.loads(design_path.read_text(encoding="utf-8")) != design:
            raise RuntimeError("existing pulse design differs; refusing to mix experiments")
    else:
        write_json(design_path, design)
        (OUT / "sources").mkdir(exist_ok=True)
        for path in design["source_sha256"]:
            shutil.copy2(ROOT / path, OUT / "sources" / path.replace("/", "__"))
    if source_hashes(list(design["source_sha256"])) != design["source_sha256"]:
        raise RuntimeError("source changed before run")
    pending = []
    statuses = []
    for case in design["cases"]:
        progress_path = OUT / case["id"] / "progress.json"
        if progress_path.exists():
            status = json.loads(progress_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                statuses.append(status)
                continue
            raise RuntimeError(f"retained incomplete case requires audit: {case['id']}")
        pending.append(case)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_case, case) for case in pending]
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            write_json(OUT / "execution_progress.json", dict(results=statuses,
                                                               completed=sum(s.get("status") == "complete" for s in statuses),
                                                               total=len(design["cases"])))
            print(json.dumps(status, ensure_ascii=False), flush=True)
    if source_hashes(list(design["source_sha256"])) != design["source_sha256"]:
        raise RuntimeError("source changed during run")
    if any(status.get("status") != "complete" for status in statuses) or len(statuses) != len(design["cases"]):
        raise RuntimeError("some pulse cases failed; retained artifacts were not overwritten")
    write_json(OUT / "execution_verification.json", dict(
        all_completed=True, count=len(statuses), source_sha256=source_hashes(list(design["source_sha256"]))
    ))
    finalize(design)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(design: dict) -> None:
    results = {case["id"]: read_json(OUT / case["id"] / "result.json") for case in design["cases"]}
    checks = []
    for case in design["cases"]:
        if case["internal_dt"] != 0.0125:
            continue
        fine = results[case["id"]]["final"]
        coarse_id = case["id"].replace("dt0.0125", "dt0.025")
        coarse = results[coarse_id]["final"]
        pha = abs(coarse["pha_live_g_l"] - fine["pha_live_g_l"]) / max(0.001, abs(fine["pha_live_g_l"]))
        biomass = max(
            abs(coarse[key] - fine[key]) / max(1e-6, abs(fine[key]))
            for key in fine if key.endswith("_live_g_l") and key not in {"pha_live_g_l"}
        )
        hv = abs(coarse["hv_mol_fraction"] - fine["hv_mol_fraction"])
        checks.append(dict(id=case["id"], pha_error=pha, biomass_error=biomass,
                           hv_fraction_error=hv,
                           passed=pha <= 0.02 and biomass <= 0.02 and hv <= 0.001))

    table = []
    for schedule in ("const_k6", "pulse_2_10", "pulse_10_2"):
        for arm in ("two", "three"):
            candidates = [c for c in design["cases"] if c["schedule"] == schedule
                          and c["arm"] == arm]
            case = next((c for c in candidates if c["internal_dt"] == 0.0125),
                        next(c for c in candidates if c["internal_dt"] == 0.025))
            final = results[case["id"]]["final"]
            table.append(dict(schedule=schedule, arm=arm, internal_dt=case["internal_dt"],
                              pha=final["pha_live_g_l"],
                              hv=final["hv_mol_fraction"], rubber=final["rubber_removed_g_l"],
                              o2_transfer=final["oxygen_transferred"],
                              o2_capacity=final["oxygen_capacity_integral"]))
    pairs = []
    for schedule in ("const_k6", "pulse_2_10", "pulse_10_2"):
        two = next(row for row in table if row["schedule"] == schedule and row["arm"] == "two")
        three = next(row for row in table if row["schedule"] == schedule and row["arm"] == "three")
        pairs.append(dict(schedule=schedule,
                          pha_delta_pct=100.0 * (three["pha"] / max(1e-12, two["pha"]) - 1.0),
                          rubber_delta_pct=100.0 * (three["rubber"] / max(1e-12, two["rubber"]) - 1.0),
                          hv_delta_mol_fraction=three["hv"] - two["hv"]))
    summary = dict(cases=len(results), convergence=checks, convergence_passed=sum(x["passed"] for x in checks),
                   table=table, paired_species=pairs,
                   pulse_vs_constant6=[row for row in table if row["schedule"] in {"const_k6", "pulse_2_10", "pulse_10_2"}],
                   biological_validation=False, global_optimality_claim=False,
                   teacher_collection_started=False, limitations=design["limitations"])
    write_json(OUT / "summary.json", summary)
    report_lines = [
        "# 酸素パルス制御の時間刻み独立性検証",
        "",
        "24時間、早期乳酸6 mmol/L・早期NH4 0.4 mmol/L、同じ初期菌体量と背景培地で、",
        "新規にkLa一定6 h^-1と半日パルス（2→10、10→2 h^-1）を2種・3種で比較した。",
        "kLa一定2/10 h^-1は同じ全背景培地を使った既存比較を参照する。",
        "パルスの切替はシミュレータの制御区間を切って厳密に12 hで行い、内部刻み0.025/0.0125 hを比較した。",
        "",
        f"数値収束（PHA・菌体量2%、3HV分率0.1ポイント）は{summary['convergence_passed']}/{len(checks)}件。",
        "",
        "## 0.0125 hでの終点比較",
        "",
        "|schedule|構成|内部刻み (h)|PHA live (g/L)|3HV mol%|ゴム除去 (g/L)|O2移動 (mmol/L)|積分容量 (mmol/L)|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        report_lines.append(f"|{row['schedule']}|{row['arm']}|{row['internal_dt']:.4f}|{row['pha']:.6f}|{100*row['hv']:.3f}|{row['rubber']:.6f}|{row['o2_transfer']:.6f}|{row['o2_capacity']:.6f}|")
    report_lines.extend([
        "",
        "## 3種−2種の差",
        "",
        "|schedule|PHA差 (%)|ゴム除去差 (%)|3HV分率差 (ポイント)|",
        "|---|---:|---:|---:|",
    ])
    for row in pairs:
        report_lines.append(f"|{row['schedule']}|{row['pha_delta_pct']:+.3f}|{row['rubber_delta_pct']:+.3f}|{100*row['hv_delta_mol_fraction']:+.3f}|")
    report_lines.extend([
        "",
        "パルスとconst_k6は同じ積分kLa容量にそろえたが、実際の酸素移動量はDO履歴とOURで変わる。",
        "const_k6は内部刻み0.025 hの対照で、パルス条件のみ0.025/0.0125 hの細分収束を確認した。",
        "したがってパルス差を単純な曝気量の差とは解釈しない。dt収束を満たさない指標は優位性の根拠に使わない。",
        "本検証はモデル上の候補スケジュール評価であり、kLa、酸素飽和、維持代謝・死滅、PHA再利用、Pf表現型は未校正である。",
        "この候補集合から全球最適解、生物学的有意差、強化学習の性能は主張しない。教師大量収集はまだ開始していない。",
    ])
    (OUT / "REPORT_JA.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    artifact_names = ["design.json", "execution_verification.json", "summary.json", "REPORT_JA.md"]
    write_json(OUT / "completion.json", dict(completed=len(results), all_completed=True,
        convergence_passed=summary["convergence_passed"], source_hashes_match=True,
        resource_and_lp_audit=True, biological_validation=False,
        artifact_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in artifact_names}))
    print(json.dumps(dict(completed=len(results), convergence_passed=summary["convergence_passed"]), ensure_ascii=False))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    design = build_design()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        write_json(OUT / "design.json", design)
        print(json.dumps(dict(cases=len(design["cases"]), workers=args.workers), ensure_ascii=False))
        return
    run_all(args.workers)


if __name__ == "__main__":
    main()
