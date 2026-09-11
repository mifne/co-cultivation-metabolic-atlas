"""Close the last unmet convergence criterion of the oxygen-schedule work.

After refining `pulse_2_10` to internal dt 0.00625 h, eleven of twelve pairs met
the uniform numerical criterion.  The single remaining failure is the 3HV mole
fraction of the two-species arm of `pulse_2_10`: 0.00475 between 0.025 and
0.0125 h, then 0.00140 between 0.0125 and 0.00625 h, against a 0.001 bound.
One further halving to 0.003125 h decides it.

The case is executed by importing the extension study's `run_case`, so the
physics, audits and output format are byte-identical to the study it extends;
only `internal_dt` differs.  Results live in their own directory with their own
design and source pinning.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.oxygen_schedule_extension_20260910 import (  # noqa: E402
    CONVERGENCE, SHARED_SOURCES, make_case, prepare, read_json, run_case,
    source_hashes, write_json, _convergence_check,
)

OUT = ROOT / "results/oxygen_schedule_closure_20260910"
EXTENSION = ROOT / "results/oxygen_schedule_extension_20260910"
PULSE = ROOT / "results/oxygen_pulse_validation_20260910_repaired_v2"
COARSE_ID = "pulse_2_10_two_dt0.00625"
SOURCES = [
    "scripts/analysis/oxygen_schedule_closure_20260910.py",
    "scripts/analysis/oxygen_schedule_extension_20260910.py",
    *SHARED_SOURCES,
    "results/oxygen_schedule_extension_20260910/design.json",
]


def _recorded_error(study: Path, case_id: str) -> float:
    """Read the previously stored 3HV step error instead of transcribing it."""
    summary = read_json(study / "summary.json")
    for check in summary["convergence"]:
        if check["id"] == case_id:
            return float(check["hv_fraction_error"])
    raise RuntimeError(f"{study.name} has no convergence record for {case_id}")


def build_design() -> dict:
    extension = read_json(EXTENSION / "design.json")
    case = make_case("pulse_2_10", "two", 0.003125, "convergence_closure")
    return dict(
        cases=[case], source_sha256=source_hashes(SOURCES),
        objective="one further halving of the internal step for the only pair that missed the 3HV criterion",
        common=extension["common"], common_initial_medium=extension["common_initial_medium"],
        convergence_criteria=CONVERGENCE,
        external_coarse_reference=dict(
            study="results/oxygen_schedule_extension_20260910",
            pairs={case["id"]: COARSE_ID},
            shared_source_sha256={p: extension["source_sha256"][p] for p in SHARED_SOURCES},
        ),
        limitations=extension["limitations"],
    )


def main() -> None:
    design = build_design()
    expected = design["external_coarse_reference"]["shared_source_sha256"]
    if source_hashes(list(expected)) != expected:
        raise RuntimeError("simulator or model sources differ from the extension study")
    prepare(OUT, design)
    case = design["cases"][0]
    progress = OUT / case["id"] / "progress.json"
    if progress.exists() and read_json(progress).get("status") == "complete":
        print(json.dumps(dict(status="already complete", id=case["id"]), ensure_ascii=False))
    else:
        started = time.monotonic()
        status = run_case(case, str(OUT))
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if status.get("status") != "complete":
            raise SystemExit(1)
        print(json.dumps(dict(wall_seconds=time.monotonic() - started), ensure_ascii=False))
    if source_hashes(list(design["source_sha256"])) != design["source_sha256"]:
        raise RuntimeError("source changed during run")

    fine = read_json(OUT / case["id"] / "result.json")
    coarse = read_json(EXTENSION / COARSE_ID / "result.json")
    check = _convergence_check(case["id"], coarse["final"], fine["final"])
    check["coarse"] = COARSE_ID
    check["coarse_study"] = EXTENSION.name
    history = [
        dict(pair="0.025 -> 0.0125",
             hv_fraction_error=_recorded_error(PULSE, "pulse_2_10_two_dt0.0125"),
             source=PULSE.name),
        dict(pair="0.0125 -> 0.00625",
             hv_fraction_error=_recorded_error(EXTENSION, COARSE_ID),
             source=EXTENSION.name),
        dict(pair="0.00625 -> 0.003125",
             hv_fraction_error=check["hv_fraction_error"], source=OUT.name),
    ]
    summary = dict(case=case, convergence=check, refinement_history=history,
                   final=fine["final"], seconds=fine["seconds"],
                   criteria=CONVERGENCE, biological_validation=False,
                   limitations=design["limitations"])
    write_json(OUT / "summary.json", summary)
    lines = [
        "# pulse_2_10 2種の3HVモル分率：最終細分",
        "",
        "統一基準（PHA・菌種別菌体2%、3HVモル分率0.001絶対）で唯一未達だった",
        "`pulse_2_10` 2種の3HVモル分率について、内部刻みをもう一段細かくした。",
        "",
        "|比較|3HV分率の差|",
        "|---|---:|",
    ]
    for row in history:
        lines.append(f"|{row['pair']} h|{row['hv_fraction_error']:.5f}|")
    lines.extend([
        "",
        f"PHA誤差 {check['pha_error']:.5f}、菌体誤差 {check['biomass_error']:.5f}、"
        f"3HV分率誤差 {check['hv_fraction_error']:.5f}。判定は{'合格' if check['passed'] else '未達'}。",
        "",
        f"終点：生菌中PHA {fine['final']['pha_live_g_l']:.6f} g/L、"
        f"3HV {100 * fine['final']['hv_mol_fraction']:.3f} mol%、"
        f"ゴム除去 {fine['final']['rubber_removed_g_l']:.6f} g/L。",
        "",
        "この細分は数値解の刻み依存性のみを扱う。kLa、酸素飽和、維持代謝・死滅、PHA再利用、Pf表現型は未校正であり、",
        "生物学的有意差も統計的有意差も主張しない。",
    ])
    (OUT / "REPORT_JA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "completion.json", dict(
        completed=1, all_completed=True, passed=check["passed"],
        source_hashes_match=True, resource_and_lp_audit=True, biological_validation=False,
        artifact_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                         for name in ("design.json", "summary.json", "REPORT_JA.md")}))
    print(json.dumps(dict(passed=check["passed"],
                          hv_fraction_error=check["hv_fraction_error"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
