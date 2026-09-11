"""Aggregate every fixed-feed oxygen condition onto one axis and ask what the third species buys.

The three studies below were run with byte-identical simulator, GEM, medium,
inoculum, maintenance, death-rate, nitrogen-policy and pH settings, and with the
same fixed feed (lactate 6 mmol/L over the first 12 h, NH4 0.4 mmol/L over the
first 4 h, rubber 10 g/L, 24 h, controller interval 0.25 h):

  results/equal_budget_comparison_20260908         constant kLa 2, 10, 50 h^-1
  results/oxygen_pulse_validation_20260910_repaired_v2   constant kLa 6, two half-day pulses
  results/oxygen_schedule_extension_20260910       constant kLa 4, 8, four-block pulses, finer steps

The source hashes recorded in each design.json are compared before anything is
plotted; if any shared source differs the aggregation refuses to run.

Time-step convergence is recomputed here from the raw endpoints with one uniform
rule for every study (PHA 2%, per-species biomass 2%, 3HV mole fraction 0.001
absolute), because the 2026-09-08 study gated on PHA and biomass only and merely
reported the 3HV difference.  Conditions whose coarse counterpart does not exist
are reported as unassessed, not as converged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/third_species_value_20260910"
EQUAL_BUDGET = ROOT / "results/equal_budget_comparison_20260908"
PULSE = ROOT / "results/oxygen_pulse_validation_20260910_repaired_v2"
EXTENSION = ROOT / "results/oxygen_schedule_extension_20260910"
CLOSURE = ROOT / "results/oxygen_schedule_closure_20260910"

SHARED_SOURCES = [
    "main.py", "src/audited_dfba.py", "src/resolved_dfba.py", "src/physiology_dfba.py",
    "src/dfba_simulator.py", "src/cultivation_numerics.py", "src/b12_evidence.py",
    "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml",
    "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml",
    "models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml",
]
CRITERIA = dict(pha_relative=0.02, biomass_relative=0.02, hv_fraction_absolute=0.001)
LIMITATIONS = [
    dict(en="Every condition is one deterministic trajectory: no replicate spread, no statistical test.",
         ja="各条件は決定論的な1軌道であり、反復のばらつきも統計検定もない。"),
    dict(en="kLa, oxygen saturation, maintenance, death, PHA remobilization and the Pf phenotype are uncalibrated.",
         ja="kLa、酸素飽和、維持代謝、死滅率、PHA再利用、Pfの表現型はいずれも未校正である。"),
    dict(en="Only the fixed early lactate / early NH4 feed is used; the feed was not re-optimized per condition.",
         ja="供給は早期乳酸・早期NH4の1通りに固定しており、条件ごとの供給再最適化を行っていない。"),
    dict(en="Fixed inoculum split (0.5/0.1/0.03 and 0.525/0.105 g/L); the split was not optimized.",
         ja="接種比は固定（3種 0.5/0.1/0.03、2種 0.525/0.105 g/L）であり、接種比の最適化はしていない。"),
    dict(en="Constant-kLa points change the oxygen budget; only the five equal-budget candidates share an integrated kLa of 144 h.",
         ja="一定kLaの各点は酸素予算そのものが異なる。kLa積分144 hで一致するのは等予算の5候補だけである。"),
    dict(en="A 3HV rise is not by itself a product-quality gain, and feeding propionate directly to the two-species arm also raises 3HV.",
         ja="3HVの上昇はそれ自体が製品品質の改善ではなく、2種にプロピオン酸を直接与えても3HVは上昇する。"),
]

# condition -> (label, mean kLa, family, per-arm (study, fine case, coarse case or None))
CONDITIONS: list[dict] = [
    dict(key="const_k2", label="kLa 2 一定", mean_kla=2.0, family="supply_scan",
         study=EQUAL_BUDGET,
         fine={"two": "refine_k2_early_early_two", "three": "refine_k2_early_early_three"},
         coarse={"two": "screen_k2_early_early_two", "three": "screen_k2_early_early_three"}),
    dict(key="const_k4", label="kLa 4 一定", mean_kla=4.0, family="supply_scan",
         study=EXTENSION,
         fine={"two": "const_k4_two_dt0.0125", "three": "const_k4_three_dt0.0125"},
         coarse={"two": "const_k4_two_dt0.025", "three": "const_k4_three_dt0.025"}),
    dict(key="const_k6", label="kLa 6 一定", mean_kla=6.0, family="equal_budget",
         study=EXTENSION,
         fine={"two": "const_k6_two_dt0.0125", "three": "const_k6_three_dt0.0125"},
         coarse={"two": (PULSE, "const_k6_two_dt0.025"), "three": (PULSE, "const_k6_three_dt0.025")}),
    dict(key="const_k8", label="kLa 8 一定", mean_kla=8.0, family="supply_scan",
         study=EXTENSION,
         fine={"two": "const_k8_two_dt0.0125", "three": "const_k8_three_dt0.0125"},
         coarse={"two": "const_k8_two_dt0.025", "three": "const_k8_three_dt0.025"}),
    dict(key="const_k10", label="kLa 10 一定", mean_kla=10.0, family="supply_scan",
         study=EQUAL_BUDGET,
         fine={"two": "screen_k10_early_early_two", "three": "screen_k10_early_early_three"},
         coarse={"two": None, "three": None}),
    dict(key="const_k50", label="kLa 50 一定", mean_kla=50.0, family="supply_scan",
         study=EQUAL_BUDGET,
         fine={"two": "refine_k50_early_early_two", "three": "refine_k50_early_early_three"},
         coarse={"two": "screen_k50_early_early_two", "three": "screen_k50_early_early_three"}),
    dict(key="pulse_2_10", label="2→10 半日パルス", mean_kla=6.0, family="equal_budget",
         study=EXTENSION,
         fine={"two": "pulse_2_10_two_dt0.00625", "three": "pulse_2_10_three_dt0.00625"},
         coarse={"two": (PULSE, "pulse_2_10_two_dt0.0125"),
                 "three": (PULSE, "pulse_2_10_three_dt0.0125")},
         # One further halving exists for the two-species arm; when it is on disk the
         # convergence of the reported dt 0.00625 value is judged against it.
         finer={"two": (CLOSURE, "pulse_2_10_two_dt0.003125")}),
    dict(key="pulse_10_2", label="10→2 半日パルス", mean_kla=6.0, family="equal_budget",
         study=PULSE,
         fine={"two": "pulse_10_2_two_dt0.0125", "three": "pulse_10_2_three_dt0.0125"},
         coarse={"two": "pulse_10_2_two_dt0.025", "three": "pulse_10_2_three_dt0.025"}),
    dict(key="pulse4_2_10", label="2/10 6時間4分割", mean_kla=6.0, family="equal_budget",
         study=EXTENSION,
         fine={"two": "pulse4_2_10_two_dt0.0125", "three": "pulse4_2_10_three_dt0.0125"},
         coarse={"two": "pulse4_2_10_two_dt0.025", "three": "pulse4_2_10_three_dt0.025"}),
    dict(key="pulse4_10_2", label="10/2 6時間4分割", mean_kla=6.0, family="equal_budget",
         study=EXTENSION,
         fine={"two": "pulse4_10_2_two_dt0.0125", "three": "pulse4_10_2_three_dt0.0125"},
         coarse={"two": "pulse4_10_2_two_dt0.025", "three": "pulse4_10_2_three_dt0.025"}),
]
TRAJECTORY_CASES = [
    ("const_k6", "kLa 6 一定", EXTENSION, "const_k6_three_dt0.0125", "const_k6_two_dt0.0125"),
    ("pulse_2_10", "2→10 半日パルス", EXTENSION,
     "pulse_2_10_three_dt0.00625", "pulse_2_10_two_dt0.00625"),
    ("pulse4_2_10", "2/10 6時間4分割", EXTENSION,
     "pulse4_2_10_three_dt0.0125", "pulse4_2_10_two_dt0.0125"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def source_hashes(paths: list[str]) -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def check_shared_sources() -> dict:
    """Every study must have been produced by the same simulator and the same GEMs."""
    current = source_hashes(SHARED_SOURCES)
    recorded = {}
    for study in (EQUAL_BUDGET, PULSE, EXTENSION):
        design = read_json(study / "design.json")
        stored = design.get("source_sha256") or design.get("sources") or {}
        missing = [p for p in SHARED_SOURCES if p not in stored]
        if missing:
            raise RuntimeError(f"{study.name} does not pin {missing}")
        differing = sorted(p for p in SHARED_SOURCES if stored[p] != current[p])
        if differing:
            raise RuntimeError(f"{study.name} was produced with different sources: {differing}")
        recorded[study.name] = {p: stored[p] for p in SHARED_SOURCES}
    return dict(current=current, per_study=recorded, identical=True)


def convergence(coarse: dict, fine: dict) -> dict:
    """One uniform endpoint rule for every study."""
    pha = abs(coarse["pha_live_g_l"] - fine["pha_live_g_l"]) / max(0.001, abs(fine["pha_live_g_l"]))
    keys = [k for k in fine
            if k.endswith("_live_g_l")
            and k not in {"pha_live_g_l", "phb_live_g_l", "phv_live_g_l"}
            and k in coarse]
    biomass = max(abs(coarse[k] - fine[k]) / max(1e-6, abs(fine[k])) for k in keys)
    hv = abs(coarse["hv_mol_fraction"] - fine["hv_mol_fraction"])
    return dict(
        pha_error=pha, biomass_error=biomass, hv_fraction_error=hv,
        pha_converged=bool(pha <= CRITERIA["pha_relative"] and biomass <= CRITERIA["biomass_relative"]),
        hv_converged=bool(hv <= CRITERIA["hv_fraction_absolute"]),
        assessed=True,
    )


def _load(study: Path, case_id: str) -> dict:
    return read_json(study / case_id / "result.json")


def collect(partial: bool) -> list[dict]:
    rows: list[dict] = []
    for condition in CONDITIONS:
        record = dict(key=condition["key"], label=condition["label"],
                      mean_kla=condition["mean_kla"], family=condition["family"],
                      study=condition["study"].name)
        missing = False
        for arm in ("two", "three"):
            fine_id = condition["fine"][arm]
            fine_path = condition["study"] / fine_id / "result.json"
            if not fine_path.exists():
                missing = True
                break
            fine = _load(condition["study"], fine_id)
            record[f"{arm}_case"] = fine_id
            record[f"{arm}_dt"] = fine["case"].get("internal_dt", fine["case"].get("dt"))
            record[f"{arm}_pha"] = fine["final"]["pha_live_g_l"]
            record[f"{arm}_pha_total"] = fine["final"]["pha_total_g_l"]
            record[f"{arm}_hv"] = fine["final"]["hv_mol_fraction"]
            record[f"{arm}_rubber"] = fine["final"]["rubber_removed_g_l"]
            record[f"{arm}_o2_transferred"] = fine["final"]["oxygen_transferred"]
            coarse_ref = condition["coarse"][arm]
            if coarse_ref is None:
                record[f"{arm}_convergence"] = dict(assessed=False, pha_converged=False,
                                                    hv_converged=False, note="no coarse counterpart")
                record[f"{arm}_coarse_case"] = None
                continue
            coarse_study, coarse_id = coarse_ref if isinstance(coarse_ref, tuple) else (condition["study"], coarse_ref)
            coarse_path = coarse_study / coarse_id / "result.json"
            if not coarse_path.exists():
                missing = True
                break
            record[f"{arm}_coarse_case"] = f"{coarse_study.name}/{coarse_id}"
            record[f"{arm}_convergence"] = convergence(_load(coarse_study, coarse_id)["final"],
                                                       fine["final"])
            finer_ref = (condition.get("finer") or {}).get(arm)
            if finer_ref is None:
                continue
            finer_study, finer_id = finer_ref
            if not (finer_study / finer_id / "result.json").exists():
                record[f"{arm}_finer_case"] = None
                continue
            # The reported value stays at the symmetric step size; only the
            # convergence judgement uses the two finest runs of this arm.
            finer = _load(finer_study, finer_id)
            check = convergence(fine["final"], finer["final"])
            check["basis"] = f"{finer_study.name}/{finer_id}"
            record[f"{arm}_finer_case"] = check["basis"]
            record[f"{arm}_convergence"] = check
        if missing:
            if not partial:
                raise RuntimeError(f"missing results for condition {condition['key']}; "
                                   f"run the extension study first or pass --partial")
            print(f"skipping incomplete condition: {condition['key']}", flush=True)
            continue
        record["pha_delta_pct"] = 100.0 * (record["three_pha"] / max(1e-12, record["two_pha"]) - 1.0)
        record["rubber_delta_pct"] = 100.0 * (record["three_rubber"] / max(1e-12, record["two_rubber"]) - 1.0)
        record["hv_delta_points"] = 100.0 * (record["three_hv"] - record["two_hv"])
        record["pha_delta_converged"] = bool(record["two_convergence"]["pha_converged"]
                                             and record["three_convergence"]["pha_converged"])
        record["hv_delta_converged"] = bool(record["two_convergence"]["hv_converged"]
                                            and record["three_convergence"]["hv_converged"])
        record["delta_assessed"] = bool(record["two_convergence"]["assessed"]
                                        and record["three_convergence"]["assessed"])
        rows.append(record)
    return rows


JAPANESE_FONT_FILES = [
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def _use_japanese_font() -> str | None:
    """matplotlib does not always index /usr/share/fonts/opentype, so register the file."""
    for path in JAPANESE_FONT_FILES:
        file = Path(path)
        if not file.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(file))
            name = font_manager.FontProperties(fname=str(file)).get_name()
        except Exception:
            continue
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
        return name
    return None


def _pha_panel(ax, rows: list[dict], zoom: bool) -> list[dict]:
    scan = [r for r in rows if r["family"] == "supply_scan"]
    pulses = [r for r in rows if r["family"] == "equal_budget" and r["key"].startswith("pulse")]
    const6 = [r for r in rows if r["key"] == "const_k6"]
    offsets = [0.86, 0.93, 1.07, 1.15]
    markers = ["^", "v", "s", "D"]
    ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
    limit = 30.0
    line = [r for r in scan if not zoom or abs(r["pha_delta_pct"]) <= limit]
    if len(line) > 1:
        ax.plot([r["mean_kla"] for r in line], [r["pha_delta_pct"] for r in line],
                "-", color="0.55", lw=1.4, zorder=2)
    for row in scan + const6:
        if zoom and abs(row["pha_delta_pct"]) > limit:
            continue
        ax.plot(row["mean_kla"], row["pha_delta_pct"], "o", ms=9, zorder=4,
                mfc=("#1f5fa8" if row["pha_delta_converged"] else "white"),
                mec="#1f5fa8", mew=1.8)
    for index, row in enumerate(pulses):
        if zoom and abs(row["pha_delta_pct"]) > limit:
            continue
        ax.plot(row["mean_kla"] * offsets[index % len(offsets)], row["pha_delta_pct"],
                markers[index % len(markers)], ms=9, zorder=4,
                mfc=("#c0392b" if row["pha_delta_converged"] else "white"),
                mec="#c0392b", mew=1.8)
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 6, 8, 10, 50])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
    ax.grid(alpha=0.25)
    if zoom:
        near = [r["pha_delta_pct"] for r in rows if abs(r["pha_delta_pct"]) <= limit]
        if near:
            margin = max(1.0, 0.25 * (max(near) - min(near)))
            ax.set_ylim(min(near) - margin, max(near) + margin)
    return pulses


def figure_axis(rows: list[dict], path: Path) -> None:
    """(a) PHA advantage against oxygen supply, (b) the same for 3HV composition."""
    fig = plt.figure(figsize=(10.0, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.35, 1.0],
                            hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    zx = fig.add_subplot(grid[0, 1])
    bx = fig.add_subplot(grid[1, :])

    pulses = _pha_panel(ax, rows, zoom=False)
    _pha_panel(zx, rows, zoom=True)
    ax.set_xlabel("平均 kLa (h$^{-1}$)")
    ax.set_ylabel("生菌中PHAの3種−2種 差 (%)")
    ax.set_title("(a) 全条件", loc="left", fontsize=11)
    zx.set_xlabel("平均 kLa (h$^{-1}$)")
    zx.set_ylabel("同左 (%)")
    zx.set_title("(b) ±30%帯の拡大（kLa 2 を除く）", loc="left", fontsize=11)
    handles = [
        Line2D([], [], color="0.55", marker="o", ms=8, mfc="#1f5fa8", mec="#1f5fa8", lw=1.4,
               label="一定kLa"),
        Line2D([], [], color="none", marker="o", ms=8, mfc="white", mec="#1f5fa8", mew=1.8,
               label="白抜き＝dt未収束/未評価"),
    ]
    handles += [Line2D([], [], color="none", marker=["^", "v", "s", "D"][i % 4], ms=8,
                       mfc=("#c0392b" if r["pha_delta_converged"] else "white"),
                       mec="#c0392b", mew=1.8, label=r["label"])
                for i, r in enumerate(pulses)]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)

    # Dumbbell: 3HV spans three orders of magnitude, so bars from zero are unreadable.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["mean_kla"])
    floor = 0.02
    for position, index in enumerate(order):
        row = rows[index]
        two = max(floor, 100.0 * row["two_hv"])
        three = max(floor, 100.0 * row["three_hv"])
        bx.plot([two, three], [position, position], color="0.6", lw=1.6, zorder=1,
                solid_capstyle="round")
        bx.plot(two, position, "o", ms=9, zorder=3, mec="#7f8c8d", mew=1.8,
                mfc=("#7f8c8d" if row["two_convergence"]["hv_converged"] else "white"))
        bx.plot(three, position, "o", ms=9, zorder=3, mec="#2e86c1", mew=1.8,
                mfc=("#2e86c1" if row["three_convergence"]["hv_converged"] else "white"))
        bx.annotate(f"{row['hv_delta_points']:+.1f} pt", (max(two, three), position),
                    textcoords="offset points", xytext=(10, -3), fontsize=8, va="center")
    bx.set_xscale("log")
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels([rows[i]["label"] for i in order], fontsize=9)
    bx.set_xlabel(f"終点 3HV モル分率 (mol%、対数軸、{floor} mol%未満は{floor}に丸めて表示)")
    bx.set_title("(c) 3HVモル分率の2種→3種の変化（灰=2種、青=3種、白抜き＝dt未収束/未評価）",
                 loc="left", fontsize=11)
    bx.grid(alpha=0.25, axis="x")
    bx.set_xlim(floor * 0.7, 600)
    bx.set_ylim(-0.6, len(order) - 0.4)
    bx.legend(handles=[
        Line2D([], [], color="none", marker="o", ms=8, mfc="#7f8c8d", mec="#7f8c8d",
               label="2種 (OR16+NS21)"),
        Line2D([], [], color="none", marker="o", ms=8, mfc="#2e86c1", mec="#2e86c1",
               label="3種 (+P. freudenreichii)"),
    ], fontsize=9, loc="upper right")

    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_equal_budget(rows: list[dict], path: Path) -> None:
    """At a fixed integrated kLa of 144 h, does the temporal pattern matter?"""
    family = [r for r in rows if r["family"] == "equal_budget"]
    if not family:
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    positions = range(len(family))
    width = 0.38
    for offset, arm, color in ((-width / 2, "two", "#7f8c8d"), (width / 2, "three", "#2e86c1")):
        bars = ax.bar([p + offset for p in positions], [r[f"{arm}_pha"] for r in family], width,
                      color=color, edgecolor="black", lw=0.6,
                      label="2種" if arm == "two" else "3種")
        for bar, row in zip(bars, family):
            if not row[f"{arm}_convergence"]["pha_converged"]:
                bar.set_hatch("//")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([r["label"] for r in family], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("終点 生菌中PHA (g/L)")
    spans = {}
    for arm in ("two", "three"):
        values = [r[f"{arm}_pha"] for r in family]
        spans[arm] = (min(values), max(values),
                      100.0 * (max(values) - min(values)) / max(1e-12, min(values)))
    ax.set_title("(a) kLa積分144 h で一致させた5候補\n"
                 f"2種 {spans['two'][0]:.3f}–{spans['two'][1]:.3f} g/L（幅{spans['two'][2]:.0f}%）、"
                 f"3種 {spans['three'][0]:.3f}–{spans['three'][1]:.3f} g/L（幅{spans['three'][2]:.0f}%）",
                 loc="left", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    bx.axhline(0.0, color="0.35", lw=1.0)
    colors = ["#2e86c1" if r["pha_delta_converged"] else "#aab7c4" for r in family]
    bars = bx.bar(list(positions), [r["pha_delta_pct"] for r in family], 0.6,
                  color=colors, edgecolor="black", lw=0.6)
    for bar, row in zip(bars, family):
        if not row["pha_delta_converged"]:
            bar.set_hatch("//")
    for pos, row in zip(positions, family):
        bx.annotate(f"3HV {row['hv_delta_points']:+.1f}pt", (pos, row["pha_delta_pct"]),
                    textcoords="offset points", xytext=(0, 6 if row["pha_delta_pct"] >= 0 else -14),
                    ha="center", fontsize=8)
    bx.set_xticks(list(positions))
    bx.set_xticklabels([r["label"] for r in family], rotation=25, ha="right", fontsize=9)
    bx.set_ylabel("PHAの3種−2種 差 (%)")
    bx.set_title("(b) 同じ曝気予算でも時間配分で差が変わる", loc="left", fontsize=10)
    bx.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _trajectory(study: Path, case: str) -> list[dict] | None:
    progress = study / case / "progress.json"
    path = study / case / "trajectory.csv"
    if not (progress.exists() and path.exists()):
        return None
    # A running case has a partially written trajectory; never plot it.
    if read_json(progress).get("status") != "complete":
        print(f"skipping incomplete trajectory: {study.name}/{case}", flush=True)
        return None
    with path.open() as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def figure_trajectories(path: Path) -> list[dict]:
    available = []
    for key, label, study, three_case, two_case in TRAJECTORY_CASES:
        three = _trajectory(study, three_case)
        two = _trajectory(study, two_case)
        if three is None or two is None:
            continue
        available.append((key, label, study, three_case, three, two_case, two))
    if not available:
        # Never leave a figure from an earlier, partial run on disk.
        for suffix in (".png", ".pdf"):
            stale = path.with_suffix(suffix)
            if stale.exists():
                stale.unlink()
                print(f"removed stale figure: {stale.name}", flush=True)
        return []
    fig, axes = plt.subplots(4, 1, figsize=(8.8, 10.8), sharex=True)
    colors = ["#1f5fa8", "#c0392b", "#27ae60"]
    used = []
    for index, (key, label, study, three_case, three, two_case, two) in enumerate(available):
        color = colors[index % len(colors)]
        used.append(dict(key=key, study=study.name, three_case=three_case, two_case=two_case,
                         three_points=len(three), two_points=len(two)))
        for rows, style, width in ((three, "-", 1.8), (two, "--", 1.3)):
            time = [r["time_h"] for r in rows]
            axes[0].plot(time, [r["pha_live_g_l"] for r in rows], style, color=color, lw=width,
                         label=(label if style == "-" else None))
            axes[1].plot(time, [100.0 * r["hv_mol_fraction"] for r in rows], style, color=color,
                         lw=width)
            axes[2].plot(time, [r["o2_mmol_l"] for r in rows], style, color=color, lw=width)
        if "Pf_live_g_l" in three[0]:
            axes[3].plot([r["time_h"] for r in three], [r["Pf_live_g_l"] for r in three],
                         "-", color=color, lw=1.8)
    axes[0].set_ylabel("生菌中PHA (g/L)")
    axes[1].set_ylabel("3HV モル分率 (mol%)")
    axes[2].set_ylabel("溶存O2 (mmol/L)")
    axes[3].set_ylabel("P. freudenreichii 生菌 (g/L)")
    axes[3].set_xlabel("時間 (h)")
    axes[0].set_title("代表軌道（同一供給、kLa積分144 h、実線=3種、破線=2種）", loc="left")
    axes[3].set_title("3種のみ（2種にはこの菌がない）", loc="left", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=colors[i % len(colors)], lw=1.8, label=row[1])
               for i, row in enumerate(available)]
    handles += [Line2D([], [], color="0.2", lw=1.8, label="3種"),
                Line2D([], [], color="0.2", lw=1.3, ls="--", label="2種")]
    axes[0].legend(handles=handles, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return used


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true",
                        help="skip conditions whose results are not on disk yet")
    args = parser.parse_args()

    provenance = check_shared_sources()
    if _JAPANESE_FONT is None:
        raise RuntimeError("no Japanese font available; figure labels would be rendered as boxes")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect(args.partial)
    if not rows:
        raise RuntimeError("no conditions available")

    fields = ["key", "label", "family", "mean_kla", "study",
              "two_case", "two_dt", "two_pha", "two_pha_total", "two_hv", "two_rubber",
              "two_o2_transferred", "three_case", "three_dt", "three_pha", "three_pha_total",
              "three_hv", "three_rubber", "three_o2_transferred",
              "pha_delta_pct", "rubber_delta_pct", "hv_delta_points",
              "pha_delta_converged", "hv_delta_converged"]
    with (OUT / "third_species_axis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    figure_axis(rows, OUT / "fig1_third_species_axis")
    figure_equal_budget(rows, OUT / "fig2_equal_budget_family")
    trajectories = figure_trajectories(OUT / "fig3_representative_trajectories")

    converged_gain = [r for r in rows if r["pha_delta_converged"] and r["pha_delta_pct"] > 0]
    converged_loss = [r for r in rows if r["pha_delta_converged"] and r["pha_delta_pct"] < 0]
    figures = ["fig1_third_species_axis", "fig2_equal_budget_family"]
    if trajectories:
        figures.append("fig3_representative_trajectories")
    summary = dict(
        conditions=len(rows), partial=bool(args.partial),
        criteria=CRITERIA, provenance=provenance,
        rows=rows, trajectories=trajectories,
        pha_gain_conditions=[r["key"] for r in converged_gain],
        pha_loss_conditions=[r["key"] for r in converged_loss],
        max_pha_gain_pct=max((r["pha_delta_pct"] for r in converged_gain), default=None),
        max_hv_gain_points=max((r["hv_delta_points"] for r in rows if r["hv_delta_converged"]),
                               default=None),
        japanese_font=_JAPANESE_FONT,
        biological_validation=False, statistical_test=False,
        global_optimality_claim=False, teacher_collection_started=False,
        limitations=LIMITATIONS,
        figure_sha256={},
    )
    figure_hashes = {}
    for name in figures:
        for suffix in (".png", ".pdf"):
            file = OUT / (name + suffix)
            figure_hashes[name + suffix] = hashlib.sha256(file.read_bytes()).hexdigest()
    figure_hashes["third_species_axis.csv"] = hashlib.sha256(
        (OUT / "third_species_axis.csv").read_bytes()).hexdigest()
    summary["figure_sha256"] = figure_hashes
    write_json(OUT / "summary.json", summary)
    print(json.dumps(dict(conditions=len(rows), figures=figures,
                          pha_gain=summary["pha_gain_conditions"],
                          pha_loss=summary["pha_loss_conditions"],
                          japanese_font=_JAPANESE_FONT), ensure_ascii=False))


_JAPANESE_FONT = _use_japanese_font()

if __name__ == "__main__":
    main()
