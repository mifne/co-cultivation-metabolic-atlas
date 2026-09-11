"""Finalize the direct-propionate comparison after the extra time-step audit."""
from pathlib import Path
import csv, hashlib, json, math

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/direct_propionate_comparison_20260909"
EXTRA = ROOT / "results/direct_propionate_refinement_20260909"


def read(path):
    return json.loads(path.read_text())


def rel(a, b, floor):
    return abs(a - b) / max(floor, abs(b))


def main():
    design = read(EXTRA / "design.json")
    cases = design["cases"]
    assert len(cases) == 36
    assert read(EXTRA / "execution_verification.json")["all_completed"]
    source_hashes = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in design["source_sha256"]
    }
    assert source_hashes == design["source_sha256"]

    results = {}
    checks = []
    for case in cases:
        case_dir = EXTRA / case["id"]
        progress = read(case_dir / "progress.json")
        result = read(case_dir / "result.json")
        settings = read(case_dir / "settings.json")
        assert progress["status"] == "complete"
        assert result["case"] == settings["case"] == case
        assert settings["initial_medium"] == design["common_initial_medium"]
        assert abs(sum(settings["initial_biomass"].values()) - 0.63) < 1e-12
        with (case_dir / "trajectory.csv").open() as handle:
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
        assert len(rows) == 97
        assert rows[-1] == result["final"]
        for index, row in enumerate(rows):
            assert abs(row["time_h"] - 0.25 * index) < 1e-9
            assert all(math.isfinite(value) for value in row.values())
            assert all(value >= -1e-8 for key, value in row.items()
                       if key.endswith("_g_l") or key.endswith("_mmol_l"))
            assert row["oxygen_transferred"] <= case["kla"] * row["time_h"] * settings["oxygen_saturation_mmol_l"] + 1e-8
            assert abs(row["pha_total_g_l"] - row["pha_live_g_l"] - row["pha_dead_g_l"]) < 1e-9
            lactate_profile = case["profile"]["lactate"]
            total_feed = (0.25 * row["time_h"] if lactate_profile == "uniform"
                          else 0.5 * min(row["time_h"], 12.0) if lactate_profile == "early"
                          else 0.5 * max(row["time_h"] - 12.0, 0.0))
            assert abs(row["lactate_delivered"] + row["propionate_delivered"] - total_feed) < 1e-7
            assert abs(row["nh4_delivered"] - 0.4 * min(1.0, row["time_h"] / (4.0 if case["profile"]["nitrogen"] == "early" else 12.0))) < 1e-7
            assert abs(row["C_residual"]) < 1e-6 and abs(row["N_residual"]) < 1e-6
            assert row["oxygen_balance_error"] <= 1e-8 and row["oxygen_kinetic_error"] <= 1e-6
            assert row["storage_excess"] <= 1e-7
        accounting = result["diagnostics"]["audited_cultivation"]["accounting"]
        assert accounting["failed_steps"] == 0
        assert max(accounting[key] for key in ("max_lp_residual", "max_dual_residual", "max_relative_duality_gap")) <= 1e-7
        assert accounting["lp_requests"] == accounting["lp_certified_requests"] + accounting.get("superseded_uncertified_requests", 0)
        results[case["id"]] = result

        parent_id = case["id"].replace("extra_", "refine_", 1)
        parent = read(BASE / parent_id / "result.json")["final"]
        final = result["final"]
        errors = {
            "pha_error": rel(parent["pha_live_g_l"], final["pha_live_g_l"], 0.001),
            "phv_error": rel(parent["phv_live_g_l"], final["phv_live_g_l"], 0.0001),
            "biomass_error": max(rel(parent[key], value, 1e-6) for key, value in final.items()
                                  if key.endswith("_live_g_l") and key not in ("pha_live_g_l", "phv_live_g_l")),
            "hv_fraction_error": abs(parent["hv_mol_fraction"] - final["hv_mol_fraction"]),
        }
        checks.append(dict(id=case["id"], **errors,
                           passed=max(errors["pha_error"], errors["phv_error"], errors["biomass_error"]) <= 0.02
                           and errors["hv_fraction_error"] <= 0.001))

    best = []
    for kla in (2.0, 10.0):
        for arm in ("three", "two_propionate", "two_mixed"):
            group = [results[case["id"]] for case in cases
                     if case["kla"] == kla and case["arm"] == arm]
            assert len(group) == 6
            for metric in ("pha_live_g_l", "phv_live_g_l"):
                winner = max(group, key=lambda item: item["final"][metric])
                best.append(dict(kla=kla, arm=arm, objective=metric,
                                 id=winner["case"]["id"], final=winner["final"]))

    summary = dict(
        completed=len(results),
        refinement_checks=checks,
        refinement_passed=sum(item["passed"] for item in checks),
        refinement_failed=[item["id"] for item in checks if not item["passed"]],
        best=best,
        biological_validation=False,
        global_optimality_claim=False,
        teacher_collection_started=False,
        limitations=design["limitations"],
    )
    (EXTRA / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = "\n".join(
        f"|{item['kla']:g}|{item['arm']}|{item['objective']}|{item['final']['pha_live_g_l']:.6f}|"
        f"{item['final']['phv_live_g_l']:.6f}|{100 * item['final']['hv_mol_fraction']:.3f}|"
        for item in best
    )
    failed = ", ".join(summary["refinement_failed"]) or "なし"
    report = f"""# プロピオン酸直接添加比較：最終監査

初回72ケースと追加36ケース（内部刻み0.00625 h）を完了した。追加36ケースの保存時点、供給積分、C/N収支、酸素収支、LP認証、ソースハッシュを再監査した。

追加細分の収束判定は、PHA・3HV絶対量・菌種別菌体量が2%以内、3HVモル分率が0.1ポイント以内とした。合格は{summary['refinement_passed']}/{len(checks)}件、未達は次の通り：{failed}。

## 最終細分での各構成の最大値

|kLa|構成|選択指標|PHA (g/L)|3HV単位質量 (g/L)|3HV (mol%)|
|---|---|---|---:|---:|---:|
{table}

PHA総量の比較と3HV組成の比較は分けて解釈する。未収束の3HV量・組成を優位性の根拠にしない。

この比較は、乳酸・プロピオン酸を同じ炭素量にそろえた有限候補探索である。還元当量、原料費、供給液の質量、実酸素消費は同一ではない。混合比も0、0.5、1に限定し、全球最適化ではない。プロピオン酸阻害、Pfの酸素表現型、PHV制御は未校正であり、Pfの分泌を直接添加で再現した因果比較ではない。

大量教師収集とRL学習は開始していない。
"""
    (EXTRA / "FINAL_REPORT_JA.md").write_text(report, encoding="utf-8")

    artifact_names = ["final_summary.json", "FINAL_REPORT_JA.md", "design.json", "execution_verification.json"]
    completion = dict(
        completed=len(results),
        all_completed=True,
        refinement_passed=summary["refinement_passed"],
        refinement_failed=summary["refinement_failed"],
        source_hashes_match=True,
        resource_and_lp_audit=True,
        biological_validation=False,
        teacher_collection_started=False,
        artifact_sha256={name: hashlib.sha256((EXTRA / name).read_bytes()).hexdigest() for name in artifact_names},
    )
    (EXTRA / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(dict(completed=len(results), passed=summary["refinement_passed"], failed=summary["refinement_failed"]), indent=2))


if __name__ == "__main__":
    main()
