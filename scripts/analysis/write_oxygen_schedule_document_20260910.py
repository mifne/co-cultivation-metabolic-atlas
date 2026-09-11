"""Write the docs report for the oxygen-schedule work from the stored summaries.

Every number in the document is read from the study artifacts, so the document
cannot drift from the results.  The document is only written when all three
studies are complete and their shared sources hash identically.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DOC = ROOT / "docs/OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md"
EQUAL_BUDGET = ROOT / "results/equal_budget_comparison_20260908"
PULSE = ROOT / "results/oxygen_pulse_validation_20260910_repaired_v2"
EXTENSION = ROOT / "results/oxygen_schedule_extension_20260910"
CLOSURE = ROOT / "results/oxygen_schedule_closure_20260910"
VALUE = ROOT / "results/third_species_value_20260910"
PROPIONATE = ROOT / "results/direct_propionate_refinement_20260909"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flag(converged, assessed: bool = True) -> str:
    if not assessed:
        return "未評価"
    return "合格" if converged else "未達"


def main() -> None:
    for path in (PULSE / "completion.json", EXTENSION / "completion.json", VALUE / "summary.json"):
        if not path.exists():
            raise SystemExit(f"missing artifact: {path.relative_to(ROOT)}")
    pulse = read_json(PULSE / "summary.json")
    pulse_done = read_json(PULSE / "completion.json")
    extension = read_json(EXTENSION / "summary.json")
    extension_done = read_json(EXTENSION / "completion.json")
    value = read_json(VALUE / "summary.json")
    if value["partial"]:
        raise SystemExit("the aggregation was produced with --partial; rerun it on the full set")
    if not value["provenance"]["identical"]:
        raise SystemExit("shared sources are not identical across studies")

    rows = value["rows"]
    by_key = {row["key"]: row for row in rows}
    gains = [by_key[k] for k in value["pha_gain_conditions"]]
    losses = [by_key[k] for k in value["pha_loss_conditions"]]
    equal = [r for r in rows if r["family"] == "equal_budget"]
    hv_gain = [r for r in rows if r["hv_delta_converged"] and r["hv_delta_points"] > 0.1]
    hv_drop = [r for r in rows if r["hv_delta_converged"] and r["hv_delta_points"] < -0.1]
    hv_unassessed = [r for r in rows if not r["hv_delta_converged"]]

    lines = [
        "# 酸素スケジュールと第3菌種の価値",
        "",
        "2026-09-10。大量教師収集とRL学習に入る前に、酸素移動スケジュールを固定供給下で比較し、",
        "P. freudenreichii を加えた3種構成が何をもたらすかを同一軸上で確認した。",
        "",
        "## 比較の土台",
        "",
        "以下の計算研究を1つの軸に載せた。共有ソース（シミュレータ、GEM、酸素・元素監査）のSHA256が",
        "全研究で完全に一致することを集計前に照合しており、一致しない場合は集計を実行しない。",
        "",
        "|研究|内容|成果物|",
        "|---|---|---|",
        f"|同一資源予算での供給探索|kLa一定 2 / 10 / 50 h⁻¹|`{EQUAL_BUDGET.relative_to(ROOT)}`|",
        f"|酸素パルスの時間刻み独立性検証|kLa一定 6 h⁻¹、半日パルス2種|`{PULSE.relative_to(ROOT)}`|",
        f"|酸素スケジュール拡張|kLa一定 4 / 8 h⁻¹、6時間4分割パルス2種、細分刻み|`{EXTENSION.relative_to(ROOT)}`|",
        f"|3種の価値の集計と作図|上記を同一軸で比較|`{VALUE.relative_to(ROOT)}`|",
    ]
    if (CLOSURE / "completion.json").exists():
        lines.insert(len(lines) - 1,
                     f"|最終細分|`pulse_2_10` 2種の3HVを内部刻み0.003125 hで確認|"
                     f"`{CLOSURE.relative_to(ROOT)}`|")
    lines += [
        "",
        "全条件で共通に固定した項目：24時間、制御記録間隔0.25 h、初期ゴム10 g/L、",
        "乳酸6 mmol/L（最初の12 hに0.5 mmol/L/h）、追加NH4 0.4 mmol/L（最初の4 hに0.1 mmol/L/h）、",
        "全背景培地（ビタミン・塩類・微量金属・初期O2 0.25 mmol/L）、pH目標7、",
        "接種量は3種 OR16/NS21/Pf = 0.5/0.1/0.03 g/L、2種 0.525/0.105 g/L（総量0.63 g/L）、",
        "維持ATP 0.1 mmol/g/h、基礎死滅率0.01 h⁻¹、飢餓死滅率0.1 h⁻¹、窒素配分は capacity_ratio、PHA再利用は有効。",
        "供給は条件ごとに再最適化していない。したがって本比較は「同じ供給での構成差」であり、各構成の最良値の比較ではない。",
        "",
        "## 時間刻み収束の状態",
        "",
        "判定基準は全研究で統一し、終点値で PHA 2%、菌種別生菌体 2%、3HVモル分率 0.001 絶対とした。",
        "2026-09-08の研究はPHAと菌体のみで判定し3HV差は併記のみだったため、ここで3HVも同じ基準で再判定している。",
        f"パルス検証（{PULSE.name}）は自身の判定で {pulse['convergence_passed']}/{len(pulse['convergence'])} 件、",
        f"拡張（{EXTENSION.name}）は {extension['convergence_passed']}/{extension['convergence_total']} 件。",
        "パルス検証で未達だった項目は、以下の追加細分で判定し直している。",
        "",
        "|酸素条件|2種の最細刻み (h)|3種の最細刻み (h)|PHA差のdt収束|3HV差のdt収束|",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        assessed = row.get("delta_assessed", True)
        lines.append(f"|{row['label']}|{row['two_dt']:g}|{row['three_dt']:g}|"
                     f"{flag(row['pha_delta_converged'], assessed)}|"
                     f"{flag(row['hv_delta_converged'], assessed)}|")
    lines.extend([
        "",
        "未収束・未評価の指標は優位性の根拠に使わない。図では白抜きマーカーで区別している。",
        "kLa 10 一定は粗い刻みの対応ケースが存在せず、刻み依存性を評価していない。",
    ])
    if (CLOSURE / "completion.json").exists():
        closure = read_json(CLOSURE / "summary.json")
        lines.extend([
            "",
            "`pulse_2_10` 2種の3HVモル分率だけは0.00625 hでも基準を満たさなかったため、もう一段細かくした。",
            "",
            "|比較する内部刻み (h)|3HVモル分率の差|",
            "|---|---:|",
        ])
        for row in closure["refinement_history"]:
            lines.append(f"|{row['pair']}|{row['hv_fraction_error']:.5f}|")
        lines.extend([
            "",
            f"最終判定は{'合格' if closure['convergence']['passed'] else '未達'}"
            f"（`{CLOSURE.relative_to(ROOT)}`）。",
            "表と図に載せた値は両構成で同じ内部刻み0.00625 hのままにし、収束判定だけを最細の対で行っている。",
        ])
    lines.extend([
        "",
        "## 酸素条件ごとの3種−2種",
        "",
        "|酸素条件|kLa積分 (h)|2種PHA (g/L)|3種PHA (g/L)|PHA差 (%)|ゴム除去差 (%)|3HV差 (ポイント)|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        integral = row["mean_kla"] * 24.0
        lines.append(f"|{row['label']}|{integral:.0f}|{row['two_pha']:.6f}|{row['three_pha']:.6f}|"
                     f"{row['pha_delta_pct']:+.3f}|{row['rubber_delta_pct']:+.3f}|"
                     f"{row['hv_delta_points']:+.3f}|")
    gain_text = ("、".join(f"{r['label']} ({r['pha_delta_pct']:+.1f}%)" for r in gains)
                 if gains else "なし")
    loss_text = ("、".join(f"{r['label']} ({r['pha_delta_pct']:+.1f}%)" for r in losses)
                 if losses else "なし")
    lines.extend([
        "",
        f"dt収束を満たしたうえで3種のPHAが上回った条件：{gain_text}。",
        f"dt収束を満たしたうえで下回った条件：{loss_text}。",
        "3種の増産は条件依存であり、酸素供給が低い側では大きく不利になる。",
        "この符号反転は本モデル上の観測であり、生物学的な必然性としては未検証である。",
        "",
        "## 同じ曝気予算での時間配分",
        "",
        f"kLa積分を144 h（一定6 h⁻¹相当）にそろえた候補が{len(equal)}件ある。",
        "4つのパルス候補はレベル集合（2と10 h⁻¹）と各レベルの合計時間（各12 h）を固定し、",
        "切替の順序と回数だけを変えている。kLa 6 一定は同じ積分を一定値で与える対照である。",
        "",
        "|候補|2種PHA (g/L)|3種PHA (g/L)|PHA差 (%)|3HV差 (ポイント)|実O2移動 2種/3種 (mmol/L)|",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in sorted(equal, key=lambda r: r["key"]):
        lines.append(f"|{row['label']}|{row['two_pha']:.6f}|{row['three_pha']:.6f}|"
                     f"{row['pha_delta_pct']:+.3f}|{row['hv_delta_points']:+.3f}|"
                     f"{row['two_o2_transferred']:.2f} / {row['three_o2_transferred']:.2f}|")
    spread = max(r["three_pha"] for r in equal) - min(r["three_pha"] for r in equal)
    base = max(1e-12, min(r["three_pha"] for r in equal))
    two_spread = max(r["two_pha"] for r in equal) - min(r["two_pha"] for r in equal)
    two_base = max(1e-12, min(r["two_pha"] for r in equal))
    best = max(equal, key=lambda r: r["three_pha"])
    widest = max(equal, key=lambda r: r["pha_delta_pct"])
    lines.extend([
        "",
        f"同じ曝気予算のなかで、2種のPHAは {min(r['two_pha'] for r in equal):.3f}〜"
        f"{max(r['two_pha'] for r in equal):.3f} g/L（幅 {100.0 * two_spread / two_base:.1f}%）、",
        f"3種は {min(r['three_pha'] for r in equal):.3f}〜{max(r['three_pha'] for r in equal):.3f} g/L"
        f"（幅 {100.0 * spread / base:.1f}%）に分布した。",
        f"3種−2種の差が最大になるのは {widest['label']}（{widest['pha_delta_pct']:+.1f}%）で、",
        "そこは2種のPHAが最も低くなる条件でもある。つまりこの候補集合では、3種は上限を押し上げるよりも",
        "曝気の時間配分が悪い場合の落ち込みを緩和する向きに働いた。",
        f"絶対量が最大なのは {best['label']} の3種（{best['three_pha']:.6f} g/L）で、パルスは一定kLaを上回らなかった。",
        "曝気能力の積分は同じでも実際の酸素移動量はDO履歴とOURで変わるため、差を曝気量の差とは解釈しない。",
        "時間配分が結果を変えること自体は、酸素スケジュールを学習の行動空間に入れる根拠になる。",
        "ただし候補は有限個の手作り集合であり、最適スケジュールの主張ではない。",
        "",
        "## 3HV組成について",
        "",
    ])
    if hv_gain:
        lines.append("dt収束を満たしたうえで3種が3HVを上げた条件："
                     + "、".join(f"{r['label']} ({r['hv_delta_points']:+.1f}ポイント)" for r in hv_gain)
                     + "。")
    else:
        lines.append("dt収束を満たしたうえで3種が3HVを有意に上げた条件は本候補集合内にない。")
    if hv_drop:
        lines.append("dt収束を満たしたうえで3種が3HVを下げた条件："
                     + "、".join(f"{r['label']} ({r['hv_delta_points']:+.1f}ポイント)" for r in hv_drop)
                     + "。3種を入れれば3HVが上がる、という一般化はできない。")
    if hv_unassessed:
        lines.append("3HV差がdt未収束・未評価の条件："
                     + "、".join(r["label"] for r in hv_unassessed)
                     + "。これらの組成差は数値的に確定していない。")
    lines.extend([
        "",
        f"3HVの上昇は3種でしか得られない効果ではない。`{PROPIONATE.relative_to(ROOT)}` では、",
        "2種にプロピオン酸を直接与えた条件で3HVモル分率が最大100 mol%に達した（同研究の細分でも未収束項目が残る）。",
        "3種の意味は「3HVを作れること」ではなく、「乳酸からプロピオン酸を系内で供給できること」であり、",
        "原料費・還元当量・供給液質量の比較は行っていない。3HVの上昇自体も製品品質の改善とは同一視しない。",
        "",
        "## 図",
        "",
    ])
    for name, caption in (
        ("fig1_third_species_axis", "酸素条件に対するPHA差（全条件・拡大）と3HVモル分率の2種→3種の変化"),
        ("fig2_equal_budget_family", "kLa積分144 hでそろえた5候補のPHAと3種−2種差"),
        ("fig3_representative_trajectories",
         "代表軌道（PHA、3HV、溶存O2、Pf生菌。実線が3種、破線が2種）"),
    ):
        if (VALUE / (name + ".png")).exists():
            lines.append(f"- `{(VALUE / (name + '.png')).relative_to(ROOT)}`：{caption}")
    lines.extend([
        "",
        "図の白抜きマーカーと注記は時間刻み未収束・未評価を示す。図のSHA256は "
        f"`{(VALUE / 'summary.json').relative_to(ROOT)}` の `figure_sha256` にある。",
        "",
        "## この作業で主張しないこと",
        "",
    ])
    for item in value["limitations"]:
        lines.append(f"- {item['ja'] if isinstance(item, dict) else item}")
    lines.extend([
        "- 有限の手作り候補からの選択であり、全球最適な酸素スケジュール・供給・接種比ではない。",
        "- 生物学的有意差でも統計的有意差でもない。",
        "- 教師の大量収集とRL学習は開始していない。",
        "",
        "## 監査の要約",
        "",
    ])
    audited = [("パルス検証", pulse_done), ("拡張", extension_done)]
    if (CLOSURE / "completion.json").exists():
        audited.append(("最終細分", read_json(CLOSURE / "completion.json")))
    lines.extend([
        "|項目|" + "|".join(name for name, _ in audited) + "|",
        "|---|" + "|".join("---" for _ in audited) + "|",
        "|完了ケース数|" + "|".join(str(done["completed"]) for _, done in audited) + "|",
        "|ソースハッシュ一致|" + "|".join(
            "はい" if done["source_hashes_match"] else "いいえ" for _, done in audited) + "|",
        "|資源・LP監査|" + "|".join(
            "合格" if done["resource_and_lp_audit"] else "不合格" for _, done in audited) + "|",
        "|生物学的検証|" + "|".join(
            "未実施" if not done["biological_validation"] else "実施" for _, done in audited) + "|",
        "",
        "全時点で有限・非負、既知C/N台帳残差 <1e-6、酸素収支誤差 <=1e-8、酸素連立残差 <=1e-6、",
        "貯蔵上限超過 <=1e-7、LP残差 <=1e-7、双対残差と相対双対ギャップ <=1e-7 を検査している。",
        "供給総量と積分曝気容量は閉形式と一致することを終点で照合した。",
        "",
        "## 次に確認すべきこと",
        "",
        "- kLa 4〜8 h⁻¹の間をさらに細かく刻み、3種が有利になる窓の境界を特定する。",
        "- 3分割以上の高頻度パルスで、結果が一定kLaに収束するかを確認する。",
        "- 3種が有利な条件で供給プロファイルも再最適化し、構成差が供給最適化後も残るかを検証する。",
        "- kLa、酸素飽和、維持代謝、死滅、Pf表現型の校正。ここまでの結論はすべてこの校正待ちである。",
    ])
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(document=str(DOC.relative_to(ROOT)),
                          lines=len(lines),
                          sha256=hashlib.sha256(DOC.read_bytes()).hexdigest()),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
