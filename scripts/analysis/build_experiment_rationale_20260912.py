"""Build a Word document laying out the computational (dFBA/GEM) evidence
that motivated the three wet-lab calibration protocols already produced
this cycle:
- LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docx (priority (1))
- MAINTENANCE_ATP_PROTOCOL_20260912.docx (priority (2))
- OXYGEN_PARTITIONING_PROTOCOL_20260912.docx (priority (3))

Every figure embedded here is a real, already-generated simulation output
from this project's results/ directory (not fabricated for this report):
- results/or16_ns21_limitation_trace_20260902/
  Figure_OR16_NS21_limitation_trace.png -- rubber depletion pace and a
  nutrient-dropout sensitivity ranking where o2_e dominates.
- results/physiology_long_validation_20260908/sensitivity.png -- Pf
  biomass/PHA trajectories under baseline vs. 10x maintenance ATP.
- results/third_species_value_20260910/fig{1,2,3}_*.png -- the
  non-monotonic oxygen-supply dependence of the 3-species vs 2-species
  PHA comparison (already embedded in THREE_SPECIES_COEXISTENCE_SUMMARY
  _20260911.docx; reused here with the same source path cited).

The priority ranking and the full 14-parameter calibration table are
taken verbatim (retyped, not re-derived) from
THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx section 8.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "docs" / "CALIBRATION_EXPERIMENT_RATIONALE_20260912.docx"

FIG_LIMITATION = ROOT / "results/or16_ns21_limitation_trace_20260902/Figure_OR16_NS21_limitation_trace.png"
FIG_SENSITIVITY = ROOT / "results/physiology_long_validation_20260908/sensitivity.png"
FIG_OXY1 = ROOT / "results/third_species_value_20260910/fig1_third_species_axis.png"
FIG_OXY2 = ROOT / "results/third_species_value_20260910/fig2_equal_budget_family.png"
FIG_OXY3 = ROOT / "results/third_species_value_20260910/fig3_representative_trajectories.png"

NAVY = RGBColor(0x0B, 0x25, 0x45)
DARK_BLUE = RGBColor(0x1F, 0x4D, 0x78)
MUTED = RGBColor(0x5B, 0x65, 0x73)
RED = RGBColor(0x9B, 0x1C, 0x1C)
GREEN = RGBColor(0x1B, 0x6E, 0x4F)


def set_cell_shading(cell, hex_color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.font.color.rgb = NAVY if level == 1 else DARK_BLUE
    return h


def add_note(doc, text, color=MUTED):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9.5)
    run.font.color.rgb = color
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def add_quote(doc, text, source):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(f'"{text}"')
    run.italic = True
    run.font.size = Pt(9.5)
    p2 = doc.add_paragraph()
    p2.paragraph_format.left_indent = Cm(1)
    run2 = p2.add_run(f"— {source}")
    run2.font.size = Pt(8.5)
    run2.font.color.rgb = MUTED


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(hdr[i], "1F4D78")
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table


def add_figure(doc, path: Path, caption: str, width_cm: float = 15.5):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED


def main() -> None:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Yu Gothic"
    normal.font.size = Pt(10.5)

    title = doc.add_heading(level=0)
    r = title.add_run("3つのWet実験の根拠データ")
    r.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run(
        "ゴム分解速度定数・維持ATP要求量・酸素分配の3プロトコルを"
        "着手した計算機内(dFBA/GEM)エビデンス — 2026年9月12日"
    )
    r2.italic = True
    r2.font.color.rgb = MUTED

    add_note(
        doc,
        "本書に掲載する図は全て、このプロジェクトのresults/配下に既に存在する"
        "実際のシミュレーション出力であり、本書のために新規作成・加工したものは"
        "ない(出典パスを各図に明記)。3つのプロトコル(LCP_ROX_ACTIVITY_ASSAY_"
        "PROTOCOL_20260911.docx、MAINTENANCE_ATP_PROTOCOL_20260912.docx、"
        "OXYGEN_PARTITIONING_PROTOCOL_20260912.docx)がなぜこの3パラメータを"
        "対象に選ばれたのか、その根拠となった計算結果を示す。",
        color=GREEN,
    )

    # ==================== 1. Overview ====================
    add_heading(doc, "1. 全体像: 優先順位とその根拠", level=1)
    doc.add_paragraph(
        "docs/THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx 第8章は、"
        "シミュレータのソースコード(dfba_simulator.py, physiology_dfba.py, "
        "audited_dfba.py, resolved_dfba.py, cultivation_numerics.py, "
        "b12_evidence.py, one_l_jar.py)を精査し、3種共存の結論に影響しうる"
        "未校正パラメータを「結論を反転させうる可能性が高い順」に7項目"
        "ランク付けした。上位3項目が今回作成した3プロトコルに対応する。"
    )
    add_table(
        doc,
        ["優先度", "パラメータ", "現在の仮定値", "影響", "対応プロトコル"],
        [
            ["①", "DEFAULT_POLYMER_RATES\n(ゴム分解速度定数)",
             "lcp_c5=2.5, roxb_c5=roxa_direct_c5=0.75,\nroxa_oligo_c30=0.25 (mmol/gDW/h)",
             "全炭素フローの起点。OR16/NS21の分解速度バランスが\nNS21へのPHA原料供給とPfへの有機酸供給を決める",
             "LCP_ROX_ACTIVITY_ASSAY_\nPROTOCOL_20260911.docx"],
            ["②", "maintenance\n(維持ATP要求量)", "未設定(空dict)→実質0",
             "ゼロ増殖時の生存・飢餓耐性。長期共存の評価に直結",
             "MAINTENANCE_ATP_\nPROTOCOL_20260912.docx"],
            ["③", "polymer_oxygen_fraction /\nhalf_saturation",
             "0.25 / 0.01 mmol/L",
             "低DO条件でのゴム分解と呼吸の酸素競合。\n低kLaでの3種優位性の符号を左右しうる",
             "OXYGEN_PARTITIONING_\nPROTOCOL_20260912.docx"],
            ["④〜⑦", "死滅率・NH4半飽和・\nPHA上限・kLa等", "(未着手)",
             "(それぞれの節を参照、THREE_SPECIES_COEXISTENCE_\nSUMMARY 第8章表)", "未作成"],
        ],
        col_widths=[1.3, 3.2, 3.8, 5.5, 3.7],
    )
    add_note(
        doc,
        "出典: THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx 第8章「実験で"
        "校正すべき主要パラメータ」の表(全14行のうち上位を抜粋)。値・定義箇所は"
        "コードと照合済みとされている。",
    )

    # ==================== 2. Priority 1 ====================
    add_heading(
        doc, "2. ①ゴム分解速度定数(Lcp/Rox) — 全炭素フローの起点", level=1
    )
    doc.add_paragraph(
        "OR16単独系(NS21との2種系)のin-silico培養トレースでは、24時間の"
        "計算でゴムがほとんど分解されていない。"
    )
    add_table(
        doc,
        ["項目", "初期値", "24h後"],
        [
            ["ゴム濃度", "10.0000 g/L", "8.8362 g/L(11.6%のみ消費)"],
            ["OR16菌体", "0.5 g/L", "0.55713 g/L"],
            ["NS21菌体", "0.1 g/L", "0.11142 g/L"],
            ["PHAモデルプール", "0 mmol/L", "0.011026 mmol/L"],
        ],
        col_widths=[4, 5, 7],
    )
    add_figure(
        doc,
        FIG_LIMITATION,
        "図1: OR16+NS21二種系の律速トレース(パネルa: 菌体量、"
        "パネルb: ゴム濃度とPHAモデルプール、パネルc: 共通増殖容量の低下、"
        "パネルd: 栄養素ドロップアウト感度ランキング)。"
        "出典: results/or16_ns21_limitation_trace_20260902/"
        "Figure_OR16_NS21_limitation_trace.png",
    )
    doc.add_paragraph(
        "この計算のゴム分解段階は、モデル上「allocated_oxygen_limited」"
        "(配分された酸素によって律速)と診断されている。"
    )
    add_quote(
        doc,
        "ゴム分解段階: allocated_oxygen_limited",
        "results/or16_ns21_limitation_trace_20260902/"
        "OR16_NS21_limitation_trace_report.md",
    )
    doc.add_paragraph(
        "つまり現行モデルでは、ゴム分解の遅さが「酵素の分解速度定数"
        "(DEFAULT_POLYMER_RATES)自体が低いから」なのか「酸素配分"
        "(polymer_oxygen_fraction)が制約しているから」なのかを、"
        "計算結果だけから切り分けられない。この2つのパラメータがいずれも"
        "実測値ではなく仮定値であるため、律速要因の特定そのものが"
        "実験データを必要としている。THREE_SPECIES_COEXISTENCE_SUMMARY "
        "第8章が①②を優先度1・3位に置いているのは、この構造的な理由による"
        "(「全炭素フローの起点」であり、NS21のPHA原料供給・Pfへの有機酸供給の"
        "上流に位置するため、ここが不正確だと下流の全ての結論が影響を受ける)。"
    )

    # ==================== 3. Priority 2 ====================
    add_heading(
        doc, "3. ②維持ATP要求量(maintenance) — 10倍の違いでPf生存曲線が"
        "大きく変わる", level=1
    )
    doc.add_paragraph(
        "PHYSIOLOGY_LONG_VALIDATION_20260908(2026-09-08)は、維持ATP・"
        "死滅率を明示的に仮定した検証用計算(いずれも「未校正の検証用仮定」と"
        "明記)を行い、baseline(維持ATP 0.1 mmol/g/h)に対しbaseline_high_"
        "maintenance(維持ATP 1 mmol/g/h、10倍)を比較した。"
    )
    add_quote(
        doc,
        "維持ATPは0.1 mmol/g/h、基礎死滅率0.01 h⁻¹、最大飢餓死滅率0.1 h⁻¹。"
        "これらは未校正の検証用仮定である。",
        "docs/PHYSIOLOGY_LONG_VALIDATION_20260908.md",
    )
    add_figure(
        doc,
        FIG_SENSITIVITY,
        "図2: baseline(維持ATP 0.1)とbaseline_high_maintenance(維持ATP 1、"
        "10倍)を含む5条件の24時間比較(生菌中PHA、死菌中PHA、"
        "P. freudenreichii生菌量、NH4濃度)。出典: "
        "results/physiology_long_validation_20260908/sensitivity.png",
    )
    doc.add_paragraph(
        "左下パネル(Live Pf, g/L)に着目すると、baseline(青)は24hで"
        "0.030→約0.0245 g/Lまで緩やかに減少するのに対し、"
        "baseline_high_maintenance(橙)は同じ24hで約0.012 g/Lまで、"
        "より急速に減少している。あわせて左上パネル(Live-cell PHA)では、"
        "高維持ATP条件はPHA蓄積の立ち上がりが他条件より遅れている"
        "(約11h時点まで蓄積がほぼゼロ)。維持ATPという1パラメータを"
        "10倍にしただけで、Pfの生存軌道とPHA蓄積タイミングの両方が"
        "明確に変化することが、この計算で既に示されている。"
    )
    add_note(
        doc,
        "この図のタイトルは元図で「Assumption and feed sensitivity; not "
        "evidence of biological stability」(仮定と供給への感度であり、"
        "生物学的な安定性の証拠ではない)と明記されている。つまりこの結果は"
        "「維持ATPが実際にこの値である」ことを示すものではなく、「この"
        "パラメータが未校正のままでは、Pfの生存・PHA蓄積タイミングに関する"
        "結論が仮定値次第で大きく変わる」ことを示す感度分析である。",
    )

    # ==================== 4. Priority 3 ====================
    add_heading(
        doc, "4. ③酸素分配(polymer_oxygen_fraction/half_saturation) — "
        "非単調な依存性と栄養素感度の両面から", level=1
    )
    doc.add_paragraph(
        "根拠は2種類ある。1つ目は、先の律速トレース(図1)のパネルdで、"
        "栄養素ドロップアウト感度ランキングにおいて酸素(o2_e)が突出して"
        "最大の影響を持つことが示されている点である。"
    )
    add_quote(
        doc,
        "最大の単独介入: o2_e(共通増殖容量 30.1%増)",
        "results/or16_ns21_limitation_trace_20260902/"
        "OR16_NS21_limitation_trace_report.md",
    )
    add_quote(
        doc,
        "酸素律速判定は固定されたpolymer_oxygen_fraction=0.25に依存し、"
        "実測DO/OURによる再推定が必要である。",
        "同上(解釈上の制約)",
    )
    doc.add_paragraph(
        "この最後の一文が本質的である。「酸素が律速している」というモデルの"
        "診断結果自体が、実測されていないpolymer_oxygen_fraction=0.25という"
        "仮定に依存しており、循環参照になっている。これを断ち切るには"
        "実測データが必要、というのがOXYGEN_PARTITIONING_PROTOCOL_"
        "20260912.docxの直接の動機である。"
    )
    doc.add_paragraph(
        "2つ目の根拠は、OXYGEN_SCHEDULE_THIRD_SPECIES_20260910"
        "(2026-09-10、dt収束確認済み)による、酸素供給条件に対する"
        "3種−2種のPHA差の非単調な挙動である。"
    )
    add_figure(
        doc,
        FIG_OXY1,
        "図3: 酸素条件(平均kLa)に対する3種−2種のPHA差(%)。"
        "kLa 2一定で-51.6%、kLa中間域(4〜8)や曝気配分の悪いパルス条件で"
        "最大+25.4%、kLa 50一定では-1.7%(パネルaの範囲は本文参照)。"
        "出典: results/third_species_value_20260910/"
        "fig1_third_species_axis.png",
    )
    add_figure(
        doc,
        FIG_OXY2,
        "図4: 同一曝気予算(kLa積分144 h)にそろえた5候補の比較。"
        "2種PHAは0.346〜0.430 g/L(幅24.1%)、3種は0.386〜0.445 g/L"
        "(幅15.3%)に分布し、3種−2種差が最大になる条件(2/10 6時間4分割、"
        "+25.4%)は同時に2種PHAが最も低い条件でもある。"
        "出典: results/third_species_value_20260910/"
        "fig2_equal_budget_family.png",
    )
    add_figure(
        doc,
        FIG_OXY3,
        "図5: 代表軌道(生菌中PHA、3HVモル分率、溶存O2、"
        "P. freudenreichii生菌量。実線=3種、破線=2種)。"
        "出典: results/third_species_value_20260910/"
        "fig3_representative_trajectories.png",
    )
    doc.add_paragraph(
        "この非単調性(中間的な酸素供給域でのみ3種が有利、極端に低い/高い"
        "供給では不利または差が消える)は、酸素が「ゴム分解」と「細胞呼吸」の"
        "間でどう配分されるかという、まさにpolymer_oxygen_fractionと"
        "polymer_oxygen_half_saturationが表現しようとしている競合構造から"
        "生まれている。これらのパラメータの実測値次第で、非単調性の形"
        "(有利になる酸素供給域の位置・幅)自体が変わりうる。"
    )

    # ==================== 5. Caveats ====================
    add_heading(doc, "5. 全体を通じた留保事項", level=1)
    add_bullets(
        doc,
        [
            "本書で示した全ての結果は、湿式実験ではなく決定論的な1軌道の"
            "計算機内(dFBA/GEM)シミュレーションである。反復・統計検定は"
            "行われていない。",
            "図2(維持ATP感度)・図1(律速トレース)は、いずれも"
            "「未校正の検証用仮定」の下での感度分析であり、示された数値"
            "(10.6%の生存量低下、30.1%の成長容量増加等)は、実際の生物学的"
            "な値を予測するものではない。「この仮定次第で結論が変わる」"
            "ことを示す証拠として扱うこと。",
            "図3〜5(酸素条件依存性)は時間刻み収束を確認済みだが、"
            "維持代謝・死滅率・PHA再利用・B12依存性・Pfの酸素表現型・kLa"
            "など、他の未校正パラメータの影響は分離されていない"
            "(THREE_SPECIES_COEXISTENCE_SUMMARY 6.2節)。",
            "①のゴム分解律速トレース(図1)はOR16+NS21の2種系であり、"
            "Pfを含む3種系での再計算ではない。",
        ],
    )

    # ==================== 6. References ====================
    add_heading(doc, "6. 参照元", level=1)
    add_bullets(
        doc,
        [
            "docs/THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx"
            "(第8章: 優先順位と全14パラメータの校正表)",
            "results/or16_ns21_limitation_trace_20260902/"
            "OR16_NS21_limitation_trace_report.md および "
            "Figure_OR16_NS21_limitation_trace.png",
            "docs/PHYSIOLOGY_LONG_VALIDATION_20260908.md および "
            "results/physiology_long_validation_20260908/sensitivity.png",
            "docs/OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md および "
            "results/third_species_value_20260910/fig1〜3_*.png",
            "docs/LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docx",
            "docs/MAINTENANCE_ATP_PROTOCOL_20260912.docx",
            "docs/OXYGEN_PARTITIONING_PROTOCOL_20260912.docx",
        ],
    )
    add_note(
        doc,
        "本書はDeepSeekへの文献検索委任を伴わない(プロジェクト内部の既存"
        "計算結果の再整理であり、外部文献調査は不要と判断した)。図・数値は"
        "全て上記の既存ファイルから直接転記・引用しており、本書のために"
        "新規のシミュレーションは実行していない。",
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
