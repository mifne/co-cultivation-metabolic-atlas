"""Build a dense, figure-heavy slide deck summarizing the 3 wet-lab
calibration protocols (Lcp/Rox activity, maintenance ATP, oxygen
partitioning) and the computational rationale behind them.

Reuses the same real result figures as
CALIBRATION_EXPERIMENT_RATIONALE_20260912.docx (no new simulations run
for this deck) and condenses the 3 protocol documents' methods/reference
tables. Built with python-pptx, 16:9 widescreen, high information density
per slide per the user's explicit request.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "docs" / "CALIBRATION_EXPERIMENTS_SLIDES_20260912.pptx"

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
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF4, 0xF6, 0xF8)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def new_deck():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)


def add_bg(slide, color=WHITE):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def add_title_bar(slide, title_text, kicker=None, color=NAVY):
    box = slide.shapes.add_textbox(Inches(0.4), Inches(0.18), SLIDE_W - Inches(0.8), Inches(0.95))
    tf = box.text_frame
    tf.word_wrap = True
    if kicker:
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = kicker
        r0.font.size = Pt(13)
        r0.font.bold = True
        r0.font.color.rgb = GREEN
        p1 = tf.add_paragraph()
    else:
        p1 = tf.paragraphs[0]
    r1 = p1.add_run()
    r1.text = title_text
    r1.font.size = Pt(26)
    r1.font.bold = True
    r1.font.color.rgb = color
    # thin rule under title
    line = slide.shapes.add_shape(1, Inches(0.4), Inches(1.12), SLIDE_W - Inches(0.8), Pt(2.2))
    line.fill.solid()
    line.fill.fore_color.rgb = DARK_BLUE
    line.line.fill.background()
    return box


def add_bullets(slide, items, left, top, width, height, size=14, color=RGBColor(0x22, 0x22, 0x22), bold_first=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            text, kwargs = item
        else:
            text, kwargs = item, {}
        p.level = kwargs.get("level", 0)
        run = p.add_run()
        run.text = ("• " if p.level == 0 else "‣ ") + text
        run.font.size = Pt(kwargs.get("size", size))
        run.font.color.rgb = kwargs.get("color", color)
        run.font.bold = kwargs.get("bold", False)
        p.space_after = Pt(4)
    return box


def add_note(slide, text, left, top, width, height, size=10.5, color=MUTED):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.italic = True
    run.font.color.rgb = color
    return box


def add_picture_fit(slide, path, left, top, max_w, max_h):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w = int(iw * ratio)
    h = int(ih * ratio)
    x = left + (max_w - w) // 2
    y = top + (max_h - h) // 2
    slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def add_table(slide, headers, rows, left, top, width, height, header_size=11, body_size=10.5, col_widths=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    gtable = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            gtable.columns[i].width = Emu(int(width * (cw / total)))
    for c, h in enumerate(headers):
        cell = gtable.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(header_size)
                r.font.bold = True
                r.font.color.rgb = WHITE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for ri, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = gtable.cell(ri, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if ri % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(body_size)
                    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return gtable


def add_page_number(slide, n):
    box = slide.shapes.add_textbox(SLIDE_W - Inches(0.7), SLIDE_H - Inches(0.45), Inches(0.5), Inches(0.35))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = str(n)
    run.font.size = Pt(10)
    run.font.color.rgb = MUTED


def main() -> None:
    prs = new_deck()
    page = 0

    # ==================== Slide 1: Title ====================
    s = blank_slide(prs)
    add_bg(s, NAVY)
    box = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(2.0))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "校正実験3プロトコル + 根拠データ"
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "ゴム分解速度定数 / 維持ATP要求量 / 酸素分配 — dFBAコンソーシアムモデル校正シリーズ"
    r2.font.size = Pt(18)
    r2.font.color.rgb = RGBColor(0xB9, 0xC8, 0xDA)
    p3 = tf.add_paragraph()
    p3.space_before = Pt(20)
    r3 = p3.add_run()
    r3.text = "2026年9月12日　天然ゴム分解・PHA変換コンソーシアム (OR16 / NS21 / P. freudenreichii)"
    r3.font.size = Pt(13)
    r3.font.color.rgb = RGBColor(0x8A, 0x9B, 0xB0)
    add_page_number(s, page)

    # ==================== Slide 2: Overview / priority table ====================
    page += 1
    s = blank_slide(prs)
    add_bg(s)
    add_title_bar(s, "全体像: なぜこの3パラメータが最優先か", kicker="1. OVERVIEW")
    add_bullets(
        s,
        [
            "THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx 第8章が、シミュレータ全ソース"
            "(dfba_simulator.py / physiology_dfba.py / audited_dfba.py / resolved_dfba.py 等)を"
            "監査し、3種共存の結論を反転させうる順に14パラメータをランク付け。",
            "上位3項目に対応する3本の湿式実験プロトコルを本サイクルで作成済み。",
            "共通する結論: 3種優位は条件依存(kLa中間域で+25.4%、kLa2で-51.6%)であり、"
            "この符号自体が未校正パラメータ次第で変わりうる。",
        ],
        Inches(0.5), Inches(1.35), Inches(12.3), Inches(1.7), size=14.5,
    )
    add_table(
        s,
        ["優先度", "パラメータ", "現在の仮定値", "影響", "対応プロトコル"],
        [
            ["①", "DEFAULT_POLYMER_RATES\n(ゴム分解速度定数)",
             "lcp_c5=2.5, roxb_c5=roxa_direct_c5=0.75,\nroxa_oligo_c30=0.25 mmol/gDW/h",
             "全炭素フローの起点。NS21のPHA原料供給と\nPfへの有機酸供給の上流",
             "LCP_ROX_ACTIVITY_ASSAY_\nPROTOCOL_20260911.docx"],
            ["②", "maintenance\n(維持ATP要求量)", "未設定(空dict)→実質0",
             "ゼロ増殖時の生存・飢餓耐性。\n長期共存の評価に直結",
             "MAINTENANCE_ATP_\nPROTOCOL_20260912.docx"],
            ["③", "polymer_oxygen_fraction /\nhalf_saturation", "0.25 / 0.01 mmol/L",
             "低DO条件でのゴム分解と呼吸の酸素競合。\n低kLaでの3種優位性の符号を左右",
             "OXYGEN_PARTITIONING_\nPROTOCOL_20260912.docx"],
            ["④〜⑦", "死滅率・NH4半飽和・PHA上限・kLa等", "(未着手)",
             "同章の残る4パラメータ(次スライド以降の対象外)", "未作成"],
        ],
        Inches(0.5), Inches(3.15), Inches(12.3), Inches(3.9),
        col_widths=[0.7, 2.6, 3.0, 3.6, 2.4], header_size=12, body_size=11,
    )
    add_page_number(s, page)

    # ==================== Slide 3: rationale ①③ limitation trace ====================
    page += 1
    s = blank_slide(prs)
    add_bg(s)
    add_title_bar(s, "根拠①③: OR16+NS21律速トレース — ゴムはほぼ分解されず、酸素が支配的",
                  kicker="2. なぜ①③が最優先か (計算結果)")
    add_bullets(
        s,
        [
            ("24h計算でゴム濃度 10.0 → 8.8362 g/L(11.6%のみ消費)。OR16 0.5→0.557 g/L、"
             "NS21 0.1→0.111 g/L と菌体はわずかに増加するのみ。", {}),
            ("ゴム分解段階はモデル上「allocated_oxygen_limited」と診断。", {"bold": True, "color": DARK_BLUE}),
            ("栄養素ドロップアウト感度ランキング(右図パネルd)で酸素(o2_e)が全アミノ酸を"
             "1桁以上引き離して最大: 単独介入で共通増殖容量+30.1%。", {}),
            ("しかし: 「酸素律速判定は固定されたpolymer_oxygen_fraction=0.25に依存し、"
             "実測DO/OURによる再推定が必要」— 診断自体が未校正パラメータの上に成立する"
             "循環構造。①(分解速度)と③(酸素配分)は計算だけでは切り分けられない。",
             {"color": RED}),
        ],
        Inches(0.5), Inches(1.35), Inches(6.3), Inches(4.8), size=13.5,
    )
    add_picture_fit(s, FIG_LIMITATION, Inches(6.95), Inches(1.35), Inches(5.9), Inches(5.3))
    add_note(
        s,
        "出典: results/or16_ns21_limitation_trace_20260902/ Figure_OR16_NS21_limitation_trace.png "
        "(a:菌体量 b:ゴム濃度とPHAプール c:共通増殖容量 d:栄養素ドロップアウト感度)。"
        "OR16+NS21の2種系、Pfは含まない。決定論的な1軌道の計算機内シミュレーション。",
        Inches(0.5), Inches(6.85), Inches(12.3), Inches(0.55),
    )
    add_page_number(s, page)

    # ==================== Slide 4: rationale ② maintenance sensitivity ====================
    page += 1
    s = blank_slide(prs)
    add_bg(s)
    add_title_bar(s, "根拠②: 維持ATPを10倍にするとPfの生存曲線が明確に変化",
                  kicker="2. なぜ②が最優先か (計算結果)")
    add_bullets(
        s,
        [
            ("PHYSIOLOGY_LONG_VALIDATION_20260908: baseline(維持ATP 0.1 mmol/g/h)と"
             "high_maintenance(1 mmol/g/h、10倍)を含む5条件を24h計算。", {}),
            ("Live Pf(左下パネル): baseline 0.030→約0.0245 g/Lに対し、high_maintenanceは"
             "約0.012 g/Lまでより急速に減少。", {"bold": True}),
            ("Live-cell PHA(左上パネル): high_maintenance条件はPHA蓄積の立ち上がりが"
             "約11h時点まで遅延(他条件は8h前後で立ち上がる)。", {}),
            ("維持ATP・基礎死滅率0.01 h⁻¹・最大飢餓死滅率0.1 h⁻¹はいずれも"
             "「未校正の検証用仮定」と明記(実測値ではない)。", {"color": MUTED}),
            ("図タイトル原文: \"Assumption and feed sensitivity; not evidence of "
             "biological stability\" — 感度分析であり生物学的実測の証拠ではない。",
             {"color": RED}),
        ],
        Inches(0.5), Inches(1.35), Inches(6.3), Inches(5.0), size=13.5,
    )
    add_picture_fit(s, FIG_SENSITIVITY, Inches(6.95), Inches(1.35), Inches(5.9), Inches(5.3))
    add_note(
        s,
        "出典: results/physiology_long_validation_20260908/sensitivity.png "
        "(baseline / high_maintenance / high_death / feed_stop / nh4_staged の5条件比較)。",
        Inches(0.5), Inches(6.85), Inches(12.3), Inches(0.5),
    )
    add_page_number(s, page)

    # ==================== Slide 5: rationale ③ oxygen non-monotonic ====================
    page += 1
    s = blank_slide(prs)
    add_bg(s)
    add_title_bar(s, "根拠③: 酸素供給条件に対する3種優位性は非単調",
                  kicker="2. なぜ③が最優先か (計算結果)")
    add_bullets(
        s,
        [
            ("OXYGEN_SCHEDULE_THIRD_SPECIES_20260910(dt収束確認済み): kLa 2一定で-51.6%、"
             "kLa中間域(4〜8)・曝気配分の悪いパルス条件で最大+25.4%、kLa 50一定で-1.7%。",
             {}),
            ("同一曝気予算(kLa積分144h)にそろえた5候補でも、2種PHA 0.346〜0.430 g/L"
             "(幅24%)、3種0.386〜0.445 g/L(幅15%)と広く分布 — 曝気の時間配分だけで"
             "結果が変わる。", {}),
            ("3種優位が最大の条件(2/10 6時間4分割)は、同時に2種PHAが最も低い条件でもある"
             "→ 3種は「上限を押し上げる」のではなく「悪条件での落ち込みを緩和する」方向"
             "に作用。", {"bold": True}),
            ("この非単調性は、酸素がゴム分解(オキシゲナーゼ)と細胞呼吸に競合配分される"
             "構造(polymer_oxygen_fraction/half_saturation)から生じている。",
             {"color": DARK_BLUE}),
        ],
        Inches(0.5), Inches(1.35), Inches(12.3), Inches(1.9), size=13.5,
    )
    add_picture_fit(s, FIG_OXY1, Inches(0.4), Inches(3.3), Inches(6.6), Inches(3.55))
    add_picture_fit(s, FIG_OXY2, Inches(7.1), Inches(3.3), Inches(5.8), Inches(3.55))
    add_note(
        s,
        "出典: results/third_species_value_20260910/fig1_third_species_axis.png, "
        "fig2_equal_budget_family.png。3HV差はdt未収束のkLa2一定・kLa10一定を除く。",
        Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.4), size=9.5,
    )
    add_page_number(s, page)

    # ==================== Slide 6: three protocols method table ====================
    page += 1
    s = blank_slide(prs)
    add_bg(s)
    add_title_bar(s, "3プロトコルの測定手法と主要文献値", kicker="3. 何をどう測るか")
    add_table(
        s,
        ["", "①Lcp/Rox活性", "②維持ATP(NGAM)", "③酸素分配"],
        [
            ["対象菌", "OR16(Lcp)、NS21(RoxA/RoxB)", "OR16・NS21・Pf",
             "OR16・NS21(オキシゲナーゼ+全菌呼吸)"],
            ["手法", "休止細胞アッセイ(Clark電極/蛍光センサーでO2消費速度計測)、"
                     "生成物HPLC定量(ODTD等)",
             "ケモスタット連続培養、Pirtプロット\nqs = μ/Ymax + ms",
             "既存Lcp/RoxアッセイをDO可変に拡張(Km(O2))+動的法拡張(Ks(O2))+"
             "基質有無OUR差分(分配実測)"],
            ["主要文献値",
             "Lcp1/2/3比活性4.02/1.17/0.22 U/mg(Gibu 2020)、RoxB 4.8 μmol/min/mg"
             "(BRENDA)",
             "S. coelicolor 7.5 mmol ATP/gDW/h(Coze 2013)、iMR558モデル "
             "1.4 mmol ATP/gDW/h(Pf自身)",
             "Lcp/RoxA/RoxBのKm(O2)は文献に前例なし(標準アッセイは酸素濃度を"
             "振っていない) — 真に新規の測定"],
            ["主要装備", "Clark型電極、天然ゴムラテックス/DPNR、50mM緩衝液系",
             "ケモスタット(0.5〜1L)、5希釈率(D=0.02〜0.15 h⁻¹)",
             "PyroScience光学式O2センサー(針型/センサースポット)推奨"
             "— 応答速度・非消費性で隔膜型電極/培養装置付属DO計より有利"],
        ],
        Inches(0.4), Inches(1.35), Inches(12.5), Inches(5.6),
        col_widths=[1.3, 3.7, 3.7, 3.8], header_size=13, body_size=11,
    )
    add_page_number(s, page)

    # ==================== Slide 7: samples & equipment ====================
    page += 1
    s = blank_slide(prs)
    add_bg(s)
    add_title_bar(s, "必要サンプル・装置(3実験共通)", kicker="4. 実施に向けて")
    add_table(
        s,
        ["カテゴリ", "サンプル/装備", "用途"],
        [
            ["生物", "OR16休止細胞懸濁液(ゴム前培養→集菌洗浄)", "①③のLcp活性・Km(O2)アッセイ"],
            ["生物", "NS21休止細胞懸濁液", "①③のRoxA/RoxB活性・Km(O2)アッセイ"],
            ["生物", "OR16/NS21/Pf 炭素源制限連続培養(各5希釈率)", "②Pirtプロット"],
            ["生物", "対数増殖期培養(ゴムあり/なし2条件)", "③OUR差分法(polymer_oxygen_fraction実測)"],
            ["基質", "天然ゴムラテックス、DPNR(Chaikumpollert法調製)、合成ポリイソプレン", "①③共通基質"],
            ["緩衝液", "50mM Tris-HCl(pH7、Lcp)、100mM リン酸K(pH7)+0.1%NP-40(Rox)", "①③"],
            ["校正液", "2% Na2SO3(0%点)、空気飽和緩衝液(100%点)", "全DO計測共通"],
            ["装備(新)", "PyroScience光学式O2センサー(針型/スポット+光ファイバー)", "①③、特に低DO域・小容量セル"],
            ["対照", "基質無添加ブランク、ゴム基質なし培養(条件A)", "内在性呼吸の差し引き"],
        ],
        Inches(0.4), Inches(1.35), Inches(12.5), Inches(5.6),
        col_widths=[1.6, 6.5, 4.4], header_size=13, body_size=11.5,
    )
    add_page_number(s, page)

    # ==================== Slide 8: next actions ====================
    page += 1
    s = blank_slide(prs)
    add_bg(s, NAVY)
    add_title_bar(s, "次のアクション", kicker="5. まとめ", color=WHITE)
    add_bullets(
        s,
        [
            ("① Lcp活性アッセイ(OR16)を先行実施 — 既存Gibu et al. 2020条件がそのまま"
             "使え、3実験中もっとも着手コストが低い。", {"color": WHITE}),
            ("③ PyroScience導入によりKm(O2)測定(小容量セル)とKs(O2)動的法"
             "(低DO域の応答遅れ懸念を解消)の両方が実施可能に。", {"color": WHITE}),
            ("② 維持ATPのPirtプロットはOR16・Pfの増殖が遅いため希釈率上限を下げて"
             "設計(D=0.02〜0.10 h⁻¹目安)。", {"color": WHITE}),
            ("④〜⑦(死滅率・NH4半飽和・PHA上限・kLa)は同じ優先順位リストの"
             "残り項目 — 本3プロトコル完了後の次サイクル対象。", {"color": WHITE}),
            ("全ての結論は「未校正パラメータ前提の計算」である点を実験完了まで"
             "保留として扱う(THREE_SPECIES_COEXISTENCE_SUMMARY全体の方針)。",
             {"color": RGBColor(0xFF, 0xC1, 0x9E), "bold": True}),
        ],
        Inches(0.6), Inches(1.5), Inches(12.0), Inches(4.8), size=16,
    )
    add_note(
        s,
        "参照文書: LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docx / "
        "MAINTENANCE_ATP_PROTOCOL_20260912.docx / OXYGEN_PARTITIONING_PROTOCOL_20260912.docx / "
        "CALIBRATION_EXPERIMENT_RATIONALE_20260912.docx",
        Inches(0.6), Inches(6.7), Inches(12.0), Inches(0.6), color=RGBColor(0x8A, 0x9B, 0xB0),
    )
    add_page_number(s, page)

    prs.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
