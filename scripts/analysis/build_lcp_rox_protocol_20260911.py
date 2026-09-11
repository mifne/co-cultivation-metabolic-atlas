"""Build the literature-grounded wet-lab protocol for measuring the
Lcp (OR16) and RoxA/RoxB (NS21) rubber-degradation rate constants.

v3 (true final): a third round of DeepSeek V4 Flash web-search deep-dives
resolved every open item from v2's "next actions" list:
- Röther, Birke & Jendrossek (2017, Bio-protocol) full text found (PMC
  open access) -- exact buffer recipes, substrate prep, O2 electrode
  calibration, HPLC conditions, and the unit definition are now in the
  protocol verbatim.
- Birke, Röther & Jendrossek (2018) full text was not directly
  accessible, but BRENDA's enzyme database entry (itself sourced from
  that paper) supplied RoxB_NS21's measured specific activity and
  molecular weight.
- Chaikumpollert et al. (2012) was identified and its DPNR preparation
  protocol found in full, supplemented by two follow-up papers from the
  same group with centrifugation-speed/time details the 2012 paper's
  excerpt lacked.
- The OR16/NS21 gene-locus-tag discrepancy flagged in v2 was resolved by
  auditing the actual SBML model directly (not literature): the model's
  R_LCP reaction already uses the gene IDs matching Gibu et al. (2020);
  the docs/MODEL_INDEX.md entry describing different IDs was simply
  wrong and has been corrected there.

A genuine unit-definition inconsistency was found across the literature
(Röther et al. 2017 defines 1 U = 1 nmol O2/min; other papers in the same
research group report activities in umol/min/mg) and is flagged rather
than silently resolved, since it is plausibly explained by different
assay scales (500 ul micro-assay vs. classical larger-volume Clark
electrode) but was not confirmed against full text.

Real URLs/DOIs and direct quotes are in tmp/deepseek_lcp_literature.md,
tmp/deepseek_rox_literature.md, tmp/deepseek_followup1-3.md, and
tmp/deepseek_deepdive_rother.md / _birke.md / _dpnr.md. Claude selected,
translated, and adapted the quoted protocols; every borrowed number is
attributed to its source.
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
OUT_PATH = ROOT / "docs" / "LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docx"

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


def add_steps(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Number")
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


def main() -> None:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Yu Gothic"
    normal.font.size = Pt(10.5)

    title = doc.add_heading(level=0)
    r = title.add_run("ゴム分解酵素(Lcp / RoxA・RoxB)活性測定プロトコル")
    r.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run(
        "OR16(Lcp)・NS21(RoxA/RoxB)のゴム分解速度定数を実測するための実験計画書 "
        "— 第3版(真の最終版) 2026年9月12日"
    )
    r2.italic = True
    r2.font.color.rgb = MUTED

    add_note(
        doc,
        "第3版での変更点: 第2版の「次のアクション」で未確認だった3文献を深掘りし、"
        "全て解決した。(1) Röther et al. (2017, Bio-protocol)の全文を取得し、"
        "緩衝液の正確な調製法・酸素電極の校正法・HPLC条件・計算式を実プロトコルとして"
        "反映(3.2節)。(2) Birke et al. (2018)の全文は未入手だが、BRENDA酵素データベース"
        "からRoxB_NS21の実測比活性・分子量を取得。(3) Chaikumpollert et al. (2012)の"
        "DPNR調製プロトコルを特定・詳細化(3.3節、新設)。(4) MODEL_INDEX.mdの遺伝子座"
        "タグの食い違いは、実際のSBMLモデルを監査して解決した(モデル自体はGibu et al. "
        "の値で正しく実装されており、ドキュメント記載が誤っていただけ。修正済み)。",
        color=GREEN,
    )

    add_heading(doc, "1. 目的と背景", level=1)
    doc.add_paragraph(
        "dFBAシミュレータ(dfba_simulator.py, DEFAULT_POLYMER_RATES)は、OR16のLcp経路と"
        "NS21のRoxA/RoxB経路のゴム分解速度を、以下の未校正の仮定値で表現している。"
    )
    add_table(
        doc,
        ["パラメータ", "現在の仮定値", "対応する酵素"],
        [
            ["lcp_c5", "2.5 mmol gDW⁻¹ h⁻¹(C5換算)", "OR16のLcp(latex-clearing protein、"
             "実際はLcp1/Lcp2/Lcp3の3パラログの合算として扱われている)"],
            ["roxb_c5", "0.75 mmol gDW⁻¹ h⁻¹", "NS21のRoxB(エンド型)"],
            ["roxa_direct_c5", "0.75 mmol gDW⁻¹ h⁻¹", "NS21のRoxA(エキソ型、ゴムへの直接作用)"],
            ["roxa_oligo_c30", "0.25 mmol gDW⁻¹ h⁻¹(C30換算)", "NS21のRoxA(オリゴマーからODTDへの変換)"],
        ],
        col_widths=[3.5, 5.5, 6.5],
    )
    doc.add_paragraph(
        "本書は、これらの値を実測で置き換えるための実験計画を、先行研究の直接引用に基づいて"
        "作成する。文献はDeepSeek V4 Flash(OpenRouterのWeb検索プラグイン, "
        "model=deepseek/deepseek-v4-flash:online)で2回に分けて実際に検索し、実在するURLと"
        "引用文を確認した上で採用している。"
    )

    add_note(
        doc,
        "【重要な発見・解決済み】 OR16のLcp遺伝子を直接特徴づけた論文(Gibu et al. 2020, "
        "Appl Microbiol Biotechnol 104:7367-7376)が見つかった。この論文が報告する遺伝子座"
        "タグ(lcp1=ACTI_59630, lcp2=ACTI_59640, lcp3=ACTI_69520)は、本プロジェクトの"
        "docs/MODEL_INDEX.md旧記載(ACTI_28730, ACTI_28740, ACTI_37800)と一致しなかった。"
        "実際のSBMLモデルを監査した結果、モデルの反応R_LCPは既にGibu et al.の遺伝子IDを"
        "使って正しく実装されており、旧ACTI_28730等は反応と接続されていない孤立エントリ"
        "だった。ドキュメント記載のみが誤っていたため、MODEL_INDEX.mdを訂正済み。",
        color=GREEN,
    )
    add_note(
        doc,
        "NS21のRoxA/RoxBについても直接の先行研究(Birke, Röther & Jendrossek 2018)が"
        "存在する。本文全文は未入手だが、BRENDA酵素データベース(下記3.2節)から"
        "RoxB_NS21の実測比活性・分子量を取得できた。RoxA_NS21固有の数値、Km/Vmax、"
        "精製収率は依然未確認であり、全文入手が引き続き必要である。",
        color=RED,
    )

    # ==================== 2. Literature review ====================
    add_heading(doc, "2. 先行研究のレビュー(直接引用)", level=1)

    add_heading(doc, "2.1 Lcp(latex-clearing protein) — OR16そのものの実測データあり", level=2)
    doc.add_paragraph(
        "OR16は3つのlcp遺伝子(lcp1, lcp2, lcp3)を持ち、それぞれをE. coliで組換え発現・"
        "精製し、酸素消費アッセイで個別に活性測定した研究が存在する。"
    )
    add_quote(
        doc,
        "A tBLASTn homology search of the genome sequence of OR16 was performed using the "
        "amino acid sequence of Lcp (AAR25849) of Streptomyces sp. strain K30 as the "
        "query, and three homologous genes, ACTI_59630, ACTI_59640, and ACTI_69520, were "
        "identified as the most closely related gene sequences; these genes were "
        "designated lcp1, lcp2, and lcp3, respectively.",
        "Gibu et al. (2020), Applied Microbiology and Biotechnology 104:7367-7376, "
        "https://doi.org/10.1007/s00253-020-10700-1",
    )
    add_quote(
        doc,
        "The activities of Lcp1, Lcp2, and Lcp3 were assayed by measuring the "
        "substrate-dependent oxygen consumption rate. Each 2-ml assay mixture contained "
        "50 mM Tris-HCl buffer, NR (final concentration was 0.2%), and purified enzyme "
        "(25 μg of protein). The reaction mixture was incubated at 30 °C, and the oxygen "
        "consumption rate was determined with an oxygen electrode (Dual Digital Model 20; "
        "Rank Brothers Ltd., Cambridge, England). One unit of enzyme activity was defined "
        "as the amount of activity that resulted in consumption of 1 μmol of O2 per 1 min.",
        "Gibu et al. (2020), 同上",
    )
    add_quote(
        doc,
        "The optimal temperature and optimal pH for the oxygen consumption activity of "
        "Lcp1-his, Lcp2-his, and Lcp3-his with NR latex were determined to be 30 °C and "
        "7.0, 35 °C and 7.5, and 30 °C and 6.5, respectively. The specific activities of "
        "Lcp1-his, Lcp2-his, and Lcp3-his were 4.02 ± 0.65 (30 °C, pH 7.0), 1.17 ± 0.07 "
        "(35 °C, pH 7.5), and 0.22 ± 0.01 (30 °C, pH 6.5) U/mg protein, respectively.",
        "Gibu et al. (2020), 同上",
    )
    doc.add_paragraph("整理すると、OR16の3パラログの実測比活性は次の通り:")
    add_table(
        doc,
        ["酵素", "遺伝子座タグ", "比活性(U/mg)", "至適温度", "至適pH"],
        [
            ["Lcp1", "ACTI_59630", "4.02 ± 0.65", "30 °C", "7.0"],
            ["Lcp2", "ACTI_59640", "1.17 ± 0.07", "35 °C", "7.5"],
            ["Lcp3", "ACTI_69520", "0.22 ± 0.01", "30 °C", "6.5"],
        ],
        col_widths=[2.5, 4.0, 4.0, 3.0, 2.5],
    )
    add_quote(
        doc,
        "The activity of Lcp1 was approximately 7 times and 18 times higher than those of "
        "Lcp2 and Lcp3, respectively, suggesting that Lcp1 is mainly involved in rubber "
        "degradation in strain OR16. Unlike the enzymatic activities of the Lcp enzymes, "
        "the transcription level of lcp3 is significantly higher than those of lcp1 and "
        "lcp2... the role of lcp3 is thought to also be important for the rubber "
        "degradation.",
        "Gibu et al. (2020), 同上",
    )
    add_quote(
        doc,
        "the transcription levels of lcp1, lcp2, and lcp3 in cells grown with NR latex "
        "were 22.2-fold, 17.1-fold, and 335-fold higher, respectively, than those in cells "
        "grown without NR latex (P < 0.05 by Student's t test).",
        "Gibu et al. (2020), 同上",
    )
    add_note(
        doc,
        "示唆: 活性(比活性)ではLcp1が支配的だが、NR誘導による転写誘導倍率はlcp3が圧倒的に"
        "大きい(335倍)。酵素量(発現量)と比活性の両方が細胞あたりの実効速度を決めるため、"
        "「どのLcpが実際のゴム分解を担うか」は活性データだけでは断定できない。dFBAモデルの"
        "lcp_c5を1つの合算値とするか、3パラログを別々に扱うかは、この転写誘導データを"
        "踏まえて再検討する価値がある。",
    )
    add_quote(
        doc,
        "The enzymatic activities are comparable with those of other reported Lcps in "
        "strain K30 (4.6 U/mg), strain VH2 (1.3 U/mg), and R. rhodochrous RPK1 (3.1 U/mg).",
        "Gibu et al. (2020), 同上(他菌株の値の引用)",
    )
    doc.add_paragraph(
        "OR16培養条件(同論文より): PYM培地(0.5% Bactopeptone, 0.3% yeast extract, "
        "0.1% MgSO₄·7H₂O, pH 7.0)、またはW最小塩培地+10 mM コハク酸ナトリウム、30°C。"
        "ゴム分解の確認には脱タンパクNR(deproteinized NR)を最終濃度0.4%(v/v)で"
        "オーバーレイ寒天培地に用いている。"
    )
    add_quote(
        doc,
        "Growth on NR or IR was examined using rubber overlay agar plates. To prepare the "
        "rubber overlay agar plates, W agar medium containing deproteinized NR "
        "(Chaikumpollert et al. 2012) or synthetic isoprene rubber at a final "
        "concentration of 0.4% (v/v) was overlaid to form a thin layer on a solid medium.",
        "Gibu et al. (2020), 同上",
    )
    add_note(
        doc,
        "DPNR(脱タンパク天然ゴムラテックス)の調製プロトコルの出典が判明した: "
        "Chaikumpollert et al. (2012)。本書のv1では「見つからなかった」としていたが、"
        "Gibu et al. (2020)の引用からこの文献を特定できた。全文入手して手順を確認すること。",
        color=GREEN,
    )

    add_heading(doc, "2.2 RoxA・RoxB — NS21そのものを対象とした先行研究(訂正版)", level=2)
    add_note(
        doc,
        "訂正: 本書v1では、以下の論文を「Kasai et al. (2017)」として引用していたが、"
        "実際の著者は Birke, J., Röther, W. & Jendrossek, D. (2018) である。"
        "Kasai et al. (2017)はNS21のlatA遺伝子を同定した別の、より早い論文であり、"
        "タンパク質の精製・生化学的特徴づけは行っていない。",
        color=RED,
    )
    doc.add_paragraph(
        "RoxA・RoxBはグラム陰性菌(Steroidobacter cummioxidans 35Y、旧Xanthomonas sp. "
        "35Y、および Rhizobacter gummiphilus NS21)が持つジヘム依存性ジオキシゲナーゼで"
        "ある。"
    )
    add_quote(
        doc,
        "No homologue for an Lcp protein but homologues for a putative RoxA and a RoxB "
        "protein (the latter identical to a previously postulated LatA-denominated rubber "
        "cleaving enzyme) were identified in the genome of strain NS21. The roxANS21 and "
        "roxBNS21 genes were separately expressed in a ΔroxA35Y/ΔroxB35Y background of S. "
        "cummioxidans 35Y and restored the ability of the mutant to produce "
        "oligoisoprenoids. The RoxANS21 and RoxBNS21 proteins were each purified and "
        "biochemically characterised.",
        "Birke, Röther & Jendrossek (2018), Applied Microbiology and Biotechnology "
        "102:10245-10257, https://doi.org/10.1007/s00253-018-9341-6",
    )
    add_quote(
        doc,
        "A latA gene was independently identified to be involved in utilisation and "
        "cleavage of rubber, but the corresponding LatA protein was not purified and "
        "characterised (Kasai et al. 2017)... it became clear that LatA represents a RoxB "
        "homologue and that R. gummiphilus NS21 harboured a second rubber oxygenase, "
        "RoxANS21. The RoxANS21 and RoxBNS21 (=LatANS21) proteins of R. gummiphilus NS21 "
        "are highly similar to RoxA35Y and RoxB35Y of S. cummioxidans 35Y... and also "
        "cleaved polyisoprene synergistically to ODTD as end product.",
        "Jendrossek & Birke (2019年頃の総説 \"Rubber oxygenases\"), "
        "Applied Microbiology and Biotechnology, "
        "https://doi.org/10.1007/s00253-018-9453-z",
    )
    add_quote(
        doc,
        "Notably, when the isolated rubber cleavage products generated by RoxBNS21 were "
        "used as substrate for RoxANS21, a subsequent HPLC analysis of the products "
        "revealed the appearance of a high ODTD peak. This clearly demonstrated that "
        "RoxANS21 is able to use the products of RoxBNS21 as substrate and confirmed the "
        "synergistic effect of the two enzymes.",
        "Birke, Röther & Jendrossek (2018), 同上(総説内での引用)",
    )
    add_note(
        doc,
        "RoxA_NS21・RoxB_NS21固有のKm・Vmax・比活性・精製収率は、今回のWeb検索(抄録・"
        "本文抜粋レベル)からは取得できなかった。Birke, Röther & Jendrossek (2018)の全文"
        "(表・図)を入手して確認する必要がある。",
    )
    doc.add_paragraph("参考として、模式生物S. cummioxidans 35Yの数値が報告されている:")
    add_table(
        doc,
        ["酵素", "分子量", "比活性", "特記事項"],
        [
            ["RoxA(35Y)", "70 kDa(ジヘム蛋白)", "2.6 U/mg(37°C)", "エキソ型、主生成物ODTD(C15)、"
             "酸化状態で酸素分子が結合済み"],
            ["RoxB(35Y)", "約70 kDa", "≈6 U/mg", "エンド型、C20以上の混合オリゴマー、"
             "as-isolated状態では酸素分子は結合せず"],
        ],
        col_widths=[3.0, 4.0, 3.0, 6.0],
    )
    add_quote(
        doc,
        "Water, adjusted to a neutral pH value, polyisoprene (purified natural rubber "
        "latex or synthetic polyisoprene) and the co-substrate dioxygen are the only "
        "compounds necessary for efficient cleavage of polyisoprene to ODTD (2.6 U oxygen "
        "consumption/mg protein at 37 °C).",
        "Jendrossek & Birke (総説), https://doi.org/10.1007/s00253-018-9453-z",
    )
    add_quote(
        doc,
        "the expression of roxB of S. cummioxidans 35Y and characterisation of the "
        "recombinantly expressed and purified protein revealed that RoxB oxidatively "
        "cleaved polyisoprene at a high specific activity of ≈ 6 U/mg. One unit of rubber "
        "oxygenase activity corresponds to the consumption of 1 μmol dioxygen per min "
        "(for assays of rubber oxygenase activity see Hiessl et al. 2014; Röther et al. "
        "2017b).",
        "Jendrossek & Birke (総説), https://doi.org/10.1007/s00253-018-9453-z",
    )
    add_note(
        doc,
        "アッセイ法の標準プロトコル論文が特定できた: Röther, Birke & Jendrossek (2017), "
        "\"Assays for the detection of rubber oxygenase activities\", Bio-protocol 7:1-14, "
        "DOI: 10.21769/BioProtoc.2188。Bio-protocol誌は再現可能な実験手順の詳細記載を"
        "目的とした専門誌であり、緩衝液組成・電極校正法等の具体的手順が記載されている"
        "可能性が高い。全文入手を最優先とする。",
        color=GREEN,
    )
    add_note(
        doc,
        "文献間の用語不整合: Lcpの構造機能論文(Birke et al. 2017, Scientific Reports)は"
        "RoxAを「endo-type, processive」と表現している箇所があるが、Rox総説群は"
        "一貫して「RoxA performs exo-cleavage」としている。本書は総説群の記述(RoxA=エキソ型)"
        "を採用するが、実験時は生成物の炭素数分布という一次データで判断すること。",
        color=RED,
    )

    # ==================== 3. Method ====================
    add_heading(doc, "3. 測定手順", level=1)

    add_heading(doc, "3.1 OR16(Lcp1・Lcp2・Lcp3)— Gibu et al. (2020)の実証済み手順", level=2)
    doc.add_paragraph(
        "OR16自身での実証済み手順であるため、これを第一選択とする。組換え精製酵素による"
        "個別活性測定と、休止菌体による全体活性測定の両方を行う。"
    )
    add_steps(
        doc,
        [
            "lcp1(ACTI_59630)、lcp2(ACTI_59640)、lcp3(ACTI_69520)遺伝子をクローニングし、"
            "10×ヒスチジンタグを付加してE. coli(Gibu et al.はRosetta-gami B(DE3)pLysSを"
            "使用)で発現させる。",
            "Ni-アフィニティカラムクロマトグラフィーで各タンパク質(Lcp1-his, Lcp2-his, "
            "Lcp3-his)を精製する。SDS-PAGEで予想分子量のバンドを確認する。",
            "反応液(2 mL)は50 mM Tris-HCl緩衝液、天然ゴム(NR)最終濃度0.2%、精製酵素25 μg"
            "タンパク質を含む。30°Cでインキュベートする。",
            "酸素電極(Dual Digital Model 20、Rank Brothers Ltd.、または同等品)で酸素消費"
            "速度を測定する。1 U = 1分あたり1 μmol O₂消費と定義する。",
            "至適pH(6.0〜8.5)・至適温度(25〜40°C)を50 mM Tris-HCl緩衝液でスクリーニングし、"
            "各パラログの至適条件を確認する(文献値: Lcp1=30°C/pH7.0、Lcp2=35°C/pH7.5、"
            "Lcp3=30°C/pH6.5)。",
            "並行して、OR16をNRオーバーレイ寒天培地またはW最小培地+コハク酸で培養し"
            "(30°C、3日間)、休止菌体での全体活性(3パラログ合算値、dFBAのlcp_c5に直接"
            "対応)を同様の酸素消費アッセイで測定する。",
            "qRT-PCRでNR存在下でのlcp1/lcp2/lcp3発現誘導も確認する(ISOGEN IIでRNA抽出、"
            "PrimeScript II逆転写、比較対象はコハク酸単独培養)。活性と発現量の両方から、"
            "どのパラログが律速かを判断する。",
        ],
    )
    doc.add_paragraph(
        "計算式: 比活性(U/mg protein)=酸素消費速度÷反応液中タンパク質量(mg)。"
        "菌体量あたりの速度(dFBAのlcp_c5に対応)は、休止菌体アッセイでの酸素消費速度を"
        "菌体乾燥重量で除して求める。"
    )
    add_note(
        doc,
        "単位定義の注意(3.2節末尾も参照): Gibu et al. (2020)は「1 U = 1 μmol O₂/min」と"
        "定義している。一方、後述のRöther et al. (2017)のBio-protocolプロトコルでは"
        "「1 U = 1 nmol O₂/min」(1000倍小さい)と明記されている。両者は反応スケールが"
        "異なる(Gibu et al.は2 mLスケール、Röther et al.は500 μLスケールの高感度蛍光"
        "センサー法)ため単位の使い分けは物理的に理解できるが、自施設で追試する際は"
        "必ず自分のアッセイの単位定義を明記し、他文献値と比較する前に単位を揃えること。",
        color=RED,
    )

    add_heading(doc, "3.2 NS21(RoxA・RoxB)— Röther et al. (2017, Bio-protocol)の実証済み手順", level=2)
    doc.add_paragraph(
        "Röther, Birke & Jendrossek (2017, Bio-protocol 7:e2188, PMC8376518)の全文を"
        "入手できた。RoxA・Lcp共通のアッセイ法として、蛍光式酸素センサー(PreSens "
        "Oxy-4-mini)による低侵襲・微量(反応液500 μL)の酸素消費測定と、HPLCによる"
        "生成物定量の2アッセイが詳細に記載されている。これを第一選択の実証済み手順とする。"
    )
    add_steps(
        doc,
        [
            "緩衝液調製: 100 mM リン酸カリウム緩衝液(KP、K₂HPO₄とKH₂PO₄でpH 7に調整)を"
            "基本緩衝液とする。",
            "基質調製: latex milk(60%、低アンモニア型)1.5 gを100 mM KP(pH7.0)10 mLに"
            "分散。この懸濁液200 μLを、0.1%(v/v) Nonidet P-40を含む100 mM KP(pH7.0)"
            "800 μLに加え、「希釈コロイドラテックス緩衝液」(全量1 mL)とする。",
            "酵素添加: 精製Lcp(またはRoxA)を2〜5 μL(タンパク質量として約2〜5 μg)加える。",
            "反応: RoxA反応は30°C、Lcp反応は37°C、連続撹拌下でインキュベートする"
            "(Gibu et al. 2020のOR16 Lcp1/2/3個別最適化条件30/35/30°Cとは異なる値である"
            "点に注意。汎用プロトコルの既定値と、OR16特異的な至適化値の両方を試すこと)。",
            "酸素センサー校正: 二点校正を行う。上限点(飽和点)は空気飽和水(O₂ 20.9%)、"
            "下限点(ゼロ点)は新しく調製した2%(w/v)亜硫酸ナトリウム(Na₂SO₃)溶液を"
            "用いる。",
            "酸素消費の初期直線勾配(nmol O₂/min)を記録し、使用タンパク質量(mg)で除して"
            "比活性(U/mg、1 U = 1 nmol O₂/min、このプロトコルでの定義)を算出する。",
            "生成物分析: 反応後、酢酸エチルで抽出し、LiChrospher 100 RP-8カラム"
            "(5 μm、250×4.6 mm)で分析する。移動相はメタノール:水の直線グラジエント"
            "(70%→100%メタノール、20分)、流速2 mL/min、検出波長210 nm。主生成物ODTD"
            "(C15)の保持時間は約12分(このカラム・グラジエント条件での値)。",
        ],
    )
    doc.add_paragraph(
        "参考として、BRENDA酵素データベース(RoxB, EC 1.13.11.87)に、Birke, Röther & "
        "Jendrossek (2018)由来と見られるNS21由来酵素の実測値が登録されている:"
    )
    add_table(
        doc,
        ["項目", "値", "条件"],
        [
            ["RoxB_NS21 比活性", "4.8 μmol/min/mg", "23°C(pHは原論文に記載なしとBRENDAに注記)"],
            ["RoxB_NS21 分子量(成熟型)", "70,800 Da", "SDS-PAGE"],
            ["RoxB_NS21 分子量(前駆体型)", "74,000 Da", "SDS-PAGE"],
        ],
        col_widths=[4.5, 4.5, 6.0],
    )
    add_note(
        doc,
        "BRENDAはNS21の学名を「Piscinibacter gummiphilus」として登録している"
        "(本プロジェクトでは Rhizobacter gummiphilus として扱っている)。この菌は"
        "Methylibium sp. NS21(最初の単離報告)→ Rhizobacter gummiphilus NS21 → "
        "(BRENDA上)Piscinibacter gummiphilusと、複数回の学名変更を経ている可能性が"
        "ある。文献検索時はこれらの旧学名・別名も検索語に含めること。RoxA_NS21固有の"
        "数値はBRENDA・今回の検索範囲のいずれからも見つからなかった。",
        color=RED,
    )
    add_steps(
        doc,
        [
            "(手順の続き)休止菌体および可能であれば精製RoxA_NS21・RoxB_NS21"
            "(異種発現系の構築が必要。Birke et al. 2018はS. cummioxidans 35Yの"
            "ΔroxA35Y/ΔroxB35Y欠損株背景で発現させている)で、上記のRöther et al. "
            "プロトコルに準じたアッセイを実施する。",
            "RoxBNS21産物をRoxANS21の基質として与える交差実験(Birke et al. 2018で実施"
            "済み、HPLCでODTDピークの出現を確認)も再現し、協調効果を定量的に確認する。",
        ],
    )

    # ==================== 3.3 DPNR preparation (new) ====================
    add_heading(doc, "3.3 基質調製: 脱タンパク天然ゴムラテックス(DPNR)", level=2)
    doc.add_paragraph(
        "Chaikumpollert, Yamamoto, Suchiva, Nghia & Kawahara (2012), \"Preparation and "
        "characterization of protein-free natural rubber\", Polymers for Advanced "
        "Technologies 23(4):825-828, DOI: 10.1002/pat.1965 の手順を用いる。"
    )
    add_steps(
        doc,
        [
            "高アンモニア天然ゴム(HANR)ラテックス、または新鮮天然ゴムラテックスを原料と"
            "する。",
            "0.1 wt% 尿素 + 1 wt% SDS + 0.025 wt% 極性有機溶媒(アセトン、エタノール、"
            "2-プロパノールのいずれか)を加え、室温でインキュベートする(関連論文"
            "[Polymers 2022, PMC9269511]では30°Cで1時間の条件を報告)。",
            "クリーム分をSDS+溶媒溶液に再分散し、乾燥ゴム分(drc)を30 wt%に調整する。",
            "遠心分離(関連論文では10,000 rpm、1時間、約12,320×gの条件を報告)で2〜3回洗浄"
            "する。",
            "遠心分離で回収後、メタノールで凝固し、減圧下・室温で恒量になるまで乾燥する。",
        ],
    )
    doc.add_paragraph(
        "効果の実測データ(Chaikumpollert et al. 2012): 窒素含有率は未処理の0.279 wt%から、"
        "2〜3回の遠心分離処理後には0.000 wt%まで低減する。溶媒の種類ではアセトンが"
        "タンパク質除去に最も効果的(1回遠心分離後の窒素含有率: アセトン0.053 wt% < "
        "2-プロパノール0.061 wt% < エタノール0.057 wt%、SDSのみの対照0.116 wt%)。"
    )
    add_note(
        doc,
        "Gibu et al. (2020)がChaikumpollert et al.のどちらの2012年論文"
        "(Polym. Adv. Technol. 23:825-828、またはColloid Polym. Sci. 290:331)を"
        "引用しているかは今回の検索範囲では確定できなかった。前者(Polym. Adv. Technol.)"
        "の内容が本節の手順の根拠である。",
    )

    # ==================== 4. Materials ====================
    add_heading(doc, "4. 必要な装置・試薬", level=1)
    add_bullets(
        doc,
        [
            "溶存酸素測定(Lcp、Gibu et al. 2020法): 酸素電極(Dual Digital Model 20, "
            "Rank Brothers Ltd.相当品)",
            "溶存酸素測定(RoxA/Lcp、Röther et al. 2017法): 蛍光式酸素センサー"
            "(PreSens Oxy-4-mini transmitter、Oxygen Sensor Spot SP-PSt3-NAU-D5-YOP)、"
            "半微量キュベット(光路長10 mm、容量1,400 μL、Hellma analytics相当)、"
            "マグネティック微小攪拌子(5×2 mm)",
            "組換えタンパク質発現・精製: E. coli発現株(例: Rosetta-gami B(DE3)pLysS)、"
            "Ni-アフィニティカラム、His-tag発現ベクター",
            "タンパク質定量: BCAアッセイキット(Pierce BCA Kit相当)",
            "HPLC: RP8カラム(Gibu et al.法: 12×4 mm, 5 μm相当 / Röther et al.法: "
            "LiChrospher 100 RP-8, 5 μm, 250×4.6 mm相当)、UV検出器(210 nm)",
            "抽出溶媒: 酢酸エチル、メタノール(HPLCグレード)",
            "緩衝液: 50 mM Tris-HCl(Lcp、Gibu et al. 2020準拠)、100 mM リン酸カリウム"
            "緩衝液(K₂HPO₄/KH₂PO₄でpH7に調整、Röther et al. 2017準拠)、"
            "1% Nonidet P-40溶液",
            "酸素電極校正用: 2%(w/v)亜硫酸ナトリウム(Na₂SO₃、ゼロ点校正用)",
            "基質: 天然ゴムラテックス(latex milk, 60%, 低アンモニア型)、"
            "脱タンパク天然ゴムラテックス(DPNR、調製法は3.3節参照)、尿素、SDS、アセトン",
            "対照: ODTD標準品(市販品の有無は未確認。RoxA反応産物からの単離・NMR/MS同定が"
            "必要な可能性)",
        ],
    )

    # ==================== 5. Pitfalls ====================
    add_heading(doc, "5. 実務上の落とし穴(文献からの示唆)", level=1)
    add_bullets(
        doc,
        [
            "OR16のLcpは単一酵素ではなく、活性・至適pH・至適温度が異なる3パラログ"
            "(Lcp1: 30°C/pH7.0/4.02 U/mg、Lcp2: 35°C/pH7.5/1.17 U/mg、Lcp3: 30°C/pH6.5/"
            "0.22 U/mg)から成る。単一条件で休止菌体アッセイを行うと、特定のパラログに"
            "偏った活性しか捉えられない可能性がある。複数のpH・温度条件で測定すること。",
            "比活性(酵素あたり)と転写誘導倍率(lcp3が335倍)は必ずしも一致しない"
            "(Gibu et al. 2020)。「主要なパラログ」の判断は活性・発現量の両方を"
            "踏まえること。",
            "生成物の保持時間(ODTDは約15.3分、Lcp由来のC35生成物は約23分)はカラム・"
            "グラジエント条件に強く依存するため、自施設の系で必ず標準物質または既知"
            "試料により再校正すること。",
            "RoxA・RoxBの生成物はオリゴマーの炭素数分布として現れるため、単一ピークの"
            "面積だけでなく、分布全体(ゲル浸透クロマトグラフィー等)を確認することが"
            "望ましい。",
            "文献間で「RoxAはエキソ型/エンド型」の表現が一致しないため、切断様式は"
            "自施設のデータ(生成物の炭素数分布)で最終判断すること。",
            "活性単位(U)の定義が文献間で異なる(1 U = 1 nmol O₂/min vs 1 μmol O₂/min、"
            "1000倍の差)。自施設のデータを報告する際は必ず単位定義を明記し、既存文献値と"
            "比較する前に単位を揃えること(3.1節末尾の注記も参照)。",
            "NS21は文献によって学名表記が異なる(Methylibium sp. NS21 → Rhizobacter "
            "gummiphilus NS21 → BRENDA上ではPiscinibacter gummiphilus)。追加の文献検索を"
            "行う際はこれらの旧学名・別名も検索語に含めること。",
            "Röther et al. (2017)の汎用プロトコルの反応温度(RoxA=30°C, Lcp=37°C)と、"
            "Gibu et al. (2020)がOR16のLcp1/2/3個別に最適化した温度(30/35/30°C)は"
            "異なる。どちらを既定値とするか、または両方でスクリーニングするかを"
            "決めておくこと。",
        ],
    )

    # ==================== 6. Next steps ====================
    add_heading(doc, "6. 次のアクション(優先順位順)", level=1)
    add_note(
        doc,
        "第2版で挙げた5項目のうち4項目(Röther et al. 2017全文、Birke et al. 2018の"
        "代替データ、Chaikumpollert et al. 2012、遺伝子座タグの食い違い)は本版で解決した。"
        "残る項目を以下に整理する。",
        color=GREEN,
    )
    add_bullets(
        doc,
        [
            "【最優先・未解決】Birke, Röther & Jendrossek (2018)の全文(表・図)を入手し、"
            "RoxA_NS21固有の比活性・分子量・Km等の数値、およびRoxB_NS21のKm・至適pH/"
            "温度を確認する(BRENDA経由で比活性・分子量の一部は取得済みだが、原論文の"
            "全数値ではない)。",
            "OR16の最初の単離報告(Gibu et al. 2020で「Imai et al.」として言及、"
            "NBRC 114529株)の完全な書誌情報を確認する。",
            "Kasai et al. (2017)(NS21のlatA遺伝子同定論文)の完全な書誌情報を確認する"
            "(プライマー設計等で参照される可能性)。",
            "Gibu et al. (2020)が引用するChaikumpollert et al.の2012年論文が"
            "Polym. Adv. Technol. 23:825-828とColloid Polym. Sci. 290:331のどちらか"
            "(または両方)かを確認する。",
            "SBMLモデル(models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml)内の"
            "孤立した旧lcp遺伝子エントリ(G_ACTI_28730等)を削除するかどうか、"
            "プロジェクト内で判断する(本書の範囲外、モデル管理者の判断事項)。",
        ],
    )

    # ==================== 7. References ====================
    add_heading(doc, "7. 参考文献", level=1)
    add_bullets(
        doc,
        [
            "Gibu, N., Arata, T., Kuboki, S., Linh, D.V., Fukuda, M., Steinbüchel, A., "
            "Kasai, D. (2020) Characterization of the genes responsible for rubber "
            "degradation in Actinoplanes sp. strain OR16. Applied Microbiology and "
            "Biotechnology 104:7367-7376. https://doi.org/10.1007/s00253-020-10700-1",
            "Birke, J. et al. (2016) Biochemical and spectroscopic characterization of "
            "purified Latex Clearing Protein (Lcp) from newly isolated rubber degrading "
            "Rhodococcus rhodochrous strain RPK1. BMC Biotechnology. "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4877957/",
            "Birke, J. et al. (2017) Structural and Functional Analysis of Latex Clearing "
            "Protein (Lcp) Provides Insight into the Enzymatic Cleavage of Rubber. "
            "Scientific Reports. https://doi.org/10.1038/s41598-017-05268-2",
            "Birke, J., Röther, W. & Jendrossek, D. (2018) Rhizobacter gummiphilus NS21 "
            "has two rubber oxygenases (RoxA and RoxB) acting synergistically in rubber "
            "utilisation. Applied Microbiology and Biotechnology 102:10245-10257. "
            "https://doi.org/10.1007/s00253-018-9341-6",
            "Birke, J., Röther, W. & Jendrossek, D. (2017) RoxB Is a Novel Type of Rubber "
            "Oxygenase That Combines Properties of Rubber Oxygenase RoxA and Latex "
            "Clearing Protein (Lcp). Applied and Environmental Microbiology 83(14). "
            "https://doi.org/10.1128/aem.00721-17",
            "Braaz, R. et al. (2005) Heme-Dependent Rubber Oxygenase RoxA of Xanthomonas "
            "sp. Cleaves the Carbon Backbone of Poly(cis-1,4-Isoprene) by a Dioxygenase "
            "Mechanism. Applied and Environmental Microbiology 71:2473-2478. "
            "https://doi.org/10.1128/aem.71.5.2473-2478.2005",
            "Schmitt, G. et al. (2019) Towards the understanding of the enzymatic "
            "cleavage of polyisoprene by the dihaem-dioxygenase RoxA. AMB Express 9:156. "
            "https://doi.org/10.1186/s13568-019-0888-0",
            "Jendrossek, D. & Birke, J. Rubber oxygenases (総説). Applied Microbiology "
            "and Biotechnology. https://doi.org/10.1007/s00253-018-9453-z",
            "Röther, W., Birke, J. & Jendrossek, D. (2017) Assays for the detection of "
            "rubber oxygenase activities. Bio-protocol 7:e2188. "
            "https://doi.org/10.21769/BioProtoc.2188 "
            "(全文確認済み、PMC8376518)",
            "Kasai, D. et al. (2017) NS21のlatA遺伝子同定論文(完全な書誌情報は未確認)",
            "Chaikumpollert, O., Yamamoto, Y., Suchiva, K., Nghia, P.T. & Kawahara, S. "
            "(2012) Preparation and characterization of protein-free natural rubber. "
            "Polymers for Advanced Technologies 23(4):825-828. "
            "https://doi.org/10.1002/pat.1965 (全文確認済み)",
            "Nanthini, J. et al. (2020) Poly(cis-1,4-isoprene)-cleavage enzymes from "
            "natural rubber-utilizing bacteria. Bioscience, Biotechnology, and "
            "Biochemistry. https://doi.org/10.1080/09168451.2020.1733927",
            "Cleavage of natural rubber by rubber oxygenases in Gram-negative bacteria "
            "(総説, 2023頃). Applied Microbiology and Biotechnology. "
            "https://doi.org/10.1007/s00253-023-12940-3",
            "BRENDA Enzyme Database, EC 1.13.11.87 (rubber oxygenase) エントリ"
            "(RoxB_NS21の比活性・分子量の出典). "
            "https://www.brenda-enzymes.info/literature.php?e=1.13.11.87&r=746860",
        ],
    )
    add_note(
        doc,
        "全ての引用は、DeepSeek V4 Flash(model=deepseek/deepseek-v4-flash:online)による"
        "Web検索で実際に取得したURLと抜粋テキストに基づく(計8回の検索、3回の深掘りを"
        "含む)。詳細な生の検索結果はtmp/deepseek_lcp_literature.md、"
        "tmp/deepseek_rox_literature.md、tmp/deepseek_followup1〜3.md、"
        "tmp/deepseek_deepdive_rother.md、tmp/deepseek_deepdive_birke.md、"
        "tmp/deepseek_deepdive_dpnr.mdを参照。本書中の引用は全てClaudeがこの生データと"
        "照合して転記しており、創作した引用は含まれない。第2版作成時に発見した1件の"
        "著者誤帰属(Kasai→Birke/Röther/Jendrossek)は訂正済み。",
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
