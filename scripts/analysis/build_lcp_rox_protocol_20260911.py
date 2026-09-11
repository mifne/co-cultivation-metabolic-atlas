"""Build the literature-grounded wet-lab protocol for measuring the
Lcp (OR16) and RoxA/RoxB (NS21) rubber-degradation rate constants.

Literature was retrieved by DeepSeek V4 Flash with OpenRouter's web-search
plugin (model="deepseek/deepseek-v4-flash:online"); real URLs/DOIs and
direct quotes are in tmp/deepseek_lcp_literature.md and
tmp/deepseek_rox_literature.md. Claude selected, translated, and adapted
the quoted protocols into a single actionable procedure; every borrowed
number/method is attributed to its source.
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
    r2 = sub.add_run("OR16(Lcp)・NS21(RoxA/RoxB)のゴム分解速度定数を実測するための実験計画書 — 2026年9月11日")
    r2.italic = True
    r2.font.color.rgb = MUTED

    add_heading(doc, "1. 目的と背景", level=1)
    doc.add_paragraph(
        "dFBAシミュレータ(dfba_simulator.py, DEFAULT_POLYMER_RATES)は、OR16のLcp経路と"
        "NS21のRoxA/RoxB経路のゴム分解速度を、以下の未校正の仮定値で表現している。"
    )
    add_table(
        doc,
        ["パラメータ", "現在の仮定値", "対応する酵素"],
        [
            ["lcp_c5", "2.5 mmol gDW⁻¹ h⁻¹(C5換算)", "OR16のLcp(latex-clearing protein)"],
            ["roxb_c5", "0.75 mmol gDW⁻¹ h⁻¹", "NS21のRoxB(エンド型)"],
            ["roxa_direct_c5", "0.75 mmol gDW⁻¹ h⁻¹", "NS21のRoxA(エキソ型、ゴムへの直接作用)"],
            ["roxa_oligo_c30", "0.25 mmol gDW⁻¹ h⁻¹(C30換算)", "NS21のRoxA(オリゴマーからODTDへの変換)"],
        ],
        col_widths=[3.5, 5.5, 6.5],
    )
    doc.add_paragraph(
        "本書は、これらの値を実測で置き換えるための実験計画を、先行研究の直接引用に基づいて"
        "作成する。文献はDeepSeek V4 Flash(OpenRouterのWeb検索プラグイン, "
        "model=deepseek/deepseek-v4-flash:online)で実際に検索し、実在するURLと引用文を"
        "確認した上で採用している。"
    )
    add_note(
        doc,
        "重要な留意点: OR16(Actinoplanes sp.)によるゴム分解を直接報告した文献は、"
        "今回の検索では見つからなかった。一方、NS21と同種(Rhizobacter gummiphilus NS21)を"
        "直接対象とした論文(Kasai et al.)が見つかっており、これは極めて重要な先行研究である。",
        color=RED,
    )

    # 2. Literature review
    add_heading(doc, "2. 先行研究のレビュー(直接引用)", level=1)

    add_heading(doc, "2.1 Lcp(latex-clearing protein) — OR16に対応", level=2)
    doc.add_paragraph(
        "Lcpはグラム陽性放線菌(Streptomyces, Rhodococcus, Gordonia, Nocardia等)が持つ、"
        "b型シトクロムのエンド型ゴム切断酵素である。OR16と同じActinoplanes属での報告は"
        "見つからなかったが、近縁の放線菌での詳細な酵素学的特徴づけが複数報告されている。"
    )
    add_quote(
        doc,
        "Purified Lcp Rr had a specific activity of 3.1 U/mg at 30 °C and degraded "
        "poly(1,4-cis-isoprene) to a mixture of oligoisoprene molecules with terminal "
        "keto and aldehyde groups. The pH optimum of Lcp Rr was higher (pH 8) than for "
        "other rubber-cleaving enzymes (≈ pH 7).",
        "Birke et al. (2016), BMC Biotechnology, "
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4877957/",
    )
    add_quote(
        doc,
        "In contrast to RoxAs that cleave rubber in an endo-type, processive manner to a "
        "single major end product (ODTD, C15 oligoisoprenoid), Lcps produce a mixture of "
        "cleavage products that differ in the number of central isoprene units.",
        "Birke et al. (2017), Scientific Reports, https://doi.org/10.1038/s41598-017-05268-2",
    )
    doc.add_paragraph(
        "測定法として、酸素消費アッセイ(OXY-4 mini apparatus, PreSens社)とHPLCベースの"
        "生成物定量アッセイの両方が確立されている:"
    )
    add_quote(
        doc,
        "poly(cis-1,4-isoprene) latex was diluted with 100 mM KP buffer, pH 7, to 0.2 % "
        "(assay volume 0.7 mL) and incubated in the presence of purified Lcp protein for "
        "2 h at a temperature as indicated... The products were extracted with 1 mL ethyl "
        "acetate..., dried, and dissolved in 100 μL methanol. Aliquots were applied to an "
        "RP8 HPLC column (12 × 4 mm, 5 μm particle size, 0.7 mL/min) with water (A) and "
        "methanol (B) as mobile phases. The concentration of B was increased from 50 % "
        "(v/v) to 100 % (v/v) within 15 min; products were detected at 210 nm.",
        "Birke et al. (2016), BMC Biotechnology, "
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4877957/",
    )
    add_note(
        doc,
        "Km・Vmax・kcatの具体的な数値は今回の検索範囲では見つからなかった。比活性(U/mg)を"
        "基礎データとして使用する。",
    )
    add_note(
        doc,
        "文献間の用語不整合に関する注記: 上記引用(Scientific Reports 2017)はRoxAを"
        "「endo-type, processive」と表現しているが、後述のRox総説(2023)は"
        "「RoxA performs exo-cleavage」と明記しており、表現が一致しない。これは原文間の"
        "実際の食い違いであり、本書はどちらかに寄せず両方をそのまま引用する。実験設計・"
        "解釈の際は、この分類が論文によって揺れることを踏まえ、生成物の炭素数分布"
        "(単一主生成物ODTD vs 複数サイズの混合物)という一次データで切断様式を判断すること。",
        color=RED,
    )

    add_heading(doc, "2.2 RoxA・RoxB — NS21に対応(直接の先行研究あり)", level=2)
    doc.add_paragraph(
        "RoxA・RoxBはグラム陰性菌(Steroidobacter cummioxidans 35Y、旧Xanthomonas sp. 35Y、"
        "および Rhizobacter gummiphilus NS21)が持つジヘム依存性ジオキシゲナーゼである。"
        "RoxAはエキソ型でC15化合物ODTDを主生成物とし、RoxBはエンド型でC20以上のオリゴマーを"
        "生成する。"
    )
    add_quote(
        doc,
        "No homologue for an Lcp protein but homologues for a putative RoxA and a RoxB "
        "protein (the latter identical to a previously postulated LatA-denominated rubber "
        "cleaving enzyme) were identified in the genome of strain NS21. The roxA_NS21 and "
        "roxB_NS21 genes were separately expressed in a ΔroxA_35Y/ΔroxB_35Y background of "
        "S. cummioxidans 35Y and restored the ability of the mutant to produce "
        "oligoisoprenoids... indicate that Gram-negative rubber-degrading bacteria "
        "generally utilise two synergistically acting rubber oxygenases (RoxA/RoxB) for "
        "efficient cleavage of polyisoprene to ODTD.",
        "Kasai et al. (2017), Applied Microbiology and Biotechnology 101:7493–7503, "
        "https://link.springer.com/article/10.1007/s00253-018-9341-6",
    )
    add_note(
        doc,
        "この論文はNS21のRoxA/RoxBを直接クローニング・異種発現・精製して生化学的に特徴づけた"
        "ものであり、本プロジェクトのモデル(NS21のRoxA/RoxB)に直接対応する最重要文献である。"
        "検索結果の抜粋には具体的な速度論数値は含まれていなかったため、着手前に全文"
        "(該当DOI)を入手し、Km/Vmax/比活性の記載があるか確認すること。",
        color=RED,
    )
    add_quote(
        doc,
        "Oxidative cleavage of poly(cis-1,4-isoprene) by rubber oxygenase RoxA... "
        "12-oxo-4,8-dimethyl-trideca-4,8-diene-1-al (ODTD; m/z 236) was the main cleavage "
        "product... revealing that RoxA cleaves rubber by a dioxygenase mechanism.",
        "Braaz et al. (2005), Applied and Environmental Microbiology 71:2473-2478, "
        "https://doi.org/10.1128/aem.71.5.2473-2478.2005",
    )
    add_quote(
        doc,
        "To determine the activity of RoxA, two different assays were applied: (i) the "
        "consumption of dissolved oxygen was determined in an OXY-4 mini apparatus... "
        "(ii) polyisoprene latex was incubated in the presence of the test enzyme for 1 "
        "or 2 h at 23 °C or 30 °C. The cleavage products were extracted with ethyl-acetate "
        "and separated by HPLC... the peak area of the cleavage product peak (ODTD) with a "
        "retention time of ≈ 15.3 min was used.",
        "Schmitt et al. (2019), AMB Express 9:156, https://doi.org/10.1186/s13568-019-0888-0",
    )

    # 3. Method
    add_heading(doc, "3. 測定手順(文献法をOR16/NS21向けに適応)", level=1)
    add_note(
        doc,
        "モデルが必要とするのは「菌体量あたりの分解速度(mmol gDW⁻¹ h⁻¹)」であるため、"
        "精製酵素の比活性(U/mg protein)ではなく、まず休止菌体(resting cell)アッセイを"
        "第一選択とする。精製酵素アッセイは、生成物の同定(Lcp型かRox型かの確認)と、"
        "将来的な比活性からの換算のための補助情報として位置づける。",
    )

    add_heading(doc, "3.1 共通の基質調製", level=2)
    add_steps(
        doc,
        [
            "天然ゴムラテックスを脱タンパク処理する(既存プロトコルの詳細な調製条件は"
            "今回の検索範囲では確認できなかったため、Birke et al. 2016のMethods原著、"
            "または類似研究の脱タンパク天然ゴムラテックス(DPNR)調製法を別途参照して"
            "確定すること)。",
            "poly(cis-1,4-isoprene)ラテックスを100 mM リン酸カリウム緩衝液(KPi)で"
            "0.2%(w/v)に希釈する(Birke et al. 2016に準拠)。",
        ],
    )

    add_heading(doc, "3.2 OR16(Lcp)の休止菌体アッセイ", level=2)
    add_steps(
        doc,
        [
            "OR16をゴム含有培地(または抑制のかからない炭素源、コハク酸等)で前培養する。",
            "集菌・洗浄し、菌体をKPi緩衝液(pH 7〜8; Lcpの至適pHはpH 8"
            "[Birke et al. 2016])に懸濁する。",
            "反応容器にDPNRラテックス(0.2%)と菌体懸濁液を加え、30°C(Lcp_Rrの至適温度"
            "[Birke et al. 2016])でインキュベートする。",
            "Clark型電極(OXY-4 mini apparatus等)で溶存酸素消費速度を連続記録する。",
            "並行して、反応液を酢酸エチルで抽出・乾固後メタノールに再溶解し、RP8 HPLCカラム"
            "(水:メタノール、50→100%を15分でグラジエント)で分離、210 nmで生成物を検出する"
            "[Birke et al. 2016のプロトコルに準拠]。",
            "菌体無添加対照、および加熱失活菌体対照を必ず並行して置く。",
        ],
    )
    doc.add_paragraph(
        "計算式: 比活性(U/mg protein)または比速度(mmol gDW⁻¹ h⁻¹)="
        "酸素消費速度(または生成物生成速度)÷菌体量。"
        "文献のLcp_Rr比活性3.1 U/mg(30°C, pH8)[Birke et al. 2016]を、酵素含量が既知の"
        "場合の換算の参考値とする(ただしOR16での実測ではないため、あくまで桁感の"
        "目安である)。"
    )

    add_heading(doc, "3.3 NS21(RoxA・RoxB)の休止菌体アッセイ", level=2)
    add_steps(
        doc,
        [
            "NS21をゴム含有培地で前培養、集菌・洗浄する。",
            "反応はKPi緩衝液中、23°Cまたは30°C[Schmitt et al. 2019]で実施する。",
            "OXY-4 mini apparatus等による酸素消費アッセイで全体の分解活性(RoxA+RoxB合算)を"
            "測定する。",
            "生成物側では、HPLC(RP8カラム、保持時間 約15.3分にODTDピーク"
            "[Schmitt et al. 2019])でODTD生成量を定量し、RoxA由来の直接分解速度"
            "(roxa_direct_c5に対応)を推定する。",
            "RoxB由来のC20以上のオリゴマー生成量は、ODTDピークとは異なる保持時間に検出"
            "されるため、これを別途定量してroxb_c5に対応させる。両ピークの分離条件は"
            "予備実験で確認する。",
            "RoxAとRoxBは協調的に作用することが報告されている"
            "[Kasai et al. 2017、Springer総説2023]ため、野生株での全体活性に加え、"
            "可能であれば組換え体または精製RoxA_NS21・RoxB_NS21を個別に用いた比較で、"
            "roxa_oligo_c30(RoxAによるオリゴマー→ODTD変換)とroxb_c5"
            "(RoxBによるゴム→オリゴマー変換)を切り分ける。",
        ],
    )
    add_note(
        doc,
        "RoxA/RoxBの協調効果(cooperative effect)が報告されているため"
        "[Springer総説2023: \"a cooperative effect on the efficiency of cleavage was "
        "determined when RoxA and RoxB were present simultaneously\"]、野生株全体の"
        "活性は単純な加算にならない可能性がある。個別酵素の活性と野生株全体活性の"
        "両方を測定し、比較すること。",
    )

    # 4. Materials
    add_heading(doc, "4. 必要な装置・試薬", level=1)
    add_bullets(
        doc,
        [
            "溶存酸素測定: Clark型電極、またはOXY-4 mini apparatus相当品(PreSens社製、"
            "文献で使用)",
            "HPLC: RP8カラム(12×4 mm, 5 μm粒子径相当)、UV検出器(210 nm)",
            "抽出溶媒: 酢酸エチル、メタノール(HPLCグレード)",
            "緩衝液: 100 mM リン酸カリウム緩衝液(pH 7およびpH 8を用意)",
            "基質: 脱タンパク天然ゴムラテックス(DPNR)",
            "対照: ODTD標準品(市販品の有無は今回の検索で確認できず、別途調査または"
            "RoxA反応産物からの単離・NMR/MS同定が必要な可能性がある)",
        ],
    )

    # 5. Pitfalls
    add_heading(doc, "5. 実務上の落とし穴(文献からの示唆)", level=1)
    add_bullets(
        doc,
        [
            "Lcp・RoxAともに至適pHや至適温度で活性・安定性が大きく変わる"
            "(Lcp_Rrの至適pHはpH8で他のゴム切断酵素[pH7付近]より高い"
            "[Birke et al. 2016])。至適条件を予備実験で必ず確認する。",
            "Lcp_Rr・Lcp_K30は37°Cで活性が不安定化し、in vitro反応は4〜8時間で頭打ちに"
            "なる一方、RoxAはより安定で70時間までODTD産生を継続したとの報告がある"
            "[Birke et al. 2016]。反応時間の設計に注意する。",
            "生成物の保持時間(ODTDは約15.3分、Lcp由来のC35生成物は約23分"
            "[Birke et al. 2016; Schmitt et al. 2019])はカラム・グラジエント条件に強く"
            "依存するため、自施設の系で必ず標準物質または既知試料により再校正すること。",
            "RoxA・RoxBの生成物はオリゴマーの炭素数分布として現れるため、単一ピークの"
            "面積だけでなく、分布全体(ゲル浸透クロマトグラフィー等)を確認することが"
            "望ましい[Springer総説2023: \"quantitatively isolated via gel permeation "
            "chromatography\"]。",
        ],
    )

    # 6. Next steps
    add_heading(doc, "6. 次のアクション", level=1)
    add_bullets(
        doc,
        [
            "Kasai et al. (2017)の全文を入手し、NS21のRoxA/RoxB比活性・速度論パラメータの"
            "記載有無を確認する(本書はアブストラクト・抜粋レベルの検索結果に基づいており、"
            "全文未確認)。",
            "Birke and Jendrossek (2014)、Röther et al. (2017a/b)など、今回の検索で"
            "参照のみ確認できた引用文献(酸素消費アッセイの詳細な緩衝液組成・校正法を含む"
            "可能性が高い)を追加で取得する。",
            "OR16(Actinoplanes属)特有の文献が存在しないため、上記の近縁種プロトコルを"
            "そのまま適用してよいか、予備実験(至適pH・温度の確認)で検証してから本実験へ"
            "進む。",
            "DPNR(脱タンパク天然ゴムラテックス)の具体的な調製プロトコルを別途確定する。",
        ],
    )

    # 7. References
    add_heading(doc, "7. 参考文献", level=1)
    add_bullets(
        doc,
        [
            "Birke, J. et al. (2016) Biochemical and spectroscopic characterization of "
            "purified Latex Clearing Protein (Lcp) from newly isolated rubber degrading "
            "Rhodococcus rhodochrous strain RPK1. BMC Biotechnology. "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4877957/",
            "Birke, J. et al. (2017) Structural and Functional Analysis of Latex Clearing "
            "Protein (Lcp) Provides Insight into the Enzymatic Cleavage of Rubber. "
            "Scientific Reports. https://doi.org/10.1038/s41598-017-05268-2",
            "Nanthini, J. et al. (2020) Poly(cis-1,4-isoprene)-cleavage enzymes from "
            "natural rubber-utilizing bacteria. Bioscience, Biotechnology, and "
            "Biochemistry. https://doi.org/10.1080/09168451.2020.1733927",
            "Braaz, R. et al. (2005) Heme-Dependent Rubber Oxygenase RoxA of Xanthomonas "
            "sp. Cleaves the Carbon Backbone of Poly(cis-1,4-Isoprene) by a Dioxygenase "
            "Mechanism. Applied and Environmental Microbiology 71:2473-2478. "
            "https://doi.org/10.1128/aem.71.5.2473-2478.2005",
            "Schmitt, G. et al. (2019) Towards the understanding of the enzymatic "
            "cleavage of polyisoprene by the dihaem-dioxygenase RoxA. AMB Express 9:156. "
            "https://doi.org/10.1186/s13568-019-0888-0",
            "Kasai, D. et al. (2017) Rhizobacter gummiphilus NS21 has two rubber "
            "oxygenases (RoxA and RoxB) acting synergistically in rubber utilisation. "
            "Applied Microbiology and Biotechnology 101:7493–7503. "
            "https://link.springer.com/article/10.1007/s00253-018-9341-6",
            "Cleavage of natural rubber by rubber oxygenases in Gram-negative bacteria "
            "(総説, 2023). Applied Microbiology and Biotechnology. "
            "https://link.springer.com/article/10.1007/s00253-023-12940-3",
            "Jendrossek & Birke (2019として上記総説内に引用) Rubber oxygenases. "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6311187/",
        ],
    )
    add_note(
        doc,
        "全ての引用は、DeepSeek V4 Flash(model=deepseek/deepseek-v4-flash:online)による"
        "Web検索で実際に取得したURLと抜粋テキストに基づく。詳細な生の検索結果・抜粋全文は"
        "tmp/deepseek_lcp_literature.md、tmp/deepseek_rox_literature.mdを参照(リポジトリの"
        "一時ファイルのため、必要であれば別途保存すること)。本書中の引用は全てClaudeが"
        "この生データと照合して転記しており、創作した引用は含まれない。",
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
