"""Build the literature-grounded wet-lab protocol for measuring the
non-growth-associated maintenance ATP requirement (NGAM / maintenance
coefficient) for the three consortium members: Actinoplanes sp. OR16,
Rhizobacter (Piscinibacter) gummiphilus NS21, and Propionibacterium
freudenreichii (Pf).

This is the second protocol in the calibration series, following the same
methodology as build_lcp_rox_protocol_20260911.py: DeepSeek V4 Flash
(model=deepseek/deepseek-v4-flash:online) was used for three separate
web-search deep-dives (one per organism / its closest relatives), and every
number quoted here is traceable to a real URL captured in
tmp/deepseek_maintenance_actinomycete.md, tmp/deepseek_maintenance_ns21.md,
and tmp/deepseek_maintenance_pf.md.

Headline finding: no organism in the consortium has a directly measured
maintenance coefficient in the literature. This protocol is therefore for a
genuinely novel measurement, using the classical Pirt chemostat method
(explicit methodology and worked example found in Tannler, Decasper & Sauer
2008, Microb Cell Fact 7:19). The literature review section supplies
order-of-magnitude priors from the closest relatives found for each
organism, to be used as starting guesses / sanity checks, not substitutes.

Two reference values (Khosravi-Darani et al. 2003 and Pflueger et al. 2018,
both for Cupriavidus necator) and one (Pena et al. 2000, Azotobacter
vinelandii) had their DOIs flagged by DeepSeek itself as "推定" (inferred,
not confirmed against the source) -- this is stated explicitly rather than
presented as a confirmed citation.
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
OUT_PATH = ROOT / "docs" / "MAINTENANCE_ATP_PROTOCOL_20260912.docx"

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
    r = title.add_run("維持ATP要求量(NGAM)測定プロトコル")
    r.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run(
        "OR16・NS21(Piscinibacter gummiphilus)・P. freudenreichiiの"
        "維持代謝パラメータを実測するための実験計画書 — 第1版 2026年9月12日"
    )
    r2.italic = True
    r2.font.color.rgb = MUTED

    add_note(
        doc,
        "本書はLcp/Rox活性測定プロトコル(LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docx)"
        "に続く、dFBAコンソーシアムモデル校正シリーズの第2弾である。DeepSeek V4 Flash"
        "(model=deepseek/deepseek-v4-flash:online)による3回のWeb検索(菌種ごと)で"
        "文献調査を行った。主要な結論: 3菌種いずれについても維持ATP要求量の直接実測値は"
        "文献上見つからなかった。したがって本測定は文献の後追いではなく、真に新規の"
        "実測データ取得である。近縁種からの参考値は、較正の初期値・妥当性確認にのみ"
        "用いること。",
        color=GREEN,
    )

    # ==================== 1. Background ====================
    add_heading(doc, "1. 目的と背景", level=1)
    doc.add_paragraph(
        "dFBAシミュレータの生理学レイヤー(src/physiology_dfba.py)は、各菌種の"
        "維持ATP要求量を `maintenance` 辞書で表現する設計になっている。各エントリは "
        "`{reaction: <ATP加水分解反応ID>, rate_mmol_g_h: <要求速度>}` の形式で、"
        "指定された速度をFBA解の中でATP maintenance反応の下限として強制する仕組みが"
        "すでに実装されている(_apply_maintenanceの該当ロジック、"
        "physiology_dfba.py:187-205)。"
    )
    doc.add_paragraph(
        "しかし現状、この `maintenance` 辞書はコンストラクタのデフォルト引数 `None` "
        "(実質的に空dict、physiology_dfba.py:38)のまま、OR16・NS21・Pfのいずれの"
        "モデルインスタンスにも値が渡されていない(プロジェクト内`grep`で確認: "
        "physiology_dfba.py以外のどのソースからも `maintenance=` へ実数値が渡された"
        "呼び出しは存在しない)。すなわち3菌種すべてで維持代謝コストがゼロとして"
        "扱われており、増殖収率・基質消費速度・共存動態のdFBA予測が過大評価に"
        "偏っている可能性が高い。"
    )
    add_note(
        doc,
        "維持ATP要求量(NGAM: non-growth-associated maintenance)は、細胞が増殖を"
        "伴わずに恒常性維持(膜電位維持、代謝回転、輸送等)のために消費するATP量。"
        "ゲノムスケールモデルでは通常、バイオマス反応から独立したATP加水分解反応"
        "(ATPM)として実装し、増殖速度によらない一定の下限フラックスを課す。",
    )
    doc.add_paragraph(
        "測定原理はPirtの古典的なケモスタットモデルに基づく。基質取り込み速度 qS "
        "(比増殖速度μの関数)を複数の希釈率Dで測定し、"
    )
    add_quote(
        doc,
        "qglc = μ / Yglc,max + mglc",
        "Tannler, Decasper & Sauer (2008), Microb Cell Fact 7:19",
    )
    doc.add_paragraph(
        "の関係でプロットすると、y切片が維持係数 mglc(または対応するmATP)を与える。"
        "この手法は放線菌・プロテオバクテリア・グラム陽性菌のいずれにも適用可能な"
        "汎用手法であり、3菌種すべてに同一の測定戦略を採用できる(3節参照)。"
    )

    # ==================== 2. Literature review ====================
    add_heading(doc, "2. 先行研究のレビュー(直接引用)", level=1)
    add_note(
        doc,
        "3菌種いずれについても「そのもの」の維持ATP実測値は見つからなかった"
        "(DeepSeek V4 Flashによる実検索で確認、創作なし)。以下は近縁種からの"
        "参考値であり、較正の初期値として使うことを想定している。",
        color=RED,
    )

    add_heading(doc, "2.1 OR16側参考値 — 放線菌(Actinomycetota)", level=2)
    doc.add_paragraph(
        "Actinoplanes属そのもの、およびゴム分解放線菌特有の維持代謝を報告した文献は"
        "見つからなかった(検索語: \"Actinoplanes\" AND \"maintenance coefficient\" / "
        "\"maintenance energy\"; ゴム分解菌 OR16 の原著論文(Gibu et al. 2020)にも"
        "維持代謝の数値データは含まれない)。"
    )
    doc.add_paragraph(
        "近縁放線菌では、Streptomyces coelicolorのケモスタットデータに基づく実測値が"
        "存在する。"
    )
    add_quote(
        doc,
        "Here, taking into account a P/O value of 2, mATP values (on a biomass "
        "basis) would be 7.545±0.3 and 4.482±0.2 mmol ATP (g cell dry weight)"
        "−1 h−1 in M145 and M1146, respectively, that is, fell well within the "
        "range (4.52–9.64 mmol ATP (g cell dry weight)−1 h−1) of those found in "
        "various bacteria and S. coelicolor grown on glucose as a carbon source.",
        "Coze, Gilard, Tcherkez, Virolle & Guyonvarch (2013), PLoS ONE 8(12):e84151, "
        "https://doi.org/10.1371/journal.pone.0084151",
    )
    add_quote(
        doc,
        "Using our own experimental data, maintenance energy should correspond "
        "to 0.498±0.019 mmol glucose (g cell dry weight)−1 h−1 in M145 and "
        "0.341±0.015 mmol glucose (g cell dry weight)−1 h−1 in M1146.",
        "同上",
    )
    doc.add_paragraph(
        "したがってS. coelicolor(野生株M145)のmATP ≈ 7.5 mmol ATP/gDW/h、グルコース"
        "消費ベースのm_s ≈ 0.5 mmol glucose/gDW/hが、OR16の較正初期値の目安として"
        "利用できる。ただしOR16はゴムを主要炭素源とする点で系統的・代謝的に大きく"
        "異なるため、あくまで桁数の妥当性確認用と位置づける。"
    )
    doc.add_paragraph(
        "Pirtプロットの手法そのものについては、放線菌ではないがBacillus属での"
        "標準的な適用例が詳細に報告されている(3節の測定手順で参照)。"
    )
    add_quote(
        doc,
        "The determined coefficient of 0.39 mmol g(cdw)-1 h-1 for B. subtilis "
        "wild type 168 compares favorably to the previously reported 0.44 mmol "
        "g(cdw)-1 h-1 of the related B. subtilis wild type 1012 ... The spo0A "
        "mutant ... exhibited an increased maintenance coefficient of 0.49 mmol "
        "g(cdw)-1 h-1.",
        "Tannler, Decasper & Sauer (2008), Microb Cell Fact 7:19, "
        "https://doi.org/10.1186/1475-2859-7-19",
    )

    add_heading(
        doc, "2.2 NS21側参考値 — ベータプロテオバクテリア・PHA生産菌", level=2
    )
    doc.add_paragraph(
        "\"Rhizobacter gummiphilus\" \"Piscinibacter gummiphilus\" のいずれの学名でも"
        "維持代謝の報告は見つからなかった。近縁のComamonadaceae科"
        "(Methylibium petroleiphilum PM1等)についても、ケモスタット/Pirt法による"
        "維持係数の報告は確認できなかった。"
    )
    add_note(
        doc,
        "副産物として、NS21の現行正式学名が Piscinibacter gummiphilus であることを"
        "NCBI/DDBJ(ACCESSION CP015118)およびKEGG(org code: rgu)で直接確認した"
        "(旧学名Rhizobacter gummiphilusはシノニムとして併記)。また、NS21固有の"
        "天然ゴム→PHA変換機構を扱った2024年の論文を新たに発見した(2.2節末尾の"
        "「関連文献の追加発見」を参照)。",
        color=GREEN,
    )
    doc.add_paragraph(
        "維持代謝の参考値としては、同じBurkholderiales目に属しPHA生産能を持つ"
        "Cupriavidus necator(旧Ralstonia eutropha)の値が最も近縁と考えられる。"
    )
    add_quote(
        doc,
        "The maintenance coefficient (m_s) for glucose was determined to be "
        "0.02 g glucose/g biomass/h under oxygen-limited conditions, and "
        "0.01 g glucose/g biomass/h under oxygen-sufficient conditions.",
        "Khosravi-Darani et al. (2003), Biotechnology and Bioengineering "
        "(DOI: 10.1002/bit.10586 — DeepSeekの検索結果ではDOIが「推定」表示。"
        "原論文での直接確認が必要)",
    )
    add_quote(
        doc,
        "The non-growth-associated maintenance (NGAM) coefficient was "
        "estimated to be 1.5 mmol ATP/gDW/h based on literature data and "
        "model fitting.",
        "Pflueger et al. (2018), Metabolic Engineering, iCN718モデル "
        "(DOI: 10.1016/j.ymben.2018.06.003 — 同上、DOI「推定」表示。文献フィッティング"
        "由来の値であり、ケモスタット直接実測ではない点に注意)",
    )
    add_quote(
        doc,
        "The maintenance coefficient (m_s) for sucrose was 0.03 g sucrose/g "
        "biomass/h under nitrogen-fixing conditions.",
        "Pena et al. (2000), Biotechnology and Bioengineering, "
        "Azotobacter vinelandii (DOI「推定」表示、γ-プロテオバクテリアで系統的には"
        "やや離れる)",
    )
    doc.add_paragraph(
        "以上より、NS21の較正初期値としてはC. necatorのNGAM値"
        "(≈1.5 mmol ATP/gDW/h、または基質ベースでm_s≈0.01-0.02 g glucose/gDW/h)を"
        "採用し、感度分析で調整するのが現実的である。"
    )
    add_note(
        doc,
        "関連文献の追加発見(維持代謝の直接データではないが、NS21のPHA代謝校正に"
        "有用): Characterization of the conversion system of natural rubber to "
        "poly(3-Hydroxyalkanoate) in Piscinibacter gummiphilus strain NS21T, "
        "New Biotechnology (2024), DOI: 10.1016/j.nbt.2024.08.507。"
        "天然ゴムおよびグルコースからのPHBV/PHB生産、phaC遺伝子の関与、窒素制限下での"
        "PHBV蓄積速度上昇、PHA depolymerase遺伝子破壊による蓄積量増加を報告している。"
        "本プロトコルの範囲外だが、優先度⑥(max_pha_fraction較正)で活用できる。",
    )

    add_heading(
        doc, "2.3 Pf側参考値 — Propionibacterium freudenreichii", level=2
    )
    doc.add_paragraph(
        "P. freudenreichiiそのもののケモスタット/Pirt法による維持係数実測値は"
        "見つからなかった。ただし、ゲノムスケール代謝モデル構築の過程で"
        "校正されたNGAM値が報告されている。"
    )
    add_quote(
        doc,
        "A reaction was also added to the metabolic network to account for "
        "nongrowth-associated ATP maintenance (NGAM). The initial value of "
        "flux through the ATP maintenance (ATPM) reaction was assumed to be "
        "0.76 based on a previously determined ATPM for P. acidipropionici.",
        "Genome Scale Constraint-Based Metabolic Reconstruction of "
        "Propionibacterium Freudenreichii DSM 20271 (2021, Research Square "
        "preprint), https://doi.org/10.21203/rs.3.rs-847511/v1",
    )
    add_quote(
        doc,
        "The assumption of biomass optimality was maintained during "
        "simulation. Nongrowth-associated maintenance ATP was calibrated to "
        "account for 1.4 mmol/gDW/hr.",
        "同上(dFBAシミュレーションによる最終校正値、iMR558モデル)",
    )
    add_quote(
        doc,
        "Growth-associated ATP maintenance (GAM) in the biomass equation was "
        "assumed to be 35 based on the P. freudenreichii theoretical "
        "calculated ATP yield (28.8) by Papoutsakis et al.",
        "同上(増殖関連ATP要求量、単位: mmol ATP/gDW biomass)",
    )
    doc.add_paragraph(
        "この論文では、B. subtilisのバイオマス組成を流用してiMR558モデルを構築し、"
        "NGAMの初期値0.76 mmol/gDW/h(近縁種P. acidipropionici由来)を、Meurice et al."
        "の嫌気バッチ培養実験データに対するdFBA適合によって1.4 mmol/gDW/hへ校正した。"
        "ケモスタット連続培養によるPirt plot直接実測ではなく、あくまでバッチ培養"
        "dFBAフィッティングによる値である点に注意が必要。"
    )
    doc.add_paragraph(
        "プロピオン酸/酢酸比についても実測との良好な一致が報告されている。"
    )
    add_quote(
        doc,
        "The ratio of propionic acid to acetate is hence calculated to be "
        "2.4, which compares well with the experimentally reported value of "
        "2.2 mmol.",
        "同上(Meurice et al.の嫌気バッチ培養実験との比較)",
    )
    doc.add_paragraph(
        "同じグラム陽性嫌気性菌としてLactobacillus delbrueckiiの実測値も参考になる。"
    )
    add_quote(
        doc,
        "The estimation of the apparent maintenance coefficient, e.g., the "
        "value of q ATP when µ is zero, gives the value of 4 mmol g-1 h-1, a "
        "value which is clearly overestimated since during the second growth "
        "phase, growth is observed while the q ATP is as low as 1.3 mmol "
        "g-1 h-1.",
        "Le Breton et al. (2003), Lait, https://doi.org/10.1051/lait:2003032",
    )
    add_quote(
        doc,
        "the maintenance coefficient (m ATP) increases from 2.6 to 3.0, 3.2 "
        "and 3.6 mmol g-1 h-1",
        "同上(µ = 0.4, 0.3, 0.2, 0.1 h-1における値、増殖速度依存的な維持係数)",
    )
    doc.add_paragraph(
        "Pfの較正初期値としては、iMR558モデルのNGAM校正値1.4 mmol ATP/gDW/hを"
        "第一候補とし、Lactobacillus delbrueckiiの実測範囲(1.3-4 mmol/g/h)を"
        "妥当性確認の参考レンジとする。"
    )

    add_heading(doc, "2.4 参考値まとめ", level=2)
    add_table(
        doc,
        ["菌種", "参考種", "維持係数", "出典", "備考"],
        [
            [
                "OR16",
                "S. coelicolor M145",
                "7.545±0.3 mmol ATP/gDW/h\n(0.498 mmol glucose/gDW/h)",
                "Coze et al. 2013",
                "ケモスタット実測。放線菌だが非ゴム分解菌",
            ],
            [
                "NS21",
                "C. necator (NGAM推定)",
                "1.5 mmol ATP/gDW/h",
                "Pflueger et al. 2018",
                "文献フィッティング由来、DOI要確認",
            ],
            [
                "NS21",
                "C. necator (実測m_s)",
                "0.01-0.02 g glucose/gDW/h",
                "Khosravi-Darani et al. 2003",
                "ケモスタット実測、DOI要確認",
            ],
            [
                "Pf",
                "P. freudenreichii (自身、モデル校正値)",
                "1.4 mmol ATP/gDW/h",
                "iMR558論文 2021",
                "バッチdFBAフィッティング、ケモスタット実測ではない",
            ],
            [
                "Pf",
                "L. delbrueckii",
                "1.3-4 mmol ATP/g/h(µ依存)",
                "Le Breton et al. 2003",
                "近縁グラム陽性嫌気性菌の実測",
            ],
        ],
        col_widths=[2.0, 3.3, 4.0, 3.0, 4.2],
    )

    # ==================== 3. Procedure ====================
    add_heading(doc, "3. 測定手順(ケモスタット連続培養・Pirtプロット法)", level=1)
    add_note(
        doc,
        "Tannler, Decasper & Sauer (2008)のBacillus属での適用例を一般化した手順。"
        "3菌種いずれにも同一の原理を適用するが、OR16は不溶性のゴムを主要炭素源とする"
        "特殊性があるため、3.4節で個別の対応を示す。",
    )

    add_heading(doc, "3.1 ケモスタット連続培養の設定", level=2)
    add_steps(
        doc,
        [
            "対象菌種を、増殖を制限する単一の可溶性炭素源(NS21・Pfはグルコースまたは"
            "既知の資化可能炭素源、OR16は3.4節参照)を含む規定培地でケモスタットに"
            "接種する。",
            "少なくとも4〜6段階の希釈率D(= 比増殖速度µに等しい、定常状態において)を"
            "設定する。最大比増殖速度µ_maxの20〜80%程度の範囲を目安にカバーする"
            "(µ_max近傍は洗い出しのリスクがあるため避ける)。",
            "各希釈率で、接種後の体積交換数5〜7回分の時間が経過し、菌体濃度(OD600または"
            "乾燥重量)・残存基質濃度が安定した定常状態に達するまで培養を継続する。",
        ],
    )
    add_quote(
        doc,
        "For continuous cultures, all physiological parameters were "
        "determined during steady state between 5 to 7 volume changes after "
        "inoculation.",
        "Tannler, Decasper & Sauer (2008), Microb Cell Fact 7:19",
    )

    add_heading(doc, "3.2 定常状態でのサンプリング・分析", level=2)
    add_steps(
        doc,
        [
            "各希釈率の定常状態において、供給培地中および培養上清中の基質濃度"
            "(S_feed, S)と、乾燥菌体重量(X, gDW/L)を測定する。",
            "比基質消費速度を qS(または P) = ΔS(またはP) × D / X の関係式から算出する"
            "(ΔSは供給培地と上清の濃度差、Dは希釈率、Xは菌体濃度)。",
            "可能であれば同時に、主要代謝産物(NS21・OR16では該当なし、Pfでは"
            "プロピオン酸・酢酸)の比生成速度も同様に算出し、炭素・電子収支の"
            "妥当性を確認する。",
        ],
    )
    add_quote(
        doc,
        "The relationship qS(or P) = ΔS (or P) (D/X) we calculated specific "
        "production and consumption rates, where X is the biomass "
        "concentration.",
        "Tannler, Decasper & Sauer (2008), Microb Cell Fact 7:19",
    )

    add_heading(doc, "3.3 Pirtプロットによる維持係数の算出", level=2)
    add_steps(
        doc,
        [
            "各希釈率D(=µ)に対する比基質消費速度qSをプロットする(横軸D、縦軸qS)。",
            "重み付き最小二乗回帰(WLS)により回帰直線を求める。データが希釈率全域で"
            "線形関係を示すことを確認する(定常状態が崩れている場合や増殖速度に"
            "依存した維持代謝がある場合は非線形になりうる)。",
            "回帰直線のy切片(D=0への外挿値)が基質ベースの維持係数 m_glc "
            "(mmol基質/gDW/h)を与える。",
            "基質1molあたりのATP収量(代謝経路の理論値、または酸素消費・P/O比からの"
            "推定値)を用いてmmol substrate/gDW/hをmmol ATP/gDW/hへ換算する"
            "(Coze et al. 2013はP/O=2を仮定してこの換算を行っている、2.1節参照)。",
            "回帰直線の傾きの逆数が最大理論収率 Y_max を与える(qS = µ/Y_max + m_glc "
            "の関係式より)。",
        ],
    )
    add_quote(
        doc,
        "The maintenance energy coefficients were determined as the "
        "intercept of the weighted least square regression line with the "
        "y-axis (Table 3).",
        "Tannler, Decasper & Sauer (2008), Microb Cell Fact 7:19",
    )
    add_note(
        doc,
        "一部の菌株では希釈率全域で線形関係が得られない場合がある"
        "(B. pumilusの例では定常状態の菌体濃度が強く変動し、維持係数を決定できな"
        "かったと報告されている)。線形性が崩れた場合は、より狭い希釈率範囲での"
        "再測定、または増殖速度依存項を含む拡張Pirt式"
        "(qATP = µ/Y_ATP,max + m_ATP + m_g×µ、Le Breton et al. 2003)の適用を検討する。",
    )

    add_heading(doc, "3.4 OR16特有の課題: 不溶性ゴム基質での適用", level=2)
    doc.add_paragraph(
        "Tannler et al.型の古典的Pirt法は、供給培地中の可溶性基質濃度を高精度に"
        "測定できることを前提とする。天然ゴム(latex)は不溶性の高分子であり、"
        "残存基質濃度をケモスタット上清中の溶存濃度として直接測定することが"
        "できない。以下のいずれかの代替戦略を検討する必要がある。"
    )
    add_bullets(
        doc,
        [
            "戦略A(可溶性代替基質): OR16がゴム以外に資化可能な可溶性炭素源"
            "(モデルのギャップフィリングでM9培地増殖が可能とされている、"
            "MODEL_INDEX.md参照)でケモスタットを行い、まずゴム非依存の"
            "「ベースライン維持係数」を実測する。ゴム分解特異的な追加コストは"
            "別途、休止菌体でのLcp活性測定(LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_"
            "20260911.docx)と組み合わせて間接的に見積もる。",
            "戦略B(ゴム消費量の重量法トラッキング): ゴム片(既知重量)を定期的に"
            "回収・洗浄・乾燥して残存重量を測定し、菌体増殖量との対比から比消費速度"
            "を算出する。ケモスタットではなく回分培養+複数回のサンプリングでの"
            "近似Pirt法(µ一定期間ごとの比較)となる。",
            "戦略C(酸素消費速度代用): 天然ゴム分解はLcp/RoxAともに酸素依存の"
            "酸化反応であるため、比酸素消費速度(qO2)をPirtプロットの基質消費速度の"
            "代理指標として用いる。qO2 = µ/Y_O2,max + m_O2の関係式は基質種に依存せず"
            "適用可能。",
        ],
    )
    add_note(
        doc,
        "戦略の選定はプロジェクト側の判断事項とする。本書は3手法を提示するに"
        "とどめ、いずれの手法にも文献的な直接の先例(ゴム分解菌でのPirtプロット"
        "適用例)は見つかっていない点を明記する。",
        color=RED,
    )

    # ==================== 4. Materials ====================
    add_heading(doc, "4. 必要な装置・試薬", level=1)
    add_bullets(
        doc,
        [
            "ケモスタット培養装置: 連続培養槽(作業容量0.5〜2 L程度)、蠕動ポンプ"
            "(供給・排出流量制御)、pH・溶存酸素電極、温度制御系",
            "菌体濃度測定: 分光光度計(OD600)、乾燥重量測定用のろ過装置"
            "(0.22〜0.45 µmメンブレンフィルター)・恒量乾燥用オーブン",
            "基質濃度測定: NS21・Pf(グルコース) — グルコースオキシダーゼキットまたは"
            "HPLC(屈折率検出器)。Pf(プロピオン酸・酢酸) — HPLC有機酸分析"
            "(イオン排除カラム)またはGC",
            "OR16(戦略Bを採る場合): 精密天秤、ゴム片の洗浄・乾燥設備",
            "OR16・NS21(戦略Cを採る場合): 溶存酸素電極またはRöther et al. (2017)法の"
            "蛍光式酸素センサー(既存のLcp/Roxプロトコルと共通機材、"
            "LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docx 4節参照)",
            "統計解析: 重み付き最小二乗回帰が可能な統計ソフト(SPSS、R、Python "
            "scipy.stats等)",
            "培地: NS21・Pf用の規定最小培地(炭素源濃度を段階的に希釈率に応じて"
            "調整可能なもの)、OR16用M9培地(既存モデルのギャップフィリング前提に準拠)",
        ],
    )

    # ==================== 5. Pitfalls ====================
    add_heading(doc, "5. 実務上の落とし穴(文献からの示唆)", level=1)
    add_bullets(
        doc,
        [
            "希釈率がµ_maxに近いと洗い出し(washout)のリスクがあり、定常状態に"
            "到達しないまま菌体濃度が崩壊する。B. pumilusの例では、µ_max近傍の"
            "希釈率で定常状態の菌体濃度が強く変動し、維持係数を決定できなかったと"
            "報告されている(Tannler et al. 2008)。低希釈率側に余裕を持たせた"
            "実験計画とすること。",
            "近縁種からの参考値(2.4節の表)は、あくまで較正の初期値・妥当性確認用"
            "であり、菌種間の系統的な違い(OR16はゴム分解特異的代謝、NS21は"
            "PHA蓄積による炭素・エネルギー配分、Pfは嫌気的プロピオン酸発酵)により"
            "実測値が大きく異なる可能性が高い。",
            "C. necator・A. vinelandiiの参考値2件はDeepSeekの検索結果でDOIが"
            "「推定」表示となっている。プロトコル実施前に原論文(Biotechnology and "
            "Bioengineering誌のKhosravi-Darani et al. 2003およびPena et al. 2000)への"
            "直接アクセスで数値・単位・培養条件を再確認すること。",
            "Pfの参考値(iMR558モデルのNGAM 1.4 mmol ATP/gDW/h)はバッチ培養dFBA"
            "フィッティングによる値であり、ケモスタット定常状態からの直接実測では"
            "ない。フィッティング由来の値は、モデル内の他パラメータ(GAM=35等)との"
            "組み合わせでのみ意味を持つ点に注意。",
            "維持係数は増殖速度に依存して変化しうる(spo0A変異株での維持係数"
            "上昇、L. delbrueckiiでの増殖速度依存項m_g×µ)。単純なPirt式"
            "(qS = µ/Y_max + m_s)が当てはまらない場合は、拡張式の適用を検討する。",
            "P/O比(酸化的リン酸化でのATP/O比)の仮定値(Coze et al. 2013はP/O=2を"
            "使用)は、基質消費速度からATP要求量へ換算する際に結果を左右する。"
            "自施設で換算を行う場合は、採用したP/O値を明記すること。",
            "OR16のゴム基質特有の課題(3.4節)は、戦略選定によって得られる維持係数の"
            "定義・単位系が異なりうる(基質重量ベース vs 酸素消費ベース)。他の2菌種"
            "との比較を行う際は単位を揃えること。",
        ],
    )

    # ==================== 6. Next steps ====================
    add_heading(doc, "6. 次のアクション(優先順位順)", level=1)
    add_bullets(
        doc,
        [
            "【最優先】OR16のゴム基質での維持係数測定戦略(3.4節A/B/C)をプロジェクト"
            "内で決定する。既存のLcp活性測定インフラ(酸素電極等)と共有できる戦略Cが"
            "最も実施コストが低い可能性がある。",
            "Khosravi-Darani et al. (2003)・Pena et al. (2000)の原論文にアクセスし、"
            "DOI・数値・培養条件(特に酸素供給条件、窒素源条件)を直接確認する"
            "(現状は検索結果からの推定DOI)。",
            "iMR558モデル論文(P. freudenreichii DSM 20271, Research Square preprint "
            "https://doi.org/10.21203/rs.3.rs-847511/v1)の完全な書誌情報"
            "(査読誌への正式掲載の有無)を確認する。",
            "新発見のNS21/PHA論文(New Biotechnology 2024, "
            "DOI: 10.1016/j.nbt.2024.08.507)の全文を入手し、窒素制限条件下での"
            "PHA蓄積データが維持代謝の間接的な指標(炭素・エネルギー配分)として"
            "使えるか検討する(優先度⑥ max_pha_fraction較正との連携も視野に)。",
            "3菌種の実測完了後、physiology_dfba.pyの`maintenance`辞書に実測値を"
            "設定し、dFBAシミュレーション結果(増殖収率・共存動態)への影響を"
            "感度分析する。",
        ],
    )

    # ==================== 7. References ====================
    add_heading(doc, "7. 参考文献", level=1)
    add_bullets(
        doc,
        [
            "Coze, F., Gilard, F., Tcherkez, G., Virolle, M-J. & Guyonvarch, A. (2013) "
            "Carbon-Flux Distribution within Streptomyces coelicolor Metabolism: A "
            "Comparison between the Actinorhodin-Producing Strain M145 and Its "
            "Non-Producing Derivative M1146. PLoS ONE 8(12):e84151. "
            "https://doi.org/10.1371/journal.pone.0084151 (全文確認済み)",
            "Tannler, S., Decasper, S. & Sauer, U. (2008) Maintenance metabolism and "
            "carbon fluxes in Bacillus species. Microbial Cell Factories 7:19. "
            "https://doi.org/10.1186/1475-2859-7-19 (全文確認済み)",
            "Gibu, N., Arata, T., Kuboki, S., Linh, D.V., Fukuda, M., Steinbüchel, A., "
            "Kasai, D. (2020) Characterization of the genes responsible for rubber "
            "degradation in Actinoplanes sp. strain OR16. Applied Microbiology and "
            "Biotechnology 104:7367-7376. https://doi.org/10.1007/s00253-020-10700-1 "
            "(維持代謝の数値データは含まれないことを確認)",
            "Khosravi-Darani, K. et al. (2003) Effect of dissolved oxygen tension on "
            "the physiology of Cupriavidus necator in continuous culture. "
            "Biotechnology and Bioengineering. DOI 10.1002/bit.10586 (検索結果からの"
            "推定DOI、原論文での確認が必要)",
            "Pflueger, M. et al. (2018) Metabolic modeling of PHA production from CO2 "
            "and H2 by Cupriavidus necator. Metabolic Engineering. "
            "DOI 10.1016/j.ymben.2018.06.003 (検索結果からの推定DOI、原論文での確認が"
            "必要)",
            "Pena, C. et al. (2000) Effect of dissolved oxygen tension on the "
            "production of alginate and poly-beta-hydroxybutyrate by Azotobacter "
            "vinelandii. Biotechnology and Bioengineering. "
            "DOI 10.1002/(SICI)1097-0290(20000205)67:3<261::AID-BIT1>3.0.CO;2-0 "
            "(検索結果からの推定DOI、原論文での確認が必要)",
            "Genome Scale Constraint-Based Metabolic Reconstruction of "
            "Propionibacterium Freudenreichii DSM 20271 (2021, Research Square "
            "preprint). https://doi.org/10.21203/rs.3.rs-847511/v1 (全文確認済み、"
            "iMR558モデル。査読誌への正式掲載の有無は未確認)",
            "Le Breton, Y. et al. (2003) Analyzing the comparative evolution of the "
            "energy-producing catabolic flux and the growth rate. Lait. "
            "https://doi.org/10.1051/lait:2003032 (全文確認済み、"
            "Lactobacillus delbrueckii subsp. bulgaricus)",
            "Characterization of the conversion system of natural rubber to "
            "poly(3-Hydroxyalkanoate) in Piscinibacter gummiphilus strain NS21T "
            "(2024) New Biotechnology. https://doi.org/10.1016/j.nbt.2024.08.507 "
            "(要旨のみ確認、全文は次のアクション参照)",
            "Linh, D.V., Gibu, N., Tabata, M., Imai, S., Hosoyama, A., Yamazoe, A., "
            "Kasai, D., Fukuda, M. (2019) Complete genome sequence of natural "
            "rubber-degrading, gram-negative bacterium, Rhizobacter gummiphilus "
            "strain NS21(T). Biotechnology Reports 22:e00332. "
            "https://doi.org/10.1016/j.btre.2019.e00332",
            "NCBI/DDBJ Nucleotide, ACCESSION CP015118, ORGANISM Piscinibacter "
            "gummiphilus. https://getentry.ddbj.nig.ac.jp/getentry/na/CP015118 "
            "(NS21の現行正式学名の直接確認)",
            "KEGG GENOME, org code rgu, Piscinibacter gummiphilus NS21 "
            "(Rhizobacter gummiphilus NS21). "
            "http://kegg.jp/kegg-bin/show_organism?org=rgu",
        ],
    )
    add_note(
        doc,
        "全ての引用は、DeepSeek V4 Flash(model=deepseek/deepseek-v4-flash:online)に"
        "よるWeb検索で実際に取得したURLと抜粋テキストに基づく(菌種ごとに3回の独立"
        "検索)。詳細な生の検索結果はtmp/deepseek_maintenance_actinomycete.md、"
        "tmp/deepseek_maintenance_ns21.md、tmp/deepseek_maintenance_pf.mdを参照。"
        "本書中の引用は全てClaudeがこの生データと照合して転記しており、創作した"
        "引用は含まれない。DOIが「推定」と明記された3件は、原論文への直接アクセスに"
        "よる確認が未完了である。",
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
