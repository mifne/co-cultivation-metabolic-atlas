"""Build the literature-grounded wet-lab protocol for measuring the two
oxygen-partitioning parameters used by the dFBA consortium model:
polymer_oxygen_fraction (fixed-split scheme in dfba_simulator.py, default
0.25) and polymer_oxygen_half_saturation (Monod-affinity scheme in
audited_dfba.py's ResolvedDFBASimulator, default 0.01 mmol/L).

This is the third protocol in the calibration series (priority #3 in
THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx section 8, tagged
"priority is high" in CALIBRATION_EXPERIMENT_PROTOCOL_20260911.docx 4.7).

Three DeepSeek V4 Flash web searches were run:
- oxygen-km-enzymes: Km(O2) for the rubber oxygenases themselves
  (Lcp/RoxA/RoxB) -- explicitly NOT found. Every located assay paper
  (Braaz 2005, Birke 2015, Hiessl 2014, Watcharakul 2016, Roether 2016)
  runs the Clark-electrode assay at ordinary air-saturated buffer, never
  varying dissolved O2 as the independent variable -- so no one has
  characterized this before, but the same assay geometry can be adapted.
- oxygen-ks-respiration: Ks(O2) for whole-cell respiration itself --
  not found directly, but strong adjacent methodology was: Harrison's
  classical respirometric method for substrate Ks (mirror-image
  applicable to O2), an in-situ pulse-respirometry protocol (Ficara-type,
  Bioprocess and Biosystems Engineering lineage), and a Japanese dynamic
  gassing-out kLa method paper that explicitly describes the critical
  dissolved-oxygen-concentration (Ccrit) concept, which is operationally
  equivalent to Ks(O2).
- oxygen-partitioning: whether anyone has quantified what fraction of
  total OUR in a rubber-degrading culture is attributable to the
  extracellular oxygenases vs. cellular respiration -- not found; every
  Lcp/RoxA/RoxB assay in the literature uses purified enzyme in vitro,
  never whole culture. Bonus finds: (a) every rubber-oxygenase paper in
  this group defines "1 U = 1 umol O2/min", reinforcing (not resolving)
  the unit-definition inconsistency flagged against Roether et al. 2017's
  Bio-protocol "1 U = 1 nmol O2/min" in the Lcp/Rox protocol; (b) a 2012
  Nagaoka University of Technology PhD dissertation that appears to be
  the original OR16/NS21 isolation report, an item that was on the
  Lcp/Rox protocol's still-open action list.

Real URLs/DOIs and direct quotes are in tmp/deepseek_oxygen_km_enzymes.md,
tmp/deepseek_oxygen_ks_respiration.md, and tmp/deepseek_oxygen_partitioning.md.
Claude selected, translated, and adapted the material; every borrowed
number/method is attributed to its source, and negative findings are
stated as such rather than silently omitted.
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
OUT_PATH = ROOT / "docs" / "OXYGEN_PARTITIONING_PROTOCOL_20260912.docx"

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
    r = title.add_run("酸素分配・酸素半飽和定数 測定プロトコル")
    r.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run(
        "polymer_oxygen_fraction・polymer_oxygen_half_saturationを実測するための"
        "実験計画書 — 第1版 2026年9月12日"
    )
    r2.italic = True
    r2.font.color.rgb = MUTED

    add_note(
        doc,
        "結論を先に述べる: 実験的に調べられるパラメータである。DeepSeek V4 Flashによる"
        "3回の独立したWeb検索の結果、このパラメータそのものの実測値は文献上どこにも"
        "存在しないことを確認した(創作ではなく、検索の結果として「見つからなかった」)。"
        "一方で、(1)ゴム分解酵素(Lcp/RoxA/RoxB)の活性測定自体は既にLCP_ROX_ACTIVITY_"
        "ASSAY_PROTOCOL_20260911.docxで確立済みの手法があり、これを酸素濃度を振る形に"
        "拡張すれば測定可能、(2)微生物呼吸の酸素飽和定数を求める一般的な手法"
        "(動的法・in situ pulse respirometry)も文献上確立されている。本書はこの2つを"
        "組み合わせ、実行可能な測定手順として提示する。",
        color=GREEN,
    )

    # ==================== 1. Background ====================
    add_heading(doc, "1. 目的と背景", level=1)
    doc.add_paragraph(
        "dFBAシミュレータは、溶存酸素(DO)を「細胞外酵素によるゴム分解"
        "(Lcp/RoxA/RoxB、いずれもO2依存の(ジ)オキシゲナーゼ)」と「各菌の細胞呼吸」の"
        "間でどう配分するかを決める、実装の異なる2つのスキームを持つ。"
    )
    doc.add_paragraph(
        "簡易版(src/dfba_simulator.py)は、各タイムステップの溶存酸素プール全体のうち"
        "固定比率(polymer_oxygen_fraction、既定値0.25)をゴム分解へ「予約」し、残りを"
        "細胞呼吸に割り当てる。コード中の注記:"
    )
    add_quote(
        doc,
        "Reserve only a configurable share of the current dissolved pool for "
        "polymer cleavage so the operator split cannot let the first process "
        "consume 100% of DO. The fraction must be calibrated from OUR/DO "
        "measurements.",
        "src/dfba_simulator.py, degrade_rubber()内のコメント",
    )
    doc.add_paragraph(
        "より精緻な実装(src/audited_dfba.py, ResolvedDFBASimulatorクラス)は、この"
        "固定比率スキームを廃止し、代わりにMichaelis-Menten型の酸素親和性項を導入する。"
    )
    add_quote(
        doc,
        "affinity = kinetic_do / (polymer_oxygen_half_saturation + kinetic_do)",
        "src/audited_dfba.py, _prepare_oxygen()",
    )
    doc.add_paragraph(
        "既定値はpolymer_oxygen_half_saturation = 0.01 mmol/L。これは、ゴム分解"
        "オキシゲナーゼ自体の見かけの酸素親和性(Km(O2)相当)を表現する意図のパラメータ"
        "である。同時に、細胞呼吸側の酸素取り込み上限にも同じ0.01という値が類似の"
        "Monod式に使われている(cellular uptake limit = "
        "max_uptake_rate × kinetic_do / (0.01 + kinetic_do))。いずれも「実測待ち」の"
        "仮定値であり、低DO条件での3種優位性の符号を左右しうるため、プロジェクトの"
        "優先順位付けで③番目に位置づけられている(THREE_SPECIES_COEXISTENCE_SUMMARY_"
        "20260911.docx 第8章)。"
    )

    # ==================== 2. Literature review ====================
    add_heading(doc, "2. 先行研究のレビュー(直接引用)", level=1)

    add_heading(
        doc, "2.1 ゴム分解オキシゲナーゼのKm(O2) — 見つからなかった", level=2
    )
    doc.add_paragraph(
        "Lcp・RoxA・RoxBの酸素に対するKm値、または酸素濃度依存的な活性測定データを"
        "報告した文献は見つからなかった。重要なのは「なぜ見つからないか」である。"
        "既存のRoxA/Lcp活性アッセイ論文(Braaz et al. 2005、Birke et al. 2015、"
        "Hiessl et al. 2014、Watcharakul et al. 2016、Röther et al. 2016)は、"
        "いずれも同一の標準条件で酸素消費速度を測定しており、酸素濃度そのものを"
        "変数として振った実験は行われていない。"
    )
    add_quote(
        doc,
        "Oxygen consumption was measured polarographically with a Clark-type "
        "oxygen electrode (Rank Brothers, Cambridge, United Kingdom) at 30°C. "
        "The reaction mixture (2 ml) contained 50 mM Tris-HCl buffer (pH 7.5), "
        "0.1% (wt/vol) poly(cis-1,4-isoprene) (synthetic rubber, 99% cis-1,4; "
        "Aldrich), and 0.1 to 0.5 mg of purified RoxA. One unit of enzyme "
        "activity was defined as the amount of enzyme that consumed 1 μmol "
        "of O2 per min under the assay conditions.",
        "Braaz, Armbruster & Jendrossek (2005), Applied and Environmental "
        "Microbiology 71(5):2473-2478, "
        "https://doi.org/10.1128/aem.71.5.2473-2478.2005",
    )
    doc.add_paragraph(
        "同一の緩衝液・電極・反応条件(空気飽和、酸素濃度を固定)が、Birke et al. "
        "(2015)、Hiessl et al. (2014)、Watcharakul et al. (2016)、Röther et al. "
        "(2016)でも繰り返し使われている(いずれも「Oxygen consumption was measured "
        "with a Clark-type oxygen electrode ... at 30°C」という同一の文言)。つまり、"
        "この4菌種・複数論文にわたる標準アッセイは、酸素を「反応の進行をモニターする"
        "読み出し」として使っているのみで、酸素自体を独立変数として振っていない。"
        "したがって本測定は、既存の確立されたアッセイ系をそのまま酸素濃度可変の"
        "実験に転用するだけで実施可能な、真に新規な測定である。"
    )
    add_note(
        doc,
        "副次的な発見: 上記5論文すべてで「1 U = 1 μmol O2/min」という単位定義が"
        "一致して使われていることを確認した。これは、LCP_ROX_ACTIVITY_ASSAY_"
        "PROTOCOL_20260911.docxで指摘した「Röther et al. (2017) Bio-protocolの"
        "1 U = 1 nmol O2/minという定義が、他の同グループ論文群(1 μmol O2/min)と"
        "1000倍異なる」という食い違いについて、後者(μmol)側がこのグループの"
        "標準的な用法であることを追加確認する結果となった。",
        color=GREEN,
    )

    add_heading(
        doc, "2.2 微生物呼吸の酸素飽和定数Ks(O2) — 直接の実測値は見つからず、"
        "転用可能な方法論を発見", level=2
    )
    doc.add_paragraph(
        "Ks(O2)そのものの実測値・標準測定法を直接報告した文献は見つからなかった。"
        "一方で、以下3件の方法論は、目的のKs(O2)測定に転用可能である。"
    )
    doc.add_paragraph("(a) Harrisonの古典的呼吸測定法(基質Ks用、酸素Ksに鏡像的に転用可能):")
    add_quote(
        doc,
        "This method is based on measurement of the rates of oxygen uptake "
        "in a culture sample supplied with different concentrations of "
        "substrate. The substrate concentration supporting an oxygen uptake "
        "rate of one half the maximum rate is presumed to equal Ks. ... a "
        "sample was taken from a carbon-limited chemostat culture, diluted "
        "1 to 5 or 1 to 10 with aerated basal medium lacking the nitrogen "
        "and the carbon components, and placed in the chamber of an oxygen "
        "electrode respirometer. Different concentrations of carbon "
        "substrate were added and the rates of oxygen uptake measured.",
        "Determination of the Monod substrate saturation constant for "
        "microbial growth, https://doi.org/10.1016/0378-1097(87)90047-4",
    )
    doc.add_paragraph(
        "この手法は本来「基質」のKsを酸素電極によるOUR測定から決めるものだが、"
        "変数を入れ替える(基質を飽和濃度で固定し、代わりに酸素濃度を振る)ことで、"
        "Ks(O2)の測定に転用できる。"
    )
    doc.add_paragraph("(b) in situ pulse respirometry(基質パルスによるKsとqO2max同時推定):")
    add_quote(
        doc,
        "In situ respirometric pulse experiments were done according to the "
        "following procedure: (i) the reactor was maintained until stable DO "
        "readings were obtained; (ii) the substrate feeding was stopped and "
        "the aeration maintained; (iii) the DO concentration slowly "
        "increased until reaching a new pseudo-stationary state, called "
        "baseline oxygen concentration (Cb) ... KS and qO2max were also "
        "estimated by respirometry from the injection of pulses of "
        "increasing concentration ... SOURex max followed a Monod-type "
        "kinetic in relation to the substrate concentration (r2 = 1.00).",
        "In situ pulse respirometric methods for the estimation of kinetic "
        "and stoichiometric parameters in aerobic microbial communities, "
        "https://doi.org/10.1016/j.bej.2011.08.001",
    )
    doc.add_paragraph(
        "この手法もパルス対象を「基質」から「酸素そのもの」に置き換えることで、"
        "DO濃度に対するOUR(またはqO2)のMonod型応答曲線を直接フィッティングできる。"
    )
    doc.add_paragraph(
        "(c) 動的法(dynamic gassing-out method)と臨界溶存酸素濃度(Ccrit)の概念:"
    )
    add_quote(
        doc,
        "動的方法では，酸素消費能力が十分に高い菌株を一定濃度以上で培養している"
        "最中に酸素供給を停止し，溶存酸素濃度を酸素消費速度に等しい速度で減少させる．"
        "低酸素状態により微生物の酸素消費速度に影響が生じる臨界溶存酸素濃度Ccritに"
        "達する前に酸素供給を再開し，酸素濃度を飽和濃度に達するまで上昇させる．この"
        "ときの溶存酸素濃度の勾配は酸素移動速度であり... dC/dt = kLa(C*−C) − QO2・X",
        "kLaの測定法，留意点, 生物工学会誌 102巻5号, "
        "https://www.jstage.jst.go.jp/article/seibutsukogaku/102/5/102_102.5_228/"
        "_pdf/-char/ja",
    )
    doc.add_paragraph(
        "この式は本来kLa測定のためのものだが、「臨界溶存酸素濃度Ccrit」という用語自体"
        "が示す通り、DOがCcritを下回ると比酸素消費速度QO2がDO濃度に依存し始める"
        "(=Ks(O2)による律速が始まる)ことが前提とされている。動的法の曝気停止フェーズを"
        "Ccrit以下まで延長し、DO低下曲線からQO2(DO)の関係を求めれば、そのままKs(O2)の"
        "推定に使える。この一般式・手法自体は文献中で明示されているが、Ccritや"
        "Ks(O2)の具体的な測定手順・数値までは今回の検索範囲では見つからなかった。"
    )

    add_heading(
        doc, "2.3 培養系全体でのOUR分配(オキシゲナーゼ vs 呼吸) — 見つからなかった",
        level=2,
    )
    doc.add_paragraph(
        "ゴム分解菌の培養系全体で「全酸素消費速度(OUR_total)のうちゴム分解酵素反応に"
        "使われる割合」を定量した報告は見つからなかった。既存のLcp/RoxA/RoxB研究は"
        "すべて精製酵素を用いたin vitroアッセイに限られており(2.1節参照)、培養系"
        "全体での酸素収支を扱った研究はない。"
    )
    doc.add_paragraph(
        "「細胞外酸化酵素の酸素消費」と「細胞呼吸の酸素消費」を分離定量する一般的な"
        "実験手法(基質特異的阻害剤を用いる方法等)についても、直接該当する報告は"
        "見つからなかった。ただし、酸素消費速度(OUR)を微生物活性の指標として用いる"
        "手法自体は確立されている。"
    )
    add_quote(
        doc,
        "The oxygen consumption rate (OCR) represents a direct, integrative "
        "readout of respiratory activity and has emerged as a sensitive "
        "phenotypic parameter for characterizing microbial physiology across "
        "different metabolic states. ... OCR measurements were additionally "
        "performed under hypoxic conditions (5% O2) by integrating the "
        "Seahorse XFe96 Analyzer with an oxygen-controlled glove box.",
        "Measuring bacterial oxygen consumption rate to probe metabolic "
        "signature and antimicrobial susceptibility, European Biophysics "
        "Journal (2026), https://doi.org/10.1007/s00249-026-01834-7",
    )
    doc.add_paragraph(
        "PHA生産(NADPH再生に伴う酸素要求)と通常呼吸との間の酸素配分についても、"
        "直接の報告は見つからなかった。"
    )
    add_note(
        doc,
        "副次的な発見: OR16・NS21の最初の単離報告と見られる文献を発見した。"
        "「天然ゴム分解菌の単離と分解酵素遺伝子の解析」(長岡技術科学大学, 平成24年"
        "(2012年)博士論文)。抜粋: 「Actinoplanes sp. OR16、Rhizobacter sp. NS21 "
        "の3株を単離した。特にNS21株はこれまでに報告例のないBetaproteobacteriaに"
        "分類される天然ゴム分解グラム陰性菌であることを明らかにした」。"
        "LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docxの未解決アクションのうち"
        "「OR16の最初の単離報告の完全な書誌情報を確認する」に対応する可能性が高い"
        "(著者名・指導教員等の詳細は原文PDFで別途確認要)。"
        "URL: https://lib.nagaokaut.ac.jp/search/drdb/h24/k661.files/"
        "k661-sinsa-yosi.pdf",
        color=GREEN,
    )

    # ==================== 3. Procedure ====================
    add_heading(doc, "3. 測定手順", level=1)
    add_note(
        doc,
        "以下は、2節で確認した「既存の確立された酵素アッセイ・呼吸測定法」を、"
        "酸素濃度を独立変数として振る形に拡張して構成した手順である。既存文献が"
        "この具体的な組み合わせを直接記述しているわけではない箇所は、その都度明記する。",
    )

    add_heading(
        doc, "3.1 ゴム分解オキシゲナーゼのKm(O2)測定(休止細胞・精製酵素共通)",
        level=2,
    )
    add_steps(
        doc,
        [
            "既存のLcp/RoxA/RoxB活性アッセイ(LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_"
            "20260911.docx 3.1〜3.2節、50 mmol/Lリン酸またはTris-HCl緩衝液、"
            "天然ゴムラテックスまたは合成ポリイソプレン基質)をそのまま踏襲する。",
            "基質(ゴム/ポリイソプレン)濃度は飽和条件(0.1% wt/volまたはそれ以上)に"
            "固定し、酵素(または休止細胞)量も固定する。",
            "反応液の溶存酸素濃度のみを変数として複数段階に調整する: 窒素バブリング"
            "による脱気と空気/酸素の再導入を組み合わせ、0(または検出限界近傍)、"
            "25%、50%、75%、100%(空気飽和)、必要なら100%超(純酸素バブリング)の"
            "系列を作る。",
            "各DO水準で、Clark型電極またはPreSens等の蛍光式酸素センサーにより初期"
            "反応速度(v、酸素消費速度)を記録する(Braaz et al. 2005、Birke et al. "
            "2015等と同一のアッセイ条件・電極を使用)。",
            "得られたv対[O2]のデータにMichaelis-Menten式 "
            "v = Vmax・[O2]/(Km(O2)+[O2]) を非線形最小二乗でフィッティングし、"
            "Km(O2)を求める。",
        ],
    )
    add_note(
        doc,
        "この酸素濃度可変プロトコルは、既存の確立されたClark電極アッセイの"
        "独立変数を基質から酸素へ入れ替えただけの拡張であり、装置・緩衝液・"
        "基質調製等の詳細はLCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docxをそのまま"
        "参照できる。文献上この具体的な酸素可変実験そのものの先例は見つかっていない"
        "(2.1節参照)。",
    )

    add_heading(doc, "3.2 全菌体呼吸のKs(O2)測定(動的法の拡張)", level=2)
    add_steps(
        doc,
        [
            "各菌種を、ゴム非存在下(内在性基質のみ、またはコハク酸等の可溶性炭素源)"
            "で対数増殖期まで培養し、密閉可能な小型容器(50〜200 mL)へ移す。",
            "曝気を停止し、DO電極(応答速度の速い光学式センサーが望ましい、"
            "kLaの測定法論文の指摘参照)でDO低下を連続記録する。",
            "臨界溶存酸素濃度Ccritを大きく下回る(ゼロに近い)水準まで意図的に"
            "酸素消費を継続させる(標準的な動的kLa法ではCcrit到達前に曝気を再開するが、"
            "本測定ではCcrit以下の領域こそが目的のデータとなる)。",
            "記録されたDO(t)曲線を時間微分し、比酸素消費速度 QO2(t) = "
            "-(1/X)・dC/dt を各時点のDO水準に対して算出する(dC/dt = kLa(C*-C) - "
            "QO2・X の関係式で、曝気停止中はkLa項がゼロになる点に注意)。",
            "QO2をDO濃度に対してプロットし、Monod式 QO2 = QO2max・DO/(Ks(O2)+DO) "
            "でフィッティングしてKs(O2)を求める。",
        ],
    )
    add_note(
        doc,
        "この手順は、Ficara型in situ pulse respirometry(2.2節(b))とkLa動的法"
        "(2.2節(c))を組み合わせて構成したものであり、Ks(O2)測定への具体的な適用例が"
        "文献上直接確認されたわけではない。特に「Ccritを大きく下回る領域まで意図的に"
        "測定を継続する」という設計は、動的kLa法の通常運用(Ccrit到達前に曝気再開)を"
        "逆手に取ったものであり、菌体への酸素欠乏ストレスに注意が必要(5節参照)。",
        color=RED,
    )

    add_heading(
        doc, "3.3 培養系全体でのOUR分配実測(polymer_oxygen_fractionの直接校正)",
        level=2,
    )
    add_steps(
        doc,
        [
            "OR16(またはNS21)の対数増殖期の菌体を用意し、同一の菌体密度・DO条件で"
            "2条件を並行測定する: (A) ゴム基質なし(内在性呼吸+可溶性炭素源呼吸のみ)、"
            "(B) ゴム基質あり(細胞呼吸+ゴム分解オキシゲナーゼ活性)。",
            "両条件とも、3.2節と同じ手法でOUR(全酸素消費速度)を記録する。",
            "OUR(B) - OUR(A) の差分を「ゴム分解オキシゲナーゼに帰属する酸素消費」と"
            "みなす。",
            "この差分をOUR(B)(全体)で割った値が、実測されたpolymer_oxygen_fraction "
            "に相当する。",
            "この測定を複数のDO水準で繰り返すことで、固定比率(0.25)が妥当かどうか、"
            "またはDO依存的に変化する(Monod型により近い)かどうかを実験的に判定できる。"
            "これにより、dFBAコードの2つの実装(固定比率版 dfba_simulator.py と "
            "Monod親和性版 audited_dfba.py)のどちらがより現実を近似しているかを"
            "経験的に決定できる。",
        ],
    )
    add_note(
        doc,
        "この「基質の有無によるOUR差分法」は、細胞外酸化酵素と細胞呼吸を分離する"
        "確立された標準手法としては文献上確認できなかったが(2.3節)、既存のOUR測定"
        "手法(3.1・3.2節で使用する同一の電極・原理)を組み合わせるだけで実施可能な、"
        "方法論的に妥当な構成である。",
    )

    # ==================== 4. Materials ====================
    add_heading(doc, "4. 必要な装置・試薬", level=1)
    add_bullets(
        doc,
        [
            "溶存酸素測定: Clark型電極(Rank Brothers社製相当、Braaz et al. 2005・"
            "Birke et al. 2015等で使用)またはPreSens等の蛍光式酸素センサー"
            "(応答速度重視、LCP_ROX_ACTIVITY_ASSAY_PROTOCOL_20260911.docx 4節と共通)",
            "酸素濃度調整: 窒素ガスボンベ(脱気用)、圧縮空気または酸素ボンベ"
            "(再酸素化用)、ガス流量計",
            "反応容器: 密閉可能な小型呼吸測定セル(半微量キュベット、1.4 mL程度、"
            "Lcp/Roxプロトコルと共通)、動的法用の中型密閉容器(50〜200 mL)",
            "緩衝液・基質: 既存のLcp/Rox活性測定と共通(50 mmol/Lリン酸緩衝液"
            "またはTris-HCl、天然ゴムラテックスまたは合成ポリイソプレン)",
            "菌体調製: 対数増殖期培養装置、遠心・洗浄設備(休止細胞法用)",
            "データ解析: 非線形最小二乗フィッティングが可能な解析ソフト"
            "(Python scipy.optimize、R nls、Prism等)",
        ],
    )

    # ==================== 5. Pitfalls ====================
    add_heading(doc, "5. 実務上の落とし穴", level=1)
    add_bullets(
        doc,
        [
            "電極の応答遅れ: 特に隔膜型DO電極は低酸素領域での応答が遅く、Km(O2)や"
            "Ks(O2)をゼロ近傍で過小評価するおそれがある。kLaの測定法論文が指摘する"
            "通り、光学式DOセンサーの方が高い攪拌・低酸素領域での追従性に優れる。",
            "菌体への酸素欠乏ストレス: 3.2節のCcrit以下までの意図的な酸素欠乏測定は、"
            "菌体の代謝状態を不可逆的に変化させうる(嫌気代謝への切り替え、ストレス"
            "応答遺伝子の発現等)。測定後の菌体生存率確認、または測定を短時間に抑える"
            "設計が必要。",
            "脱気操作自体が酵素活性に影響する可能性: 窒素バブリングによる激しい撹拌は、"
            "特に精製酵素を使う場合に失活のリスクがある。脱気は緩やかに行い、対照"
            "(脱気操作をしない通常条件)との比較で活性保持を確認すること。",
            "OUR差分法(3.3節)の解釈上の注意: ゴム基質の存在自体が細胞呼吸そのものを"
            "変化させる可能性がある(例: ゴム由来代謝物の取り込みによる呼吸の亢進・"
            "抑制)。OUR(B)-OUR(A)を単純にオキシゲナーゼ由来と解釈する前に、ゴム"
            "基質存在下での細胞呼吸自体の変化がないことを、別途(ゴム分解を阻害した"
            "対照等で)確認することが望ましい。",
            "OR16の菌糸状生育: OR16は放線菌であり菌糸状に生育するため、休止細胞"
            "懸濁液の均一性・電極周辺での酸素拡散に偏りが生じうる(LCP_ROX_ACTIVITY_"
            "ASSAY_PROTOCOL_20260911.docxでも同様の注意点を指摘済み)。",
            "既存文献のアッセイ条件(30°C、pH 7.0〜7.5)は空気飽和条件下で最適化された"
            "ものであり、低酸素条件でも同じpH・温度最適性が成り立つ保証はない。"
            "Km(O2)測定と並行して、低DO条件でのpH・温度依存性も記録しておくことが"
            "望ましい。",
        ],
    )

    # ==================== 6. Next steps ====================
    add_heading(doc, "6. 次のアクション(優先順位順)", level=1)
    add_bullets(
        doc,
        [
            "【最優先】3.1節のKm(O2)測定をOR16由来Lcpで先行実施する。既存のGibu et "
            "al. (2020)由来アッセイ条件がそのまま使えるため、3つの測定の中で最も"
            "着手コストが低い。",
            "3.3節のOUR差分法による polymer_oxygen_fraction の直接実測を、複数DO"
            "水準で行い、固定比率モデルとMonod親和性モデルのどちらが実データに"
            "近いかを判定する。この結果はdfba_simulator.pyとaudited_dfba.pyの"
            "どちらの酸素配分ロジックを今後の標準とするかの意思決定材料になる。",
            "長岡技術科学大学の2012年博士論文(2.3節で発見)の全文を入手し、著者名・"
            "指導教員・OR16/NS21単離条件の詳細を確認する。LCP_ROX_ACTIVITY_ASSAY_"
            "PROTOCOL_20260911.docxの未解決アクションと統合する。",
            "Ks(O2)測定(3.2節)は、動的法をCcrit以下まで延長する非標準的な運用のため、"
            "予備実験で菌体への影響(生存率・代謝の可逆性)を先に確認してから本測定に"
            "進む。",
        ],
    )

    # ==================== 7. References ====================
    add_heading(doc, "7. 参考文献", level=1)
    add_bullets(
        doc,
        [
            "Braaz, R., Armbruster, W. & Jendrossek, D. (2005) Heme-Dependent "
            "Rubber Oxygenase RoxA of Xanthomonas sp. Cleaves the Carbon "
            "Backbone of Poly(cis-1,4-Isoprene) by a Dioxygenase Mechanism. "
            "Applied and Environmental Microbiology 71(5):2473-2478. "
            "https://doi.org/10.1128/aem.71.5.2473-2478.2005",
            "Birke, J., Röther, W. & Jendrossek, D. (2015) Latex Clearing "
            "Protein (Lcp) of Streptomyces sp. Strain K30 Is a b-Type "
            "Cytochrome and Differs from Rubber Oxygenase A (RoxA) in Its "
            "Biophysical Properties. Applied and Environmental Microbiology "
            "81(11):3793-3799. https://doi.org/10.1128/aem.00275-15",
            "Hiessl, S., Böse, D., Oetermann, S., Eggers, J., Pietruszka, J. "
            "& Steinbüchel, A. (2014) Latex Clearing Protein—an Oxygenase "
            "Cleaving Poly(cis-1,4-Isoprene) Rubber at the cis Double Bonds. "
            "Applied and Environmental Microbiology 80(17):5231-5240. "
            "https://doi.org/10.1128/aem.01502-14",
            "Watcharakul, S., Röther, W., Birke, J., Umsakul, K., Hodgson, B. "
            "& Jendrossek, D. (2016) Biochemical and spectroscopic "
            "characterization of purified Latex Clearing Protein (Lcp) from "
            "newly isolated rubber degrading Rhodococcus rhodochrous strain "
            "RPK1. BMC Microbiology 16(1):92. "
            "https://doi.org/10.1186/s12866-016-0703-x",
            "Röther, W., Austen, S., Birke, J. & Jendrossek, D. (2016) "
            "Cleavage of Rubber by the Latex Clearing Protein (Lcp) of "
            "Streptomyces sp. Strain K30: Molecular Insights. Applied and "
            "Environmental Microbiology 82(22):6593-6602. "
            "https://doi.org/10.1128/aem.02176-16",
            "Determination of the Monod substrate saturation constant for "
            "microbial growth. https://doi.org/10.1016/0378-1097(87)90047-4 "
            "(Harrison法、著者・誌名は検索結果に明記されず原文での確認が必要)",
            "In situ pulse respirometric methods for the estimation of "
            "kinetic and stoichiometric parameters in aerobic microbial "
            "communities. https://doi.org/10.1016/j.bej.2011.08.001 "
            "(著者・誌名は検索結果に明記されず原文での確認が必要)",
            "kLaの測定法，留意点. 生物工学会誌 102巻5号. "
            "https://www.jstage.jst.go.jp/article/seibutsukogaku/102/5/"
            "102_102.5_228/_pdf/-char/ja (著者名は検索結果に明記されず原文での"
            "確認が必要)",
            "Determination of Kinetic and Stoichiometric Parameters of "
            "Pseudomonas putida F1 by Chemostat and In Situ Pulse "
            "Respirometry. https://doi.org/10.2202/1934-2659.1304",
            "Measuring bacterial oxygen consumption rate to probe metabolic "
            "signature and antimicrobial susceptibility. European "
            "Biophysics Journal (2026). "
            "https://doi.org/10.1007/s00249-026-01834-7",
            "天然ゴム分解菌の単離と分解酵素遺伝子の解析. 長岡技術科学大学博士論文 "
            "(平成24年度). "
            "https://lib.nagaokaut.ac.jp/search/drdb/h24/k661.files/"
            "k661-sinsa-yosi.pdf (著者名は検索結果に明記されず原文での確認が必要、"
            "OR16/NS21の原単離報告の可能性)",
            "Röther, W., Birke, J. & Jendrossek, D. (2017) Assays for the "
            "detection of rubber oxygenase activities. Bio-protocol 7:e2188. "
            "https://doi.org/10.21769/BioProtoc.2188 (LCP_ROX_ACTIVITY_ASSAY_"
            "PROTOCOL_20260911.docxで全文確認済み)",
        ],
    )
    add_note(
        doc,
        "全ての引用は、DeepSeek V4 Flash(model=deepseek/deepseek-v4-flash:online)"
        "による3回の独立したWeb検索(酵素Km(O2)、呼吸Ks(O2)、酸素分配実測)で実際に"
        "取得したURLと抜粋テキストに基づく。詳細な生の検索結果は"
        "tmp/deepseek_oxygen_km_enzymes.md、tmp/deepseek_oxygen_ks_respiration.md、"
        "tmp/deepseek_oxygen_partitioning.mdを参照。本書中の引用は全てClaudeが"
        "この生データと照合して転記しており、創作した引用は含まれない。3項目とも"
        "「そのもの」の実測値・直接的な先例は見つからなかったが、この事実自体を"
        "隠さず明記した上で、確立された近縁手法からの拡張として実行可能な測定手順を"
        "構成した。",
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
