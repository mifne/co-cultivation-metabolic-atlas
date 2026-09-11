"""Build the Word protocol for the wet-lab calibration experiments needed
to put the three-species (OR16 / NS21 / P. freudenreichii) coexistence
model on measured ground.

Source material: DeepSeek V4 Flash extraction of the existing experiment
plans plus standard-method knowledge (tmp/deepseek_protocol_materials_20260911.md),
verified by Claude against docs/FEED_VALIDATION_EXPERIMENT.md,
docs/ONE_L_JAR_FERMENTER_CONTROL_DESIGN.md and docs/THREE_PUMP_FEED_STRATEGY.md.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "docs" / "CALIBRATION_EXPERIMENT_PROTOCOL_20260911.docx"

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
    r = title.add_run("3種共培養モデル 校正実験プロトコル")
    r.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run(
        "dFBAシミュレータの未校正パラメータを実測で決めるための手順 — 2026年9月11日"
    )
    r2.italic = True
    r2.font.color.rgb = MUTED

    # 1. Scope
    add_heading(doc, "1. 目的と適用範囲", level=1)
    doc.add_paragraph(
        "本プロトコルは、docs/THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx 第8章で"
        "特定した未校正パラメータを実測で決定し、あわせて本モデルの中心的な予測"
        "「3種の優位性は酸素供給条件に対して非単調に変化する」を実験的に検証するための"
        "手順をまとめたものである。"
    )
    doc.add_paragraph(
        "既存の実験計画(FEED_VALIDATION_EXPERIMENT.md、ONE_L_JAR_FERMENTER_CONTROL_DESIGN.md、"
        "THREE_PUMP_FEED_STRATEGY.md)と重複しない範囲を扱い、装置構成・培地・採取時点は"
        "可能な限り既存計画に合わせてある。"
    )

    add_heading(doc, "1.1 先に決めるべき重大な不一致", level=2)
    add_note(
        doc,
        "既存の1 Lジャー設計・流加戦略・供給検証計画は、いずれも OR16 + NS21 + "
        "L. plantarum (WCFS1) の3種系として書かれており、P. freudenreichii への言及は"
        "1件もない(3ドキュメントを全文検索して確認)。一方、本プロトコルが検証対象とする"
        "計算結果は OR16 + NS21 + P. freudenreichii の3種系のものである。",
        color=RED,
    )
    doc.add_paragraph(
        "したがって、実験に入る前に次のいずれかを決定する必要がある。"
    )
    add_bullets(
        doc,
        [
            "案A: 計算研究に合わせ、第3菌種を P. freudenreichii として装置・流加設計を"
            "改訂する(P3ポンプのD-mannitolはWCFS1支援用のため、Pf用の基質・CO2供給・"
            "嫌気性への対応が新たに必要)。",
            "案B: 既存装置計画に合わせ、WCFS1系で先に制御系を確立し、Pf系はPhase 1の"
            "単離菌パラメータ取得までに留める。",
            "案C: 両立させる。Phase 0-1(装置校正・単離菌パラメータ)はどちらの系でも"
            "共通に使えるため先行実施し、Phase 3の3種比較実験の時点で系を確定する。",
        ],
    )
    add_note(
        doc,
        "本プロトコルは案Cを前提に構成している。Phase 0とPhase 1の大半は第3菌種の"
        "選択に依存しないため、決定を待たずに着手できる。",
    )

    # 2. Overview
    add_heading(doc, "2. 全体の流れ", level=1)
    add_table(
        doc,
        ["Phase", "内容", "菌の要否", "目安期間", "得られるもの"],
        [
            ["Phase 0", "装置・培地の物理校正", "不要\n(培地のみ)", "2〜3日",
             "kLa、酸素飽和濃度、緩衝能"],
            ["Phase 1", "単離菌の生理パラメータ取得", "各菌の単独培養", "4〜8週間",
             "維持係数、死滅率、Ks、Vmax、PHA上限、ゴム分解速度、Pf表現型、B12要求性"],
            ["Phase 2", "2菌(OR16+NS21)基準系の確認", "2菌共培養", "1〜2週間",
             "基準系の再現性、測定系のばらつき(CV)の実測値"],
            ["Phase 3", "酸素条件依存性の検証(決定的実験)", "2菌 vs 3菌", "3〜4週間",
             "モデル予測の成否判定"],
            ["Phase 4", "1 Lジャー流加・長期運転", "3菌", "既存計画に接続",
             "168 h運転、制御系確立"],
        ],
        col_widths=[1.8, 4.2, 2.6, 2.0, 5.2],
    )
    doc.add_paragraph()
    add_note(
        doc,
        "Phase 0はPhase 1と並行実施できる。Phase 3はPhase 0のkLa校正が終わっていることが"
        "前提条件となる(酸素条件を再現できなければ検証にならない)。",
    )

    # 3. Phase 0
    add_heading(doc, "3. Phase 0: 装置・培地の物理校正(菌不要)", level=1)
    doc.add_paragraph(
        "最も安価で短時間に終わり、かつPhase 3の前提となるため最初に行う。"
        "モデルのkLaは現在デフォルト50 h⁻¹、酸素飽和濃度は0.25 mmol/Lという仮定値である。"
    )

    add_heading(doc, "3.1 kLa測定(動的ガス抜き法)", level=2)
    add_steps(
        doc,
        [
            "1 Lジャーに実際に使用する培地(またはまず水)を初期液量600〜750 mL入れ、"
            "30 °Cで温度平衡させる。",
            "窒素ガスを吹き込み、DOをゼロ近くまで低下させる。",
            "窒素を止め、所定の通気量・撹拌速度で空気通気を再開し、DO上昇曲線を"
            "1秒間隔で記録する。",
            "撹拌速度200/300/400/500/600 rpm × 通気量0.5/1.0/1.5 vvm の組合せで繰り返す。",
            "天然ゴムを初期投入量(10 g/L相当)加えた条件でも同じ測定を繰り返し、"
            "粘度・粒子分散によるkLa低下を定量する。",
            "DO電極自体の応答時定数を別途測定し(電極を飽和水から脱酸素水へ移した際の"
            "応答曲線)、一次遅れ補正を行う。",
        ],
    )
    doc.add_paragraph(
        "計算式: ln((C* − C₀)/(C* − C_t)) = kLa × t の傾きから kLa (h⁻¹) を求める。"
    )
    add_note(
        doc,
        "落とし穴: 電極の応答遅れを補正しないとkLaを過小評価する。ゴム粒子が電極膜に"
        "付着すると応答が鈍るため、条件ごとに電極を洗浄・再校正する。",
    )

    add_heading(doc, "3.2 酸素飽和濃度 C* の確定", level=2)
    doc.add_paragraph(
        "30 °C・大気圧・実使用培地での飽和溶存酸素濃度を、DOプローブ校正(100%飽和点)と"
        "必要に応じWinkler法で確認する。純水30 °Cの標準値は約0.25 mmol/L(約8 mg/L)だが、"
        "塩濃度により低下するため、実培地で測る。"
    )

    add_heading(doc, "3.3 緩衝能とpH滴定曲線", level=2)
    doc.add_paragraph(
        "モデルは緩衝液50 mmol/L、pKa 7.21、pH目標6.5を仮定している。実培地を2 M NaOHと"
        "1 M HClで滴定し、pH 6.0〜7.5の範囲の緩衝能(mmol/L/pH)を実測する。"
        "これは長期運転での塩基消費量とイオン蓄積の予測に直結する。"
    )

    # 4. Phase 1
    add_heading(doc, "4. Phase 1: 単離菌の生理パラメータ取得", level=1)
    add_note(
        doc,
        "すべて単独培養で行う。混合培養では菌種間相互作用が交絡し、パラメータを"
        "分離できないため。菌体定量は種特異的qPCR(既存計画で採用予定の手法)を用い、"
        "ゴム共存下ではODを菌体量の指標に使わない。",
    )

    add_heading(doc, "4.1 維持係数(維持ATP要求量)— 最優先", level=2)
    doc.add_paragraph("手法: 連続培養(ケモスタット)によるPirt plot。")
    add_steps(
        doc,
        [
            "各菌について、炭素源制限の連続培養を確立する(500 mL〜1 L容)。",
            "希釈率 D を 0.02, 0.04, 0.06, 0.10, 0.15 h⁻¹ の5点に設定する"
            "(増殖の遅いOR16・Pfでは上限を下げる)。",
            "各希釈率で5滞留時間以上経過後、定常状態を確認してサンプリングする。",
            "菌体乾燥重量(gDCW/L)と基質消費速度を測定し、比基質消費速度 q_s を算出する。",
            "μ(= D)に対して q_s をプロットする。",
        ],
    )
    doc.add_paragraph(
        "計算式: q_s = μ/Y_max + m_s。傾きが 1/Y_max、切片が維持係数 m_s "
        "(mmol gDW⁻¹ h⁻¹)。ATP要求量へは基質の代謝経路の化学量論比で換算する。"
    )
    add_note(
        doc,
        "目安: 従属栄養細菌で 0.1〜5 mmol gDW⁻¹ h⁻¹ のオーダー(一般論であり、"
        "本株の値の予測ではない)。所要期間は5条件で2〜3週間。"
        "落とし穴: 低希釈率では定常維持が難しく、死菌蓄積があると維持係数を過大評価する。"
        "CFUまたは生死染色を併用する。",
    )

    add_heading(doc, "4.2 基礎死滅率・飢餓死滅率", level=2)
    doc.add_paragraph("手法: 長期培養のCFU生存曲線。")
    add_steps(
        doc,
        [
            "対数増殖期の菌体を集菌・洗浄する。",
            "飢餓条件(炭素源・窒素源なし)と非飢餓条件(最小限の基質)に分けて懸濁する。",
            "0, 6, 12, 24, 48, 72, 96, 168 時間でサンプリングする。",
            "段階希釈して寒天プレートに塗布し、CFUを計数する。",
            "LIVE/DEAD染色を併用し、培養不能だが生存している細胞(VBNC)による"
            "過小評価を補正する。",
        ],
    )
    doc.add_paragraph("計算式: k_d = −ln(N_t/N_0)/t (h⁻¹)。指数的死滅が成立する区間で平均する。")
    add_note(
        doc,
        "OR16固有の注意: 放線菌は菌糸状に生育するためCFU計数が難しい。ホモジナイズ/"
        "菌糸断片化の条件を先に最適化する。胞子を形成する場合、死滅率を過小評価しうる。",
    )

    add_heading(doc, "4.3 NH4半飽和定数(Monod Ks)", level=2)
    doc.add_paragraph(
        "手法: 窒素制限ケモスタット。希釈率を0.02〜0.10 h⁻¹で5点変え、各定常状態で"
        "残留NH4濃度 S と菌体濃度 X を測る。μ = μ_max·S/(K_s + S) を非線形最小二乗で"
        "フィッティングする。モデルの現在値は0.1 mmol/L。"
    )
    add_note(
        doc,
        "NS21では窒素制限がPHA蓄積を誘導するため、増殖とPHA蓄積のカップリングを"
        "同時に記録する(4.5と同一実験で採取できる)。残留NH4が検出限界を下回ると"
        "Ks推定精度が落ちるため、高感度定量(イオンクロマトまたは酵素法)を用いる。",
    )

    add_heading(doc, "4.4 基質最大取り込み速度(Vmax相当)", level=2)
    doc.add_paragraph(
        "手法: 休止細胞法。洗浄菌体を既知濃度に調整し、単一基質を飽和濃度で与え、"
        "0, 5, 10, 15, 30, 60分で上清の基質濃度を測る。q_s = (S₀ − S_t)/(t·X)。"
        "複数の初期濃度で行えばMichaelis-Menten式からVmaxとKmが得られる。"
        "モデルは全菌一律に20.0 mmol gDW⁻¹ h⁻¹ を仮定しているため、菌種別・基質別の"
        "実測値で置き換える価値が高い。"
    )
    add_note(
        doc,
        "対象基質は、既存流加設計にあるコハク酸・グルタミン酸・マンニトール・"
        "酵母エキス、およびNH4と酸素を優先する。内在性基質による"
        "バックグラウンド消費の対照を必ず置く。",
    )

    add_heading(doc, "4.5 PHA最大蓄積率と組成(3HB/3HV)", level=2)
    doc.add_paragraph(
        "手法: 酸メタノリシス後のGC-FID分析。モデルの上限0.80 g/gDCWは数値暴走防止の"
        "安全値であり実測値ではないため、NS21の実際の上限を決める。"
    )
    add_steps(
        doc,
        [
            "NS21を窒素制限条件で培養し、0, 12, 24, 48, 72 時間で10〜20 mLずつ採取する。",
            "遠心・洗浄後、凍結乾燥または105 °Cで恒量まで乾燥し、乾燥菌体重量を記録する。",
            "乾燥菌体10〜20 mgに、内部標準を含むメタノール:硫酸 = 85:15 (v/v) を1〜2 mL加える。",
            "密閉して100 °Cで4時間(または60 °Cで16時間)加熱する。",
            "冷却後クロロホルム1 mLと蒸留水1 mLを加えて撹拌、遠心してクロロホルム層を回収する。",
            "GC-FIDで分析する(昇温例: 80 °C 1分 → 10 °C/min → 250 °C 5分保持)。",
            "標準PHB/PHBVのメタノリシス生成物で検量線を作成し、3HB・3HVのmol%を算出する。",
        ],
    )
    doc.add_paragraph(
        "計算式: PHA含有率(% DCW) = PHA重量/乾燥菌体重量 × 100。"
        "3HV mol% = 3HVメチルエステル量/全モノマー量 × 100。"
    )
    add_note(
        doc,
        "落とし穴: 密閉加熱は圧力上昇を伴うため耐圧容器を用いる。ゴム粒子が"
        "サンプルに混入するとPHA定量を妨害するため、ゴム存在下では密度勾配遠心などで"
        "菌体と分離する。菌体由来の脂質もメタノリシスされるためバックグラウンド補正を行う。",
    )

    add_heading(doc, "4.6 ゴム分解速度定数(Lcp/Rox活性)— 最優先", level=2)
    doc.add_paragraph(
        "手法: 休止細胞アッセイ(まずこちらを推奨)、必要に応じ精製酵素アッセイ。"
        "モデルは lcp_c5 = 2.5、roxb_c5 = roxa_direct_c5 = 0.75、roxa_oligo_c30 = 0.25 "
        "(mmol gDW⁻¹ h⁻¹, C5換算)を仮定している。"
    )
    add_steps(
        doc,
        [
            "ゴム含有培地でOR16(またはNS21)を培養し、集菌・洗浄して休止細胞懸濁液を作る。",
            "50 mmol/Lリン酸緩衝液(pH 7.0)を酸素飽和させ、30 °Cで平衡化する。",
            "天然ゴムラテックス(脱タンパク処理・超音波で均一分散)を過剰量加える。",
            "休止細胞を加えて反応を開始し、Clark型電極で酸素消費速度を5〜10分連続記録する。",
            "反応停止後の上清からODTD/オリゴイソプレノイドをHPLCまたはLC-MSで定量する。",
            "基質無添加の対照で内在性呼吸を差し引く。",
        ],
    )
    add_note(
        doc,
        "重要: モデルは「グルコース存在下でLcp発現が抑制される」と仮定している"
        "(半飽和0.5 mmol/L)。活性測定時は抑制のかからない炭素源(コハク酸等)で"
        "前培養し、抑制曲線自体もグルコース濃度を振って別途取得する。"
        "ゴムラテックスの粒径・分散状態が活性を大きく左右するため、調製条件を固定して記録する。",
    )

    add_heading(doc, "4.7 酸素分配と酸素半飽和定数", level=2)
    doc.add_paragraph(
        "モデルは「全酸素在庫の25%をゴム分解へ割り当てる」(polymer_oxygen_fraction = 0.25)"
        "と、ゴム分解の酸素半飽和定数0.01 mmol/Lを仮定している。この2つは低kLa条件での"
        "3種優位性の符号を左右するため、優先度が高い。"
    )
    add_steps(
        doc,
        [
            "DO電極付きの密閉小型容器(50〜200 mL)に菌体懸濁液を入れる。",
            "通気を止めて密閉し、DO低下速度(全酸素消費速度 OUR_total)を記録する。",
            "ゴム基質を加えない対照で同じ測定を行い、呼吸のみの OUR_respiration を得る。",
            "差分から OUR_rubber = OUR_total − OUR_respiration を求める。",
            "DOレベルを0.5, 1, 2, 5, 10, 20 %飽和に変えて繰り返し、DO依存曲線を得る。",
        ],
    )
    doc.add_paragraph(
        "計算式: polymer_oxygen_fraction = OUR_rubber / OUR_total。"
        "酸素半飽和定数は v = Vmax·[O₂]/(K_O₂ + [O₂]) をDO依存データにフィットして求める。"
    )
    add_note(
        doc,
        "目安: 好気性細菌の呼吸の K_O₂ は 0.01〜0.1 mmol/L(空気飽和の約0.5〜5%)の"
        "オーダーとされる。ゴム分解酵素側の値は確度の高い一般値がないため、実測が必要。"
        "呼吸阻害剤で分離する方法もあるが、ゴム分解酵素への副作用の確認が要る。",
    )

    add_heading(doc, "4.8 P. freudenreichii の酸素表現型とプロピオン酸生産", level=2)
    doc.add_paragraph(
        "モデルはPfの酸素交換を閉じた境界として扱い、強制的に嫌気型としている。"
        "この仮定は3種の酸素競合の解釈そのものを左右する。"
    )
    add_steps(
        doc,
        [
            "Pfを嫌気条件(N₂/CO₂ = 80:20 等)で前培養する。培地は事前還元し、"
            "レサズリン等の酸化還元指示薬を入れる。",
            "本培養を3条件で行う: 厳密嫌気(O₂ < 0.1 %)、微好気(O₂ 1〜5 %)、好気(大気)。",
            "経時的に菌体量、プロピオン酸、酢酸、乳酸、コハク酸をHPLC"
            "(有機酸カラム)で定量する。",
            "各条件での比増殖速度とプロピオン酸比生産速度を比較する。",
        ],
    )
    add_note(
        doc,
        "注意: Propionibacterium属はCO₂要求性を示すことが知られており、ガス相のCO₂または"
        "重炭酸塩添加が必要。増殖が遅い(世代時間2〜5時間程度)ため培養は48〜72時間必要。"
        "プロピオン酸蓄積による自己阻害があるためpH制御を行う。至適CO₂濃度は株依存で、"
        "確度の高い一般値がないため予備検討が必要。",
    )

    add_heading(doc, "4.9 B12要求性と授受", level=2)
    doc.add_paragraph(
        "現行モデルにはPf由来B12をOR16/NS21が受け取る経路が実装されていない"
        "(b12_evidence.py はGPRからpseudo遺伝子を除くのみ)。実験で要求性の有無を"
        "確認すれば、この欠落を埋めるべきか判断できる。"
    )
    add_steps(
        doc,
        [
            "OR16・NS21をB12不含最小培地で2〜3回継代し、細胞内B12を枯渇させる。",
            "B12を0, 0.1, 1, 10, 100 nMで添加した培養で増殖を比較する。",
            "B12なしで有意に増殖しない場合、要求性ありと判定する。",
            "Pf培養上清のB12濃度をLC-MS/MS(またはL. leichmanniiによる微生物学的定量)で測る。",
            "共培養上清のB12濃度と要求菌の増殖を対応づける。",
        ],
    )
    add_note(
        doc,
        "注意: B12は光分解するため遮光する。要求濃度はnMオーダーと低いため、"
        "器具由来のコンタミに注意。微生物学的定量法は擬似コバラミンも拾うため、"
        "特異性が要るときはLC-MS/MSを使う。",
    )

    # 5. Phase 2
    add_heading(doc, "5. Phase 2: 2菌基準系の確認と測定ばらつきの実測", level=1)
    doc.add_paragraph(
        "OR16 + NS21 の2菌系を、Phase 3で使う予定の条件(まずkLa一定条件)で反復培養し、"
        "PHA量・菌体量・ゴム除去率の生物学的反復間のばらつき(変動係数 CV)を実測する。"
        "この値がPhase 3の必要反復数を決めるため、Phase 3より先に必ず行う。"
    )
    add_note(
        doc,
        "本工程の主目的はパラメータ取得ではなく「測定系のノイズの大きさを知ること」である。"
        "少なくとも5反復を推奨する(3反復ではCVの推定自体が不安定)。",
    )

    # 6. Phase 3
    add_heading(doc, "6. Phase 3: 酸素条件依存性の検証(決定的実験)", level=1)
    doc.add_paragraph(
        "本モデルの最も特徴的な予測は「3種の優位性が酸素供給条件に対して非単調に変化する」"
        "ことである。ここを検証できれば、計算上の仮説が実験的な知見に変わる。"
    )

    add_heading(doc, "6.1 検出可能性からみた条件の選び方", level=2)
    doc.add_paragraph(
        "モデルが予測する3種−2種のPHA差は条件によって大きさが3桁近く違う。"
        "実験には測定ばらつきがあるため、小さい差の条件を先に試しても結論が出ない。"
        "下表は、生物学的反復のCVを15%と仮定したときに、差を検出するために必要な"
        "1群あたりの反復数の概算である(両側α=0.05、検出力80%)。"
    )
    add_table(
        doc,
        ["酸素条件", "予測されるPHA差", "必要反復数(CV=15%)", "実験としての現実性"],
        [
            ["kLa 2 一定", "−51.6 %", "計算上2未満\n→実務上の最小3", "◎ 最も検出しやすい。最優先で実施"],
            ["2/10 6時間4分割", "+25.4 %", "約6", "◎ 実施可能。3種が有利な側の代表"],
            ["10/2 6時間4分割", "+5.7 %", "約109", "× 現実的でない"],
            ["10→2 半日パルス", "+5.6 %", "約113", "× 現実的でない"],
            ["kLa 4 一定", "+4.2 %", "約200", "× 現実的でない"],
            ["kLa 6 一定", "+3.5 %", "約288", "× 現実的でない"],
            ["kLa 50 一定", "−1.7 %", "約1200", "× 現実的でない"],
        ],
        col_widths=[3.4, 3.0, 3.6, 6.0],
    )
    doc.add_paragraph()
    add_note(
        doc,
        "反復数は n ≈ 2·CV²·(z₀.₉₇₅ + z₀.₈)² / δ² による概算。CV = 15%は仮定値であり、"
        "Phase 2で実測したCVに置き換えて再計算すること。CVが10%に下がれば"
        "+25.4%の条件は3反復で足りるが、+5.7%の条件は依然として約48反復を要する。",
    )
    doc.add_paragraph(
        "結論として、最初に実施すべきは「kLa 2 一定」と「2/10 6時間4分割パルス」の2条件"
        "である。前者はモデルが3種の大幅な不利(−51.6%)を、後者は大幅な有利(+25.4%)を"
        "予測しており、符号が逆かつ効果量が大きいため、モデルの非単調性という主張そのものを"
        "少ない反復で検証できる。"
    )

    add_heading(doc, "6.2 実験デザイン", level=2)
    add_bullets(
        doc,
        [
            "構成: 3種(OR16/NS21/Pf = 0.5/0.1/0.03 g/L)と、総接種量を揃えた2種"
            "(OR16/NS21 = 0.525/0.105 g/L)の2群。モデル比較と同じ接種比を用いる。",
            "酸素条件: 「kLa 2 h⁻¹ 相当一定」「2→10 h⁻¹ を6時間ごとに4分割」の2条件。"
            "kLaの実現はPhase 0で作成した撹拌・通気とkLaの対応表を用いる。",
            "共通条件: 24時間、30 °C、pH 6.5制御、初期ゴム10 g/L、乳酸6 mmol/L"
            "(最初の12時間に0.5 mmol/L/h)、追加NH4 0.4 mmol/L(最初の4時間に0.1 mmol/L/h)。"
            "これはモデル比較で固定した条件と同一である。",
            "採取時点: 0, 4, 8, 12, 24時間(既存計画の採取点に合わせる)。",
            "測定項目: 種特異的qPCRによる各菌絶対量、PHA量と3HB/3HV組成(GC-FID)、"
            "残存ゴム重量とFTIR/GPC、有機酸(乳酸・プロピオン酸・酢酸)、NH4、DO・pH・"
            "塩基消費の連続記録。",
            "反復: 各条件・各構成で最低6反復(Phase 2で実測したCVに応じて再計算)。",
        ],
    )

    add_heading(doc, "6.3 事前に固定する判定基準", level=2)
    add_bullets(
        doc,
        [
            "主要評価項目: 24時間後の培養液当たりPHA量(g/L)。",
            "モデル予測が支持されたと判定する条件: kLa 2相当条件で3種が2種を有意に下回り、"
            "かつ2/10分割パルス条件で3種が2種を有意に上回ること(符号の非単調性の再現)。",
            "モデル予測が棄却されたと判定する条件: 両条件とも差が有意でない、または"
            "符号が予測と逆であること。",
            "統計処理: 事前に検定手法(例: Welchのt検定または混合効果モデル)と"
            "多重比較補正を決めて記録し、結果を見てから変更しない。",
            "副次評価項目: ゴム除去率、3HVモル分率、各菌の生残。これらは主要評価項目の"
            "解釈のためであり、単独で成否判定に用いない。",
        ],
    )
    add_note(
        doc,
        "計算研究では「同一総接種量の2種に対しPHAが5%超増加、ゴム除去の低下が5%以内」を"
        "探索用の陽性基準としていたが、これは統計的有意水準ではない。実験では上記のとおり"
        "検出力に基づく基準を別途設定すること。",
    )

    # 7. Phase 4
    add_heading(doc, "7. Phase 4: 1 Lジャー流加・長期運転(既存計画への接続)", level=1)
    doc.add_paragraph(
        "既存の ONE_L_JAR_FERMENTER_CONTROL_DESIGN.md / THREE_PUMP_FEED_STRATEGY.md の"
        "設計に従う。装置構成は栄養ポンプ4本(P1 コハク酸二Na、P2 glutamate/MSG、"
        "P3 D-mannitol、P4 酵母エキス)と酸・塩基ポンプ2本(2 M NaOH、1 M HCl)、"
        "初期液量600〜750 mL、30 °C、pH 6.5、撹拌200〜600 rpm、手動採取1回5 mL。"
    )
    add_bullets(
        doc,
        [
            "第3菌種をPfに変更する場合、P3(D-mannitol、WCFS1支援用)の用途見直しと、"
            "CO₂供給・嫌気/微好気制御の追加が必要になる。",
            "既存計画で「未解決」とされているシミュレータ側の課題(流加による液量増加・"
            "希釈の未計算、手動採取による液量減少の未反映、pH-statのイオン収支未計上、"
            "温度依存性の欠如、撹拌アクションが実RPMでなく未校正kLaである点)は、"
            "Phase 0の実測値を入れる際に同時に修正する。",
            "長期目標は168時間運転。24時間の対照運転が安定してから延長する。",
        ],
    )

    # 8. Method reference
    add_heading(doc, "8. 測定法リファレンス", level=1)
    add_table(
        doc,
        ["測定対象", "手法", "主な装置・試薬", "所要時間の目安"],
        [
            ["kLa", "動的ガス抜き法(または亜硫酸塩酸化法)", "DO電極、マスフローコントローラ", "1条件30分〜1時間"],
            ["維持係数", "ケモスタットによるPirt plot", "連続培養装置、基質定量系", "5条件で2〜3週間"],
            ["死滅率", "CFU生存曲線(＋生死染色)", "寒天培地、インキュベータ", "7〜14日"],
            ["Monod Ks", "窒素制限ケモスタット", "連続培養装置、NH4高感度定量", "5条件で2〜3週間"],
            ["取り込み速度", "休止細胞法", "振盪培養器、HPLC/酵素法", "1菌1基質あたり半日"],
            ["PHA量・組成", "酸メタノリシス + GC-FID", "GC-FID、耐圧バイアル、標準PHBV", "乾燥除き1日"],
            ["ゴム分解活性", "休止細胞アッセイ(酸素電極＋生成物定量)", "Clark型電極、HPLC/LC-MS", "培養3〜7日＋測定半日"],
            ["ゴム分解の進行", "重量減少、FTIR、SEC/GPC、ODTD定量", "FTIR、GPC、LC-MS", "サンプルごと1日"],
            ["菌種別菌体量", "種特異的qPCR/ddPCR", "qPCR装置、菌種固有プライマー", "1バッチ半日"],
            ["有機酸", "HPLC(有機酸カラム)", "HPLC、標準品", "1バッチ半日"],
            ["B12", "LC-MS/MS または微生物学的定量", "LC-MS/MS", "半日〜1日"],
        ],
        col_widths=[3.2, 4.6, 4.4, 3.0],
    )

    # 9. Pitfalls
    add_heading(doc, "9. この3菌に固有の落とし穴", level=1)
    add_heading(doc, "9.1 OR16(放線菌・固形基質)", level=2)
    add_bullets(
        doc,
        [
            "天然ゴムは不溶性の固形基質であり、沈降・凝集によりサンプリングの代表性が"
            "損なわれる。採取直前に撹拌を上げるなどの均一化手順を固定し、記録する。",
            "菌糸状に生育するためOD測定もCFU計数も素直に使えない。qPCRを主軸にし、"
            "CFUを使う場合は菌糸断片化条件を先に確立する。",
            "ゴム粒子と菌体が分離しにくく、乾燥菌体重量にゴムが混入する。PHA定量前に"
            "密度勾配遠心等での分離が必要。",
            "グルコース存在下でLcp発現が抑制されるとモデルは仮定している。活性測定の"
            "前培養炭素源の選択に注意する。",
        ],
    )
    add_heading(doc, "9.2 NS21", level=2)
    add_bullets(
        doc,
        [
            "PHA蓄積は窒素制限で誘導されるため、増殖相とPHA蓄積相を分けた二段階運転が"
            "必要。P2(glutamate)停止のタイミングが結果を左右する。",
            "3HVの生成にはゴム中間体(C30/ODTD)またはプロピオン酸が必要という前提で"
            "モデル化されている(phv_requires_rubber_intermediate = True)。"
            "この前提自体を、プロピオン酸単独添加試験で確認する価値がある。",
            "PhaZ(PHA分解)は現行モデルで完全に閉じている。飢餓時のPHA再動員の有無を"
            "実験で確認しないと、長期培養の予測が外れる可能性がある。",
        ],
    )
    add_heading(doc, "9.3 P. freudenreichii", level=2)
    add_bullets(
        doc,
        [
            "通性嫌気性で、酸素存在下ではプロピオン酸生産が落ち酢酸生産へ傾くとされる。"
            "共培養では他菌の酸素消費で局所的に嫌気化するため、槽全体のDOだけでは"
            "Pfの代謝モードを説明できない可能性がある。",
            "CO₂要求性があるため、ガス相CO₂または重炭酸塩の添加が必要。至適濃度は"
            "株依存で、予備検討が要る。",
            "増殖が遅く、接種量も小さい(0.03 g/L)ため、24時間の実験では検出限界付近に"
            "なりうる。qPCRの定量下限を事前に確認する。",
        ],
    )

    # 10. Open decisions
    add_heading(doc, "10. 着手前に決めるべきこと", level=1)
    add_bullets(
        doc,
        [
            "第3菌種をP. freudenreichiiとするかL. plantarum(WCFS1)とするか(第1.1節)。",
            "ケモスタット装置を確保できるか。確保できない場合、維持係数とKsはバッチ法での"
            "代替(精度は落ちる)になる。",
            "qPCRプライマーの設計・検証(既存計画ではgene registryの菌種固有・単一コピー"
            "候補から選定し、標準曲線・増幅効率・融解曲線・交差増幅なしを事前確認する"
            "方針が定められている)。",
            "GC-FID、HPLC、FTIR、GPC、LC-MS/MSのうち自前で使えるものと外注するものの切り分け。",
            "Phase 1の全項目を実施するか、Phase 3の決定的実験を先行させるか。"
            "後者を選ぶ場合でも、Phase 0のkLa校正だけは必須である。",
        ],
    )

    add_heading(doc, "11. 本プロトコルが主張しないこと", level=1)
    add_bullets(
        doc,
        [
            "本文中の「一般に報告される値のオーダー」は、当該株の実測値の予測ではなく、"
            "実験計画時の桁合わせの目安である。",
            "特定の文献を引用していない。手法名は教科書的に確立したものを挙げているが、"
            "具体的な出典の確認は実施者が行うこと。",
            "必要反復数の概算は仮定したCVに強く依存する。Phase 2の実測値で必ず再計算すること。",
            "本プロトコルの実施はモデルの正しさを保証しない。検証の結果としてモデルが"
            "棄却される可能性も同等に扱う。",
        ],
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
