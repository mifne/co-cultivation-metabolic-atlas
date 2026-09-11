"""Build the Word report summarizing what has been learned about
OR16 + NS21 + P. freudenreichii three-species coexistence value.

Content was synthesized by Claude from a DeepSeek V4 Flash extraction
(tmp/deepseek_coexistence_extraction_20260911.md) that was itself verified
against the primary source docs under docs/.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results" / "third_species_value_20260910"
OUT_PATH = ROOT / "docs" / "THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx"

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


def add_note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9.5)
    run.font.color.rgb = MUTED
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(hdr_cells[i], "1F4D78")
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

    # base style
    normal = doc.styles["Normal"]
    normal.font.name = "Yu Gothic"
    normal.font.size = Pt(10.5)

    title = doc.add_heading(level=0)
    r = title.add_run("OR16 + NS21 + P. freudenreichii\n三種共存系の価値：現状のまとめ")
    r.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run("計算機内(dFBA/GEM)研究のレビュー — 2026年9月11日時点")
    r2.italic = True
    r2.font.color.rgb = MUTED

    add_note(
        doc,
        "本書は docs/ 配下の一次資料(PFREUDENREICHII_VALUE_AUDIT, PFREUDENREICHII_CAUSAL_AUDIT, "
        "AUDITED_SYMBIOSIS_INTERPRETATION, CULTIVATION_MODEL_AUDIT/REPAIR, PHYSIOLOGY_IMPLEMENTATION/"
        "LONG_VALIDATION, RESOLVED_DYNAMICS, EQUAL_BUDGET_COMPARISON, OXYGEN_SCHEDULE_THIRD_SPECIES, "
        "WCFS1_COEXISTENCE_OPTIMIZATION, THREE_PUMP_FEED_STRATEGY, RL_FEDBATCH_COCULTURE_MILESTONES, "
        "ONE_L_JAR_*, NS21_PHA_MODEL_CURATION)を基に作成した。数値・限界事項はすべて原文の記述に基づく。",
    )

    # 1. Purpose / scope
    add_heading(doc, "1. 目的とスコープ", level=1)
    doc.add_paragraph(
        "天然ゴム分解・PHA変換を行う OR16 + NS21 の2種コンソーシアムに、"
        "プロピオン酸産生菌 P. freudenreichii (Pf) を第3の菌種として加えることに"
        "どのような価値があるかを、ゲノムスケール代謝モデル(GEM)と動的フラックスバランス"
        "解析(dFBA)による計算機内シミュレーションで検証してきた一連の研究を整理する。"
    )
    doc.add_paragraph(
        "着想は、乳酸を系内でプロピオン酸へ変換できる Pf を加えることで、NS21のPHA合成に"
        "3-ヒドロキシ吉草酸(3HV)前駆体を供給し、PHA量や組成を改善できるのではないか、"
        "という代謝的な仮説である。"
    )
    add_note(
        doc,
        "重要な前提：ここで扱う結果はすべて湿式実験ではなく計算機内シミュレーションである。"
        "統計的検定は一切行われておらず、各条件は決定論的な1軌道の比較である。",
    )

    # 2. Executive summary
    add_heading(doc, "2. 結論の要約", level=1)
    add_bullets(
        doc,
        [
            "3種優位は条件依存である。中程度の酸素供給(kLa 4〜8 h⁻¹)や、曝気の時間配分が悪い"
            "パルス条件では3種が2種を上回る(最大 +25.4%)一方、酸素供給が非常に低い条件(kLa 2 h⁻¹)"
            "では大きく下回る(-51.6%)。",
            "時間刻み(dt)を粗くすると3種優位が過大評価される。最初期の1時間刻み計算では"
            "+20.19%の増加が見えたが、45秒・22.5秒刻みまで細分化すると優位は縮小・反転し"
            "(-2.62%〜-5.55%)、頑健な優位ではないことが確認された。",
            "3種の効果は「PHA生産の上限を押し上げる」ものではなく、「曝気の時間配分が悪い場合の"
            "落ち込みを緩和する」方向に働いている、というのが現時点で最も整合的な解釈である。",
            "3-ヒドロキシ吉草酸(3HV)組成の上昇は3種特有の効果ではない。2種にプロピオン酸を"
            "直接与えても3HVは上昇するため、Pfの意味は「3HVを作れること」ではなく"
            "「乳酸からプロピオン酸を系内で供給できること」にある。",
            "維持代謝、死滅率、PHA再利用、B12依存性、Pfの酸素表現型、kLaなど、結果を左右する"
            "主要パラメータは実測値による校正がまだ行われていない。現段階の数値はすべて"
            "「校正待ち」であり、実培養での性能予測として扱うことはできない。",
        ],
    )

    # 3. Timeline
    add_heading(doc, "3. 研究の変遷", level=1)
    doc.add_paragraph(
        "3種共存の価値評価は、単発の結論ではなく、時間刻みの粗さや実装上の不整合を"
        "段階的に洗い出しながら評価基準を厳しくしていく反復プロセスとして進められてきた。"
    )
    add_table(
        doc,
        ["日付", "研究", "3種 vs 2種の結果", "位置づけ"],
        [
            ["2026-09-08", "P. freudenreichii 追加価値 再評価\n(VALUE_AUDIT)",
             "1時間刻みで約+20%。0.2/0.1時間刻みで逆転", "最初の楽観的シグナル、dt依存を発見"],
            ["2026-09-08", "1時間刻み陽性の因果診断\n(CAUSAL_AUDIT)",
             "+20.19%の90.4%はNH4閾値スケジュール要因。\n22.5秒刻みで-2.62%〜-5.55%",
             "「陽性」の原因を分解、頑健性を否定"],
            ["2026-09-08", "培養モデル監査/修復\n(MODEL_AUDIT/REPAIR)",
             "41項目の実装修正後、基準細刻みで-2.600%、\n低kLa最細刻みで+3.712%。5%超基準は非該当",
             "実装の不整合(PHA上限処理・酸素配分等)を是正"],
            ["2026-09-08", "生理モデル実装/長期検証\n(PHYSIOLOGY_IMPLEMENTATION/\nLONG_VALIDATION)",
             "dt収束は良好(終点PHA差0.2%前後)。\n3HV最大差0.09485ポイント(基準0.1に近接)",
             "維持代謝・死滅・元素会計を実装、収束確認"],
            ["2026-09-08", "同一資源予算での供給探索\n(EQUAL_BUDGET_COMPARISON)",
             "5%超増産・ゴム除去低下5%以内を工学的候補とする\n設計を固定(結果は次段で実行)",
             "評価基準・候補設計の事前登録"],
            ["2026-09-10", "酸素スケジュールと第3菌種の価値\n(OXYGEN_SCHEDULE_THIRD_SPECIES)",
             "kLa4/6/8一定、パルス条件で+1.8%〜+25.4%。\nkLa2一定で-51.6%、kLa50一定で-1.7%",
             "現時点で最も整理された比較。dt収束済み"],
        ],
        col_widths=[2.2, 4.0, 5.5, 4.0],
    )
    add_note(
        doc,
        "「陽性」の判定基準(同一総接種量の2種に対しPHAが5%超増加、ゴム除去の低下が5%以内)は"
        "AUDITED_SYMBIOSIS_INTERPRETATION_20260908.md で定義された探索用の基準であり、"
        "測定誤差から導いた統計的有意水準ではない。",
    )

    # 4. Key numeric results
    add_heading(doc, "4. 酸素条件ごとの3種−2種比較(最新・dt収束済み)", level=1)
    doc.add_paragraph(
        "OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md より、時間刻み収束を確認した条件のみを示す"
        "(3HV差が未収束の kLa 2一定・kLa 10一定は組成差を割愛)。"
    )
    add_table(
        doc,
        ["酸素条件", "kLa積分(h)", "2種PHA(g/L)", "3種PHA(g/L)", "PHA差(%)", "ゴム除去差(%)"],
        [
            ["kLa 2 一定", "48", "0.048818", "0.023622", "-51.612", "-0.260"],
            ["kLa 4 一定", "96", "0.188870", "0.196861", "+4.231", "+3.504"],
            ["kLa 6 一定", "144", "0.429802", "0.444764", "+3.481", "+0.718"],
            ["kLa 8 一定", "192", "0.594276", "0.612011", "+2.984", "-0.876"],
            ["kLa 10 一定", "240", "0.691782", "0.697130", "+0.773", "-2.139"],
            ["kLa 50 一定", "1200", "0.750835", "0.738437", "-1.651", "-3.902"],
            ["2→10 半日パルス", "144", "0.394558", "0.401599", "+1.784", "+3.160"],
            ["10→2 半日パルス", "144", "0.387878", "0.409681", "+5.621", "+1.665"],
            ["2/10 6時間4分割", "144", "0.346469", "0.434311", "+25.354", "+20.493"],
            ["10/2 6時間4分割", "144", "0.364902", "0.385758", "+5.716", "+1.894"],
        ],
        col_widths=[3.2, 2.2, 2.6, 2.6, 2.2, 2.6],
    )
    doc.add_paragraph()
    doc.add_paragraph(
        "同じ曝気予算(kLa積分144 h)にそろえた5候補で見ると、2種のPHAは0.346〜0.430 g/L"
        "(幅24.1%)、3種は0.386〜0.445 g/L(幅15.3%)に分布する。3種−2種の差が最大になる"
        "「2/10 6時間4分割」は、同時に2種のPHAが最も低くなる条件でもある。つまりこの候補"
        "集合の範囲では、3種は生産量の上限を押し上げるというより、曝気の時間配分が悪い場合の"
        "落ち込みを緩和する方向に働いている。絶対量が最大になるのは「kLa 6 一定」の3種"
        "(0.444764 g/L)であり、どのパルス条件も一定kLaを上回っていない。"
    )

    # 5. Mechanism / interpretation
    add_heading(doc, "5. なぜ条件依存になるのか：現時点の解釈", level=1)
    add_bullets(
        doc,
        [
            "Pfは乳酸をプロピオン酸へ変換し、系内でNS21に3HV前駆体を供給できる代謝経路を"
            "持つ。この経路自体は計算上確認されている。",
            "しかし3種を加えることは同時に、酸素・窒素・培地成分を3菌で分配することでもある。"
            "酸素供給が乏しい条件(kLa 2)では、この分配コストがプロピオン酸供給の利益を"
            "上回り、3種が大きく不利になる。",
            "酸素供給が十分すぎる条件(kLa 50)でも3種の優位は消える。優位が見られるのは"
            "中間的な酸素供給域、特に供給の時間配分が非効率な場合に限られる。",
            "3HV組成の上昇はPfの専売特許ではない。2種にプロピオン酸を直接投与しても3HVは"
            "上昇するため、Pfの本質的な価値は「3HVを作れること」ではなく「乳酸から"
            "プロピオン酸を系内合成できること」に求めるべきである。",
        ],
    )

    # 6. Cross-cutting caveats
    add_heading(doc, "6. 横断的な注意点", level=1)

    add_heading(doc, "6.1 時間刻み依存性", level=2)
    doc.add_paragraph(
        "3種優位の大きさは時間刻みに強く依存する。1時間刻みでの+20.19%は、細分化する"
        "につれて縮小・反転し(45秒刻みで3種0.482322 g/L対2種0.496733 g/L、22.5秒刻みで"
        "3種が-2.62%〜-5.55%劣位)、その後の実装修正を経た比較でも「基準細刻みで-2.600%、"
        "低kLa最細刻みで+3.712%」と、5%超の探索基準を安定して満たす条件は当初確認"
        "されなかった。2026-09-10の酸素スケジュール研究で、dt収束を確認したうえで"
        "初めて+25.4%までの明確な優位条件が複数見つかっている。"
    )

    add_heading(doc, "6.2 未校正パラメータ", level=2)
    doc.add_paragraph(
        "維持代謝要求、飢餓死滅率、PHA再利用、B12依存性、kLa、酸素飽和濃度、Pfの酸素表現型、"
        "ゴム分解酵素の発現動態など、結果を左右する多数のパラメータが実測値による校正を"
        "受けていない。これらはすべて「校正待ち」であり、現在の結論はこの校正が完了するまで"
        "保留扱いとするのが適切である。"
    )

    add_heading(doc, "6.3 背景培地とB12の扱い", level=2)
    doc.add_paragraph(
        "初期培地からグルコースと酵母エキスを除いても、アミノ酸・核酸関連物質・ビタミン等が"
        "残存し、無視できない炭素・窒素源となる(背景有機炭素 約8.89 mmol-C/L、既知全窒素"
        "約3.06 mmol-N/L)。したがって「外部乳酸0」は「ゴムのみが炭素源」を意味しない。"
        "また、Pf由来B12をOR16/NS21が受け取る明示的な経路はモデルに実装されていないため、"
        "現行モデルからB12媒介の共生効果を主張することはできない。"
    )

    add_heading(doc, "6.4 周辺モデル・インフラの状況", level=2)
    add_bullets(
        doc,
        [
            "L. plantarum(WCFS1)GEMを2022年版に更新・修復。定義10成分+協調LPにより、"
            "完全16成分×4条件に比べ供給質量78.0%減・成分数37.5%減で三種の終点/初期比を"
            "約1.121へ揃えることに成功。ただし旧iNF517/iBT721で学習したGPUサロゲートとは"
            "非互換であり、新モデルでの再学習が必要。",
            "4ポンプ・pH制御付き1Lジャーファーメンターの制御設計とスモークテストを実施"
            "(17/17実行で局所判定合格)。ただしfeedは濃度加算のみで液量変化・希釈を扱わず、"
            "kLaも未校正であるため、現状はソフトウェア経路の疎通確認にとどまる。",
            "強化学習による三種共培養・流加制御のマイルストーンを設計中。低流加・最大酸素"
            "移動の固定ルールを強い対照とし、RLがこれを下回る場合は「RLの利点」を主張しない"
            "方針が明記されている。",
        ],
    )

    # 7. Next steps
    # 7. Novelty / interest assessment
    add_heading(doc, "7. 科学的な面白さ・新規性の評価", level=1)
    add_note(
        doc,
        "本章は、DeepSeek V4 Flash(OpenRouter経由)にドキュメント・生データ・"
        "シミュレータのソースコード(src/audited_dfba.py, physiology_dfba.py, "
        "cultivation_numerics.py, resolved_dfba.py, b12_evidence.py 等)を精査させ、"
        "得られた指摘のうちClaudeが原文・コードと突き合わせて検証できたものだけを"
        "採用して構成している。具体的な行番号の引用は、モデルが不正確な番号を"
        "挙げる傾向が確認されたため用いず、関数名・変数名で参照する。",
    )

    add_heading(doc, "7.1 面白いと言える理由", level=2)
    add_bullets(
        doc,
        [
            "酸素供給に対する非単調な共生価値: 3種優位はkLa中間域(4〜8 h⁻¹)や曝気配分の"
            "悪いパルス条件で最大+25.4%に達する一方、酸素が極端に少ない条件(kLa 2)では"
            "-51.6%まで逆転する。同じ曝気予算(kLa積分144 h)でも時間配分だけで結果が"
            "変わる点は、微生物生態学の資源比理論・ストレス勾配仮説(競争と協力が資源条件に"
            "応じて転換するという考え方)と関連しうる現象である。",
            "Pfの役割の明確な切り分け: 「PHA生産の上限を押し上げる」のではなく「曝気配分が"
            "悪い場合の落ち込みを緩和する」というロバスト性向上効果としての位置づけは、"
            "コンソーシアム設計における共生パートナーの価値の測り方として一般性のある視点。",
            "計算科学としての誠実さ: 時間刻み収束の徹底確認、SHA256によるソース一致照合、"
            "否定的結果(kLa 2で-51.6%等)や自らの初期の楽観的シグナル(1時間刻みで+20%)が"
            "細分化で崩れたことを隠さず報告する姿勢は、この分野の計算研究として模範的。",
        ],
    )

    add_heading(doc, "7.2 面白さを割り引く理由", level=2)
    add_bullets(
        doc,
        [
            "結論はすべて未校正パラメータの上に成り立っている。維持代謝、死滅率、kLa、"
            "Pfの酸素表現型、B12授受など、結果の符号を反転させうるパラメータが未校正の"
            "まま使われており、現時点の「発見」は特定の仮定群のもとでの計算上の観測に"
            "とどまる。",
            "手法自体(プロピオン酸添加によるPHBV組成制御)は、Cupriavidus necator等で"
            "報告例のある比較的古典的な戦略であり、新規性は主に「天然ゴム分解菌×"
            "プロピオン酸産生菌」という組み合わせに由来する。",
            "B12媒介共生など、想定されうる主要な生物学的機構がモデルに実装されていない"
            "(b12_evidence.py はGPRからpseudo遺伝子を除去するのみで、B12の授受経路自体は"
            "追加されていない)。",
            "反復・統計検定を伴わない決定論的な1軌道比較であり、湿式実験による検証は"
            "まだ行われていない。",
        ],
    )

    add_heading(doc, "7.3 総合評価", level=2)
    doc.add_paragraph(
        "方法論の質は高く、「酸素条件・時間配分に対する非単調な共生価値」という発見は、"
        "理論的にも実務的にも追いかける価値がある。しかし現時点では仮説生成段階にあり、"
        "説得力を増すには次章で挙げる未校正パラメータの実測、少なくとも1条件での湿式"
        "実験による再現、パラメータ感度解析が必要である。これらが揃えば、微生物生態学と"
        "合成生物学の境界領域での貢献になり得る、というのがバランスの取れた評価である。"
    )

    # 8. Calibration parameters
    add_heading(doc, "8. 実験で校正すべき主要パラメータ", level=1)
    doc.add_paragraph(
        "シミュレータのソースコード(dfba_simulator.py, physiology_dfba.py, "
        "audited_dfba.py, resolved_dfba.py, cultivation_numerics.py, b12_evidence.py, "
        "one_l_jar.py)と関連監査ドキュメントを精査し、3種共存の結論に影響しうる"
        "未校正パラメータを洗い出した。値・定義箇所はすべてコードと照合済みである。"
    )
    add_table(
        doc,
        ["パラメータ", "現在の仮定値", "定義箇所", "影響", "測定方法(概要)"],
        [
            ["DEFAULT_POLYMER_RATES\n(ゴム分解速度定数)", "lcp_c5=2.5,\nroxb_c5=roxa_direct_c5=0.75,\nroxa_oligo_c30=0.25\n(mmol/gDW/h)",
             "dfba_simulator.py\ndFBASimulator", "ゴム分解速度、C30/ODTD供給量、\n3種の炭素フロー全体の起点",
             "精製Lcp/Rox酵素のin vitroアッセイ、\nまたは単離菌のゴム分解培養"],
            ["maintenance\n(維持ATP要求量)", "未設定(空dict)\n→実質0", "physiology_dfba.py\nPhysiologyDFBASimulator",
             "ゼロ増殖時の生存・飢餓耐性。\n長期共存の評価に直結",
             "連続培養/休止細胞実験でPirt式から\n維持係数を決定"],
            ["basal_death_rates /\nstarvation_death_rates", "未設定(空dict)\n→実質0 h⁻¹", "physiology_dfba.py\nPhysiologyDFBASimulator",
             "菌体量減少・長期安定性。\nゼロだと共存を人為的に安定化",
             "長期培養での生菌数(CFU)の\n経時変化(飢餓/非飢餓条件)"],
            ["polymer_oxygen_fraction", "0.25", "dfba_simulator.py\ndFBASimulator",
             "ゴム分解に割く酸素比率。\n低DO時の分解速度・酸素競合",
             "OUR測定とゴム分解速度の同時計測"],
            ["polymer_oxygen_half_saturation", "0.01 mmol/L", "audited_dfba.py\nAuditedDFBASimulator",
             "ゴム分解酵素の酸素親和性", "DO制御下での分解速度のDO依存性"],
            ["nitrogen_half_saturation", "0.1 mmol/L", "resolved_dfba.py\nResolvedDFBASimulator",
             "NH4取り込みのMonod定数。\n窒素制限応答とPHA配分",
             "NH4制限連続培養での希釈率-\n残留NH4濃度の関係"],
            ["max_pha_fraction_g_gdcw", "0.80 g/g\n(数値暴走防止の安全上限)", "dfba_simulator.py\ndFBASimulator",
             "PHA蓄積上限。生産量の\n理論上限を規定", "窒素制限培養でのPHA含有率の\n経時変化(GC/重量法)"],
            ["kLa・KLaCalibration", "デフォルト50 h⁻¹\n(校正パラメータは仮想値)", "dfba_simulator.py\none_l_jar.py KLaCalibration",
             "酸素供給速度。好気代謝・分解・\nPHA生産すべての律速要因",
             "動的ガス抜き法/亜硫酸塩酸化法"],
            ["oxygen_saturation", "0.25 mmol/L", "resolved_dfba.py\nResolvedDFBASimulator",
             "培地への酸素溶解度", "Winkler法またはDOプローブ校正"],
            ["max_uptake_rate", "20.0 mmol/gDW/h\n(全菌共通)", "dfba_simulator.py\ndFBASimulator",
             "基質取り込み上限。競争下の\n資源配分・増殖速度",
             "基質飽和実験(消費速度と\n菌体量から算出)"],
            ["ph_control_target /\nbuffer_mmol_l / pKa", "target=6.5,\nbuffer=50 mmol/L, pKa=7.21", "dfba_simulator.py /\naudited_dfba.py",
             "pH制御・酸塩基添加量。\n各菌の増殖・代謝活性",
             "培養中のpH滴定曲線、\n各菌の至適/阻害pH域"],
            ["B12授受パラメータ", "未校正(注釈のみ、\n授受経路は未実装)", "b12_evidence.py\ninspect_and_curate",
             "B12供給による代謝補助の\n有無・大きさ", "最小培地でのB12要求性、\n添加有無での増殖比較"],
            ["Pfの酸素表現型", "閉鎖(嫌気的)と仮定", "CULTIVATION_MODEL_AUDIT\nB02(コード上は交換境界で表現)",
             "低DO条件での競争・生存。\n酸素分配の妥当性",
             "Pf単離株の微好気/嫌気培養での\n増殖・生存確認"],
        ],
        col_widths=[3.0, 3.2, 3.2, 3.6, 3.8],
    )
    doc.add_paragraph()
    add_heading(doc, "優先順位(結論を反転させうる可能性が高い順)", level=2)
    add_bullets(
        doc,
        [
            "① DEFAULT_POLYMER_RATES(ゴム分解速度) — 全炭素フローの起点。OR16/NS21の"
            "分解速度バランスがNS21へのPHA原料供給とPfへの有機酸供給を決める。",
            "② maintenance(維持ATP要求量) — 実質ゼロだと飢餓耐性を過大評価し、低基質条件"
            "でのPf・OR16の早期死滅を見逃す。",
            "③ polymer_oxygen_fraction と polymer_oxygen_half_saturation(酸素分配・親和性)"
            " — 低DO条件でのゴム分解と呼吸の酸素競合を決め、低kLaでの3種優位性の符号を"
            "左右しうる。",
            "④ basal_death_rates / starvation_death_rates(死滅率) — ゼロのままでは、"
            "競争に負けた菌(特に低接種量のPf)も生き残り続け、共存が人為的に安定化する。",
            "⑤ nitrogen_half_saturation(NH4半飽和定数) — NH4枯渇タイミングとPHA蓄積"
            "開始時期を左右し、3種のPHA優位性の大きさに影響する。",
            "⑥ max_pha_fraction_g_gdcw(PHA上限) — 現在の80%はNS21の実測値ではなく数値"
            "暴走防止の安全上限であり、実際の上限次第でPHA優位の大きさが変わる。",
            "⑦ kLa・酸素飽和濃度 — 酸素供給は好気代謝全体の律速因子であり、実機のkLaが"
            "校正されない限り、酸素条件依存の3種優位性を実培養へ外挿できない。",
        ],
    )

    add_heading(doc, "9. 次の課題", level=1)
    add_bullets(
        doc,
        [
            "kLa 4〜8 h⁻¹の間をさらに細かく刻み、3種が有利になる窓の境界を特定する。",
            "3分割以上の高頻度パルスで、結果が一定kLaに収束するかを確認する。",
            "3種が有利な条件で供給プロファイルも再最適化し、構成差が供給最適化後も残るかを"
            "検証する。",
            "kLa、酸素飽和、維持代謝、死滅率、Pf表現型の実測値による校正。ここまでの結論は"
            "すべてこの校正待ちの状態にある。",
            "上記の校正・境界探索が完了した後に、大量教師データ収集とRL学習フェーズへ進む。",
        ],
    )

    # 8. Methodology note
    add_heading(doc, "10. 方法論に関する注記", level=1)
    doc.add_paragraph(
        "本書で扱う結果はすべて、ゲノムスケール代謝モデル(GEM)と動的フラックスバランス解析"
        "(dFBA)による計算機内シミュレーションである。各比較は決定論的な1軌道であり、反復"
        "実験のばらつきや統計的検定は含まれない。多くの研究で、比較前に評価基準・条件設計"
        "を固定し、共有ソース(シミュレータ・GEM・監査コード)のSHA256ハッシュが全研究で"
        "一致することを事前に照合してから集計する運用がとられている。"
    )

    # 9. Figures
    add_heading(doc, "11. 図(酸素スケジュールと第3菌種の価値, 2026-09-10)", level=1)
    fig1 = FIG_DIR / "fig1_third_species_axis.png"
    fig2 = FIG_DIR / "fig2_equal_budget_family.png"
    fig3 = FIG_DIR / "fig3_representative_trajectories.png"

    if fig1.exists():
        doc.add_paragraph("図1: 酸素条件に対するPHA差(全条件)と3HVモル分率の2種→3種の変化")
        doc.add_picture(str(fig1), width=Cm(16))
    if fig2.exists():
        doc.add_paragraph("図2: 同一曝気予算(kLa積分144 h)でそろえた5候補のPHA比較")
        doc.add_picture(str(fig2), width=Cm(16))
    if fig3.exists():
        doc.add_paragraph(
            "図3: 代表軌道(PHA/3HV/溶存O2/Pf生菌量。実線=3種、破線=2種)"
        )
        doc.add_picture(str(fig3), width=Cm(16))
    add_note(
        doc,
        "図の白抜きマーカーおよび元図中の注記は時間刻み未収束・未評価を示す"
        "(詳細は OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md を参照)。",
    )

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
