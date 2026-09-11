from __future__ import annotations

import json
from pathlib import Path
from datetime import date

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DOCX_PATH = RESULTS / "program_inspection_report.docx"
FIG_PATH = RESULTS / "program_architecture_diagram.png"

NAVY = "0B2545"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
CALLOUT = "F4F6F9"
GREEN = "1B6E4F"
GOLD = "7A5A00"
RED = "9B1C1C"
MUTED = "5B6573"


def font(path: str, size: int, bold: bool = False):
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def make_architecture_figure(path: Path) -> None:
    """Create a compact, publication-style architecture diagram with PIL."""
    img = Image.new("RGB", (1800, 1040), "white")
    d = ImageDraw.Draw(img)
    f_title = font(r"C:\Windows\Fonts\segoeuib.ttf", 36, True)
    f_box = font(r"C:\Windows\Fonts\segoeui.ttf", 25)
    f_small = font(r"C:\Windows\Fonts\segoeui.ttf", 21)
    f_note = font(r"C:\Windows\Fonts\segoeui.ttf", 20)
    d.text((70, 35), "3-GEM dFBA-RL program architecture", fill="#0B2545", font=f_title)

    boxes = {
        "cli": (80, 150, 360, 290, "CLI / config", "main.py\nargs + paths", "#E8EEF5"),
        "models": (470, 150, 790, 290, "SBML GEM loader", "3 XML models\nCOBRApy objects", "#E8EEF5"),
        "sim": (900, 150, 1230, 290, "dFBA simulator", "state + bounds\nΔt integration", "#DCEAF7"),
        "env": (1340, 150, 1710, 290, "RL environment", "16 observations\n5 continuous actions", "#DCEAF7"),
        "solver": (600, 390, 1200, 560, "FBA solver router", "CPU exact LP / cuOpt GPU /\nCUDA surrogate + HiGHS guard", "#E7F1EC"),
        "ppo": (1340, 390, 1710, 560, "PPO MlpPolicy", "SB3 + VecNormalize\n~142k trainable params", "#F4EBDD"),
        "out": (840, 690, 1260, 850, "Artifacts", "checkpoints · TensorBoard\nJSON · CSV · figures", "#E8EEF5"),
        "tests": (120, 690, 590, 850, "Tests / diagnostics", "42 collected\nresiduals · fallbacks", "#FDEDEC"),
    }
    for x1, y1, x2, y2, title, body, fill in boxes.values():
        d.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=fill, outline="#5B6573", width=3)
        tw = d.textbbox((0, 0), title, font=f_box)[2]
        d.text(((x1+x2-tw)//2, y1+24), title, fill="#0B2545", font=f_box)
        for i, line in enumerate(body.split("\n")):
            tw = d.textbbox((0, 0), line, font=f_small)[2]
            d.text(((x1+x2-tw)//2, y1+78+i*32), line, fill="#263238", font=f_small)

    def arrow(a, b, color="#2E74B5", width=5):
        d.line((a[0], a[1], b[0], b[1]), fill=color, width=width)
        import math
        ang = math.atan2(b[1]-a[1], b[0]-a[0])
        L = 18
        pts = [(b[0], b[1]), (b[0]-L*math.cos(ang-0.45), b[1]-L*math.sin(ang-0.45)), (b[0]-L*math.cos(ang+0.45), b[1]-L*math.sin(ang+0.45))]
        d.polygon(pts, fill=color)

    arrow((360, 220), (470, 220))
    arrow((790, 220), (900, 220))
    arrow((1230, 220), (1340, 220))
    arrow((1065, 290), (900, 390), color="#1B6E4F")
    arrow((1120, 390), (1340, 475), color="#1B6E4F")
    arrow((1340, 250), (1200, 690), color="#2E74B5")
    arrow((1525, 560), (1150, 690), color="#7A5A00")
    arrow((600, 770), (840, 770), color="#9B1C1C")
    d.text((85, 930), "Exploration path: GPU surrogate batches requests; exact HiGHS remains the scientific audit/fallback path.", fill="#5B6573", font=f_note)
    d.text((85, 970), "GPU memory is assigned by process/shard (not pooled). Joint FBA is a performance experiment, not an OptCom objective.", fill="#5B6573", font=f_note)
    img.save(path, dpi=(220, 220))


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, dxa: int):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_cell_text(cell, text, bold=False, color="263238", size=9.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")


def set_row_cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant = tr_pr.find(qn("w:cantSplit"))
    if cant is None:
        cant = OxmlElement("w:cantSplit")
        tr_pr.append(cant)


def add_table(doc, headers, rows, widths, header_fill=LIGHT_GRAY, font_size=9.2):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    for i, h in enumerate(headers):
        set_cell_shading(table.rows[0].cells[i], header_fill)
        set_cell_text(table.rows[0].cells[i], h, bold=True, color=NAVY, size=font_size)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            if ridx % 2 == 1:
                set_cell_shading(cells[i], "FAFBFC")
            set_cell_text(cells[i], value, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_callout(doc, label, text, fill=CALLOUT, label_color=NAVY):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    set_row_cant_split(table.rows[0])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.08
    r = p.add_run(label + "  ")
    r.bold = True; r.font.color.rgb = RGBColor.from_string(label_color); r.font.size = Pt(10.5)
    r2 = p.add_run(text)
    r2.font.size = Pt(10.5); r2.font.color.rgb = RGBColor.from_string("263238")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.5 if level == 0 else 0.75)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.167
    p.add_run(text)
    return p


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.size = Pt(9); run.font.color.rgb = RGBColor.from_string(MUTED)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def set_run_font(run, size=None, bold=None, color=None, name="Calibri"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if color: run.font.color.rgb = RGBColor.from_string(color)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    return p


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.10
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix); set_run_font(r, bold=True, color=NAVY)
        r2 = p.add_run(text[len(bold_prefix):]); set_run_font(r2)
    else:
        r = p.add_run(text); set_run_font(r)
    return p


def build_doc():
    make_architecture_figure(FIG_PATH)
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(1.0); sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(1.0); sec.right_margin = Inches(1.0)
    sec.header_distance = Inches(0.492); sec.footer_distance = Inches(0.492)

    # standard_business_brief token map
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"; normal.font.size = Pt(11); normal.font.color.rgb = RGBColor.from_string("263238")
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri"); normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.10
    for level, size, color, before, after in [(1,16,BLUE,16,8),(2,13,BLUE,12,6),(3,12,DARK_BLUE,8,4)]:
        style = doc.styles[f"Heading {level}"]
        style.font.name = "Calibri"; style.font.size = Pt(size); style.font.bold = True; style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri"); style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.paragraph_format.space_before = Pt(before); style.paragraph_format.space_after = Pt(after); style.paragraph_format.line_spacing = 1.10
    # Header/footer
    hp = sec.header.paragraphs[0]
    hp.text = "co-cultivation | Program Inspection Report"
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in hp.runs: set_run_font(r, size=9, color=MUTED)
    fp = sec.footer.paragraphs[0]; add_page_number(fp)

    # Title page
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(60); p.paragraph_format.space_after = Pt(8)
    r = p.add_run("co-cultivation プログラム点検報告書"); set_run_font(r, size=26, bold=True, color=NAVY)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(18)
    r = p.add_run("3-GEM dFBA + PPO 強化学習システムの構造・実装レビュー"); set_run_font(r, size=15, color=BLUE)
    add_callout(doc, "レビュー範囲", "main.py、src/、models/sbml/final_consortium/、tests/、docs/、results/ を読み取り、実行経路・データフロー・テスト・既存ベンチマークを点検。コードの変更は行っていない。", fill=LIGHT_BLUE)
    add_table(doc, ["項目", "内容"], [
        ("点検日", "2026-08-31"),
        ("対象", "天然ゴム分解・PHA変換コンソーシアム（OR16 / NS21 / L. plantarum）"),
        ("主要技術", "COBRApy、dFBA、Stable-Baselines3 PPO、SciPy/HiGHS、cuOpt、PyTorch CUDA surrogate"),
        ("点検方式", "静的読解 + 既存ベンチマークの確認 + pytest の実行"),
    ], [1900, 7460], font_size=10)
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(24)
    r = p.add_run("判定の読み方"); set_run_font(r, size=12, bold=True, color=DARK_BLUE)
    add_body(doc, "本報告書の「現状」はコードと保存済みログに基づく事実、「注意」は再現性・科学的妥当性・運用安全性の観点からの点検所見、「推奨」は次に実施すべき具体的な対応を示す。")
    doc.add_page_break()

    add_heading(doc, "1. エグゼクティブサマリー", 1)
    add_callout(doc, "総合判定", "研究用プロトタイプとしては、3つのGEMを用いたdFBAとPPOを接続し、GPUサロゲート、厳密CPU監査、並列環境、診断ログまで一通り備えた構成になっている。一方、既定CLIがCPU/GLPKであること、cuOptの実運用経路が数値失敗時にフォールバックすること、評価環境の受け渡しがコード上分離されていないことから、論文・本番計算では実行モードを明示し、評価経路を整理する必要がある。", fill="FFF7E6", label_color=GOLD)
    add_body(doc, "プログラムは、(i) SBMLモデルをCOBRApyへ読み込むモデル層、(ii)動的境界条件と物質収支を更新するdFBA層、(iii) FBAのCPU/GPUソルバーを切り替える層、(iv) Gymnasium環境とPPOを接続するRL層、(v)ログ・チェックポイント・図表を出力する評価層、の5層に分かれる。")
    add_bullet(doc, "3 GEMの実体ファイルは存在し、モデル選択テストでは OR16、NS21、L. plantarum の意図した3種が選択されることを確認。")
    add_bullet(doc, "FBAは separate（種ごとに3 LP）と joint（ブロック対角の1 LP）を切り替えられる。jointは共有培地をLP内で最適化するOptComではなく、独立LPをまとめた性能実験である。")
    add_bullet(doc, "GPUサロゲートは、化学量論の零空間／実行可能辞書をGPUで一括評価し、OOD・境界違反・残差異常・定期監査失敗時に厳密HiGHSへ戻す設計である。")
    add_bullet(doc, "現行ログでは、RTX 4060 Laptop上の16環境でCPU HiGHS 12.24 transition/s、GPU surrogate 85.83 transition/s（7.01倍）が記録されている。ただしこれは近似探索経路の測定であり、cuOptの厳密LPが同じ速度で動くことを意味しない。")

    add_heading(doc, "2. システム構造", 1)
    doc.add_picture(str(FIG_PATH), width=Inches(6.5))
    cap = doc.add_paragraph("図1　プログラムの主要モジュールとデータフロー。青は制御フロー、緑はFBAソルバー分岐、赤は検証・診断、黄はPPO学習を示す。")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER; cap.paragraph_format.space_after = Pt(4)
    for r in cap.runs: set_run_font(r, size=9, color=MUTED)
    add_body(doc, "実行の中心は main.py の train サブコマンドである。CLI引数からモデル・ソルバー・GPU割当・PPO設定を構築し、SubprocVecEnv または DummyVecEnv で環境を作成する。各環境の1ステップは、アクションを栄養供給量とKLaへ変換し、dFBASimulator が動的境界を設定して3種のFBAを解き、共有培地・バイオマス・PHA・ゴム残量を更新する。")
    add_table(doc, ["層", "主なファイル", "責務", "点検結果"], [
        ("入口・設定", "main.py / config.py", "CLI、モデルパス、学習・評価・GPU設定", "実行モードは豊富。ただし既定値はCPU/GLPK。"),
        ("モデル", "src/utils.py; models/sbml/...", "SBML読込、3種の選択、初期条件", "意図した3種を選択。読込失敗は静かにスキップされる。"),
        ("dFBA", "src/dfba_simulator.py", "境界更新、FBA、状態積分、質量収支、ログ", "本体は1311行。機能豊富だが責務が集中。"),
        ("ソルバー", "src/cuopt_solver.py; community_solver.py; fba_surrogate*.py", "CPU exact、cuOpt、GPU surrogate、joint LP", "監査・残差検査・フォールバックあり。"),
        ("RL", "src/rl_environment.py; ppo_agent.py", "観測16、行動5、報酬、PPO MLP、正規化", "PPOは約142kパラメータ。評価経路に改善余地。"),
        ("検証・出力", "tests/; callbacks.py; results/", "pytest、科学指標、TensorBoard、JSON/CSV/図", "42テストを収集。GPU/近似のラベル分離が必要。"),
    ], [1500, 2400, 3100, 2360], font_size=8.8)

    add_heading(doc, "3. 実行アルゴリズムとデータフロー", 1)
    add_heading(doc, "3.1 1ステップの処理", 2)
    steps = [
        ("1", "PPOが16次元の正規化観測から5次元の連続アクションを出力。"),
        ("2", "アクションを菌種別栄養補給、酵母エキス供給、動的KLaへ変換。"),
        ("3", "dFBAが培地濃度、ミカエリス・メンテン型取り込み上限、物理的消費上限、LCP誘導、ecFBA様総フラックス制約を反映。"),
        ("4", "separateでは3種それぞれのLPを解く。surrogateでは同じ時刻の3種要求をGPUサービスへまとめ、候補を一括評価。"),
        ("5", "解から増殖率、交換フラックス、PHA、CO2、培地濃度を更新し、ゴム分解を外部酵素モデルで進める。"),
        ("6", "報酬 = ゴム分解増分×10 + PHA増分×500 − pHペナルティ − 生存閾値ペナルティ。全種が0.01未満なら終了。"),
        ("7", "PPOがrolloutを収集し、GAE・クリッピング付き更新、チェックポイント・TensorBoard・履歴保存を行う。"),
    ]
    add_table(doc, ["段階", "処理"], steps, [900, 8460], font_size=9.4)
    add_heading(doc, "3.2 観測・行動・モデル規模", 2)
    add_table(doc, ["項目", "実装値", "意味"], [
        ("観測", "16次元（3種バイオマス + pH/DO/Glc/ゴム/C30/ODTD/PHA/BS/AA3/時刻/phase）", "すべて0–1へ正規化。"),
        ("行動", "5次元 Box[0,1]", "OR16/NS21/LP個別補給、酵母エキス、KLa。遺伝子KOは現行行動空間に直接は含まれない。"),
        ("PPO", "MlpPolicy; pi=[256,256], vf=[256,256]", "SB3標準PPO。main.py既定はn_steps=4096、batch=512、gamma=0.999。"),
        ("概算パラメータ", "約141,835（約142k）", "actor約71,434 + critic約70,401。"),
        ("時間刻み", "dt=0.2 h", "max_steps=840なら168 h/episode。"),
    ], [1900, 3000, 4460], font_size=9)

    add_heading(doc, "4. 使用モデルと計算規模", 1)
    add_body(doc, "final_consortium にあるXMLをCOBRApyで読み込んだ実測規模を以下に示す。反応数・代謝物数・遺伝子数は、LPの大きさとモデル読込／境界更新コストの目安であり、実行時間は疎性、固定境界、目的関数、ソルバー設定にも依存する。")
    add_table(doc, ["GEM", "役割", "反応", "代謝物", "遺伝子", "交換反応"], [
        ("Actinoplanes sp. OR16", "Lcp型ゴム分解", "2,585", "1,679", "1,593", "285"),
        ("Rhizobacter gummiphilus NS21", "Rox型ゴム分解 + PHA", "2,796", "1,787", "1,390", "335"),
        ("Lactobacillus plantarum", "代謝安定化 / BS", "756", "651", "516", "106"),
        ("合計", "3 GEM", "6,137", "4,117", "3,499", "726"),
    ], [2300, 2350, 900, 900, 900, 1010], font_size=9)
    add_callout(doc, "重要な解釈", "3 GEMのjoint LPは反応変数6,137、質量収支行4,117のブロック対角問題である。shared mediumの更新はLP解後に行われるため、jointと呼んでも種間交換をLPの同一目的で最適化する構造ではない。", fill=LIGHT_BLUE)

    add_heading(doc, "5. GPU・CPUの実装点検", 1)
    add_heading(doc, "5.1 GPU経路", 2)
    add_body(doc, "GPUは2種類の経路を持つ。cuOptは厳密LPの候補だが、現行RTX 4060ではランク削減PDLP・crossover・数値監査が必要で、保存済みログでは単一環境のcuOptがCPU GLPKより速いとは限らない。実運用の探索高速化は、HiGHSで生成した可行フラックス辞書をGPUでバッチ評価する surrogate 経路で実現している。")
    add_table(doc, ["経路", "計算場所", "保証", "点検所見"], [
        ("GLPK / SciPy-HiGHS", "CPU", "厳密LP（実装上の基準）", "最終評価・論文数値に適する。"),
        ("cuOpt", "GPU", "残差・境界・目的値監査後に採用", "失敗時はCPUへ戻る。current logでは数値エラー／速度不足が観測済み。"),
        ("CUDA surrogate", "GPU推論 + CPU監査", "零空間／可行辞書、OOD・残差・定期HiGHS監査", "探索用。近似採用率とfallback数を必ず併記する。"),
        ("GPU割当", "プロセス環境変数", "rank→GPUのround-robin", "VRAM共有ではなく協調的なプロセス並列。"),
    ], [1700, 1700, 3000, 1960], font_size=9)
    add_heading(doc, "5.2 保存済みベンチマーク", 2)
    add_table(doc, ["測定", "結果", "意味"], [
        ("16環境 end-to-end", "CPU HiGHS 12.24 transition/s; GPU surrogate 85.83 transition/s; 7.01倍", "GPU探索経路の速度差。surrogate acceptance 99.42%、CPU fallback 17。"),
        ("GPU saturation", "batch 128で平均GPU利用率98.3%、VRAM最大7,935 MiB", "バッチを増やせばGPUを高稼働できるが、スループットはbatch 64が最大。"),
        ("exact FBA collection", "4 workersで7.44 exact LP/s", "CPU exact LPは並列化しても飽和点がある。"),
        ("3-GPU提案", "GPUあたり24 GB、合計72 GB（非共有）", "seed/agent/環境群を独立shardにする前提。実機は未測定。"),
    ], [2200, 3550, 2610], font_size=9)
    add_callout(doc, "購入判断に関する注意", "保存済みの7.01倍は「RTX 4060上のGPU surrogate探索 vs CPU HiGHS」のend-to-end値であり、「cuOpt厳密LP vs CPU HiGHS」の値ではない。提案ワークステーションの性能値は予測で、購入資料では測定値と予測値を明確に分離する必要がある。", fill="FFF7E6", label_color=GOLD)

    add_heading(doc, "6. テストと再現性の点検", 1)
    add_body(doc, "WSL Ubuntu上でpytestを実行した。全体の収集数は42件。主要テストをグループ別に実行した結果、モデル／構造17件、ソルバー・サロゲート・科学整合性17件、scientific_rigor 5件が合格した。cuOpt CUDA実機専用テストは環境依存でスキップされる可能性があるため、合格数だけでGPU実機の正しさを保証しない。")
    add_table(doc, ["テスト群", "実行結果", "確認対象"], [
        ("project_integrity + model_selection + cuOpt safety", "17 passed（CUDA実機依存項目は環境依存）", "コアファイル、3種選択、基本cuOptガード"),
        ("community / surrogate / science", "17 passed, 2 warnings, 44.89 s", "joint分割、GPU surrogate、科学反応、PHA/BS、ゴム分解"),
        ("scientific_rigor", "5 passed, 28.69 s", "質量収支、無炭素増殖、pHバッファ、ecFBA様制約"),
        ("全体 collection", "42 tests collected", "テストスイート全体の規模"),
    ], [2700, 2700, 3960], font_size=9)
    add_heading(doc, "7. 点検で見つかった注意点", 1)
    add_table(doc, ["優先度", "所見", "根拠", "影響"], [
        ("P0", "cuOptは厳密LPの本番既定にしない", "docs/GPU_CUOPT.mdに、RTX 4060で数値エラー・CPUより遅い測定が記録されている。", "速度と再現性を誤認する可能性。"),
        ("P1", "既定CLIがCPU/GLPK", "main.pyのtrain既定値は solver-backend=glpk、device=cpu、n-envs=4。", "GPUを購入しても明示引数なしでは使われない。"),
        ("P1", "評価経路の分離が不十分", "train_agentでeval_envを作るが、agent.evaluate()には渡していない。evaluate()はself.envを使う。", "訓練環境と評価環境の条件が混ざる。"),
        ("P1", "評価分解率の初期値取得に注意", "ppo_agent.evaluate()は最初のstep後のinfoからinitial_rubberを設定する。", "分解率を過小評価し得る。"),
        ("P1", "SBML読込失敗が静かにスキップ", "utils.load_sbml_models()は例外を握りつぶし、部分集合を返す。selectが不足分を補完する。", "意図しない菌種で学習するリスク。"),
        ("P2", "仕様書の観測次元表記が不一致", "SYSTEM_ARCHITECTURE.mdはnum_species+11と記載するが、rl_environment.pyは+13で3種なら16。", "文書・実装・論文記述の不整合。"),
        ("P2", "dFBASimulatorへ責務が集中", "境界更新、ソルバー、状態更新、物理モデル、ログが1,311行に集約。", "変更時の回帰範囲が広い。"),
    ], [800, 2250, 3800, 2510], font_size=8.3)

    add_heading(doc, "8. 推奨する改善順序", 1)
    for text in [
        "実行プロファイルを固定する：exploration_gpu_surrogate、exact_cpu_validation、cuopt_probe の3つを設定ファイルまたはCLIプリセットとして分離し、backend/device/n_envs/surrogate_dirをログに必ず保存する。",
        "評価経路を修正する：評価用VecEnvをagent.evaluate(eval_env=...)へ明示的に渡し、初期ゴム量をreset直後に保存する。評価JSONにはsolver backend、surrogate acceptance、fallback、seedを含める。",
        "モデル読込をfail-fast化する：3種のcanonical filenameとmodel fingerprintを必須条件にし、読込失敗や不足時は補完せず停止する。",
        "近似と厳密の比較を標準化する：同一seed・同一action列・同一初期状態で、state vector、objective、S v残差、bound violation、wall timeを比較する。",
        "テストを3層化する：軽量unit、3-GEM integration、GPU/performance acceptanceを分け、GPU実機がない場合はskip理由と未検証項目をレポートへ出す。",
        "遺伝子KO探索を追加する場合は、KO候補を外側の設計ループ（またはaction mask付きの離散ヘッド）に置く。現行の5次元連続行動だけでは遺伝子ノックアウトは表現していない。",
    ]:
        add_bullet(doc, text)

    add_heading(doc, "9. まとめ", 1)
    add_callout(doc, "結論", "現行コードは、3 GEMの代謝モデルをdFBAで時間発展させ、その状態をPPOが制御する研究用統合基盤として成立している。GPU高速化の実体は、厳密LPそのものを常にGPUで置換することではなく、GPUサロゲートのバッチ探索とCPU HiGHS監査を組み合わせる設計である。次の実装優先度は、GPUをさらに“使う”ことよりも、実行プロファイルの固定、評価経路の明示、モデル読込のfail-fast化、近似／厳密の証跡分離である。", fill=LIGHT_BLUE)
    add_body(doc, "この順序を守れば、研究結果の再現性を保ちながら、RTX PRO 4000 ×3 の独立shard運用や、遺伝子KOを含む大規模探索へ段階的に拡張できる。")

    add_heading(doc, "付録A. 参照した主要ファイル", 1)
    add_table(doc, ["分類", "ファイル", "用途"], [
        ("入口", "main.py", "train/evaluate CLI、環境作成、GPU・solver引数"),
        ("dFBA", "src/dfba_simulator.py", "状態、境界、FBA、質量収支、診断"),
        ("RL", "src/rl_environment.py; src/ppo_agent.py", "観測・行動・報酬、PPO"),
        ("GPU", "src/cuopt_solver.py; src/fba_surrogate.py; src/fba_surrogate_service.py; src/gpu_assignment.py", "cuOpt、surrogate、batch、GPU割当"),
        ("モデル", "models/sbml/final_consortium/*.xml", "OR16、NS21、L. plantarumのGEM"),
        ("検証", "tests/*.py", "42件のpytestスイート"),
        ("証跡", "results/workstation_procurement_evidence.json; results/rollout_cpu_vs_gpu_rtx4060.json", "性能・資源・速度比較"),
        ("仕様", "docs/MASTER_INDEX.md; docs/SYSTEM_ARCHITECTURE.md; docs/GPU_CUOPT.md; docs/GPU_SURROGATE_ARCHITECTURE.md", "設計意図と既知の制約"),
    ], [1500, 4300, 3060], font_size=8.8)
    add_body(doc, "注：本報告書は2026-08-31時点のワークスペースを対象とした点検記録であり、提案ワークステーション上の実機性能を新たに保証するものではない。")
    doc.save(DOCX_PATH)


if __name__ == "__main__":
    build_doc()
    print(DOCX_PATH)
