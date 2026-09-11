"""Revise the laboratory GPU workstation procurement DOCX.

The source document is preserved.  This script performs targeted edits,
repairs numbering/captions, adds the missing projection and acceptance-test
sections, and writes a clean revised copy.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph


NAVY = "263E58"
LIGHT_BLUE = "EAF1FA"
LIGHT_YELLOW = "FFF2CC"
WHITE = "FFFFFF"


def find_paragraph(doc: Document, prefix: str) -> Paragraph:
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise ValueError(f"Paragraph not found: {prefix!r}")


def set_paragraph_text(paragraph: Paragraph, text: str, *, bold_prefix: str | None = None) -> None:
    paragraph.clear()
    if bold_prefix and text.startswith(bold_prefix):
        first = paragraph.add_run(bold_prefix)
        first.bold = True
        paragraph.add_run(text[len(bold_prefix) :])
    else:
        paragraph.add_run(text)


def remove_numbering(paragraph: Paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is not None:
        p_pr.remove(num_pr)


def insert_paragraph_before(anchor: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    anchor._p.addprevious(new_p)
    paragraph = Paragraph(new_p, anchor._parent)
    if style:
        paragraph.style = style
    if text:
        paragraph.add_run(text)
    return paragraph


def insert_paragraph_after(anchor: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    anchor._p.addnext(new_p)
    paragraph = Paragraph(new_p, anchor._parent)
    if style:
        paragraph.style = style
    if text:
        paragraph.add_run(text)
    return paragraph


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=95, start=115, bottom=95, end=115) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def create_numbering(doc: Document, *, bullet: bool) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(node.get(qn("w:abstractNumId"))) for node in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(node.get(qn("w:numId"))) for node in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    lvl.append(start)
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet" if bullet else "decimal")
    lvl.append(num_fmt)
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "•" if bullet else "%1.")
    lvl.append(lvl_text)
    suff = OxmlElement("w:suff")
    suff.set(qn("w:val"), "tab")
    lvl.append(suff)
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "520")
    tabs.append(tab)
    p_pr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "520")
    ind.set(qn("w:hanging"), "260")
    p_pr.append(ind)
    lvl.append(p_pr)
    if bullet:
        r_pr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        fonts.set(qn("w:ascii"), "Arial")
        fonts.set(qn("w:hAnsi"), "Arial")
        r_pr.append(fonts)
        lvl.append(r_pr)
    abstract.append(lvl)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    numbering.append(num)
    return num_id


def apply_numbering(paragraph: Paragraph, num_id: int) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    old = p_pr.find(qn("w:numPr"))
    if old is not None:
        p_pr.remove(old)
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    num_pr.append(ilvl)
    num_pr.append(num)
    p_pr.append(num_pr)


def add_hyperlink(paragraph: Paragraph, label: str, url: str) -> None:
    rel_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "4472C4")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.extend((color, underline))
    text = OxmlElement("w:t")
    text.text = label
    run.extend((r_pr, text))
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_reference(doc: Document, anchor: Paragraph, label: str, url: str, bullet_num_id: int) -> Paragraph:
    paragraph = insert_paragraph_before(anchor, style="Normal")
    paragraph.add_run(label + "：")
    add_hyperlink(paragraph, url, url)
    apply_numbering(paragraph, bullet_num_id)
    return paragraph


def set_image_alt_text(paragraph: Paragraph, title: str, description: str) -> None:
    for doc_pr in paragraph._p.xpath(".//wp:docPr"):
        doc_pr.set("title", title)
        doc_pr.set("descr", description)


def resize_shape(shape, width_inches: float) -> None:
    ratio = shape.height / shape.width
    width = Inches(width_inches)
    shape.width = width
    shape.height = int(width * ratio)


def resize_drawing_paragraph(paragraph: Paragraph, width_inches: float) -> None:
    """Resize either an inline or floating drawing while preserving aspect ratio."""
    extents = paragraph._p.xpath(".//wp:extent")
    if not extents:
        raise RuntimeError("Drawing paragraph has no wp:extent")
    extent = extents[0]
    old_cx = int(extent.get("cx"))
    old_cy = int(extent.get("cy"))
    new_cx = int(Inches(width_inches))
    new_cy = int(new_cx * old_cy / old_cx)
    extent.set("cx", str(new_cx))
    extent.set("cy", str(new_cy))
    for child_extent in paragraph._p.xpath(".//a:xfrm/a:ext"):
        child_extent.set("cx", str(new_cx))
        child_extent.set("cy", str(new_cy))


def convert_drawing_to_inline(paragraph: Paragraph, width_inches: float) -> None:
    """Replace a floating/anchored drawing with an inline copy of the same image."""
    blips = paragraph._p.xpath(".//a:blip")
    if not blips:
        raise RuntimeError("No image relationship in drawing paragraph")
    rel_id = blips[0].get(qn("r:embed"))
    blob = paragraph.part.related_parts[rel_id].blob
    paragraph.clear()
    paragraph.style = "Normal"
    remove_numbering(paragraph)
    paragraph.add_run().add_picture(BytesIO(blob), width=Inches(width_inches))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def style_caption(paragraph: Paragraph) -> None:
    paragraph.style = "Caption"
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.keep_with_next = False
    remove_numbering(paragraph)
    for run in paragraph.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(89, 98, 110)


def replace_all(doc: Document, old: str, new: str) -> None:
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if old in run.text:
                            run.text = run.text.replace(old, new)


def revise(source: Path, output: Path) -> None:
    doc = Document(str(source))

    # Keep the established visual system while making the document status explicit.
    set_paragraph_text(doc.paragraphs[1], "要件定義・見積依頼書（3 GEM dFBA／強化学習・実測／予測統合版）")
    set_paragraph_text(doc.paragraphs[2], "改訂日：2026年8月28日　｜　用途：研究室導入・正式見積依頼")

    executive = (
        "導入判断　購入希望ワークステーションの第一目的は、3種の微生物GEMを組み込んだdFBA環境を多数並列実行し、"
        "強化学習によって流加培養条件を設計することである。現行RTX 4060 Laptopで、CPU HiGHS版13.05秒に対し、"
        "物理制約付きGPUサロゲート版は2.63秒となり、4.99 ± 0.40倍（n = 5、95%信頼区間）の高速化を実測した。"
        "これは、各stepで反復される3 GEMのFBA評価が主要な高速化対象であることを示す。一方、end-to-end時間には環境更新、"
        "スケジューリング、CPU–GPU間転送も含まれるため、律速はGPU上のFBA評価と、それへ仕事を供給するCPU側処理の複合である。"
        "このため、24コア48スレッドCPU、128 GB ECC RDIMM、RTX PRO 4000 Blackwell非SFF版3基を一体として導入する。"
        "GPUは共有VRAMではなく、seed・agent・16環境群の独立shardとして運用する。本体・組立は約267.7万円、3年保守は別枠上限10万円とする。"
    )
    set_paragraph_text(doc.paragraphs[3], executive, bold_prefix="導入判断　")

    set_paragraph_text(doc.paragraphs[4], "1. 研究目的と研究室内ユースケース")
    set_paragraph_text(
        doc.paragraphs[5],
        "本機の中心用途は、3種の微生物ゲノムスケール代謝モデル（GEM）を組み合わせた動的フラックスバランス解析（dFBA）環境で、"
        "流加量・組成・タイミング等を強化学習により探索することである。各環境stepでは培地状態と菌体量を更新し、3 GEMの代謝フラックスを評価する。"
        "探索段はGPUサロゲートで高スループット化し、採択候補と最終報告値はCPU HiGHSによる厳密FBAで監査する。"
        "同じ計算基盤を、GEM構築、GTDB・メタゲノム解析、空き時間のラボ管理AIおよびMDにも利用する。",
    )
    set_paragraph_text(doc.paragraphs[11], "1.1 過去の実測・ログ・公式値")
    doc.paragraphs[11].style = "Heading 2"
    doc.paragraphs[12].clear()
    doc.paragraphs[12].style = "Normal"
    remove_numbering(doc.paragraphs[12])

    set_paragraph_text(
        doc.paragraphs[14],
        "実測による結論　受入試験の対象はゴム分解菌、PHA蓄積菌、乳酸菌の3 GEMである。CPU HiGHS版ではFBA solveが1ステップ時間の89%を占めた。"
        "物理制約付きGPUサロゲートへ置換し、同一24ステップを各5回比較した結果、全体実行時間は平均13.05秒から2.63秒へ短縮し、"
        "ペア速度比4.99 ± 0.40倍（n = 5、95%信頼区間）を確認した。別の16環境・960 transition実測では、GPU推論がwall timeの41.8%、"
        "環境更新その他が58.2%であった。したがって、GPUは主要な高速化対象であるが、購入構成ではCPU側の供給処理も同時に増強する必要がある。",
        bold_prefix="実測による結論　",
    )
    set_paragraph_text(doc.paragraphs[15], "2.1 実測資源使用量とボトルネックの解釈")
    doc.paragraphs[15].paragraph_format.page_break_before = True

    # Remove empty/corrupt list paragraphs that rendered as orphan numbers.
    for paragraph in doc.paragraphs:
        if not paragraph.text.strip() and not paragraph._p.xpath(".//a:blip"):
            paragraph.style = "Normal"
            remove_numbering(paragraph)
            paragraph.paragraph_format.space_after = Pt(0)

    # The document contains three scientific figures followed by one
    # motherboard photograph. One scientific figure uses a floating anchor,
    # so identify them by body order rather than python-docx inline_shapes.
    all_image_paragraphs = [p for p in doc.paragraphs if p._p.xpath(".//a:blip")]
    if len(all_image_paragraphs) != 4:
        raise RuntimeError(f"Expected 4 image paragraphs, found {len(all_image_paragraphs)}")
    image_paragraphs = all_image_paragraphs[:3]

    convert_drawing_to_inline(image_paragraphs[1], 6.65)
    resize_drawing_paragraph(image_paragraphs[0], 6.15)
    resize_drawing_paragraph(image_paragraphs[1], 6.65)
    resize_drawing_paragraph(image_paragraphs[2], 6.65)
    for p in image_paragraphs:
        p.style = "Normal"
        remove_numbering(p)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True

    set_image_alt_text(
        image_paragraphs[0],
        "現行機の資源使用量",
        "RTX 4060 Laptopで測定したGPU使用率、VRAM、ホストCPU負荷と、提案構成の容量比を示す4パネル図。",
    )
    set_image_alt_text(
        image_paragraphs[1],
        "RTX 4060実測性能",
        "同時環境数に対するGPU使用率、FBA処理量、VRAM、およびCPU HiGHSとのend-to-end比較を示す4パネル図。",
    )
    set_image_alt_text(
        image_paragraphs[2],
        "購入構成の予測挙動",
        "Threadripper 9960X、128 GB RAM、RTX PRO 4000 Blackwell 3基のGPU段、end-to-end処理量、固定仕事時間、容量比の予測。",
    )
    set_image_alt_text(
        all_image_paragraphs[3],
        "ASUS Pro WS TRX50-SAGE WIFIのPCIeスロット配置",
        "ASUS公式の実基板写真に、3基のfull-height single-slot GPUを配置するPCIe領域、拡大枠、40.64 mmの間隔を追記した図。",
    )

    caption1 = insert_paragraph_after(
        image_paragraphs[0],
        "図1　現行RTX 4060 LaptopにおけるGPU使用率、VRAM、ホスト負荷、および提案構成の公称容量比。"
        "使用率と処理量は実測、ハードウェア容量比は仕様値であり、アプリケーション性能の保証値ではない。",
        "Caption",
    )
    style_caption(caption1)

    heading22 = insert_paragraph_before(image_paragraphs[1], "2.2 GPU処理量とend-to-end高速化", "Heading 2")
    heading22.paragraph_format.page_break_before = True
    heading22.paragraph_format.keep_with_next = True
    # A small explicit inset avoids a Word floating-object wrap artifact that
    # otherwise masks the first glyph at the top of this page.
    heading22.paragraph_format.left_indent = Pt(6)
    heading22.paragraph_format.first_line_indent = Pt(0)
    caption2 = insert_paragraph_after(
        image_paragraphs[1],
        "図2　現行RTX 4060 Laptopでの同時環境数に対するGPU使用率、3-GEM FBA候補処理量、VRAM使用量、および16環境rolloutのend-to-end比較。"
        "最大GPU使用率と最大処理量のbatchは一致しないため、初期運用では処理量が最大となったbatch 64を採用する。",
        "Caption",
    )
    style_caption(caption2)

    # Replace the prior misplaced caption and its automatic list marker.
    old_caption = find_paragraph(doc, "図1　工程別ピーク資源")
    old_caption.clear()
    old_caption.style = "Normal"
    remove_numbering(old_caption)
    old_caption.paragraph_format.space_after = Pt(0)

    heading3 = insert_paragraph_before(image_paragraphs[2], "3. 予測挙動と購入構成の妥当性", "Heading 1")
    heading3.paragraph_format.page_break_before = True
    heading3.paragraph_format.keep_with_next = True
    intro3 = insert_paragraph_after(
        heading3,
        "予測は、現行16環境rolloutの実測85.83 transition/s、GPU推論時間比41.8%、生産カーネル5,581 prediction/sを基準とし、"
        "RTX PRO 4000の公式メモリ帯域・FP32性能と、独立shardの並列効率を用いて外挿した。1 GPUあたり16環境を初期単位とし、3 GPUへ独立に割り当てる。",
        "Normal",
    )
    intro3.paragraph_format.keep_with_next = True
    caption3 = insert_paragraph_after(
        image_paragraphs[2],
        "図3　現行実測から外挿したThreadripper 9960X＋RTX PRO 4000 Blackwell非SFF版3基＋128 GB RAMの予測挙動。"
        "線・棒は計画シナリオ、帯・whiskerは低位–高位の工学的シナリオ範囲であり、信頼区間ではない。VRAMは3基間で共有しない。",
        "Caption",
    )
    style_caption(caption3)
    interpretation = insert_paragraph_after(
        caption3,
        "計画シナリオ　3 GPUでGPU段5.94倍、独立した16環境shard×3のend-to-end処理量は約300 transition/s（現行比3.50倍）、"
        "固定仕事時間は現行の28.6%と予測する。低位–高位範囲は約249–328 transition/sである。これらは発注前予測であり、"
        "納品後に同一入力・同一step数・同一seed構成で再測定する。",
        "Callout",
    )
    set_paragraph_text(interpretation, interpretation.text, bold_prefix="計画シナリオ　")

    section4 = find_paragraph(doc, "4. 一本化した推奨構成")
    section4.paragraph_format.page_break_before = True

    # Correct official GPU power and derived totals throughout the document.
    replace_all(doc, "140 W", "145 W")
    replace_all(doc, "420 W", "435 W")
    replace_all(doc, "約890 W", "約905 W")
    replace_all(doc, "連続負荷約60%", "連続負荷約60%")

    # Add the current official single-precision figure to the principal specification row.
    gpu_spec_cell = doc.tables[3].cell(1, 2)
    gpu_spec_cell.text = "各24 GB GDDR7 ECC、672 GB/s、37 TFLOPS（FP32）、145 W、full-height・single-slot。同一SKU。"
    doc.tables[4].cell(2, 1).text = "435 W"
    doc.tables[4].cell(2, 2).text = "145 W × 3（NVIDIA公式）"
    doc.tables[4].cell(4, 1).text = "約905 W"
    doc.tables[4].cell(5, 2).text = "連続負荷約60%、独立16-pin×3を必須確認"
    doc.tables[11].cell(3, 1).text = "145 W"

    gpu_power_paragraph = find_paragraph(doc, "各GPUは最大")
    set_paragraph_text(gpu_power_paragraph, gpu_power_paragraph.text.replace("140 W", "145 W"))

    # The supplied board image already contains a crop box and dimension arrows.
    set_paragraph_text(
        find_paragraph(doc, "図4　ASUS公式の実基板写真"),
        "図4　ASUS公式の実基板写真とPCIe領域拡大。白枠・拡大線・40.64 mm寸法矢印は、本資料で配置確認用に追記した。",
    )
    set_paragraph_text(
        find_paragraph(doc, "写真出典：ASUS公式製品ページ"),
        "写真出典：ASUS公式製品ページ『Pro WS TRX50-SAGE WIFI front view』。図中の白枠・拡大線・寸法矢印は概略表示であり、"
        "最終的なスロット位置、リンク幅、カード間隔、コネクター干渉は公式マニュアルと実機組立で確認する。",
    )

    # Bring GTDB-Tk wording in line with current R232 documentation.
    doc.tables[2].cell(2, 2).text = "R232の細菌分類は約140 GB、--full_treeは約950 GB、参照DB保存は約100 GB。--scratch_dirでpplacerのRAMを削減可能。"
    doc.tables[2].cell(2, 3).text = "研究室実績とはversion／DB／入力条件が異なり得る。128 GBでは公式推奨値を下回るため、同一データ回帰試験と外部計算への切替条件を設ける。"
    set_paragraph_text(
        find_paragraph(doc, "GTDB-Tkの扱い"),
        "GTDB-Tkの扱い　研究室では48 GB DDR5＋swapで細菌分類を完走しているが、現行R232の公式要件は細菌約140 GB、--full_tree約950 GBである。"
        "128 GBは同一version・DB・入力でのswap削減と安定化には有効だが、R232の保証容量ではない。回帰試験と現行DB試験を分け、peak RSS・最大swap・iowait・経過時間を記録し、"
        "OOMまたはswap常用時はHPC／cloudへ切り替える。",
        bold_prefix="GTDB-Tkの扱い　",
    )

    # Clarify the secondary AI workload without treating a one-GPU launch as guaranteed.
    set_paragraph_text(
        find_paragraph(doc, "採用モデル"),
        "採用モデル　Qwen3.8-27Bは2026年8月14日に公開された27B denseのオープンウェイト・マルチモーダルモデルで、公式例では262,144 token contextを扱う。"
        "本機では全文脈を常用せず、4-bit量子化＋RAGで必要箇所だけを検索する。24 GB GPU 1基への収容は量子化形式、KV cache、画像入力、同時数で変わるため、"
        "『指定条件で起動し、VRAM・応答時間・安定性を測定する』ことを受入条件とする。",
        bold_prefix="採用モデル　",
    )

    # Repair all body lists with a dedicated true bullet definition.
    bullet_num_id = create_numbering(doc, bullet=True)
    for paragraph in doc.paragraphs:
        if paragraph.style.name == "List Bullet" and paragraph.text.strip():
            apply_numbering(paragraph, bullet_num_id)

    # Add a missing acceptance-test section before the existing summary.
    summary = find_paragraph(doc, "10.要約")
    summary.paragraph_format.page_break_before = True
    heading9 = insert_paragraph_before(summary, "9. 納品時の受入試験", "Heading 1")
    heading9.paragraph_format.page_break_before = True
    lead9 = insert_paragraph_before(
        summary,
        "性能予測を購入後の保証値と混同しないため、納品時はハードウェア認識、安定性、および実アプリケーションの再現試験を分けて記録する。"
        "業者検収は以下の項目、研究室検収は3-GEM dFBA benchmarkを中心とする。",
        "Normal",
    )

    acceptance_rows = [
        ("GPU認識", "同一SKUのRTX PRO 4000を3基認識。各24 GB ECC、driver／CUDA整合。", "nvidia-smi -L／-q、CUDA sample、GPU 0–2の個別ログ"),
        ("PCIe構成", "G5_1／G5_2／G5_3で最大リンク幅x16／x16／x8を確認。", "lspciおよび負荷中のPCIe link generation／width"),
        ("CPU・RAM", "9960Xの24 core／48 thread、128 GB ECC、A1／C1配置を確認。", "BIOS／OS認識、ECC状態、MemTest一晩error 0件"),
        ("電力・冷却", "CPU＋3 GPU同時負荷を60分実施し、error・shutdown・thermal throttlingなし。", "CPU／GPU温度、clock、power、fanの時系列ログと組立写真"),
        ("3-shard実行", "各GPUへ独立した16環境jobを1本ずつ割り当て、3本を同時完走。", "transition/s、GPU使用率、VRAM、fallback件数、seed、入力hash"),
        ("ストレージ・OS", "Ubuntu 24.04 LTS、2 TB NVMe、256 GB swap、300 GB以上のscratch空き。", "OS／driver／CUDA版、fioまたはSMART、df、swapon結果"),
    ]
    table = doc.add_table(rows=1, cols=3)
    table.style = doc.tables[3].style
    table.autofit = False
    table.columns[0].width = Inches(1.15)
    table.columns[1].width = Inches(2.95)
    table.columns[2].width = Inches(2.75)
    headers = ("区分", "受入条件", "提出ログ・証跡")
    for i, text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.width = table.columns[i].width
        cell.text = text
        set_cell_shading(cell, NAVY)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9)
    set_repeat_table_header(table.rows[0])
    for row_index, row_data in enumerate(acceptance_rows, start=1):
        cells = table.add_row().cells
        for col_index, text in enumerate(row_data):
            cells[col_index].width = table.columns[col_index].width
            cells[col_index].text = text
            cells[col_index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cells[col_index])
            if row_index % 2 == 0:
                set_cell_shading(cells[col_index], "F4F6F8")
            for paragraph in cells[col_index].paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)
        cells[0].paragraphs[0].runs[0].bold = True
    table._tbl.getparent().remove(table._tbl)
    summary._p.addprevious(table._tbl)
    acceptance_note = insert_paragraph_before(
        summary,
        "判定上の注意　約300 transition/sは計画シナリオであり、納品時の最低保証値ではない。実機評価では、同一入力・同一step数・同一batch・同一seed構成を固定し、"
        "単GPU性能、3 GPU総処理量、GPU解採用率、CPU fallback、および再現性を併記する。",
        "Callout",
    )
    set_paragraph_text(acceptance_note, acceptance_note.text, bold_prefix="判定上の注意　")

    # Strengthen the final procurement request and use the corrected 145 W figure implicitly via the BOM.
    set_paragraph_text(
        find_paragraph(doc, "依頼文"),
        "依頼文　税込約270万円を目標に、OR16、NS21、L. plantarumの3 GEMを用いるdFBA強化学習で流加培養条件を最適化する計算機を依頼する。"
        "RTX 4060 Laptop上では、CPU HiGHS版13.05秒に対してGPUサロゲート版2.63秒、4.99 ± 0.40倍（n = 5、95%信頼区間）を実測した。"
        "提案機ではThreadripper 9960X、TRX50、指定OWC 128 GB ECC RDIMM、RTX PRO 4000 Blackwell非SFF版24 GB×3を採用し、GPUごとにseed・agent・16環境群を独立shardとして割り当てる。"
        "GPU段5.94倍、end-to-end約3.50倍は発注前の計画予測であり、納品後に同一benchmarkで再測定する。RAMはA1／C1の2チャネル運用とし、QVL未掲載品のため発注前互換保証とMemTestを必須とする。"
        "2 TB TLC NVMeに256 GB swapと300 GB以上のscratchを確保し、1,500 W級電源から各GPUへ独立したCEM5 16-pinを供給する。本体・組立は約267.7万円、3年保守は別枠上限10万円とする。"
        "Qwen3.8-27B 4-bit、細菌GTDB／メタゲノム、MDは副用途とし、主目的のRL計算と時間分離する。",
        bold_prefix="依頼文　",
    )

    # Add a compact, authoritative source section after the summary.
    heading11 = doc.add_paragraph("11. 参照資料", style="Heading 1")
    heading11.paragraph_format.page_break_before = True
    doc.add_paragraph("ハードウェア仕様・副用途要件は、以下の公式資料で確認した（閲覧日：2026年8月28日）。")
    source_anchor = doc.add_paragraph()
    add_reference(doc, source_anchor, "AMD Ryzen Threadripper 9960X 製品仕様", "https://www.amd.com/en/products/processors/ryzen-threadripper/9000-series/amd-ryzen-threadripper-9960x.html", bullet_num_id)
    add_reference(doc, source_anchor, "NVIDIA RTX PRO 4000 Blackwell データシート", "https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-4000/workstation-datasheet-rtx-pro-4000-nvidia-us-web.pdf", bullet_num_id)
    add_reference(doc, source_anchor, "ASUS Pro WS TRX50-SAGE WIFI 技術仕様", "https://www.asus.com/us/motherboards-components/motherboards/workstation/pro-ws-trx50-sage-wifi/techspec/", bullet_num_id)
    add_reference(doc, source_anchor, "ASUS Pro WS TRX50-SAGE WIFI User's Manual", "https://dlcdnets.asus.com/pub/ASUS/mb/SocketsTR5/Pro_WS_TRX50-SAGE_WIFI/E23238_Pro_WS_TRX50-SAGE_WIFI_EM_V3_WEB.pdf", bullet_num_id)
    add_reference(doc, source_anchor, "Qwen3.8 公式リポジトリ", "https://github.com/QwenLM/Qwen3.8", bullet_num_id)
    add_reference(doc, source_anchor, "GTDB-Tk 2.7.2 hardware requirements", "https://ecogenomics.github.io/GTDBTk/installing/index.html", bullet_num_id)
    source_anchor._element.getparent().remove(source_anchor._element)
    internal = doc.add_paragraph("内部再現資料", style="Heading 2")
    doc.add_paragraph(
        "現行実測：results/gpu_saturation_2048_multistream_rtx4060.json、results/parallel_env_16_long_multistream_rtx4060.json。"
        "予測条件・出力：results/workstation_predicted_behavior.json。図の再生成：scripts/create_workstation_prediction_figure.py。"
    )

    # Update footer revision date while preserving PAGE/NUMPAGES fields.
    for section in doc.sections:
        for paragraph in section.footer.paragraphs:
            for run in paragraph.runs:
                if "2026-08-25" in run.text:
                    run.text = run.text.replace("2026-08-25", "2026-08-28")

    # Keep headings/captions with their following content and avoid isolated lines.
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True
        if paragraph.style.name == "Callout":
            paragraph.paragraph_format.keep_together = True
        paragraph.paragraph_format.widow_control = True

    # Document metadata: identify the deliverable, not the editing tool/user.
    props = doc.core_properties
    props.title = "研究室用GPUワークステーション 要件定義・見積依頼書"
    props.subject = "3 GEM dFBA強化学習用ワークステーションの導入根拠・仕様・受入試験"
    props.comments = "2026-08-28 revision: measured/predicted separation, official specifications, acceptance criteria, and sources."

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    revise(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
