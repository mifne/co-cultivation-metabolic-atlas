from pathlib import Path
from docx import Document

from revise_workstation_docx import find_paragraph, set_paragraph_text


SRC = Path(r"C:\Users\tmkri\Downloads\260819_研究室用GPUワークステーション_完成版_図解価格追補.docx")
OUT = Path(r"C:\Users\tmkri\Downloads\260819_研究室用GPUワークステーション_完成版_図解価格追補_要約修正版.docx")


doc = Document(str(SRC))
paragraph = find_paragraph(doc, "依頼文")
if paragraph is None:
    raise RuntimeError("summary paragraph beginning with 依頼文 was not found")

old = "RTX 4060 Laptop上では、CPU HiGHS版13.05秒に対してGPUサロゲート版2.63秒、4.99 ± 0.40倍（n = 5、95%信頼区間）を実測した。"
new = (
    "RTX 4060 Laptop上で、ゴム分解菌・PHA蓄積菌・乳酸菌の3 GEMによる24ステップの同一dFBAケースを"
    "end-to-endで各5回実行したところ、CPU HiGHS版の1試行あたり平均経過時間は13.05秒、GPUサロゲート版は"
    "2.63秒で、ペア速度比4.99 ± 0.40倍（n = 5、95%信頼区間）を実測した。"
)
text = paragraph.text
if old not in text:
    raise RuntimeError("target timing sentence was not found in the summary paragraph")
set_paragraph_text(paragraph, text.replace(old, new), bold_prefix="依頼文　")
doc.save(str(OUT))
print(OUT)
