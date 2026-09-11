import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra
model = cobra.io.read_sbml_model('models/sbml/final_consortium/Lactobacillus_plantarum.xml')

# バイオサーファクタント、脂質、脂肪酸に関連する反応を検索
keywords = ['surfactant', 'rhamno', 'lipid', 'deca', 'fatty', 'acid']
found = []
for r in model.reactions:
    if any(k.lower() in r.name.lower() or k.lower() in r.id.lower() for k in keywords):
        found.append(f"{r.id}: {r.name}")

print(f"--- Potential Biosurfactant/Lipid Pathways in LP ({len(found)} found) ---")
for f in found[:20]: # 最初の20件
    print(f)
