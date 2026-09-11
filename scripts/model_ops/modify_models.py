import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import cobra
from pathlib import Path

def modify_sj_model(model_path, output_path):
    """SJモデルを利他的分解者に改変 (Arg欠損 + イソプレン漏出)"""
    print(f"🧬 SJモデル改変中: {model_path}")
    model = cobra.io.read_sbml_model(str(model_path))
    
    # 1. アルギニン生合成経路のノックアウト (Arg- 要求性)
    # 例: argH (argininosuccinate lyase) 相当の反応を停止
    reactions_to_ko = [r for r in model.reactions if 'ARGSL' in r.id or 'ARGN' in r.id]
    for rxn in reactions_to_ko:
        rxn.knock_out()
    print(f"  - Arg生合成経路をノックアウト ({len(reactions_to_ko)} reactions)")

    # 2. イソプレン取り込みの制限
    # EX_isoprene_e がない場合は作成し、取り込み(LB)を制限
    if 'EX_isoprene_e' in model.reactions:
        # 生存に最低限必要な分だけ許容 (例: 0.1)
        model.reactions.EX_isoprene_e.lower_bound = -0.1 
    
    model.id = "Sphingobium_japonicum_altruistic"
    cobra.io.write_sbml_model(model, str(output_path))
    print(f"  ✅ 保存完了: {output_path}")

def modify_pp_model(model_path, output_path):
    """PPモデルをPHA生産に特化 (phaZ欠損)"""
    print(f"🧬 PPモデル改変中: {model_path}")
    model = cobra.io.read_sbml_model(str(model_path))
    
    # 1. PHA分解酵素 (phaZ) のノックアウト
    # iJN1463では 'PHAD' や特定のIDで定義されている可能性がある
    pha_depol = [r for r in model.reactions if 'PHAD' in r.id or 'phaZ' in r.name.lower()]
    for rxn in pha_depol:
        rxn.knock_out()
    print(f"  - PHA分解経路をノックアウト ({len(pha_depol)} reactions)")
    
    model.id = "Pseudomonas_putida_PHA_specialist"
    cobra.io.write_sbml_model(model, str(output_path))
    print(f"  ✅ 保存完了: {output_path}")

if __name__ == "__main__":
    sbml_dir = Path("models/sbml")
    output_dir = Path("models/sbml/modified")
    output_dir.mkdir(exist_ok=True)
    
    # 元のファイルを指定 (SJがPPのクローンであることを逆手に取り、共通のiJN1463ベースで改変)
    pp_base = sbml_dir / "Pseudomonas_putida_KT2440.xml"
    
    modify_sj_model(pp_base, output_dir / "Sphingobium_japonicum_altruistic.xml")
    modify_pp_model(pp_base, output_dir / "Pseudomonas_putida_PHA_specialist.xml")
