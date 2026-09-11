import pytest
import cobra
import os

def test_or16_lcp_genes_present():
    """OR16モデルに3つのLCP遺伝子が存在することを確認"""
    model_path = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    assert os.path.exists(model_path)
    
    model = cobra.io.read_sbml_model(model_path)
    
    # 期待される locus tags
    expected_genes = ["ACTI_59630", "ACTI_59640", "ACTI_69520"]
    
    for gene_id in expected_genes:
        # CarveMe/COBRApy の ID プレフィックスを考慮 (G_... など)
        found = any(gene_id in g.id for g in model.genes)
        assert found, f"Gene {gene_id} not found in model"

def test_lcp_reaction_gpr():
    """R_LCP 反応が3つの遺伝子と関連付けられていることを確認"""
    model_path = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    model = cobra.io.read_sbml_model(model_path)
    
    assert "R_LCP" in model.reactions
    lcp_rxn = model.reactions.get_by_id("R_LCP")
    
    gpr_str = lcp_rxn.gene_reaction_rule
    assert "ACTI_59630" in gpr_str
    assert "ACTI_59640" in gpr_str
    assert "ACTI_69520" in gpr_str
    assert "ACTI_28730" not in gpr_str
    # 'or' 関係であることを確認
    assert "or" in gpr_str.lower()
    assert lcp_rxn.bounds == (0.0, 0.0)
    assert lcp_rxn.check_mass_balance() == {}

def test_lcp_metabolic_connectivity():
    """LCP反応がゴム代謝経路に正しく組み込まれていることを確認"""
    model_path = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    model = cobra.io.read_sbml_model(model_path)
    
    lcp_rxn = model.reactions.get_by_id("R_LCP")
    
    # 生成物が C30 オリゴマーであり、それが輸送反応に繋がっているか
    products = [m.id for m in lcp_rxn.products]
    assert any("C30_oligo" in p for p in products)
    
    # C30_oligo を消費する反応があるか
    oligo_e = next(m for m in lcp_rxn.products if "C30_oligo" in m.id)
    consuming_rxns = [r.id for r in oligo_e.reactions if r.id != "R_LCP"]
    assert "R_C30t" in consuming_rxns or any("C30t" in rid for rid in consuming_rxns)
    
    # 輸送されて細胞内オリゴマー C30_oligo_c に到達するか
    t_rxn_id = next(rid for rid in consuming_rxns if "C30t" in rid or rid == "R_C30t")
    t_rxn = model.reactions.get_by_id(t_rxn_id)
    oligo_c = next(m for m in t_rxn.products if "C30_oligo_c" in m.id or "C30_oligo" in m.id)
    
    # 細胞内オリゴマーが R_C30_cat 反応によって分解されるか
    cat_rxns = [r.id for r in oligo_c.reactions if "C30_cat" in r.id or r.id == "R_C30_cat"]
    assert len(cat_rxns) > 0
    
    # 分解物が最終的に中心代謝アセチルCoA (accoa_c) またはプロピオニルCoA (ppcoa_c) に到達するか
    cat_rxn_id = cat_rxns[0]
    cat_rxn = model.reactions.get_by_id(cat_rxn_id)
    cat_products = [m.id for m in cat_rxn.products]
    assert "accoa_c" in cat_products or "ppcoa_c" in cat_products
    assert cat_rxn.check_mass_balance() == {}
