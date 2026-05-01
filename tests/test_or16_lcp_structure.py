import pytest
import cobra
import os

def test_or16_lcp_genes_present():
    """OR16モデルに3つのLCP遺伝子が存在することを確認"""
    model_path = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    assert os.path.exists(model_path)
    
    model = cobra.io.read_sbml_model(model_path)
    
    # 期待される locus tags
    expected_genes = ["ACTI_28730", "ACTI_28740", "ACTI_37800"]
    
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
    assert "ACTI_28730" in gpr_str
    assert "ACTI_28740" in gpr_str
    assert "ACTI_37800" in gpr_str
    # 'or' 関係であることを確認
    assert "or" in gpr_str.lower()

def test_lcp_metabolic_connectivity():
    """LCP反応がゴム代謝経路に正しく組み込まれていることを確認"""
    model_path = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    model = cobra.io.read_sbml_model(model_path)
    
    lcp_rxn = model.reactions.get_by_id("R_LCP")
    
    # 生成物がアルデヒドであり、それが次の ALDH 反応に繋がっているか
    products = [m.id for m in lcp_rxn.products]
    assert any("isoprenoid_aldehyde" in p for p in products)
    
    # isoprenoid_aldehyde_c を消費する反応があるか
    aldehyde = next(m for m in lcp_rxn.products if "isoprenoid_aldehyde" in m.id)
    consuming_rxns = [r.id for r in aldehyde.reactions if r.id != "R_LCP"]
    assert "ISOP_ALDH" in consuming_rxns
    
    # 最終的に accoa_c または ppcoa_c に到達するか (ISOP_ACS)
    aldh_rxn = model.reactions.get_by_id("ISOP_ALDH")
    acid = next(m for m in aldh_rxn.products if "isoprenoid_acid" in m.id)
    acs_rxns = [r.id for r in acid.reactions if r.id == "ISOP_ACS"]
    assert len(acs_rxns) > 0
    
    acs_rxn = model.reactions.get_by_id("ISOP_ACS")
    acs_products = [m.id for m in acs_rxn.products]
    assert "accoa_c" in acs_products or "ppcoa_c" in acs_products
