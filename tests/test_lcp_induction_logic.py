import pytest
import numpy as np
import cobra
import sys
import os

# src ディレクトリをパスに追加
sys.path.append(os.path.join(os.getcwd(), "src"))
from dfba_simulator import dFBASimulator

def test_lcp_induction_calculation():
    """ゴム存在下でのLCPフラックス上限計算が正しいことを確認"""
    # 最小限のモデルセットアップ
    model_or16 = cobra.io.read_sbml_model("models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml")
    models = {"Actinoplanes_sp_OR16": model_or16}
    
    sim = dFBASimulator(
        models=models,
        initial_biomass={"Actinoplanes_sp_OR16": 0.1},
        initial_metabolites={},
        initial_rubber=10.0
    )
    
    # 1. ゴムのみ、グルコースなしの状態
    met_concs = {"o2_e": 0.25} # グルコース 0
    sim.state.rubber_concentration = 10.0
    
    sim.set_uptake_constraints("Actinoplanes_sp_OR16", met_concs)
    lcp_rxn = model_or16.reactions.get_by_id("R_LCP")
    
    # 計算値: v_base(0.1) * (1 + 43.01 * (10/11)) = 0.1 * (1 + 39.1) = 4.01
    assert lcp_rxn.upper_bound > 3.5
    bound_rubber_only = lcp_rxn.upper_bound
    
    # 2. グルコースによる抑制 (CCR)
    met_concs_with_glc = {"o2_e": 0.25, "glc__D_e": 2.0}
    sim.set_uptake_constraints("Actinoplanes_sp_OR16", met_concs_with_glc)
    
    assert lcp_rxn.upper_bound < bound_rubber_only
    # 高濃度グルコース下では大幅に抑制されるはず
    assert lcp_rxn.upper_bound < 1.0

def test_lcp_no_rubber_no_induction():
    """ゴムがない場合、LCPフラックスが基礎レベルであることを確認"""
    model_or16 = cobra.io.read_sbml_model("models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml")
    models = {"Actinoplanes_sp_OR16": model_or16}
    
    sim = dFBASimulator(
        models=models,
        initial_biomass={"Actinoplanes_sp_OR16": 0.1},
        initial_metabolites={},
        initial_rubber=0.0
    )
    
    sim.state.rubber_concentration = 0.0
    sim.set_uptake_constraints("Actinoplanes_sp_OR16", {"o2_e": 0.25})
    
    lcp_rxn = model_or16.reactions.get_by_id("R_LCP")
    # 基礎レベル (例: 0.1)
    assert lcp_rxn.upper_bound <= 0.5 
