import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
import matplotlib.pyplot as plt
import cobra
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
from dfba_simulator import dFBASimulator

def run_lcp_validation():
    # モデルの読み込み
    model_or16 = cobra.io.read_sbml_model("models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml")
    models = {"Actinoplanes_sp_OR16": model_or16}
    
    # シミュレーターの初期化
    sim = dFBASimulator(
        models=models,
        initial_biomass={"Actinoplanes_sp_OR16": 0.1},
        initial_metabolites={"o2_e": 0.25, "glc__D_e": 0.0},
        initial_rubber=100.0,
        dt=1.0 # 1時間刻み
    )
    
    results = []
    
    # 1. 最初はゴムのみで成長 (誘導を観察)
    for t in range(48):
        state = sim.step(rubber_degradation_rates={}, nutrient_supplementation={})
        lcp_flux = model_or16.reactions.get_by_id("R_LCP").flux
        lcp_bound = model_or16.reactions.get_by_id("R_LCP").upper_bound
        results.append({
            "time": state.time,
            "biomass": state.species["Actinoplanes_sp_OR16"].biomass,
            "glc": state.metabolites.get("glc__D_e", 0.0),
            "lcp_flux": lcp_flux,
            "lcp_bound": lcp_bound,
            "rubber": state.rubber_concentration
        })
        
    # 2. 48時間目にグルコースを大量投入 (抑制を観察)
    sim.state.metabolites["glc__D_e"] = 10.0
    for t in range(24):
        state = sim.step(rubber_degradation_rates={}, nutrient_supplementation={})
        lcp_flux = model_or16.reactions.get_by_id("R_LCP").flux
        lcp_bound = model_or16.reactions.get_by_id("R_LCP").upper_bound
        results.append({
            "time": state.time,
            "biomass": state.species["Actinoplanes_sp_OR16"].biomass,
            "glc": state.metabolites.get("glc__D_e", 0.0),
            "lcp_flux": lcp_flux,
            "lcp_bound": lcp_bound,
            "rubber": state.rubber_concentration
        })

    df = pd.DataFrame(results)
    df.to_csv("lcp_validation_results.csv", index=False)
    
    print("Simulation completed. Results saved to lcp_validation_results.csv")
    print(df.tail(30))

if __name__ == "__main__":
    run_lcp_validation()
