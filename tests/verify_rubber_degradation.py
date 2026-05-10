import unittest
import numpy as np
import cobra
from pathlib import Path
from src.dfba_simulator import dFBASimulator

class TestRubberDegradation(unittest.TestCase):
    def setUp(self):
        sbml_dir = Path("models/sbml/final_consortium")
        self.models = {
            'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
            'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml")),
            'LP': cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))
        }
        self.initial_metabolites = {
            'glc__D_e': 0.0, 'nh4_e': 10.0, 'o2_e': 0.25, 'pi_e': 10.0,
            'yeast_extract_e': 0.0, 'h2o_e': 55000.0, 'h_e': 1e-4
        }
        self.initial_biomass = {'OR16': 1.0, 'NS21': 1.0, 'LP': 1.0}

    def test_rubber_degradation_and_growth(self):
        """ゴム分解が発生し、かつ増殖（Biomass増加）が物理的に可能であることを検証"""
        sim = dFBASimulator(
            models=self.models,
            initial_biomass=self.initial_biomass,
            initial_metabolites=self.initial_metabolites,
            initial_rubber=100.0,
            dt=5.0 # 長めに回して増殖を確認
        )
        
        state = sim.step({}, {})
        
        # 検証1: ゴム濃度が減少していること (細胞外キネティクスによる)
        self.assertLess(state.rubber_concentration, 100.0, "Rubber concentration did not decrease via kinetics.")
        
        # 検証2: 環境中にオリゴマーが蓄積していること
        c30_conc = state.metabolites.get('C30_oligo_e', 0.0)
        odtd_conc = state.metabolites.get('odtd_e', 0.0)
        print(f"\n[Test Result] C30_oligo_e in environment: {c30_conc:.4e}")
        print(f"[Test Result] odtd_e in environment: {odtd_conc:.4e}")
        self.assertTrue(c30_conc > 0 or odtd_conc > 0, "No oligomers were produced in the environment.")

        # 検証3: 増殖率が正であること
        or16_mu = state.species['OR16'].growth_rate
        ns21_mu = state.species['NS21'].growth_rate
        print(f"[Test Result] Rubber remaining: {state.rubber_concentration:.4f}")
        print(f"[Test Result] OR16 growth rate: {or16_mu:.4e}")
        print(f"[Test Result] NS21 growth rate: {ns21_mu:.4e}")
        
        self.assertGreater(or16_mu, 1e-6, "OR16 is NOT GROWING on oligomers.")
        self.assertGreater(ns21_mu, 1e-6, "NS21 is NOT GROWING on oligomers.")

if __name__ == '__main__':
    unittest.main()
