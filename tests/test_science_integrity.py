
import unittest
import cobra
import os
import numpy as np
from src.dfba_simulator import dFBASimulator, ConsortiumState, SpeciesState

class TestScienceIntegrity(unittest.TestCase):
    def setUp(self):
        self.model_dir = "models/sbml/final_consortium"
        self.lp_path = os.path.join(self.model_dir, "Lactobacillus_plantarum.xml")
        self.ns21_path = os.path.join(self.model_dir, "Rhizobacter_gummiphilus_NS21.xml")
        self.or16_path = os.path.join(self.model_dir, "Actinoplanes_sp_OR16_lcp.xml")

    def test_lp_biosurfactant_secretion(self):
        """Test if LP secretes biosurfactant under growth conditions."""
        model = cobra.io.read_sbml_model(self.lp_path)
        # 糖と栄養素を十分に与える
        for rxn in model.exchanges:
            rxn.lower_bound = -10
        
        solution = model.optimize()
        print(f"\nLP Solver status: {solution.status}")
        # COBRAでは EX_biosurfactant_e
        bs_flux = solution.fluxes.get("EX_biosurfactant_e", 0.0)
        print(f"LP Biosurfactant secretion flux: {bs_flux}")
        self.assertGreater(bs_flux, 1e-6, "LP should secrete biosurfactant")

    def test_ns21_pha_compartment(self):
        """Test if PHA in NS21 is intracellular (cytosol)."""
        model = cobra.io.read_sbml_model(self.ns21_path)
        pha = model.metabolites.get_by_id("pha_c")
        print(f"\nNS21 PHA compartment: {pha.compartment}")
        self.assertEqual(pha.compartment, "C_c", "PHA should be in cytosol (C_c)")

    def test_or16_rubber_degradation_products(self):
        """Test if OR16 degrades rubber without ODTD."""
        model = cobra.io.read_sbml_model(self.or16_path)
        # ODTDが含まれていないことを確認
        odtd_ids = [m.id for m in model.metabolites if "odtd" in m.id.lower()]
        print(f"\nOR16 ODTD-like metabolites: {odtd_ids}")
        self.assertEqual(len(odtd_ids), 0, "OR16 should NOT have ODTD pathway")
        
        # 主要な炭素源（糖など）の摂取を禁止し、ゴムを唯一の主要炭素源にする
        # 完全に閉じるのではなく、主要なものに限定することで生存に必要な微量栄養素を確保
        major_carbon_sources = ["EX_glc__D_e", "EX_glc__aD_e", "EX_fru_e", "EX_sucr_e", "EX_malt_e"]
        for rid in major_carbon_sources:
            if rid in model.reactions:
                model.reactions.get_by_id(rid).lower_bound = 0
            
        # LCP反応が動作することを確認（目的関数をLCPに設定）
        model.objective = "R_LCP"
        model.reactions.EX_rubber_e.lower_bound = -10
        # 他の境界条件を十分に開放
        for rxn in model.exchanges:
            if rxn.id not in major_carbon_sources and rxn.id != "EX_rubber_e":
                 rxn.lower_bound = min(rxn.lower_bound, -10)
        
        solution = model.optimize()
        print(f"OR16 Solver status (Objective=LCP): {solution.status}")
        lcp_flux = solution.fluxes.get("R_LCP", 0.0)
        print(f"OR16 LCP flux: {lcp_flux}")
        self.assertGreater(lcp_flux, 1e-6, "OR16 should be capable of running LCP reaction")

    def test_simulator_bs_boost_application(self):
        """Test if both NS21 and OR16 benefit from BS boost in the simulator."""
        models = {
            'LP': cobra.Model('LP'),
            'NS21': cobra.Model('NS21'),
            'OR16': cobra.Model('OR16')
        }
        initial_biomass = {'LP': 1.0, 'NS21': 1.0, 'OR16': 1.0}
        initial_metabolites = {'biosurfactant_e': 1.0}
        initial_rubber = 100.0
        
        simulator = dFBASimulator(
            models=models,
            initial_biomass=initial_biomass,
            initial_metabolites=initial_metabolites,
            initial_rubber=initial_rubber
        )
        
        simulator.state.species['NS21'].metabolite_uptake = {'R_EX_rubber_e': -1.0}
        simulator.state.species['OR16'].metabolite_uptake = {'EX_rubber_e': -1.0}
        
        simulator.degrade_rubber({})
        consumed_with_bs = initial_rubber - simulator.state.rubber_concentration
        
        simulator.state.rubber_concentration = initial_rubber
        simulator.state.metabolites['biosurfactant_e'] = 0.0
        simulator.degrade_rubber({})
        consumed_without_bs = initial_rubber - simulator.state.rubber_concentration
        
        print(f"\nRubber consumed (with BS): {consumed_with_bs}")
        print(f"Rubber consumed (without BS): {consumed_without_bs}")
        self.assertGreater(consumed_with_bs, consumed_without_bs, "BS should boost rubber degradation")

if __name__ == '__main__':
    unittest.main()
