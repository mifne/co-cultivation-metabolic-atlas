
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
        """The hypothetical biosurfactant route must not be forced."""
        model = cobra.io.read_sbml_model(self.lp_path)
        # 糖と栄養素を十分に与える
        for rxn in model.exchanges:
            rxn.lower_bound = -10
        
        solution = model.optimize()
        print(f"\nLP Solver status: {solution.status}")
        # COBRAでは EX_biosurfactant_e
        bs_flux = solution.fluxes.get("EX_biosurfactant_e", 0.0)
        print(f"LP Biosurfactant secretion flux: {bs_flux}")
        self.assertAlmostEqual(bs_flux, 0.0, places=12)
        # The curated WCFS1 2022 model omits the earlier hypothetical,
        # gene-free glycolipoprotein route. If a curated replacement adds one,
        # it must remain optional and have gene support.
        if "R_GLYCOLIPOPROTEIN_SYN_SEC" in model.reactions:
            reaction = model.reactions.get_by_id("R_GLYCOLIPOPROTEIN_SYN_SEC")
            self.assertEqual(reaction.lower_bound, 0.0)
            self.assertNotEqual(reaction.gene_reaction_rule, "")

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
        
        # 不溶性ポリマーの切断は細胞外速度式が担当し、FBA内では二重計上を防ぐ。
        lcp = model.reactions.get_by_id("R_LCP")
        self.assertEqual(lcp.bounds, (0.0, 0.0))
        self.assertEqual(lcp.check_mass_balance(), {})
        self.assertEqual(
            set(g.id for g in lcp.genes),
            {"ACTI_59630", "ACTI_59640", "ACTI_69520"},
        )

    def test_uncalibrated_biosurfactant_does_not_change_rubber_rate(self):
        """Biosurfactant boost is not applied before an experimental calibration."""
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
        
        simulator.state.species['NS21'].metabolite_uptake = {'EX_rubber_bulk_e': -1.0}
        simulator.state.species['OR16'].metabolite_uptake = {'EX_rubber_bulk_e': -1.0}
        
        simulator.degrade_rubber({})
        consumed_with_bs = initial_rubber - simulator.state.rubber_concentration
        
        simulator.state.rubber_concentration = initial_rubber
        simulator.state.metabolites['biosurfactant_e'] = 0.0
        simulator.degrade_rubber({})
        consumed_without_bs = initial_rubber - simulator.state.rubber_concentration
        
        print(f"\nRubber consumed (with BS): {consumed_with_bs}")
        print(f"Rubber consumed (without BS): {consumed_without_bs}")
        self.assertAlmostEqual(consumed_with_bs, consumed_without_bs, places=12)

if __name__ == '__main__':
    unittest.main()
