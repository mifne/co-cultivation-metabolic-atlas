import unittest
import numpy as np
from src.dfba_simulator import dFBASimulator, ConsortiumState, SpeciesState

class MockModel:
    def __init__(self, id):
        self.id = id
        self.reactions = []
        self.exchanges = []

class TestDegradationNameMatching(unittest.TestCase):
    def test_degrade_rubber_with_full_names(self):
        """フルネームの菌種が存在する場合にゴム分解が行われるか検証"""
        # モックモデル（初期化に必要）
        models = {
            'Actinoplanes_sp_OR16_lcp': MockModel('OR16'),
            'Rhizobacter_gummiphilus_NS21': MockModel('NS21')
        }
        
        sim = dFBASimulator(
            models=models,
            initial_biomass={'Actinoplanes_sp_OR16_lcp': 1.0, 'Rhizobacter_gummiphilus_NS21': 1.0},
            initial_metabolites={},
            initial_rubber=100.0,
            dt=1.0
        )
        
        # 初期化直後の状態確認
        self.assertEqual(sim.state.rubber_concentration, 100.0)
        
        # degrade_rubber 実行
        # 現状のバグがあれば、名前が 'OR16' ではないため分解がスキップされるはず
        sim.degrade_rubber({})
        
        print(f"\n[Test] Rubber after degradation: {sim.state.rubber_concentration}")
        
        # 検証: ゴムが減少していること
        self.assertLess(sim.state.rubber_concentration, 100.0, 
                        "Rubber degradation skipped due to name mismatch!")

if __name__ == '__main__':
    unittest.main()
