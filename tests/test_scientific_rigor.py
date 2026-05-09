import pytest
import numpy as np
import cobra
import os
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv

def get_test_models():
    sbml_dir = 'models/sbml/final_consortium'
    models = {f.replace('.xml', ''): cobra.io.read_sbml_model(os.path.join(sbml_dir, f)) 
              for f in os.listdir(sbml_dir) if f.endswith('.xml')}
    return models

def test_mass_balance_per_step():
    """
    Test 1: 毎ステップの炭素収支が 0.1% 未満であることを検証 (TDD: 厳密な質量保存)
    """
    models = get_test_models()
    initial_biomass = {name: 0.1 for name in models.keys()}
    initial_metabolites = {
        'glc__D_e': 100.0, 'nh4_e': 5.0, 'o2_e': 0.25, 'h_e': 0.0001,
        'arg__L_e': 1.0, 'trp__L_e': 1.0, 'leu__L_e': 1.0,
        'rubber_fragment_e': 0.0, 'odtd_e': 0.0
    }
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        dt=1.0
    )
    
    # C-mol conversion (rough estimation for verification)
    C_CONV = {
        'rubber': 5 * (1000 / 68.12),
        'biomass': 40.65,
        'pha': 4.0, 'co2': 1.0, 'odtd': 15.0, 'fragment': 5.0,
        'glc': 6.0, 'arg': 6.0, 'trp': 11.0, 'leu': 6.0
    }

    def calc_total_c(state, sim):
        c = state.rubber_concentration * C_CONV['rubber']
        c += sum(s.biomass for s in state.species.values()) * C_CONV['biomass']
        c += sum(s.pha_accumulated for s in state.species.values()) * C_CONV['pha']
        c += sim.cumulative_co2_emission * C_CONV['co2']
        for m, val in state.metabolites.items():
            key = m.replace('__D_e','').replace('__L_e','').replace('_e','')
            if key in C_CONV:
                c += val * C_CONV[key]
        return c

    initial_c = calc_total_c(simulator.state, simulator)
    
    # Execute 10 steps
    for _ in range(10):
        # Action: OR16 lcp induction
        state = simulator.step({'Actinoplanes_sp_OR16_lcp': 0.01}, {})
        current_c = calc_total_c(state, simulator)
        
        error = abs(current_c - initial_c) / initial_c
        assert error < 0.001, f"Mass balance error too high: {error*100:.4f}% at time {state.time}"

def test_no_growth_without_carbon():
    """
    Test 2: 炭素源がない場合に増殖しないことを検証 (TDD: 生物学的妥当性)
    """
    models = get_test_models()
    initial_biomass = {name: 0.1 for name in models.keys()}
    # No carbon sources (no glucose, no rubber)
    initial_metabolites = {'o2_e': 0.25, 'nh4_e': 10.0}
    
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=0.0, # NO RUBBER
        dt=1.0
    )
    
    state = simulator.step({}, {})
    for name, s in state.species.items():
        assert s.growth_rate <= 1e-6, f"{name} grew without carbon source!"

def test_ph_buffering_logic():
    """
    Test 3: LPによるpHバッファリングが機能しているか (TDD: メカニズムの因果関係)
    """
    # 詳細はsimulatorの実装に依存するが、h_eの蓄積がbuffering_poolで緩和されるべき
    models = get_test_models()
    initial_biomass = {'Lactobacillus_plantarum': 1.0}
    initial_metabolites = {'glc__D_e': 100.0, 'o2_e': 0.25, 'h_e': 0.0001}
    
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        dt=1.0
    )
    
    # 強制的に酸を出すような状況（あるいはシミュレーション時間の経過）
    # simulator.buffering_pool が正しく減少しているか、あるいはpHの低下が抑制されているかを確認
    initial_h = simulator.state.metabolites['h_e']
    state = simulator.step({}, {})
    # LPは乳酸菌なので通常pHを下げる側だが、buffering_poolがそれを食い止めているはず
    # ここではbuffering_poolが存在すること自体のチェック
    assert hasattr(simulator, 'buffering_pool'), "Simulator missing buffering_pool"
    assert simulator.buffering_pool > 0

def test_shrinking_core_model_existence():
    """
    Test 4: ゴム分解の物理モデル（収縮核モデル等）が実装されているか (TDD: 物理的制約)
    """
    # 現状のコードに実装されているかチェック
    # もし未実装ならこのテストは失敗し、実装を促す
    import inspect
    from src.dfba_simulator import dFBASimulator
    source = inspect.getsource(dFBASimulator.step)
    
    # 表面積(surface area)や半径(radius)などの物理的キーワードが含まれているか
    source = inspect.getsource(dFBASimulator.degrade_rubber)
    physical_keywords = ['surface', 'radius', 'shrinking', 'area']
    found = any(kw in source.lower() for kw in physical_keywords)
    
    # 暫定的に、未実装の場合は失敗させる（TDDの「Red」フェーズ）
    assert found, "Shrinking Core Model or similar physical surface area constraint not found in simulator.degrade_rubber()"

def test_ecfba_constraint_existence():
    """
    Test 5: 酵素量制限 (ecFBA) 的な制約が考慮されているか (TDD: 代謝容量の妥当性)
    """
    import inspect
    from src.dfba_simulator import dFBASimulator
    # step または _apply_dynamic_constraints で、フラックスの合計値に対する制約があるかチェック
    source = inspect.getsource(dFBASimulator)
    
    ec_keywords = ['enzyme', 'protein', 'capacity', 'total_flux', 'crowding']
    found = any(kw in source.lower() for kw in ec_keywords)
    
    assert found, "Enzyme capacity constraint or total flux limitation (ecFBA-like) not found in simulator"

if __name__ == "__main__":
    pytest.main([__file__])
