import pytest
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def test_rubber_degradation_and_pha_accumulation():
    """
    TDD検証: 修正されたモデルとシミュレーターにおいて、ゴムが実際に分解され、
    その結果としてPHAが蓄積されることを証明する結合テスト。
    """
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    
    initial_biomass, initial_metabolites = get_initial_params(models)

    # 理想環境のセットアップ (完全なMinimal Media + ゴム)
    clean_metabolites = {}
    # 全ての代謝物を一度ゼロにリセットする
    for met in initial_metabolites.keys():
        clean_metabolites[met] = 0.0

    # 無機塩類と酸素などを無尽蔵に
    inorganic_salts = ['o2_e', 'pi_e', 'so4_e', 'ca2_e', 'mg2_e', 'fe3_e', 'fe2_e', 'mn2_e', 'zn2_e', 'cu2_e', 'cobalt2_e', 'cl_e', 'k_e', 'na_e', 'ni2_e', 'h2o_e', 'h_e', 'co2_e']
    for met in inorganic_salts:
        if met in clean_metabolites:
            clean_metabolites[met] = 1000.0

    # 初期状態のnh4_eを制限して窒素枯渇(Nitrogen Limitation)を促す
    clean_metabolites['nh4_e'] = 10.0 # 少量だけ与える

    initial_rubber = 100.0
    # max_uptake_rateを大きくして酸素制限をなくす
    sim = dFBASimulator(        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=clean_metabolites,
        initial_rubber=initial_rubber,
        volume=1.0,
        dt=1.0, # 1時間ステップ
        max_uptake_rate=1000.0 # 制限解除
    )

    # 1. 初期状態の確認
    assert sim.state.rubber_concentration == initial_rubber

    # 2. 短時間のシミュレーション実行 (24時間)
    # 理想的なpH(7.0)と栄養補給を維持する「神の手」コントローラー
    max_rubber_uptake = 0.0
    for t in range(24):
        # 枯渇を防ぐための栄養補充 (nh4_eは補充しない)
        for met in ['o2_e', 'pi_e', 'so4_e', 'ca2_e', 'mg2_e', 'fe3_e', 'mn2_e', 'zn2_e', 'cu2_e', 'cobalt2_e', 'cl_e', 'k_e', 'na_e', 'ni2_e']:
            sim.state.metabolites[met] = 1000.0

        # 手動でOptimizeして詳細を確認 (OR16)
        or16_model = sim.models['Actinoplanes_sp_OR16_lcp']
        sim.set_uptake_constraints('Actinoplanes_sp_OR16_lcp', sim.state.metabolites)
        sol_or16 = or16_model.optimize()
        if sol_or16 and sol_or16.status == 'optimal':
            rubber_flux = sol_or16.fluxes.get('EX_rubber_e', 0.0)
        else:
            rubber_flux = 0.0

        # 手動でOptimizeして詳細を確認 (NS21)
        ns21_model = sim.models['Rhizobacter_gummiphilus_NS21']
        sim.set_uptake_constraints('Rhizobacter_gummiphilus_NS21', sim.state.metabolites)
        sol_ns21 = ns21_model.optimize()
        if sol_ns21 and sol_ns21.status == 'optimal':
            ns21_pha_flux = sol_ns21.fluxes.get('EX_pha_c', sol_ns21.fluxes.get('EX_phb_c', 0.0))
        else:
            ns21_pha_flux = 0.0

        # 「神の手」によるpH 7.0 の強制リセット (テスト環境専用)
        sim.buffer_base = sim.buffer_total / 2.0
        sim.buffer_acid = sim.buffer_total / 2.0
        sim.state.metabolites['h_e'] = 10**-4.0 # 単位はmMなので10^-4 mM = 10^-7 M = pH 7.0

        sim.step({}, {})
        # Debug print
        or16_key = [k for k in models.keys() if 'OR16' in k][0]
        or16_state = sim.state.species[or16_key]
        ns21_key = [k for k in models.keys() if 'NS21' in k][0]
        ns21_state = sim.state.species[ns21_key]

        current_uptake = or16_state.metabolite_uptake.get('rubber_e', 0.0)
        max_rubber_uptake = max(max_rubber_uptake, current_uptake)
        
        or16_odtd_sec = or16_state.metabolite_secretion.get('odtd_e', 0.0)
        or16_frag_sec = or16_state.metabolite_secretion.get('rubber_fragment_e', 0.0)
        ns21_odtd_up = ns21_state.metabolite_uptake.get('odtd_e', 0.0)
        ns21_frag_up = ns21_state.metabolite_uptake.get('rubber_fragment_e', 0.0)
        env_odtd = sim.state.metabolites.get('odtd_e', 0.0)
        env_frag = sim.state.metabolites.get('rubber_fragment_e', 0.0)
        env_nh4 = sim.state.metabolites.get('nh4_e', 0.0)

        or16_nh4_sec = or16_state.metabolite_secretion.get('nh4_e', 0.0)
        ns21_nh4_sec = ns21_state.metabolite_secretion.get('nh4_e', 0.0)
        lp_key = [k for k in models.keys() if 'Lactobacillus' in k][0]
        lp_state = sim.state.species[lp_key]
        lp_nh4_sec = lp_state.metabolite_secretion.get('nh4_e', 0.0)

        print(f"t={t:02d}: NH4={env_nh4:.2f}, NH4_sec(OR16={or16_nh4_sec:.2f}, NS21={ns21_nh4_sec:.2f}, LP={lp_nh4_sec:.2f}) | Rubber={sim.state.rubber_concentration:.2f}, "
              f"OR16(Uptake: rub={current_uptake:.2f}, Sec: odtd={or16_odtd_sec:.2f}, frag={or16_frag_sec:.2f}) | "
              f"Env(ODTD={env_odtd:.2f}, Frag={env_frag:.2f}) | "
              f"NS21(Uptake: odtd={ns21_odtd_up:.2f}, frag={ns21_frag_up:.2f}, PHA_flux={ns21_pha_flux:.4f}, PHA_acc={ns21_state.pha_accumulated:.4f})")

    # 3. 結果のアサーション (検証)

    # A. ゴムが明確に減少していること
    final_rubber = sim.state.rubber_concentration
    degraded_amount = initial_rubber - final_rubber
    print(f"\n[Test Result] Rubber Degraded: {degraded_amount:.4f} g/L")
    assert final_rubber < initial_rubber, "Rubber concentration did not decrease!"
    assert degraded_amount > 0.1, "Degradation is too slow to be practical."

    # B. OR16がゴムを取り込んでいること (シミュレーション中の最大値)
    print(f"[Test Result] OR16 Max Rubber Uptake Flux: {max_rubber_uptake:.4f} mmol/gDW/h")
    assert max_rubber_uptake > 0.01, "OR16 is not actively taking up rubber!" # 吸収は正の値として記録される

    # C. NS21がPHAを蓄積していること
    ns21_key = [k for k in models.keys() if 'NS21' in k][0]
    ns21_state = sim.state.species[ns21_key]
    final_pha = ns21_state.pha_accumulated
    print(f"[Test Result] NS21 Accumulated PHA: {final_pha:.4f} mmol")
    assert final_pha > 0.0, "NS21 did not accumulate any PHA!"

    print("✅ All TDD assertions passed. The metabolic bucket brigade is functional.")
