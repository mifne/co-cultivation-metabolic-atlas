import cobra

# 1. モデルの読み込み
model_path = 'models/sbml/final_consortium/Lactobacillus_plantarum.xml'
model = cobra.io.read_sbml_model(model_path)

# 2. 新しい代謝物 'biosurfactant_e' の追加
bs = cobra.Metabolite(
    'biosurfactant_e',
    formula='', # 10 kDa 複合体
    name='Glycolipoprotein biosurfactant (10 kDa)',
    compartment='e'
)

# 3. 擬似反応 'R_GLYCOLIPOPROTEIN_SYN_SEC' の作成
# IDマッピングの安全な取得
def get_met(mid):
    try: return model.metabolites.get_by_id(mid)
    except:
        # 代替案
        alt = {'glu__L_c': 'glu__L_c', 'asp__L_c': 'asp__L_c'} # 存在するはず
        return model.metabolites.get_by_id(alt.get(mid, mid))

reaction = cobra.Reaction('R_GLYCOLIPOPROTEIN_SYN_SEC')
reaction.name = 'Glycolipoprotein biosurfactant biosynthesis and secretion'
reaction.subsystem = 'Secondary Metabolism'
reaction.lower_bound = 0.  # 非可逆
reaction.upper_bound = 1000.

# 化学量論の設定
stoichiometry = {
    get_met('udpg_c'): -22.5,
    get_met('udpgal_c'): -8.3,
    get_met('hdeacp_c'): -6.0,
    get_met('ocdacp_c'): -3.2, # 2.0 (Oleoyl) + 1.2 (Stearoyl)
    get_met('phe__L_c'): -4.8,
    get_met('ile__L_c'): -4.4,
    get_met('pro__L_c'): -4.2,
    get_met('leu__L_c'): -2.6,
    get_met('val__L_c'): -1.7,
    get_met('glu__L_c'): -1.2,
    get_met('asp__L_c'): -1.2,
    get_met('gly_c'): -5.3,
    get_met('atp_c'): -168.0,
    get_met('h2o_c'): -168.0,
    # 産物
    bs: 1.0,
    get_met('udp_c'): 30.8,
    get_met('ACP_c'): 9.2,
    get_met('adp_c'): 168.0,
    get_met('pi_c'): 168.0,
    get_met('h_c'): 168.0
}
reaction.add_metabolites(stoichiometry)

# 4. 交換反応 'EX_biosurfactant_e' の追加 (dFBAでの蓄積監視用)
ex_bs = model.add_boundary(bs, type="exchange")
ex_bs.id = 'EX_biosurfactant_e'

model.add_reactions([reaction])

# 5. モデルの保存
cobra.io.write_sbml_model(model, model_path)
print(f"✅ Successfully updated {model_path} with Glycolipoprotein synthesis reaction.")

