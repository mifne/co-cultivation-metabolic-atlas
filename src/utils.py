import cobra
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable

from src.gene_id_registry import validate_canonical_gene_ids


# Defined 24 h reserve predicted by the corrected three-member shared-medium
# LP at a maintenance growth target of 0.005 h^-1. These are modeling start
# values (mM), not a validated wet-lab recipe. The four amino acids were found
# with iBT721; the repaired 2022 WCFS1 network additionally requires a trace
# pyridoxamine (vitamin B6) pool. Experimental dropout tests must confirm each
# requirement.
COEXISTENCE_DEFINED_MEDIUM_24H = {
    "ile__L_e": 0.5160,
    "phe__L_e": 0.4080,
    "pro__L_e": 0.4800,
    "gln__L_e": 0.0720,
    "pydam_e": 0.0240,
}

# One-pump, chemically defined continuous feed. Values are concentrations
# added per hour at scale=1.0. Common carbon and the four initially depleted
# WCFS1 amino-acid pools are included with the four cost-aware rescue solutes
# and the pyridoxamine requirement exposed by the repaired 2022 network.
COEXISTENCE_DEFINED_FEED_MMOL_L_H = {
    "glc__D_e": 0.5000,
    "arg__L_e": 0.0500,
    "trp__L_e": 0.0500,
    "leu__L_e": 0.0500,
    "glu__L_e": 0.0500,
    "ile__L_e": 0.0215,
    "pro__L_e": 0.0200,
    "phe__L_e": 0.0170,
    "gln__L_e": 0.0030,
    "pydam_e": 0.0010,
}


def validate_consortium_model_identity(species_name: str, model) -> None:
    """Reject a known cross-species model substitution before simulation."""

    if "Lactobacillus_plantarum" not in species_name and "Lactiplantibacillus_plantarum" not in species_name:
        return
    gene_ids = {gene.id for gene in model.genes}
    if model.id == "iNF517" or any(gene.startswith("LLMG_RS") for gene in gene_ids):
        raise ValueError(
            "The selected 'L. plantarum' GEM is iNF517 from Lactococcus lactis "
            "subsp. cremoris MG1363. Use the curated WCFS1 2022 model instead."
        )
    if not any(gene.startswith("lp_") for gene in gene_ids):
        raise ValueError(
            f"The selected L. plantarum model {model.id!r} lacks WCFS1 lp_* locus tags"
        )

def load_sbml_models(sbml_dir: Path) -> dict:
    """
    SBMLファイルからモデルを読み込み
    """
    models = {}
    sbml_files = list(sbml_dir.glob('*.xml')) + list(sbml_dir.glob('*.sbml'))
    
    if not sbml_files:
        raise FileNotFoundError(f"SBMLファイルが見つかりません: {sbml_dir}")
    
    for sbml_file in sbml_files:
        species_name = sbml_file.stem
        try:
            model = cobra.io.read_sbml_model(str(sbml_file))
            models[species_name] = model
        except Exception:
            continue
    
    if not models:
        raise ValueError(f"有効なSBMLモデルが1つも読み込めませんでした: {sbml_dir}")
    
    return models


def select_consortium_models(models: dict, profile: str = "legacy3") -> dict:
    """
    Select the explicitly requested organisms; never fill gaps with another GEM.
    """
    if profile not in {"legacy3", "pf-helper3", "or16-ns21"}:
        raise ValueError(f"Unknown consortium profile: {profile}")
    # SBML file stems include reconstruction/version suffixes.  The previous
    # exact-name-only selection silently substituted Bacillus for NS21/LP when
    # the canonical names did not exist, so training and benchmarks were not
    # actually using the intended three-member consortium.
    priority_species = [
        (
            ('Actinoplanes_sp_OR16_lcp',),
            lambda name: 'OR16' in name,
            'Engine 1: Lcp分解',
        ),
        (
            ('Rhizobacter_gummiphilus_NS21', 'NS21_lcp_pha'),
            lambda name: 'NS21' in name,
            'Engine 2: Rox分解 + PHA蓄積',
        ),
        (
            ('Lactobacillus_plantarum', 'Lactobacillus_plantarum_iNF517'),
            lambda name: 'Lactobacillus_plantarum' in name,
            'Stabilizer: 代謝安定化',
        ),
    ]
    if profile == "or16-ns21":
        priority_species = priority_species[:2]
    elif profile == "pf-helper3":
        priority_species[2] = (
            ('Propionibacterium_freudenreichii_shermanii',),
            lambda name: 'freudenreichii' in name.lower(),
            'Helper candidate: P. freudenreichii',
        )
    selected = {}
    for exact_names, matcher, role in priority_species:
        matches = [name for name in models if name in exact_names or matcher(name)]
        if len(matches) != 1:
            raise ValueError(f"Profile {profile} requires exactly one {role}; found {matches or 'none'}")
        key = matches[0]
        validate_consortium_model_identity(key, models[key])
        validate_canonical_gene_ids(key, models[key])
        selected[key] = models[key]
        print(f"  ✅ {role}: {key}")
    return selected


def select_or16_ns21_models(models: dict) -> dict:
    """Select exactly the OR16 and NS21 production GEMs without fallback.

    A two-member experiment must not silently substitute an unrelated SBML
    model, so this selector fails closed when either organism is absent or
    ambiguous, as does the explicit three-member profile selector.
    """

    exact_names = (
        "Actinoplanes_sp_OR16_lcp",
        "Rhizobacter_gummiphilus_NS21",
    )
    selected = {}
    for exact, marker in zip(exact_names, ("OR16", "NS21")):
        matches = [name for name in models if name == exact]
        if not matches:
            matches = [name for name in models if marker in name]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {marker} model, found {matches or 'none'}"
            )
        name = matches[0]
        validate_canonical_gene_ids(name, models[name])
        selected[name] = models[name]
    return selected


def get_initial_params(models: dict) -> Tuple[dict, dict]:
    """
    初期パラメータを取得（M9 + LP生存用サプリメント）
    """
    initial_biomass = {
        name: 0.5 if 'OR16' in name else 0.1 
        for name in models.keys()
    }
    
    initial_metabolites = {
        'glc__D_e': 0.1, 'nh4_e': 2.0, 'pi_e': 50.0, 'o2_e': 0.25,
        'so4_e': 2.0, 'mg2_e': 2.0, 'ca2_e': 0.1, 'k_e': 10.0, 'cl_e': 10.0,
        'fe3_e': 0.1, 'fe2_e': 0.1, 'h_e': 0.0001, 'h2o_e': 55000.0, 'co2_e': 1.0,
        'zn2_e': 0.01, 'mn2_e': 0.1, 'cu2_e': 0.01, 'cu_e': 0.01, 'cobalt2_e': 0.01,
        'ni2_e': 0.01, 'mobd_e': 0.01,
        'nac_e': 0.1, 'ribflv_e': 0.1, 'pnto__R_e': 0.1, 'thm_e': 0.1,
        'btn_e': 0.1, '4abz_e': 0.1, 'fol_e': 0.1, 'nicnt_e': 0.1,
        'ade_e': 0.1, 'gua_e': 0.1, 'ura_e': 0.1, 'xan_e': 0.1, 'orot_e': 0.1,
        'ins_e': 0.1, 'thymd_e': 0.1,
        
        # --- 必須微量栄養素 (Background Nutrients) ---
        'salchs4fe_e': 0.01, 'tsul_e': 0.01, 'ump_e': 0.01, 'xtsn_e': 0.01, # OR16
        # Phosphotyrosine was previously added as an NS21 nutrient, but its
        # only downstream hydrolysis reaction is gene-free and unnecessary;
        # ordinary glutamate/tyrosine metabolism is used instead.
        'istfrnB_e': 0.01, 'tyrp_e': 0.0,                                   # NS21
        'glu__L_e': 0.01, 'leu__L_e': 0.01, 'nmn_e': 0.01, 'ppi_e': 0.01,   # LP & others
        'arg__L_e': 0.01, 'trp__L_e': 0.01,                                 # Others

        # The corrected WCFS1 supplements are delivered continuously by one
        # common feed pump, not front-loaded here (which causes luxury uptake).
        'ile__L_e': 0.0, 'phe__L_e': 0.0, 'pro__L_e': 0.0, 'gln__L_e': 0.0,

        'yeast_extract_e': 1.0,
        '2mba_e': 0.0,
        'C30_oligo_e': 0.0, 'odtd_e': 0.0,
        'mlttr_e': 0.0, 'ptrc_e': 0.0, 'mnl_e': 0.0,
        'lac__L_e': 0.0, 'ppa_e': 0.0, 'ac_e': 0.0, 'b12_e': 0.0
    }
    return initial_biomass, initial_metabolites


def create_mock_models() -> dict:
    """
    モックのCOBRAモデルを作成
    """
    print("⚠️  モックモデルを使用")
    models = {}
    species_names = ['Sphingobium_japonicum', 'Pseudomonas_putida_KT2440', 'Lactobacillus_plantarum']
    for species in species_names:
        model = cobra.Model(f'{species}_mock')
        glc = cobra.Metabolite('glc_e', compartment='e', name='Glucose')
        nh4 = cobra.Metabolite('nh4_e', compartment='e', name='Ammonium')
        biomass = cobra.Metabolite('biomass', compartment='c', name='Biomass')
        
        ex_glc = cobra.Reaction('EX_glc_e')
        ex_glc.add_metabolites({glc: -1})
        ex_glc.bounds = (-100, 1000)
        
        ex_nh4 = cobra.Reaction('EX_nh4_e')
        ex_nh4.add_metabolites({nh4: -1})
        ex_nh4.bounds = (-100, 1000)
        
        biomass_rxn = cobra.Reaction('BIOMASS')
        biomass_rxn.add_metabolites({glc: -10, nh4: -1, biomass: 1})
        biomass_rxn.lower_bound = 0
        biomass_rxn.upper_bound = 1000
        
        model.add_reactions([ex_glc, ex_nh4, biomass_rxn])
        model.objective = 'BIOMASS'
        models[species] = model
    return models
