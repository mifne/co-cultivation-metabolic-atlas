import cobra
import pytest
from cobra.io import read_sbml_model
from pathlib import Path

from src.utils import select_consortium_models, validate_consortium_model_identity


def test_selects_intended_versioned_three_species_names():
    names = [
        "Bacillus_subtilis_168_iYO844",
        "Actinoplanes_sp_OR16_lcp",
        "NS21_lcp_pha",
        "Lactobacillus_plantarum",
    ]
    models = {name: cobra.Model(name) for name in names}
    marker = cobra.Reaction("WCFS1_marker")
    marker.gene_reaction_rule = "lp_0001"
    models["Lactobacillus_plantarum"].add_reactions([marker])
    selected = select_consortium_models(models)
    assert list(selected) == [
        "Actinoplanes_sp_OR16_lcp",
        "NS21_lcp_pha",
        "Lactobacillus_plantarum",
    ]


def test_rejects_lactococcus_inf517_mislabelled_as_l_plantarum():
    model = cobra.Model("iNF517")
    marker = cobra.Reaction("lactococcus_marker")
    marker.gene_reaction_rule = "LLMG_RS00010"
    model.add_reactions([marker])

    try:
        validate_consortium_model_identity("Lactobacillus_plantarum", model)
    except ValueError as exc:
        assert "Lactococcus lactis" in str(exc)
    else:
        raise AssertionError("mislabelled iNF517 model was accepted")


def test_production_l_plantarum_model_is_wcfs1_not_inf517():
    path = Path("models/sbml/final_consortium/Lactobacillus_plantarum.xml")
    model = read_sbml_model(str(path))
    assert model.id == "WCFS1_Koduru2022_repaired"
    assert len(model.reactions) == 1209
    assert len(model.genes) == 913
    assert sum(gene.id.startswith("lp_") for gene in model.genes) >= 912
    assert "GntK" not in model.genes
    assert model.reactions.get_by_id("GNK").gene_reaction_rule == "lp_1250"
    validate_consortium_model_identity("Lactobacillus_plantarum", model)


def test_missing_requested_organism_is_not_replaced_with_unrelated_model():
    names = ['Actinoplanes_sp_OR16_lcp', 'Rhizobacter_gummiphilus_NS21', 'Bacillus_subtilis']
    with pytest.raises(ValueError, match='exactly one'):
        select_consortium_models({name: cobra.Model(name) for name in names})


def test_explicit_pf_profile_selects_pf_and_rejects_ambiguous_reconstructions():
    names = ['Actinoplanes_sp_OR16_lcp', 'Rhizobacter_gummiphilus_NS21',
             'Propionibacterium_freudenreichii_shermanii', 'Bacillus_subtilis']
    models = {name: cobra.Model(name) for name in names}
    assert list(select_consortium_models(models, profile='pf-helper3')) == names[:3]
    models['Other_freudenreichii_version'] = cobra.Model('Other')
    with pytest.raises(ValueError, match='exactly one'):
        select_consortium_models(models, profile='pf-helper3')
