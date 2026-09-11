import csv
import re
from pathlib import Path

from cobra.io import read_sbml_model

from src.gene_id_registry import resolve_gene_id


MODEL_DIR = Path("models/sbml/final_consortium")
REGISTRY_DIR = Path("models/gene_registry")


def test_production_gprs_use_strain_locus_tags():
    expectations = {
        "Actinoplanes_sp_OR16_lcp.xml": re.compile(r"^ACTI_[0-9]+$"),
        "Rhizobacter_gummiphilus_NS21.xml": re.compile(r"^A4W93_[0-9]+$"),
        "Lactobacillus_plantarum.xml": re.compile(
            r"^(lp_[0-9]+|pWCFS[0-9]+_[0-9]+)$"
        ),
    }
    for filename, pattern in expectations.items():
        model = read_sbml_model(str(MODEL_DIR / filename))
        unexpected = {
            gene.id
            for gene in model.genes
            if gene.id != "spontaneous"
            and pattern.fullmatch(gene.id) is None
        }
        assert unexpected == set()


def test_legacy_fasta_and_protein_ids_resolve_to_same_ns21_gene():
    expected = "CCG_NS21_A4W93_01825"
    assert resolve_gene_id("NS21", "A4W93_01825") == expected
    assert (
        resolve_gene_id("NS21", "lcl_CP015118_1_prot_ARN18758_1_365")
        == expected
    )
    assert resolve_gene_id("NS21", "ARN18758.1") == expected


def test_gene_symbols_are_retained_but_not_used_as_unique_identifiers():
    with (REGISTRY_DIR / "gene_id_aliases.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    fusa = [
        row
        for row in rows
        if row["organism_key"] == "NS21" and row["alias"] == "fusA"
    ]
    assert len(fusa) >= 2
    assert all(row["resolvable"] == "False" for row in fusa)


def test_known_wcfs1_alias_is_collapsed_and_unknown_typo_is_quarantined():
    model = read_sbml_model(str(MODEL_DIR / "Lactobacillus_plantarum.xml"))
    assert "GntK" not in model.genes
    assert "lp1406" not in model.genes
    assert model.reactions.get_by_id("GNK").gene_reaction_rule == "lp_1250"
    assert model.reactions.get_by_id("DAPRPL").gene_reaction_rule == "lp_2019"
    assert resolve_gene_id("WCFS1", "GntK") == "CCG_WCFS1_lp_1250"

    with (REGISTRY_DIR / "gene_id_registry.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    unresolved = [
        row for row in rows if row["mapping_status"] == "retired_unresolved"
    ]
    assert [(row["organism_key"], row["canonical_model_gene_id"]) for row in unresolved] == [
        ("WCFS1", "lp1406")
    ]
