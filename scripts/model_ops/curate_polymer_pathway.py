#!/usr/bin/env python3
"""Curate the extracellular natural-rubber pathway in the production GEMs.

The insoluble polymer is deliberately *not* an FBA nutrient. Lcp/Rox
cleavage is represented here as atom-balanced annotation reactions, while the
corresponding rates are applied by :class:`src.dfba_simulator.dFBASimulator`.
Only the soluble products produced by that kinetic layer can enter the GEMs.

The C30 species is a coarse-grained mean pool, not a claim that Lcp or RoxB
has a single C30 product. Experimental work reports a distribution of
oligo-isoprenoids, whereas RoxA forms the C15 product ODTD processively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cobra
from cobra import Metabolite, Reaction

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "sbml" / "final_consortium"
OR16_PATH = MODEL_DIR / "Actinoplanes_sp_OR16_lcp.xml"
NS21_PATH = MODEL_DIR / "Rhizobacter_gummiphilus_NS21.xml"

OR16_LCP_GENES = ("ACTI_59630", "ACTI_59640", "ACTI_69520")
NS21_ROXB_GENE = "A4W93_01825"  # latA1; endo-type RoxB orthologue
NS21_ROXA_GENE = "A4W93_07150"  # latA2; processive exo-type RoxA

SOURCES = {
    "OR16_lcp": "https://doi.org/10.1128/AEM.01275-20",
    "NS21_genome": "https://doi.org/10.1128/MRA.00118-19",
    "RoxA_RoxB_synergy": "https://doi.org/10.1128/AEM.01721-18",
    "product_distribution": "https://doi.org/10.1128/AEM.01285-14",
}


def _get_metabolite(model: cobra.Model, metabolite_id: str):
    for candidate in (metabolite_id, f"M_{metabolite_id}"):
        if candidate in model.metabolites:
            return model.metabolites.get_by_id(candidate)
    raise KeyError(f"{metabolite_id} is not present in {model.id}")


def _upsert_metabolite(model, metabolite_id, name, formula, compartment):
    try:
        metabolite = _get_metabolite(model, metabolite_id)
    except KeyError:
        metabolite = Metabolite(
            metabolite_id, name=name, formula=formula, compartment=compartment
        )
        model.add_metabolites([metabolite])
    metabolite.name = name
    metabolite.formula = formula
    metabolite.charge = 0
    return metabolite


def _upsert_reaction(model: cobra.Model, reaction_id: str) -> Reaction:
    if reaction_id in model.reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        reaction.subtract_metabolites(reaction.metabolites)
        return reaction
    reaction = Reaction(reaction_id)
    model.add_reactions([reaction])
    return reaction


def _set_exchange(model, reaction_id, metabolite, bounds):
    reaction = _upsert_reaction(model, reaction_id)
    reaction.name = f"Exchange for {metabolite.name}"
    reaction.add_metabolites({metabolite: -1.0})
    reaction.bounds = bounds
    return reaction


def _set_external_annotation(reaction, source_key, assumption):
    # The kinetic dFBA layer owns these rates; zero FBA bounds prevent carbon
    # from being consumed independently by both representations.
    reaction.bounds = (0.0, 0.0)
    reaction.notes = {
        "model_role": "extracellular kinetic reaction; disabled in FBA",
        "source": SOURCES[source_key],
        "coarse_graining": assumption,
    }


def _remove_obsolete_polymer_objects(model, reaction_ids, metabolite_ids):
    """Remove superseded routes that could bypass the curated pathway."""
    obsolete_reactions = [
        model.reactions.get_by_id(item) for item in reaction_ids if item in model.reactions
    ]
    if obsolete_reactions:
        model.remove_reactions(obsolete_reactions, remove_orphans=False)
    obsolete_metabolites = [
        model.metabolites.get_by_id(item)
        for item in metabolite_ids
        if item in model.metabolites and not model.metabolites.get_by_id(item).reactions
    ]
    if obsolete_metabolites:
        model.remove_metabolites(obsolete_metabolites, destructive=False)


def _balanced_c30_catabolism(model, c30_c):
    """Element/charge-balanced pathway-level beta-oxidation equivalent."""
    return {
        c30_c: -1.0,
        # The representative C30 pool has two terminal oxygens, whereas two
        # ODTD-equivalent C15 products contain four; one additional O2 closes
        # that oxidative processing step before beta oxidation.
        _get_metabolite(model, "o2_c"): -1.0,
        _get_metabolite(model, "coa_c"): -12.0,
        _get_metabolite(model, "h2o_c"): -8.0,
        _get_metabolite(model, "nad_c"): -8.0,
        _get_metabolite(model, "fad_c"): -6.0,
        _get_metabolite(model, "accoa_c"): 6.0,
        _get_metabolite(model, "ppcoa_c"): 6.0,
        _get_metabolite(model, "nadh_c"): 8.0,
        _get_metabolite(model, "fadh2_c"): 6.0,
        _get_metabolite(model, "h_c"): 8.0,
    }


def _balanced_odtd_catabolism(model, odtd_c):
    return {
        odtd_c: -1.0,
        _get_metabolite(model, "coa_c"): -6.0,
        _get_metabolite(model, "h2o_c"): -4.0,
        _get_metabolite(model, "nad_c"): -4.0,
        _get_metabolite(model, "fad_c"): -3.0,
        _get_metabolite(model, "accoa_c"): 3.0,
        _get_metabolite(model, "ppcoa_c"): 3.0,
        _get_metabolite(model, "nadh_c"): 4.0,
        _get_metabolite(model, "fadh2_c"): 3.0,
        _get_metabolite(model, "h_c"): 4.0,
    }


def curate_or16(model: cobra.Model) -> cobra.Model:
    rubber = _upsert_metabolite(
        model, "M_rubber_bulk_e", "Natural-rubber cis-1,4-isoprene unit", "C5H8", "C_e"
    )
    c30_e = _upsert_metabolite(
        model, "M_C30_oligo_e",
        "Representative oxygenated oligo-isoprenoid pool (C30 mean)",
        "C30H48O2", "C_e",
    )
    c30_c = _upsert_metabolite(
        model, "M_C30_oligo_c",
        "Representative oxygenated oligo-isoprenoid pool (C30 mean)",
        "C30H48O2", "C_c",
    )

    lcp = _upsert_reaction(model, "R_LCP")
    lcp.name = "Lcp-catalysed oxidative endo-cleavage of natural rubber"
    lcp.add_metabolites(
        {rubber: -6.0, _get_metabolite(model, "o2_e"): -1.0, c30_e: 1.0}
    )
    lcp.gene_reaction_rule = " or ".join(OR16_LCP_GENES)
    _set_external_annotation(
        lcp, "OR16_lcp",
        "C30 is a carbon-balanced representative of the measured C20+ product distribution.",
    )

    transport = _upsert_reaction(model, "R_C30t")
    transport.name = "Hypothetical ATP-coupled uptake of oxygenated C30 oligomer"
    transport.add_metabolites({
        c30_e: -1.0,
        _get_metabolite(model, "atp_c"): -1.0,
        _get_metabolite(model, "h2o_c"): -1.0,
        c30_c: 1.0,
        _get_metabolite(model, "adp_c"): 1.0,
        _get_metabolite(model, "pi_c"): 1.0,
        _get_metabolite(model, "h_c"): 1.0,
    })
    transport.bounds = (0.0, 0.5)
    transport.notes = {
        "evidence": "hypothesis; transporter and ATP cost require experimental validation"
    }

    catabolism = _upsert_reaction(model, "R_C30_cat")
    catabolism.name = "Balanced coarse-grained C30 beta-oxidation equivalent"
    catabolism.add_metabolites(_balanced_c30_catabolism(model, c30_c))
    catabolism.bounds = (0.0, 1000.0)
    catabolism.notes = {
        "evidence": "element-balanced pathway-level closure; redox stoichiometry is a modelling assumption"
    }

    _set_exchange(model, "EX_rubber_bulk_e", rubber, (0.0, 0.0))
    _set_exchange(model, "EX_C30_oligo_e", c30_e, (-1000.0, 1000.0))
    _remove_obsolete_polymer_objects(
        model,
        (
            "RUBBERt", "ISOP_ALDH", "EX_rubber_e",
            "EX_rubber_fragment_e", "EX_oligomer_e",
        ),
        (
            "rubber_high_e", "rubber_c", "isoprenoid_aldehyde_c",
            "isoprenoid_acid_c", "rubber_fragment_e", "M_rubber_e",
            "M_rubber_c", "M_isoprenoid_aldehyde_c",
            "M_isoprenoid_acid_c", "M_rubber_fragment_e", "oligomer_e",
        ),
    )
    return model


def curate_ns21(model: cobra.Model) -> cobra.Model:
    rubber = _upsert_metabolite(
        model, "M_rubber_bulk_e", "Natural-rubber cis-1,4-isoprene unit", "C5H8", "C_e"
    )
    c30_e = _upsert_metabolite(
        model, "M_C30_oligo_e",
        "Representative oxygenated oligo-isoprenoid pool (C30 mean)",
        "C30H48O2", "C_e",
    )
    c30_c = _upsert_metabolite(
        model, "M_C30_oligo_c",
        "Representative oxygenated oligo-isoprenoid pool (C30 mean)",
        "C30H48O2", "C_c",
    )
    odtd_e = _upsert_metabolite(model, "M_odtd_e", "ODTD", "C15H24O2", "C_e")
    odtd_c = _upsert_metabolite(model, "M_odtd_c", "ODTD", "C15H24O2", "C_c")
    oxygen = _get_metabolite(model, "o2_e")

    roxb = _upsert_reaction(model, "R_ROXB")
    roxb.name = "RoxB-catalysed oxidative endo-cleavage of natural rubber"
    roxb.add_metabolites({rubber: -6.0, oxygen: -1.0, c30_e: 1.0})
    roxb.gene_reaction_rule = NS21_ROXB_GENE
    _set_external_annotation(
        roxb, "RoxA_RoxB_synergy",
        "C30 is a carbon-balanced representative of the RoxB C20+ product distribution.",
    )

    roxa = _upsert_reaction(model, "R_ROXA")
    roxa.name = "RoxA conversion of an oxygenated C30 oligomer to ODTD"
    roxa.add_metabolites({c30_e: -1.0, oxygen: -1.0, odtd_e: 2.0})
    roxa.gene_reaction_rule = NS21_ROXA_GENE
    _set_external_annotation(
        roxa, "RoxA_RoxB_synergy",
        "C30 represents a RoxB-generated/free-ended oligomer available to processive RoxA.",
    )

    # RoxA is not absolutely dependent on RoxB; it can start from accessible
    # polymer chain ends.
    roxa_bulk = _upsert_reaction(model, "R_ROXA_BULK")
    roxa_bulk.name = "RoxA processive cleavage from a natural-rubber chain end"
    roxa_bulk.add_metabolites({rubber: -3.0, oxygen: -1.0, odtd_e: 1.0})
    roxa_bulk.gene_reaction_rule = NS21_ROXA_GENE
    _set_external_annotation(
        roxa_bulk, "RoxA_RoxB_synergy",
        "Three C5 units to one C15 ODTD is a balanced chain-end cleavage equivalent.",
    )

    odtd_transport = _upsert_reaction(model, "R_ODTDt")
    odtd_transport.name = "ODTD uptake (mechanism unresolved)"
    odtd_transport.add_metabolites({odtd_e: -1.0, odtd_c: 1.0})
    odtd_transport.bounds = (0.0, 1000.0)
    odtd_transport.notes = {"evidence": "transport mechanism and GPR are unresolved"}

    odtd_catabolism = _upsert_reaction(model, "R_ODTD_cat")
    odtd_catabolism.name = "Balanced coarse-grained ODTD beta-oxidation equivalent"
    odtd_catabolism.add_metabolites(_balanced_odtd_catabolism(model, odtd_c))
    odtd_catabolism.bounds = (0.0, 1000.0)
    odtd_catabolism.notes = {
        "evidence": "element-balanced pathway-level closure; redox stoichiometry is a modelling assumption"
    }

    # Direct intracellular C30 uptake would bypass extracellular RoxA.
    c30_transport = _upsert_reaction(model, "R_C30t")
    c30_transport.name = "Direct C30 uptake disabled; extracellular RoxA route required"
    c30_transport.add_metabolites({c30_e: -1.0, c30_c: 1.0})
    c30_transport.bounds = (0.0, 0.0)
    if "R_C30_cat" in model.reactions:
        c30_catabolism = model.reactions.get_by_id("R_C30_cat")
        c30_catabolism.subtract_metabolites(c30_catabolism.metabolites)
        c30_catabolism.add_metabolites(_balanced_c30_catabolism(model, c30_c))
        c30_catabolism.bounds = (0.0, 0.0)
        c30_catabolism.notes = {"model_role": "disabled bypass; use RoxA then ODTD"}

    _set_exchange(model, "EX_rubber_bulk_e", rubber, (0.0, 0.0))
    _set_exchange(model, "EX_C30_oligo_e", c30_e, (-1000.0, 1000.0))
    _set_exchange(model, "EX_odtd_e", odtd_e, (-1000.0, 1000.0))
    _remove_obsolete_polymer_objects(
        model,
        ("LATA1", "LATA2", "EX_rubber_e", "EX_rubber_fragment_e", "EX_oligomer_e"),
        ("rubber_high_e", "rubber_fragment_e"),
    )
    return model


def validate_curated_model(model: cobra.Model, reaction_ids: tuple[str, ...]):
    failures = {}
    for reaction_id in reaction_ids:
        residual = model.reactions.get_by_id(reaction_id).check_mass_balance()
        if residual:
            failures[reaction_id] = residual
    if failures:
        raise ValueError(f"Unbalanced curated reactions in {model.id}: {failures}")


def curate_files(or16_path: Path = OR16_PATH, ns21_path: Path = NS21_PATH):
    or16 = curate_or16(cobra.io.read_sbml_model(or16_path))
    ns21 = curate_ns21(cobra.io.read_sbml_model(ns21_path))
    validate_curated_model(or16, ("R_LCP", "R_C30t", "R_C30_cat"))
    validate_curated_model(
        ns21,
        ("R_ROXB", "R_ROXA", "R_ROXA_BULK", "R_ODTDt", "R_ODTD_cat", "R_C30_cat"),
    )
    cobra.io.write_sbml_model(or16, or16_path)
    cobra.io.write_sbml_model(ns21, ns21_path)
    return {
        "or16": str(or16_path), "ns21": str(ns21_path),
        "or16_lcp_genes": list(OR16_LCP_GENES),
        "ns21_roxb_gene": NS21_ROXB_GENE,
        "ns21_roxa_gene": NS21_ROXA_GENE,
        "sources": SOURCES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--or16", type=Path, default=OR16_PATH)
    parser.add_argument("--ns21", type=Path, default=NS21_PATH)
    args = parser.parse_args()
    print(json.dumps(curate_files(args.or16, args.ns21), indent=2))


if __name__ == "__main__":
    main()
