#!/usr/bin/env python3
"""Curate PHB/PHBV metabolism in the NS21 production GEM.

The previous ``PHB_syn`` was a one-step aggregate attached to A4W93_02445.
Current PGAP annotation identifies that protein as DUF3141, whereas the
experimentally supported PHA-synthase locus is A4W93_10485 (current PGAP
A4W93_RS10540).  This script keeps the assembly locus tags used by the model,
but records the current aliases and replaces the aggregate with explicit
PhaB/PhaC reactions for both 3HB and 3HV repeat units.

PHA turnover reactions are included for auditability but disabled.  Their
rates belong in the calibrated dynamic layer; enabling them in a steady-state
LP would create an unconstrained storage/degradation cycle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cobra
from cobra import Metabolite, Reaction
from cobra.manipulation.delete import remove_genes


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    ROOT / "models" / "sbml" / "final_consortium"
    / "Rhizobacter_gummiphilus_NS21.xml"
)

OLD_PHA_GENE = "A4W93_02445"
PHAC = "A4W93_10485"  # current PGAP alias: A4W93_RS10540
PHAA = "A4W93_10490"  # current PGAP alias: A4W93_RS10545
PHAB = "A4W93_10495"  # current PGAP alias: A4W93_RS10550
PHAZ_CANDIDATES = ("A4W93_09590", "A4W93_20030")

SOURCES = {
    "conversion": "https://doi.org/10.1016/j.nbt.2024.08.507",
    "genome": "https://doi.org/10.1128/MRA.00118-19",
    "assembly": "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_002116905.1/",
    "phac_motif": "https://doi.org/10.1038/s41598-017-05509-4",
}


def _met(model: cobra.Model, metabolite_id: str):
    for candidate in (metabolite_id, f"M_{metabolite_id}"):
        if candidate in model.metabolites:
            return model.metabolites.get_by_id(candidate)
    raise KeyError(f"{metabolite_id} is absent from {model.id}")


def _upsert_metabolite(
    model: cobra.Model,
    metabolite_id: str,
    name: str,
    formula: str,
    charge: int,
    compartment: str = "C_c",
) -> Metabolite:
    try:
        metabolite = _met(model, metabolite_id)
    except KeyError:
        metabolite = Metabolite(
            metabolite_id,
            name=name,
            formula=formula,
            charge=charge,
            compartment=compartment,
        )
        model.add_metabolites([metabolite])
    metabolite.name = name
    metabolite.formula = formula
    metabolite.charge = charge
    return metabolite


def _upsert_reaction(model: cobra.Model, reaction_id: str) -> Reaction:
    if reaction_id in model.reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        reaction.subtract_metabolites(reaction.metabolites)
    else:
        reaction = Reaction(reaction_id)
        model.add_reactions([reaction])
    return reaction


def _set_reaction(
    model: cobra.Model,
    reaction_id: str,
    name: str,
    stoichiometry: dict,
    gene_rule: str,
    bounds: tuple[float, float],
    notes: dict,
) -> Reaction:
    reaction = _upsert_reaction(model, reaction_id)
    reaction.name = name
    reaction.add_metabolites(stoichiometry)
    reaction.gene_reaction_rule = gene_rule
    reaction.bounds = bounds
    reaction.notes = notes
    return reaction


def _assert_balanced(reactions: list[Reaction]) -> None:
    failures = {}
    for reaction in reactions:
        imbalance = reaction.check_mass_balance()
        if imbalance:
            failures[reaction.id] = imbalance
    if failures:
        raise RuntimeError(f"PHA curation created unbalanced reactions: {failures}")


def curate(model: cobra.Model) -> cobra.Model:
    # Correct charges for CoA thioesters imported with zero charge in the
    # draft.  The formulas already match the corresponding ModelSEED species.
    _upsert_metabolite(
        model, "3hbcoa__R_c", "(R)-3-hydroxybutanoyl-CoA", 
        "C25H38N7O18P3S", -4,
    )
    _upsert_metabolite(
        model, "3optcoa_c", "3-oxopentanoyl-CoA", "C26H38N7O18P3S", -4,
    )
    _upsert_metabolite(
        model, "R_3hptcoa_c", "(R)-3-hydroxypentanoyl-CoA",
        "C26H40N7O18P3S", -4,
    )
    _upsert_metabolite(
        model, "R_3hpt_c", "(R)-3-hydroxypentanoate", "C5H9O3", -1,
    )
    phb = _upsert_metabolite(
        model, "pha_c", "PHB storage repeat unit (3HB)", "C4H6O2", 0,
    )
    phv = _upsert_metabolite(
        model, "phv_c", "PHBV storage repeat unit (3HV)", "C5H8O2", 0,
    )

    aacoa = _met(model, "aacoa_c")
    optcoa = _met(model, "3optcoa_c")
    hbcoa = _met(model, "3hbcoa__R_c")
    hvcoa = _met(model, "R_3hptcoa_c")
    h = _met(model, "h_c")
    h2o = _met(model, "h2o_c")
    nadph = _met(model, "nadph_c")
    nadp = _met(model, "nadp_c")
    coa = _met(model, "coa_c")
    bhb = _met(model, "bhb_c")
    hval = _met(model, "R_3hpt_c")

    common_notes = {
        "evidence": "pathway structure supported; kinetic capacity not calibrated",
        "experimental_source": SOURCES["conversion"],
        "genome_source": SOURCES["genome"],
        "annotation_source": SOURCES["assembly"],
    }
    reactions = [
        _set_reaction(
            model, "PHB_PhaB", "PhaB acetoacetyl-CoA reductase (3HB branch)",
            {aacoa: -1, h: -1, nadph: -1, hbcoa: 1, nadp: 1},
            PHAB, (0.0, 1000.0), dict(common_notes),
        ),
        _set_reaction(
            model, "PHV_PhaB", "PhaB 3-oxopentanoyl-CoA reductase (3HV branch)",
            {optcoa: -1, h: -1, nadph: -1, hvcoa: 1, nadp: 1},
            PHAB, (0.0, 1000.0),
            {
                **common_notes,
                "substrate_scope": "3-oxopentanoyl-CoA use is a model hypothesis requiring flux/metabolite validation",
            },
        ),
        _set_reaction(
            model, "PHB_syn", "PhaC-catalysed PHB repeat-unit polymerization",
            {hbcoa: -1, coa: 1, phb: 1}, PHAC, (0.0, 1000.0),
            {**common_notes, "phaC_motif_source": SOURCES["phac_motif"]},
        ),
        _set_reaction(
            model, "PHV_syn", "PhaC-catalysed PHBV 3HV-unit polymerization",
            {hvcoa: -1, coa: 1, phv: 1}, PHAC, (0.0, 1000.0),
            {
                **common_notes,
                "phaC_motif_source": SOURCES["phac_motif"],
                "monomer_assignment": "NS21 produces PHBV from natural rubber; composition is not fixed in the LP",
            },
        ),
    ]

    pha_z_rule = " or ".join(PHAZ_CANDIDATES)
    reactions.extend([
        _set_reaction(
            model, "PHB_PhaZ", "Candidate PhaZ-catalysed PHB hydrolysis",
            {phb: -1, h2o: -1, bhb: 1, h: 1}, pha_z_rule, (0.0, 0.0),
            {
                **common_notes,
                "model_role": "disabled audit reaction; turnover requires dynamic calibration",
                "localization_warning": "A4W93_20030 is annotated as extracellular type-2 PHA depolymerase",
            },
        ),
        _set_reaction(
            model, "PHV_PhaZ", "Candidate PhaZ-catalysed PHV hydrolysis",
            {phv: -1, h2o: -1, hval: 1, h: 1}, pha_z_rule, (0.0, 0.0),
            {
                **common_notes,
                "model_role": "disabled audit reaction; turnover requires dynamic calibration",
                "localization_warning": "A4W93_20030 is annotated as extracellular type-2 PHA depolymerase",
            },
        ),
    ])

    for reaction_id, metabolite, label in (
        ("EX_pha_c", phb, "Intracellular PHB storage accounting sink"),
        ("EX_phv_c", phv, "Intracellular PHBV 3HV storage accounting sink"),
    ):
        reaction = _upsert_reaction(model, reaction_id)
        reaction.name = label
        reaction.add_metabolites({metabolite: -1})
        reaction.bounds = (0.0, 1000.0)
        reaction.notes = {
            "model_role": "dynamic intracellular storage sink; not extracellular secretion",
            "experimental_source": SOURCES["conversion"],
        }

    # PhaA is already represented by ACACT1r; retain its alternative isozymes
    # and document the pathway-relevant NS21 locus rather than duplicating the
    # same chemistry.
    acact = model.reactions.get_by_id("ACACT1r")
    acact.notes = {
        **dict(acact.notes),
        "pha_pathway_locus": PHAA,
        "current_pgap_alias": "A4W93_RS10545",
    }

    _assert_balanced(reactions)

    if OLD_PHA_GENE in model.genes and not model.genes.get_by_id(OLD_PHA_GENE).reactions:
        remove_genes(model, [model.genes.get_by_id(OLD_PHA_GENE)], remove_reactions=False)

    model.notes = {
        **dict(model.notes),
        "ns21_pha_curation": (
            "Explicit PhaA/PhaB/PhaC PHB-PHBV model; old A4W93_02445 "
            "assignment removed; PhaZ reactions disabled pending calibration."
        ),
        "ns21_pha_evidence": "; ".join(SOURCES.values()),
    }
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input
    model = cobra.io.read_sbml_model(str(args.input))
    curate(model)
    cobra.io.write_sbml_model(model, str(output))
    print(f"Wrote curated NS21 PHA model: {output}")


if __name__ == "__main__":
    main()
