#!/usr/bin/env python3
"""Repair and activate the 2022 Lactiplantibacillus plantarum WCFS1 GEM.

The BioModels MODEL2210190005 deposit contains a disconnected biomass
precursor.  Its biomass reaction consumes a lysine/D-Ala peptidoglycan pool
whose upstream identifiers do not form a connected synthesis route.  WCFS1 is
experimentally known to use an mDAP/D-lactate peptidoglycan precursor, which is
also the biomass component used by the experimentally constrained iBT721
model.  The 2022 network contains that pathway but omits UGMDDS2, the reaction
joining D-Ala-D-Lac to the UDP-MurNAc-mDAP precursor.

This script preserves the larger 2022 network and its GPRs while applying
traceable corrections:

1. restore the published iBT721 WCFS1 biomass composition using metabolites
   already present in the 2022 model; and
2. restore UGMDDS2 from iBT721 with its WCFS1 locus tag ``lp_0518``.
3. collapse the duplicated gene-symbol alias ``GntK`` into its WCFS1 locus
   tag ``lp_1250`` while retaining the alias in the gene registry.

No exchange, sink, demand, or nutrient-bypass reaction is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cobra import Reaction
from cobra.io import read_sbml_model, write_sbml_model


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gene_id_registry import (  # noqa: E402
    canonicalize_model_gene_ids,
    load_registry_config,
    profile_by_key,
)
SOURCE_2022 = (
    ROOT
    / "models/sbml/reference/Lactiplantibacillus_plantarum_WCFS1_Koduru2022.xml"
)
BIOMASS_REFERENCE = (
    ROOT
    / "models/sbml/reference/Lactiplantibacillus_plantarum_WCFS1_iBT721.xml"
)
REPAIRED_REFERENCE = (
    ROOT
    / "models/sbml/reference/Lactiplantibacillus_plantarum_WCFS1_Koduru2022_repaired.xml"
)
PRODUCTION_PATH = ROOT / "models/sbml/final_consortium/Lactobacillus_plantarum.xml"
PROVENANCE_PATH = (
    ROOT / "models/sbml/reference/Lplantarum_WCFS1_Koduru2022_repair.json"
)

EXPECTED_SOURCE_ID = "M_LbPt"
EXPECTED_SOURCE_SHA256 = "80f4ee5f146394faefe6f9e2bf98ab49e6bf1e5a78bba48fd9f3db51e75aa238"
EXPECTED_BIOMASS_REFERENCE_SHA256 = (
    "6e67165128100f25ac547f8cf877cb0241257f72053f0109c02ba858539a8c6a"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_iBT721_metabolite_id(identifier: str) -> str:
    return (
        identifier.replace("_LSQBKTc_RSQBKT", "_c")
        .replace("_LSQBKTe_RSQBKT", "_e")
        .replace("PROT_LPL_v60", "PROT_LPL_v6.0")
    )


def fully_open_growth(model, uptake_bound: float = 20.0) -> float:
    diagnostic = model.copy()
    diagnostic.objective = diagnostic.reactions.get_by_id("biomass")
    for exchange in diagnostic.exchanges:
        exchange.lower_bound = min(float(exchange.lower_bound), -abs(uptake_bound))
        exchange.upper_bound = max(float(exchange.upper_bound), 1000.0)
    solution = diagnostic.optimize()
    return float(solution.objective_value or 0.0)


def repair_model(source_2022: Path, biomass_reference: Path):
    if sha256(source_2022) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "The 2022 source GEM checksum changed; review it before applying curation"
        )
    if sha256(biomass_reference) != EXPECTED_BIOMASS_REFERENCE_SHA256:
        raise RuntimeError(
            "The iBT721 biomass reference checksum changed; review it before curation"
        )

    model = read_sbml_model(str(source_2022))
    reference = read_sbml_model(str(biomass_reference))
    if model.id != EXPECTED_SOURCE_ID:
        raise RuntimeError(f"Unexpected 2022 source model id: {model.id!r}")
    if "UGMDDS2" in model.reactions:
        raise RuntimeError("The 2022 source already contains UGMDDS2; re-audit required")

    biomass = model.reactions.get_by_id("biomass")
    original_biomass = {
        metabolite.id: float(coefficient)
        for metabolite, coefficient in biomass.metabolites.items()
    }
    if original_biomass.get("peptido_LPL_c") != -0.1462:
        raise RuntimeError("The expected disconnected 2022 biomass component was not found")

    reference_biomass = reference.reactions.get_by_id("biomass_LPL60")
    repaired_biomass = {}
    mapping = {}
    for source_metabolite, coefficient in reference_biomass.metabolites.items():
        target_id = _map_iBT721_metabolite_id(source_metabolite.id)
        if target_id not in model.metabolites:
            raise RuntimeError(
                f"Cannot map iBT721 biomass metabolite {source_metabolite.id!r} "
                f"to the 2022 model ({target_id!r})"
            )
        repaired_biomass[model.metabolites.get_by_id(target_id)] = float(coefficient)
        mapping[source_metabolite.id] = target_id

    biomass.subtract_metabolites(dict(biomass.metabolites))
    biomass.add_metabolites(repaired_biomass)
    biomass.notes["curation"] = (
        "Restored the experimentally constrained WCFS1 iBT721 biomass composition. "
        "The deposited 2022 biomass used a disconnected lysine/D-Ala "
        "peptidoglycan pool; WCFS1 uses the mDAP/D-lactate branch represented "
        "by PGlac2_c."
    )

    reference_ugmdds2 = reference.reactions.get_by_id("UGMDDS2")
    ugmdds2 = Reaction(
        "UGMDDS2",
        name=reference_ugmdds2.name,
        lower_bound=0.0,
        upper_bound=1000.0,
    )
    ugmdds2.add_metabolites(
        {
            model.metabolites.get_by_id("alalac_c"): -1.0,
            model.metabolites.get_by_id("atp_c"): -1.0,
            model.metabolites.get_by_id("ugmd_c"): -1.0,
            model.metabolites.get_by_id("adp_c"): 1.0,
            model.metabolites.get_by_id("h_c"): 1.0,
            model.metabolites.get_by_id("pi_c"): 1.0,
            model.metabolites.get_by_id("ugmdalac_c"): 1.0,
        }
    )
    ugmdds2.gene_reaction_rule = reference_ugmdds2.gene_reaction_rule
    ugmdds2.annotation = dict(reference_ugmdds2.annotation)
    ugmdds2.notes["curation"] = (
        "Restored from iBT721. This joins D-Ala-D-Lac to the mDAP-containing "
        "UDP-MurNAc precursor and completes the experimentally supported WCFS1 "
        "D-lactate peptidoglycan pathway."
    )
    model.add_reactions([ugmdds2])

    model.id = "WCFS1_Koduru2022_repaired"
    model.name = (
        "Lactiplantibacillus plantarum WCFS1 "
        "(Koduru2022 network; curated biomass and D-Lac peptidoglycan)"
    )
    model.objective = biomass
    model.notes["curation"] = (
        "MODEL2210190005 network repaired using the WCFS1-specific iBT721 "
        "biomass equation and UGMDDS2 reaction. No exchange or nutrient-bypass "
        "reaction was added."
    )
    config = load_registry_config()
    gene_curation = canonicalize_model_gene_ids(
        model, profile_by_key(config, "WCFS1")
    )
    return model, original_biomass, mapping, gene_curation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_2022)
    parser.add_argument("--biomass-reference", type=Path, default=BIOMASS_REFERENCE)
    parser.add_argument("--reference-output", type=Path, default=REPAIRED_REFERENCE)
    parser.add_argument("--production-output", type=Path, default=PRODUCTION_PATH)
    parser.add_argument("--provenance-output", type=Path, default=PROVENANCE_PATH)
    args = parser.parse_args()

    source_model = read_sbml_model(str(args.source))
    before_growth = fully_open_growth(source_model)
    model, original_biomass, mapping, gene_curation = repair_model(
        args.source, args.biomass_reference
    )
    after_growth = fully_open_growth(model)
    if before_growth > 1e-9:
        raise RuntimeError(f"Expected blocked source biomass, got {before_growth:g} h^-1")
    if after_growth <= 1e-6:
        raise RuntimeError("Repaired model still cannot synthesize biomass")

    for output in (args.reference_output, args.production_output):
        output.parent.mkdir(parents=True, exist_ok=True)
        write_sbml_model(model, str(output))
        round_trip = read_sbml_model(str(output))
        if round_trip.id != model.id or len(round_trip.genes) != len(model.genes):
            raise RuntimeError(f"SBML round-trip validation failed for {output}")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_2022": {
            "path": str(args.source.relative_to(ROOT)),
            "biomodels_accession": "MODEL2210190005",
            "sha256": sha256(args.source),
            "model_id": source_model.id,
            "reactions": len(source_model.reactions),
            "metabolites": len(source_model.metabolites),
            "genes": len(source_model.genes),
        },
        "biomass_reference": {
            "path": str(args.biomass_reference.relative_to(ROOT)),
            "biomodels_accession": "MODEL1507180045",
            "pubmed": "17062565",
            "sha256": sha256(args.biomass_reference),
        },
        "scientific_support": {
            "D_lactate_peptidoglycan_pubmed": "8808932",
            "mDAP_peptidoglycan_pubmed": "21949063",
        },
        "changes": [
            {
                "type": "replace_biomass_stoichiometry",
                "reaction": "biomass",
                "removed_component": "peptido_LPL_c",
                "restored_component": "PGlac2_c",
                "source_reaction": "iBT721 biomass_LPL60",
                "metabolite_mapping": mapping,
                "original_2022_biomass": original_biomass,
            },
            {
                "type": "restore_reaction",
                "reaction": "UGMDDS2",
                "source_reaction": "iBT721 UGMDDS2",
                "gpr": model.reactions.get_by_id("UGMDDS2").gene_reaction_rule,
            },
            {
                "type": "canonicalize_gene_identifiers",
                "policy": "strain-specific locus_tag in GPR; project IDs and aliases in sidecar registry",
                "audit": gene_curation,
            },
        ],
        "validation": {
            "source_fully_open_growth_per_h": before_growth,
            "repaired_fully_open_growth_per_h": after_growth,
            "no_exchange_or_sink_added": True,
            "output_model_id": model.id,
            "reactions": len(model.reactions),
            "metabolites": len(model.metabolites),
            "genes": len(model.genes),
        },
        "outputs": {
            "reference": str(args.reference_output.relative_to(ROOT)),
            "reference_sha256": sha256(args.reference_output),
            "production": str(args.production_output.relative_to(ROOT)),
            "production_sha256": sha256(args.production_output),
        },
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
