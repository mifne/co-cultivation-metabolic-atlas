#!/usr/bin/env python3
"""Import traceable Lactiplantibacillus plantarum WCFS1 reference GEMs.

The production repository previously labelled the *Lactococcus lactis*
iNF517 model as *L. plantarum*.  This importer downloads two independent
WCFS1 reference models from BioModels, verifies their checksums and repairs
only dangling SBML ``groups`` references in the 2022 deposit.  Group metadata
does not participate in FBA; reactions, bounds, GPRs and stoichiometry are
left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from cobra.io import read_sbml_model, write_sbml_model


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "models" / "sbml" / "reference"
PRODUCTION_PATH = (
    ROOT / "models" / "sbml" / "final_consortium" / "Lactobacillus_plantarum.xml"
)


@dataclass(frozen=True)
class Deposit:
    key: str
    accession: str
    revision: int
    filename: str
    sha256: str
    output_name: str
    expected_model_id: str

    @property
    def url(self) -> str:
        return (
            "https://www.biomodels.org/services/download/get-files/"
            f"{self.accession}/{self.revision}/{self.filename}"
        )


DEPOSITS = {
    "ibt721": Deposit(
        key="ibt721",
        accession="MODEL1507180045",
        revision=2,
        filename="MODEL1507180045_url.xml",
        sha256="4a2544adc4e1facfe25d89394b25116394071f745ddd52e5ff4534a7762328e5",
        output_name="Lactiplantibacillus_plantarum_WCFS1_iBT721.xml",
        expected_model_id="MODEL1507180045",
    ),
    "koduru2022": Deposit(
        key="koduru2022",
        accession="MODEL2210190005",
        revision=2,
        filename="LbPt.xml",
        sha256="031b07a24bc3101fd97331dc93073ab88b27bb2d398d46d680aef32066536a53",
        output_name="Lactiplantibacillus_plantarum_WCFS1_Koduru2022.xml",
        expected_model_id="M_LbPt",
    ),
}


def _download(deposit: Deposit, destination: Path) -> None:
    request = urllib.request.Request(
        deposit.url,
        headers={"User-Agent": "co-cultivation-model-import/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != deposit.sha256:
        raise ValueError(
            f"Checksum mismatch for {deposit.accession}: {digest} != {deposit.sha256}"
        )


def _remove_dangling_group_members(source: Path, destination: Path) -> list[str]:
    """Remove invalid annotation-only group members and return their IDs."""

    tree = ET.parse(source)
    root = tree.getroot()
    all_ids = {
        element.get("id")
        for element in root.iter()
        if element.get("id") is not None
    }
    removed: list[str] = []
    parents = {child: parent for parent in root.iter() for child in parent}
    for element in list(root.iter()):
        id_ref = next(
            (
                value
                for attribute, value in element.attrib.items()
                if attribute.rsplit("}", 1)[-1] == "idRef"
            ),
            None,
        )
        if id_ref is not None and id_ref not in all_ids:
            parent = parents.get(element)
            if parent is not None:
                parent.remove(element)
                removed.append(id_ref)
    tree.write(
        destination,
        encoding="UTF-8",
        xml_declaration=True,
    )
    return sorted(set(removed))


def import_deposit(deposit: Deposit, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lplantarum_gem_") as temp_name:
        temp_dir = Path(temp_name)
        downloaded = temp_dir / deposit.filename
        _download(deposit, downloaded)
        load_path = downloaded
        removed_group_refs: list[str] = []
        if deposit.key == "koduru2022":
            load_path = temp_dir / "group_repaired.xml"
            removed_group_refs = _remove_dangling_group_members(downloaded, load_path)

        model = read_sbml_model(str(load_path))
        if model.id != deposit.expected_model_id:
            raise ValueError(
                f"Unexpected model ID for {deposit.accession}: {model.id!r}"
            )
        if not any(gene.id.startswith("lp_") for gene in model.genes):
            raise ValueError(
                f"{deposit.accession} lacks the expected WCFS1 lp_* locus tags"
            )
        biomass_candidates = [
            reaction
            for reaction in model.reactions
            if "biomass" in reaction.id.lower() or "biomass" in reaction.name.lower()
        ]
        if not biomass_candidates:
            raise ValueError(f"No biomass reaction found in {deposit.accession}")
        # The original objective is retained when valid.  Koduru2022 already
        # points to R_biomass; this assignment also makes the intent explicit
        # after normalization by COBRApy.
        if deposit.key == "koduru2022":
            biomass = next(
                (
                    reaction
                    for reaction in biomass_candidates
                    if reaction.id in {"R_biomass", "biomass"}
                ),
                biomass_candidates[0],
            )
            model.objective = biomass
            model.objective.direction = "max"

        output_path = output_dir / deposit.output_name
        write_sbml_model(model, str(output_path))
        reloaded = read_sbml_model(str(output_path))
        if len(reloaded.reactions) != len(model.reactions):
            raise RuntimeError("Reaction count changed during SBML normalization")

    record = {
        **asdict(deposit),
        "source_url": deposit.url,
        "output_path": str(output_path.relative_to(ROOT)),
        "removed_dangling_group_references": removed_group_refs,
        "model_id": reloaded.id,
        "model_name": reloaded.name,
        "reactions": len(reloaded.reactions),
        "metabolites": len(reloaded.metabolites),
        "genes": len(reloaded.genes),
        "lp_locus_tags": sum(gene.id.startswith("lp_") for gene in reloaded.genes),
    }
    return record


def activate_ibt721(reference_path: Path, production_path: Path) -> dict[str, object]:
    """Write a simulator-compatible WCFS1 model without altering chemistry."""

    model = read_sbml_model(str(reference_path))
    if model.id != DEPOSITS["ibt721"].expected_model_id:
        raise ValueError(f"Refusing to activate unexpected model {model.id!r}")
    for metabolite in model.metabolites:
        if metabolite.id.endswith("_LSQBKTe_RSQBKT"):
            metabolite.compartment = "e"
        elif metabolite.id.endswith("_LSQBKTc_RSQBKT"):
            metabolite.compartment = "c"
    model.id = "iBT721_WCFS1"
    model.name = "Lactiplantibacillus plantarum WCFS1 (iBT721; Teusink 2006)"
    model.annotation.update(
        {
            "biomodels.db": "MODEL1507180045",
            "pubmed": "17062565",
            "ncbi.assembly": "GCF_000203855.1",
        }
    )
    production_path.parent.mkdir(parents=True, exist_ok=True)
    write_sbml_model(model, str(production_path))
    activated = read_sbml_model(str(production_path))
    if activated.id != "iBT721_WCFS1" or len(activated.genes) != len(model.genes):
        raise RuntimeError("Activated WCFS1 model failed round-trip validation")
    return {
        "production_path": str(production_path.relative_to(ROOT)),
        "source_reference": str(reference_path.relative_to(ROOT)),
        "model_id": activated.id,
        "reactions": len(activated.reactions),
        "metabolites": len(activated.metabolites),
        "genes": len(activated.genes),
        "objective_reactions": [
            reaction.id
            for reaction in activated.reactions
            if abs(float(reaction.objective_coefficient)) > 1e-12
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(DEPOSITS),
        default=tuple(DEPOSITS),
    )
    parser.add_argument("--output-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument(
        "--activate-ibt721",
        action="store_true",
        help="Replace the production LP file with the verified WCFS1 iBT721 model",
    )
    args = parser.parse_args()

    records = [
        import_deposit(DEPOSITS[key], args.output_dir)
        for key in args.models
    ]
    provenance_path = args.output_dir / "Lplantarum_WCFS1_provenance.json"
    provenance_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    payload: dict[str, object] = {"references": records}
    if args.activate_ibt721:
        payload["activation"] = activate_ibt721(
            args.output_dir / DEPOSITS["ibt721"].output_name,
            PRODUCTION_PATH,
        )
        (args.output_dir / "Lplantarum_WCFS1_activation.json").write_text(
            json.dumps(payload["activation"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
