import json
from pathlib import Path
import re

import cobra
import pytest

from src.dfba_simulator import ConsortiumState, SpeciesState, dFBASimulator


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "models" / "sbml" / "final_consortium"
    / "Rhizobacter_gummiphilus_NS21.xml"
)


@pytest.fixture(scope="module")
def ns21():
    return cobra.io.read_sbml_model(str(MODEL_PATH))


def test_curated_pha_genes_and_balanced_reactions(ns21):
    expected_rules = {
        "PHB_PhaB": "A4W93_10495",
        "PHV_PhaB": "A4W93_10495",
        "PHB_syn": "A4W93_10485",
        "PHV_syn": "A4W93_10485",
    }
    for reaction_id, gene_id in expected_rules.items():
        reaction = ns21.reactions.get_by_id(reaction_id)
        assert reaction.gene_reaction_rule == gene_id
        assert reaction.check_mass_balance() == {}

    assert "A4W93_02445" not in ns21.genes
    assert ns21.metabolites.get_by_id("pha_c").formula == "C4H6O2"
    assert ns21.metabolites.get_by_id("phv_c").formula == "C5H8O2"


def test_selected_phac_has_conserved_catalytic_motif_but_old_assignment_does_not():
    sequences = {}
    header = ""
    for line in (ROOT / "models" / "genome" / "NS21.faa").read_text().splitlines():
        if line.startswith(">"):
            header = line
            sequences[header] = []
        else:
            sequences[header].append(line.strip())

    def sequence(locus_tag):
        match = next(parts for title, parts in sequences.items() if locus_tag in title)
        return "".join(match)

    assert re.search(r"G.C.G", sequence("A4W93_10485"))
    assert not re.search(r"G.C.G", sequence("A4W93_02445"))


def test_phbv_has_independent_3hb_and_3hv_sinks(ns21):
    assert ns21.reactions.get_by_id("EX_pha_c").reaction == "pha_c --> "
    assert ns21.reactions.get_by_id("EX_phv_c").reaction == "phv_c --> "
    assert ns21.reactions.get_by_id("PHB_syn").metabolites != (
        ns21.reactions.get_by_id("PHV_syn").metabolites
    )


def test_phac_and_phab_knockouts_disable_their_two_branches(ns21):
    with ns21:
        ns21.genes.get_by_id("A4W93_10485").knock_out()
        assert ns21.reactions.get_by_id("PHB_syn").bounds == (0, 0)
        assert ns21.reactions.get_by_id("PHV_syn").bounds == (0, 0)
    with ns21:
        ns21.genes.get_by_id("A4W93_10495").knock_out()
        assert ns21.reactions.get_by_id("PHB_PhaB").bounds == (0, 0)
        assert ns21.reactions.get_by_id("PHV_PhaB").bounds == (0, 0)


def test_phaz_is_explicit_but_disabled_pending_dynamic_calibration(ns21):
    for reaction_id in ("PHB_PhaZ", "PHV_PhaZ"):
        reaction = ns21.reactions.get_by_id(reaction_id)
        assert reaction.bounds == (0, 0)
        assert reaction.check_mass_balance() == {}
        assert "dynamic calibration" in reaction.notes["model_role"]


def test_current_pgap_aliases_resolve_to_model_locus_tags():
    config = json.loads((ROOT / "config" / "gene_id_registry.json").read_text())
    ns21_config = next(
        item for item in config["organisms"] if item["organism_key"] == "NS21"
    )
    aliases = ns21_config["manual_aliases"]
    assert aliases["A4W93_RS10540"]["canonical_locus_tag"] == "A4W93_10485"
    assert aliases["A4W93_RS10550"]["canonical_locus_tag"] == "A4W93_10495"


def test_internal_phbv_sink_is_discovered_by_dfba(ns21):
    simulator = dFBASimulator.__new__(dFBASimulator)
    simulator.models = {"Rhizobacter_gummiphilus_NS21": ns21}
    exchange_map = simulator._identify_exchange_reactions()
    assert exchange_map["Rhizobacter_gummiphilus_NS21"]["pha_c"] == "EX_pha_c"
    assert exchange_map["Rhizobacter_gummiphilus_NS21"]["phv_c"] == "EX_phv_c"


def test_dfba_tracks_phb_and_3hv_without_extracellular_leakage():
    simulator = dFBASimulator.__new__(dFBASimulator)
    simulator.dt = 1.0
    simulator.max_pha_fraction_g_gdcw = 0.5
    simulator.buffer_total = 10.0
    simulator.buffer_base = 5.0
    simulator.buffer_acid = 5.0
    simulator.pKa = 7.0
    simulator.exchange_reactions = {
        "NS21": {"pha_c": "EX_pha_c", "phv_c": "EX_phv_c"}
    }
    simulator.state = ConsortiumState(
        time=0.0,
        species={
            "NS21": SpeciesState(1.0, 0.0, {}, {}),
        },
        metabolites={},
        rubber_concentration=0.0,
    )

    simulator._update_environment(
        {"NS21": {"EX_pha_c": 10.0, "EX_phv_c": 10.0}}
    )
    state = simulator.state.species["NS21"]
    polymer_mass = (
        state.phb_accumulated * simulator.PHB_REPEAT_G_PER_MMOL
        + state.phv_accumulated * simulator.PHV_REPEAT_G_PER_MMOL
    )
    assert polymer_mass == pytest.approx(1.0)
    assert state.pha_accumulated == pytest.approx(
        state.phb_accumulated + state.phv_accumulated
    )
    assert "pha_c" not in simulator.state.metabolites
    assert "phv_c" not in simulator.state.metabolites
