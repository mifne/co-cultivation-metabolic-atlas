"""Tests for the opt-in P. freudenreichii exchange curation (src/pf_curation.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cobra
import pytest

from src.metabolite_ids import canonical_metabolite_id
from src.pf_curation import SEED_TO_SHARED_POOL, curate_pf_exchanges

ROOT = Path(__file__).resolve().parents[1]
PF_SBML = ROOT / "models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml"
CONSORTIUM = [ROOT / "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml",
              ROOT / "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"]


def _met(mid, formula, compartment="c", name=None):
    return cobra.Metabolite(mid, formula=formula, compartment=compartment, name=name or mid)


def synthetic_pf():
    """Single-compartment ModelSEED layout like P_sherm_model.xml.

    glucose (already converted, EX_glc__D_e) -> 6 CO2 in the cytosol; CO2 leaves
    through a diffusion transport to ``S_cpd00011_ext`` and the secretion-only
    boundary ``Ex_S_cpd00011_ext``.  Hg2+ and H2S are open sinks, as in the
    source model; ``cpd99999`` has no shared pool.
    """
    m = cobra.Model("pf_synthetic")
    m.compartments = {"c": "Cytoplasm", "e": ""}
    glc_e = _met("glc__D_e", "C6H12O6", "e")
    glc_c = _met("S_cpd00027_c0", "C6H12O6")
    co2_c, co2_x = _met("S_cpd00011_c0", "CO2"), _met("S_cpd00011_ext", "CO2", name="CO2")
    hg_c, hg_x = _met("S_cpd00531_c0", "Hg"), _met("S_cpd00531_ext", "Hg", name="Hg2+")
    hs_c, hs_x = _met("S_cpd00239_c0", "H2S"), _met("S_cpd00239_ext", "H2S", name="H2S")
    xx_c, xx_x = _met("S_cpd99999_c0", "C2H4O2"), _met("S_cpd99999_ext", "C2H4O2", name="X")
    bio = _met("S_biomass_ext", None)

    def rxn(rid, stoich, bounds):
        r = cobra.Reaction(rid)
        r.add_metabolites(stoich)
        r.bounds = bounds
        return r

    m.add_reactions([
        rxn("EX_glc__D_e", {glc_e: -1}, (-10, 0)),
        rxn("GLCt", {glc_e: -1, glc_c: 1}, (0, 1000)),
        rxn("OXID", {glc_c: -1, co2_c: 6}, (0, 1000)),
        rxn("CO2t", {co2_c: -1, co2_x: 1}, (-1000, 1000)),
        rxn("Ex_S_cpd00011_ext", {co2_x: -1}, (0, 1000)),
        rxn("HGt", {hg_x: -1, hg_c: 1}, (-1000, 1000)),
        rxn("Ex_S_cpd00531_ext", {hg_x: -1}, (-1000, 1000)),
        rxn("H2St", {hs_x: -1, hs_c: 1}, (-1000, 1000)),
        rxn("Ex_S_cpd00239_ext", {hs_x: -1}, (-1000, 1000)),
        rxn("XXt", {xx_c: -1, xx_x: 1}, (-1000, 1000)),
        rxn("Ex_S_cpd99999_ext", {xx_x: -1}, (0, 1000)),
        rxn("biomass_c0", {glc_c: -0.01, bio: 1}, (0, 1000)),
        rxn("Ex_S_biomass_ext", {bio: -1}, (0, 1000)),
    ])
    m.objective = "OXID"
    return m


def signature(model):
    return (sorted((r.id, tuple(r.bounds), tuple(sorted((x.id, c) for x, c in r.metabolites.items())),
                    json.dumps(r.annotation, sort_keys=True)) for r in model.reactions),
            sorted((x.id, x.compartment, x.formula) for x in model.metabolites))


def test_synthetic_conversion_outcome():
    source = synthetic_pf()
    assert (len(source.exchanges), len(source.demands), len(source.sinks)) == (1, 3, 2)
    before = source.optimize()
    assert before.fluxes["Ex_S_cpd00011_ext"] == pytest.approx(60)

    model, report = curate_pf_exchanges(source)
    # copy semantics: the input keeps its legacy ids
    assert "Ex_S_cpd00011_ext" in source.reactions and "EX_co2_e" not in source.reactions

    ex = model.reactions.EX_co2_e
    assert ex in model.exchanges and ex.bounds == (0, 1000)
    assert {m.id: c for m, c in ex.metabolites.items()} == {"co2_e": -1}
    assert model.metabolites.co2_e.compartment == "e"
    assert model.metabolites.co2_e.annotation["seed.compound"] == "cpd00011"
    assert "CO2t" in {r.id for r in model.metabolites.co2_e.reactions}   # source transport kept
    assert len(model.reactions) == len(source.reactions)                   # nothing added
    assert model.optimize().fluxes["EX_co2_e"] == pytest.approx(60)
    assert model.reactions.EX_cpd99999_e.bounds == (0, 1000)               # Pf-specific pool
    assert model.reactions.EX_cpd00531_e.bounds == (0, 1000)               # Hg uptake closed
    assert model.reactions.EX_h2s_e.bounds == (-1000, 1000)                # H2S: medium-gated
    assert "Ex_S_biomass_ext" in model.reactions
    assert [s["id"] for s in report["skipped"]] == ["Ex_S_biomass_ext"]
    assert [c["id"] for c in report["closed_sinks"]] == ["EX_cpd00531_e"]
    assert {e["id"] for e in report["original_uptake_allowed"]} == {"EX_h2s_e", "EX_cpd00531_e"}
    assert (report["counts_after"]["exchanges"], report["counts_after"]["demands"],
            report["counts_after"]["sinks"]) == (5, 1, 0)
    for entry in report["converted"]:
        assert not (entry["bounds_after"][0] < 0 and not entry["bounds_before"][0] < 0)


def test_synthetic_idempotent_and_options():
    model, _ = curate_pf_exchanges(synthetic_pf())
    again, report = curate_pf_exchanges(model)
    assert signature(again) == signature(model)
    assert report["summary"]["converted"] == 0 and report["summary"]["closed_sinks"] == 0

    open_model, _ = curate_pf_exchanges(synthetic_pf(), close_toxic_sinks=False)
    assert open_model.reactions.EX_cpd00531_e.bounds == (-1000, 1000)
    closed, rep = curate_pf_exchanges(open_model, close_toxic_sinks=True)
    assert closed.reactions.EX_cpd00531_e.bounds == (0, 1000)
    assert signature(closed) == signature(model)

    source = synthetic_pf()
    same, _ = curate_pf_exchanges(source, inplace=True)
    assert same is source and "EX_co2_e" in source.reactions


def test_synthetic_duplicate_pool_falls_back_deterministically():
    model = synthetic_pf()
    co2 = _met("co2_e", "CO2", "e")
    r = cobra.Reaction("EX_co2_e")
    r.add_metabolites({co2: -1})
    model.add_reactions([r])
    curated, report = curate_pf_exchanges(model)
    assert report["duplicates"] == [{"source": "Ex_S_cpd00011_ext", "wanted": "co2_e", "used": "cpd00011_e",
                                     "reason": "co2_e / EX_co2_e already present in model"}]
    assert "EX_cpd00011_e" in curated.reactions


@pytest.fixture(scope="module")
def real_pf():
    if not PF_SBML.is_file():
        pytest.skip("Pf SBML missing")
    return cobra.io.read_sbml_model(str(PF_SBML))


@pytest.fixture(scope="module")
def curated_real(real_pf):
    return curate_pf_exchanges(real_pf)


def test_real_model_counts_and_bounds(real_pf, curated_real):
    model, report = curated_real
    assert report["counts_before"]["exchanges"] == 23 and report["counts_after"]["exchanges"] == 195
    assert report["counts_after"]["demands"] == 1 and report["counts_after"]["sinks"] == 0
    assert len(model.reactions) == len(real_pf.reactions) and len(model.metabolites) == len(real_pf.metabolites)
    for entry in report["converted"]:
        old = real_pf.reactions.get_by_id(entry["old_id"]).bounds
        new = model.reactions.get_by_id(entry["new_id"]).bounds
        assert new[0] >= old[0] and (new[0] >= 0 or old[0] < 0)
        assert new[1] == old[1]
    for pool in ("co2_e", "succ_e", "for_e", "pyr_e", "glyc_e"):
        rxn = model.reactions.get_by_id(f"EX_{pool}")
        assert rxn in model.exchanges and model.metabolites.get_by_id(pool).compartment == "e"
    for rid in ("EX_cpd00531_e", "EX_cd2_e", "EX_cpd04097_e"):
        assert model.reactions.get_by_id(rid).lower_bound == 0
    pools = [canonical_metabolite_id(next(iter(r.metabolites)).id) for r in model.exchanges]
    assert len(pools) == len(set(pools))
    for rid in ("rxnnew73_c0", "rxnnew74_c0", "rxnnew75_c0", "rxnnew78_c0"):
        assert "curation_note" in model.reactions.get_by_id(rid).annotation
        assert model.reactions.get_by_id(rid).reaction == real_pf.reactions.get_by_id(rid).reaction
    assert model.slim_optimize() == pytest.approx(real_pf.slim_optimize(), rel=1e-9)


def test_real_model_idempotent_and_roundtrip(curated_real, tmp_path):
    model, _ = curated_real
    again, report = curate_pf_exchanges(model)
    assert report["summary"]["converted"] == 0
    assert signature(again) == signature(model)
    path = tmp_path / "pf.xml"
    cobra.io.write_sbml_model(model, str(path))
    back = cobra.io.read_sbml_model(str(path))
    assert len(back.exchanges) == 195
    assert all(back.reactions.get_by_id(r.id).bounds == r.bounds for r in model.reactions)


@pytest.mark.skipif(not all(p.is_file() for p in CONSORTIUM), reason="consortium SBML missing")
def test_shared_pool_map_matches_consortium_annotations(real_pf):
    seed = {}
    for path in CONSORTIUM:
        for r in cobra.io.read_sbml_model(str(path)).exchanges:
            met = next(iter(r.metabolites))
            ids = met.annotation.get("seed.compound", [])
            for cpd in [ids] if isinstance(ids, str) else ids:
                seed.setdefault(cpd, set()).add(canonical_metabolite_id(met.id))
    for cpd, pool in SEED_TO_SHARED_POOL.items():
        assert pool in seed.get(cpd, set()), (cpd, pool)
        assert f"S_{cpd}_ext" in real_pf.metabolites
    heavy = lambda f: sorted((e, n) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f or "") if e != "H")
    assert heavy(real_pf.metabolites.S_cpd00036_ext.formula) == heavy("C4H4O4")
