"""Opt-in exchange curation for the P. freudenreichii subsp. shermanii GEM.

The published pan-model reconstruction (Machado et al. 2020, Genes 11:1115,
``P_sherm_model.xml``) is a single-compartment SBML file: every extracellular
species is written as ``S_<id>_ext`` with compartment ``c``.  The ``_ext``
species are nevertheless *extracellular*: each one is already connected to its
cytosolic ``S_<id>_c0`` counterpart by the model's own transport reaction
(e.g. ``rxn05467_c0`` CO2 diffusion, ``rxn10159_c0`` succinate:fumarate
antiporter).  ``scripts/analysis/screen_propionibacterium_helper.py`` fixed 23
of these boundaries by renaming the metabolite to a shared BiGG pool id,
setting its compartment to ``e`` and renaming ``Ex_S_<id>_ext`` to
``EX_<pool>``.  The remaining 173 were left as ``c`` boundaries, which COBRA
classifies as demands (secretion only) or sinks, so their flux never reaches a
shared extracellular pool.

:func:`curate_pf_exchanges` applies exactly the same convention to the rest.
Because the transport step c0 <-> ext already exists in the source network, no
transport reaction is added: the ``_ext`` species simply becomes the ``e``
metabolite of the pool.  Bounds are preserved, so nothing that was
secretion-only becomes uptake-capable.

Nothing in the repository calls this function by default.  It returns a
curated **copy** unless ``inplace=True``; it is idempotent.
"""
from __future__ import annotations

import re
from typing import Any

import cobra

__all__ = ["curate_pf_exchanges", "SEED_TO_SHARED_POOL", "pf_specific_pool"]

# ModelSEED compound -> shared pool id of the other two consortium species.
# Derived 2026-09-30 from the ``seed.compound`` annotations of the exchange
# metabolites of Actinoplanes sp. OR16 and Rhizobacter gummiphilus NS21
# (models/sbml/final_consortium), accepted only when the ModelSEED id matches a
# single pool and the heavy-atom formula (H ignored: protonation) agrees with
# the Pf ``_ext`` metabolite.  tests/test_pf_curation.py re-derives it.
SEED_TO_SHARED_POOL: dict[str, str] = {
    "cpd00011": "co2_e",  # CO2
    "cpd00020": "pyr_e",  # Pyruvate
    "cpd00023": "glu__L_e",  # L-Glutamate
    "cpd00024": "akg_e",  # 2-Oxoglutarate
    "cpd00033": "gly_e",  # Glycine
    "cpd00035": "ala__L_e",  # L-Alanine
    "cpd00036": "succ_e",  # Succinate
    "cpd00039": "lys__L_e",  # L-Lysine
    "cpd00041": "asp__L_e",  # L-Aspartate
    "cpd00046": "cmp_e",  # CMP
    "cpd00047": "for_e",  # Formate
    "cpd00051": "arg__L_e",  # L-Arginine
    "cpd00053": "gln__L_e",  # L-Glutamine
    "cpd00054": "ser__L_e",  # L-Serine
    "cpd00060": "met__L_e",  # L-Methionine
    "cpd00063": "ca2_e",  # Ca2+
    "cpd00064": "orn_e",  # Ornithine
    "cpd00065": "trp__L_e",  # L-Tryptophan
    "cpd00066": "phe__L_e",  # L-Phenylalanine
    "cpd00069": "tyr__L_e",  # L-Tyrosine
    "cpd00073": "urea_e",  # Urea
    "cpd00075": "no2_e",  # Nitrite
    "cpd00076": "sucr_e",  # Sucrose
    "cpd00080": "glyc3p_e",  # Glycerol-3-phosphate
    "cpd00082": "fru_e",  # D-Fructose
    "cpd00084": "cys__L_e",  # L-Cysteine
    "cpd00092": "ura_e",  # Uracil
    "cpd00100": "glyc_e",  # Glycerol
    "cpd00105": "rib__D_e",  # D-Ribose
    "cpd00106": "fum_e",  # Fumarate
    "cpd00107": "leu__L_e",  # L-Leucine
    "cpd00108": "gal_e",  # Galactose
    "cpd00116": "meoh_e",  # Methanol
    "cpd00117": "ala__D_e",  # D-Alanine
    "cpd00118": "ptrc_e",  # Putrescine
    "cpd00119": "his__L_e",  # L-Histidine
    "cpd00121": "inost_e",  # L-Inositol
    "cpd00129": "pro__L_e",  # L-Proline
    "cpd00130": "mal__L_e",  # L-Malate
    "cpd00132": "asn__L_e",  # L-Asparagine
    "cpd00137": "cit_e",  # Citrate
    "cpd00139": "glyclt_e",  # Glycolate
    "cpd00142": "acac_e",  # Acetoacetate
    "cpd00154": "xyl__D_e",  # Xylose
    "cpd00156": "val__L_e",  # L-Valine
    "cpd00161": "thr__L_e",  # L-Threonine
    "cpd00162": "etha_e",  # Aminoethanol
    "cpd00164": "glcur_e",  # Glucuronate
    "cpd00185": "arab__D_e",  # D-Arabinose
    "cpd00204": "co_e",  # CO
    "cpd00207": "gua_e",  # Guanine
    "cpd00208": "lcts_e",  # LACT
    "cpd00209": "no3_e",  # Nitrate
    "cpd00220": "ribflv_e",  # Riboflavin
    "cpd00226": "hxan_e",  # HYXN
    "cpd00227": "hom__L_e",  # L-Homoserine
    "cpd00239": "h2s_e",  # H2S
    "cpd00249": "uri_e",  # Uridine
    "cpd00266": "crn_e",  # Carnitine
    "cpd00268": "tsul_e",  # H2S2O3
    "cpd00280": "galur_e",  # D-Galacturonate
    "cpd00281": "4abut_e",  # GABA
    "cpd00305": "thm_e",  # Thiamin
    "cpd00307": "csn_e",  # Cytosine
    "cpd00309": "xan_e",  # XAN
    "cpd00322": "ile__L_e",  # L-Isoleucine
    "cpd00359": "indole_e",  # indol
    "cpd00363": "etoh_e",  # Ethanol
    "cpd00382": "raffin_e",  # Melitose
    "cpd00396": "rmn_e",  # L-Rhamnose
    "cpd00438": "dad_2_e",  # Deoxyadenosine
    "cpd00540": "glyb_e",  # BET
    "cpd00588": "sbt__D_e",  # Sorbitol
    "cpd00589": "tag__D_e",  # D-Tagatose
    "cpd00637": "met__D_e",  # D-Methionine
    "cpd00794": "tre_e",  # TRHL
    "cpd01012": "cd2_e",  # Cd2+
    "cpd01017": "cgly_e",  # Cys-Gly
    "cpd01030": "salcn_e",  # Salicin
    "cpd01242": "drib_e",  # Thyminose
    "cpd01262": "malttr_e",  # Amylotriose
    "cpd01307": "abt__D_e",  # D-Lyxitol
    "cpd01553": "2hxmp_e",  # Saligenin
    "cpd01711": "ibt_e",  # Isobutyrate
    "cpd03161": "peamn_e",  # Phenethylamine
    "cpd03696": "arbt_e",  # Ursin
    "cpd03725": "fe3dcit_e",  # Fe(III)dicitrate
    "cpd11581": "gly_asn__L_e",  # gly-asn-L
    "cpd11586": "ala_L_glu__L_e",  # ala-L-glu-L
    "cpd11588": "gly_pro__L_e",  # gly-pro-L
    "cpd11589": "gly_asp__L_e",  # gly-asp-L
    "cpd11590": "met_L_ala__L_e",  # met-L-ala-L
    "cpd11592": "gly_glu__L_e",  # gly-glu-L
    "cpd11593": "ala_L_asp__L_e",  # ala-L-asp-L
    "cpd15378": "4hba_e",  # 4-Hydroxy-benzylalcohol
    "cpd15380": "5drib_e",  # 5'-deoxyribose
    "cpd15605": "glyphe_e",  # Gly-Phe
    # Ambiguous in the other species: OR16 uses glcn__D_e, NS21 uses glcn_e for
    # the same cpd00222 (C6H11O7).  glcn_e is the BiGG universal id; chosen
    # deterministically and reported as a duplicate-pool finding.
    "cpd00222": "glcn_e",  # GLCN
}

# ModelSEED ids whose annotation matches a shared pool but which are NOT merged.
SHARED_POOL_VETOED: dict[str, tuple[str, str]] = {
    "cpd05178": ("3mb_e", "Pf metabolite has name/formula '#N/A'; identity with 3-methylbutanoate (C5H9O2) cannot be formula-checked"),
    "cpd11594": ("dextrin_e", "Pf dextrin is C36H62O31, OR16/NS21 dextrin is C12H20O10; merging would break carbon accounting"),
}
AMBIGUOUS_SHARED_POOLS: dict[str, list[str]] = {"cpd00222": ["glcn__D_e", "glcn_e"]}

# The 23 boundaries converted by screen_propionibacterium_helper.py (source id -> pool).
ALREADY_CONVERTED: dict[str, str] = {
    "cpd00001": "h2o_e", "cpd00009": "pi_e", "cpd00013": "nh4_e", "cpd00027": "glc__D_e",
    "cpd00030": "mn2_e", "cpd00034": "zn2_e", "cpd00048": "so4_e", "cpd00058": "cu2_e",
    "cpd00067": "h_e", "cpd00099": "cl_e", "cpd00104": "btn_e", "cpd00141": "ppa_e",
    "cpd00149": "cobalt2_e", "cpd00159": "lac__L_e", "cpd00205": "k_e", "cpd00254": "mg2_e",
    "cpd00644": "pnto__R_e", "cpd00971": "na1_e", "cpd03424": "b12_e", "cpd10515": "fe2_e",
    "cpd10516": "fe3_e", "cpd00029": "ac_e", "cpd00007": "o2_pfreud_e",
}

# Source-model sinks (lower bound -1000).  Heavy metals are closed for uptake
# when close_toxic_sinks=True; H2S keeps its original bounds (see report).
TOXIC_METAL_SEEDS = {"cpd00531": "Hg2+", "cpd01012": "Cd2+", "cpd04097": "Pb"}
H2S_SEED = "cpd00239"

PSEUDO_REACTION_NOTES = {
    "rxnnew73_c0": "pseudo macromolecule synthesis (protein); element/charge imbalance inherited unchanged from source P_sherm_model.xml: product cpd11463 uses the ModelSEED repeat-unit formula C4H5N2O3R2, and ATP hydrolysis water is not written; stoichiometry not changed (no documented correction in repo or source)",
    "rxnnew74_c0": "pseudo macromolecule synthesis (RNA); element/charge imbalance inherited unchanged from source P_sherm_model.xml: product cpd11613 uses the ModelSEED repeat-unit formula C15H23O19P3R3; stoichiometry not changed (no documented correction in repo or source)",
    "rxnnew75_c0": "pseudo macromolecule synthesis (DNA); element/charge imbalance inherited unchanged from source P_sherm_model.xml: product cpd11461 uses the ModelSEED repeat-unit formula C15H23O13P2R3; stoichiometry not changed (no documented correction in repo or source)",
    "rxnnew78_c0": "hypothetical anaerobic BluB (fumarate as acceptor); imbalance inherited unchanged from source and equals exactly the missing formula of cpdnew28 (Dialurate, no formula in model): C4H4N2O4, charge -2; no R pseudo-element involved; stoichiometry not changed",
}

_EX_S = re.compile(r"^Ex_S_(?P<key>.+)_ext$")
_NOTE_KEY = "curation_note"


def pf_specific_pool(key: str) -> str:
    """Deterministic Pf-only pool id for a compound with no shared equivalent."""
    return f"{key}_e"


def _counts(model: cobra.Model) -> dict[str, int]:
    return {
        "reactions": len(model.reactions),
        "metabolites": len(model.metabolites),
        "exchanges": len(model.exchanges),
        "demands": len(model.demands),
        "sinks": len(model.sinks),
        "boundary": len(model.boundary),
        "legacy_Ex_S_boundaries": sum(1 for r in model.boundary if _EX_S.match(r.id)),
    }


def _target_pool(key: str) -> tuple[str, str]:
    if key in SEED_TO_SHARED_POOL:
        return SEED_TO_SHARED_POOL[key], "shared"
    return pf_specific_pool(key), "pf_specific"


def curate_pf_exchanges(
    model: cobra.Model,
    *,
    close_toxic_sinks: bool = True,
    verbose: bool = False,
    inplace: bool = False,
) -> tuple[cobra.Model, dict[str, Any]]:
    """Convert Pf's remaining ``Ex_S_*_ext`` boundaries into pool exchanges.

    Returns ``(curated_model, report)``.  By default the input is not modified
    (a ``model.copy()`` is curated); pass ``inplace=True`` to edit it.
    Applying the function to its own output changes nothing (idempotent).

    * ``S_<id>_ext`` (compartment ``c``) -> ``<pool>`` in compartment ``e``;
      ``Ex_S_<id>_ext`` -> ``EX_<pool>``.  Bounds are copied unchanged, so a
      secretion-only boundary stays secretion-only.  The existing source
      transport reaction (c0 <-> ext) is what carries the flux; none is added.
    * ``<pool>`` is the shared BiGG id when :data:`SEED_TO_SHARED_POOL` maps
      it, otherwise ``<ModelSEED id>_e``.  If the target metabolite or
      ``EX_<pool>`` already exists the Pf-specific id is used (reported under
      ``duplicates``); if that too exists the boundary is skipped.
    * ``Ex_S_biomass_ext`` (biomass drain, not a chemical) is left untouched.
    * ``close_toxic_sinks``: Hg2+/Cd2+/Pb uptake bound set to 0.  H2S becomes
      ``EX_h2s_e`` with its original bounds; uptake is then governed by the
      medium (``h2s_e`` concentration), like every other exchange.
    * rxnnew73/74/75/78 receive ``annotation['curation_note']``; their
      stoichiometry is not changed.  No reaction, gene or metabolite is added.
    """
    if not inplace:
        model = model.copy()
    report: dict[str, Any] = {
        "mode": "inplace" if inplace else "copy",
        "converted": [], "already_converted": [], "skipped": [], "duplicates": [],
        "vetoed_shared_mappings": [], "ambiguous_shared_pools": [],
        "original_uptake_allowed": [], "closed_sinks": [], "annotations": {},
        "added_reactions": [], "counts_before": _counts(model),
    }
    for key, pool in sorted(ALREADY_CONVERTED.items()):
        if f"EX_{pool}" in model.reactions:
            report["already_converted"].append({"source": f"Ex_S_{key}_ext", "id": f"EX_{pool}", "pool": pool,
                                                "bounds": list(model.reactions.get_by_id(f"EX_{pool}").bounds)})
    if "e" not in model.compartments:
        model.compartments = {"e": "extracellular"}

    for rxn in sorted((r for r in model.reactions if _EX_S.match(r.id)), key=lambda r: r.id):
        key = _EX_S.match(rxn.id).group("key")
        stoich = {m.id: c for m, c in rxn.metabolites.items()}
        if key == "biomass":
            report["skipped"].append({"id": rxn.id, "reason": "biomass drain (S_biomass_ext produced by biomass_c0), not a chemical pool"})
            continue
        if stoich != {f"S_{key}_ext": -1}:
            report["skipped"].append({"id": rxn.id, "reason": f"unexpected stoichiometry {stoich}"})
            continue
        met = model.metabolites.get_by_id(f"S_{key}_ext")
        pool, kind = _target_pool(key)
        if key in SHARED_POOL_VETOED:
            report["vetoed_shared_mappings"].append({"source": met.id, "candidate": SHARED_POOL_VETOED[key][0],
                                                     "used": pool, "reason": SHARED_POOL_VETOED[key][1]})
        if key in AMBIGUOUS_SHARED_POOLS:
            report["ambiguous_shared_pools"].append({"source": met.id, "candidates": AMBIGUOUS_SHARED_POOLS[key], "used": pool})
        if pool in model.metabolites or f"EX_{pool}" in model.reactions:
            fallback = pf_specific_pool(key)
            report["duplicates"].append({"source": rxn.id, "wanted": pool, "used": fallback,
                                         "reason": f"{pool} / EX_{pool} already present in model"})
            pool, kind = fallback, "pf_specific_duplicate"
            if pool in model.metabolites or f"EX_{pool}" in model.reactions:
                report["skipped"].append({"id": rxn.id, "reason": f"fallback {pool} also present"})
                continue
        partners = sorted(r.id for r in met.reactions if r is not rxn)
        bounds = tuple(rxn.bounds)
        old_rxn, old_met = rxn.id, met.id
        met.id = pool
        met.compartment = "e"
        if re.fullmatch(r"cpd\d+", key):
            met.annotation["seed.compound"] = key
        rxn.id = f"EX_{pool}"
        rxn.name = f"{met.name} exchange"
        note = f"pf_curation: converted from {old_rxn} on {old_met}; bounds unchanged {list(bounds)}"
        entry = {"old_id": old_rxn, "new_id": rxn.id, "old_metabolite": old_met, "pool": pool,
                 "pool_kind": kind, "name": met.name, "formula": met.formula,
                 "bounds_before": list(bounds), "transport_partners": partners}
        if bounds[0] < 0:
            report["original_uptake_allowed"].append({"id": rxn.id, "old_id": old_rxn, "bounds": list(bounds)})
        if not partners:
            entry["warning"] = "no transport/partner reaction: exchange is isolated"
        rxn.annotation[_NOTE_KEY] = note
        entry["bounds_after"] = list(rxn.bounds)
        report["converted"].append(entry)

    if close_toxic_sinks:
        for key, label in sorted(TOXIC_METAL_SEEDS.items()):
            candidates = [f"EX_{p}" for p in {_target_pool(key)[0], pf_specific_pool(key)}] + [f"Ex_S_{key}_ext"]
            for rid in candidates:
                if rid in model.reactions and model.reactions.get_by_id(rid).lower_bound < 0:
                    r = model.reactions.get_by_id(rid)
                    before = list(r.bounds)
                    r.lower_bound = 0.0
                    r.annotation[_NOTE_KEY] = (r.annotation.get(_NOTE_KEY, "pf_curation") +
                                               f"; toxic metal ({label}) uptake closed, original bounds {before}")
                    report["closed_sinks"].append({"id": rid, "metal": label, "bounds_before": before,
                                                   "bounds_after": list(r.bounds)})
                    for e in report["converted"]:
                        if e["new_id"] == rid:
                            e["bounds_after"] = list(r.bounds)
    for rid, note in PSEUDO_REACTION_NOTES.items():
        if rid in model.reactions:
            r = model.reactions.get_by_id(rid)
            r.annotation[_NOTE_KEY] = note
            report["annotations"][rid] = {"note": note, "imbalance": {k: round(v, 6) for k, v in r.check_mass_balance().items()}}
    report["counts_after"] = _counts(model)
    report["summary"] = {
        "converted": len(report["converted"]),
        "converted_shared": sum(e["pool_kind"] == "shared" for e in report["converted"]),
        "converted_pf_specific": sum(e["pool_kind"] != "shared" for e in report["converted"]),
        "already_converted": len(report["already_converted"]),
        "skipped": len(report["skipped"]), "duplicates": len(report["duplicates"]),
        "closed_sinks": len(report["closed_sinks"]),
    }
    if verbose:
        b, a = report["counts_before"], report["counts_after"]
        print(f"pf_curation: converted {len(report['converted'])}, skipped {len(report['skipped'])}, "
              f"duplicates {len(report['duplicates'])}, closed sinks {len(report['closed_sinks'])}; "
              f"exchanges {b['exchanges']}->{a['exchanges']}, demands {b['demands']}->{a['demands']}, "
              f"sinks {b['sinks']}->{a['sinks']}")
    return model, report
