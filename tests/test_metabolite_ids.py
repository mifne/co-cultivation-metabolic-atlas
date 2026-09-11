from src.metabolite_ids import canonical_metabolite_id


def test_canonical_metabolite_id_matches_wcfs1_and_bigg_encodings():
    assert canonical_metabolite_id("M_glc-D_e") == "glc__D_e"
    assert canonical_metabolite_id("glc__D_LSQBKTe_RSQBKT") == "glc__D_e"
    assert canonical_metabolite_id("cys-L_e") == "cys__L_e"
    assert canonical_metabolite_id("btd-RR_e") == "btd__RR_e"


def test_canonical_metabolite_id_does_not_erase_stereochemistry():
    assert canonical_metabolite_id("lac-L_e") != canonical_metabolite_id("lac-D_e")
    assert canonical_metabolite_id("M_co2_e") == "co2_e"
