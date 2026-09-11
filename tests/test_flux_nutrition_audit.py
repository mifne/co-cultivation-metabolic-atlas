from pathlib import Path

from scripts.analysis.audit_flux_and_nutrition import (
    forced_exchange_rows,
    polymer_reaction_rows,
    zero_feasible_exchange_copies,
)
from src.utils import load_sbml_models, select_consortium_models


MODEL_DIR = Path("models/sbml/final_consortium")


def _models():
    return select_consortium_models(load_sbml_models(MODEL_DIR))


def test_production_models_have_no_compulsory_exchange_flux():
    raw = _models()
    forced_before = forced_exchange_rows(raw)
    assert forced_before == []
    normalized = zero_feasible_exchange_copies(raw)
    assert forced_exchange_rows(raw) == forced_before
    assert forced_exchange_rows(normalized) == []


def test_polymer_audit_confirms_curated_pathway():
    rows = polymer_reaction_rows(_models())
    by_key = {(row["species"], row["reaction"]): row for row in rows}
    or16 = next(name for name in _models() if "OR16" in name)
    ns21 = next(name for name in _models() if "NS21" in name)
    assert by_key[(or16, "R_LCP")]["mass_balance_status"] == "balanced"
    assert by_key[(or16, "R_LCP")]["gpr_status"] == "match"
    assert by_key[(or16, "R_C30_cat")]["mass_balance_status"] == "balanced"
    assert by_key[(ns21, "R_ROXA")]["mass_balance_status"] == "balanced"
    assert by_key[(ns21, "R_ROXA")]["gpr_status"] == "match"
    assert by_key[(ns21, "R_ROXB")]["gpr_status"] == "match"
    assert by_key[(ns21, "R_ROXA_BULK")]["gpr_status"] == "match"
    assert by_key[(ns21, "R_ODTD_cat")]["mass_balance_status"] == "balanced"
