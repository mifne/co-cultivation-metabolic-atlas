"""Backward-compatible entry point for centralized polymer curation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra

from scripts.model_ops.curate_polymer_pathway import curate_ns21, validate_curated_model


def modify_ns21_model(input_path, output_path):
    model = curate_ns21(cobra.io.read_sbml_model(input_path))
    validate_curated_model(
        model,
        ("R_ROXB", "R_ROXA", "R_ROXA_BULK", "R_ODTDt", "R_ODTD_cat", "R_C30_cat"),
    )
    cobra.io.write_sbml_model(model, output_path)


if __name__ == "__main__":
    target = "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
    modify_ns21_model(target, target)
