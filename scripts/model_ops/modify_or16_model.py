"""Backward-compatible entry point for centralized polymer curation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra

from scripts.model_ops.curate_polymer_pathway import curate_or16, validate_curated_model


def modify_or16_model(input_path, output_path):
    model = curate_or16(cobra.io.read_sbml_model(input_path))
    validate_curated_model(model, ("R_LCP", "R_C30t", "R_C30_cat"))
    cobra.io.write_sbml_model(model, output_path)


if __name__ == "__main__":
    target = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    modify_or16_model(target, target)
