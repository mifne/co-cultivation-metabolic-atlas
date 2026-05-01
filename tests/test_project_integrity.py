import os
import pytest
from pathlib import Path

# Core files that MUST exist for the project to be functional
CORE_FILES = [
    "main.py",
    "src/dfba_simulator.py",
    "src/ppo_agent.py",
    "src/rl_environment.py",
    "src/callbacks.py",
    "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml",
    "models/sbml/final_consortium/Lactobacillus_plantarum.xml",
    "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml",
    "docs/MASTER_INDEX.md",
    "docs/MODEL_INDEX.md",
    "docs/SYSTEM_ARCHITECTURE.md",
    "docs/PROJECT_MANAGEMENT.md",
]

@pytest.mark.parametrize("file_path", CORE_FILES)
def test_core_files_exist(file_path):
    """
    Ensure that all core files defined in CORE_FILES list exist in the workspace.
    """
    path = Path(file_path)
    assert path.exists(), f"Core file missing: {file_path}"

def test_src_directory_structure():
    """
    Ensure the src directory is intact.
    """
    src_path = Path("src")
    assert src_path.is_dir()
    # Check for __init__.py if it should exist (optional, but good practice)
    # assert (src_path / "__init__.py").exists()

def test_models_directory_structure():
    """
    Ensure the final_consortium models directory is intact.
    """
    models_path = Path("models/sbml/final_consortium")
    assert models_path.is_dir()
    xml_files = list(models_path.glob("*.xml"))
    assert len(xml_files) >= 3, "Expected at least 3 consortium models"
