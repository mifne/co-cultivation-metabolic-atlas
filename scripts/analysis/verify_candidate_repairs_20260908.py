"""Load the staged core solely into this test process; live workers stay frozen."""
from pathlib import Path
import importlib.util,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src
spec=importlib.util.spec_from_file_location('src.audited_dfba',ROOT/'results/cultivation_model_audit_20260908/audited_dfba_candidate.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
spec.loader.exec_module(module);src.audited_dfba=module
import pytest
raise SystemExit(pytest.main(['tests/test_audited_dfba.py','tests/test_audited_production_repairs.py','-q']))
