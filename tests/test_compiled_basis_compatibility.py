import hashlib
from src.compiled_basis_compatibility import equivalent_offline_compiler


def test_only_online_evaluator_changes_are_compatible():
    previous=b'import numpy as np\ndef compile_basis(): return 1\nclass GpuBasisEvaluator: pass\n'
    digest=hashlib.sha256(previous).hexdigest()
    assert equivalent_offline_compiler(previous,previous.replace(b'pass',b'value=2'),digest)
    assert not equivalent_offline_compiler(previous,previous.replace(b'return 1',b'return 2'),digest)
    assert not equivalent_offline_compiler(previous,previous,digest+'0')
