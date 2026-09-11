import numpy as np
import pytest
from src.compiled_basis_artifact import save_anchor,load_anchor
from src.gpu_certified_basis import compile_basis
from tests.test_gpu_revised_basis import problem


def test_nonpickle_artifact_roundtrip_and_overwrite_guard(tmp_path):
    p=problem()
    original=compile_basis(p.a,p.rhs,p.lower,p.upper,p.c,p.neq)
    path=tmp_path/"basis.npz"
    save_anchor(path,original)
    restored=load_anchor(path)
    for key in ("basic","active","kind","row_kind","cpu_anchor_values"):
        np.testing.assert_array_equal(original[key],restored[key])
    np.testing.assert_array_equal(original["inverse"].toarray(),restored["inverse"].toarray())
    np.testing.assert_array_equal(original["lp"].a.toarray(),restored["lp"].a.toarray())
    with np.load(path,allow_pickle=False) as data:
        assert all(data[k].dtype.kind!="O" for k in data.files)
    with pytest.raises(FileExistsError):save_anchor(path,original)


def test_missing_fields_fail_closed(tmp_path):
    path=tmp_path/"invalid.npz"
    np.savez(path,rhs=np.ones(1))
    with pytest.raises(KeyError):load_anchor(path)
