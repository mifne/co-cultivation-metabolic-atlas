"""Integrity fingerprints must retain their historical byte representation."""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.lp_zero_face import ZeroFaceReduction, ZeroFaceWitness


def legacy(plan):
    digest = hashlib.sha256()
    for name in ('columns','rows','fixed_columns','fixed_values','explicit_fixed_columns',
                 'forced_zero_columns','skipped_small_singleton_rows',
                 'removed_duplicate_rows','removed_zero_rows'):
        value=np.asarray(getattr(plan,name),dtype='<f8' if name=='fixed_values' else '<i8')
        digest.update(name.encode('ascii'))
        digest.update(np.asarray(value.shape,dtype='<i8').tobytes())
        digest.update(value.tobytes())
    digest.update(json.dumps(dict(witnesses=[asdict(w) for w in plan.witnesses],
        duplicates=[asdict(d) for d in plan.duplicate_equalities],
        singleton=plan.fix_singleton_equalities,floor=plan.singleton_min_coefficient,
        deduplicate=plan.remove_duplicate_equalities),sort_keys=True,
        separators=(',',':'),allow_nan=False).encode())
    return digest.hexdigest()


@pytest.mark.parametrize('singletons',[False,True])
@pytest.mark.parametrize('duplicates',[False,True])
def test_exact_legacy_fingerprint_and_mutation_detection(singletons,duplicates):
    p=(csr_matrix([[1.,1.,0.,0.],[0.,-1.,2.,0.],[0.,0.,0.,1.],
                   [0.,0.,0.,1.],[0.,0.,0.,2.]]),
       np.array([0.,0.,1.,1.,3.]),np.zeros(4),np.ones(4),np.arange(4.),4)
    plan=ZeroFaceReduction(p,fix_singleton_equalities=singletons,
                           remove_duplicate_equalities=duplicates)
    assert plan.witnesses and (plan.duplicate_equalities or not duplicates)
    assert plan._map_fingerprint()==legacy(plan)==plan._proof_fingerprint
    old=plan.witnesses[0]
    plan.witnesses=(ZeroFaceWitness(old.row,old.orientation,
        old.newly_fixed_columns,tuple(v+1. for v in old.newly_fixed_coefficients)),*plan.witnesses[1:])
    assert plan._map_fingerprint()==legacy(plan)
    with pytest.raises(ValueError,match='proof'): plan.validate_integrity()


def test_subclass_extra_field_is_not_silently_ignored():
    @dataclass(frozen=True)
    class ExtraWitness(ZeroFaceWitness):
        extra: int = 7
    p=(csr_matrix([[1.,1.]]),np.zeros(1),np.zeros(2),np.ones(2),np.ones(2),1)
    plan=ZeroFaceReduction(p)
    w=plan.witnesses[0]
    plan.witnesses=(ExtraWitness(w.row,w.orientation,w.newly_fixed_columns,w.newly_fixed_coefficients),)
    assert plan._map_fingerprint()==legacy(plan)
    with pytest.raises(ValueError,match='proof'): plan.validate_integrity()
