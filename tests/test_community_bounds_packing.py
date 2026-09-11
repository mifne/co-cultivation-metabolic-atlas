"""Exact scalar-rule equivalence for live community bound packing."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.community_solver import CooperativeCommunityFbaSolver


def legacy(solver, models, max_crossfeed_uptake):
    bounds = []
    lookup = {(species,index):reaction_id for terms in solver._exchange_terms.values()
              for species,index,_,reaction_id in terms}
    for species,(start,end) in solver._offsets.items():
        originals = solver._original_exchange_bounds.get(species,{})
        for local_index,reaction in enumerate(models[species].reactions):
            lower,upper = map(float,reaction.bounds)
            reaction_id = lookup.get((species,start+local_index))
            if reaction_id is not None:
                original_lower = float(originals.get(reaction_id,(lower,upper))[0])
                if original_lower < 0:
                    lower = min(lower,max(original_lower,-abs(max_crossfeed_uptake)))
            bounds.append((lower if np.isfinite(lower) else None,
                           upper if np.isfinite(upper) else None))
    return bounds


@pytest.mark.parametrize('maximum',[0.,.1,4.,-4.,np.inf,np.nan])
def test_bounds_packing_preserves_live_crossfeed_and_nonfinite_rules(maximum):
    solver = CooperativeCommunityFbaSolver.__new__(CooperativeCommunityFbaSolver)
    solver._offsets = {'a':(0,5),'b':(5,9),'empty':(9,9)}
    solver._exchange_terms = {'first':[('a',0,-1.,'x'),('a',2,1.,'y'),('b',6,-1.,'z')],
                              'alias':[('a',0,-1.,'overwrites_x'),('b',99,1.,'ignored')]}
    solver._original_exchange_bounds = {'a':{'overwrites_x':(-8,1000),'y':(0,0)},
                                        'b':{'z':(-3,1000)}}
    models = {'a':SimpleNamespace(reactions=[SimpleNamespace(bounds=b) for b in
                    [(-.1,1000),(-np.inf,np.inf),(0,0),(np.nan,2),(-0.,0.)]]),
              'b':SimpleNamespace(reactions=[SimpleNamespace(bounds=b) for b in
                    [(-2,7),(-.2,4),(1,np.nan),(3,np.inf)]]),
              'empty':SimpleNamespace(reactions=[])}
    before = [r.bounds for m in models.values() for r in m.reactions]
    actual = solver._bounds(models,maximum)
    assert actual == legacy(solver,models,maximum)
    assert all(isinstance(row,tuple) and all(v is None or type(v) is float for v in row) for row in actual)
    assert [r.bounds for m in models.values() for r in m.reactions] == before
    models['a'].reactions[0].bounds = (-9,3)
    solver._original_exchange_bounds['b']['z'] = (-1,10)
    assert solver._bounds(models,maximum) == legacy(solver,models,maximum)
