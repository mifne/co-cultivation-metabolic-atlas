import numpy as np
import pytest
from src.gpu_hybrid_lp import route_rejected, observable_projection


def test_certificate_precedes_dispersion_and_low_spread_does_not_accept():
    direct, repair, unknown = route_rejected([True, False, False, False],
        [100., 100., .01, 0.], [4, 4, 4, 1], 1.)
    assert direct.tolist() == [False, True, False, True]
    assert repair.tolist() == [False, False, True, False]
    assert unknown.tolist() == [False, False, False, True]


def test_nonfinite_dispersion_is_unknown_not_confident():
    direct, repair, unknown = route_rejected([False]*3, [np.nan, np.inf, 0.], [3]*3, 1.)
    assert direct.tolist() == [True, True, False]
    assert repair.tolist() == [False, False, True]
    assert unknown.tolist() == [True, True, False]
    with pytest.raises(ValueError): route_rejected([False], [0.], [3], -1.)


def test_observables_exclude_internal_cycles_and_auxiliaries():
    metadata = dict(growth_terms={'a':(0, 2.)}, exchange_terms={'m':[('a', 1, -1., 'm_e')]},
        reaction_ids=['biomass', 'exchange', 'PHB_syn', 'internal'], reaction_species=['a']*4)
    matrix, scales, names = observable_projection(metadata, 6)
    np.testing.assert_array_equal(matrix@np.array([1., 2., 3., 1e6, 1e6, 1e6]), [2., -2., 3.])
    assert len(names) == 3 and (scales > 0).all()
