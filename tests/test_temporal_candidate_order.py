import numpy as np
import pytest
from src.temporal_candidate_order import temporal_candidate_order,temporal_candidate_order_device


def test_within_budget_cannot_evict_current_query_candidate():
    order = np.array([[2,0,1], [1,2,0]])
    old = order.copy()
    assert temporal_candidate_order(order, [1,2], 1, 'prepend').tolist() == [[1],[2]]
    assert temporal_candidate_order(order, [1,2], 1, 'within-budget').tolist() == [[2],[1]]
    assert temporal_candidate_order(order, [1,2], 2, 'within-budget').tolist() == [[2,0],[2,1]]
    np.testing.assert_array_equal(order, old)


@pytest.mark.parametrize('limit',[0,1,2,3,8])
def test_candidate_set_preserved_and_budget_never_expanded(limit):
    order = np.array([[2,0,1], [1,2,0]])
    expected = order[:, :limit or 3]
    for previous in ([0,0],[-1,-1],[9,9]):
        got = temporal_candidate_order(order, previous, limit, 'within-budget')
        np.testing.assert_array_equal(np.sort(got,axis=1),np.sort(expected,axis=1))
        np.testing.assert_array_equal(temporal_candidate_order(order, previous, limit, 'off'),expected)


@pytest.mark.parametrize('policy,limit,previous',[('bad',1,[0]),('off',-1,[0]),('off',True,[0]),
    ('off',1,[0.,]),('off',1,[0,1])])
def test_invalid_policy_input_fails(policy,limit,previous):
    with pytest.raises(ValueError):temporal_candidate_order(np.array([[0,1]]),previous,limit,policy)


@pytest.mark.parametrize('policy',['prepend','within-budget','off'])
@pytest.mark.parametrize('limit',[0,1,2,8])
def test_device_algorithm_same_topk_set_and_order(policy,limit):
    order=np.array([[2,0,1],[1,2,0],[0,2,1],[2,1,0]])
    prior=np.array([1,2,-1,99])
    np.testing.assert_array_equal(temporal_candidate_order_device(np,order,prior,limit,policy),
        temporal_candidate_order(order,prior,limit,policy))
