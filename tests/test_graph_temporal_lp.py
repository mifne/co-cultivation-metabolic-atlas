import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.graph_temporal_lp import (LPGraphBatch, GraphTemporalLP, GraphTemporalSession,
                                   differentiable_lp_residuals)


def problems():
    a = csr_matrix([[1., -2., 0.], [0., 1., 3.]])
    return [(a, np.array([1., 4.]), np.array([0., -np.inf, 2.]),
             np.array([10., np.inf, 2.]), np.array([1., 0., -1.]), 1)]


def graph(ps=None, **kwargs):
    return LPGraphBatch.from_problems(ps or problems(), stage='exchange', model_identity='test-v1',
                                      device='cpu', **kwargs)


def test_input_only_snapshot_keeps_all_rows_and_box_masks():
    ps = problems()
    g = graph(ps)
    ps[0][0].data[:] = 99
    assert g.coefficient[0].tolist() == [1., -2., 1., 3.]
    v, r, _ = g.features(torch.float32)
    assert v.shape == (1, 3, 6) and r.shape == (1, 2, 2)
    assert v[0, 1, 3:5].tolist() == [0., 0.]
    p = GraphTemporalLP(hidden=8)(g)
    assert p.x.shape == (1, 3) and p.y.shape == (1, 2)
    assert 0 <= p.x[0, 0] <= 10 and p.x[0, 2] == 2 and p.y[0, 1] <= 0
    assert p.x.isfinite().all() and p.y.isfinite().all()


def test_backpropagation_includes_graph_and_temporal_parameters():
    torch.manual_seed(3)
    model = GraphTemporalLP(hidden=8)
    first = model(graph())
    second = model(graph(), first.state)
    (second.x.square().sum()+second.y.square().sum()).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    assert model.to_row[0].weight.grad.abs().sum() > 0
    assert model.variable_gru.weight_hh.grad.abs().sum() > 0


def test_variable_permutation_equivariance_and_independent_environments():
    torch.manual_seed(7)
    model = GraphTemporalLP(hidden=8).double().eval()
    p = problems()[0]
    order = np.array([2, 0, 1])
    permuted = (p[0][:, order], p[1], p[2][order], p[3][order], p[4][order], p[-1])
    original, changed = model(graph()), model(graph([permuted]))
    torch.testing.assert_close(changed.x, original.x[:, order], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(changed.y, original.y, rtol=1e-12, atol=1e-12)
    batch = model(graph([p, permuted]))
    torch.testing.assert_close(batch.x[0], original.x[0])
    torch.testing.assert_close(batch.x[1], changed.x[0])


def test_causal_history_is_transactional_reorderable_and_resettable():
    torch.manual_seed(9)
    session = GraphTemporalSession(GraphTemporalLP(hidden=8).eval())
    g = graph(problems()*2)
    first, token = session.propose(g, ['a', 'b'])
    again, _ = session.propose(g, ['a', 'b'])
    torch.testing.assert_close(first.x, again.x)
    with pytest.raises(ValueError):session.commit(token, again, [True, True])
    session.commit(token, first, [True, False])
    with pytest.raises(ValueError):
        session.commit(token, first, [True, False])
    following, token2 = session.propose(g, ['b', 'a'])
    torch.testing.assert_close(following.x[0], first.x[1])
    assert not torch.equal(following.x[1], first.x[0])
    session.reset(['a'])
    with pytest.raises(ValueError):
        session.commit(token2, following, [True, True])
    reset, _ = session.propose(g, ['a', 'b'])
    torch.testing.assert_close(reset.x, first.x)


def test_different_stage_or_model_cannot_reuse_history():
    session = GraphTemporalSession(GraphTemporalLP(hidden=4).eval())
    proposal, token = session.propose(graph(), [0])
    session.commit(token, proposal, [True])
    changed = LPGraphBatch.from_problems(problems(), stage='maxmin', model_identity='test-v1', device='cpu')
    with pytest.raises(ValueError):session.propose(changed, [0])
    with pytest.raises(ValueError):session.propose(graph(problems()*2), [0, 0])


def test_gnn_only_is_explicit_and_rejects_temporal_state():
    model = GraphTemporalLP(hidden=4, temporal=False)
    proposal = model(graph())
    with pytest.raises(ValueError):model(graph(), proposal.state)
    with pytest.raises(ValueError):GraphTemporalSession(model)


def test_nonfinite_problem_and_impossible_infinite_bounds_are_rejected():
    p = problems()[0]
    with pytest.raises(ValueError):graph([(p[0], p[1]*np.nan, *p[2:])])
    with pytest.raises(ValueError):graph([(*p[:2], np.full(3, np.inf), np.full(3, np.inf), *p[4:])])


def test_dynamic_coefficients_and_rhs_change_input_without_resetting_same_pattern():
    p = problems()[0]
    changed = (p[0]*2, p[1]+3, *p[2:])
    first, second = graph(), graph([changed])
    assert first.identity == second.identity
    assert not torch.equal(first.coefficient, second.coefficient)
    assert not torch.equal(first.features(torch.float32)[1], second.features(torch.float32)[1])
    changed_structure = (csr_matrix(np.ones((2, 3))), *p[1:])
    assert graph([changed_structure]).identity != first.identity


def test_physics_losses_match_independent_full_lp_residuals_and_have_gradients():
    from scripts.probe_downstream_gpu_coverage import paired_certificate
    g=graph()
    x=torch.tensor([[1.,-2.,3.]],requires_grad=True,dtype=torch.float64)
    y=torch.tensor([[.2,.4]],requires_grad=True,dtype=torch.float64)
    losses=differentiable_lp_residuals(g,x,y)
    expected=paired_certificate(problems()[0],x.detach().numpy()[0],y.detach().numpy()[0])
    for key,value in losses.items():
        assert value.item()==pytest.approx(expected[key],abs=1e-12)
    sum(v.sum() for v in losses.values()).backward()
    assert x.grad.isfinite().all() and y.grad.isfinite().all()
