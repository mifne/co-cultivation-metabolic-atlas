import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from src.graph_temporal_lp import LPGraphBatch
from src.supervised_graph_lp import SupervisedGraphLP,supervised_loss,checkpoint,load_checkpoint


def graph(identity='unit'):
    p=(csr_matrix([[1.,-1.,0.],[0.,1.,1.]]),np.array([0.,2.]),np.zeros(3),
        np.array([2.,2.,1.]),np.array([0.,0.,-1.]),1)
    return LPGraphBatch.from_problems([p],stage='maxmin',model_identity=identity,device='cpu')


def model(g,temporal=True):
    return SupervisedGraphLP(g.identity,[.2,.2,1.],[1.,1.,1.],[0.,0.],[1.,1.],hidden=4,temporal=temporal)


def test_full_space_gradient_includes_graph_and_gru():
    torch.manual_seed(1);g=graph();m=model(g)
    one=m(g);two=m(g,one.state)
    loss,_=supervised_loss(m,g,two,torch.tensor([[.4,.4,1.]]),torch.zeros((1,2)),physics_mode='original_worst')
    loss.backward()
    for name in ('variable_embedding','row_embedding'):
        assert getattr(m,name).grad.isfinite().all()
    assert m.to_row[0].weight.grad.abs().sum()>0
    assert m.variable_gru.weight_hh.grad.abs().sum()>0
    assert two.x.shape==(1,3) and two.y.shape==(1,2)
    assert torch.all(two.x>=g.lower) and torch.all(two.x<=g.upper)


def test_checkpoint_owned_and_round_trip(tmp_path):
    torch.manual_seed(2);g=graph();m=model(g).eval()
    p=tmp_path/'model.pt';torch.save(checkpoint(m,dict(stage='maxmin')),p)
    restored,meta=load_checkpoint(p,device='cpu')
    torch.testing.assert_close(restored(g).x,m(g).x)
    assert meta==dict(stage='maxmin') and not restored.training
    with pytest.raises(ValueError,match='identity'):restored(graph('different-GEM'))


def test_normalization_is_owned_and_positive():
    g=graph();center=torch.zeros(3)
    m=SupervisedGraphLP(g.identity,center,torch.ones(3),torch.zeros(2),torch.ones(2),hidden=4)
    center[0]=99
    assert m.center_x[0]==0
    with pytest.raises(ValueError):SupervisedGraphLP(g.identity,center,[0.,1.,1.],[0.,0.],[1.,1.])


def test_gnn_ablation_has_no_temporal_parameters():
    g=graph();m=model(g,False)
    assert not hasattr(m,'variable_gru')
    assert m(g).x.isfinite().all()


def test_training_physics_is_not_a_certificate():
    g=graph();m=model(g);proposal=m(g)
    for mode in ('scaled_mean','original_worst'):
        loss,parts=supervised_loss(m,g,proposal,proposal.x.detach(),proposal.y.detach(),physics_mode=mode)
        assert loss.isfinite() and 'accepted' not in parts
    with pytest.raises(ValueError):supervised_loss(m,g,proposal,proposal.x,proposal.y,physics_mode='relax')
