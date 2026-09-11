"""Train-only scaled, identity-bound full-space GNN/GRU proposal model.

No low-rank decoder and no claim that network output solves the LP. Per-node
embeddings apply only to the declared immutable graph coordinates; this model
does not claim permutation/general-GEM generalization without remapping them.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .graph_temporal_lp import GraphTemporalLP


class SupervisedGraphLP(GraphTemporalLP):
    def __init__(self,identity,center_x,scale_x,center_y,scale_y,*,hidden=16,rounds=2,temporal=True):
        super().__init__(hidden=hidden,rounds=rounds,temporal=temporal)
        if not isinstance(identity,str) or not identity:raise ValueError('Immutable graph identity required')
        self.identity=identity
        for key,value in dict(center_x=center_x,scale_x=scale_x,center_y=center_y,scale_y=scale_y).items():
            v=torch.as_tensor(value,dtype=torch.float64).detach().clone()
            if v.ndim!=1 or not bool(v.isfinite().all()) or ('scale' in key and not bool((v>0).all())):
                raise ValueError('Finite 1D training statistics and positive scales required')
            self.register_buffer(key,v)
        if self.center_x.shape!=self.scale_x.shape or self.center_y.shape!=self.scale_y.shape:
            raise ValueError('Training-statistic dimensions disagree')
        self.variable_embedding=nn.Parameter(torch.zeros(len(self.center_x),hidden))
        self.row_embedding=nn.Parameter(torch.zeros(len(self.center_y),hidden))
        nn.init.normal_(self.variable_embedding,std=.02);nn.init.normal_(self.row_embedding,std=.02)
        # Begin close to the TRAIN-only mean, not the midpoint of +/-1000.
        nn.init.normal_(self.primal_head.weight,std=.001);nn.init.zeros_(self.primal_head.bias)
        nn.init.normal_(self.dual_head.weight,std=.001);nn.init.zeros_(self.dual_head.bias)

    def encode_nodes(self,graph,variables,rows):
        if graph.identity!=self.identity:raise ValueError('Checkpoint graph/model/stage identity mismatch')
        if variables.shape[1]!=len(self.center_x) or rows.shape[1]!=len(self.center_y):
            raise ValueError('Checkpoint node coordinates differ')
        return (F.silu(self.variable_encoder(variables)+self.variable_embedding),
                F.silu(self.row_encoder(rows)+self.row_embedding))

    def decode_heads(self,graph,raw_x,raw_y):
        x=self.center_x+self.scale_x*raw_x
        x=torch.maximum(graph.lower,torch.minimum(graph.upper,x))
        y=self.center_y+self.scale_y*raw_y
        eq=torch.arange(graph.rhs.shape[1],device=y.device)<graph.neq
        return x,torch.where(eq,y,y.clamp_max(0.))


def supervised_loss(model,graph,proposal,target_x,target_y,*,physics_mode='scaled_mean'):
    """Full-vector teacher loss plus current mass balance/inequality physics.

Row normalization is a training conditioning choice, never a runtime gate.
All normalizers depend on training scales and CURRENT input coefficients.
"""
    x,y=proposal.x,proposal.y
    lx=((x-target_x)/model.scale_x).square().mean()
    ly=((y-target_y)/model.scale_y).square().mean()
    activity=torch.zeros_like(graph.rhs)
    activity.index_add_(1,graph.row,graph.coefficient*x[:,graph.column])
    scale=torch.zeros_like(graph.rhs)
    scale.index_add_(1,graph.row,graph.coefficient.abs()*model.scale_x[graph.column])
    residual=activity-graph.rhs
    eq=torch.arange(graph.rhs.shape[1],device=x.device)<graph.neq
    residual=torch.where(eq,residual,residual.clamp_min(0.))
    if physics_mode=='scaled_mean':
        physics=(residual/scale.clamp_min(1e-3)).square().mean()
    elif physics_mode=='original_worst':
        # Original-unit rare violations must not vanish in a large-row mean.
        # log1p tempers the gradient of very large early errors. This loss is
        # still NOT a tolerance or a substitute for the unchanged runtime gate.
        log_error=torch.log1p(residual.abs()).square()
        physics=log_error.mean()+log_error.amax(1).mean()
    else:raise ValueError('Unknown physics training objective')
    obj=((graph.cost*(x-target_x)).sum(1)/
         (graph.cost*target_x).sum(1).abs().clamp_min(.005)).square().mean()
    weight=2. if physics_mode=='original_worst' else .2
    return lx+.1*ly+weight*physics+.1*obj,dict(primal=lx,dual=ly,physics=physics,objective=obj)


def checkpoint(model,metadata):
    return dict(schema='supervised_full_graph_lp_v1',identity=model.identity,
        config=dict(hidden=model.hidden,rounds=model.rounds,temporal=model.temporal),
        state_dict=model.state_dict(),metadata=metadata)


def load_checkpoint(path,*,device='cuda'):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('schema')!='supervised_full_graph_lp_v1':raise ValueError('Unknown graph checkpoint schema')
    state=data['state_dict']
    model=SupervisedGraphLP(data['identity'],*(state[k] for k in
        ('center_x','scale_x','center_y','scale_y')),**data['config'])
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),data['metadata']
