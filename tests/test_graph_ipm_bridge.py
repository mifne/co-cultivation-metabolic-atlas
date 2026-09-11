import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch

from src.graph_temporal_lp import LPGraphBatch, GraphTemporalLP, GraphTemporalSession
from src.gpu_batched_ipm import GpuBatchedIPM
from src.graph_ipm_bridge import diagnostic_graph_ipm
from scripts.probe_downstream_gpu_coverage import paired_certificate


def problems():
    return [(csr_matrix([[1.,1.]]), np.array([r]), np.zeros(2), np.full(2,np.inf),
             np.array([1.,2.]), 1) for r in (1.,2.)]


def graph(ps=None):
    return LPGraphBatch.from_problems(ps or problems(), stage='exchange',
        model_identity='toy-gpu-bridge-v1', device='cuda')


def test_random_gnn_gru_is_only_a_proposal_gpu_corrector_must_certify():
    torch.manual_seed(20260905)
    session=GraphTemporalSession(GraphTemporalLP(hidden=8).cuda().eval())
    with GpuBatchedIPM(problems()) as solver:
        g=graph()
        result=diagnostic_graph_ipm(g,solver,session=session,environment_ids=['a','b'],iterations=30)
        assert result['accepted'].all(),result['metrics']
        assert result['cpu_lp_calls']==0 and result['factor_count']>0
        assert result['proposal_trained_status']=='unspecified_not_a_validated_artifact'
        for p,x,y in zip(problems(),result['x'].get(),result['y'].get()):
            assert paired_certificate(p,x,y)['certificate_passed']
        # No hidden state survives a failed certified step.
        failed=diagnostic_graph_ipm(g,solver,session=session,environment_ids=['a','b'],iterations=0)
        assert not failed['accepted'].any() and not session._history


def test_graph_input_order_or_values_must_match_solver_and_mode_is_explicit():
    model=GraphTemporalLP(hidden=4,temporal=False).cuda().eval()
    with GpuBatchedIPM(problems()) as solver:
        with pytest.raises(ValueError):diagnostic_graph_ipm(graph(problems()[::-1]),solver,model=model)
        with pytest.raises(ValueError):diagnostic_graph_ipm(graph(),solver)
        model.train()
        with pytest.raises(ValueError):diagnostic_graph_ipm(graph(),solver,model=model)
