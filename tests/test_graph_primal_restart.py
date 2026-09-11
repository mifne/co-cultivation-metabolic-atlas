import dataclasses
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.graph_temporal_lp import LPGraphBatch
from src.gpu_forest_ipm import ForestGpuBatchedIPM
from src.graph_primal_restart import bind_graph_restart


def problems():
    return [(csr_matrix([[1.,-1.,0.],[0.,0.,1.]]),np.array([0.,1.]),np.zeros(3),
             np.array([2.,2.,2.]),np.array([-1.,-1.,1.]),1)]


def graph():
    return LPGraphBatch.from_problems(problems(),stage='maxmin',model_identity='unit-graph',device='cuda')


def test_learned_restart_needs_current_certificate_and_owns_primal():
    import cupy as cp
    with ForestGpuBatchedIPM(problems(),allow_box_dual=True,globalized=True,second_forest=True) as s:
        g=graph();x=cp.array([[1.,1.,.5]])
        bound=bind_graph_restart(s,g,x,checkpoint_sha256='a'*64)
        assert 'NOT_certified' in bound.metadata['source']
        x[:]=999
        r=s.solve(internal_warm_start=bound,iterations=0)
        assert not r['accepted'].any()
        assert np.max(r['x'].get())<999
        assert not bound.metadata['current_CPU_solution_used']


def test_original_optimal_candidate_can_pass_without_newton():
    import cupy as cp
    with ForestGpuBatchedIPM(problems(),allow_box_dual=True,globalized=True,second_forest=True) as s:
        bound=bind_graph_restart(s,graph(),cp.array([[2.,2.,0.]]),checkpoint_sha256='a'*64)
        r=s.solve(internal_warm_start=bound,iterations=0)
        assert r['accepted'].all() and r['factor_count']==0


def test_reject_stale_hash_shape_or_generation():
    import cupy as cp
    with ForestGpuBatchedIPM(problems(),allow_box_dual=True,globalized=True) as s:
        g=graph();x=cp.array([[2.,2.,0.]])
        with pytest.raises(ValueError,match='identity'):
            bind_graph_restart(s,dataclasses.replace(g,problem_hashes=('wrong',)),x,checkpoint_sha256='a'*64)
        with pytest.raises(ValueError,match='primal'):
            bind_graph_restart(s,g,x[:,:2],checkpoint_sha256='a'*64)
        bound=bind_graph_restart(s,g,x,checkpoint_sha256='a'*64)
        s.numeric_update_generation=1
        with pytest.raises(ValueError,match='changed'):bound.initialize(s)
