"""Explicit input/hash-bound learned primal restart for the diagnostic GPU IPM.

Same centered restart as the previous-primal comparator, but the primal comes
from a trained graph model. Never mislabel this as a certified previous state.
"""
import weakref
from .gpu_ipm_warm_state import BoundGpuWarmState
from .gpu_ipm_reoptimization import centered_bound_restart


def bind_graph_restart(solver,graph,full_x,*,checkpoint_sha256,mu=1e-5):
    cp=solver.cp;solver.factor._context()
    if not isinstance(checkpoint_sha256,str) or len(checkpoint_sha256)!=64:
        raise ValueError('Checkpoint SHA256 provenance required')
    if tuple(graph.problem_hashes)!=tuple(solver.problem_hashes):
        raise ValueError('Graph and CURRENT original LP identity/order differ')
    if (not isinstance(full_x,cp.ndarray) or full_x.dtype!=cp.float64
        or full_x.shape!=(solver.batch,solver.full_n) or full_x.device.id!=solver.factor.device
        or not bool(cp.all(cp.isfinite(full_x)))):
        raise ValueError('Finite full original primal on current CUDA device required')
    # Same forward compression as ForestGpuBatchedIPM.solve, never an
    # inverse/PCA fitted on evaluation targets. Numerical reduction is a
    # proposal; the full original LP is still certified after the solve.
    x=(solver._forest_compression@full_x.ravel()).reshape(solver.batch,solver.original_n)
    x=x[solver.batch_index,solver.columns]
    if solver.secondary_forest_plans:
        y=cp.zeros((solver.batch,solver.secondary_forest_map.original_m),dtype=cp.float64)
        x,_=solver.secondary_forest_map.compress(x,y)
    arrays,info=centered_bound_restart(solver,x,mu=mu)
    return BoundGpuWarmState(weakref.ref(solver),arrays,dict(
        source='supervised_GNN_or_GNN_GRU_primal_proposal_NOT_certified_previous_state',
        checkpoint_sha256=checkpoint_sha256,target_problem_sha256=tuple(solver.problem_hashes),
        graph_identity=graph.identity,bound_dual_restart=info,current_CPU_solution_used=False,
        current_LP_requires_new_certificate=True,learned_row_dual_used_in_restart=False),
        getattr(solver,'numeric_update_generation',0),tuple(solver.problem_hashes))
