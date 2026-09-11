"""Explicit diagnostic bridge: GNN/GRU proposal -> GPU IPM -> full LP gate.

No trained-artifact claim, production registration, CPU LP, or host round-trip
of primal/dual vectors. Public setup/control/certificate summaries stay on host.
The prototype model and IPM must be validated separately before live adoption.
"""
import time


def diagnostic_graph_ipm(graph, solver, *, model=None, session=None,
                         environment_ids=None, iterations=60):
    import cupy as cp
    import torch
    if (model is None) == (session is None):
        raise ValueError('Supply exactly one graph model or transactional session')
    if graph.problem_hashes != solver.problem_hashes:
        raise ValueError('Graph inputs/order differ from the exact LP batch to correct')
    if graph.cost.device.type != 'cuda' or graph.cost.device.index != solver.factor.device:
        raise ValueError('Graph and numerical solver must share the NVIDIA CUDA device')
    active_model = model if session is None else session.model
    if active_model.training:
        raise ValueError('Diagnostic inference requires explicit model.eval()')
    if session is not None and environment_ids is None:
        raise ValueError('Temporal inference requires explicit environment IDs')
    before = time.perf_counter()
    with torch.inference_mode():
        if session is None:
            proposal = model(graph)
            token = None
        else:
            proposal, token = session.propose(graph, environment_ids)
        # DLPack performs stream-aware exchange of GPU buffers. No .cpu()/.get()
        # occurs here. Keep proposal owners alive until the numerical call ends.
        x = cp.from_dlpack(proposal.x.contiguous())
        y = cp.from_dlpack(proposal.y.contiguous())
        cp.cuda.get_current_stream().synchronize()
        proposal_seconds = time.perf_counter()-before
        result = solver.solve(initial_x=x, initial_y=y, iterations=iterations)
    if session is not None:
        session.commit(token, proposal, result['accepted'])
    result['graph_proposal_seconds'] = proposal_seconds
    result['graph_plus_correction_seconds'] = time.perf_counter()-before
    result['graph_identity'] = graph.identity
    result['graph_model_parameters'] = sum(p.numel() for p in active_model.parameters())
    result['proposal_path'] = 'experimental_gnn_gru' if active_model.temporal else 'experimental_gnn'
    result['proposal_trained_status'] = 'unspecified_not_a_validated_artifact'
    return result
