from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import torch

from src.gpu_batch_qp import BatchedCooperativeQpProjector
from src.cooperative_gpu_service import CooperativeGpuQpService


def test_live_objective_multitarget_microbatch_and_retry(monkeypatch):
    constructor_options = {}
    class Backend:
        metadata = {"context_layout":[["biomass_g_l", 1], ["shared_supply_mmol_l_h", 1],
            ["flux_lower_bounds", 2], ["flux_upper_bounds", 2]]}
        decision_indices = torch.tensor([0, 1])
        decision_scale = torch.ones(2)
        decision_weight = torch.ones(2)
        def rank_device(self, contexts, top_k):
            rows = [[2., 2.], [3., 3.]] if top_k == 2 else [[0., 0.], [1., 1.], [.3, .2], [.2, .3]]
            return SimpleNamespace(fluxes=torch.tensor([rows]*len(contexts)), reference_weights=None,
                predicted_decision_fluxes=torch.tensor([[.4, .3]]*len(contexts)))
    def solver(*args, **kwargs):
        constructor_options.update(kwargs)
        return SimpleNamespace(_surrogate_dictionary=Backend(), cpu_lp_stage_calls=0,
            optimize_live_objectives=kwargs["optimize_live_objectives"],
            _gpu_qp_projector=BatchedCooperativeQpProjector(["toy"], ["carbon_e"],
                [(0, 0, 0, 1.), (0, 0, 1, 1.)], device="cpu", maximum_iterations=200))
    monkeypatch.setattr("src.cooperative_gpu_service.CooperativeCommunityFbaSolver", solver)
    service = CooperativeGpuQpService({}, {}, "unused", device="cpu", candidates=2,
        retry_candidates=4, optimize_live_objectives=True, multioutput_strength=30,
        batch_window_ms=30, max_batch_size=2)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(service.predict, [np.asarray([1, 1, 0, 0, 1, 1])]*2))
        assert constructor_options["optimize_live_objectives"] is True
        assert all(row["feasible"] and row["retry_used"] for row in results)
        assert all(sum(row["fluxes"]) <= 1.0003 for row in results)
        diagnostics = service.diagnostics()
        assert diagnostics["requests"] == diagnostics["retry_recovered"] == 2
        assert diagnostics["online_cpu_lp_stage_calls"] == 0
        assert diagnostics["multioutput_strength"] == 30
    finally:
        service.close()
