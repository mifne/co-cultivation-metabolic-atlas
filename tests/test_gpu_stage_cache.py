import numpy as np
import pytest

from src.gpu_stage_cache_backend import objective_stage, StageAwareDeviceEngine
from src.gpu_device_bounded_simplex import GpuDeviceBoundedSimplex


def test_four_objective_namespaces_are_distinct():
    metabolites = ["glc__D_e", "h2o_e", "h_e", "oh1_e", "nh4_e"]
    assert objective_stage([0, 0, -1], 2, metabolites) == "maxmin"
    assert objective_stage([-1, -2, 0], 2, metabolites) == "aggregate"
    assert objective_stage([0, 0, 0, 1, 1, 1, 1, 1], 2, metabolites) == "exchange_primary"
    assert objective_stage([0, 0, 0, 1, 0, 0, 0, 1], 2, metabolites) == "exchange_tie"
    with pytest.raises(ValueError):
        objective_stage([0, 0, 0, 1, 0, 0, 0, .9], 2, metabolites)


def test_gpu_repairs_use_the_same_explicit_stage_key(monkeypatch):
    engine = object.__new__(StageAwareDeviceEngine)
    engine.history = []
    engine.original_objective_stage = "exchange_primary"
    def solve(self, c, **kwargs):
        assert kwargs["stage_key"] == "exchange_primary"
        self.history.append({})
        return "unchanged_result"
    monkeypatch.setattr(GpuDeviceBoundedSimplex, "solve", solve)
    assert engine.solve(np.ones(3), stage_key="parsimony") == "unchanged_result"
    assert engine.history[-1]["warm_cache_namespace"] == "exchange_primary"
    engine.original_objective_stage = None
    with pytest.raises(RuntimeError):
        engine.solve(np.ones(3))
