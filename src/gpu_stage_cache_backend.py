"""Isolated device prototype with separate primary/tie basis caches.

The original presolved backend labels both exchange objectives 'parsimony'.
Consequently the extra tie LP overwrites the primary LP's previous-step
basis. Preserve separate caches by explicit original objective layout. This
changes warm starts only, never objectives, bounds or acceptance tolerances.
"""
import numpy as np

from .gpu_device_bounded_simplex import GpuDeviceBoundedSimplex
from .gpu_lexicographic_lp import PresolvedCuOptBackend


def objective_stage(c, n_fluxes, aux_metabolites):
    c = np.asarray(c, dtype=float)
    if c.shape == (n_fluxes+1,):
        if np.all(c[:-1] == 0) and c[-1] == -1:
            return "maxmin"
        return "aggregate"
    if c.shape != (n_fluxes+1+len(aux_metabolites),) or np.any(c[:n_fluxes+1] != 0):
        raise ValueError("Unknown original LP objective layout; do not alias its cache")
    if np.all(c[n_fluxes+1:] == 1):
        return "exchange_primary"
    tie_cost = np.array([0. if m in {"h2o_e", "h_e", "oh1_e"} else 1. for m in aux_metabolites])
    if np.array_equal(c[n_fluxes+1:], tie_cost):
        return "exchange_tie"
    raise ValueError("Unknown exchange objective; explicit stage mapping required")


class StageAwareDeviceEngine(GpuDeviceBoundedSimplex):
    original_objective_stage = None

    def solve(self, c, **kwargs):
        if self.original_objective_stage is None:
            raise RuntimeError("Original objective stage not set")
        result = super().solve(c, **dict(kwargs, stage_key=self.original_objective_stage))
        self.history[-1]["warm_cache_namespace"] = self.original_objective_stage
        return result


class StageSeparatedPresolvedBackend(PresolvedCuOptBackend):
    def __init__(self, layout, **kwargs):
        super().__init__(method="tableau", **kwargs)
        self.engine = StageAwareDeviceEngine(**kwargs)
        self.validation_engine = self.engine
        self.n_fluxes = layout.n_fluxes
        self.aux_metabolites = [met for met, terms in layout._exchange_terms.items() for _ in terms]
        self.name = "presolved_gpu_device_separate_objective_basis_cache_experimental"

    def solve(self, c, **kwargs):
        stage = objective_stage(c, self.n_fluxes, self.aux_metabolites)
        self.engine.original_objective_stage = stage
        try:
            result = super().solve(c, **kwargs)
            self.history[-1]["original_objective_stage"] = stage
            return result
        finally:
            self.engine.original_objective_stage = None
