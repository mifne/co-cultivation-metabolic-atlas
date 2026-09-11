"""Experimental GPU-only LP tie-break on the existing exchange-optimal face.

Does not modify the CPU reference, GEM or the first three objective stages.
An additional objective is an explicit algorithmic selection policy, not a
claim that observed organisms have this nutrient preference. Not qualified.
"""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix, vstack
from .gpu_lexicographic_lp import PresolvedCuOptBackend


class GpuExchangeTieBreakBackend:
    name = "gpu_simplex_exchange_tie_experimental"
    method = "tableau_exchange_tie"

    def __init__(self, layout, *, inner=None, **kwargs):
        self.inner = inner if inner is not None else PresolvedCuOptBackend(method="tableau", **kwargs)
        self.engine = getattr(self.inner, "engine", None)
        self.history = []
        self.n_fluxes = layout.n_fluxes
        self.aux_metabolites = [met for met, terms in layout._exchange_terms.items() for _ in terms]
        self.actual_gpu_lp_calls = 0
        self.tie_lp_calls = 0
        self.primary_face_relative_allowance = 1e-10

    def solve(self, c, **kwargs):
        c = np.asarray(c, dtype=float)
        primary = self.inner.solve(c, **kwargs)
        self.actual_gpu_lp_calls += 1
        record = dict(self.inner.history[-1])
        record["actual_gpu_lp_calls"] = 1
        is_parsimony = (len(c) == self.n_fluxes + 1 + len(self.aux_metabolites)
            and np.all(c[:self.n_fluxes+1] == 0)
            and np.all(c[self.n_fluxes+1:] == 1))
        if not primary.success or not is_parsimony:
            self.history.append(record)
            return primary
        objective = np.zeros_like(c)
        objective[self.n_fluxes+1:] = [0. if met in {"h2o_e", "h_e", "oh1_e"} else 1.
                                      for met in self.aux_metabolites]
        allowance = self.primary_face_relative_allowance * max(1., abs(primary.fun))
        tie_kwargs = dict(kwargs,
            A_ub=vstack((kwargs["A_ub"], csr_matrix(c[None])), format="csr"),
            b_ub=np.r_[kwargs["b_ub"], primary.fun+allowance])
        selected = self.inner.solve(objective, **tie_kwargs)
        self.actual_gpu_lp_calls += 1
        self.tie_lp_calls += 1
        record.update(tie_break=self.inner.history[-1], actual_gpu_lp_calls=2,
            primary_face_allowance=allowance,primary_optimum=primary.fun,
            selection_policy="minimize_exchange_L1_excluding_water_proton_hydroxide_on_primary_optimal_face",
            total_seconds=record["total_seconds"]+self.inner.history[-1]["total_seconds"])
        if selected.success:
            primary_value=float(c@selected.x)
            record["selected_primary_value"]=primary_value
            record["primary_objective_loss"]=primary_value-primary.fun
            if primary_value > primary.fun+allowance+1e-8:
                selected=SimpleNamespace(success=False,x=None,fun=None,
                    message="Primary exchange objective lost during tie-break")
            else:
                record["max_original_residual"]=max(record["max_original_residual"],
                    self.inner.history[-1]["max_original_residual"])
                record["objective"]=primary_value
                selected=SimpleNamespace(success=True,x=selected.x,fun=primary_value,message="GPU exchange tie-break")
        record["success"]=selected.success
        self.history.append(record)
        return selected
