"""Offline-compiled candidate bases, GPU-only batch selection and certification.

No interpolation, online CPU optimizer, host acceptance decisions, or implicit
zero flux. A miss remains invalid; the caller must stop or explicitly repair.
This is an LP component, not a GPU implementation of the whole environment.
"""
import numpy as np
from .gpu_certified_basis import GpuBasisEvaluator


class GpuBasisBank:
    def __init__(self, anchors, variable_rows):
        if not anchors:
            raise ValueError("At least one offline compiled basis is required")
        self.evaluators=[GpuBasisEvaluator(a,variable_rows) for a in anchors]
        self.root=self.evaluators[0]
        self.cp=self.root.cp
        self.offsets=[]
        for candidate in self.evaluators:
            # This also rejects changes outside the declared row-update space.
            offset=self.root.prepare_host([candidate.anchor["lp"]])["delta"]
            self.offsets.append(offset)
        self.offline_cpu_lp_calls=sum(a["offline_cpu_lp_calls"] for a in anchors)

    def prepare_host(self,problems):
        return self.root.prepare_host(problems)

    def evaluate_device(self,**inputs):
        cp=self.cp
        batch,n=inputs["lower"].shape
        accepted=cp.zeros(batch,dtype=bool)
        values=cp.full((batch,n),cp.nan)
        selected=cp.full(batch,-1,dtype=cp.int32)
        objective=cp.full(batch,cp.nan)
        metrics={k:cp.full(batch,cp.inf) for k in ("primal_residual","dual_violation","relative_kkt_gap")}
        best_score=cp.full(batch,cp.inf)
        best_index=cp.zeros(batch,dtype=cp.int32)
        best_dual_feasible=cp.zeros(batch,dtype=bool)
        best_metrics={k:cp.full(batch,cp.inf) for k in metrics}
        for index,(candidate,offset) in enumerate(zip(self.evaluators,self.offsets)):
            result=candidate.evaluate_device(**dict(inputs,delta=inputs["delta"]-offset))
            choose=~accepted&result["accepted"]
            values=cp.where(choose[:,None],result["values"],values)
            objective=cp.where(choose,result["objective"],objective)
            selected=cp.where(choose,index,selected)
            for key in metrics: metrics[key]=cp.where(choose,result[key],metrics[key])
            accepted|=result["accepted"]
            score=cp.maximum(result["primal_residual"]/candidate.primal_tolerance,
                cp.maximum(result["dual_violation"]/candidate.dual_tolerance,result["relative_kkt_gap"]/candidate.gap_tolerance))
            # Prefer a dual-feasible basis, which can be repaired directly
            # with the unmodified objective. The old mixed residual score
            # could instead select a both-primal-and-dual-infeasible basis.
            dual_feasible=result["basis_dual_violation"]<=1e-8
            improve=cp.isfinite(score)&((dual_feasible&~best_dual_feasible)|
                ((dual_feasible==best_dual_feasible)&(score<best_score)))
            best_dual_feasible=cp.where(improve,dual_feasible,best_dual_feasible)
            best_score=cp.where(improve,score,best_score)
            best_index=cp.where(improve,index,best_index)
            for key in metrics:best_metrics[key]=cp.where(improve,result[key],best_metrics[key])
        return dict(accepted=accepted,values=values,objective=objective,candidate_index=selected,
            **metrics,best_candidate_index=best_index,best_rejected_metrics=best_metrics,
            best_candidate_dual_feasible=best_dual_feasible,
            cpu_lp_calls=0,scope="GPU basis-bank LP evaluation; misses are rejected")
