"""Opt-in host dFBA adapter for an offline compiled GPU basis bank.

LP assembly/normalization and result transfer remain on the CPU in this adapter.
Only bank evaluation and numerical certification are device-resident. No bank
miss is silently accepted and there is no online CPU LP fallback.
"""
import time
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix,vstack

from .lp_bounds import split_variable_bounds


def lp_arrays(c,*,A_eq=None,b_eq=None,A_ub=None,b_ub=None,bounds=None,**ignored):
    c=np.asarray(c,dtype=float); n=len(c)
    eq=csr_matrix((0,n)) if A_eq is None else csr_matrix(A_eq)
    ub=csr_matrix((0,n)) if A_ub is None else csr_matrix(A_ub)
    matrix=vstack((eq,ub),format="csr")
    rhs=np.r_[np.zeros(eq.shape[0]) if b_eq is None else b_eq,
              np.zeros(ub.shape[0]) if b_ub is None else b_ub]
    lower,upper=split_variable_bounds(bounds,n,checked=False)
    return matrix,rhs,lower,upper,c,eq.shape[0]


def stage_key(c,matrix,neq,n_fluxes):
    if len(c)==n_fluxes+1:
        name="maxmin" if np.count_nonzero(c)==1 and c[-1]==-1 else "aggregate"
    else:
        name="exchange" if np.all(c[n_fluxes+1:]==1) else "exchange_tie"
    return (name,matrix.shape[0],matrix.shape[1],neq)


class GpuCompiledCommunityBackend:
    name="gpu_compiled_basis_bank_experimental"
    method="compiled_basis_KKT"

    def __init__(self,coordinates,banks,*,repair_pivots=0,optimal_face_tie=False,warm_updates=False,capture_repair=False):
        self.coordinates=coordinates
        self.banks=banks
        self.history=[]
        self.repair_pivots=repair_pivots
        self.repair_cache=OrderedDict()
        self.optimal_face_tie=optimal_face_tie
        self.last_exchange=None
        self.warm_updates=warm_updates
        self.previous_states={}
        self.capture_repair=capture_repair
        if optimal_face_tie and repair_pivots<1:raise ValueError("Optimal-face solve requires a GPU pivot budget")

    def _repair_for(self,bank,key,index):
        from .gpu_revised_basis import GpuRevisedBasis
        cache_key=(key,index)
        if cache_key not in self.repair_cache:
            while len(self.repair_cache)>=3:self.repair_cache.popitem(last=False)
            candidate=bank.evaluators[index]
            self.repair_cache[cache_key]=GpuRevisedBasis(candidate.anchor,candidate.variable_rows,max_pivots=self.repair_pivots,capture_safe=self.capture_repair,adapter=candidate)
        self.repair_cache.move_to_end(cache_key)
        return self.repair_cache[cache_key]

    def _face_solve(self,arrays,started,record):
        from .gpu_optimal_face import solve_on_primary_face
        if self.last_exchange is None:raise ValueError("Missing primary exchange state")
        solver,primary,original_c=self.last_exchange
        self.last_exchange=None
        matrix,rhs,lower,upper,c,neq=arrays
        if not np.array_equal(matrix[-1].toarray().ravel(),original_c):
            raise ValueError("Secondary row does not preserve original primary cost")
        problem=self.coordinates.normalize(matrix[:-1],rhs[:-1],lower,upper,c,neq)
        inputs=solver.prepare_host([problem])
        cp=solver.cp
        allowance=cp.asarray([rhs[-1]])-primary["objective"]
        result=solve_on_primary_face(solver,primary,inputs,inputs["c"],allowance)
        record.update(success=bool(result["accepted"].get()[0]),selection_route="reduced_cost_face_original_LP_certificate",
            repair_pivots=int(result["pivots"].get()[0]),fixed_variables=int(result["fixed_variables"].get()[0]),
            face_multiplier=float(result["multiplier"].get()[0]),
            face_metrics={k:float(result[k].get()[0]) for k in ("primal_residual","dual_violation","relative_kkt_gap")},
            total_seconds=time.perf_counter()-started)
        values=objective=None
        if record["success"]:
            values=result["values"].get()[0];objective=float(result["objective"].get()[0])
            record.update(max_original_residual=float(result["primal_residual"].get()[0]),objective=objective)
        else:record["status"]="optimal_face_original_certificate_failed"
        self.history.append(record)
        return SimpleNamespace(success=record["success"],x=values,fun=objective,message=str(record))

    def solve(self,c,**kwargs):
        started=time.perf_counter()
        arrays=lp_arrays(c,**kwargs)
        matrix,rhs,lower,upper,c,neq=arrays
        key=stage_key(c,matrix,neq,self.coordinates.n_fluxes)
        record=dict(stage=key[0],success=False,cpu_lp_calls=0,cpu_optimization_allowed=False,
            cpu_iteration_counts=dict(simplex_iteration_count=0,ipm_iteration_count=0,crossover_iteration_count=0),
            scope="GPU LP kernel with CPU LP assembly/state updates, not fully resident dFBA")
        values=objective=None
        if self.optimal_face_tie and key[0]=="exchange_tie":
            try:
                return self._face_solve(arrays,started,record)
            except ValueError as error:
                record.update(success=False,status="unsupported_face: "+str(error),total_seconds=time.perf_counter()-started)
                self.history.append(record)
                return SimpleNamespace(success=False,x=None,fun=None,message=str(record))
        if key in self.banks:
            bank=self.banks[key]
            try:
                problem=self.coordinates.normalize(*arrays)
                result=bank.evaluate_device(**bank.prepare_host([problem]))
                record["success"]=bool(result["accepted"].get()[0])
                record["candidate_index"]=int(result["candidate_index"].get()[0])
                record["bank_seconds"]=time.perf_counter()-started
                if not record["success"]:
                    best=int(result["best_candidate_index"].get()[0])
                    record["best_candidate_index"]=best
                    record['best_candidate_dual_feasible']=bool(result['best_candidate_dual_feasible'].get()[0])
                    record["best_rejected_metrics"]={k:float(v.get()[0]) for k,v in result["best_rejected_metrics"].items()}
                    if self.repair_pivots:
                        setup_started=time.perf_counter()
                        if self.warm_updates and key in self.previous_states:
                            previous=self.previous_states[key]
                            repair=previous['owner']
                            result=repair.run_device(**repair.prepare_host([problem]),warm_start=previous,update_warm_matrix=True,diagnostics=not self.capture_repair)
                            record['warm_attempt_success']=bool(result['accepted'].get()[0])
                            record['warm_attempt_pivots']=int(result['pivots'].get()[0])
                            record['warm_attempt_metrics']={k:float(result[k].get()[0]) for k in
                                ('primal_residual','dual_violation','relative_kkt_gap')}
                            if not record['warm_attempt_success'] and 'trace' in result:record['warm_failure_trace']=result['trace'].get().tolist()
                        if not record.get('warm_attempt_success',False):
                            repair=self._repair_for(bank,key,best)
                            record["repair_setup_seconds"]=time.perf_counter()-setup_started
                            result=repair.run_device(**repair.prepare_host([problem]),diagnostics=not self.capture_repair)
                        record["success"]=bool(result["accepted"].get()[0])
                        if not record['success'] and 'trace' in result:record['failure_trace']=result['trace'].get().tolist()
                        record['graph_compilation_seconds_cumulative']=repair.graph_compilation_seconds
                        record["repair_pivots"]=int(result["pivots"].get()[0])
                        record["repair_metrics"]={k:float(result[k].get()[0]) for k in
                            ("primal_residual","dual_violation","relative_kkt_gap")}
                        record["repair_cpu_lp_calls"]=result["cpu_lp_calls"]
                if record["success"]:
                    values=result["values"].get()[0]
                    objective=float(result["objective"].get()[0])
                    record.update(max_original_residual=float(result["primal_residual"].get()[0]),
                        dual_violation=float(result["dual_violation"].get()[0]),
                        relative_kkt_gap=float(result["relative_kkt_gap"].get()[0]),objective=objective)
                    if self.warm_updates and 'warm_state' in result:
                        self.previous_states[key]=result['warm_state']
                    if self.optimal_face_tie and key[0]=="exchange":
                        if "warm_state" not in result:
                            repair=self._repair_for(bank,key,record["candidate_index"])
                            result=repair.run_device(**repair.prepare_host([problem]),pivot_budget=0)
                            if not bool(result["accepted"].get()[0]):raise ValueError("Primary warm-state certification failed")
                        self.last_exchange=(result["warm_state"]["owner"],result,c.copy())
                else:
                    record["status"]="bank_miss_no_cpu_fallback"
            except ValueError as error:
                record["success"]=False
                values=objective=None
                record["status"]="unsupported_structure: "+str(error)
        else:
            record["status"]="unknown_stage_no_cpu_fallback"
        record["total_seconds"]=time.perf_counter()-started
        self.history.append(record)
        return SimpleNamespace(success=record["success"],x=values,fun=objective,message=str(record))
