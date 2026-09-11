"""Reusable reduced GPU LP service with strict original-coordinate gates.

The workspace keeps arithmetic, CSR/transpose and graph buffers on CUDA.
Host normalization/reduction, pattern validation, certificate decisions and
the compatibility output bridge remain explicit. This class has no CPU LP
fallback and is NOT a fully GPU-resident dFBA environment.
"""
from __future__ import annotations

import time
import numpy as np

from .gpu_pdhg_corrector import _validated_problem
from .gpu_reduced_pdhg import GpuReducedPdhgCorrector
from .gpu_pdhg_workspace import GpuPdhgWorkspace


def _same_pattern(left, right):
    return (left.shape == right.shape and np.array_equal(left.indptr,right.indptr)
            and np.array_equal(left.indices,right.indices))


class GpuResidentReducedPdhg(GpuReducedPdhgCorrector):
    """Persistent fixed-pattern solver; changed patterns require a rebuild.

    ``try_update`` returns False for batch/order/CSR changes, without mutation.
    Changed equality fingerprints or invalid numeric input raise. Only normal
    successful updates can be solved, and every solve resets extrapolation and
    the accepted mask from explicitly supplied original-coordinate warm starts.
    """
    name = 'gpu_resident_reduced_pdhg_original_certificate'

    def __init__(self, problems, *, environment_ids=None, chunk_size=64,
                 use_graph=True, plan=None, **kwargs):
        started = time.perf_counter()
        super().__init__(problems,plan=plan,**kwargs)
        self.environment_ids = tuple(range(self.batch)) if environment_ids is None else tuple(environment_ids)
        self.workspace = GpuPdhgWorkspace(self.a,batch_size=self.batch,
            n_variables=self.n,n_constraints=self.m,rhs=self.rhs,c=self.c,
            lower=self.lower,upper=self.upper,tau=self.tau,sigma=self.sigma,
            inequality_mask=self.inequality_mask,theta=self.theta,
            env_ids=self.environment_ids,chunk_size=chunk_size,use_graph=use_graph)
        self.setup_timing.update(workspace_setup_seconds=self.workspace.setup_seconds,
            graph_capture_seconds=self.workspace.capture_seconds,
            graph_upload_seconds=self.workspace.graph_upload_seconds,
            setup_total_seconds=time.perf_counter()-started)
        self.last_update_timing = dict(reused=False,reason='initial_setup',total_seconds=0.)
        self._valid = True

    def try_update(self, problems, *, environment_ids=None):
        started = time.perf_counter()
        original = tuple(_validated_problem(p) for p in problems)
        ids = tuple(range(len(original))) if environment_ids is None else tuple(environment_ids)
        def rebuild(reason):
            self.last_update_timing = dict(reused=False,reason=reason,total_seconds=time.perf_counter()-started)
            return False
        if len(original) != self.batch or ids != self.environment_ids:
            return rebuild('batch_or_environment_order_changed')
        # Fingerprint, homogeneity, dimensions and bound intersections must be
        # validated before touching any persistent device buffer.
        reductions = tuple(self.plan.reduce(p) for p in original)
        reduced = tuple(r.problem for r in reductions)
        if not all(_same_pattern(p[0],q[0]) for p,q in zip(original,self.original_problems)):
            return rebuild('original_csr_pattern_changed')
        if not all(_same_pattern(p[0],q[0]) for p,q in zip(reduced,self.reduced_problems)):
            return rebuild('reduced_csr_pattern_changed')
        cp = self.cp
        preparation = time.perf_counter()-started
        self.workspace.synchronize()
        self._valid = False
        before = time.perf_counter()
        # CSR positions and dimensions have already matched; do not replace
        # captured buffers. Each coefficient, including dynamic performance
        # constraints, is copied from the current unmodified LP.
        cp.copyto(self.a.data,cp.asarray(np.concatenate([p[0].data for p in reduced])))
        cp.copyto(self._original_a.data,cp.asarray(np.concatenate([p[0].data for p in original])))
        for destination,index in ((self.rhs,1),(self.lower,2),(self.upper,3),(self.c,4)):
            cp.copyto(destination,cp.asarray(np.concatenate([p[index] for p in reduced])))
        for destination,index in zip(self._original_assembled[2:],(1,2,3,4)):
            cp.copyto(destination,cp.asarray(np.concatenate([p[index] for p in original])))
        cp.copyto(self.row_lower,cp.where(self.inequality_mask,-cp.inf,self.rhs))
        cp.copyto(self._lower_witness,cp.asarray(np.stack([r.lower_witness for r in reductions])))
        cp.copyto(self._upper_witness,cp.asarray(np.stack([r.upper_witness for r in reductions])))
        absolute = self.a.copy()
        absolute.data = cp.abs(absolute.data)
        row_sum = cp.asarray(absolute.sum(axis=1)).ravel()
        col_sum = cp.asarray(absolute.sum(axis=0)).ravel()
        cp.copyto(self.sigma,cp.where(row_sum>0.,self.step_safety/cp.maximum(row_sum,cp.finfo(cp.float64).tiny),1.)*self.primal_weight)
        cp.copyto(self.tau,cp.where(col_sum>0.,self.step_safety/cp.maximum(col_sum,cp.finfo(cp.float64).tiny),1.)/self.primal_weight)
        self.workspace.update_problem(a=self.a,rhs=self.rhs,c=self.c,lower=self.lower,
            upper=self.upper,tau=self.tau,sigma=self.sigma,
            inequality_mask=self.inequality_mask,theta=self.theta,env_ids=ids)
        cp.cuda.runtime.deviceSynchronize()
        self.original_problems = original
        self.reductions,self.reduced_problems = reductions,reduced
        self.problems = reduced
        self._last_lift = None
        self._valid = True
        self.last_update_timing = dict(reused=True,reason='same_pattern_value_update',
            host_preparation_seconds=preparation,device_update_seconds=time.perf_counter()-before,
            total_seconds=time.perf_counter()-started)
        return True

    def solve(self, initial_x=None, initial_y=None, *, iterations=1000, check_interval=64):
        if not self._valid:
            raise RuntimeError('An interrupted update invalidated this solver; rebuild it')
        iterations = self._valid_budget(iterations,'iterations',allow_zero=True)
        check_interval = self._valid_budget(check_interval,'check_interval',allow_zero=False)
        started = time.perf_counter()
        cp = self.cp
        before = time.perf_counter()
        full_x = self._warm_array(initial_x,(self.batch,self.original_n),'initial_x',
            lambda:cp.zeros((self.batch,self.original_n),dtype=cp.float64))
        full_y = self._warm_array(initial_y,(self.batch,self.original_m),'initial_y',
            lambda:cp.zeros((self.batch,self.original_m),dtype=cp.float64))
        z = (self._compression @ full_x.ravel()).reshape(self.batch,self.n)
        yr = full_y[:,self._kept_rows].copy()
        self.workspace.reset(x=z,y=yr)
        warm_seconds = time.perf_counter()-before
        correction_seconds = certificate_seconds = control_seconds = 0.
        accepted = np.zeros(self.batch,dtype=bool)
        accepted_iteration = np.full(self.batch,-1,dtype=np.int64)
        checkpoints = []
        completed = 0
        while True:
            z,yr = self.workspace.state()
            before = time.perf_counter()
            metrics = self._certificate(z,yr)
            certificate_elapsed = time.perf_counter()-before
            certificate_seconds += certificate_elapsed
            checkpoints.append(self._checkpoint(completed,metrics,certificate_elapsed))
            now = np.array([r['certificate_passed'] for r in metrics],dtype=bool)
            accepted_iteration[~accepted & now] = completed
            accepted |= now
            if accepted.all() or completed == iterations:
                break
            before = time.perf_counter()
            self.workspace.set_accepted(cp.asarray(accepted))
            self.workspace.synchronize()
            control_seconds += time.perf_counter()-before
            count = min(check_interval,iterations-completed)
            before = time.perf_counter()
            self.workspace.run(count)
            self.workspace.synchronize()
            correction_seconds += time.perf_counter()-before
            completed += count
        # A frozen block was accepted at an earlier checkpoint, but its final
        # lifted certificate is the authority for the returned candidate.
        # Never let a previous pass mask a final numerical certification loss.
        accepted &= now
        accepted_iteration[~accepted] = -1
        # Return OWNED reduced arrays, not borrowed workspace views that would
        # silently change after reset/update/next solve. Original lift already
        # owns its arrays, including rejected candidates for diagnostics.
        before = time.perf_counter()
        reduced_x,reduced_y = z.copy(),yr.copy()
        cp.cuda.runtime.deviceSynchronize()
        copy_seconds = time.perf_counter()-before
        for index,row in enumerate(metrics):
            row.update(accepted_iteration=int(accepted_iteration[index]),
                       success=bool(accepted[index]),solver=self.name)
        timing = dict(warm_start_transfer_seconds=warm_seconds,
            correction_seconds=correction_seconds,certificate_seconds=certificate_seconds,
            host_control_seconds=control_seconds,owned_result_copy_seconds=copy_seconds,
            solve_total_seconds=time.perf_counter()-started)
        x,y = self._last_lift
        result = dict(x=x,y=y,reduced_x=reduced_x,reduced_y=reduced_y,
            accepted=accepted,all_accepted=bool(accepted.all()),iterations_run=completed,
            metrics=metrics,checkpoints=checkpoints,timing=timing,
            cpu_lp_calls=0,primal_weight=self.primal_weight,equality_reduction=True,
            scope='Resident FP64 reduced GPU iterations; original-LP certification; '
                  'host input/update/control remains; no CPU optimizer or fallback')
        self.history.append(dict(batch=self.batch,iterations_run=completed,
            all_accepted=bool(accepted.all()),accepted=accepted.tolist(),
            metrics=[dict(r) for r in metrics],timing=dict(timing),cpu_lp_calls=0,
            update=dict(self.last_update_timing),execution='graph' if self.workspace.use_graph else 'fused_loop'))
        return result
