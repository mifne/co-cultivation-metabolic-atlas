"""Opt-in equality-preserving GPU correction with original-LP certification.

The fixed homogeneous equality forest is analyzed once on the host. Dynamic
LP coefficients/bounds/objectives are transformed on the host, without a CPU
optimizer. PDHG, primal expansion, bound-normal allocation, dual restoration,
and the ORIGINAL-unit certificate run on CUDA. This is not fully GPU-resident
dFBA: input preparation and certificate decisions still cross the host bridge.
"""
from __future__ import annotations

import time
import numpy as np

from .gpu_pdhg_corrector import GpuPdhgCorrector, _validated_problem
from .gpu_block_lp import certify_blocks_device
from .lp_equality_reduction import HomogeneousEqualityReduction


class GpuReducedPdhgCorrector(GpuPdhgCorrector):
    """Iterate in an exact reduced space; accept ONLY a lifted original LP.

    Public warm starts and returned x/y use original coordinates. The saved
    plan may be reused only when its equality fingerprint still matches. No
    current CPU reference solution, CPU fallback, or relaxed gate is used.
    """

    name = 'gpu_fp64_equality_reduced_pdhg_original_certificate'

    def __init__(self, problems, *, plan=None, **kwargs):
        started = time.perf_counter()
        original = tuple(_validated_problem(p) for p in problems)
        if not original:
            raise ValueError('At least one independent LP is required')
        shape, neq = original[0][0].shape, original[0][-1]
        if any(p[0].shape != shape or p[-1] != neq for p in original):
            raise ValueError('Original LP batch requires identical dimensions and equality counts')
        before = time.perf_counter()
        self.plan = HomogeneousEqualityReduction.from_problem(original[0]) if plan is None else plan
        plan_build_seconds = time.perf_counter()-before if plan is None else 0.
        before = time.perf_counter()
        self.reductions = tuple(self.plan.reduce(p) for p in original)
        self.reduced_problems = tuple(r.problem for r in self.reductions)
        host_reduction_seconds = time.perf_counter()-before
        super().__init__(self.reduced_problems, **kwargs)
        self.original_problems = original
        self.original_m, self.original_n = shape
        self.original_neq = neq
        cp = self.cp
        from cupyx.scipy.sparse import csr_matrix, block_diag
        before = time.perf_counter()
        self._original_a = block_diag([csr_matrix(p[0]) for p in original], format='csr')
        original_rhs, original_lower, original_upper, original_c = [
            cp.asarray(np.concatenate([p[i] for p in original]), dtype=cp.float64)
            for i in (1, 2, 3, 4)]
        self._original_c = original_c
        self._original_assembled = (self._original_a, None, original_rhs,
                                    original_lower, original_upper, original_c)
        self._T = block_diag([csr_matrix(self.plan.T)]*self.batch, format='csr')
        self._compression = block_diag([csr_matrix(self.plan.compression)]*self.batch, format='csr')
        self._dual_lift = block_diag([csr_matrix(self.plan.dual_lift_map)]*self.batch, format='csr')
        self._kept_rows = cp.asarray(self.plan.kept_rows)
        self._eliminated_rows = cp.asarray(self.plan.eliminated_rows)
        self._lower_witness = cp.asarray(np.stack([r.lower_witness for r in self.reductions]))
        self._upper_witness = cp.asarray(np.stack([r.upper_witness for r in self.reductions]))
        self._fallback_witness = cp.asarray(np.stack([r.fallback_witness for r in self.reductions]))
        self._weights = cp.asarray(self.plan.weights, dtype=cp.float64)
        self._batch_indices = cp.arange(self.batch)[:, None]
        cp.cuda.runtime.deviceSynchronize()
        self.setup_timing.update(
            equality_plan_build_seconds=plan_build_seconds,
            host_equality_reduction_seconds=host_reduction_seconds,
            original_lift_setup_seconds=time.perf_counter()-before,
            setup_total_seconds=time.perf_counter()-started,
            original_variables=self.original_n, original_rows=self.original_m,
            reduced_variables=self.n, reduced_rows=self.m,
            eliminated_equalities=len(self.plan.eliminated_rows),
            equality_fingerprint=self.plan.equality_fingerprint,
        )
        self._last_lift = None

    def expand_and_lift(self, z, y_reduced):
        """Device-only expansion and dual postsolve (no LP solve on host).

        A transformed box normal is assigned to the ORIGINAL bound that
        produced the tight intersection endpoint, not distributed as T*r.
        Eliminated equality multipliers then complete stationarity.
        """
        cp = self.cp
        original_x = (self._T @ z.ravel()).reshape(self.batch, self.original_n)
        original_y = cp.zeros((self.batch, self.original_m), dtype=cp.float64)
        original_y[:, self._kept_rows] = y_reduced
        q = self._original_c - self._original_a.T @ original_y.ravel()
        reduced_cost = (self._T.T @ q).reshape(self.batch, self.n)
        witness = cp.where(reduced_cost >= 0., self._lower_witness, self._upper_witness)
        witness = cp.where(witness >= 0, witness, self._fallback_witness)
        normal = cp.zeros((self.batch, self.original_n), dtype=cp.float64)
        # Groups have disjoint original variables, hence no duplicate scatter
        # indices within an environment. Negative T weights reverse signs.
        normal[self._batch_indices, witness] = reduced_cost/self._weights[witness]
        if len(self.plan.eliminated_rows):
            removed_y = self._dual_lift @ (q-normal.ravel())
            original_y[:, self._eliminated_rows] = removed_y.reshape(self.batch, -1)
        return original_x, original_y

    def _certificate(self, x, y):
        original_x, original_y = self.expand_and_lift(x, y)
        self._last_lift = (original_x, original_y)
        return certify_blocks_device(
            self.original_problems, self._original_assembled,
            original_x.ravel(), original_y.ravel(), cp=self.cp,
            allow_box_dual=False,
        )

    def solve(self, initial_x=None, initial_y=None, *, iterations=1000, check_interval=50):
        started = time.perf_counter()
        cp = self.cp
        before = time.perf_counter()
        full_x = self._warm_array(initial_x, (self.batch, self.original_n), 'initial_x',
                                 lambda: cp.zeros((self.batch, self.original_n), dtype=cp.float64))
        full_y = self._warm_array(initial_y, (self.batch, self.original_m), 'initial_y',
                                 lambda: cp.zeros((self.batch, self.original_m), dtype=cp.float64))
        z = (self._compression @ full_x.ravel()).reshape(self.batch, self.n)
        y = full_y[:, self._kept_rows].copy()
        cp.cuda.runtime.deviceSynchronize()
        compression_seconds = time.perf_counter()-before
        result = super().solve(initial_x=z, initial_y=y, iterations=iterations, check_interval=check_interval)
        # Every final state was certified at the last checkpoint. Reuse that
        # expansion, rather than doing another postsolve after the timed solve.
        result['reduced_x'], result['reduced_y'] = result['x'], result['y']
        result['x'], result['y'] = self._last_lift
        result['timing']['original_warm_compression_seconds'] = compression_seconds
        result['timing']['solve_total_seconds'] = time.perf_counter()-started
        result['equality_reduction'] = True
        result['scope'] = ('Exact homogeneous equality reduction; GPU reduced-space PDHG, '
                           'device primal/dual restoration, strict ORIGINAL-LP certificate. '
                           'Host structural/input preparation remains; no CPU LP fallback.')
        self.history[-1]['timing'] = dict(result['timing'])
        self.history[-1]['equality_reduction'] = True
        return result
