"""Experimental forest -> zero-face -> GPU IPM with full original LP gates.

Forest construction and sparse map assembly are host setup, not CPU LP solves.
Numeric compression, IPM, both primal/dual postsolves and full certificates are
CUDA operations. Every environment owns its own structural forest and bound
witnesses. No full-model bounds, objective or certificate tolerance is changed.
"""
import time

import numpy as np
from scipy.sparse import block_diag

from .gpu_block_lp import assemble_blocks, certify_blocks_device
from .gpu_pdhg_corrector import _validated_problem
from .gpu_zero_face_ipm import ZeroFaceGpuBatchedIPM
from .lp_equality_reduction import HomogeneousEqualityReduction
from .lp_trace import problem_hash
from .lp_zero_face import _freeze_problem


def prepare_forest_batch(problems, *, reuse_equality_proofs=False):
    """CPU setup only; independent plans, common original/reduced dimensions."""
    full = tuple(_freeze_problem(_validated_problem(p)) for p in problems)
    if not full:
        raise ValueError('Nonempty full original LP batch required')
    shape, neq = full[0][0].shape, full[0][-1]
    if any(p[0].shape != shape or p[-1] != neq for p in full):
        raise ValueError('Common full original dimensions and equality count required')
    if type(reuse_equality_proofs) is not bool:raise ValueError('Boolean equality proof reuse required')
    if reuse_equality_proofs:
        from .lp_equality_reduction import prepare_independent_forest_plans
        plans=prepare_independent_forest_plans(full)
    else:
        plans = tuple(HomogeneousEqualityReduction.from_problem(p) for p in full)
    reductions = tuple(plan.reduce(p) for plan, p in zip(plans, full))
    reduced = tuple(r.problem for r in reductions)
    reduced_shape, reduced_neq = reduced[0][0].shape, reduced[0][-1]
    if any(p[0].shape != reduced_shape or p[-1] != reduced_neq for p in reduced):
        raise ValueError('Regroup environments with different forest-reduced dimensions')
    return full, plans, reductions


class ForestGpuBatchedIPM(ZeroFaceGpuBatchedIPM):
    """Public warm starts/results are full coordinates; parents remain local.

    Parent ``original_problems/original_n/original_m`` intentionally describe
    the intermediate forest LP, so its zero-face gather/postsolve is unchanged.
    ``full_problems/full_n/full_m`` identify the LP used for every acceptance.
    """
    name = 'experimental_forest_zero_face_gpu_ipm_full_original_certificate'

    def __init__(self, problems, *, allow_box_dual=False, reuse_equality_proofs=False, **kwargs):
        if type(allow_box_dual) is not bool:
            raise ValueError('allow_box_dual must be boolean')
        self.allow_box_dual=allow_box_dual
        before = time.perf_counter()
        self.full_problems, self.forest_plans, self.forest_reductions = prepare_forest_batch(
            problems,reuse_equality_proofs=reuse_equality_proofs)
        self.full_m, self.full_n = self.full_problems[0][0].shape
        self.full_neq = self.full_problems[0][-1]
        self.forest_problems = tuple(r.problem for r in self.forest_reductions)
        host_seconds = time.perf_counter()-before
        super().__init__(self.forest_problems, reuse_equality_proofs=reuse_equality_proofs, **kwargs)
        parent_setup_seconds = self.setup_seconds
        try:
            self._setup_forest_device()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f'Forest setup cleanup also failed: {cleanup}')
            raise
        self.problem_hashes = tuple(problem_hash(p) for p in self.full_problems)
        self._forest_certificate_seconds = 0.
        self.setup_seconds = time.perf_counter()-before
        self.forest_setup_timing = dict(host_forest_setup_seconds=host_seconds,
            parent_zero_face_ipm_setup_seconds=parent_setup_seconds,
            total_setup_seconds=self.setup_seconds)

    def _setup_forest_device(self):
        cp = self.cp
        from cupyx.scipy.sparse import csr_matrix
        before = time.perf_counter()
        packed = assemble_blocks(self.full_problems)
        self._full_assembled = (csr_matrix(packed[0]), *(cp.asarray(v) for v in packed[1:]))
        self._full_at = self._full_assembled[0].T.tocsr()
        self._forest_t = csr_matrix(block_diag([p.T for p in self.forest_plans], format='csr'))
        self._forest_tt = self._forest_t.T.tocsr()
        self._forest_compression = csr_matrix(block_diag(
            [p.compression for p in self.forest_plans], format='csr'))
        self._forest_dual_lift = csr_matrix(block_diag(
            [p.dual_lift_map for p in self.forest_plans], format='csr'))
        self._forest_kept_rows = cp.asarray(np.stack([p.kept_rows for p in self.forest_plans]))
        self._forest_eliminated_rows = cp.asarray(np.stack([p.eliminated_rows for p in self.forest_plans]))
        self._forest_lower_witness = cp.asarray(np.stack([r.lower_witness for r in self.forest_reductions]))
        self._forest_upper_witness = cp.asarray(np.stack([r.upper_witness for r in self.forest_reductions]))
        self._forest_fallback_witness = cp.asarray(np.stack([r.fallback_witness for r in self.forest_reductions]))
        self._forest_weights = cp.asarray(np.stack([p.weights for p in self.forest_plans]))
        self._forest_batch = cp.arange(self.batch)[:, None]
        cp.cuda.get_current_stream().synchronize()
        self.forest_device_setup_seconds = time.perf_counter()-before

    def expand_forest_device(self, z, y):
        """Forest-coordinate pair -> full original pair, entirely on CUDA."""
        self.factor._context()
        cp = self.cp
        for value, shape in ((z, (self.batch, self.original_n)),
                             (y, (self.batch, self.original_m))):
            if (not isinstance(value, cp.ndarray) or value.shape != shape
                    or value.dtype != cp.float64 or value.device.id != self.factor.device):
                raise ValueError('Exact-shape FP64 forest arrays on the bound CUDA device required')
        full_x = (self._forest_t @ z.ravel()).reshape(self.batch, self.full_n)
        full_y = cp.zeros((self.batch, self.full_m), dtype=cp.float64)
        full_y[self._forest_batch, self._forest_kept_rows] = y
        q = self._full_assembled[5]-self._full_at @ full_y.ravel()
        reduced_cost = (self._forest_tt @ q).reshape(self.batch, self.original_n)
        witness = cp.where(reduced_cost >= 0., self._forest_lower_witness, self._forest_upper_witness)
        witness = cp.where(witness >= 0, witness, self._forest_fallback_witness)
        normal = cp.zeros((self.batch, self.full_n), dtype=cp.float64)
        # Each group's original variables are disjoint, including when forests
        # differ across environments. Use that environment's signed T weight.
        normal[self._forest_batch, witness] = reduced_cost/self._forest_weights[self._forest_batch, witness]
        if self._forest_eliminated_rows.shape[1]:
            removed_y = self._forest_dual_lift @ (q-normal.ravel())
            full_y[self._forest_batch, self._forest_eliminated_rows] = removed_y.reshape(self.batch, -1)
        return full_x, full_y

    def certificate(self, x, y):
        before = time.perf_counter()
        forest_x, forest_y = super().lift_device(x, y)
        full_x, full_y = self.expand_forest_device(forest_x, forest_y)
        metrics = certify_blocks_device(self.full_problems, self._full_assembled,
            full_x.ravel(), full_y.ravel(), cp=self.cp,allow_box_dual=self.allow_box_dual,
            require_direct_dual=bool(self.near_equality_plans or self.second_forest or self.fix_singleton_equalities))
        self._forest_certificate_seconds += time.perf_counter()-before
        return metrics

    def _materialize_box_certificate(self,result,warm_x,warm_y,has_warm):
        """Return the actual certified full pair, not merely alternate metrics.

        A zero row-dual is used only where the unchanged full-original gate
        selected its analytic box certificate. Intermediate forest/reduced
        duals remain diagnostic iterates and need not certify that full x.
        The returned pair is independently rechecked with the alternative
        disabled. Certified iteration-zero caller pairs retain their exact
        original coordinates, including redundant equality multipliers.
        """
        cp=self.cp
        before=time.perf_counter()
        previous_metrics=result['metrics']
        previous_accepted=np.asarray(result['accepted'],dtype=bool).copy()
        box=np.array([m.get('certificate_source')=='analytic_box_bound'
                      for m in previous_metrics],dtype=bool)&previous_accepted
        preserved_warm=np.zeros(self.batch,dtype=bool)
        if has_warm and np.any(previous_accepted&(result['accepted_iteration']==0)):
            warm_metrics=certify_blocks_device(self.full_problems,self._full_assembled,
                warm_x.ravel(),warm_y.ravel(),cp=cp,allow_box_dual=False,
                require_direct_dual=bool(self.near_equality_plans or self.second_forest or self.fix_singleton_equalities))
            preserved_warm=(previous_accepted&(result['accepted_iteration']==0)
                &np.array([m['certificate_passed'] for m in warm_metrics],dtype=bool))
            select=cp.asarray(preserved_warm)[:,None]
            result['x']=cp.where(select,warm_x,result['x'])
            result['y']=cp.where(select,warm_y,result['y'])
            box&=~preserved_warm
        result['y']=cp.where(cp.asarray(box)[:,None],0.,result['y'])
        verified=certify_blocks_device(self.full_problems,self._full_assembled,
            result['x'].ravel(),result['y'].ravel(),cp=cp,allow_box_dual=False,
            require_direct_dual=bool(self.near_equality_plans or self.second_forest or self.fix_singleton_equalities))
        passed=np.array([m['certificate_passed'] for m in verified],dtype=bool)
        result['accepted']=previous_accepted&passed
        result['accepted_iteration']=np.asarray(result['accepted_iteration']).copy()
        result['accepted_iteration'][~result['accepted']]=-1
        if np.any(previous_accepted&~passed):
            result['status']='returned_pair_certificate_failed'
        metrics=[]
        for i,row in enumerate(verified):
            row=dict(row,returned_pair_certificate_passed=bool(passed[i]))
            if box[i]:
                row['certificate_source']='analytic_box_bound'
                row['solver_dual_relative_kkt_gap']=previous_metrics[i].get(
                    'solver_dual_relative_kkt_gap')
            metrics.append(row)
        result['metrics']=metrics
        result['analytic_box_dual_materialized']=box.tolist()
        result['original_certified_warm_pair_preserved']=preserved_warm.tolist()
        result['returned_pair_certificates_without_alternate']=verified
        result['returned_pair_certificate_seconds']=time.perf_counter()-before

    def solve(self, *, initial_x=None, initial_y=None, **kwargs):
        self.factor._context()
        before = time.perf_counter()
        full_x = self._initial(initial_x, (self.batch, self.full_n))
        full_y = self._initial(initial_y, (self.batch, self.full_m))
        z = (self._forest_compression @ full_x.ravel()).reshape(self.batch, self.original_n)
        y = full_y[self._forest_batch, self._forest_kept_rows]
        self.cp.cuda.get_current_stream().synchronize()
        compression_seconds = time.perf_counter()-before
        self._forest_certificate_seconds = 0.
        result = super().solve(initial_x=z, initial_y=y, **kwargs)
        result['forest_x'], result['forest_y'] = result['x'], result['y']
        result['x'], result['y'] = self.expand_forest_device(result['forest_x'], result['forest_y'])
        result['allow_box_dual']=self.allow_box_dual
        result['intermediate_dual_scope']='forest_y and reduced_y are diagnostic iterates, not certified full-original output duals'
        if self.allow_box_dual:
            self._materialize_box_certificate(result,full_x,full_y,
                initial_x is not None or initial_y is not None)
        if getattr(self,'device_numeric_update_active',False):
            result['zero_face']['objective_offsets']=self._device_zero_face_objective_offsets.get().tolist()
            result['exact_equalities']['plan_summaries_describe_initial_layout_not_current_numeric_inputs']=True
            result['device_numeric_update']=dict(generation=self.numeric_update_generation,
                current_full_problem_sha256=self.problem_hashes,
                intermediate_host_snapshots_are_layout_only=True,
                current_device_numeric_buffers_used_for_all_certificates=True)
        self.cp.cuda.get_current_stream().synchronize()
        result['parent_zero_face_solve_seconds'] = result['total_seconds']
        result['total_seconds'] = time.perf_counter()-before
        result['forest_full_certificate_seconds_including_initial'] = self._forest_certificate_seconds
        result['forest_warm_compression_seconds'] = compression_seconds
        result['forest'] = dict(full_variables=self.full_n, full_rows=self.full_m,
            forest_variables=self.original_n, forest_rows=self.original_m,
            eliminated_equalities=[len(p.eliminated_rows) for p in self.forest_plans],
            equality_fingerprints=[p.equality_fingerprint for p in self.forest_plans],
            setup_timing=dict(self.forest_setup_timing,
                             device_forest_setup_seconds=self.forest_device_setup_seconds))
        result['scope'] = ('Experimental per-environment exact equality forest and zero-face setup; '
            'GPU IPM and both primal/dual postsolves; unchanged FULL ORIGINAL LP certificate; '
            'host setup/control/summary remains, no CPU LP calls.')
        return result
