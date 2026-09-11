"""Opt-in dictionary-independent FP64 batched primal-dual IPM prototype.

Mehrotra predictor/corrector with one cuDSS symbolic analysis and one numeric
factorization per iteration, reused for two Newton right-hand sides. All
numeric factorization/iteration arithmetic uses CUDA. CSR assembly, analysis
reordering, Python iteration control and certificate summaries remain on host.
This is not yet a production or fully device-resident dFBA backend.

No CPU optimizer, hidden fallback, objective alteration, row deletion, or
uncertified success. KKT regularization stabilizes the Newton SYSTEM only;
acceptance always tests the unchanged original LP. Exhaustion fails closed.
"""
import time

import numpy as np
from scipy.sparse import block_diag, bmat, csr_matrix, diags, eye, vstack

from .gpu_block_lp import assemble_blocks, certify_blocks_device
from .gpu_pdhg_corrector import _validated_problem
from .gpu_sparse_factor import UniformCudssFactor


def constraint_form(problem):
    """Equivalent Ex=b, Gx<=h; exact fixed bounds become equalities.

    Keeping both inequalities for x_j=constant would give an empty strict
    interior. No variable or original row is dropped. Invalid infinities fail.
    """
    a, rhs, lo, hi, c, neq = _validated_problem(problem)
    if np.isposinf(lo).any() or np.isneginf(hi).any():
        raise ValueError('Impossible infinite bounds')
    fixed = np.isfinite(lo) & (lo == hi)
    il = np.flatnonzero(np.isfinite(lo) & ~fixed)
    iu = np.flatnonzero(np.isfinite(hi) & ~fixed)
    identity = eye(len(c), format='csr')
    e = vstack((a[:neq], identity[fixed]), format='csr')
    g = vstack((a[neq:], -identity[il], identity[iu]), format='csr')
    return e, np.r_[rhs[:neq], lo[fixed]], g, np.r_[rhs[neq:], -lo[il], hi[iu]], fixed, il, iu


def uniform_kkt_pattern(matrices):
    """Full symmetric union pattern, including all regularization diagonals."""
    size = matrices[0].shape[0]
    keys = [np.repeat(np.arange(size, dtype=np.int64), np.diff(a.indptr))*size+a.indices
            for a in matrices]
    union = np.union1d(np.concatenate(keys), np.arange(size, dtype=np.int64)*(size+1))
    pattern = csr_matrix((np.ones(len(union)), (union//size, union % size)), shape=(size, size))
    values = np.zeros((len(matrices), len(union)))
    for i, (a, key) in enumerate(zip(matrices, keys)):
        values[i, np.searchsorted(union, key)] = a.data
    diagonal = np.searchsorted(union, np.arange(size, dtype=np.int64)*(size+1))
    return pattern, values, diagonal


class GpuBatchedIPM:
    name = 'experimental_uniform_cudss_predictor_corrector'
    _condense_bounds = False

    def __init__(self, problems, *, regularization=1e-9, equilibration_rounds=3,
                 newton_refinements=2, newton_relative_tolerance=1e-8,
                 matrix_type='symmetric', original_newton_target=False,
                 newton_krylov_iterations=0,krylov_coordinates='full',
                 globalized=False,forcing_eta=0.05,regularization_retries=0,
                 predictor_corrector=False,predictor_affine_fraction=1.,
                 ipm_initialization='legacy',regularization_schedule='fixed',
                 device_checked_solves=False,krylov_microkernels='none',factor_refinements=2,
                 equality_row_scaling=False,krylov_defer_lane_checks=False,
                 retain_internal_state=False,reuse_gmres_workspace=False,fused_solve_guards=False,
                 factor_layout='uniform',factor_ordering='default',factor_algorithm='default',factor_precision='float64',
                 newton_backend='augmented',centrality_corrections=0,step_policy='common',shared_factor_group_size=1,
                 certified_primal_extrapolation=False):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as device_csr
        before = time.perf_counter()
        if (type(centrality_corrections) is not int or not 0<=centrality_corrections<=4
                or (centrality_corrections and (not globalized or not predictor_corrector))):
            raise ValueError('Zero to four centrality corrections require globalized predictor-corrector')
        self.centrality_corrections=centrality_corrections
        if step_policy not in ('common','primal_dual') or (step_policy=='primal_dual' and
                (not globalized or centrality_corrections)):
            raise ValueError('Separate primal/dual steps require globalized mode without extra centrality correctors')
        self.step_policy=step_policy
        if type(certified_primal_extrapolation) is not bool or (certified_primal_extrapolation and not globalized):
            raise ValueError('Certified primal extrapolation requires globalized mode')
        self.certified_primal_extrapolation=certified_primal_extrapolation
        if (type(shared_factor_group_size) is not int or shared_factor_group_size<1 or
            (shared_factor_group_size>1 and (not self._condense_bounds or not globalized or krylov_coordinates!='full'
                or newton_krylov_iterations<1 or newton_backend!='augmented' or fused_solve_guards
                or factor_precision!='float64' or factor_ordering!='default' or factor_algorithm!='default'))):
            raise ValueError('Shared factors require full original globalized Krylov without compound factor modes')
        self.shared_factor_group_size=shared_factor_group_size
        self._shared_newton=None;self._shared_factor_enabled=False
        if newton_backend not in ('augmented','dual_schur'):raise ValueError('Invalid Newton backend')
        if newton_backend=='dual_schur' and (not self._condense_bounds or not globalized
                or krylov_coordinates!='full' or factor_precision!='float64' or fused_solve_guards
                or factor_ordering!='default' or factor_algorithm!='default' or equality_row_scaling):
            raise ValueError('Dual Schur requires globalized FP64 bound condensation and full Krylov without compound options')
        self.newton_backend=newton_backend;self._dual_schur=None
        if type(fused_solve_guards) is not bool or (fused_solve_guards and not device_checked_solves):
            raise ValueError('Fused solve guards require device_checked_solves')
        self.fused_solve_guards=fused_solve_guards
        if factor_layout not in ('uniform','block_diagonal'):raise ValueError('Invalid factor layout')
        self.factor_layout=factor_layout
        if factor_ordering not in ('default','amd') or factor_algorithm not in ('default','alternate'):
            raise ValueError('Invalid factor ordering or algorithm')
        self.factor_ordering=factor_ordering;self.factor_algorithm=factor_algorithm
        if factor_precision not in ('float64','float32'):raise ValueError('Invalid factor precision')
        if factor_precision=='float32' and not globalized:
            raise ValueError('FP32 factors require globalized original-Newton residual certification')
        self.factor_precision=factor_precision
        if type(factor_refinements) is not int or not 0 <= factor_refinements <= 2:
            raise ValueError('cuDSS refinement budget must be an integer from 0 to 2')
        self.factor_refinements=factor_refinements
        if type(equality_row_scaling) is not bool:
            raise ValueError('equality_row_scaling must be boolean')
        self.equality_row_scaling=equality_row_scaling
        if type(krylov_defer_lane_checks) is not bool:
            raise ValueError('krylov_defer_lane_checks must be boolean')
        self.krylov_defer_lane_checks=krylov_defer_lane_checks
        if type(retain_internal_state) is not bool:
            raise ValueError('retain_internal_state must be boolean')
        self.retain_internal_state=retain_internal_state
        self._last_internal_state=None
        if type(reuse_gmres_workspace) is not bool:
            raise ValueError('reuse_gmres_workspace must be boolean')
        self.reuse_gmres_workspace=reuse_gmres_workspace
        self._gmres_workspaces={}
        if not np.isfinite(regularization) or not 0 < regularization <= 1e-2:
            raise ValueError('Positive finite Newton regularization <=1e-2 required')
        if (type(equilibration_rounds) is not int or not 0 <= equilibration_rounds <= 8
                or type(newton_refinements) is not int or not 0 <= newton_refinements <= 5):
            raise ValueError('Invalid bounded equilibration/refinement budget')
        if not np.isfinite(newton_relative_tolerance) or not 0 < newton_relative_tolerance <= 1e-6:
            raise ValueError('Strict finite Newton residual tolerance required')
        self.problems = [_validated_problem(p) for p in problems]
        if not self.problems:
            raise ValueError('Nonempty LP batch required')
        from .lp_trace import problem_hash
        self.problem_hashes = tuple(problem_hash(p) for p in self.problems)
        self.cp = cp
        self.regularization = float(regularization)
        self.equilibration_rounds = equilibration_rounds
        self.newton_refinements = newton_refinements
        self.newton_relative_tolerance = float(newton_relative_tolerance)
        if matrix_type not in ('symmetric','general'):
            raise ValueError('Select symmetric LDL or general GPU LU explicitly')
        self.matrix_type=matrix_type
        if type(globalized) is not bool:
            raise ValueError('globalized must be boolean')
        if not np.isfinite(forcing_eta) or not 0<forcing_eta<=0.1:
            raise ValueError('Forcing eta must be finite in (0,0.1]')
        self.globalized=globalized
        self.forcing_eta=float(forcing_eta)
        if (type(regularization_retries) is not int or not 0<=regularization_retries<=2
                or (regularization_retries and not globalized)):
            raise ValueError('Regularization retries require globalized mode and an integer budget 0..2')
        self.regularization_retries=regularization_retries
        if type(predictor_corrector) is not bool or (predictor_corrector and not globalized):
            raise ValueError('Safeguarded predictor_corrector requires globalized mode')
        self.predictor_corrector=predictor_corrector
        if (isinstance(predictor_affine_fraction,bool) or not np.isfinite(predictor_affine_fraction) or not 0.<predictor_affine_fraction<=1.
                or (not predictor_corrector and predictor_affine_fraction!=1.)):
            raise ValueError('Predictor fraction in (0,1] requires predictor_corrector when nondefault')
        self.predictor_affine_fraction=float(predictor_affine_fraction)
        if (ipm_initialization not in ('legacy','balanced')
                or (not globalized and ipm_initialization!='legacy')):
            raise ValueError('Balanced initialization is an explicit globalized IPM option')
        self.ipm_initialization=ipm_initialization
        if (regularization_schedule not in ('fixed','barrier')
                or (not globalized and regularization_schedule!='fixed')):
            raise ValueError('Barrier regularization is an explicit globalized IPM option')
        self.regularization_schedule=regularization_schedule
        if type(device_checked_solves) is not bool:
            raise ValueError('device_checked_solves must be boolean')
        if krylov_microkernels not in ('none','mgs','all'):
            raise ValueError('Choose explicit none, mgs or all Krylov microkernels')
        self.device_checked_solves=device_checked_solves
        self.krylov_microkernels=krylov_microkernels
        if globalized:
            original_newton_target=True
        if type(original_newton_target) is not bool:
            raise ValueError('original_newton_target must be boolean')
        self.original_newton_target=original_newton_target
        if (type(newton_krylov_iterations) is not int or not 0<=newton_krylov_iterations<=32
                or (newton_krylov_iterations and not original_newton_target)):
            raise ValueError('Krylov budget must be 0..32 and explicitly target original Newton equations')
        self.newton_krylov_iterations=newton_krylov_iterations
        if (krylov_coordinates not in ('full','condensed')
                or (krylov_coordinates=='condensed' and not self._condense_bounds)):
            raise ValueError('Condensed Krylov requires an explicitly bound-condensed solver')
        self.krylov_coordinates=krylov_coordinates
        self.batch = len(self.problems)
        self.m, self.n = self.problems[0][0].shape
        self.neq = self.problems[0][-1]
        forms = [constraint_form(p) for p in self.problems]
        if equality_row_scaling:
            from .gpu_ipm_equality_scaling import scale_equality_form
            scaled=[scale_equality_form(f) for f in forms]
            forms=[pair[0] for pair in scaled]
            self.equality_scale=cp.asarray(np.stack([pair[1] for pair in scaled]))
        if any(p[0].shape != (self.m, self.n) or p[-1] != self.neq for p in self.problems):
            raise ValueError('Shared LP dimensions/equality count required')
        if any(not np.array_equal(f[k], forms[0][k]) for f in forms for k in (4, 5, 6)):
            raise ValueError('Shared fixed/finite bound masks required within a uniform batch')
        self.ne, self.ng = forms[0][0].shape[0], forms[0][2].shape[0]
        if self.ng == 0:
            raise ValueError('This IPM prototype requires at least one inequality/bound')
        self.size = self.n+self.ne+self.ng
        self.q=self.m-self.neq
        self.condensed_size=self.n+self.ne+self.q
        linear_ng=self.q if self._condense_bounds else self.ng
        linear_size=self.n+self.ne+linear_ng
        self.fixed = cp.asarray(np.flatnonzero(forms[0][4]))
        self.il, self.iu = cp.asarray(forms[0][5]), cp.asarray(forms[0][6])
        self.e = device_csr(block_diag([f[0] for f in forms], format='csr'))
        self.g = device_csr(block_diag([f[2] for f in forms], format='csr'))
        self.et, self.gt = self.e.T.tocsr(), self.g.T.tocsr()
        self.b, self.h = [cp.asarray(np.stack([f[k] for f in forms])) for k in (1, 3)]
        self.c = cp.asarray(np.stack([p[4] for p in self.problems]))
        matrices = [bmat([[diags(np.ones(self.n)), f[0].T, f[2][:linear_ng].T],
                          [f[0], diags(-np.ones(self.ne)), None],
                          [f[2][:linear_ng], None, diags(-np.ones(linear_ng))]], format='csr') for f in forms]
        pattern, values, diagonal = uniform_kkt_pattern(matrices)
        self.values, self.diagonal = cp.asarray(values), cp.asarray(diagonal)
        self.kkt_rows = cp.asarray(np.repeat(np.arange(linear_size), np.diff(pattern.indptr)))
        self.kkt_columns = cp.asarray(pattern.indices)
        self.values[:, self.diagonal[:self.n]] = self.regularization
        self.values[:, self.diagonal[self.n:self.n+self.ne]] = -self.regularization
        packed = assemble_blocks(self.problems)
        self.assembled = (device_csr(packed[0]), *(cp.asarray(v) for v in packed[1:]))
        self.factor = UniformCudssFactor(pattern, batch_size=self.batch,matrix_type=matrix_type,
                                       refinement_steps=factor_refinements,execution_layout=factor_layout,
                                       ordering=factor_ordering,algorithm=factor_algorithm,precision=factor_precision)
        self._row_max = cp.ElementwiseKernel(
            'raw float64 values, raw int32 offsets, int32 rows, int32 nnz',
            'float64 maximum',
            'int row = i % rows; int batch = i / rows; double v = 0.; '
            'for (int k = offsets[row]; k < offsets[row+1]; ++k) '
            'v = fmax(v, fabs(values[batch * nnz + k])); maximum = v;',
            'uniform_ipm_csr_row_max')
        self.closed = False
        if newton_backend=='dual_schur':
            try:
                from .gpu_dual_schur import GpuDualSchurNewton
                self._dual_schur=GpuDualSchurNewton(self)
            except BaseException:
                self.close();raise
        if shared_factor_group_size>1:
            try:
                from .gpu_shared_newton import GpuSharedNewtonFactor
                self._shared_newton=GpuSharedNewtonFactor(self,shared_factor_group_size)
            except BaseException:
                self.close();raise
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds = time.perf_counter()-before

    def _equilibrate_kkt(self):
        """Invertible symmetric scaling DKD; never changes the LP or residuals."""
        cp = self.cp
        values = self.values.copy()
        scaling = cp.ones((self.batch, self.factor.n), dtype=cp.float64)
        for _ in range(self.equilibration_rounds):
            largest = self._row_max(values, self.factor.indptr, np.int32(self.factor.n),
                np.int32(self.factor.nnz), cp.empty_like(scaling))
            d = 1./cp.sqrt(cp.maximum(largest, 1e-300))
            scaling *= d
            values *= d[:, self.kkt_rows]*d[:, self.kkt_columns]
        return values, scaling

    def _factor_newton(self,ratio):
        # A changed preconditioner delta must refresh both signed blocks.
        # The original LP and its certificate remain untouched.
        self.values[:, self.diagonal[:self.n]]=self.regularization
        self.values[:, self.diagonal[self.n:self.n+self.ne]]=-self.regularization
        self.values[:, self.diagonal[self.n+self.ne:]]=-ratio
        values,scaling=self._equilibrate_kkt()
        self.factor.factor(values)
        return scaling

    def _solve_newton(self,rhs,ratio,scaling):
        return scaling*self._factor_solve((scaling*rhs)[:,:,None])[:,:,0]

    def _factor_solve(self,rhs):
        """Internal opt-in device validation; public factor API stays strict."""
        if getattr(self,'_shared_newton',None) is not None and self._shared_factor_enabled:
            return self._shared_newton.solve(rhs)
        if getattr(self,'device_checked_solves',False):
            if getattr(self,'fused_solve_guards',False):
                return self.factor._solve_device_checked_fused(rhs)
            return self.factor._solve_device_checked(rhs)
        return self.factor.solve(rhs)

    def _gmres_workspace(self,width):
        if not self.reuse_gmres_workspace:
            return None
        from .gpu_newton_krylov import GmresWorkspace
        key=(width,self.newton_krylov_iterations)
        if key not in self._gmres_workspaces:
            self._gmres_workspaces[key]=GmresWorkspace((self.batch,width),xp=self.cp,
                max_iterations=self.newton_krylov_iterations)
        return self._gmres_workspaces[key]

    def _krylov_direction(self,rhs,answer,ratio,scaling,need):
        """Full-coordinate K0 refinement; subclasses may avoid bound expansion."""
        from .gpu_newton_krylov import batched_gmres
        cp=self.cp
        weight=cp.ones_like(rhs)
        for first,last in ((0,self.n),(self.n,self.n+self.ne),
                           (self.n+self.ne,self.size)):
            if first<last:
                weight[:,first:last]=1./cp.maximum(1.,
                    cp.max(cp.abs(rhs[:,first:last]),axis=1))[:,None]
        return batched_gmres(cp.where(need[:,None],weight*rhs,0.),
            cp.where(need[:,None],answer,0.),
            lambda v:weight*self._kkt_mv(v,ratio,regularized=False),
            lambda v:self._solve_newton(v/weight,ratio,scaling),
            self._direction_error,xp=cp,max_iterations=self.newton_krylov_iterations,
            tolerance=self.newton_relative_tolerance,
            microkernels=getattr(self,'krylov_microkernels','none'),
            defer_lane_checks=getattr(self,'krylov_defer_lane_checks',False),
            workspace=self._gmres_workspace(self.size) if getattr(self,'reuse_gmres_workspace',False) else None)

    def _kkt_mv(self, vector, ratio, *, regularized=True):
        dx, dy, dz = (vector[:, :self.n], vector[:, self.n:self.n+self.ne],
                      vector[:, self.n+self.ne:])
        reg=self.regularization if regularized else 0.
        return self.cp.concatenate((reg*dx+self._mv(self.et, dy, self.n)
            +self._mv(self.gt, dz, self.n), self._mv(self.e, dx, self.ne)-reg*dy,
            self._mv(self.g, dx, self.ng)-ratio*dz), axis=1)

    def _direction_error(self,rhs,residual):
        cp=self.cp
        errors=[]
        for start,end in ((0,self.n),(self.n,self.n+self.ne),(self.n+self.ne,self.size)):
            if start<end:
                errors.append(cp.max(cp.abs(residual[:,start:end]),axis=1)
                    /cp.maximum(1.,cp.max(cp.abs(rhs[:,start:end]),axis=1)))
        return cp.max(cp.stack(errors),axis=0)

    def _mv(self, matrix, value, columns):
        return (matrix @ value.ravel()).reshape(self.batch, columns)

    def _row_dual(self, y, z):
        native_y=(y*self.equality_scale if getattr(self,'equality_row_scaling',False) else y)
        return self.cp.concatenate((-native_y[:, :self.neq], -z[:, :self.m-self.neq]), axis=1)

    def certificate(self, x, y):
        return certify_blocks_device(self.problems, self.assembled, x.ravel(), y.ravel(), cp=self.cp)

    def _initial(self, supplied, shape):
        cp = self.cp
        if supplied is None:
            return cp.zeros(shape, dtype=cp.float64)
        # This interface requires device arrays, e.g. CuPy.from_dlpack(Torch).
        if (not isinstance(supplied, cp.ndarray) or supplied.shape != shape
                or supplied.device.id != self.factor.device or supplied.dtype != cp.float64
                or not bool(cp.isfinite(supplied).all())):
            raise ValueError('Finite FP64 CUDA warm start with the exact shape required')
        return supplied.copy()

    def export_internal_state(self, *, environment_ids, stage, step):
        from .gpu_ipm_warm_state import export_internal_state
        return export_internal_state(self, environment_ids=environment_ids, stage=stage, step=step)

    def solve(self, *, initial_x=None, initial_y=None, iterations=60, check_interval=1,
              capture_failure=False,internal_warm_start=None,factor_reuse_interval=1):
        if self.closed:
            raise RuntimeError('Closed IPM workspace')
        # Cold solves retain independent factors; only causally bound warm
        # states enable the experimental cross-environment preconditioner.
        self._shared_factor_enabled=(self._shared_newton is not None and internal_warm_start is not None)
        if self.globalized:
            from .gpu_globalized_ipm import solve_globalized_ipm
            return solve_globalized_ipm(self,initial_x=initial_x,initial_y=initial_y,
                iterations=iterations,check_interval=check_interval,capture_failure=capture_failure,
                regularization_retries=self.regularization_retries,
                predictor_corrector=self.predictor_corrector,
                predictor_affine_fraction=self.predictor_affine_fraction,
                ipm_initialization=self.ipm_initialization,
                regularization_schedule=self.regularization_schedule,
                internal_warm_start=internal_warm_start,factor_reuse_interval=factor_reuse_interval)
        if type(factor_reuse_interval) is not int or factor_reuse_interval != 1:
            raise ValueError('Factor reuse requires globalized condensed IPM')
        if internal_warm_start is not None or self.retain_internal_state:
            raise ValueError('Internal state retention/transfer requires globalized IPM')
        if (type(iterations) is not int or iterations < 0 or type(check_interval) is not int
                or check_interval < 1):
            raise ValueError('Invalid iteration/check budget')
        cp = self.cp
        if type(capture_failure) is not bool:
            raise ValueError('capture_failure must be boolean')
        self.failure_snapshot = None
        before = time.perf_counter()
        x = self._initial(initial_x, (self.batch, self.n))
        original_y = self._initial(initial_y, (self.batch, self.m))
        y = cp.zeros((self.batch, self.ne), dtype=cp.float64)
        y[:, :self.neq] = -original_y[:, :self.neq]
        z = cp.ones((self.batch, self.ng), dtype=cp.float64)
        z[:, :self.m-self.neq] = cp.maximum(1., -original_y[:, self.neq:])
        reduced = self.c-self._mv(self.assembled[0].T, original_y, self.n)
        y[:, self.neq:] = -reduced[:, self.fixed]
        if self.equality_row_scaling:
            y /= self.equality_scale
        k = self.m-self.neq
        z[:, k:k+len(self.il)] = cp.maximum(1., reduced[:, self.il])
        z[:, k+len(self.il):] = cp.maximum(1., -reduced[:, self.iu])
        s = cp.maximum(1., self.h-self._mv(self.g, x, self.ng))
        metrics = self.certificate(x, original_y)
        accepted = np.array([r['certificate_passed'] for r in metrics])
        done = cp.asarray(accepted)
        accepted_iteration = np.where(accepted, 0, -1)
        # Freeze the certified pair separately: initialization may interiorize
        # multipliers, so it must not replace a valid caller-supplied solution.
        result_x, result_y = x.copy(), original_y.copy()
        checkpoints = [dict(iteration=0, accepted=int(accepted.sum()), metrics=list(metrics))]
        factor_seconds = solve_seconds = certificate_seconds = 0.
        completed = 0
        start_factor_count, start_solve_count = self.factor.factor_count, self.factor.solve_count
        status = 'iteration_limit'
        newton_history = []
        krylov_history = []
        krylov_termination_history = []
        def step(value, delta):
            ratio = cp.where(delta < 0., -value/cp.minimum(delta, -1e-300), cp.inf)
            return cp.minimum(1., cp.min(ratio, axis=1))
        for iteration in range(1, iterations+1):
            if accepted.all():
                break
            rd = self.c+self._mv(self.et, y, self.n)+self._mv(self.gt, z, self.n)
            rp = self._mv(self.e, x, self.ne)-self.b
            rg = self._mv(self.g, x, self.ng)+s-self.h
            mu = cp.sum(s*z, axis=1)/self.ng
            stamp = time.perf_counter()
            scaling=self._factor_newton(s/z)
            cp.cuda.get_current_stream().synchronize()
            factor_seconds += time.perf_counter()-stamp
            last_linear = {}
            def direction(rc):
                rhs = cp.concatenate((-rd, -rp, rc/z-rg), axis=1)
                answer = self._solve_newton(rhs,s/z,scaling)
                def measure(value):
                    residual=rhs-self._kkt_mv(value,s/z,regularized=not self.original_newton_target)
                    return residual,self._direction_error(rhs,residual)
                residual,error=measure(answer)
                # Retain the best iterate independently for each environment.
                # Internal cuDSS refinement is not assumed to be monotone.
                for _ in range(self.newton_refinements):
                    need=~done & (error>self.newton_relative_tolerance)
                    if not bool(cp.any(need)):break
                    trial=answer+self._solve_newton(cp.where(need[:,None],residual,0.),s/z,scaling)
                    trial_residual,trial_error=measure(trial)
                    better=need & cp.isfinite(trial_error) & (trial_error<error)
                    answer=cp.where(better[:,None],trial,answer)
                    residual=cp.where(better[:,None],trial_residual,residual)
                    error=cp.where(better,trial_error,error)
                    if not bool(cp.any(better)):break
                if self.newton_krylov_iterations:
                    # K_delta is only a preconditioner. Minimize and recheck
                    # the unregularized K_0 residual, preserving its blocks.
                    need=~done & (error>self.newton_relative_tolerance)
                    if bool(cp.any(need)):
                        improved,diagnostic=self._krylov_direction(rhs,answer,s/z,scaling,need)
                        improved_residual,improved_error=measure(improved)
                        better=need & cp.isfinite(improved_error) & (improved_error<error)
                        answer=cp.where(better[:,None],improved,answer)
                        residual=cp.where(better[:,None],improved_residual,residual)
                        error=cp.where(better,improved_error,error)
                        krylov_history.append(diagnostic['iterations'])
                        krylov_termination_history.append(diagnostic['termination'])
                    else:
                        krylov_history.append(cp.zeros(self.batch,dtype=cp.int64))
                        krylov_termination_history.append(cp.zeros(self.batch,dtype=cp.int32))
                if capture_failure:
                    last_linear.update(rhs=rhs, answer=answer, residual=residual)
                dx, dy, dz = (answer[:, :self.n], answer[:, self.n:self.n+self.ne],
                              answer[:, self.n+self.ne:])
                ds = -rg-self._mv(self.g, dx, self.ng)
                return dx, dy, dz, ds, error
            stamp = time.perf_counter()
            dx_a, dy_a, dz_a, ds_a, error_a = direction(s*z)
            affine_reliable=cp.isfinite(error_a) & (error_a<=self.newton_relative_tolerance)
            if not bool(cp.all(done | affine_reliable)):
                cp.cuda.get_current_stream().synchronize()
                solve_seconds+=time.perf_counter()-stamp
                status='unreliable_affine_direction'
                newton_history.append(cp.stack((mu,cp.zeros_like(mu),cp.zeros_like(mu),error_a,error_a),axis=1))
                if capture_failure:
                    self.failure_snapshot={k:v.copy() for k,v in dict(last_linear,
                        kkt_values=self.values,scaling=scaling,x=x,y=y,z=z,s=s,rd=rd,rp=rp,rg=rg).items()}
                break
            ap, ad = step(s, ds_a), step(z, dz_a)
            mu_aff = cp.sum((s+ap[:, None]*ds_a)*(z+ad[:, None]*dz_a), axis=1)/self.ng
            sigma = cp.clip(mu_aff/mu, 0., 1.)**3
            dx, dy, dz, ds, error = direction(s*z+ds_a*dz_a-sigma[:, None]*mu[:, None])
            ap, ad = .995*step(s, ds), .995*step(z, dz)
            newton_history.append(cp.stack((mu, ap, ad, error_a, error), axis=1))
            reliable = (cp.isfinite(error_a) & cp.isfinite(error)
                & (error_a <= self.newton_relative_tolerance) & (error <= self.newton_relative_tolerance))
            if not bool(cp.all(done | reliable)):
                # Do not advance an environment with an unreliable direction.
                # This prototype stops the batch; accepted pairs stay frozen.
                cp.cuda.get_current_stream().synchronize()
                solve_seconds += time.perf_counter()-stamp
                status = 'unreliable_newton_direction'
                if capture_failure:
                    self.failure_snapshot = {k: v.copy() for k, v in dict(last_linear,
                        kkt_values=self.values, scaling=scaling, x=x, y=y, z=z, s=s,
                        rd=rd, rp=rp, rg=rg).items()}
                break
            active = ~done[:, None]
            x = cp.where(active, x+ap[:, None]*dx, x)
            s = cp.where(active, s+ap[:, None]*ds, s)
            y = cp.where(active, y+ad[:, None]*dy, y)
            z = cp.where(active, z+ad[:, None]*dz, z)
            cp.cuda.get_current_stream().synchronize()
            solve_seconds += time.perf_counter()-stamp
            completed = iteration
            if not bool(cp.isfinite(x).all() & cp.isfinite(y).all() & cp.isfinite(z).all()
                        & cp.isfinite(s).all() & (s > 0.).all() & (z > 0.).all()):
                status = 'nonfinite_or_noninterior_iterate'
                break
            if iteration % check_interval == 0 or iteration == iterations:
                stamp = time.perf_counter()
                candidate_y = self._row_dual(y, z)
                latest = self.certificate(x, candidate_y)
                newly = ~accepted & np.array([r['certificate_passed'] for r in latest])
                select = cp.asarray(~accepted)[:, None]
                result_x = cp.where(select, x, result_x)
                result_y = cp.where(select, candidate_y, result_y)
                for i in range(self.batch):
                    if not accepted[i]:
                        metrics[i] = latest[i]
                accepted |= newly
                accepted_iteration[newly] = iteration
                done = cp.asarray(accepted)
                checkpoints.append(dict(iteration=iteration, accepted=int(accepted.sum()),
                                        metrics=list(metrics)))
                certificate_seconds += time.perf_counter()-stamp
        if accepted.all():
            status = 'certified'
        return dict(x=result_x, y=result_y, metrics=metrics, status=status,
            accepted=accepted, accepted_iteration=accepted_iteration, iterations=completed,
            checkpoints=checkpoints, cpu_lp_calls=0, analysis_count=self.factor.analysis_count,
            newton_diagnostics=(cp.stack(newton_history).get().tolist() if newton_history else []),
            newton_diagnostic_columns=['mu', 'primal_step', 'dual_step', 'affine_relative_residual',
                                      'corrector_relative_residual'],
            equilibration_rounds=self.equilibration_rounds, newton_refinements=self.newton_refinements,
            newton_relative_tolerance=self.newton_relative_tolerance,
            matrix_type=self.matrix_type,
            regularization=self.regularization,
            original_newton_target=self.original_newton_target,
            newton_krylov_iterations=self.newton_krylov_iterations,
            krylov_coordinates=self.krylov_coordinates,
            krylov_iterations_by_direction=(cp.stack(krylov_history).get().tolist() if krylov_history else []),
            krylov_termination_by_direction=(cp.stack(krylov_termination_history).get().tolist()
                                             if krylov_termination_history else []),
            krylov_termination_codes={0:'converged_or_not_needed',1:'iteration_limit',
                                      2:'breakdown',3:'nonfinite'},
            newton_error_definition='maximum of per-block infinity residual / max(1, block RHS infinity)',
            factor_count=self.factor.factor_count-start_factor_count,
            solve_count=self.factor.solve_count-start_solve_count,
            factor_seconds=factor_seconds, triangular_solve_and_update_seconds=solve_seconds,
            certificate_seconds=certificate_seconds, total_seconds=time.perf_counter()-before,
            setup_seconds=self.setup_seconds, kkt_dimension=self.size, kkt_nnz=self.factor.nnz,
            factored_dimension=self.factor.n,
            scope='FP64 GPU numeric iterations; host setup/control/summary; not production dFBA')

    def close(self):
        if not self.closed:
            try:
                if getattr(self,'_dual_schur',None) is not None:self._dual_schur.close()
                if getattr(self,'_shared_newton',None) is not None:self._shared_newton.close()
            finally:
                self.factor.close()
                self.closed = True

    @property
    def numerical_factor(self):
        if getattr(self,'_shared_newton',None) is not None and self._shared_factor_enabled:return self._shared_newton.factor
        return self._dual_schur.factor if getattr(self,'_dual_schur',None) is not None else self.factor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
