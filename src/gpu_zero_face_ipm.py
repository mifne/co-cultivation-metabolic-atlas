"""Exact zero-face setup and GPU-only primal/dual postsolve around IPM.

No model bounds are edited: reduced working problems are copies, justified by
stored zero-face witnesses. Every accepted result uses the unchanged ORIGINAL
LP certificate. Host proof construction is setup, not a CPU LP fallback.
"""
import time

import numpy as np
from .gpu_condensed_ipm import GpuCondensedBatchedIPM
from .gpu_block_lp import assemble_blocks,certify_blocks_device
from .lp_zero_face import ZeroFaceReduction


_LIFT_SOURCE=r'''
extern "C" __global__ void zero_face_dual_lift(
    const int* env_offsets,const int* witness_rows,const int* orientations,
    const int* new_offsets,const int* new_columns,const double* new_values,
    const int* row_offsets,const int* row_columns,const double* row_values,
    double* y,double* q) {
    const int env=blockIdx.x, lane=threadIdx.x;
    for (int w=env_offsets[env+1]-1;w>=env_offsets[env];--w) {
        double delta=0.;
        if(orientations[w]==0) {
            // A singleton a*x=0 with zero-crossing bounds fixes an interior
            // variable. Its equality multiplier is signed, not one-sided.
            const int k=new_offsets[w];
            if(lane==0)delta=q[new_columns[k]]/new_values[k];
            delta=__shfl_sync(0xffffffff,delta,0);
        } else {
            double maximum=0.;
            for (int k=new_offsets[w]+lane;k<new_offsets[w+1];k+=32)
                maximum=fmax(maximum,orientations[w]*q[new_columns[k]]/new_values[k]);
            for (int offset=16;offset>0;offset>>=1)
                maximum=fmax(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
            delta=orientations[w]*__shfl_sync(0xffffffff,maximum,0);
        }
        int row=witness_rows[w];
        if(lane==0)y[row]+=delta;
        for(int k=row_offsets[row]+lane;k<row_offsets[row+1];k+=32)
            q[row_columns[k]]-=delta*row_values[k];
        __syncwarp();
    }
}
'''


class ZeroFaceGpuBatchedIPM(GpuCondensedBatchedIPM):
    def __init__(self,problems,*,exact_equalities=False,bounded_near_equality=None,
                 second_forest=False,fix_singleton_equalities=False,reuse_equality_proofs=False,**kwargs):
        before=time.perf_counter()
        if type(exact_equalities) is not bool:
            raise ValueError('exact_equalities must be boolean')
        self.exact_equalities=exact_equalities
        if type(reuse_equality_proofs) is not bool:raise ValueError('Boolean equality proof reuse required')
        self.reuse_equality_proofs=reuse_equality_proofs
        if type(second_forest) is not bool:
            raise ValueError('second_forest must be boolean')
        self.second_forest=second_forest
        if type(fix_singleton_equalities) is not bool:
            raise ValueError('fix_singleton_equalities must be boolean')
        self.fix_singleton_equalities=fix_singleton_equalities
        self.face_plan_reuse_count=0
        self.face_plans=[]
        for p in problems:
            plan=None
            if reuse_equality_proofs and self.face_plans:
                try:
                    # Existing rebind authenticates the prior proof, checks
                    # CURRENT equality/RHS and zero signs, and reconstructs
                    # independent current bounds/witness snapshots. No solved
                    # primal/dual or unchecked numeric data is shared.
                    plan=self.face_plans[-1].rebind(p)
                except ValueError:
                    # Compatibility failure requires fresh proof discovery,
                    # not a relaxed reduction or a CPU optimization fallback.
                    pass
            if plan is None:
                plan=ZeroFaceReduction(p,fix_singleton_equalities=fix_singleton_equalities)
            else:
                self.face_plan_reuse_count+=1
            self.face_plans.append(plan)
        if not self.face_plans:raise ValueError('Nonempty LP batch required')
        self.original_problems=[p.original for p in self.face_plans]
        if any(not len(p.columns) for p in self.face_plans):
            raise ValueError('Zero-variable reductions require an analytic solver, not this IPM prototype')
        self.original_m,self.original_n=self.original_problems[0][0].shape
        if any(p[0].shape!=(self.original_m,self.original_n) for p in self.original_problems):
            raise ValueError('Common original LP dimensions required')
        if any(p[-1]!=self.original_problems[0][-1] for p in self.original_problems):
            raise ValueError('Common original LP equality count required')
        reduced=[p.reduced for p in self.face_plans]
        if any(p[0].shape!=reduced[0][0].shape or p[-1]!=reduced[0][-1] for p in reduced):
            raise ValueError('Regroup LPs with different exact reduced dimensions before GPU batching')
        self.near_equality_plans=[]
        self.near_equality_setup_seconds=0.
        if bounded_near_equality is not None:
            from .lp_bounded_near_equality import BoundedNearEqualityReduction
            stamp=time.perf_counter()
            # Candidate IDs are proposals only. Every stored coefficient, RHS
            # and finite bound is re-proved for EACH incoming working LP.
            self.near_equality_plans=[BoundedNearEqualityReduction(
                p,bounded_near_equality,max_defect=1e-12) for p in reduced]
            reduced=[p.reduced for p in self.near_equality_plans]
            self.near_equality_setup_seconds=time.perf_counter()-stamp
        self.secondary_forest_plans=[]
        self.secondary_forest_reductions=[]
        if second_forest:
            from .lp_equality_reduction import HomogeneousEqualityReduction
            if reuse_equality_proofs:
                from .lp_equality_reduction import prepare_independent_forest_plans
                self.secondary_forest_plans=list(prepare_independent_forest_plans(reduced))
            else:
                self.secondary_forest_plans=[HomogeneousEqualityReduction.from_problem(p) for p in reduced]
            self.secondary_forest_reductions=[plan.reduce(p)
                for plan,p in zip(self.secondary_forest_plans,reduced)]
            reduced=[r.problem for r in self.secondary_forest_reductions]
        self.equality_plans=[]
        self.equality_setup_seconds=0.
        if exact_equalities:
            from .lp_exact_equalities import ExactEqualityReduction,equality_fingerprint
            stamp=time.perf_counter()
            # Reuse is added only through the plan's equality/RHS identity
            # contract; different state data must never inherit an unchecked map.
            verified={}
            for problem in reduced:
                key=equality_fingerprint(problem)
                plan=ExactEqualityReduction(problem,reuse_from=verified.get(key))
                verified.setdefault(key,plan)
                self.equality_plans.append(plan)
            reduced=[plan.reduced for plan in self.equality_plans]
            self.equality_setup_seconds=time.perf_counter()-stamp
            if any(p[0].shape!=reduced[0][0].shape or p[-1]!=reduced[0][-1] for p in reduced):
                raise ValueError('Regroup LPs with different exactly verified equality reductions')
        super().__init__(reduced,**kwargs)
        try:
            self._setup_postsolve(before)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f'Postsolve setup cleanup also failed: {cleanup}')
            raise

    def _setup_postsolve(self,before):
        cp=self.cp
        from cupyx.scipy.sparse import csr_matrix
        self.problem_hashes=tuple(p.original_hash for p in self.face_plans)
        packed=assemble_blocks(self.original_problems)
        self.original_assembled=(csr_matrix(packed[0]),*(cp.asarray(v) for v in packed[1:]))
        self.original_at=self.original_assembled[0].T.tocsr()
        self.columns=cp.asarray(np.stack([p.columns for p in self.face_plans]))
        self.rows=cp.asarray(np.stack([p.rows for p in self.face_plans]))
        if self.secondary_forest_plans:
            from .gpu_forest_map import GpuForestMap
            self.secondary_forest_map=GpuForestMap(self.secondary_forest_plans,
                self.secondary_forest_reductions,cp=cp,reuse_equality_proofs=self.reuse_equality_proofs)
        if self.near_equality_plans:
            for plan in self.near_equality_plans:
                plan.validate_integrity()
            self.near_equality_keep=cp.asarray(np.stack([p.rows for p in self.near_equality_plans]))
        if self.equality_plans:
            from scipy.sparse import block_diag
            for plan in self.equality_plans:
                plan.validate_integrity()
            self.equality_keep=cp.asarray(np.stack([p.rows for p in self.equality_plans]))
            self.equality_dual_compression=csr_matrix(block_diag(
                [p.dual_compression for p in self.equality_plans],format='csr'))
        duplicate_rows=[];duplicate_representatives=[];duplicate_signs=[]
        for i,p in enumerate(self.face_plans):
            for d in p.duplicate_equalities:
                duplicate_rows.append(i*self.original_m+d.row)
                duplicate_representatives.append(i*self.original_m+d.representative_row)
                duplicate_signs.append(d.sign)
        self.duplicate_rows=cp.asarray(duplicate_rows,dtype=cp.int64)
        self.duplicate_representatives=cp.asarray(duplicate_representatives,dtype=cp.int64)
        self.duplicate_signs=cp.asarray(duplicate_signs,dtype=cp.float64)
        self.batch_index=cp.arange(self.batch)[:,None]
        self.fixed_template=cp.zeros((self.batch,self.original_n),dtype=cp.float64)
        env_offsets=[0];wrows=[];orientation=[];noffsets=[0];ncols=[];nvalues=[]
        for i,p in enumerate(self.face_plans):
            self.fixed_template[i,cp.asarray(p.fixed_columns)]=cp.asarray(p.fixed_values)
            for w in p.witnesses:
                wrows.append(i*self.original_m+w.row)
                if w.orientation not in ('min','max','equality'):
                    raise ValueError('Unknown zero-face witness orientation')
                if w.orientation=='equality' and len(w.newly_fixed_columns)!=1:
                    raise ValueError('Singleton equality witness must fix exactly one column')
                orientation.append({'min':-1,'max':1,'equality':0}[w.orientation])
                ncols.extend(i*self.original_n+np.asarray(w.newly_fixed_columns))
                nvalues.extend(w.newly_fixed_coefficients)
                noffsets.append(len(ncols))
            env_offsets.append(len(wrows))
        self.proof_arrays=[cp.asarray(v,dtype=cp.int32) for v in
                           (env_offsets,wrows,orientation,noffsets,ncols)]
        self.proof_values=cp.asarray(nvalues,dtype=cp.float64)
        self.lift_kernel=cp.RawKernel(_LIFT_SOURCE,'zero_face_dual_lift')
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds=time.perf_counter()-before

    def lift_device(self,x,y):
        self.factor._context()
        cp=self.cp
        for value,shape in ((x,(self.batch,self.n)),(y,(self.batch,self.m))):
            if (not isinstance(value,cp.ndarray) or value.shape!=shape
                    or value.dtype!=cp.float64 or value.device.id!=self.factor.device):
                raise ValueError('Exact-shape FP64 arrays on the bound CUDA device required for postsolve')
        # Do not introduce a full-device finite reduction/synchronization on
        # every certificate. Nonfinite proposals remain uncertified by the
        # unchanged original-LP finite gate after this algebraic mapping.
        full_y=cp.zeros((self.batch,self.original_m),dtype=cp.float64)
        if self.equality_plans:
            # Zero multipliers on the proved-redundant rows. This is an actual
            # dual lift, not just reporting a certificate for a different pair.
            face_y=cp.zeros((self.batch,self.equality_plans[0].original[0].shape[0]),dtype=cp.float64)
            face_y[self.batch_index,self.equality_keep]=y
        else:
            face_y=y
        if self.secondary_forest_plans:
            x,face_y=self.secondary_forest_map.expand(x,face_y)
        if self.near_equality_plans:
            lifted_y=cp.zeros(self.rows.shape,dtype=cp.float64)
            lifted_y[self.batch_index,self.near_equality_keep]=face_y
            face_y=lifted_y
        full_x=self.fixed_template.copy()
        full_x[self.batch_index,self.columns]=x
        full_y[self.batch_index,self.rows]=face_y
        a=self.original_assembled[0]
        q=self.original_assembled[5]-self.original_at@full_y.ravel()
        self.lift_kernel((self.batch,),(32,),(*self.proof_arrays,self.proof_values,
            a.indptr,a.indices,a.data,full_y,q))
        return full_x,full_y

    def certificate(self,x,y):
        full_x,full_y=self.lift_device(x,y)
        return certify_blocks_device(self.original_problems,self.original_assembled,
                                      full_x.ravel(),full_y.ravel(),cp=self.cp,
                                      require_direct_dual=bool(self.near_equality_plans or self.second_forest or self.fix_singleton_equalities))

    def solve(self,*,initial_x=None,initial_y=None,**kwargs):
        self.factor._context()
        before=time.perf_counter()
        full_x=self._initial(initial_x,(self.batch,self.original_n))
        full_y=self._initial(initial_y,(self.batch,self.original_m))
        # Preserve A^T y in the remaining variable space when a caller's dual
        # places weight on removed duplicate equalities. Representatives are
        # retained rows (never other removed duplicates), so one scatter-add
        # handles shared representatives without in-place dependency hazards.
        if self.duplicate_rows.size:
            self.cp.add.at(full_y.ravel(),self.duplicate_representatives,
                           self.duplicate_signs*full_y.ravel()[self.duplicate_rows])
        reduced_y=full_y[self.batch_index,self.rows]
        reduced_x=full_x[self.batch_index,self.columns]
        if self.near_equality_plans:
            reduced_y=reduced_y[self.batch_index,self.near_equality_keep]
        if self.secondary_forest_plans:
            reduced_x,reduced_y=self.secondary_forest_map.compress(reduced_x,reduced_y)
        if self.equality_plans:
            reduced_y=(self.equality_dual_compression@reduced_y.ravel()).reshape(self.batch,self.m)
        result=super().solve(initial_x=reduced_x,
                             initial_y=reduced_y,**kwargs)
        result['reduced_x'],result['reduced_y']=result['x'],result['y']
        result['x'],result['y']=self.lift_device(result['x'],result['y'])
        self.cp.cuda.get_current_stream().synchronize()
        result['core_solve_seconds']=result['total_seconds']
        result['total_seconds']=time.perf_counter()-before
        result['zero_face']=dict(original_variables=self.original_n,original_rows=self.original_m,
            fix_singleton_equalities=self.fix_singleton_equalities,
            singleton_equality_counts=[sum(w.orientation=='equality' for w in p.witnesses) for p in self.face_plans],
            skipped_small_singleton_counts=[len(p.skipped_small_singleton_rows) for p in self.face_plans],
            reduced_variables=self.n,reduced_rows=self.m,
            forced_zero_counts=[len(p.forced_zero_columns) for p in self.face_plans],
            explicit_fixed_counts=[len(p.explicit_fixed_columns) for p in self.face_plans],
            removed_row_counts=[len(p.original[1])-len(p.rows) for p in self.face_plans],
            removed_duplicate_counts=[len(p.removed_duplicate_rows) for p in self.face_plans],
            objective_offsets=[p.objective_offset for p in self.face_plans])
        result['exact_equalities']=dict(enabled=self.exact_equalities,
            host_setup_seconds=self.equality_setup_seconds,
            plans=[p.summary for p in self.equality_plans],
            scope='Only exact verified equality identities; CUDA warm-dual compression and zero-injection dual lift; original LP gates unchanged')
        result['bounded_near_equality']=dict(enabled=bool(self.near_equality_plans),
            host_setup_seconds=self.near_equality_setup_seconds,
            plans=[p.summary for p in self.near_equality_plans],
            scope='NOT an exactly equivalent LP reduction: bounded working relaxation only; removed dual zero-lift; all original rows and direct dual objective must pass')
        result['second_forest']=dict(enabled=self.second_forest,
            eliminated_rows=[len(p.eliminated_rows) for p in self.secondary_forest_plans],
            scope='Second algebraic two-term equality pass after zero-face; no small-slack fixing; original full-row/direct-dual gates')
        return result
