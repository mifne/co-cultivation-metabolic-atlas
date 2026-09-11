"""Fixed-layout current LP reductions on CUDA, never an LP solution cache.

The bridge still validates/packs current host inputs and hashes them. All
numeric reductions and certificate/operator updates are staged on GPU. The
first construction is explicit host work. A changed equality, zero sign or
CSR support rejects this route; a fresh GPU topology must then be built.
"""
import threading
import time
import weakref

import numpy as np

from .gpu_lp_numeric import DeviceLPBatch
from .gpu_pdhg_corrector import _validated_problem
from .lp_trace import problem_hash
from .lp_zero_face import _freeze_problem


def _pack(problems, xp):
    return DeviceLPBatch(xp.asarray(np.concatenate([p[0].data for p in problems])),
        *(xp.asarray(np.stack([p[i] for p in problems])) for i in (1,2,3,4)))


def _meta(array, xp):
    return (array.shape, array.dtype.str,
            array.device.id if xp.__name__=='cupy' else None,
            array.data.ptr if xp.__name__=='cupy' else array.__array_interface__['data'][0])


class _Owned:
    def __init__(self, xp):
        self.xp=xp
        self._owned=[]

    def own(self, name, value):
        array=self.xp.array(value, copy=True)
        snapshot=array.copy()
        setattr(self,name,array)
        self._owned.append((name,array,snapshot,_meta(array,self.xp)))
        return array

    def integrity_flags(self):
        flags=[]
        for name,array,snapshot,metadata in self._owned:
            if getattr(self,name) is not array or _meta(array,self.xp)!=metadata:
                raise ValueError('A compiled layout array was replaced or moved: '+name)
            flags.append(self.xp.array_equal(array,snapshot))
        return flags


class _Layout(_Owned):
    def __init__(self, problems, xp):
        super().__init__(xp)
        self.problems=tuple(_freeze_problem(_validated_problem(p)) for p in problems)
        if not self.problems: raise ValueError('Nonempty layout required')
        self.batch=len(self.problems)
        self.m,self.n=self.problems[0][0].shape
        self.neq=self.problems[0][-1]
        if any(p[0].shape!=(self.m,self.n) or p[-1]!=self.neq for p in self.problems):
            raise ValueError('Uniform batch dimensions/equality counts required')
        self.nnz=sum(p[0].nnz for p in self.problems)
        self._dimensions=(self.batch,self.m,self.n,self.neq,self.nnz)
        base=_pack(self.problems,xp)
        for name,array in zip(('data','rhs','lower','upper','c'),base.arrays()):
            self.own('base_'+name,array)
        offsets=np.cumsum([0,*[p[0].nnz for p in self.problems[:-1]]])
        equal_slots=np.concatenate([offset+np.arange(p[0].indptr[self.neq],dtype=np.int64)
                                    for offset,p in zip(offsets,self.problems)])
        self.own('equal_slots',equal_slots)

    def validate_metadata(self, value):
        if (self.batch,self.m,self.n,self.neq,self.nnz)!=self._dimensions:
            raise ValueError('Compiled layout dimensions were modified')
        if not isinstance(value,DeviceLPBatch): raise ValueError('DeviceLPBatch required')
        shapes=((self.nnz,),(self.batch,self.m),*((self.batch,self.n),)*3)
        for array,shape in zip(value.arrays(),shapes):
            if (not isinstance(array,self.xp.ndarray) or array.shape!=shape
                    or array.dtype!=self.xp.float64
                    or (self.xp.__name__=='cupy' and array.device.id!=self.base_data.device.id)):
                raise ValueError('Current LP arrays have wrong shape/dtype/device')

    def guards(self, value, *, zero_signs=False):
        self.validate_metadata(value)
        xp=self.xp
        flags=self.integrity_flags()
        flags.extend([xp.all(xp.isfinite(value.data)),xp.all(value.data!=0.),
            xp.all(xp.isfinite(value.rhs)),xp.all(xp.isfinite(value.c)),
            xp.all(xp.isfinite(value.lower)|xp.isneginf(value.lower)),
            xp.all(xp.isfinite(value.upper)|xp.isposinf(value.upper)),
            xp.all(value.lower<=value.upper),
            xp.array_equal(value.data[self.equal_slots],self.base_data[self.equal_slots]),
            xp.array_equal(value.rhs[:,:self.neq],self.base_rhs[:,:self.neq]),
            xp.array_equal(xp.isfinite(value.lower),xp.isfinite(self.base_lower)),
            xp.array_equal(xp.isfinite(value.upper),xp.isfinite(self.base_upper)),
            xp.array_equal(value.lower==value.upper,self.base_lower==self.base_upper),
            xp.all(xp.where(self.base_lower==self.base_upper,value.lower==self.base_lower,True))])
        if zero_signs:
            flags.extend([xp.array_equal(xp.sign(value.lower),xp.sign(self.base_lower)),
                          xp.array_equal(xp.sign(value.upper),xp.sign(self.base_upper))])
        return flags


def _forest_linear(inputs, outputs, plans):
    """Compile current inequality A*T sums, including structurally cancelled slots.

    Equality inputs are separately required unchanged, so their already
    proved transformed coefficients remain constants; no new roundoff-based
    equality or dependent-row discovery is performed during an update.
    """
    pointers=[0]; indices=[]; weights=[]; destinations=[]
    input_offset=output_offset=0
    for before,after,plan in zip(inputs,outputs,plans):
        a,out=before[0],after[0]
        for row in range(after[-1],out.shape[0]):
            old_row=int(plan.kept_rows[row])
            sums={}
            for slot in range(a.indptr[old_row],a.indptr[old_row+1]):
                col=int(a.indices[slot]); group=int(plan.original_to_reduced[col])
                sums.setdefault(group,[]).append((input_offset+slot,float(plan.weights[col])))
            stored={int(out.indices[k]):output_offset+k
                    for k in range(out.indptr[row],out.indptr[row+1])}
            if not set(stored).issubset(sums):
                raise ValueError('Stored forest inequality is outside the compiled source map')
            for group,terms in sorted(sums.items()):
                indices.extend(t[0] for t in terms); weights.extend(t[1] for t in terms)
                pointers.append(len(indices)); destinations.append(stored.get(group,-1))
        input_offset+=a.nnz; output_offset+=out.nnz
    return (np.array(pointers,dtype=np.int64),np.array(indices,dtype=np.int64),
            np.array(weights,dtype=np.float64),np.array(destinations,dtype=np.int64))


class _Forest(_Owned):
    def __init__(self, before, after, plans, xp):
        super().__init__(xp)
        from .gpu_segmented_linear import DeviceSegmentedLinear
        from .gpu_forest_numeric_bounds import DeviceForestBounds
        self.before,self.after=before,after
        pointers,indices,weights,destinations=_forest_linear(before.problems,after.problems,plans)
        self.linear=DeviceSegmentedLinear(pointers,indices,weights,before.nnz,cp=xp)
        self.bounds=DeviceForestBounds(plans,cp=xp)
        keep=destinations>=0
        self.own('selected',np.flatnonzero(keep))
        self.own('destinations',destinations[keep])
        self.own('cancelled',np.flatnonzero(~keep))
        self.own('rows',np.stack([p.kept_rows for p in plans]))
        self.own('lanes',np.arange(before.batch)[:,None])

    def stage(self, value):
        xp=self.xp
        self.before.validate_metadata(value)
        flags=self.integrity_flags()
        transformed=self.linear.apply(value.data)
        flags.extend([xp.all(xp.isfinite(transformed)),xp.all(transformed[self.cancelled]==0.),
                      xp.all(transformed[self.selected]!=0.)])
        data=self.after.base_data.copy()
        data[self.destinations]=transformed[self.selected]
        lo,hi,c,lw,uw,fw,valid=self.bounds.apply(value.lower,value.upper,value.c)
        flags.append(xp.all(valid))
        return DeviceLPBatch(data,value.rhs[self.lanes,self.rows],lo,hi,c),(lw,uw,fw),flags


def _selection_map(inputs,outputs,rows,columns):
    indices=[]; offset=0
    for before,after,rr,cc in zip(inputs,outputs,rows,columns):
        a,out=before[0],after[0]
        for r,original_row in enumerate(rr):
            source={int(a.indices[k]):offset+k for k in range(a.indptr[original_row],a.indptr[original_row+1])}
            for k in range(out.indptr[r],out.indptr[r+1]):
                col=int(cc[out.indices[k]])
                if col not in source: raise ValueError('Current selection layout lacks an original coefficient')
                indices.append(source[col])
        offset+=a.nnz
    return np.asarray(indices,dtype=np.int64)


class _RowColumn(_Owned):
    def __init__(self,before,after,rows,columns,xp,*,face_plans=None):
        super().__init__(xp)
        from .gpu_segmented_linear import DeviceSegmentedLinear
        self.before,self.after=before,after
        self.own('gather',_selection_map(before.problems,after.problems,rows,columns))
        self.own('rows',np.stack(rows));self.own('columns',np.stack(columns))
        self.own('lanes',np.arange(before.batch)[:,None])
        self.substitution=None;self.objective_offset=None
        if face_plans is not None:
            pointers=[0];indices=[];weights=[];offset=0;zero=[]
            cost_ptr=[0];cost_indices=[];cost_weights=[]
            for lane,(problem,plan) in enumerate(zip(before.problems,face_plans)):
                fixed=dict(zip(map(int,plan.fixed_columns),map(float,plan.fixed_values)))
                a=problem[0]
                for row in range(before.m):
                    for k in range(a.indptr[row],a.indptr[row+1]):
                        coefficient=fixed.get(int(a.indices[k]),0.)
                        if coefficient!=0.: indices.append(offset+k);weights.append(coefficient)
                    pointers.append(len(indices))
                zero.extend(lane*before.m+int(row) for row in plan.removed_zero_rows)
                for col,weight in fixed.items():
                    if weight!=0.:cost_indices.append(lane*before.n+col);cost_weights.append(weight)
                cost_ptr.append(len(cost_indices));offset+=a.nnz
            self.substitution=DeviceSegmentedLinear(np.asarray(pointers,dtype=np.int64),
                np.asarray(indices,dtype=np.int64),np.asarray(weights,dtype=np.float64),before.nnz,cp=xp)
            self.objective_offset=DeviceSegmentedLinear(np.asarray(cost_ptr,dtype=np.int64),
                np.asarray(cost_indices,dtype=np.int64),np.asarray(cost_weights,dtype=np.float64),
                before.batch*before.n,cp=xp)
            self.own('zero_rows',np.asarray(zero,dtype=np.int64))
            self.own('zero_equality',np.asarray(zero,dtype=np.int64)%before.m<before.neq)

    def stage(self,value):
        xp=self.xp
        self.before.validate_metadata(value)
        flags=self.integrity_flags();rhs=value.rhs
        objective_offsets=None
        if self.substitution is not None:
            product=self.substitution.apply(value.data).reshape(self.before.batch,self.before.m)
            adjusted=rhs-product
            flags.extend([xp.all(xp.isfinite(product)),xp.all(xp.isfinite(adjusted))])
            zero_values=adjusted.ravel()[self.zero_rows]
            flags.append(xp.all(xp.where(self.zero_equality,zero_values==0.,zero_values>=0.)))
            rhs=adjusted
            objective_offsets=self.objective_offset.apply(value.c.ravel())
            flags.append(xp.all(xp.isfinite(objective_offsets)))
        rhs=rhs[self.lanes,self.rows]
        # Input equality values/RHS and the fixed values were checked against
        # the original proof. Reuse that proof's numeric equality RHS exactly.
        rhs[:,:self.after.neq]=self.after.base_rhs[:,:self.after.neq]
        result=DeviceLPBatch(value.data[self.gather],rhs,
            value.lower[self.lanes,self.columns],value.upper[self.lanes,self.columns],
            value.c[self.lanes,self.columns])
        return result,flags,objective_offsets


class DeviceReductionPipeline:
    """GPU value transformation only; caller owns commit and LP certification."""
    def __init__(self,solver,cp=None):
        from .gpu_ipm_numeric_update import _existing_host_state
        self.cp=solver.cp if cp is None else cp
        old=_existing_host_state(solver)
        if solver.near_equality_plans or getattr(solver,'equality_row_scaling',False):
            raise ValueError('No approximate equality transform/row scaling in device update')
        self.host_states=dict(full=old.full_problems,forest=old.forest_problems,
            face=tuple(p.reduced for p in old.face_plans),
            secondary=tuple(r.problem for r in old.secondary_forest_reductions)
                if old.secondary_forest_reductions else tuple(p.reduced for p in old.face_plans),
            core=old.problems)
        self.layouts={name:_Layout(problems,self.cp) for name,problems in self.host_states.items()}
        self.first=_Forest(self.layouts['full'],self.layouts['forest'],solver.forest_plans,self.cp)
        self.face=_RowColumn(self.layouts['forest'],self.layouts['face'],
            [p.rows for p in old.face_plans],[p.columns for p in old.face_plans],self.cp,
            face_plans=old.face_plans)
        self.second=(_Forest(self.layouts['face'],self.layouts['secondary'],
            solver.secondary_forest_plans,self.cp) if old.secondary_forest_reductions else None)
        self.exact=(_RowColumn(self.layouts['secondary'],self.layouts['core'],
            [p.rows for p in old.equality_plans],
            [np.arange(self.layouts['secondary'].n) for p in old.equality_plans],self.cp)
            if old.equality_plans else None)

    def stage(self,full):
        flags=self.layouts['full'].guards(full)
        forest,first_witnesses,newflags=self.first.stage(full);flags.extend(newflags)
        flags.extend(self.layouts['forest'].guards(forest,zero_signs=True))
        face,newflags,offsets=self.face.stage(forest);flags.extend(newflags)
        flags.extend(self.layouts['face'].guards(face))
        secondary=face;second_witnesses=None
        if self.second is not None:
            secondary,second_witnesses,newflags=self.second.stage(face);flags.extend(newflags)
            flags.extend(self.layouts['secondary'].guards(secondary))
        core=secondary
        if self.exact is not None:
            core,newflags,_unused=self.exact.stage(secondary);flags.extend(newflags)
        flags.extend(self.layouts['core'].guards(core))
        return dict(full=full,forest=forest,face=face,secondary=secondary,core=core),\
            first_witnesses,second_witnesses,flags,offsets

    def auxiliary_static_targets(self):
        objects=[self.first.linear,self.first.bounds]
        if self.face.substitution is not None:
            objects.extend([self.face.substitution,self.face.objective_offset])
        if self.second is not None:objects.extend([self.second.linear,self.second.bounds])
        return [item for obj in objects for item in obj.static_targets]

    def preflight_flags(self):
        # Run before any gather/raw kernel: a corrupt index must never be
        # executed merely because a later commit would have been rejected.
        objects=[*self.layouts.values(),self.first,self.face]
        if self.second is not None:objects.append(self.second)
        if self.exact is not None:objects.append(self.exact)
        flags=[flag for obj in objects for flag in obj.integrity_flags()]
        auxiliary=[self.first.linear,self.first.bounds]
        if self.face.substitution is not None:
            auxiliary.extend([self.face.substitution,self.face.objective_offset])
        if self.second is not None:auxiliary.extend([self.second.linear,self.second.bounds])
        for obj in auxiliary:
            if callable(getattr(obj,'_context',None)):obj._context()
            if hasattr(obj,'_check_layout'):obj._check_layout()
            for _name,array,snapshot in obj.static_targets:
                flags.append(self.cp.array_equal(array,snapshot))
        return flags


class DeviceNumericUpdatePlan:
    """Transactional host-input bridge to fixed-layout GPU numeric updates.

    Current full host inputs are retained and hashed truthfully. Intermediate
    host LP snapshots become layout-only and are NOT silently reused by the
    older host numeric-update API. All solve/certificate numerical buffers
    use the staged CURRENT device values. No CPU LP is called.
    """
    def __init__(self,solver):
        from .gpu_forest_ipm import ForestGpuBatchedIPM
        from .gpu_ipm_device_payload import DeviceIPMPayloadBinder
        from .gpu_ipm_warm_state import coordinate_signature
        started=time.perf_counter()
        if type(solver) is not ForestGpuBatchedIPM or not solver.globalized:
            raise ValueError('Concrete globalized ForestGpuBatchedIPM required')
        if getattr(solver,'device_numeric_update_active',False):
            raise ValueError('Cannot compile new maps from layout-only host snapshots')
        solver.factor._context()
        self._solver=weakref.ref(solver);self.cp=solver.cp
        if self.cp.__name__!='cupy':raise ValueError('Actual CUDA solver required')
        self.thread=threading.get_ident();self.device=solver.factor.device
        self.stream=solver.factor.stream
        self.signature=coordinate_signature(solver)
        self.generation=getattr(solver,'numeric_update_generation',0)
        self.current_hashes=tuple(solver.problem_hashes)
        if tuple(problem_hash(p) for p in solver.full_problems)!=self.current_hashes:
            raise ValueError('The initial original LP snapshots/hash disagree')
        # Establish trust once, not only immutability after construction:
        # canonical independent forests, current zero-face/exact proofs, and
        # every intermediate numeric snapshot must form a coherent chain.
        from .gpu_ipm_numeric_update import prepare_host_rebind
        from .gpu_forest_map import _prepare_host_maps
        original,verified=prepare_host_rebind(solver,solver.full_problems)
        proof_reuse=getattr(solver,'reuse_equality_proofs',False)
        _prepare_host_maps(solver.forest_plans,solver.forest_reductions,reuse_equality_proofs=proof_reuse)
        if solver.secondary_forest_plans:
            _prepare_host_maps(solver.secondary_forest_plans,solver.secondary_forest_reductions,
                reuse_equality_proofs=proof_reuse)
        groups=lambda state:(state.full_problems,state.forest_problems,
            tuple(p.reduced for p in state.face_plans),
            tuple(r.problem for r in state.secondary_forest_reductions),state.problems)
        if any(tuple(problem_hash(p) for p in a)!=tuple(problem_hash(p) for p in b)
               for a,b in zip(groups(original),groups(verified))):
            raise ValueError('Initial intermediate host snapshots fail independent reproof')
        # Eliminate lazy numeric operators before taking the fixed inventory.
        if hasattr(solver,'_condensed_h')!=hasattr(solver,'_condensed_ht'):
            raise ValueError('Incomplete condensed operator cache')
        if solver.q and not hasattr(solver,'_condensed_h'):
            cp=self.cp
            rows=(cp.arange(solver.batch)[:,None]*solver.ng+cp.arange(solver.q)).ravel()
            solver._condensed_h=solver.g[rows,:].tocsr()
            solver._condensed_ht=solver._condensed_h.T.tocsr()
        self.pipeline=DeviceReductionPipeline(solver)
        self.binder=DeviceIPMPayloadBinder(solver,self.pipeline.host_states)
        self._mutable_work_paths={('values',),('factor','values')}
        self._numeric_expected=[(path,target,target.copy()) for path,target in self.binder.numeric_targets
                                if path not in self._mutable_work_paths]
        self._static_bindings=tuple(self.binder.static_targets)
        self.stream.synchronize()
        self.setup_seconds=time.perf_counter()-started

    def _context(self):
        solver=self._solver()
        if solver is None:raise ValueError('The original solver no longer exists')
        solver.factor._context()
        if (threading.get_ident()!=self.thread or self.cp.cuda.runtime.getDevice()!=self.device
                or self.cp.cuda.get_current_stream().ptr!=self.stream.ptr
                or getattr(solver,'numeric_update_generation',0)!=self.generation
                or tuple(solver.problem_hashes)!=self.current_hashes):
            raise ValueError('Device numeric plan context/current generation changed')
        return solver

    def _preflight(self,solver):
        from .gpu_ipm_numeric_update import _target
        flags=self.pipeline.preflight_flags()
        for path,target,expected in self._static_bindings:
            if path and path[0]!='__binder__' and _target(solver,path) is not target:
                raise ValueError('A static solver target was replaced')
            flags.append(self.cp.array_equal(target,expected))
        for path,target,expected in self._numeric_expected:
            if _target(solver,path) is not target:
                raise ValueError('A current numeric operator was replaced')
            flags.append(self.cp.array_equal(target,expected))
        if not bool(self.cp.all(self.cp.stack(flags))):
            raise ValueError('Static layout or current operator was altered before numeric update')

    def pack_current_host(self,problems):
        layout=self.pipeline.layouts['full']
        full=tuple(_freeze_problem(_validated_problem(p)) for p in problems)
        if len(full)!=layout.batch:raise ValueError('Current ordered batch size changed')
        for current,base in zip(full,layout.problems):
            a,b=current[0],base[0]
            if (a.shape!=b.shape or current[-1]!=base[-1]
                    or not np.array_equal(a.indptr,b.indptr) or not np.array_equal(a.indices,b.indices)):
                raise ValueError('Original CSR/equality topology changed; fresh GPU structure required')
        return full,_pack(full,self.cp),tuple(problem_hash(p) for p in full)

    def rebind(self,problems):
        from .gpu_ipm_numeric_update import NumericRebindRejected
        started=time.perf_counter()
        try:
            solver=self._context()
            self._preflight(solver)
            preflight_seconds=time.perf_counter()-started
            stamp=time.perf_counter()
            full,current,hashes=self.pack_current_host(problems)
            host_pack_seconds=time.perf_counter()-stamp
            stamp=time.perf_counter()
            batches,first,second,flags,offsets=self.pipeline.stage(current)
            copies=self.binder.stage(batches,first,second)
            # Source buffers are fully computed BEFORE any target is modified.
            # Every dynamic source receives a fresh snapshot, except native
            # factor scratch which is legitimately overwritten by Newton.
            expected=[(path,target,source.copy()) for path,target,source in copies
                      if path not in self._mutable_work_paths]
            validity=self.cp.stack(flags)
            if not bool(self.cp.all(validity)):
                failed=np.flatnonzero(~validity.get()).tolist()
                raise ValueError('Current GPU reduction/topology guards failed at checks '+str(failed))
            if not bool(self.cp.all(self.cp.isfinite(offsets))):
                raise ValueError('Current zero-face objective offset is nonfinite')
            self.stream.synchronize()
            transform_seconds=time.perf_counter()-stamp
            self._context()
        except ValueError as error:
            raise NumericRebindRejected(str(error)) from error
        factor=solver.factor;analysis=factor.analysis_count
        old_hashes=self.current_hashes
        commit_started=time.perf_counter()
        factor.factored=False;solver._last_internal_state=None;solver.failure_snapshot=None
        try:
            for _path,target,source in copies:self.cp.copyto(target,source)
            self.stream.synchronize()
            solver.full_problems=full
            solver.problem_hashes=hashes
            solver.numeric_update_generation=self.generation+1
            solver.device_numeric_update_active=True
            solver.host_reduced_snapshots_are_layout_only=True
            solver._device_zero_face_objective_offsets=offsets
            solver._forest_certificate_seconds=0.
            self.current_hashes=hashes;self.generation=solver.numeric_update_generation
            self._numeric_expected=expected
            result=dict(generation=self.generation,old_problem_hashes=old_hashes,new_problem_hashes=hashes,
                native_factor_object_preserved=True,symbolic_analysis_count_before=analysis,
                symbolic_analysis_count_after=factor.analysis_count,numerical_factor_invalidated=True,
                automatic_warm_state_reuse=False,cpu_lp_calls=0,
                device_numeric_reduction=True,host_reduced_snapshots_are_layout_only=True,
                numeric_plan_setup_seconds=self.setup_seconds,
                preflight_seconds=preflight_seconds,host_pack_and_hash_seconds=host_pack_seconds,
                device_transform_staging_and_guard_seconds=transform_seconds,
                commit_seconds=time.perf_counter()-commit_started,total_seconds=time.perf_counter()-started,
                copied_device_payloads=[repr(path) for path,_target,_source in copies],
                scope='Current host input pack/hash and control remain; numeric reductions/operator updates on GPU; '
                      'new full-original certification required; no CPU optimizer')
            solver.last_numeric_update=result
            return result
        except BaseException as error:
            factor.failed=True
            try:self.stream.synchronize()
            except BaseException as draining:error.add_note('Device update drain failed: '+str(draining))
            raise
