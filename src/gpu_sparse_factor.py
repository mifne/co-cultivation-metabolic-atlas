"""Experimental FP64 cuDSS uniform-batch factors with one symbolic analysis.

Version-pinned to the installed 0.7 C ABI; no library upgrade, CPU numerical
factorization, LP solve, inverse dictionary, or implicit host-memory execution.
Analysis includes CPU reordering. Numeric factorization and solves use CUDA.
The NVIDIA runtime remains separately installed under its own license.
"""
import ctypes as ct
from importlib.metadata import distribution
import threading

import numpy as np
from scipy.sparse import csr_matrix


class CudssError(RuntimeError):
    pass


def _library():
    package=distribution('nvidia-cudss-cu12')
    files=[package.locate_file(p) for p in package.files if str(p).endswith('/libcudss.so.0')]
    if len(files)!=1:raise CudssError('Exactly one installed cuDSS CUDA12 runtime is required')
    lib=ct.CDLL(str(files[0]))
    pointer=ct.c_void_p; out=ct.POINTER(pointer); integer=ct.c_int; size=ct.c_size_t; i64=ct.c_int64
    signatures=dict(cudssCreate=[out],cudssDestroy=[pointer],
        cudssConfigCreate=[out],cudssConfigDestroy=[pointer],
        cudssDataCreate=[pointer,out],cudssDataDestroy=[pointer,pointer],
        cudssSetStream=[pointer,pointer],cudssGetProperty=[integer,ct.POINTER(integer)],
        cudssConfigSet=[pointer,integer,pointer,size],
        cudssDataGet=[pointer,pointer,integer,pointer,size,ct.POINTER(size)],
        cudssExecute=[pointer,integer,pointer,pointer,pointer,pointer,pointer],
        cudssMatrixCreateDn=[out,i64,i64,i64,pointer,integer,integer],
        cudssMatrixCreateCsr=[out,i64,i64,i64,pointer,pointer,pointer,pointer,
                            integer,integer,integer,integer,integer],
        cudssMatrixDestroy=[pointer])
    for name,args in signatures.items():
        function=getattr(lib,name);function.argtypes=args;function.restype=integer
    version=[]
    for property_id in range(3):
        value=integer();status=lib.cudssGetProperty(property_id,ct.byref(value))
        if status:raise CudssError(f'cuDSS version query failed: {status}')
        version.append(value.value)
    if tuple(version[:2])!=(0,7):
        raise CudssError(f'Unsupported cuDSS ABI {version}; this adapter requires 0.7.x')
    return lib,tuple(version)


def canonical_square_pattern(matrix):
    """Copy indices, preserve explicit zeros, and reject malformed storage."""
    a=csr_matrix(matrix,dtype=np.float64,copy=True)
    if (a.shape[0]!=a.shape[1] or a.shape[0]<1 or not a.has_canonical_format
            or not np.isfinite(a.data).all() or max(a.shape+(a.nnz,))>=2**31):
        raise ValueError('Finite canonical nonempty square int32 CSR required')
    return a


class UniformCudssFactor:
    """Private device buffers; factor once per value update, solve many RHS.

    Calls are single-thread/single-device/single-stream. Returned solutions are
    copies and cannot be overwritten by a later solve. Closing drains the bound
    stream before destroying cuDSS wrappers and factors. Device arrays remain
    owned by the Python instance until it is released (CuPy may pool storage).
    """
    def __init__(self,pattern,*,batch_size=1,nrhs=1,matrix_type='symmetric',refinement_steps=2,
                 execution_layout='uniform', ordering='default', algorithm='default', precision='float64'):
        import cupy as cp
        if ordering not in ('default','amd') or algorithm not in ('default','alternate'):
            raise ValueError('Unsupported cuDSS 0.7 ordering or factor algorithm')
        if precision not in ('float64','float32'):raise ValueError('Unsupported native factor precision')
        self.precision=precision
        self.ordering=ordering;self.algorithm=algorithm
        self._algorithm_snapshot=(ordering,algorithm,precision)
        if (isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1
                or isinstance(nrhs,bool) or not isinstance(nrhs,int) or nrhs<1
                or matrix_type not in ('symmetric','general','spd')
                or not isinstance(refinement_steps,int) or refinement_steps<0):
            raise ValueError('Invalid cuDSS batch, RHS count, matrix type, or refinement')
        a=canonical_square_pattern(pattern)
        if precision=='float32' and np.any(np.abs(a.data)>np.finfo(np.float32).max):
            raise ValueError('Initial matrix exceeds native FP32 finite range')
        if execution_layout not in ('uniform','block_diagonal') or (execution_layout=='block_diagonal' and nrhs!=1):
            raise ValueError('Block diagonal native layout currently requires one RHS')
        if execution_layout=='block_diagonal' and max(a.shape[0]*batch_size,a.nnz*batch_size)>=2**31:
            raise ValueError('Block diagonal native layout exceeds int32 capacity')
        if matrix_type!='general' and (a-a.T).nnz:
            raise ValueError('Symmetric factorization requires a symmetric full matrix')
        transpose_positions=None
        if matrix_type!='general':
            rows=np.repeat(np.arange(a.shape[0],dtype=np.int64),np.diff(a.indptr))
            keys=rows*a.shape[0]+a.indices
            transposed=a.indices.astype(np.int64)*a.shape[0]+rows
            transpose_positions=np.searchsorted(keys,transposed)
            if (np.any(transpose_positions>=a.nnz)
                    or not np.array_equal(keys[transpose_positions],transposed)):
                raise ValueError('Symmetric storage must include both triangles, including explicit zeros')
        self.cp=cp;self.batch=batch_size;self.n=a.shape[0];self.nnz=a.nnz;self.nrhs=nrhs
        self.host_pattern=a;self.device=cp.cuda.runtime.getDevice()
        self.execution_layout=execution_layout
        self.stream=cp.cuda.get_current_stream();self.thread=threading.get_ident()
        self.closed=False;self.factored=False;self.failed=False
        self.analysis_count=0;self.factor_count=0;self.solve_count=0
        self.objects={k:ct.c_void_p() for k in ('handle','config','data','a','x','b')}
        self.lib,self.version=_library()
        self.indptr=cp.asarray(a.indptr,dtype=cp.int32);self.indices=cp.asarray(a.indices,dtype=cp.int32)
        # Public logical dimensions/operators stay per-lane. Only the native
        # execution layout changes; no off-diagonal environmental coupling.
        self._native_n=self.n if execution_layout=='uniform' else self.n*self.batch
        self._native_nnz=self.nnz if execution_layout=='uniform' else self.nnz*self.batch
        self._native_batch=self.batch if execution_layout=='uniform' else 1
        if execution_layout=='block_diagonal':
            self._native_host_indptr=np.r_[np.concatenate([
                a.indptr[:-1].astype(np.int64)+lane*a.nnz for lane in range(self.batch)]),self._native_nnz].astype(np.int32)
            self._native_host_indices=np.concatenate([
                a.indices.astype(np.int64)+lane*self.n for lane in range(self.batch)]).astype(np.int32)
            self._native_host_indptr.flags.writeable=False;self._native_host_indices.flags.writeable=False
            self._native_indptr=cp.asarray(self._native_host_indptr)
            self._native_indices=cp.asarray(self._native_host_indices)
        else:
            self._native_indptr=self.indptr;self._native_indices=self.indices
        self._native_layout_snapshot=(execution_layout,self.n,self.nnz,self.batch,self.nrhs,
            self._native_n,self._native_nnz,self._native_batch,
            self._native_indptr,self._native_indices)
        self._native_buffer_metadata=tuple((v.shape,v.strides,v.dtype.str,v.device.id,v.data.ptr)
            for v in (self._native_indptr,self._native_indices))
        self.transpose_positions=(None if transpose_positions is None else cp.asarray(transpose_positions))
        self.values=cp.empty((batch_size,a.nnz),dtype=cp.float64)
        # Each uniform-batch RHS is a separate column-major n x nrhs matrix.
        self.rhs=cp.zeros((batch_size,nrhs,self.n),dtype=cp.float64)
        self.solution=cp.zeros_like(self.rhs)
        # External operators, residuals and certificates remain FP64. FP32 is
        # only a numerical preconditioner/proposal and has no accuracy promise.
        native_dtype=cp.float32 if precision=='float32' else cp.float64
        self._native_values=self.values if precision=='float64' else cp.empty_like(self.values,dtype=native_dtype)
        self._native_rhs=self.rhs if precision=='float64' else cp.empty_like(self.rhs,dtype=native_dtype)
        self._native_solution=self.solution if precision=='float64' else cp.empty_like(self.solution,dtype=native_dtype)
        self._numeric_native_snapshot=tuple((v,v.shape,v.strides,v.dtype.str,v.device.id,v.data.ptr)
            for v in (self._native_values,self._native_rhs,self._native_solution))
        # Internal solves keep numerical rejection on the device. These are
        # lifetime lane counters, not acceptance flags or a CPU fallback.
        self.internal_solve_count=0
        self._internal_invalid_rhs_count=cp.zeros(batch_size,dtype=cp.int64)
        self._internal_nonfinite_output_count=cp.zeros(batch_size,dtype=cp.int64)
        self.values[:]=cp.asarray(a.data)[None]
        try:
            for name,args in (('cudssCreate',(ct.byref(self.objects['handle']),)),
                ('cudssConfigCreate',(ct.byref(self.objects['config']),)),
                ('cudssDataCreate',(self.objects['handle'],ct.byref(self.objects['data'])))):
                self._call(name,*args)
            self._call('cudssSetStream',self.objects['handle'],ct.c_void_p(self.stream.ptr))
            # cuDSS 0.7: AMD=CUDSS_ALG_3, alternate factor=CUDSS_ALG_1.
            # Apply before symbolic analysis; never change the numerical LP.
            for key,value in ((0,3 if ordering=='amd' else 0),(1,1 if algorithm=='alternate' else 0)):
                if value:
                    scalar=ct.c_int(value)
                    self._call('cudssConfigSet',self.objects['config'],key,ct.byref(scalar),ct.sizeof(scalar))
            # 0.7 enum IDs, checked against installed cudss.h. Hybrid modes off.
            # cuDSS 0.7 rejects deterministic mode with internal refinement.
            # Keep refinement for accuracy; do not claim bitwise determinism.
            for key,value in ((19,self._native_batch),(20,-1),(12,0),(16,0),(6,refinement_steps),(25,0)):
                scalar=ct.c_int(value)
                self._call('cudssConfigSet',self.objects['config'],key,ct.byref(scalar),ct.sizeof(scalar))
            if precision=='float32':
                # The default FP32 1e-5 pivot perturbation exceeds our scaled
                # Newton regularization. FP64 outer residual checks decide
                # whether this experimental low-precision factor is usable.
                epsilon=ct.c_double(1e-12)
                self._call('cudssConfigSet',self.objects['config'],10,ct.byref(epsilon),ct.sizeof(epsilon))
                cp.copyto(self._native_values,self.values,casting='unsafe')
            native_type=0 if precision=='float32' else 1
            self._call('cudssMatrixCreateCsr',ct.byref(self.objects['a']),self._native_n,self._native_n,self._native_nnz,
                self._native_indptr.data.ptr,0,self._native_indices.data.ptr,self._native_values.data.ptr,
                10,native_type,{'general':0,'symmetric':1,'spd':3}[matrix_type],0,0)
            for key,buffer in (('x',self._native_solution),('b',self._native_rhs)):
                self._call('cudssMatrixCreateDn',ct.byref(self.objects[key]),self._native_n,nrhs,self._native_n,
                    buffer.data.ptr,native_type,0)
            self._execute(3);self.analysis_count=1
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:error.add_note(f'Cleanup also failed: {cleanup}')
            raise

    def _call(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:raise CudssError(f'{name} failed with cuDSS status {status}')

    def _context(self):
        if self.closed or self.failed:raise CudssError('cuDSS factor is closed or failed')
        if hasattr(self,'_algorithm_snapshot') and (self.ordering,self.algorithm,self.precision)!=self._algorithm_snapshot:
            raise CudssError('Native factor algorithm changed after analysis')
        if hasattr(self,'_numeric_native_snapshot'):
            for value,expected in zip((self._native_values,self._native_rhs,self._native_solution),self._numeric_native_snapshot):
                if value is not expected[0] or (value.shape,value.strides,value.dtype.str,value.device.id,value.data.ptr)!=expected[1:]:
                    raise CudssError('Native numeric buffer identity or metadata changed')
        if (self.thread!=threading.get_ident() or self.cp.cuda.runtime.getDevice()!=self.device
                or self.cp.cuda.get_current_stream().ptr!=self.stream.ptr):
            raise CudssError('cuDSS factor cannot change thread, CUDA device, or stream')
        if hasattr(self,'_native_layout_snapshot'):
            current=(self.execution_layout,self.n,self.nnz,self.batch,self.nrhs,
                self._native_n,self._native_nnz,self._native_batch)
            if (current!=self._native_layout_snapshot[:8]
                    or self._native_indptr is not self._native_layout_snapshot[8]
                    or self._native_indices is not self._native_layout_snapshot[9]):
                raise CudssError('Native factor execution layout changed')
            if tuple((v.shape,v.strides,v.dtype.str,v.device.id,v.data.ptr)
                    for v in (self._native_indptr,self._native_indices))!=self._native_buffer_metadata:
                raise CudssError('Native factor coordinate metadata changed')

    def _execute(self,phase):
        d=self.objects
        mixed=getattr(self,'precision','float64')=='float32'
        if mixed and phase==4:
            self.cp.copyto(self._native_values,self.values,casting='unsafe')
            if not bool(self.cp.all(self.cp.isfinite(self._native_values))):
                raise CudssError('Native FP32 matrix conversion overflow')
        if mixed and phase==1008:
            # Lane/RHS scaling avoids overflow during FP64 -> FP32 packing.
            scale=self.cp.maximum(1.,self.cp.max(self.cp.abs(self.rhs),axis=2,keepdims=True))
            self.cp.copyto(self._native_rhs,self.rhs/scale,casting='unsafe')
        self._call('cudssExecute',d['handle'],phase,d['config'],d['data'],d['a'],d['x'],d['b'])
        if mixed and phase==1008:
            self.cp.multiply(self._native_solution,scale,out=self.solution)

    def _array(self,value,shape):
        cp=self.cp
        if not isinstance(value,cp.ndarray) or value.dtype!=cp.float64 or value.shape!=shape:
            raise ValueError('FP64 device array with exact batch shape required')
        if value.device.id!=self.device or not bool(cp.isfinite(value).all()):
            raise ValueError('Nonfinite or wrong-device input')
        return value

    def factor(self,values):
        self._context();values=self._array(values,(self.batch,self.nnz))
        if self.transpose_positions is not None and not bool(
                self.cp.all(values==values[:,self.transpose_positions])):
            raise ValueError('Symmetric numeric values must match in both triangles')
        self.factored=False
        try:
            self.cp.copyto(self.values,values)
            self._execute(4)  # Repeat numeric factorization, retaining analysis.
            self.factor_count+=1;self.factored=True
        except BaseException:
            self.failed=True;self.stream.synchronize();raise

    def solve(self,rhs):
        self._context()
        if not self.factored:raise CudssError('No current numeric factorization')
        rhs=self._array(rhs,(self.batch,self.n,self.nrhs))
        try:
            self.cp.copyto(self.rhs,rhs.transpose(0,2,1));self._execute(1008)
            self.solve_count+=1
            if not bool(self.cp.isfinite(self.solution).all()):
                raise CudssError('cuDSS returned a nonfinite solution')
            return self.solution.transpose(0,2,1).copy()
        except BaseException:
            self.failed=True;self.stream.synchronize();raise

    def _solve_device_checked(self,rhs):
        """Internal IPM proposal, without per-solve finite-result host reads.

        Context, factor state, array shape/type/device and cuDSS status remain
        mandatory checks. Every nonfinite RHS lane is replaced with zero
        BEFORE cuDSS sees it. Invalid input OR nonfinite output is represented
        by an all-NaN returned lane, so callers cannot silently accept a
        sanitized zero solution. The IPM caller MUST retain its device finite
        checks and original full-Newton/LP acceptance gates.

        The returned allocation is independent of the reusable solution
        buffer; subsequent solves cannot overwrite an Arnoldi direction.
        Numerical input/output rejections accumulate on device and do not
        poison other independent lanes. cuDSS/CUDA call errors still fail the
        entire workspace and drain its bound stream, as in public solve().
        This method is not a replacement for the strict public API.
        """
        self._context()
        if not self.factored:raise CudssError('No current numeric factorization')
        cp=self.cp
        # Metadata is host-resident; unlike _array(), this does not materialize
        # a device reduction as a Python bool. Never omit the device guard.
        if (not isinstance(rhs,cp.ndarray) or rhs.dtype!=cp.float64
                or rhs.shape!=(self.batch,self.n,self.nrhs)):
            raise ValueError('FP64 device array with exact batch shape required')
        if rhs.device.id!=self.device:
            raise ValueError('Wrong-device internal right-hand side')
        try:
            input_finite=cp.all(cp.isfinite(rhs),axis=(1,2))
            safe_rhs=cp.where(input_finite[:,None,None],rhs,0.)
            cp.copyto(self.rhs,safe_rhs.transpose(0,2,1))
            self._execute(1008)
            self.solve_count+=1
            self.internal_solve_count+=1
            output_finite=cp.all(cp.isfinite(self.solution),axis=(1,2))
            self._internal_invalid_rhs_count+=(~input_finite).astype(cp.int64)
            self._internal_nonfinite_output_count+=(~output_finite).astype(cp.int64)
            valid=input_finite&output_finite
            return cp.where(valid[:,None,None],self.solution.transpose(0,2,1),cp.nan)
        except BaseException:
            self.failed=True;self.stream.synchronize();raise

    def internal_diagnostics(self):
        """Return independent device-counter copies; never download vectors."""
        self._context()
        return dict(solve_count=self.internal_solve_count,
            invalid_rhs_count=self._internal_invalid_rhs_count.copy(),
            nonfinite_output_count=self._internal_nonfinite_output_count.copy(),
            scope='Lifetime internal-solve lane counts; NaN proposals must fail caller numerical gates')

    def _solve_columns_device_checked(self,rhs):
        """Multi-RHS proposals with finite guards independent for each column.

        Lifetime counters remain native-group counts; invalid/nonfinite RHS
        columns are marked NaN independently and cannot poison good columns.
        Full original Newton/LP acceptance belongs to the caller.
        """
        self._context()
        if not self.factored:raise CudssError('No current numeric factorization')
        cp=self.cp
        if (not isinstance(rhs,cp.ndarray) or rhs.shape!=(self.batch,self.n,self.nrhs)
                or rhs.dtype!=cp.float64 or rhs.device.id!=self.device):
            raise ValueError('Exact FP64 native multi-RHS shape/device required')
        try:
            finite=cp.all(cp.isfinite(rhs),axis=1)
            cp.copyto(self.rhs,cp.where(finite[:,None,:],rhs,0.).transpose(0,2,1))
            self._execute(1008);self.solve_count+=1;self.internal_solve_count+=1
            output_finite=cp.all(cp.isfinite(self.solution),axis=2)
            self._internal_invalid_rhs_count+=cp.any(~finite,axis=1).astype(cp.int64)
            self._internal_nonfinite_output_count+=cp.any(~output_finite,axis=1).astype(cp.int64)
            return cp.where((finite&output_finite)[:,None,:],self.solution.transpose(0,2,1),cp.nan)
        except BaseException:
            self.failed=True;self.stream.synchronize();raise

    def _solve_device_checked_fused(self,rhs):
        """Same finite-lane contract as the internal API, with two guard kernels."""
        self._context()
        if not self.factored:raise CudssError('No current numeric factorization')
        from .gpu_solve_guards import FusedSolveGuards
        if not hasattr(self,'_fused_solve_guards'):
            self._fused_solve_guards=FusedSolveGuards(self)
        # Reject public metadata errors without poisoning a healthy factor.
        cp=self.cp
        if (not isinstance(rhs,cp.ndarray) or rhs.shape!=(self.batch,self.n,self.nrhs)
                or rhs.dtype!=cp.float64 or rhs.device.id!=self.device):
            raise ValueError('Exact-shape FP64 RHS on the bound device required')
        try:
            self._fused_solve_guards.pack(rhs)
            self._execute(1008)
            self.solve_count+=1;self.internal_solve_count+=1
            return self._fused_solve_guards.finish()
        except BaseException:
            self.failed=True;self.stream.synchronize();raise

    def info(self):
        self._context();self.stream.synchronize()
        value=ct.c_int();written=ct.c_size_t()
        self._call('cudssDataGet',self.objects['handle'],self.objects['data'],0,
            ct.byref(value),ct.sizeof(value),ct.byref(written))
        if written.value!=ct.sizeof(value):raise CudssError('Unexpected cuDSS info shape')
        return value.value

    def close(self):
        if self.closed:return
        self.closed=True;errors=[]
        try:self.stream.synchronize()
        except BaseException as error:errors.append(error)
        for key in ('b','x','a','data','config','handle'):
            obj=self.objects[key]
            if not obj:continue
            name={'b':'cudssMatrixDestroy','x':'cudssMatrixDestroy','a':'cudssMatrixDestroy',
                  'data':'cudssDataDestroy','config':'cudssConfigDestroy','handle':'cudssDestroy'}[key]
            try:self._call(name,*((self.objects['handle'],obj) if key=='data' else (obj,)))
            except BaseException as error:errors.append(error)
            self.objects[key]=ct.c_void_p()
        if errors:raise BaseExceptionGroup('cuDSS cleanup failed',errors)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
