"""FP64 capture-safe primitives for fixed-shape experimental GPU LP kernels.

Uses the documented cuBLAS C API with a private handle (no CuPy/library
monkeypatch), a CSR kernel, and pivoted small dense solves. No CPU numerical
solve, host pivot selection, lower precision or relaxed acceptance gates.
"""
import ctypes
import numpy as np

SOURCE=r'''
extern "C" __global__ void csrmm(const int* ptr,const int* idx,const double* val,
 const double* x,double* out,int rows,int cols,int width){
 int lane=threadIdx.x%32;
 long long work=((long long)blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(work>=(long long)rows*width)return;
 int i=work/width,j=work%width;
 double sum=0.;
 for(int k=ptr[i]+lane;k<ptr[i+1];k+=32)sum+=val[k]*x[(long long)idx[k]*width+j];
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
 if(!lane)out[work]=sum;
}
extern "C" __global__ void small_solve(double* work,double* out,int n,int width,const int* active_size){
 int b=blockIdx.x,t=threadIdx.x;
 int active=active_size?min(n,max(0,active_size[0])):n;
 double* a=work+(long long)b*n*width;
 __shared__ int pivot;
 __shared__ double scale;
 extern __shared__ double shared[];
 double* pivot_row=shared;
 double* factors=shared+width;
 int nrhs=width-n;
 int active_width=active+nrhs;
 for(int k=0;k<active;k++){
   if(t==0){pivot=k;double best=fabs(a[(long long)k*width+k]);
     for(int i=k+1;i<active;i++){double v=fabs(a[(long long)i*width+k]);if(v>best){best=v;pivot=i;}}
   }
   __syncthreads();
   for(int z=t;z<active_width;z+=blockDim.x){int j=z<active?z:n+z-active;
     double v=a[(long long)k*width+j];
     a[(long long)k*width+j]=a[(long long)pivot*width+j];a[(long long)pivot*width+j]=v;}
   __syncthreads();
   if(t==0)scale=a[(long long)k*width+k];
   __syncthreads();
   for(int z=t;z<active_width;z+=blockDim.x){int j=z<active?z:n+z-active;
     double value=a[(long long)k*width+j]/scale;
     a[(long long)k*width+j]=value;pivot_row[z]=value;}
   __syncthreads();
   // Cache factors before updating any column, then distribute contiguous
   // row elements across threads. This removes the old per-thread serial
   // row loop and its uncoalesced global-memory traffic without a race.
   for(int i=t;i<active;i+=blockDim.x)factors[i]=a[(long long)i*width+k];
   __syncthreads();
   for(int z=t;z<active*active_width;z+=blockDim.x){
     int i=z/active_width,local=z%active_width;
     int j=local<active?local:n+local-active;
     if(i!=k){if(j==k)a[(long long)i*width+j]=0.;
       else a[(long long)i*width+j]-=factors[i]*pivot_row[local];}}
   __syncthreads();
 }
 for(int z=t;z<n*nrhs;z+=blockDim.x)out[(long long)b*n*nrhs+z]=a[(long long)(z/nrhs)*width+n+z%nrhs];
}
extern "C" __global__ void small_lu(double* all,int* pivots,int n,const int* active_size){
 int b=blockIdx.x,t=threadIdx.x;
 int active=active_size?min(n,max(0,active_size[0])):n;
 double* a=all+(long long)b*n*n;
 __shared__ int pivot;
 __shared__ double diagonal;
 for(int k=0;k<active;k++){
   if(t==0){pivot=k;double best=fabs(a[(long long)k*n+k]);
     for(int i=k+1;i<active;i++){double v=fabs(a[(long long)i*n+k]);if(v>best){best=v;pivot=i;}}
     pivots[b*n+k]=pivot;}
   __syncthreads();
   for(int j=t;j<active;j+=blockDim.x){double v=a[(long long)k*n+j];
     a[(long long)k*n+j]=a[(long long)pivot*n+j];a[(long long)pivot*n+j]=v;}
   __syncthreads();
   if(t==0)diagonal=a[(long long)k*n+k];
   __syncthreads();
   for(int i=k+1+t;i<active;i+=blockDim.x)a[(long long)i*n+k]/=diagonal;
   __syncthreads();
   int remain=active-k-1;
   for(int z=t;z<remain*remain;z+=blockDim.x){int i=k+1+z/remain,j=k+1+z%remain;
     a[(long long)i*n+j]-=a[(long long)i*n+k]*a[(long long)k*n+j];}
   __syncthreads();
 }
}
extern "C" __global__ void small_lu_solve(const double* all,const int* pivots,
 double* rhs,int n,int nrhs,int transpose,const int* active_size){
 int b=blockIdx.x,t=threadIdx.x;
 int active=active_size?min(n,max(0,active_size[0])):n;
 const double* a=all+(long long)b*n*n;
 double* x=rhs+(long long)b*n*nrhs;
 if(!transpose){
   for(int j=t;j<nrhs;j+=blockDim.x)for(int k=0;k<active;k++){
     int p=pivots[b*n+k];double v=x[k*nrhs+j];x[k*nrhs+j]=x[p*nrhs+j];x[p*nrhs+j]=v;}
   __syncthreads();
 }
 // Non-transpose: unit L then U. Transpose: U^T then unit L^T.
 for(int k=0;k<active;k++){
   if(transpose)for(int j=t;j<nrhs;j+=blockDim.x)x[k*nrhs+j]/=a[(long long)k*n+k];
   __syncthreads();
   for(int z=t;z<(active-k-1)*nrhs;z+=blockDim.x){int i=k+1+z/nrhs,j=z%nrhs;
     double v=transpose?a[(long long)k*n+i]:a[(long long)i*n+k];
     x[i*nrhs+j]-=v*x[k*nrhs+j];}
   __syncthreads();
 }
 for(int k=active-1;k>=0;k--){
   if(!transpose)for(int j=t;j<nrhs;j+=blockDim.x)x[k*nrhs+j]/=a[(long long)k*n+k];
   __syncthreads();
   for(int z=t;z<k*nrhs;z+=blockDim.x){int i=z/nrhs,j=z%nrhs;
     double v=transpose?a[(long long)k*n+i]:a[(long long)i*n+k];
     x[i*nrhs+j]-=v*x[k*nrhs+j];}
   __syncthreads();
 }
 if(transpose)for(int j=t;j<nrhs;j+=blockDim.x)for(int k=active-1;k>=0;k--){
   int p=pivots[b*n+k];double v=x[k*nrhs+j];x[k*nrhs+j]=x[p*nrhs+j];x[p*nrhs+j]=v;}
}
'''


class CaptureMath:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        # CuPy has loaded the runtime and cuBLAS before this module is used.
        self.lib=ctypes.CDLL('libcublas.so.12')
        self.handle=ctypes.c_void_p()
        self.lib.cublasCreate_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cublasSetStream_v2.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        self.lib.cublasDestroy_v2.argtypes=[ctypes.c_void_p]
        self._check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)))
        self.set_workspace=getattr(self.lib,'cublasSetWorkspace_v2',None) or self.lib.cublasSetWorkspace
        self.set_workspace.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        # Conditional CUDA graph bodies cannot contain allocation nodes.
        # Give cuBLAS a persistent workspace instead of its async allocator.
        self.workspace=cp.empty(32*1024*1024,dtype=cp.uint8)
        p=ctypes.c_void_p;i=ctypes.c_int;ll=ctypes.c_longlong
        self.gemm=self.lib.cublasDgemmStridedBatched
        self.gemm.argtypes=[p,i,i,i,i,i,p,p,i,ll,p,i,ll,p,p,i,ll,i]
        self.one=ctypes.c_double(1.);self.zero=ctypes.c_double(0.)
        self.csr_kernel=cp.RawKernel(SOURCE,'csrmm',options=('--fmad=false',))
        self.solve_kernel=cp.RawKernel(SOURCE,'small_solve',options=('--fmad=false',))
        self.lu_kernel=cp.RawKernel(SOURCE,'small_lu',options=('--fmad=false',))
        self.lu_solve_kernel=cp.RawKernel(SOURCE,'small_lu_solve',options=('--fmad=false',))
        self.csr_kernel.compile();self.solve_kernel.compile();self.lu_kernel.compile();self.lu_solve_kernel.compile()

    @staticmethod
    def _check(status):
        if status:raise RuntimeError(f'cuBLAS status {status}')

    def close(self):
        """Release private cuBLAS allocations when an operator is retired."""
        handle=getattr(self,'handle',None)
        if handle is not None and handle.value:
            self._check(self.lib.cublasDestroy_v2(handle))
            handle.value=None

    def __del__(self):
        try:self.close()
        except Exception:pass  # CUDA can already be unloaded at shutdown.

    def mm(self,a,b):
        cp=self.cp
        if hasattr(a,'indptr'):
            if a.format!='csr' or a.indices.dtype!=cp.int32 or a.indptr.dtype!=cp.int32:
                raise ValueError('CSR FP64 with int32 indices required')
            x=cp.ascontiguousarray(b)
            if x.ndim!=2 or a.shape[1]!=x.shape[0]:raise ValueError('CSR dimension mismatch')
            out=cp.empty((a.shape[0],x.shape[1]),dtype=cp.float64)
            work=out.size*32
            self.csr_kernel(((work+255)//256,),(256,),
                (a.indptr,a.indices,a.data,x,out,np.int32(a.shape[0]),np.int32(a.shape[1]),np.int32(x.shape[1])))
            return out
        if a.dtype!=cp.float64 or b.dtype!=cp.float64:raise ValueError('FP64 only')
        if a.ndim not in (2,3) or b.ndim not in (2,3):raise ValueError('2D/3D only')
        m,k=a.shape[-2:];kb,n=b.shape[-2:]
        if k!=kb:raise ValueError('GEMM dimension mismatch')
        ba=1 if a.ndim==2 else a.shape[0];bb=1 if b.ndim==2 else b.shape[0]
        batch=max(ba,bb)
        if ba not in (1,batch) or bb not in (1,batch):raise ValueError('Batch mismatch')
        a=cp.ascontiguousarray(a);b=cp.ascontiguousarray(b)
        out=cp.empty((batch,m,n),dtype=cp.float64)
        if not m or not n or not k:
            out.fill(0.)
            return out[0] if a.ndim==2 and b.ndim==2 else out
        self._check(self.lib.cublasSetStream_v2(self.handle,cp.cuda.get_current_stream().ptr))
        # setStream resets the workspace; bind it again after each stream set.
        self._check(self.set_workspace(self.handle,self.workspace.data.ptr,self.workspace.nbytes))
        self._check(self.gemm(self.handle,0,0,n,m,k,ctypes.byref(self.one),
            b.data.ptr,n,0 if bb==1 else k*n,a.data.ptr,k,0 if ba==1 else m*k,
            ctypes.byref(self.zero),out.data.ptr,n,m*n,batch))
        return out[0] if a.ndim==2 and b.ndim==2 else out

    def solve(self,a,b,active_size=None):
        """FP64 solve; active_size requires an EXACT identity-padded block.

        The revised solver creates this structure from unused zero U/V slots.
        Never use active_size to truncate a general dense matrix.
        """
        cp=self.cp
        if a.ndim!=3 or b.ndim!=3 or a.shape[:2]!=b.shape[:2] or a.shape[1]!=a.shape[2]:
            raise ValueError('Batched square solve required')
        if a.dtype!=cp.float64 or b.dtype!=cp.float64:raise ValueError('FP64 only')
        n=a.shape[1];batch=a.shape[0];nrhs=b.shape[2]
        work=cp.ascontiguousarray(cp.concatenate((a,b),axis=2))
        out=cp.empty_like(b,order='C')
        if active_size is not None and (active_size.dtype!=cp.int32 or active_size.shape!=(1,)):
            raise ValueError('Active prefix must be a one-element device int32 array')
        self.solve_kernel((batch,),(256,),(work,out,np.int32(n),np.int32(n+nrhs),
            np.uint64(0) if active_size is None else active_size),shared_mem=(2*n+nrhs)*8)
        return out

    def einsum(self,signature,a,b):
        if signature in ('bkn,bn->bk','brm,bm->br'):
            return self.mm(a,b[:,:,None])[:,:,0]
        if signature in ('bkn,bk->bn','bmr,bm->br','brm,br->bm'):
            return self.mm(a.transpose(0,2,1),b[:,:,None])[:,:,0]
        if signature=='bmr,br->bm':return self.mm(a,b[:,:,None])[:,:,0]
        raise ValueError(f'Unsupported contraction {signature}')

    def factor(self,a,active_size=None):
        """One pivoted FP64 LU, reused only for this exact small matrix."""
        cp=self.cp
        if a.ndim!=3 or a.shape[1]!=a.shape[2] or a.dtype!=cp.float64:
            raise ValueError('Batched FP64 square matrix required')
        if active_size is not None and (active_size.dtype!=cp.int32 or active_size.shape!=(1,)):
            raise ValueError('Active prefix must be a one-element device int32 array')
        lu=a.copy(order='C');batch,n,_=a.shape;pivots=cp.empty((batch,n),dtype=cp.int32)
        self.lu_kernel((batch,),(256,),(lu,pivots,np.int32(n),np.uint64(0) if active_size is None else active_size))
        return lu,pivots,active_size

    def solve_factored(self,factor,b,transpose=False):
        cp=self.cp;lu,pivots,active_size=factor
        if b.ndim!=3 or b.shape[:2]!=lu.shape[:2] or b.dtype!=cp.float64:
            raise ValueError('Batched FP64 compatible RHS required')
        out=b.copy(order='C');batch,n,nrhs=b.shape
        self.lu_solve_kernel((batch,),(256,),(lu,pivots,out,np.int32(n),np.int32(nrhs),np.int32(transpose),
            np.uint64(0) if active_size is None else active_size))
        return out
